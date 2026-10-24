from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import main
from backend.agent.orchestrator import AgentOrchestrator as RealAgentOrchestrator
from backend.agent.schemas import ActionDecision, AgentAction
from backend.llm.base import LLMProvider
from backend.llm.data_policy import ExternalModelDataPolicy
from backend.llm.schemas import LLMRequest, LLMResponse
from backend.llm.template_provider import TemplateProvider
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import RetrievalResponse, RetrievalResult
from backend.storage.logger import LearningLogger, sha256_bytes


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-assignment-progress")
    original = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "assignments.db")
    main.learning_logger.seed_demo_identity()
    main.learning_logger.ensure_demo_assignments()
    client = TestClient(main.app)
    client._original_logger = original  # type: ignore[attr-defined]
    return client


def _restore(client: TestClient) -> None:
    main.learning_logger = client._original_logger  # type: ignore[attr-defined]


def _login(client: TestClient, username: str) -> tuple[str, dict]:
    response = client.post("/api/auth/login", json={"username": username, "password": "demo123456"})
    assert response.status_code == 200
    payload = response.json()
    if payload.get("must_change_password"):
        changed = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {payload['access_token']}"},
            json={"current_password": "demo123456", "new_password": f"{username}-changed-123"},
        )
        assert changed.status_code == 200
        payload = changed.json()
    return payload["access_token"], payload["user"]


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _first_assignment(client: TestClient, token: str) -> tuple[dict, dict]:
    payload = client.get("/api/assignments", headers=_headers(token)).json()
    assignment = payload["assignments"][0]
    return assignment, assignment["steps"][0]


def test_student_updates_only_own_assignment_progress(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token_1, _ = _login(client, "student_demo")
        token_2, _ = _login(client, "student_demo2")
        assignment, step = _first_assignment(client, token_1)

        response = client.post(
            f"/api/assignments/{assignment['id']}/progress",
            headers=_headers(token_1),
            json={
                "step_id": step["id"],
                "status": "in_progress",
                "progress_percent": 40,
                "note": "Finished the topic definition.",
            },
        )
        assert response.status_code == 200
        assert response.json()["progress"]["progress_percent"] == 40

        own = client.get(f"/api/assignments/{assignment['id']}/progress", headers=_headers(token_1))
        other = client.get(f"/api/assignments/{assignment['id']}/progress", headers=_headers(token_2))
        assert own.json()["progress"]["progress_percent"] == 40
        assert other.json()["progress"] is None
    finally:
        _restore(client)


def test_practice_guide_can_attach_assignment_and_step(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, user = _login(client, "student_demo")
        headers = _headers(token)
        assignment, step = _first_assignment(client, token)
        sid = "assignment-practice-session"
        assert client.post("/api/learning/sessions", headers=headers, json={"client_session_id": sid}).status_code == 200

        response = client.post(
            "/api/practice/guide",
            headers=headers,
            json={
                "topic": "5G NAS replay-risk experiment",
                "client_session_id": sid,
                "assignment_id": assignment["id"],
                "step_id": step["id"],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["assignment_id"] == assignment["id"]
        assert payload["step_id"] == step["id"]
        run = main.learning_logger.agent_run(payload["run_id"])
        assert run is not None
        assert run["state"]["assignment_context_assignment_id"] == assignment["id"]
        assert run["state"]["assignment_context_step_id"] == step["id"]
        assert run["state"]["learning_context"]["step_title"] == step["title"]

        timeline = client.get(f"/api/students/{user['id']}/timeline", headers=headers).json()["events"]
        guide_events = [item for item in timeline if item["event_type"] == "PRACTICE_GUIDE_GENERATED"]
        assert guide_events
        assert str(guide_events[-1]["assignment_id"]) == str(assignment["id"])
        assert guide_events[-1]["metadata"]["step_id"] == step["id"]
    finally:
        _restore(client)


def test_chat_passes_validated_assignment_scope_into_agent(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    captured = {}

    class RecordingOrchestrator:
        def __init__(self, **_kwargs):
            pass

        def run(self, question, **kwargs):
            captured.update(kwargs)
            return {
                "answer": "已记录当前实验上下文。",
                "citations": [],
                "confidence": 0.5,
                "abstained": False,
                "module": "5G NAS",
                "intent": "practice_guide",
                "run_id": "assignment-context-run",
                "run_status": "COMPLETED",
                "current_step": 1,
                "requires_user_input": False,
                "pending_question": None,
                "trace_id": "assignment-context-trace",
                "provider": "template",
                "model": "test",
                "fallback_used": False,
                "stop_reason": "ANSWERED",
                "tool_calls": [],
                "delivery_mode": "local",
            }

    try:
        token, _ = _login(client, "student_demo")
        headers = _headers(token)
        assignment, step = _first_assignment(client, token)
        monkeypatch.setattr(main, "AgentOrchestrator", RecordingOrchestrator)
        monkeypatch.setattr(main, "_check_agent_result_before_business_write", lambda **_kwargs: None)
        monkeypatch.setattr(main, "_record_question", lambda *_args, **_kwargs: 991)

        response = client.post(
            "/api/chat",
            headers=headers,
            json={
                "question": "结合当前步骤说明评价指标。",
                "course_id": assignment["course_id"],
                "assignment_id": assignment["id"],
                "step_id": step["id"],
            },
        )

        assert response.status_code == 200, response.text
        assert captured["assignment_id"] == assignment["id"]
        assert captured["step_id"] == step["id"]
    finally:
        _restore(client)


def test_project31_http_chain_projects_trusted_assignment_context(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    captured = {"planner": [], "requests": [], "planner_calls": 0, "answer_calls": 0}

    class ContextProvider(LLMProvider):
        provider_name = "deepseek-test-double"
        model_name = "deepseek-test-model"

        def configure_course_policy(
            self,
            *,
            course_id,
            use_course_references,
            currently_authorized,
        ):
            self.data_policy = ExternalModelDataPolicy.from_ids({int(course_id)})
            self.course_reference_status = self.data_policy.require_course_context(
                course_id=course_id,
                use_course_references=use_course_references,
                currently_authorized=currently_authorized,
            )
            return self.course_reference_status

        def set_outbound_guard(self, guard):
            self.guard = guard

        def decide_next_action(self, state_summary, available_tools):
            raise AssertionError("server-owned state path must be used")

        def decide_from_state(self, state, available_tools):
            captured["planner_calls"] += 1
            captured["planner"].append(
                self.data_policy.planner_summary(
                    state,
                    status=self.course_reference_status,
                )
            )
            if state.evidence:
                return ActionDecision(
                    action=AgentAction.ANSWER,
                    reason="current evidence is ready",
                    task_complete=True,
                )
            return ActionDecision(
                action=AgentAction.RETRIEVE,
                arguments={"query": "隔离日志 对照 评价指标"},
                reason="retrieve current course evidence",
            )

        def generate_answer(self, request: LLMRequest) -> LLMResponse:
            raise AssertionError("practice guide must use structured generation")

        def generate_structured(self, request, schema):
            captured["answer_calls"] += 1
            projected = self.data_policy.project_request(
                request,
                status=self.course_reference_status,
            )
            captured["requests"].append(projected)
            return TemplateProvider().generate_structured(projected, schema)

    try:
        token, user = _login(client, "student_demo")
        headers = _headers(token)
        assignment, step = _first_assignment(client, token)
        class_id = client.get(
            "/api/classes", headers=headers, params={"course_id": assignment["course_id"]}
        ).json()[0]["id"]
        main.learning_logger.upsert_assignment_progress(
            user_id=int(user["id"]),
            assignment_id=int(assignment["id"]),
            step_id=int(step["id"]),
            status="stuck",
            progress_percent=60,
            note="已完成字段清洗，正在比较正常序列与重复序列的误报率。",
        )
        active_report = main.learning_logger.create_stuck_report(
            user_id=int(user["id"]),
            assignment_id=int(assignment["id"]),
            step_id=int(step["id"]),
            module="5G NAS",
            reason="误报率计算口径不确定，希望教师确认评价指标。",
        )
        main.learning_logger.update_stuck_report_status(int(active_report["id"]), "confirmed")
        resolved_sentinel = "RESOLVED_PROJECT31_SENTINEL"
        resolved_report = main.learning_logger.create_stuck_report(
            user_id=int(user["id"]),
            assignment_id=int(assignment["id"]),
            step_id=int(step["id"]),
            module="5G NAS",
            reason=resolved_sentinel,
        )
        main.learning_logger.update_stuck_report_status(int(resolved_report["id"]), "resolved")
        other_user = main.learning_logger.user_by_username("student_demo2")
        assert other_user is not None
        other_sentinel = "OTHER_STUDENT_PROJECT31_SENTINEL"
        main.learning_logger.upsert_assignment_progress(
            user_id=int(other_user["id"]),
            assignment_id=int(assignment["id"]),
            step_id=int(step["id"]),
            status="stuck",
            progress_percent=15,
            note=other_sentinel,
        )
        main.learning_logger.create_stuck_report(
            user_id=int(other_user["id"]),
            assignment_id=int(assignment["id"]),
            step_id=int(step["id"]),
            module="5G NAS",
            reason=other_sentinel,
        )
        course_text = (
            "研究目标是比较隔离日志中的正常序列和重复序列。"
            "公开防御说明参考 https://public.example/home/guide。"
            "方法采用同一字段集合设计受控对照。工具只需离线解析器和本地统计表。"
            "实验步骤1：实验记录输入编号、预期转移、实际转移、拒绝原因和处理时延。"
            "结果分析比较命中、误报和处理时延。创新点是可复核的对照记录。"
            "风险是只使用隔离合成数据，不执行真实网络攻击。"
        )
        document = Document(
            source_path=tmp_path / "project31.md",
            filename="project31.md",
            file_type="md",
            title="Project 31 course evidence",
            text=course_text,
            pages=[Page(text=course_text, page=1, section="5G NAS", source_type="markdown")],
        )
        saved = main.learning_logger.save_document(
            document=document,
            content_hash=sha256_bytes(course_text.encode("utf-8")),
            chunks=[course_text],
            byte_size=len(course_text.encode("utf-8")),
            index_status="INDEXED",
            course_id=int(assignment["course_id"]),
        )
        main.learning_logger.set_document_publication(int(saved["id"]), published=True, actor_id=1)
        chunk = main.learning_logger.chunks_for_document(int(saved["id"]))[0]

        class Retriever:
            def retrieve(self, query, top_k=None, course_id=None, **_kwargs):
                return RetrievalResponse(
                    results=[
                        RetrievalResult(
                            document_id=int(saved["id"]),
                            chunk_id=int(chunk["id"]),
                            source=document.filename,
                            title=document.title,
                            snippet=course_text,
                            page=1,
                            section="5G NAS",
                            source_type="md",
                            keyword_score=1.0,
                            vector_score=0.0,
                            final_score=1.0,
                            metadata={"content_hash": saved["content_hash"]},
                        )
                    ],
                    abstained=False,
                    threshold=0.18,
                    fallback_used=True,
                )

        provider = ContextProvider()
        monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
        monkeypatch.setattr(
            main,
            "AgentOrchestrator",
            lambda **kwargs: RealAgentOrchestrator(retriever=Retriever(), **kwargs),
        )
        session_id = "project31-context-session"
        assert client.post(
            "/api/learning/sessions",
            headers=headers,
            json={
                "client_session_id": session_id,
                "course_id": assignment["course_id"],
                "class_id": class_id,
            },
        ).status_code == 200

        response = client.post(
            "/api/chat",
            headers=headers,
            json={
                "question": "结合我当前选择的实验步骤，说明下一步应如何在隔离日志中记录对照与评价指标；若看不到步骤上下文请明说。",
                "client_session_id": session_id,
                "course_id": assignment["course_id"],
                "class_id": class_id,
                "assignment_id": assignment["id"],
                "step_id": step["id"],
                "use_course_references": True,
            },
        )

        assert response.status_code == 200, response.text
        assert response.json()["run_status"] == "COMPLETED", response.json()
        assert captured["planner_calls"] == 2
        assert captured["answer_calls"] == 1
        planner_payload = "\n".join(captured["planner"])
        assert assignment["title"] in planner_payload
        assert step["title"] in planner_payload
        assert "60%" in planner_payload
        assert "误报率计算口径不确定" in planner_payload
        assert "https://public.example/home/guide" in planner_payload
        assert other_sentinel not in planner_payload
        answer_context = captured["requests"][0].metadata["learning_context"]
        assert answer_context["progress_percent"] == 60
        assert "assignment_id" not in str(answer_context)
        assert "step_id" not in str(answer_context)
        assert "https://public.example/home/guide" in str(captured["requests"][0].evidence)
        assert other_sentinel not in str(captured["requests"][0].model_dump(mode="json"))
        assert all(item.document_id is None for item in captured["requests"][0].evidence)
        assert all(item.chunk_id is None for item in captured["requests"][0].evidence)
        visible_answer = response.json()["answer"]
        assert assignment["title"] in visible_answer
        assert step["title"] in visible_answer
        assert "60%" in visible_answer
        assert "需要帮助" in visible_answer
        assert "已确认" in visible_answer
        assert "误报率计算口径不确定" in visible_answer
        assert resolved_sentinel not in visible_answer
        assert other_sentinel not in visible_answer
        for internal_identifier in (
            str(assignment["id"]),
            str(step["id"]),
            str(active_report["id"]),
            str(user["id"]),
        ):
            assert f"_id={internal_identifier}" not in visible_answer
        for internal_key in (
            "user_id",
            "course_id",
            "class_id",
            "assignment_id",
            "step_id",
            "document_id",
            "chunk_id",
            "source_path",
        ):
            assert internal_key not in captured["requests"][0].metadata
    finally:
        _restore(client)


def test_active_stuck_report_enters_teacher_analysis_and_can_be_updated(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        student_token, _ = _login(client, "student_demo")
        teacher_token, _ = _login(client, "teacher_demo")
        student_headers = _headers(student_token)
        teacher_headers = _headers(teacher_token)
        assignment, step = _first_assignment(client, student_token)
        sid = "stuck-report-session"
        client.post("/api/learning/sessions", headers=student_headers, json={"client_session_id": sid})

        report = client.post(
            "/api/stuck-reports",
            headers=student_headers,
            json={
                "assignment_id": assignment["id"],
                "step_id": step["id"],
                "module": "5G NAS",
                "reason": "Cannot decide a safe replay-risk evaluation metric.",
                "client_session_id": sid,
            },
        )
        assert report.status_code == 200
        report_id = report.json()["report"]["id"]

        analysis = client.get("/api/teacher/stuck-analysis", headers=teacher_headers).json()
        assert any(item.get("id") == report_id and item["source"] == "active" for item in analysis["items"])
        class_id = client.get("/api/classes", headers=teacher_headers).json()[0]["id"]
        in_class = client.get(
            "/api/teacher/stuck-analysis",
            headers=teacher_headers,
            params={"class_id": class_id},
        ).json()
        out_of_class = client.get(
            "/api/teacher/stuck-analysis",
            headers=teacher_headers,
            params={"class_id": 9999},
        )
        assert any(item.get("id") == report_id for item in in_class["items"])
        assert out_of_class.status_code == 403

        confirmed = client.post(
            f"/api/teacher/stuck-reports/{report_id}/action",
            headers=teacher_headers,
            json={"action": "confirm"},
        )
        assert confirmed.status_code == 200
        assert confirmed.json()["report"]["status"] == "confirmed"
    finally:
        _restore(client)


def test_repeated_same_step_questions_create_explainable_suspected_stuck(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        student_token, _ = _login(client, "student_demo")
        teacher_token, _ = _login(client, "teacher_demo")
        student_headers = _headers(student_token)
        teacher_headers = _headers(teacher_token)
        assignment, step = _first_assignment(client, student_token)
        sid = "repeated-question-session"
        client.post("/api/learning/sessions", headers=student_headers, json={"client_session_id": sid})

        page_view = client.post(
            "/api/learning/events",
            headers=student_headers,
            json={
                "client_session_id": sid,
                "event_type": "PAGE_VIEW",
                "page_name": "assignment",
                "assignment_id": str(assignment["id"]),
                "metadata": {"step_id": step["id"]},
            },
        )
        assert page_view.status_code == 200
        assert client.get("/api/teacher/stuck-analysis", headers=teacher_headers).json()["items"] == []

        for index in range(2):
            response = client.post(
                "/api/learning/events",
                headers=student_headers,
                json={
                    "client_session_id": sid,
                    "event_type": "QUESTION_ASKED",
                    "page_name": "student_chat",
                    "module": "5G NAS",
                    "assignment_id": str(assignment["id"]),
                    "metadata": {"step_id": step["id"], "index": index},
                },
            )
            assert response.status_code == 200

        analysis = client.get(
            "/api/teacher/stuck-analysis",
            headers=teacher_headers,
            params={"assignment_id": assignment["id"], "step_id": step["id"], "module": "5G NAS"},
        ).json()
        suspected = [item for item in analysis["items"] if item["source"] == "suspected"]
        assert suspected
        assert suspected[0]["evidence"]["question_event_count"] == 2
        assert "reason" in suspected[0]
    finally:
        _restore(client)


def test_permissions_and_invalid_assignment_step(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        student_token, _ = _login(client, "student_demo")
        assignment, step = _first_assignment(client, student_token)

        forbidden = client.get("/api/teacher/stuck-analysis", headers=_headers(student_token))
        assert forbidden.status_code == 403

        invalid = client.post(
            f"/api/assignments/{assignment['id']}/progress",
            headers=_headers(student_token),
            json={"step_id": step["id"] + 9999, "status": "in_progress", "progress_percent": 5},
        )
        assert invalid.status_code == 400
        assert invalid.json()["error_code"] == "STEP_NOT_IN_ASSIGNMENT"
    finally:
        _restore(client)


@pytest.mark.parametrize("request_class_id", [None, 82])
def test_step_only_scope_checks_resolved_assignment_class(monkeypatch, request_class_id):
    class ScopeLogger:
        @staticmethod
        def assignment_step(step_id: int) -> dict:
            assert step_id == 91
            return {"id": 91, "assignment_id": 41}

        @staticmethod
        def assignment(assignment_id: int) -> dict:
            assert assignment_id == 41
            return {"id": 41, "course_id": 7, "class_id": 81}

        @staticmethod
        def user_in_class(user_id: int, class_id: int, course_id: int) -> bool:
            return (user_id, class_id, course_id) == (5, 81, 7)

    monkeypatch.setattr(main, "learning_logger", ScopeLogger())
    monkeypatch.setattr(main, "_user_can_access_course", lambda *_args, **_kwargs: True)

    with pytest.raises(HTTPException) as caught:
        main._validate_request_scope(
            current_user={"id": 5, "role": "student"},
            course_id=7,
            class_id=request_class_id,
            step_id=91,
        )

    assert caught.value.status_code == 409
    assert caught.value.detail["error_code"] == "ASSIGNMENT_CONTEXT_CONFLICT"


def test_scope_checks_cross_course_step_pair_and_course_level_positive(monkeypatch):
    assignments = {
        41: {"id": 41, "course_id": 7, "class_id": None},
        42: {"id": 42, "course_id": 8, "class_id": None},
    }
    steps = {
        91: {"id": 91, "assignment_id": 41},
        92: {"id": 92, "assignment_id": 42},
    }

    class ScopeLogger:
        @staticmethod
        def assignment_step(step_id: int) -> dict | None:
            return steps.get(step_id)

        @staticmethod
        def assignment(assignment_id: int) -> dict | None:
            return assignments.get(assignment_id)

    monkeypatch.setattr(main, "learning_logger", ScopeLogger())
    monkeypatch.setattr(main, "_user_can_access_course", lambda *_args, **_kwargs: True)

    main._validate_request_scope(
        current_user={"id": 5, "role": "student"},
        course_id=7,
        class_id=None,
        assignment_id=41,
        step_id=91,
    )
    with pytest.raises(HTTPException) as caught:
        main._validate_request_scope(
            current_user={"id": 5, "role": "student"},
            course_id=7,
            class_id=None,
            assignment_id=41,
            step_id=92,
        )

    assert caught.value.status_code == 409
    assert caught.value.detail["error_code"] == "STEP_CONTEXT_CONFLICT"

    with pytest.raises(HTTPException) as cross_course:
        main._validate_request_scope(
            current_user={"id": 5, "role": "student"},
            course_id=7,
            class_id=None,
            assignment_id=42,
        )
    assert cross_course.value.status_code == 409
    assert cross_course.value.detail["error_code"] == "ASSIGNMENT_CONTEXT_CONFLICT"


def test_assignment_event_scope_is_rejected_before_event_write(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, _ = _login(client, "student_demo")
        headers = _headers(token)
        assignment, step = _first_assignment(client, token)
        classes = client.get("/api/classes", headers=headers).json()
        class_id = int(classes[0]["id"])
        session_id = "assignment-event-scope"
        created = client.post(
            "/api/learning/sessions",
            headers=headers,
            json={
                "client_session_id": session_id,
                "course_id": int(assignment["course_id"]),
                "class_id": class_id,
            },
        )
        assert created.status_code == 200

        original_assignment = main.learning_logger.assignment

        def assignment_with_other_class(assignment_id: int):
            item = original_assignment(assignment_id)
            if item is not None:
                item = {**item, "class_id": class_id + 1000}
            return item

        monkeypatch.setattr(main.learning_logger, "assignment", assignment_with_other_class)
        with main.learning_logger._connect() as conn:
            before = conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0]

        response = client.post(
            "/api/learning/events",
            headers=headers,
            json={
                "client_session_id": session_id,
                "event_type": "QUESTION_ASKED",
                "assignment_id": str(assignment["id"]),
                "metadata": {"step_id": step["id"]},
            },
        )

        with main.learning_logger._connect() as conn:
            after = conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0]
        assert response.status_code in {403, 409}
        assert response.json()["error_code"] in {
            "CLASS_FORBIDDEN",
            "ASSIGNMENT_CONTEXT_CONFLICT",
        }
        assert after == before
    finally:
        _restore(client)


def test_session_a_cannot_attach_assignment_b_before_model_or_writes(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, user = _login(client, "student_demo")
        headers = _headers(token)
        assignment_a, _ = _first_assignment(client, token)
        class_a = client.get(
            "/api/classes",
            headers=headers,
            params={"course_id": assignment_a["course_id"]},
        ).json()[0]
        with main.learning_logger._connect() as conn:
            course_b = int(
                conn.execute(
                    "INSERT INTO courses (slug, name, description, assessment_method) VALUES (?, ?, '', '')",
                    ("assignment-scope-b", "Assignment Scope B"),
                ).lastrowid
            )
            class_b = int(
                conn.execute(
                    "INSERT INTO classes (course_id, slug, name, term, teacher_id) VALUES (?, ?, ?, '', NULL)",
                    (course_b, "assignment-class-b", "Assignment Class B"),
                ).lastrowid
            )
            conn.execute(
                "INSERT INTO class_members (class_id, user_id) VALUES (?, ?)",
                (class_b, int(user["id"])),
            )
            assignment_b = int(
                conn.execute(
                    "INSERT INTO assignments (course_id, module, title, description, visibility) VALUES (?, 'B', 'B task', '', 'course')",
                    (course_b,),
                ).lastrowid
            )
            step_b = int(
                conn.execute(
                    "INSERT INTO assignment_steps (assignment_id, step_index, title, description, module) VALUES (?, 1, 'B step', '', 'B')",
                    (assignment_b,),
                ).lastrowid
            )
        session_id = "assignment-session-a-task-b"
        assert client.post(
            "/api/learning/sessions",
            headers=headers,
            json={
                "client_session_id": session_id,
                "course_id": int(assignment_a["course_id"]),
                "class_id": int(class_a["id"]),
            },
        ).status_code == 200
        monkeypatch.setattr(
            main,
            "create_llm_provider",
            lambda: (_ for _ in ()).throw(AssertionError("model must not be created")),
        )
        with main.learning_logger._connect() as conn:
            before = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("agent_runs", "agent_steps", "questions", "learning_events")
            }

        response = client.post(
            "/api/practice/guide",
            headers=headers,
            json={
                "topic": "must be rejected before business work",
                "client_session_id": session_id,
                "assignment_id": assignment_b,
                "step_id": step_b,
            },
        )

        with main.learning_logger._connect() as conn:
            after = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in before
            }
        assert response.status_code == 409
        assert response.json()["error_code"] == "ASSIGNMENT_CONTEXT_CONFLICT"
        assert after == before
    finally:
        _restore(client)


@pytest.mark.parametrize(
    ("protected_field", "protected_value"),
    [
        ("assignment_id", 1),
        ("step_id", 1),
        ("client_session_id", "anonymous-protected-session"),
    ],
)
def test_anonymous_practice_rejects_protected_scope_before_generation_or_writes(
    tmp_path,
    monkeypatch,
    protected_field,
    protected_value,
):
    client = _client(tmp_path, monkeypatch)
    try:
        generated: list[str] = []
        original_guide = main.answerer.practice_guide

        def tracked_guide(topic: str):
            generated.append(topic)
            return original_guide(topic)

        monkeypatch.setattr(main.answerer, "practice_guide", tracked_guide)
        with main.learning_logger._connect() as conn:
            before = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("agent_runs", "agent_steps", "questions", "learning_events")
            }

        response = client.post(
            "/api/practice/guide",
            json={"topic": "anonymous scope must fail closed", protected_field: protected_value},
        )

        with main.learning_logger._connect() as conn:
            after = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in before
            }
        assert response.status_code == 401
        assert response.json()["error_code"] == "AUTH_REQUIRED"
        assert generated == []
        assert after == before
    finally:
        _restore(client)
