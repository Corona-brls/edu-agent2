from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend import main
from backend.auth.password import hash_password
from backend.connectors.base import ExternalConnector
from backend.connectors.registry import ConnectorRegistry
from backend.connectors.schemas import ConnectorHealth, ExternalResource, ResourceQuery
from backend.innovation.service import InnovationService
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import HybridRetriever, RetrievalResponse, RetrievalResult
from backend.resources.normalizer import content_hash
from backend.storage.logger import LearningLogger, sha256_bytes


class EmptyPortal:
    def search(self, query):
        return {"items": [], "total": 0}


class EmptyRetriever:
    def retrieve(self, query, top_k=None, course_id=None):
        class Response:
            results = []

        return Response()


class EmptyMCP:
    def call_tool(self, *args, **kwargs):
        class Result:
            success = False
            data = {}

        class Call:
            result = Result()
            fallback_used = False

        return Call()


class LocalPaperClueMCP:
    def call_tool(self, tool_name, *args, **kwargs):
        class Result:
            success = True
            data = {
                "results": [
                    {
                        "title": "Uploaded markdown research notes",
                        "snippet": "A local course note mentioning papers.",
                        "source": "notes.md",
                        "resource_type": "course_document",
                        "verification_status": "local_course_clue",
                    }
                ],
                "verified_papers": [],
                "result_type": "local_course_clues",
            }

        class Call:
            result = Result()
            fallback_used = True

        return Call()


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-innovation-suite-long-enough")
    original_logger = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "innovation.db")
    main.learning_logger.seed_demo_identity()
    _seed_evidence(main.learning_logger, tmp_path)
    client = TestClient(main.app)
    client._original_logger = original_logger  # type: ignore[attr-defined]
    return client


def _restore(client: TestClient) -> None:
    main.learning_logger = client._original_logger  # type: ignore[attr-defined]


def _login(client: TestClient, username: str = "student_demo") -> tuple[str, dict]:
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


def _commit_verified_resources(logger: LearningLogger, resources: list[ExternalResource]) -> None:
    for resource in resources:
        query = {"query": resource.title, "page": 1, "page_size": 10, "allow_remote": True}
        run_id = logger.create_connector_sync_run(
            connector_name=resource.source_name,
            query=query,
            status="RUNNING",
            cache_hit=False,
        )
        logger.commit_verified_sync(
            run_id=run_id,
            connector_name=resource.source_name,
            origin_kind="local_catalog" if resource.source_name == "standard_source" else "remote",
            query=query,
            resources=[{"resource": resource.model_dump(mode="json"), "content_hash": content_hash(resource)}],
            ttl_hours=24,
            latency_ms=0,
        )


def _seed_evidence(logger: LearningLogger, tmp_path: Path) -> None:
    document = Document(
        source_path=tmp_path / "innovation.md",
        filename="innovation.md",
        file_type=".md",
        title="5G NAS innovation notes",
        text="5G NAS replay detection and authentication experiments require baselines.",
        pages=[Page(text="5G NAS replay detection and authentication experiments require baselines.", page=1, section="notes", source_type="markdown")],
    )
    saved = logger.save_document(
        document=document,
        content_hash=sha256_bytes(document.text.encode("utf-8")),
        chunks=["5G NAS replay detection and authentication experiments require baselines."],
        byte_size=len(document.text),
        index_status="INDEXED",
        course_id=logger.default_course_id(),
    )
    logger.set_document_publication(int(saved["id"]), published=True, actor_id=1)
    resources = [
        ExternalResource(
            external_id="paper-nas-innovation",
            resource_type="paper",
            source_name="semantic_scholar",
            title="5G NAS Authentication Security Evaluation",
            abstract="A paper metadata record for evaluating NAS authentication.",
            url="https://example.test/paper",
            published_at="2025",
            tags=["5G NAS", "security"],
            module="5G NAS",
        ),
        ExternalResource(
            external_id="project-nas-lab",
            resource_type="project",
            source_name="github",
            title="nas-security-lab",
            abstract="A teaching project for NAS security experiments.",
            url="https://github.com/example/nas-security-lab",
            published_at="2026",
            tags=["project", "lab"],
            module="5G NAS",
            raw_metadata={
                "full_name": "example/nas-security-lab",
                "private": False,
                "visibility": "public",
            },
        ),
    ]
    _commit_verified_resources(logger, resources)


def test_innovation_migration_tables_exist(tmp_path):
    logger = LearningLogger(tmp_path / "innovation.db")
    with logger._connect() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'innovation_%'"
            ).fetchall()
        }

    assert {"innovation_sessions", "innovation_questions", "innovation_ideas"}.issubset(tables)


def test_start_session_recommends_two_source_types_and_persists(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, _ = _login(client)
        payload = {
            "topic": "5G NAS",
            "current_plan": "设计一个认证重放检测实验，比较基线和改进方案。",
        }
        response = client.post("/api/innovation/sessions", headers=_headers(token), json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["id"] > 0
        assert body["status"] == "ACTIVE"
        assert len(body["questions"]) >= 3
        types = {case["type"] for case in body["cases"]}
        assert {"paper", "project"}.issubset(types)
        source_fields = {
            "source_name",
            "origin_kind",
            "delivery_mode",
            "fetched_at",
            "url",
            "external_id",
            "provenance_verified",
            "topic_category",
        }
        assert all(source_fields.issubset(case) for case in body["cases"])
        assert body["limitations"]
        stored = main.learning_logger.innovation_session(int(body["id"]))
        assert stored["topic"] == "5G NAS"
        assert stored["cases"] == body["cases"]
        fetched = client.get(f"/api/innovation/sessions/{body['id']}", headers=_headers(token))
        assert fetched.status_code == 200
        assert fetched.json()["cases"] == body["cases"]
    finally:
        _restore(client)


def test_continuous_answer_and_finalize_outputs_evidence_and_methods(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, user = _login(client)
        headers = _headers(token)
        sid = "innovation-session-events"
        assert client.post("/api/learning/sessions", headers=headers, json={"client_session_id": sid}).status_code == 200
        start = client.post(
            "/api/innovation/sessions",
            headers=headers,
            json={
                "topic": "5G NAS",
                "current_plan": "比较 NAS 认证异常检测方案，关注误报率和延迟。",
                "client_session_id": sid,
            },
        ).json()
        question = start["questions"][0]
        answered = client.post(
            f"/api/innovation/sessions/{start['id']}/answers",
            headers=headers,
            json={
                "question_id": question["id"],
                "answer": "我会设置基线组、改进组，并记录误报率、检测延迟和重放样本覆盖率。",
                "client_session_id": sid,
            },
        ).json()
        finalized = client.post(
            f"/api/innovation/sessions/{start['id']}/finalize",
            headers=headers,
            json={"client_session_id": sid},
        ).json()
        timeline = client.get(f"/api/students/{user['id']}/timeline", headers=headers).json()["events"]

        assert answered["answered_question"]["answer"].startswith("我会设置基线组")
        assert answered["follow_up_question"]
        assert finalized["status"] == "FINALIZED"
        assert finalized["final_ideas"]
        first = finalized["final_ideas"][0]
        assert first["evidence"]
        assert {
            "source_name",
            "origin_kind",
            "delivery_mode",
            "fetched_at",
            "external_id",
            "provenance_verified",
        }.issubset(first["evidence"][0])
        assert first["experiment_method"]
        assert first["risks"]
        assert "公网目标" in " ".join(first["risks"])
        event_types = [item["event_type"] for item in timeline]
        assert "INNOVATION_SESSION_STARTED" in event_types
        assert "INNOVATION_QUESTION_ANSWERED" in event_types
    finally:
        _restore(client)


def test_finalized_session_accepts_new_answer_and_rebuilds_ideas_without_duplicates(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, _ = _login(client)
        headers = _headers(token)
        started_response = client.post(
            "/api/innovation/sessions",
            headers=headers,
            json={
                "topic": "5G NAS",
                "current_plan": "先整理现有方案，再根据追问补充实验约束。",
            },
        )
        assert started_response.status_code == 200
        started = started_response.json()
        session_id = int(started["id"])

        first_finalize = client.post(
            f"/api/innovation/sessions/{session_id}/finalize",
            headers=headers,
            json={},
        )
        assert first_finalize.status_code == 200
        assert first_finalize.json()["status"] == "FINALIZED"

        with main.learning_logger._connect() as conn:
            first_idea_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM innovation_ideas WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            )

        marker = "POST_FINALIZE_NEW_ANSWER_5G_NAS"
        answered = client.post(
            f"/api/innovation/sessions/{session_id}/answers",
            headers=headers,
            json={"question_id": started["questions"][0]["id"], "answer": marker},
        )
        assert answered.status_code == 200
        assert answered.json()["answered_question"]["answer"] == marker

        second_finalize = client.post(
            f"/api/innovation/sessions/{session_id}/finalize",
            headers=headers,
            json={},
        )
        assert second_finalize.status_code == 200
        rebuilt = second_finalize.json()
        assert rebuilt["status"] == "FINALIZED"
        assert marker in str(rebuilt["final_ideas"])

        stored_questions = main.learning_logger.innovation_questions(session_id)
        stored_session = main.learning_logger.innovation_session(session_id)
        with main.learning_logger._connect() as conn:
            rebuilt_idea_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM innovation_ideas WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            )

        assert any(item["answer"] == marker for item in stored_questions)
        assert stored_session is not None
        assert stored_session["final_ideas"] == rebuilt["final_ideas"]
        assert rebuilt_idea_count == len(rebuilt["final_ideas"])
        assert rebuilt_idea_count == first_idea_count
    finally:
        _restore(client)


def test_innovation_terminal_permissions_and_rejections_have_no_side_effects(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        main.learning_logger.upsert_user(
            username="admin_demo",
            display_name="Admin Demo",
            password_hash=hash_password("demo123456"),
            role="admin",
            must_change_password=False,
        )
        owner_token, owner = _login(client, "student_demo")
        teacher_token, _ = _login(client, "teacher_demo")
        admin_token, _ = _login(client, "admin_demo")
        owner_headers = _headers(owner_token)
        teacher_headers = _headers(teacher_token)
        admin_headers = _headers(admin_token)

        started_response = client.post(
            "/api/innovation/sessions",
            headers=owner_headers,
            json={"topic": "5G NAS", "current_plan": "验证终态会话的角色与撤权边界。"},
        )
        assert started_response.status_code == 200
        started = started_response.json()
        session_id = int(started["id"])
        question_id = int(started["questions"][0]["id"])

        def mutable_counts() -> tuple[int, int, int]:
            with main.learning_logger._connect() as conn:
                return (
                    int(conn.execute("SELECT COUNT(*) FROM innovation_questions").fetchone()[0]),
                    int(conn.execute("SELECT COUNT(*) FROM innovation_ideas").fetchone()[0]),
                    int(conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0]),
                )

        def assert_rejected_without_side_effects(request, expected_status: int = 403) -> None:
            before = mutable_counts()
            response = request()
            assert response.status_code == expected_status
            assert mutable_counts() == before

        before_admin_get = mutable_counts()
        admin_get = client.get(
            f"/api/innovation/sessions/{session_id}",
            headers=admin_headers,
        )
        assert admin_get.status_code == 200
        assert mutable_counts() == before_admin_get

        for request in (
            lambda: client.get(f"/api/innovation/sessions/{session_id}", headers=teacher_headers),
            lambda: client.post(
                f"/api/innovation/sessions/{session_id}/answers",
                headers=teacher_headers,
                json={"question_id": question_id, "answer": "teacher must not write"},
            ),
            lambda: client.post(
                f"/api/innovation/sessions/{session_id}/finalize",
                headers=teacher_headers,
                json={},
            ),
            lambda: client.post(
                f"/api/innovation/sessions/{session_id}/answers",
                headers=admin_headers,
                json={"question_id": question_id, "answer": "admin is read only"},
            ),
            lambda: client.post(
                f"/api/innovation/sessions/{session_id}/finalize",
                headers=admin_headers,
                json={},
            ),
        ):
            assert_rejected_without_side_effects(request)

        with main.learning_logger._connect() as conn:
            conn.execute("DELETE FROM class_members WHERE user_id = ?", (int(owner["id"]),))

        for request in (
            lambda: client.get(f"/api/innovation/sessions/{session_id}", headers=owner_headers),
            lambda: client.post(
                f"/api/innovation/sessions/{session_id}/answers",
                headers=owner_headers,
                json={"question_id": question_id, "answer": "revoked owner must not write"},
            ),
            lambda: client.post(
                f"/api/innovation/sessions/{session_id}/finalize",
                headers=owner_headers,
                json={},
            ),
        ):
            assert_rejected_without_side_effects(request)
    finally:
        _restore(client)


def test_student_cannot_access_other_students_innovation_session(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token_1, _ = _login(client, "student_demo")
        token_2, _ = _login(client, "student_demo2")
        created = client.post(
            "/api/innovation/sessions",
            headers=_headers(token_1),
            json={"topic": "5G NAS", "current_plan": "做一个防御性实验。"},
        ).json()

        denied = client.get(f"/api/innovation/sessions/{created['id']}", headers=_headers(token_2))

        assert denied.status_code == 403
    finally:
        _restore(client)


def test_existing_session_list_is_bounded_and_current_student_scope_only(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        owner_token, owner = _login(client, "student_demo")
        other_token, _ = _login(client, "student_demo2")
        teacher_token, _ = _login(client, "teacher_demo")
        owner_headers = _headers(owner_token)
        first = client.post(
            "/api/innovation/sessions",
            headers=owner_headers,
            json={"topic": "当前班级 ACTIVE", "current_plan": "隔离教学计划"},
        ).json()
        second = client.post(
            "/api/innovation/sessions",
            headers=owner_headers,
            json={"topic": "当前班级 FINALIZED", "current_plan": "隔离教学计划"},
        ).json()
        assert client.post(
            f"/api/innovation/sessions/{second['id']}/finalize",
            headers=owner_headers,
            json={},
        ).status_code == 200
        with main.learning_logger._connect() as conn:
            other_class = int(conn.execute(
                "INSERT INTO classes (course_id, slug, name) VALUES (?, ?, ?)",
                (int(first["course_id"]), "innovation-resume-other-class", "隔离另一班"),
            ).lastrowid)
            conn.execute(
                "INSERT INTO class_members (class_id, user_id) VALUES (?, ?)",
                (other_class, int(owner["id"])),
            )
        other_scope = client.post(
            "/api/innovation/sessions",
            headers=owner_headers,
            json={
                "topic": "另一班的会话",
                "current_plan": "不得出现在原班级列表",
                "course_id": first["course_id"],
                "class_id": other_class,
            },
        )
        assert other_scope.status_code == 200
        original_scope = {"course_id": first["course_id"], "class_id": first["class_id"]}
        with main.learning_logger._connect() as conn:
            before = tuple(int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in (
                "innovation_sessions", "innovation_questions", "innovation_ideas", "learning_events",
            ))
        listed = client.get("/api/innovation/sessions", headers=owner_headers, params=original_scope)
        assert listed.status_code == 200
        items = listed.json()["items"]
        assert {item["id"] for item in items} == {first["id"], second["id"]}
        assert all(set(item) == {"id", "topic", "status", "created_at", "updated_at"} for item in items)
        assert {item["status"] for item in items} == {"ACTIVE", "FINALIZED"}
        assert [item["id"] for item in items] == [
            item["id"] for item in client.get("/api/innovation/sessions", headers=owner_headers, params=original_scope).json()["items"]
        ]
        assert [item["id"] for item in client.get(
            "/api/innovation/sessions", headers=owner_headers,
            params={"course_id": first["course_id"], "class_id": other_class},
        ).json()["items"]] == [other_scope.json()["id"]]
        assert client.get("/api/innovation/sessions", headers=_headers(other_token), params=original_scope).json()["items"] == []
        assert client.get("/api/innovation/sessions", headers=_headers(teacher_token), params=original_scope).status_code == 403
        assert client.get(f"/api/innovation/sessions/{first['id']}", headers=_headers(other_token)).status_code == 403
        with main.learning_logger._connect() as conn:
            after = tuple(int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in (
                "innovation_sessions", "innovation_questions", "innovation_ideas", "learning_events",
            ))
        assert after == before
    finally:
        _restore(client)


def test_existing_session_list_and_get_reject_revoked_owner(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, owner = _login(client, "student_demo")
        headers = _headers(token)
        created = client.post(
            "/api/innovation/sessions", headers=headers,
            json={"topic": "撤权前会话", "current_plan": "隔离教学计划"},
        ).json()
        params = {"course_id": created["course_id"], "class_id": created["class_id"]}
        with main.learning_logger._connect() as conn:
            conn.execute("DELETE FROM class_members WHERE user_id = ?", (int(owner["id"]),))
        assert client.get("/api/innovation/sessions", headers=headers, params=params).status_code == 403
        assert client.get(f"/api/innovation/sessions/{created['id']}", headers=headers).status_code == 403
        with main.learning_logger._connect() as conn:
            conn.execute("UPDATE users SET enabled = 0 WHERE id = ?", (int(owner["id"]),))
        assert client.get("/api/innovation/sessions", headers=headers, params=params).status_code == 401
        assert client.get(f"/api/innovation/sessions/{created['id']}", headers=headers).status_code == 401
        with main.learning_logger._connect() as conn:
            conn.execute("UPDATE users SET enabled = 1, token_version = token_version + 1 WHERE id = ?", (int(owner["id"]),))
        assert client.get("/api/innovation/sessions", headers=headers, params=params).status_code == 401
        assert client.get(f"/api/innovation/sessions/{created['id']}", headers=headers).status_code == 401
    finally:
        _restore(client)


def test_existing_session_list_has_stable_twenty_item_limit(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, owner = _login(client, "student_demo")
        classes = main.learning_logger.classes_for_user(int(owner["id"]))
        selected = classes[0]
        ids = []
        for number in range(22):
            item = main.learning_logger.create_innovation_session(
                user_id=int(owner["id"]), course_id=int(selected["course_id"]),
                class_id=int(selected["id"]), topic=f"隔离摘要 {number}",
                current_plan="不得进入列表", cases=[], limitations=[], questions=[],
            )
            ids.append(int(item["id"]))
        response = client.get(
            "/api/innovation/sessions", headers=_headers(token),
            params={"course_id": selected["course_id"], "class_id": selected["id"]},
        )
        assert response.status_code == 200
        assert [item["id"] for item in response.json()["items"]] == list(reversed(ids[-20:]))
        assert all("current_plan" not in item and "cases" not in item for item in response.json()["items"])
    finally:
        _restore(client)


def test_question_id_must_belong_to_innovation_session(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, _ = _login(client, "student_demo")
        headers = _headers(token)
        first = client.post(
            "/api/innovation/sessions",
            headers=headers,
            json={"topic": "5G NAS", "current_plan": "设计一个防御性评测实验。"},
        ).json()
        second = client.post(
            "/api/innovation/sessions",
            headers=headers,
            json={"topic": "LLM 漏洞检测", "current_plan": "比较误报率和证据引用。"},
        ).json()

        response = client.post(
            f"/api/innovation/sessions/{first['id']}/answers",
            headers=headers,
            json={"question_id": second["questions"][0]["id"], "answer": "尝试跨会话回答。"},
        )

        assert response.status_code == 404
        assert response.json()["error_code"] == "INNOVATION_QUESTION_NOT_FOUND"
    finally:
        _restore(client)


def test_no_source_cases_are_not_fabricated(tmp_path):
    logger = LearningLogger(tmp_path / "empty-innovation.db")
    logger.seed_demo_identity()
    user = logger.user_by_username("student_demo")
    service = InnovationService(
        logger,
        resource_portal=EmptyPortal(),
        retriever=EmptyRetriever(),
        mcp_gateway=EmptyMCP(),
    )

    started = service.start_session(
        user=user,
        topic="不存在的专题",
        current_plan="只有一个粗略想法。",
    )
    finalized = service.finalize_session(user=user, session_id=int(started["id"]))

    assert started["cases"] == []
    assert finalized["final_ideas"]
    assert all(not idea["evidence"] for idea in finalized["final_ideas"])
    assert all(idea["model_suggestion"] is True for idea in finalized["final_ideas"])


def test_innovation_live_resource_queries_share_authenticated_user_source_rate_limit(tmp_path, monkeypatch):
    class EmptyLiveConnector(ExternalConnector):
        enabled = True
        supports_remote = True

        def __init__(self, name: str) -> None:
            self.name = name

        async def health_check(self) -> ConnectorHealth:
            return ConnectorHealth(name=self.name, status="ok")

        async def search(self, query: ResourceQuery) -> list[ExternalResource]:
            return []

    client = _client(tmp_path, monkeypatch)
    original_registry = main.connector_registry
    main.connector_registry = ConnectorRegistry(
        [
            EmptyLiveConnector("semantic_scholar"),
            EmptyLiveConnector("arxiv"),
            EmptyLiveConnector("github"),
            EmptyLiveConnector("nvd"),
        ]
    )
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_MAX_CALLS", 1)
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_WINDOW_SECONDS", 60.0)
    monkeypatch.setattr(InnovationService, "_rag_resources", lambda self, query, course_id: [])
    main._RESOURCE_RATE_BUCKETS.clear()
    try:
        first_token, _ = _login(client, "student_demo")
        other_token, _ = _login(client, "student_demo2")
        first = client.post(
            "/api/innovation/sessions",
            headers=_headers(first_token),
            json={"topic": "rate topic one", "current_plan": "plan one"},
        )
        limited = client.post(
            "/api/innovation/sessions",
            headers=_headers(first_token),
            json={"topic": "rate topic two", "current_plan": "plan two"},
        )
        isolated_actor = client.post(
            "/api/innovation/sessions",
            headers=_headers(other_token),
            json={"topic": "rate topic three", "current_plan": "plan three"},
        )

        assert first.status_code == 200
        assert limited.status_code == 429
        assert limited.json()["error_code"] == "RESOURCE_RATE_LIMIT"
        assert int(limited.headers["Retry-After"]) >= 1
        assert isolated_actor.status_code == 200
    finally:
        main._RESOURCE_RATE_BUCKETS.clear()
        main.connector_registry = original_registry
        _restore(client)


def test_innovation_does_not_consume_remote_quota_when_connectors_are_disabled(tmp_path, monkeypatch):
    class DisabledLiveConnector(ExternalConnector):
        enabled = False
        supports_remote = True

        def __init__(self, name: str) -> None:
            self.name = name

        async def health_check(self) -> ConnectorHealth:
            return ConnectorHealth(name=self.name, enabled=False, status="disabled")

        async def search(self, query: ResourceQuery) -> list[ExternalResource]:
            raise AssertionError("disabled connector must not be called")

    client = _client(tmp_path, monkeypatch)
    original_registry = main.connector_registry
    main.connector_registry = ConnectorRegistry(
        [
            DisabledLiveConnector("semantic_scholar"),
            DisabledLiveConnector("arxiv"),
            DisabledLiveConnector("github"),
            DisabledLiveConnector("nvd"),
        ]
    )
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_MAX_CALLS", 1)
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_WINDOW_SECONDS", 60.0)
    monkeypatch.setattr(InnovationService, "_rag_resources", lambda self, query, course_id: [])
    main._RESOURCE_RATE_BUCKETS.clear()
    try:
        token, _ = _login(client, "student_demo")
        first = client.post(
            "/api/innovation/sessions",
            headers=_headers(token),
            json={"topic": "offline topic one", "current_plan": "plan one"},
        )
        second = client.post(
            "/api/innovation/sessions",
            headers=_headers(token),
            json={"topic": "offline topic two", "current_plan": "plan two"},
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert main._RESOURCE_RATE_BUCKETS == {}
    finally:
        main._RESOURCE_RATE_BUCKETS.clear()
        main.connector_registry = original_registry
        _restore(client)


def test_innovation_local_paper_search_clues_are_not_labeled_as_papers(tmp_path):
    logger = LearningLogger(tmp_path / "local-paper-clue.db")
    logger.seed_demo_identity()
    user = logger.user_by_username("student_demo")
    service = InnovationService(
        logger,
        resource_portal=EmptyPortal(),
        retriever=EmptyRetriever(),
        mcp_gateway=LocalPaperClueMCP(),
    )

    started = service.start_session(
        user=user,
        topic="local paper clue topic",
        current_plan="Use local notes as a starting point.",
    )

    assert started["cases"] == []


def test_innovation_rag_scope_error_never_retries_without_course_scope(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-l0-scope-long-enough")
    monkeypatch.setenv("VECTOR_STORE_DIR", str((tmp_path / "l0-chroma").resolve()))
    logger = LearningLogger(tmp_path / "l0-scope.db")
    original_logger = main.learning_logger
    main.learning_logger = logger

    with logger._connect() as conn:
        course_a = int(
            conn.execute(
                "INSERT INTO courses (slug, name, description, assessment_method) VALUES ('l0-a', 'L0 A', '', '')"
            ).lastrowid
        )
        course_b = int(
            conn.execute(
                "INSERT INTO courses (slug, name, description, assessment_method) VALUES ('l0-b', 'L0 B', '', '')"
            ).lastrowid
        )
        class_a = int(
            conn.execute(
                "INSERT INTO classes (course_id, slug, name, term) VALUES (?, 'l0-a-class', 'L0 A Class', '2026')",
                (course_a,),
            ).lastrowid
        )
        class_b = int(
            conn.execute(
                "INSERT INTO classes (course_id, slug, name, term) VALUES (?, 'l0-b-class', 'L0 B Class', '2026')",
                (course_b,),
            ).lastrowid
        )
        student_b = int(
            conn.execute(
                """
                INSERT INTO users (username, display_name, password_hash, role, enabled, must_change_password)
                VALUES ('l0_student_b', 'L0 Student B', ?, 'student', 1, 0)
                """,
                (hash_password("l0-password-123"),),
            ).lastrowid
        )
        conn.execute(
            "INSERT INTO class_members (class_id, user_id) VALUES (?, ?)",
            (class_b, student_b),
        )

    marker = "L0_COURSE_A_PRIVATE_MARKER"
    document = Document(
        source_path=tmp_path / "l0-a-private.md",
        filename="l0-a-private.md",
        file_type=".md",
        title="L0 course A private document",
        text=marker,
        pages=[Page(text=marker, page=1, section="L0", source_type="markdown")],
    )
    saved = logger.save_document(
        document=document,
        content_hash=sha256_bytes(marker.encode("utf-8")),
        chunks=[{"chunk_text": marker, "page": 1, "section": "L0", "source_type": "markdown"}],
        byte_size=len(marker.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_a,
        class_id=class_a,
    )
    logger.set_document_publication(int(saved["id"]), published=True, actor_id=1)
    chunk = next(
        item
        for item in logger.document_chunks(course_id=course_a)
        if int(item["document_id"]) == int(saved["id"])
    )
    unscoped_result = RetrievalResult(
        document_id=int(saved["id"]),
        chunk_id=int(chunk["id"]),
        source=document.filename,
        title=document.title,
        snippet=marker,
        page=1,
        section="L0",
        source_type="markdown",
        keyword_score=1.0,
        vector_score=0.0,
        final_score=1.0,
    )
    calls: list[int | None] = []

    def scope_breaking_retrieve(self, query, top_k=None, course_id=None):
        calls.append(course_id)
        if course_id is not None:
            raise TypeError("simulated scoped retriever failure")
        return RetrievalResponse(
            results=[unscoped_result],
            abstained=False,
            threshold=0.0,
            fallback_used=True,
        )

    monkeypatch.setattr(HybridRetriever, "retrieve", scope_breaking_retrieve)
    monkeypatch.setattr(
        InnovationService,
        "_on_demand_resources",
            lambda self, topic, *, actor_id=None, before_remote=None: ([], []),
    )
    monkeypatch.setattr(InnovationService, "_mcp_resources", lambda self, topic: [])
    monkeypatch.setattr(
        "backend.innovation.service.ResourcePortalService.search",
        lambda self, query: {"items": [], "total": 0},
    )

    try:
        with TestClient(main.app) as client:
            login = client.post(
                "/api/auth/login",
                json={"username": "l0_student_b", "password": "l0-password-123"},
            )
            assert login.status_code == 200
            headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
            response = client.post(
                "/api/innovation/sessions",
                headers=headers,
                json={
                    "topic": "L0 scoped retrieval",
                    "current_plan": "Use only evidence from course B.",
                    "course_id": course_b,
                },
            )

        assert response.status_code == 200
        body = response.json()
        stored = logger.innovation_session(int(body["id"]))
        assert (
            calls == [course_b]
            and marker not in str(body["cases"])
            and marker not in str(stored["cases"])
        ), {
            "http_status": response.status_code,
            "calls": calls,
            "response_cases": body["cases"],
            "stored_cases": stored["cases"],
        }
        assert "Innovation scoped RAG retrieval failed: TypeError" in caplog.text
        assert "simulated scoped retriever failure" not in caplog.text
    finally:
        main.learning_logger = original_logger


def test_innovation_rag_discards_results_not_owned_by_selected_course(tmp_path):
    logger = LearningLogger(tmp_path / "l0-result-scope.db")
    with logger._connect() as conn:
        course_a = int(
            conn.execute(
                "INSERT INTO courses (slug, name, description, assessment_method) VALUES ('l0-result-a', 'L0 Result A', '', '')"
            ).lastrowid
        )
        course_b = int(
            conn.execute(
                "INSERT INTO courses (slug, name, description, assessment_method) VALUES ('l0-result-b', 'L0 Result B', '', '')"
            ).lastrowid
        )

    def save_result(course_id: int, filename: str, marker: str) -> RetrievalResult:
        document = Document(
            source_path=tmp_path / filename,
            filename=filename,
            file_type=".md",
            title=filename,
            text=marker,
            pages=[Page(text=marker, page=1, section="L0", source_type="markdown")],
        )
        saved = logger.save_document(
            document=document,
            content_hash=sha256_bytes(marker.encode("utf-8")),
            chunks=[{"chunk_text": marker, "page": 1, "section": "L0", "source_type": "markdown"}],
            byte_size=len(marker.encode("utf-8")),
            index_status="INDEXED",
            course_id=course_id,
        )
        logger.set_document_publication(int(saved["id"]), published=True, actor_id=1)
        chunk = next(
            item
            for item in logger.document_chunks(course_id=course_id)
            if int(item["document_id"]) == int(saved["id"])
        )
        return RetrievalResult(
            document_id=int(saved["id"]),
            chunk_id=int(chunk["id"]),
            source=filename,
            title=filename,
            snippet=marker,
            page=1,
            section="L0",
            source_type="markdown",
            keyword_score=1.0,
            vector_score=0.0,
            final_score=1.0,
        )

    result_a = save_result(course_a, "l0-result-a.md", "L0_RESULT_A_PRIVATE")
    result_b = save_result(course_b, "l0-result-b.md", "L0_RESULT_B_VISIBLE")

    class MixedScopeRetriever:
        def retrieve(self, query, top_k=None, course_id=None):
            assert course_id == course_b
            return RetrievalResponse(
                results=[result_a, result_b],
                abstained=False,
                threshold=0.0,
                fallback_used=False,
            )

    service = InnovationService(
        logger,
        resource_portal=EmptyPortal(),
        retriever=MixedScopeRetriever(),
        mcp_gateway=EmptyMCP(),
    )
    resources = service._rag_resources("L0 result scope", course_id=course_b)

    assert [item["resource_id"] for item in resources] == [
        f"rag:{result_b.document_id}:{result_b.chunk_id}"
    ]
    assert "L0_RESULT_B_VISIBLE" in str(resources)
    assert "L0_RESULT_A_PRIVATE" not in str(resources)
    assert resources[0]["origin_kind"] == "course_document"
    assert resources[0]["delivery_mode"] == "local_rag"
    assert resources[0]["fetched_at"]
    assert resources[0]["provenance_verified"] is True
    assert resources[0]["metadata"]["course_id"] == course_b


def test_innovation_http_scope_matrix_and_rejected_requests_do_not_write(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-l0-http-matrix-long-enough")
    monkeypatch.setenv("VECTOR_STORE_DIR", str((tmp_path / "l0-http-chroma").resolve()))
    logger = LearningLogger(tmp_path / "l0-http.db")
    original_logger = main.learning_logger
    main.learning_logger = logger

    password_hash = hash_password("l0-password-123")
    with logger._connect() as conn:
        user_ids = {}
        for username, role in (
            ("l0_teacher_a", "teacher"),
            ("l0_teacher_b", "teacher"),
            ("l0_student_a", "student"),
            ("l0_student_b_matrix", "student"),
            ("l0_outsider", "student"),
            ("l0_admin", "admin"),
        ):
            user_ids[username] = int(
                conn.execute(
                    """
                    INSERT INTO users (username, display_name, password_hash, role, enabled, must_change_password)
                    VALUES (?, ?, ?, ?, 1, 0)
                    """,
                    (username, username, password_hash, role),
                ).lastrowid
            )
        course_a = int(
            conn.execute(
                "INSERT INTO courses (slug, name, description, assessment_method) VALUES ('l0-http-a', 'L0 HTTP A', '', '')"
            ).lastrowid
        )
        course_b = int(
            conn.execute(
                "INSERT INTO courses (slug, name, description, assessment_method) VALUES ('l0-http-b', 'L0 HTTP B', '', '')"
            ).lastrowid
        )
        class_a = int(
            conn.execute(
                """
                INSERT INTO classes (course_id, slug, name, term, teacher_id)
                VALUES (?, 'l0-http-a-class', 'L0 HTTP A Class', '2026', ?)
                """,
                (course_a, user_ids["l0_teacher_a"]),
            ).lastrowid
        )
        class_b = int(
            conn.execute(
                """
                INSERT INTO classes (course_id, slug, name, term, teacher_id)
                VALUES (?, 'l0-http-b-class', 'L0 HTTP B Class', '2026', ?)
                """,
                (course_b, user_ids["l0_teacher_b"]),
            ).lastrowid
        )
        conn.executemany(
            "INSERT INTO class_members (class_id, user_id) VALUES (?, ?)",
            [
                (class_a, user_ids["l0_student_a"]),
                (class_b, user_ids["l0_student_b_matrix"]),
            ],
        )

    def save_document(course_id: int, class_id: int, filename: str, marker: str) -> None:
        document = Document(
            source_path=tmp_path / filename,
            filename=filename,
            file_type=".md",
            title=filename,
            text=marker,
            pages=[Page(text=marker, page=1, section="L0", source_type="markdown")],
        )
        saved = logger.save_document(
            document=document,
            content_hash=sha256_bytes(marker.encode("utf-8")),
            chunks=[{"chunk_text": marker, "page": 1, "section": "L0", "source_type": "markdown"}],
            byte_size=len(marker.encode("utf-8")),
            index_status="INDEXED",
            course_id=course_id,
            class_id=class_id,
        )
        logger.set_document_publication(int(saved["id"]), published=True, actor_id=1)

    save_document(course_a, class_a, "l0-http-a.md", "L0_HTTP_A_PRIVATE_MARKER")
    save_document(course_b, class_b, "l0-http-b.md", "L0_HTTP_B_VISIBLE_MARKER")
    monkeypatch.setattr(
        InnovationService,
        "_on_demand_resources",
            lambda self, topic, *, actor_id=None, before_remote=None: ([], []),
    )
    monkeypatch.setattr(InnovationService, "_mcp_resources", lambda self, topic: [])

    def counts() -> tuple[int, int]:
        with logger._connect() as conn:
            return (
                int(conn.execute("SELECT COUNT(*) FROM innovation_sessions").fetchone()[0]),
                int(conn.execute("SELECT COUNT(*) FROM innovation_questions").fetchone()[0]),
            )

    try:
        with TestClient(main.app) as client:
            def headers(username: str) -> dict[str, str]:
                login = client.post(
                    "/api/auth/login",
                    json={"username": username, "password": "l0-password-123"},
                )
                assert login.status_code == 200
                return {"Authorization": f"Bearer {login.json()['access_token']}"}

            student_a = headers("l0_student_a")
            student_b = headers("l0_student_b_matrix")
            outsider = headers("l0_outsider")
            teacher_a = headers("l0_teacher_a")
            teacher_b = headers("l0_teacher_b")
            admin = headers("l0_admin")
            before = counts()
            anonymous = client.post(
                "/api/innovation/sessions",
                json={"topic": "L0", "current_plan": "anonymous", "course_id": course_b},
            )
            cross_student = client.post(
                "/api/innovation/sessions",
                headers=student_a,
                json={"topic": "L0", "current_plan": "cross course", "course_id": course_b},
            )
            unassigned = client.post(
                "/api/innovation/sessions",
                headers=outsider,
                json={"topic": "L0", "current_plan": "no membership", "course_id": course_b},
            )
            cross_teacher = client.get(
                "/api/resources/portal",
                headers=teacher_a,
                params={"course_id": course_b},
            )
            after_rejections = counts()
            teacher_b_portal = client.get(
                "/api/resources/portal",
                headers=teacher_b,
                params={"course_id": course_b, "resource_type": "course_document"},
            )
            admin_portal = client.get(
                "/api/resources/portal",
                headers=admin,
                params={"course_id": course_b, "resource_type": "course_document"},
            )
            own_course = client.post(
                "/api/innovation/sessions",
                headers=student_b,
                json={
                    "topic": "L0_HTTP_B_VISIBLE_MARKER",
                    "current_plan": "Use the indexed evidence from course B.",
                    "course_id": course_b,
                },
            )
            after_success = counts()

        assert anonymous.status_code == 401
        assert cross_student.status_code == 403
        assert cross_student.json()["error_code"] == "COURSE_FORBIDDEN"
        assert unassigned.status_code == 403
        assert unassigned.json()["error_code"] == "COURSE_FORBIDDEN"
        assert cross_teacher.status_code == 403
        assert cross_teacher.json()["error_code"] == "COURSE_FORBIDDEN"
        assert after_rejections == before
        assert teacher_b_portal.status_code == 200
        assert admin_portal.status_code == 200
        assert own_course.status_code == 200
        assert "L0_HTTP_B_VISIBLE_MARKER" in str(own_course.json()["cases"])
        assert "L0_HTTP_A_PRIVATE_MARKER" not in str(own_course.json()["cases"])
        assert after_success[0] == before[0] + 1
        assert after_success[1] > before[1]
    finally:
        main.learning_logger = original_logger
