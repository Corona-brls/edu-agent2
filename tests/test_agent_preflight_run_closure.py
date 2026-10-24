from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.agent.state import AgentRunStatus, AgentState
from backend.analytics.events import LearningEventService
from backend.auth.service import create_access_token
from backend.core.config import get_settings
from backend.llm.deepseek_provider import DeepSeekProvider
from backend.llm.schemas import EvidenceItem
from backend.llm.template_provider import TemplateProvider
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import RetrievalResponse, RetrievalResult
from backend.storage.logger import LearningLogger, sha256_bytes


PRIVATE_MARKER = "R1A_PRIVATE_COURSE_MARKER"
LATE_CHAT_CANDIDATE = "R1A_LATE_CHAT_CANDIDATE"
LATE_CONTINUE_CANDIDATE = "R1A_LATE_CONTINUE_CANDIDATE"
LATE_PRACTICE_CANDIDATE = "R1A_LATE_PRACTICE_CANDIDATE"
LATE_ROUTE_CANDIDATE = "迟到候选正文"
ATOMIC_MODEL_CANDIDATE = "R1C2_ATOMIC_MODEL_CANDIDATE_6E91"
ATOMIC_PRIVATE_MARKER = "R1C2_ATOMIC_PRIVATE_MARKER_29B4"
DISABLED_ACCOUNT_CANDIDATE = "R1C4_DISABLED_ACCOUNT_CANDIDATE_81E3"
REMOVED_MEMBER_CANDIDATE = "R1C4_REMOVED_MEMBER_CANDIDATE_4A72"
UNEXPECTED_LATER_STAGE = "R1C4_UNEXPECTED_LATER_STAGE_6F90"
RETRY_FAILURE_MARKER = "R1C5_RETRY_FAILURE_MARKER_93B7"
BEFORE_CREATE_CANDIDATE = "R1C6A_BEFORE_CREATE_CANDIDATE_5D28"
C6A2_SECOND_PLANNER_CANDIDATE = "R1C6A2_SECOND_PLANNER_CANDIDATE_7D31"

PRACTICE_TEXT = (
    f"{PRIVATE_MARKER} 的研究目标是比较正常认证与重放检测在课程隔离日志中的差异。"
    f"{PRIVATE_MARKER} 的方法设计是使用同一字段集合分别标注正常认证样本与重放检测样本。"
    f"{PRIVATE_MARKER} 的工具选择是离线日志解析器、表格统计工具和本地可视化工具。"
    f"{PRIVATE_MARKER} 的实验步骤一是准备隔离日志；步骤二是标注正常认证；"
    "步骤三是标注重放检测；步骤四是运行离线解析；步骤五是比较结果；步骤六是保存记录。"
    f"{PRIVATE_MARKER} 的结果分析只比较认证事件数量、检测命中情况和误报情况。"
    f"{PRIVATE_MARKER} 的创新点是把两类样本放入同一受控字段框架中进行可复核比较。"
    f"{PRIVATE_MARKER} 的风险与注意事项是仅使用隔离样本，不接触公网目标，不收集真实用户数据。"
)


class _NoLiveGateway:
    is_available = False

    @staticmethod
    def get_agent_tool_catalog(*, role: str):
        del role
        return []


class _StoredDocumentRetriever:
    def __init__(
        self,
        *,
        document_id: int,
        chunk_id: int,
        text: str = PRACTICE_TEXT,
        source: str = "private-course.md",
        title: str = "Private course",
        section: str = "practice",
    ) -> None:
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.title = title
        self.section = section
        self.calls: list[dict[str, int | str | None]] = []

    def retrieve(self, query: str, *, course_id: int | None = None, top_k: int | None = None):
        self.calls.append({"query": query, "course_id": course_id, "top_k": top_k})
        return RetrievalResponse(
            results=[
                RetrievalResult(
                    document_id=self.document_id,
                    chunk_id=self.chunk_id,
                    source=self.source,
                    title=self.title,
                    snippet=self.text,
                    page=1,
                    section=self.section,
                    source_type="md",
                    keyword_score=1.0,
                    vector_score=0.0,
                    final_score=1.0,
                    metadata={"content_hash": "test-content-hash"},
                )
            ],
            abstained=False,
            threshold=0.18,
            fallback_used=False,
        )


class _ScriptedCompletions:
    def __init__(
        self,
        responses: list[str],
        *,
        mutate_on_call: dict[int, object] | None = None,
        block_on_call: int | None = None,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.responses = list(responses)
        self.mutate_on_call = dict(mutate_on_call or {})
        self.block_on_call = block_on_call
        self.entered = entered
        self.release = release
        self.requests: list[dict] = []

    def create(self, **request):
        self.requests.append(deepcopy(request))
        call_number = len(self.requests)
        callback = self.mutate_on_call.get(call_number)
        if callable(callback):
            callback()
        if self.block_on_call == call_number:
            assert self.entered is not None
            assert self.release is not None
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise AssertionError("timed out waiting to release blocked fake model call")
        if not self.responses:
            raise AssertionError("unexpected model call")
        return {"choices": [{"message": {"content": self.responses.pop(0)}}]}


class _BeforeCreateBarrierDeepSeekProvider(DeepSeekProvider):
    """Pause one transport attempt after outbound-before guard but before SDK create."""

    def __init__(
        self,
        *,
        settings,
        client,
        barrier_on_chat_once: int,
        entered: threading.Event,
        release: threading.Event,
        finished: threading.Event | None = None,
    ) -> None:
        super().__init__(settings=settings, client=client)
        self.barrier_on_chat_once = barrier_on_chat_once
        self.entered = entered
        self.release = release
        self.finished = finished
        self.chat_once_count = 0
        self.logical_stages: list[str] = []

    def decide_next_action(self, state_summary, available_tools):
        self.logical_stages.append("planner")
        return super().decide_next_action(state_summary, available_tools)

    def generate_answer(self, request):
        self.logical_stages.append("answer")
        return super().generate_answer(request)

    def generate_structured(self, request, schema):
        self.logical_stages.append("structured")
        return super().generate_structured(request, schema)

    def _chat_once(
        self,
        messages,
        *,
        response_format=None,
        outbound_guard=None,
        extra_body=None,
        sdk_max_retries=None,
    ):
        self.chat_once_count += 1
        call_number = self.chat_once_count
        try:
            if call_number == self.barrier_on_chat_once:
                self.entered.set()
                if not self.release.wait(timeout=5):
                    raise AssertionError("timed out waiting to release pre-create barrier")
            return super()._chat_once(
                messages,
                response_format=response_format,
                outbound_guard=outbound_guard,
                extra_body=extra_body,
                sdk_max_retries=sdk_max_retries,
            )
        finally:
            if call_number == self.barrier_on_chat_once and self.finished is not None:
                self.finished.set()


def _planner(action: str, *, reason: str) -> str:
    payload = {
        "action": action,
        "arguments": {"query": "课程实践资料"} if action == "retrieve" else {},
        "reason": reason,
        "task_complete": action == "answer",
    }
    return json.dumps(payload, ensure_ascii=False)


def _practice_candidate() -> str:
    claim = {
        "claim_text": LATE_PRACTICE_CANDIDATE,
        "source_id": "S1",
        "evidence_span_id": "S1:text:1",
    }
    return json.dumps(
        {
            "research_goal": [claim],
            "method_design": [claim],
            "tools": [claim],
            "experiment_steps": [claim],
            "result_analysis": [claim],
            "innovation_points": [claim],
            "risks": [claim],
        },
        ensure_ascii=False,
    )


def _provider(
    course_id: int,
    responses: list[str],
    *,
    mutate_on_call=None,
    block_on_call: int | None = None,
    entered: threading.Event | None = None,
    release: threading.Event | None = None,
):
    completions = _ScriptedCompletions(
        responses,
        mutate_on_call=mutate_on_call,
        block_on_call=block_on_call,
        entered=entered,
        release=release,
    )
    settings = get_settings()
    provider = DeepSeekProvider(
        settings=settings,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    assert course_id in settings.external_model_course_allowlist
    return provider, completions


@pytest.fixture
def isolated_http_context(tmp_path, monkeypatch):
    database = (tmp_path / "r1a.db").resolve()
    knowledge = (tmp_path / "knowledge").resolve()
    vectors = (tmp_path / "vectors").resolve()
    uploads = (tmp_path / "uploads").resolve()
    knowledge.mkdir()
    vectors.mkdir()
    uploads.mkdir()
    monkeypatch.setenv("SQLITE_DB_PATH", str(database))
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(knowledge))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(vectors))
    monkeypatch.setenv("UPLOAD_DIR", str(uploads))
    monkeypatch.setenv("JWT_SECRET_KEY", "r1a-isolated-jwt-secret-0123456789")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-r1a-key")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "5")

    storage = LearningLogger(database)
    context = storage.seed_demo_identity()
    course_id = int(context["course_id"])
    class_id = int(context["class_id"])
    monkeypatch.setenv("EXTERNAL_MODEL_COURSE_ALLOWLIST", str(course_id))
    with storage._connect() as conn:
        conn.execute("UPDATE users SET must_change_password = 0")
    user = storage.user_by_username("student_demo")
    assert user is not None

    document = Document(
        source_path=knowledge / "private-course.md",
        filename="private-course.md",
        file_type=".md",
        title="Private course",
        text=PRACTICE_TEXT,
        pages=[Page(text=PRACTICE_TEXT, page=1, section="practice", source_type="md")],
    )
    saved = storage.save_document(
        document=document,
        content_hash=sha256_bytes(PRACTICE_TEXT.encode("utf-8")),
        chunks=[{"chunk_text": PRACTICE_TEXT, "page": 1, "section": "practice", "source_type": "md"}],
        byte_size=len(PRACTICE_TEXT.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
        class_id=class_id,
        uploaded_by_user_id=int(user["id"]),
    )
    document_id = int(saved["id"])
    storage.set_document_publication(document_id, published=True, actor_id=int(user["id"]))
    chunk_id = int(storage.chunks_for_document(document_id)[0]["id"])

    session_id = "r1a-session"
    LearningEventService(storage).create_session(
        user=user,
        client_session_id=session_id,
        course_id=course_id,
        class_id=class_id,
    )
    token, _ = create_access_token(storage.user(int(user["id"])) or user)
    headers = {"Authorization": f"Bearer {token}"}

    monkeypatch.setattr(main, "learning_logger", storage)
    monkeypatch.setattr(main, "mcp_gateway", _NoLiveGateway())
    retriever = _StoredDocumentRetriever(document_id=document_id, chunk_id=chunk_id)
    monkeypatch.setattr("backend.agent.orchestrator.HybridRetriever", lambda **_kwargs: retriever)
    client = TestClient(main.app)
    yield SimpleNamespace(
        storage=storage,
        user=user,
        course_id=course_id,
        class_id=class_id,
        document_id=document_id,
        chunk_id=chunk_id,
        session_id=session_id,
        headers=headers,
        client=client,
        retriever=retriever,
    )
    client.close()


def _counts(storage: LearningLogger) -> tuple[int, int]:
    with storage._connect() as conn:
        questions = int(conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0])
        events = int(conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0])
    return questions, events


def test_real_http_plain_course_answer_sends_only_relevant_complete_span(
    isolated_http_context,
    monkeypatch,
):
    """The production Answer handoff must not approve an unrelated same-chunk tail."""

    ctx = isolated_http_context
    query_token = "R1C1R_QUERY_TOKEN_7B42"
    selected_marker = "R1C1R_SELECTED_MARKER_29C1"
    tail_marker = "R1C1R_UNSELECTED_TAIL_46F8"
    identity_marker = "user_id=R1C1R_IDENTITY_SENTINEL_58D3"
    path_marker = r"C:\R1C1R_PATH_SENTINEL_84A6\private.md"
    selected = f"{selected_marker} {query_token} is the relevant complete sentence."
    tail = f"{tail_marker} is an unrelated complete sentence."
    canonical_text = f"{selected} {tail}"
    document = Document(
        source_path=(ctx.storage.db_path.parent / "r1c1r-course.md").resolve(),
        filename="r1c1r-course.md",
        file_type=".md",
        title="R1C1R_TITLE_SENTINEL.md",
        text=canonical_text,
        pages=[Page(text=canonical_text, page=1, section="scope", source_type="md")],
    )
    saved = ctx.storage.save_document(
        document=document,
        content_hash=sha256_bytes(canonical_text.encode("utf-8")),
        chunks=[{"chunk_text": canonical_text, "page": 1, "section": "scope", "source_type": "md"}],
        byte_size=len(canonical_text.encode("utf-8")),
        index_status="INDEXED",
        course_id=ctx.course_id,
        class_id=ctx.class_id,
        uploaded_by_user_id=int(ctx.user["id"]),
    )
    document_id = int(saved["id"])
    ctx.storage.set_document_publication(
        document_id,
        published=True,
        actor_id=int(ctx.user["id"]),
    )
    chunk_id = int(ctx.storage.chunks_for_document(document_id)[0]["id"])
    retriever = _StoredDocumentRetriever(
        document_id=document_id,
        chunk_id=chunk_id,
        text=canonical_text,
        source=path_marker,
        title="R1C1R_TITLE_SENTINEL.md",
        section=identity_marker,
    )
    monkeypatch.setattr("backend.agent.orchestrator.HybridRetriever", lambda **_kwargs: retriever)
    provider, completions = _provider(
        ctx.course_id,
        [
            _planner("retrieve", reason="collect the requested course evidence"),
            f"{query_token} is supported by the selected course statement [S1].",
        ],
    )
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)

    response = ctx.client.post(
        "/api/chat",
        headers=ctx.headers,
        json={
            "question": f"请根据课程资料解释 {query_token}",
            "course_id": ctx.course_id,
            "class_id": ctx.class_id,
            "client_session_id": ctx.session_id,
            "use_course_references": True,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["run_status"] == AgentRunStatus.COMPLETED
    assert len(completions.requests) == 2
    outbound = json.dumps(completions.requests[-1]["messages"], ensure_ascii=False)
    assert selected_marker in outbound
    assert f"[S1:text:1] {selected}" in outbound
    assert tail_marker not in outbound
    assert "R1C1R_TITLE_SENTINEL.md" not in outbound
    assert identity_marker not in outbound
    assert path_marker not in outbound
    run = ctx.storage.agent_run(response.json()["run_id"])
    assert run is not None
    assert run["state"]["evidence"][0]["text"] == canonical_text
    assert ctx.storage.chunks_for_document(document_id)[0]["chunk_text"] == canonical_text


def _raw_run(storage: LearningLogger, run_id: str) -> dict:
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT question_id, status, current_step, tool_call_count, state_json,
                   pending_question, final_answer, stop_reason
            FROM agent_runs WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


def _create_trusted_requires_input_run(ctx, *, run_id: str) -> AgentState:
    evidence = EvidenceItem(
        source_id="S1",
        text=PRACTICE_TEXT,
        source="private-course.md",
        title="Private course",
        document_id=ctx.document_id,
        chunk_id=ctx.chunk_id,
        page=1,
        section="practice",
        source_type="md",
        resource_type="course",
        origin_kind="course_document",
        delivery_mode="local_rag",
        trusted=True,
    )
    state = AgentState(
        run_id=run_id,
        user_id=str(ctx.user["id"]),
        course_id=ctx.course_id,
        class_id=ctx.class_id,
        user_goal="比较正常认证与重放检测",
        intent="course_qa",
        status=AgentRunStatus.REQUIRES_INPUT,
        current_step=1,
        tool_call_count=0,
        pending_question="请补充期望输出",
        stop_reason="REQUIRES_INPUT",
        evidence=[evidence],
        required_sources=["course"],
        completed_sources=["course"],
    )
    state.add_user_message(state.user_goal)
    ctx.storage.create_agent_run(
        run_id=state.run_id,
        user_id=state.user_id,
        goal=state.user_goal,
        intent=state.intent,
        status=state.status.value,
        max_steps=state.max_steps,
        max_tool_calls=state.max_tool_calls,
        state=state.model_dump(mode="json"),
        course_id=ctx.course_id,
        class_id=ctx.class_id,
    )
    ctx.storage.update_agent_run(
        run_id=state.run_id,
        status=state.status.value,
        state=state.model_dump(mode="json"),
        stop_reason=state.stop_reason,
        final_answer=None,
        pending_question=state.pending_question,
    )
    ctx.storage.create_agent_step(
        run_id=state.run_id,
        step_id=f"{run_id}-prior-step",
        step_number=1,
        action="ask_user",
        status=AgentRunStatus.REQUIRES_INPUT.value,
        decision={"question": state.pending_question},
        observation={"summary": "trusted prior snapshot"},
        trace_id=state.trace_id,
    )
    ctx.storage.log_agent_trace(
        trace_id=state.trace_id,
        run_id=state.run_id,
        step_id=f"{run_id}-prior-step",
        question_id=None,
        event_type="prior_snapshot",
        message="trusted prior trace",
        metadata={"safe": True},
    )
    return state


def _audit_after(storage: LearningLogger, run_id: str, checkpoint: dict) -> dict[str, list[dict]]:
    with storage._connect() as conn:
        steps = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM agent_steps WHERE run_id = ? AND id > ? ORDER BY id",
                (run_id, checkpoint["step_watermark"]),
            ).fetchall()
        ]
        traces = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM agent_traces WHERE run_id = ? AND id > ? ORDER BY id",
                (run_id, checkpoint["trace_watermark"]),
            ).fetchall()
        ]
        models = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM model_calls WHERE run_id = ? AND id > ? ORDER BY id",
                (run_id, checkpoint["model_call_watermark"]),
            ).fetchall()
        ]
        tools = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM tool_calls WHERE run_id = ? AND id > ? ORDER BY id",
                (run_id, checkpoint["tool_call_watermark"]),
            ).fetchall()
        ]
    audit = {"steps": steps, "traces": traces, "models": models, "tools": tools}
    watermarks = {
        "steps": "step_watermark",
        "traces": "trace_watermark",
        "models": "model_call_watermark",
        "tools": "tool_call_watermark",
    }
    for category, rows in audit.items():
        watermark = int(checkpoint[watermarks[category]])
        for row in rows:
            assert row["run_id"] == run_id
            assert int(row["id"]) > watermark
    return audit


def _assert_safe_attempt(
    audit: dict[str, list[dict]],
    *,
    run_id: str,
    error_code: str,
    forbidden: str,
    expected_model_count: int,
    expected_tool_count: int,
    extra_forbidden: tuple[str, ...] = (),
) -> None:
    assert audit["steps"]
    assert audit["traces"]
    assert len(audit["models"]) == expected_model_count
    assert len(audit["tools"]) == expected_tool_count
    attempt_step_ids = {str(row["step_id"]) for row in audit["steps"]}
    for category in ("steps", "traces", "models", "tools"):
        for row in audit[category]:
            assert row["run_id"] == run_id
    assert all(row["status"] == "FAILED" for row in audit["steps"])
    assert all(row["action"] == "security_preflight" for row in audit["steps"])
    assert all(row["error_code"] == error_code for row in audit["steps"])
    assert all(row["event_type"] == "model_context_rejected" for row in audit["traces"])
    if expected_model_count:
        for row in audit["models"]:
            assert str(row["step_id"]) in attempt_step_ids
            assert int(row["success"]) == 0
            assert row["error_code"] == error_code
            assert json.loads(row["metadata_json"])["error_code"] == error_code
    else:
        assert audit["models"] == []
    if expected_tool_count:
        for row in audit["tools"]:
            assert str(row["step_id"]) in attempt_step_ids
            assert int(row["success"]) == 0
            assert json.loads(row["arguments_json"]) == {}
            assert json.loads(row["result_json"])["error_code"] == error_code
    else:
        assert audit["tools"] == []
    serialized = json.dumps(audit, ensure_ascii=False, default=str)
    assert forbidden not in serialized
    for marker in extra_forbidden:
        assert marker not in serialized
    assert PRIVATE_MARKER not in serialized
    assert "planning" not in serialized


def test_real_http_new_chat_closes_safely_when_account_is_disabled_during_answer(
    isolated_http_context,
    monkeypatch,
):
    ctx = isolated_http_context
    entered = threading.Event()
    release = threading.Event()
    provider, completions = _provider(
        ctx.course_id,
        [
            _planner("retrieve", reason="collect course evidence"),
            f"{DISABLED_ACCOUNT_CANDIDATE} [S1]",
            UNEXPECTED_LATER_STAGE,
        ],
        block_on_call=2,
        entered=entered,
        release=release,
    )
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
    before_counts = _counts(ctx.storage)
    outcome: dict[str, object] = {}

    def post_chat() -> None:
        try:
            outcome["response"] = ctx.client.post(
                "/api/chat",
                headers=ctx.headers,
                json={
                    "question": "请解释课程实践资料",
                    "course_id": ctx.course_id,
                    "class_id": ctx.class_id,
                    "client_session_id": ctx.session_id,
                    "use_course_references": True,
                },
            )
        except BaseException as exc:  # pragma: no cover - surfaced in the main test thread
            outcome["error"] = exc

    request_thread = threading.Thread(target=post_chat, name="r1c4-disable-chat", daemon=True)
    account_disabled = False
    request_thread.start()
    try:
        assert entered.wait(timeout=3), "Answer generation did not reach the controlled barrier"
        with ctx.storage._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET enabled = 0 WHERE id = ?",
                (int(ctx.user["id"]),),
            )
            assert cursor.rowcount == 1
        account_disabled = True
        release.set()
        request_thread.join(timeout=5)
    finally:
        release.set()
        request_thread.join(timeout=5)
        if account_disabled:
            with ctx.storage._connect() as conn:
                conn.execute(
                    "UPDATE users SET enabled = 1 WHERE id = ?",
                    (int(ctx.user["id"]),),
                )

    assert not request_thread.is_alive(), "controlled HTTP worker did not stop"
    restored_user = ctx.storage.user(int(ctx.user["id"]))
    assert restored_user is not None
    assert bool(restored_user["enabled"]) is True
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    response = outcome["response"]
    assert response.status_code == 403, response.text  # type: ignore[union-attr]
    detail = response.json()  # type: ignore[union-attr]
    assert detail["error_code"] == "MODEL_CONTEXT_REVOKED"
    assert detail["course_reference_status"] == "revoked"
    run_id = detail["run_id"]
    assert run_id
    assert len(completions.requests) == 2
    assert all(request.get("messages") for request in completions.requests)
    assert completions.responses == [UNEXPECTED_LATER_STAGE]
    assert _counts(ctx.storage) == before_counts

    run = ctx.storage.agent_run(run_id)
    assert run is not None
    assert run["status"] == AgentRunStatus.FAILED
    assert run["current_step"] == 0
    assert run["tool_call_count"] == 0
    assert run["pending_question"] is None
    assert run["final_answer"] is None
    assert run["stop_reason"] == "MODEL_CONTEXT_REVOKED"
    assert run["state"]["status"] == AgentRunStatus.FAILED
    assert run["state"]["final_answer"] is None
    assert DISABLED_ACCOUNT_CANDIDATE not in json.dumps(run["state"], ensure_ascii=False)
    audit = _audit_after(
        ctx.storage,
        run_id,
        {"step_watermark": 0, "trace_watermark": 0, "model_call_watermark": 0, "tool_call_watermark": 0},
    )
    _assert_safe_attempt(
        audit,
        run_id=run_id,
        error_code="MODEL_CONTEXT_REVOKED",
        forbidden=DISABLED_ACCOUNT_CANDIDATE,
        expected_model_count=1,
        expected_tool_count=0,
        extra_forbidden=(UNEXPECTED_LATER_STAGE,),
    )


def test_real_http_continue_restores_snapshot_when_membership_is_removed_during_answer(
    isolated_http_context,
    monkeypatch,
):
    ctx = isolated_http_context
    state = _create_trusted_requires_input_run(ctx, run_id="r1c4-membership-run")
    checkpoint = ctx.storage.agent_attempt_checkpoint(state.run_id)
    raw_before = _raw_run(ctx.storage, state.run_id)
    prior_steps = deepcopy(ctx.storage.agent_steps(state.run_id))
    prior_traces = deepcopy(ctx.storage.agent_trace_rows(state.run_id))
    before_counts = _counts(ctx.storage)
    entered = threading.Event()
    release = threading.Event()
    provider, completions = _provider(
        ctx.course_id,
        [
            _planner("answer", reason="supplement received"),
            f"{REMOVED_MEMBER_CANDIDATE} [S1]",
            UNEXPECTED_LATER_STAGE,
        ],
        block_on_call=2,
        entered=entered,
        release=release,
    )
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
    outcome: dict[str, object] = {}

    def post_continue() -> None:
        try:
            outcome["response"] = ctx.client.post(
                f"/api/agent/runs/{state.run_id}/continue",
                headers=ctx.headers,
                json={
                    "user_input": "继续比较正常认证与重放检测",
                    "client_session_id": ctx.session_id,
                    "use_course_references": True,
                },
            )
        except BaseException as exc:  # pragma: no cover - surfaced in the main test thread
            outcome["error"] = exc

    request_thread = threading.Thread(target=post_continue, name="r1c4-remove-member", daemon=True)
    membership_removed = False
    request_thread.start()
    try:
        assert entered.wait(timeout=3), "Answer generation did not reach the controlled barrier"
        with ctx.storage._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM class_members WHERE class_id = ? AND user_id = ?",
                (ctx.class_id, int(ctx.user["id"])),
            )
            assert cursor.rowcount == 1
        membership_removed = True
        release.set()
        request_thread.join(timeout=5)
    finally:
        release.set()
        request_thread.join(timeout=5)
        if membership_removed:
            with ctx.storage._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO class_members (class_id, user_id) VALUES (?, ?)",
                    (ctx.class_id, int(ctx.user["id"])),
                )

    assert not request_thread.is_alive(), "controlled HTTP worker did not stop"
    assert ctx.storage.user_in_class(
        int(ctx.user["id"]),
        ctx.class_id,
        ctx.course_id,
    )
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    response = outcome["response"]
    assert response.status_code == 403, response.text  # type: ignore[union-attr]
    detail = response.json()  # type: ignore[union-attr]
    assert detail["error_code"] == "MODEL_CONTEXT_REVOKED"
    assert detail["course_reference_status"] == "revoked"
    assert detail["run_id"] == state.run_id
    assert len(completions.requests) == 2
    assert all(request.get("messages") for request in completions.requests)
    assert completions.responses == [UNEXPECTED_LATER_STAGE]
    assert _raw_run(ctx.storage, state.run_id) == raw_before
    assert _counts(ctx.storage) == before_counts
    assert ctx.storage.agent_steps(state.run_id)[: len(prior_steps)] == prior_steps
    assert ctx.storage.agent_trace_rows(state.run_id)[: len(prior_traces)] == prior_traces

    audit = _audit_after(ctx.storage, state.run_id, checkpoint)
    _assert_safe_attempt(
        audit,
        run_id=state.run_id,
        error_code="MODEL_CONTEXT_REVOKED",
        forbidden=REMOVED_MEMBER_CANDIDATE,
        expected_model_count=1,
        expected_tool_count=0,
        extra_forbidden=(UNEXPECTED_LATER_STAGE,),
    )
    restored = ctx.storage.agent_run(state.run_id)
    assert restored is not None
    assert restored["status"] == AgentRunStatus.REQUIRES_INPUT
    assert restored["pending_question"] == "请补充期望输出"
    assert restored["current_step"] == 1
    assert restored["tool_call_count"] == 0
    assert restored["final_answer"] is None
    assert restored["stop_reason"] == "REQUIRES_INPUT"
    assert restored["state"] == json.loads(raw_before["state_json"])

    legal_provider, legal_calls = _provider(
        ctx.course_id,
        [
            _planner("answer", reason="membership restored"),
            "研究目标是比较正常认证与重放检测在课程隔离日志中的差异 [S1]",
        ],
    )
    monkeypatch.setattr(main, "create_llm_provider", lambda: legal_provider)
    resumed = ctx.client.post(
        f"/api/agent/runs/{state.run_id}/continue",
        headers=ctx.headers,
        json={
            "user_input": "恢复成员资格后继续",
            "client_session_id": ctx.session_id,
            "use_course_references": True,
        },
    )
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["run_id"] == state.run_id
    assert resumed.json()["run_status"] == AgentRunStatus.COMPLETED
    assert len(legal_calls.requests) == 2
    assert REMOVED_MEMBER_CANDIDATE not in resumed.text


def test_real_http_retry_rechecks_authorization_before_second_physical_create(
    isolated_http_context,
    monkeypatch,
):
    ctx = isolated_http_context
    provider, completions = _provider(ctx.course_id, [])
    provider.settings = replace(provider.settings, llm_max_retries=1)
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
    requested_backoffs: list[float] = []
    monkeypatch.setattr(
        "backend.llm.deepseek_provider.time.sleep",
        lambda seconds: requested_backoffs.append(float(seconds)),
    )
    before_counts = _counts(ctx.storage)
    account_disabled = False

    def fail_first_physical_create(**request):
        nonlocal account_disabled
        completions.requests.append(deepcopy(request))
        with ctx.storage._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET enabled = 0 WHERE id = ?",
                (int(ctx.user["id"]),),
            )
            assert cursor.rowcount == 1
        account_disabled = True
        raise TimeoutError(RETRY_FAILURE_MARKER)

    completions.create = fail_first_physical_create
    try:
        response = ctx.client.post(
            "/api/chat",
            headers=ctx.headers,
            json={
                "question": "请解释课程实践资料",
                "course_id": ctx.course_id,
                "class_id": ctx.class_id,
                "client_session_id": ctx.session_id,
                "use_course_references": True,
            },
        )
    finally:
        if account_disabled:
            with ctx.storage._connect() as conn:
                conn.execute(
                    "UPDATE users SET enabled = 1 WHERE id = ?",
                    (int(ctx.user["id"]),),
                )

    restored_user = ctx.storage.user(int(ctx.user["id"]))
    assert restored_user is not None
    assert bool(restored_user["enabled"]) is True
    assert response.status_code == 403, response.text
    detail = response.json()
    assert detail["error_code"] == "MODEL_CONTEXT_REVOKED"
    assert detail["course_reference_status"] == "revoked"
    run_id = detail["run_id"]
    assert run_id
    assert len(completions.requests) == 1
    assert completions.requests[0].get("messages")
    assert len(requested_backoffs) == 1
    assert requested_backoffs[0] >= 0
    assert _counts(ctx.storage) == before_counts

    run = ctx.storage.agent_run(run_id)
    assert run is not None
    assert run["status"] == AgentRunStatus.FAILED
    assert run["current_step"] == 0
    assert run["tool_call_count"] == 0
    assert run["pending_question"] is None
    assert run["final_answer"] is None
    assert run["stop_reason"] == "MODEL_CONTEXT_REVOKED"
    assert run["state"]["status"] == AgentRunStatus.FAILED
    assert run["state"]["final_answer"] is None
    assert RETRY_FAILURE_MARKER not in json.dumps(run["state"], ensure_ascii=False)
    audit = _audit_after(
        ctx.storage,
        run_id,
        {"step_watermark": 0, "trace_watermark": 0, "model_call_watermark": 0, "tool_call_watermark": 0},
    )
    _assert_safe_attempt(
        audit,
        run_id=run_id,
        error_code="MODEL_CONTEXT_REVOKED",
        forbidden=RETRY_FAILURE_MARKER,
        expected_model_count=0,
        expected_tool_count=0,
    )


def test_real_chat_rechecks_unpublish_after_retrieval_before_second_planner_create(
    isolated_http_context,
    monkeypatch,
):
    ctx = isolated_http_context
    session_id = "r1c6a2-chat-planner-session"
    LearningEventService(ctx.storage).create_session(
        user=ctx.user,
        client_session_id=session_id,
        course_id=ctx.course_id,
        class_id=ctx.class_id,
    )
    before_counts = _counts(ctx.storage)
    entered = threading.Event()
    release = threading.Event()
    worker_finished = threading.Event()
    completions = _ScriptedCompletions(
        [
            _planner("retrieve", reason="collect course evidence before replanning"),
            C6A2_SECOND_PLANNER_CANDIDATE,
        ]
    )
    settings = replace(
        get_settings(),
        llm_timeout_seconds=10,
        agent_timeout_seconds=15,
    )
    provider = _BeforeCreateBarrierDeepSeekProvider(
        settings=settings,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        barrier_on_chat_once=2,
        entered=entered,
        release=release,
        finished=worker_finished,
    )
    assert ctx.course_id in settings.external_model_course_allowlist
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
    outcome: dict[str, object] = {}

    def post_chat() -> None:
        try:
            outcome["response"] = ctx.client.post(
                "/api/chat",
                headers=ctx.headers,
                json={
                    "question": "请根据课程资料给出实验步骤",
                    "course_id": ctx.course_id,
                    "class_id": ctx.class_id,
                    "client_session_id": session_id,
                    "use_course_references": True,
                },
            )
        except BaseException as exc:  # pragma: no cover - surfaced in the main test thread
            outcome["error"] = exc

    request_thread = threading.Thread(
        target=post_chat,
        name="r1c6a2-chat-before-planner2-create",
        daemon=True,
    )
    unpublished = False
    request_thread.start()
    try:
        assert entered.wait(timeout=3), "Planner#2 did not reach the pre-create barrier"
        assert provider.chat_once_count == 2
        assert provider.logical_stages == ["planner", "planner"]
        assert len(completions.requests) == 1
        assert completions.responses == [C6A2_SECOND_PLANNER_CANDIDATE]
        assert len(ctx.retriever.calls) == 1
        assert ctx.retriever.calls[0]["course_id"] == ctx.course_id
        with ctx.storage._connect() as conn:
            rows = conn.execute("SELECT run_id FROM agent_runs ORDER BY id").fetchall()
        assert len(rows) == 1
        in_flight_run_id = str(rows[0]["run_id"])
        in_flight = ctx.storage.agent_run(in_flight_run_id)
        assert in_flight is not None
        assert in_flight["state"]["intent"] == "practice_guide"
        assert in_flight["state"]["completed_sources"] == ["course"]
        assert len(in_flight["state"]["evidence"]) == 1
        assert int(in_flight["state"]["evidence"][0]["document_id"]) == ctx.document_id
        assert int(in_flight["state"]["evidence"][0]["chunk_id"]) == ctx.chunk_id
        current_document = ctx.storage.document(ctx.document_id)
        assert current_document is not None
        assert current_document["index_status"] == "INDEXED"
        assert current_document["published_to_students"] is True
        ctx.storage.set_document_publication(
            ctx.document_id,
            published=False,
            actor_id=int(ctx.user["id"]),
        )
        unpublished = True
        assert ctx.storage.document(ctx.document_id)["published_to_students"] is False
        release.set()
        assert worker_finished.wait(timeout=5), "Planner#2 worker did not finish"
        request_thread.join(timeout=5)
    finally:
        release.set()
        worker_finished.wait(timeout=5)
        request_thread.join(timeout=5)
        if unpublished:
            ctx.storage.set_document_publication(
                ctx.document_id,
                published=True,
                actor_id=int(ctx.user["id"]),
            )

    assert not request_thread.is_alive(), "controlled HTTP worker did not stop"
    assert worker_finished.is_set()
    restored_document = ctx.storage.document(ctx.document_id)
    assert restored_document is not None
    assert restored_document["published_to_students"] is True
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    response = outcome["response"]
    assert response.status_code == 403, response.text  # type: ignore[union-attr]
    detail = response.json()  # type: ignore[union-attr]
    assert detail["error_code"] == "MODEL_CONTEXT_REVOKED"
    assert detail["course_reference_status"] == "revoked"
    run_id = detail["run_id"]
    assert run_id == in_flight_run_id
    assert len(completions.requests) == 1
    assert all(request.get("messages") for request in completions.requests)
    assert PRIVATE_MARKER not in json.dumps(completions.requests, ensure_ascii=False)
    assert completions.responses == [C6A2_SECOND_PLANNER_CANDIDATE]
    assert _counts(ctx.storage) == before_counts

    run = ctx.storage.agent_run(run_id)
    assert run is not None
    assert run["status"] == AgentRunStatus.FAILED
    assert run["final_answer"] is None
    assert run["stop_reason"] == "MODEL_CONTEXT_REVOKED"
    assert run["state"]["status"] == AgentRunStatus.FAILED
    assert run["state"]["final_answer"] is None
    assert C6A2_SECOND_PLANNER_CANDIDATE not in json.dumps(run["state"], ensure_ascii=False)
    audit = _audit_after(
        ctx.storage,
        run_id,
        {"step_watermark": 0, "trace_watermark": 0, "model_call_watermark": 0, "tool_call_watermark": 0},
    )
    _assert_safe_attempt(
        audit,
        run_id=run_id,
        error_code="MODEL_CONTEXT_REVOKED",
        forbidden=C6A2_SECOND_PLANNER_CANDIDATE,
        expected_model_count=1,
        expected_tool_count=0,
    )


def test_real_continue_rechecks_session_immediately_before_answer_sdk_create(
    isolated_http_context,
    monkeypatch,
):
    ctx = isolated_http_context
    state = _create_trusted_requires_input_run(ctx, run_id="r1c6a-pre-create-run")
    checkpoint = ctx.storage.agent_attempt_checkpoint(state.run_id)
    raw_before = _raw_run(ctx.storage, state.run_id)
    prior_steps = deepcopy(ctx.storage.agent_steps(state.run_id))
    prior_traces = deepcopy(ctx.storage.agent_trace_rows(state.run_id))
    before_counts = _counts(ctx.storage)
    entered = threading.Event()
    release = threading.Event()
    worker_finished = threading.Event()
    completions = _ScriptedCompletions(
        [
            _planner("answer", reason="supplement received"),
            f"{BEFORE_CREATE_CANDIDATE} [S1]",
        ]
    )
    settings = replace(
        get_settings(),
        llm_timeout_seconds=10,
        agent_timeout_seconds=15,
    )
    provider = _BeforeCreateBarrierDeepSeekProvider(
        settings=settings,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        barrier_on_chat_once=2,
        entered=entered,
        release=release,
        finished=worker_finished,
    )
    assert ctx.course_id in settings.external_model_course_allowlist
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
    outcome: dict[str, object] = {}

    def post_continue() -> None:
        try:
            outcome["response"] = ctx.client.post(
                f"/api/agent/runs/{state.run_id}/continue",
                headers=ctx.headers,
                json={
                    "user_input": "继续比较正常认证与重放检测",
                    "client_session_id": ctx.session_id,
                    "use_course_references": True,
                },
            )
        except BaseException as exc:  # pragma: no cover - surfaced in the main test thread
            outcome["error"] = exc

    request_thread = threading.Thread(
        target=post_continue,
        name="r1c6a-before-answer-create",
        daemon=True,
    )
    request_thread.start()
    try:
        assert entered.wait(timeout=3), "Answer did not reach the pre-create barrier"
        assert provider.chat_once_count == 2
        assert len(completions.requests) == 1
        LearningEventService(ctx.storage).end_session(
            user=ctx.user,
            client_session_id=ctx.session_id,
        )
        release.set()
        assert worker_finished.wait(timeout=5), "Answer worker did not finish"
        request_thread.join(timeout=5)
    finally:
        release.set()
        worker_finished.wait(timeout=5)
        request_thread.join(timeout=5)

    assert not request_thread.is_alive(), "controlled HTTP worker did not stop"
    assert worker_finished.is_set()
    ended_session = ctx.storage.learning_session_by_client_id(ctx.session_id)
    assert ended_session is not None
    assert ended_session["ended_at"] is not None
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    response = outcome["response"]
    assert response.status_code == 409, response.text  # type: ignore[union-attr]
    detail = response.json()  # type: ignore[union-attr]
    assert detail["error_code"] == "SESSION_ENDED"
    assert detail["course_reference_status"] == "revoked"
    assert detail["run_id"] == state.run_id
    assert _raw_run(ctx.storage, state.run_id) == raw_before
    assert _counts(ctx.storage) == before_counts
    assert ctx.storage.agent_steps(state.run_id)[: len(prior_steps)] == prior_steps
    assert ctx.storage.agent_trace_rows(state.run_id)[: len(prior_traces)] == prior_traces

    audit = _audit_after(ctx.storage, state.run_id, checkpoint)
    _assert_safe_attempt(
        audit,
        run_id=state.run_id,
        error_code="SESSION_ENDED",
        forbidden=BEFORE_CREATE_CANDIDATE,
        expected_model_count=1,
        expected_tool_count=0,
    )
    actual_physical_create_count = len(completions.requests)
    if actual_physical_create_count != 1:
        pytest.fail(
            "Answer SDK create ran after the learning session ended inside the "
            "outbound-before-guard to physical-create window; "
            f"actual_create_count={actual_physical_create_count}",
            pytrace=False,
        )


def test_real_strict_practice_rechecks_pending_after_structured_create_before_return(
    isolated_http_context,
    monkeypatch,
):
    ctx = isolated_http_context
    session_id = "r1c6a2-strict-practice-session"
    LearningEventService(ctx.storage).create_session(
        user=ctx.user,
        client_session_id=session_id,
        course_id=ctx.course_id,
        class_id=ctx.class_id,
    )
    before_counts = _counts(ctx.storage)
    entered = threading.Event()
    release = threading.Event()
    worker_finished = threading.Event()
    completions = _ScriptedCompletions(
        [
            _planner("retrieve", reason="collect strict practice evidence"),
            _planner("answer", reason="course evidence is complete"),
            _practice_candidate(),
        ],
        block_on_call=3,
        entered=entered,
        release=release,
    )
    settings = replace(
        get_settings(),
        llm_timeout_seconds=10,
        agent_timeout_seconds=15,
    )

    class _StructuredWorkerTrackingProvider(DeepSeekProvider):
        def __init__(self):
            super().__init__(
                settings=settings,
                client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            )
            self.chat_once_count = 0

        def _chat_once(
            self,
            messages,
            *,
            response_format=None,
            outbound_guard=None,
            extra_body=None,
            sdk_max_retries=None,
        ):
            self.chat_once_count += 1
            call_number = self.chat_once_count
            try:
                return super()._chat_once(
                    messages,
                    response_format=response_format,
                    outbound_guard=outbound_guard,
                    extra_body=extra_body,
                    sdk_max_retries=sdk_max_retries,
                )
            finally:
                if call_number == 3:
                    worker_finished.set()

    provider = _StructuredWorkerTrackingProvider()
    assert ctx.course_id in settings.external_model_course_allowlist
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
    outcome: dict[str, object] = {}

    def post_practice() -> None:
        try:
            outcome["response"] = ctx.client.post(
                "/api/practice/guide",
                headers=ctx.headers,
                json={
                    "topic": "请根据课程资料生成实践指导",
                    "course_id": ctx.course_id,
                    "class_id": ctx.class_id,
                    "client_session_id": session_id,
                    "use_course_references": True,
                },
            )
        except BaseException as exc:  # pragma: no cover - surfaced in the main test thread
            outcome["error"] = exc

    request_thread = threading.Thread(
        target=post_practice,
        name="r1c6a2-practice-structured-inflight",
        daemon=True,
    )
    request_thread.start()
    try:
        assert entered.wait(timeout=3), "Structured create did not reach the in-flight barrier"
        assert provider.chat_once_count == 3
        assert len(completions.requests) == 3
        assert all(request.get("messages") for request in completions.requests)
        assert len(ctx.retriever.calls) == 1
        assert ctx.retriever.calls[0]["course_id"] == ctx.course_id
        current_document = ctx.storage.document(ctx.document_id)
        assert current_document is not None
        assert current_document["index_status"] == "INDEXED"
        assert current_document["published_to_students"] is True
        ctx.storage.update_document_index_status(ctx.document_id, "PENDING")
        pending_document = ctx.storage.document(ctx.document_id)
        assert pending_document is not None
        assert pending_document["index_status"] == "PENDING"
        assert pending_document["published_to_students"] is False
        release.set()
        assert worker_finished.wait(timeout=5), "Structured worker did not finish"
        request_thread.join(timeout=5)
    finally:
        release.set()
        worker_finished.wait(timeout=5)
        request_thread.join(timeout=5)

    assert not request_thread.is_alive(), "controlled HTTP worker did not stop"
    assert worker_finished.is_set()
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    response = outcome["response"]
    assert response.status_code == 403, response.text  # type: ignore[union-attr]
    detail = response.json()  # type: ignore[union-attr]
    assert detail["error_code"] == "MODEL_CONTEXT_REVOKED"
    assert detail["course_reference_status"] == "revoked"
    run_id = detail["run_id"]
    assert run_id
    assert len(completions.requests) == 3
    assert completions.responses == []
    assert _counts(ctx.storage) == before_counts
    pending_document = ctx.storage.document(ctx.document_id)
    assert pending_document is not None
    assert pending_document["index_status"] == "PENDING"
    assert pending_document["published_to_students"] is False

    run = ctx.storage.agent_run(run_id)
    assert run is not None
    assert run["status"] == AgentRunStatus.FAILED
    assert run["final_answer"] is None
    assert run["stop_reason"] == "MODEL_CONTEXT_REVOKED"
    assert run["state"]["status"] == AgentRunStatus.FAILED
    assert run["state"]["final_answer"] is None
    assert LATE_PRACTICE_CANDIDATE not in json.dumps(run["state"], ensure_ascii=False)
    audit = _audit_after(
        ctx.storage,
        run_id,
        {"step_watermark": 0, "trace_watermark": 0, "model_call_watermark": 0, "tool_call_watermark": 0},
    )
    _assert_safe_attempt(
        audit,
        run_id=run_id,
        error_code="MODEL_CONTEXT_REVOKED",
        forbidden=LATE_PRACTICE_CANDIDATE,
        expected_model_count=2,
        expected_tool_count=0,
    )


@pytest.mark.parametrize(
    ("endpoint", "candidate", "mutation", "expected_calls"),
    [
        ("chat", LATE_CHAT_CANDIDATE, "unpublish", 2),
        ("practice", LATE_PRACTICE_CANDIDATE, "reindex", 3),
    ],
)
def test_real_new_run_is_failed_and_sanitized_when_context_changes_after_model_return(
    isolated_http_context,
    monkeypatch,
    endpoint,
    candidate,
    mutation,
    expected_calls,
):
    ctx = isolated_http_context
    before_counts = _counts(ctx.storage)

    def revoke():
        if mutation == "unpublish":
            ctx.storage.set_document_publication(
                ctx.document_id,
                published=False,
                actor_id=int(ctx.user["id"]),
            )
        else:
            ctx.storage.update_document_index_status(ctx.document_id, "PENDING")

    responses = [_planner("retrieve", reason="collect course evidence")]
    if endpoint == "practice":
        responses.append(_planner("answer", reason="evidence collected"))
        responses.append(_practice_candidate())
    else:
        responses.append(candidate)
    provider, completions = _provider(
        ctx.course_id,
        responses,
        mutate_on_call={expected_calls: revoke},
    )
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)

    if endpoint == "chat":
        response = ctx.client.post(
            "/api/chat",
            headers=ctx.headers,
            json={
                "question": "请解释课程实践资料",
                "course_id": ctx.course_id,
                "class_id": ctx.class_id,
                "client_session_id": ctx.session_id,
                "use_course_references": True,
            },
        )
    else:
        response = ctx.client.post(
            "/api/practice/guide",
            headers=ctx.headers,
            json={
                "topic": "请根据课程资料生成实践指导",
                "course_id": ctx.course_id,
                "class_id": ctx.class_id,
                "client_session_id": ctx.session_id,
                "use_course_references": True,
            },
        )

    assert response.status_code == 403, (response.text, len(completions.requests))
    detail = response.json()
    assert detail["error_code"] == "MODEL_CONTEXT_REVOKED"
    run_id = detail["run_id"]
    assert run_id
    assert len(completions.requests) == expected_calls
    assert _counts(ctx.storage) == before_counts

    run = ctx.storage.agent_run(run_id)
    assert run is not None
    assert run["status"] == AgentRunStatus.FAILED
    assert run["current_step"] == 0
    assert run["tool_call_count"] == 0
    assert run["pending_question"] is None
    assert run["final_answer"] is None
    assert run["stop_reason"] == "MODEL_CONTEXT_REVOKED"
    assert run["state"]["status"] == AgentRunStatus.FAILED
    assert run["state"]["final_answer"] is None
    assert run["state"]["course_id"] == ctx.course_id
    assert run["state"]["class_id"] == ctx.class_id
    assert "请" in run["state"]["user_goal"]
    assert candidate not in json.dumps(run["state"], ensure_ascii=False)

    audit = _audit_after(
        ctx.storage,
        run_id,
        {"step_watermark": 0, "trace_watermark": 0, "model_call_watermark": 0, "tool_call_watermark": 0},
    )
    _assert_safe_attempt(
        audit,
        run_id=run_id,
        error_code="MODEL_CONTEXT_REVOKED",
        forbidden=candidate,
        expected_model_count=expected_calls - 1,
        expected_tool_count=0,
    )


def test_route_final_preflight_clears_an_already_completed_real_run(
    isolated_http_context,
    monkeypatch,
):
    ctx = isolated_http_context
    before_counts = _counts(ctx.storage)
    provider, completions = _provider(
        ctx.course_id,
        [
            _planner("retrieve", reason="collect course evidence"),
            (
                f"{LATE_ROUTE_CANDIDATE}：研究目标是比较正常认证与重放检测"
                "在课程隔离日志中的差异 [S1]"
            ),
        ],
    )
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
    original_run = main.AgentOrchestrator.run
    revoked = False

    def run_then_revoke(self, *args, **kwargs):
        nonlocal revoked
        result = original_run(self, *args, **kwargs)
        assert result["run_status"] == AgentRunStatus.COMPLETED
        assert LATE_ROUTE_CANDIDATE in result["answer"]
        if not revoked:
            ctx.storage.set_document_publication(
                ctx.document_id,
                published=False,
                actor_id=int(ctx.user["id"]),
            )
            revoked = True
        return result

    monkeypatch.setattr(main.AgentOrchestrator, "run", run_then_revoke)
    response = ctx.client.post(
        "/api/chat",
        headers=ctx.headers,
        json={
            "question": "比较正常认证与重放检测",
            "course_id": ctx.course_id,
            "class_id": ctx.class_id,
            "client_session_id": ctx.session_id,
            "use_course_references": True,
        },
    )

    assert response.status_code == 403, response.text
    detail = response.json()
    assert detail["error_code"] == "MODEL_CONTEXT_REVOKED"
    run_id = detail["run_id"]
    assert run_id
    assert revoked is True
    assert len(completions.requests) == 2
    assert _counts(ctx.storage) == before_counts
    run = ctx.storage.agent_run(run_id)
    assert run is not None
    assert run["status"] == AgentRunStatus.FAILED
    assert run["stop_reason"] == "MODEL_CONTEXT_REVOKED"
    assert run["final_answer"] is None
    assert LATE_ROUTE_CANDIDATE not in json.dumps(run["state"], ensure_ascii=False)
    audit = _audit_after(
        ctx.storage,
        run_id,
        {"step_watermark": 0, "trace_watermark": 0, "model_call_watermark": 0, "tool_call_watermark": 0},
    )
    _assert_safe_attempt(
        audit,
        run_id=run_id,
        error_code="MODEL_CONTEXT_REVOKED",
        forbidden=LATE_ROUTE_CANDIDATE,
        expected_model_count=2,
        expected_tool_count=0,
    )


def test_real_continue_restores_exact_snapshot_after_session_ends_and_can_resume_with_new_session(
    isolated_http_context,
    monkeypatch,
):
    ctx = isolated_http_context
    evidence = EvidenceItem(
        source_id="S1",
        text=PRACTICE_TEXT,
        source="private-course.md",
        title="Private course",
        document_id=ctx.document_id,
        chunk_id=ctx.chunk_id,
        page=1,
        section="practice",
        source_type="md",
        resource_type="course",
        origin_kind="course_document",
        delivery_mode="local_rag",
        trusted=True,
    )
    state = AgentState(
        run_id="r1a-existing-run",
        user_id=str(ctx.user["id"]),
        course_id=ctx.course_id,
        class_id=ctx.class_id,
        user_goal="比较正常认证与重放检测",
        intent="course_qa",
        status=AgentRunStatus.REQUIRES_INPUT,
        current_step=1,
        tool_call_count=0,
        pending_question="请补充期望输出",
        stop_reason="REQUIRES_INPUT",
        evidence=[evidence],
        required_sources=["course"],
        completed_sources=["course"],
    )
    state.add_user_message(state.user_goal)
    ctx.storage.create_agent_run(
        run_id=state.run_id,
        user_id=state.user_id,
        goal=state.user_goal,
        intent=state.intent,
        status=state.status.value,
        max_steps=state.max_steps,
        max_tool_calls=state.max_tool_calls,
        state=state.model_dump(mode="json"),
        course_id=ctx.course_id,
        class_id=ctx.class_id,
    )
    ctx.storage.update_agent_run(
        run_id=state.run_id,
        status=state.status.value,
        state=state.model_dump(mode="json"),
        stop_reason=state.stop_reason,
        final_answer=state.final_answer,
        pending_question=state.pending_question,
    )
    ctx.storage.create_agent_step(
        run_id=state.run_id,
        step_id="prior-step",
        step_number=1,
        action="ask_user",
        status=AgentRunStatus.REQUIRES_INPUT.value,
        decision={"question": state.pending_question},
        observation={"summary": "trusted prior snapshot"},
        trace_id=state.trace_id,
    )
    ctx.storage.log_agent_trace(
        trace_id=state.trace_id,
        run_id=state.run_id,
        step_id="prior-step",
        question_id=None,
        event_type="prior_snapshot",
        message="trusted prior trace",
        metadata={"safe": True},
    )
    checkpoint = ctx.storage.agent_attempt_checkpoint(state.run_id)
    raw_before = _raw_run(ctx.storage, state.run_id)
    prior_steps = deepcopy(ctx.storage.agent_steps(state.run_id))
    prior_traces = deepcopy(ctx.storage.agent_trace_rows(state.run_id))
    before_counts = _counts(ctx.storage)

    def end_session():
        LearningEventService(ctx.storage).end_session(
            user=ctx.user,
            client_session_id=ctx.session_id,
        )

    provider, completions = _provider(
        ctx.course_id,
        [
            _planner("answer", reason="supplement received"),
            f"{LATE_CONTINUE_CANDIDATE} [S1]",
        ],
        mutate_on_call={2: end_session},
    )
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
    response = ctx.client.post(
        f"/api/agent/runs/{state.run_id}/continue",
        headers=ctx.headers,
        json={
            "user_input": "继续比较正常认证与重放检测",
            "client_session_id": ctx.session_id,
            "use_course_references": True,
        },
    )

    assert response.status_code == 409
    detail = response.json()
    assert detail["error_code"] == "SESSION_ENDED"
    assert detail["run_id"] == state.run_id
    assert len(completions.requests) == 2
    assert _raw_run(ctx.storage, state.run_id) == raw_before
    assert _counts(ctx.storage) == before_counts
    assert ctx.storage.agent_steps(state.run_id)[: len(prior_steps)] == prior_steps
    assert ctx.storage.agent_trace_rows(state.run_id)[: len(prior_traces)] == prior_traces
    audit = _audit_after(ctx.storage, state.run_id, checkpoint)
    _assert_safe_attempt(
        audit,
        run_id=state.run_id,
        error_code="SESSION_ENDED",
        forbidden=LATE_CONTINUE_CANDIDATE,
        expected_model_count=1,
        expected_tool_count=0,
    )

    restored = ctx.storage.agent_run(state.run_id)
    assert restored is not None
    assert restored["status"] == AgentRunStatus.REQUIRES_INPUT
    assert restored["pending_question"] == "请补充期望输出"
    assert restored["current_step"] == 1
    assert restored["tool_call_count"] == 0
    assert restored["final_answer"] is None
    assert restored["stop_reason"] == "REQUIRES_INPUT"
    assert restored["state"] == json.loads(raw_before["state_json"])

    new_session_id = "r1a-restored-session"
    LearningEventService(ctx.storage).create_session(
        user=ctx.user,
        client_session_id=new_session_id,
        course_id=ctx.course_id,
        class_id=ctx.class_id,
    )
    legal_provider, legal_calls = _provider(
        ctx.course_id,
        [
            _planner("answer", reason="restored context is current"),
            "研究目标是比较正常认证与重放检测在课程隔离日志中的差异 [S1]",
        ],
    )
    monkeypatch.setattr(main, "create_llm_provider", lambda: legal_provider)
    resumed = ctx.client.post(
        f"/api/agent/runs/{state.run_id}/continue",
        headers=ctx.headers,
        json={
            "user_input": "继续比较正常认证与重放检测",
            "client_session_id": new_session_id,
            "use_course_references": True,
        },
    )
    assert resumed.status_code == 200, resumed.text
    assert len(legal_calls.requests) == 2
    assert resumed.json()["run_id"] == state.run_id
    assert resumed.json()["run_status"] == AgentRunStatus.COMPLETED, resumed.json()
    assert LATE_CONTINUE_CANDIDATE not in resumed.text


def test_real_continue_projection_invalid_closes_before_sdk_without_template_or_business_write(
    isolated_http_context,
    monkeypatch,
):
    ctx = isolated_http_context
    evidence = EvidenceItem(
        source_id="R1C1_FORGED_SOURCE_2A90",
        text=PRACTICE_TEXT,
        source="private-course.md",
        title="Private course",
        document_id=ctx.document_id,
        chunk_id=ctx.chunk_id,
        page=1,
        section="practice",
        source_type="md",
        resource_type="course",
        origin_kind="course_document",
        delivery_mode="local_rag",
        trusted=True,
    )
    state = AgentState(
        run_id="r1c1-invalid-projection-run",
        user_id=str(ctx.user["id"]),
        course_id=ctx.course_id,
        class_id=ctx.class_id,
        user_goal="比较正常认证与重放检测",
        intent="course_qa",
        status=AgentRunStatus.REQUIRES_INPUT,
        current_step=1,
        pending_question="请补充期望输出",
        stop_reason="REQUIRES_INPUT",
        evidence=[evidence],
        required_sources=["course"],
        completed_sources=["course"],
    )
    state.add_user_message(state.user_goal)
    ctx.storage.create_agent_run(
        run_id=state.run_id,
        user_id=state.user_id,
        goal=state.user_goal,
        intent=state.intent,
        status=state.status.value,
        max_steps=state.max_steps,
        max_tool_calls=state.max_tool_calls,
        state=state.model_dump(mode="json"),
        course_id=ctx.course_id,
        class_id=ctx.class_id,
    )
    ctx.storage.update_agent_run(
        run_id=state.run_id,
        status=state.status.value,
        state=state.model_dump(mode="json"),
        stop_reason=state.stop_reason,
        final_answer=None,
        pending_question=state.pending_question,
    )
    checkpoint = ctx.storage.agent_attempt_checkpoint(state.run_id)
    raw_before = _raw_run(ctx.storage, state.run_id)
    before_counts = _counts(ctx.storage)
    provider, completions = _provider(ctx.course_id, [])
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)

    def forbid_template(*_args, **_kwargs):
        raise AssertionError("projection failure must not enter Template fallback")

    monkeypatch.setattr(TemplateProvider, "generate_answer", forbid_template)
    response = ctx.client.post(
        f"/api/agent/runs/{state.run_id}/continue",
        headers=ctx.headers,
        json={
            "user_input": "继续比较",
            "client_session_id": ctx.session_id,
            "use_course_references": True,
        },
    )

    assert response.status_code == 409, response.text
    assert response.json()["error_code"] == "MODEL_CONTEXT_PROJECTION_INVALID"
    assert completions.requests == []
    assert _raw_run(ctx.storage, state.run_id) == raw_before
    assert _counts(ctx.storage) == before_counts
    audit = _audit_after(ctx.storage, state.run_id, checkpoint)
    assert audit["steps"]
    assert audit["traces"]
    assert {row["status"] for row in audit["steps"]} == {"FAILED"}
    assert {row["error_code"] for row in audit["steps"]} == {
        "MODEL_CONTEXT_PROJECTION_INVALID"
    }
    assert {row["event_type"] for row in audit["traces"]} == {
        "model_context_rejected"
    }
    assert audit["models"] == []
    assert audit["tools"] == []
    assert "继续比较" not in json.dumps(ctx.storage.agent_run(state.run_id), ensure_ascii=False)


def test_rejected_attempt_transaction_rolls_back_and_repeated_close_is_idempotent(tmp_path):
    storage = LearningLogger(tmp_path / "atomic.db")
    context = storage.seed_demo_identity()
    state = AgentState(
        run_id="atomic-run",
        user_id="1",
        course_id=int(context["course_id"]),
        class_id=int(context["class_id"]),
        user_goal="atomic closure",
    )
    state.add_user_message(state.user_goal)
    storage.create_agent_run(
        run_id=state.run_id,
        user_id=state.user_id,
        goal=state.user_goal,
        intent=state.intent,
        status=state.status.value,
        max_steps=state.max_steps,
        max_tool_calls=state.max_tool_calls,
        state=state.model_dump(mode="json"),
        course_id=state.course_id,
        class_id=state.class_id,
    )
    checkpoint = storage.agent_attempt_checkpoint(state.run_id)
    storage.create_agent_step(
        run_id=state.run_id,
        step_id="attempt-step",
        step_number=1,
        action="planning",
        status="RUNNING",
        decision={"candidate": ATOMIC_MODEL_CANDIDATE},
        trace_id=state.trace_id,
    )
    storage.log_model_call(
        run_id=state.run_id,
        step_id="attempt-step",
        question_id=None,
        provider="deepseek",
        model="fake-r1c2-model",
        purpose="generate_answer",
        success=True,
        metadata={
            "candidate": ATOMIC_MODEL_CANDIDATE,
            "private_marker": ATOMIC_PRIVATE_MARKER,
        },
    )
    storage.log_tool_call(
        run_id=state.run_id,
        step_id="attempt-step",
        question_id=None,
        tool_name="r1c2_test_tool",
        arguments={"query": ATOMIC_PRIVATE_MARKER},
        result={"candidate": ATOMIC_MODEL_CANDIDATE},
        success=True,
        latency_ms=1.0,
        fallback_used=False,
    )
    storage.log_agent_trace(
        trace_id=state.trace_id,
        run_id=state.run_id,
        step_id="attempt-step",
        question_id=None,
        event_type="model_candidate",
        message=ATOMIC_MODEL_CANDIDATE,
        metadata={"private_marker": ATOMIC_PRIVATE_MARKER},
    )
    before = _raw_run(storage, state.run_id)
    audit_before = _audit_after(storage, state.run_id, checkpoint)
    before_serialized = json.dumps(audit_before, ensure_ascii=False, default=str)
    assert ATOMIC_MODEL_CANDIDATE in before_serialized
    assert ATOMIC_PRIVATE_MARKER in before_serialized
    with storage._connect() as conn:
        conn.execute(
            """
            CREATE TRIGGER reject_tool_closure
            BEFORE UPDATE ON tool_calls
            BEGIN
                SELECT RAISE(ABORT, 'synthetic closure failure');
            END
            """
        )
    with pytest.raises(sqlite3.IntegrityError, match="synthetic closure failure"):
        storage.close_rejected_agent_attempt(
            run_id=state.run_id,
            checkpoint=checkpoint,
            error_code="MODEL_CONTEXT_REVOKED",
            trace_id=state.trace_id,
            restore_run=False,
        )
    assert _raw_run(storage, state.run_id) == before
    assert _audit_after(storage, state.run_id, checkpoint) == audit_before

    with storage._connect() as conn:
        conn.execute("DROP TRIGGER reject_tool_closure")
    for _ in range(2):
        storage.close_rejected_agent_attempt(
            run_id=state.run_id,
            checkpoint=checkpoint,
            error_code="MODEL_CONTEXT_REVOKED",
            trace_id=state.trace_id,
            restore_run=False,
        )
    run = storage.agent_run(state.run_id)
    assert run is not None
    assert run["status"] == AgentRunStatus.FAILED
    assert run["final_answer"] is None
    assert run["stop_reason"] == "MODEL_CONTEXT_REVOKED"
    assert len(storage.agent_trace_rows(state.run_id)) == 1
    audit = _audit_after(storage, state.run_id, checkpoint)
    _assert_safe_attempt(
        audit,
        run_id=state.run_id,
        error_code="MODEL_CONTEXT_REVOKED",
        forbidden=ATOMIC_MODEL_CANDIDATE,
        expected_model_count=1,
        expected_tool_count=1,
        extra_forbidden=(ATOMIC_PRIVATE_MARKER,),
    )
