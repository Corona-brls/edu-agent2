from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import main
from backend.agent.state import AgentRunStatus, AgentState
from backend.auth.service import create_access_token
from backend.core.config import get_settings
from backend.llm.deepseek_provider import DeepSeekProvider
from backend.llm.schemas import EvidenceItem, LLMRequest
from backend.schemas import APIError
from backend.storage.logger import LearningLogger


ALLOWED_SPAN_MARKER = "R1C3_ALLOWED_SPAN_MARKER_73A1"
_AUDIT_TABLES = (
    "agent_runs",
    "agent_steps",
    "agent_traces",
    "model_calls",
    "tool_calls",
    "questions",
    "learning_events",
)


class _CountingCompletions:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, **request):
        self.requests.append(deepcopy(request))
        return {"choices": [{"message": {"content": "allowed response"}}]}


def _provider(
    course_id: int,
    *,
    allow_course: bool,
) -> tuple[DeepSeekProvider, _CountingCompletions]:
    completions = _CountingCompletions()
    settings = replace(
        get_settings(),
        deepseek_api_key="fake-r1c3-key",
        external_model_course_allowlist=(frozenset({course_id}) if allow_course else frozenset()),
        llm_max_retries=0,
        llm_timeout_seconds=2.0,
        agent_timeout_seconds=2.0,
    )
    provider = DeepSeekProvider(
        settings=settings,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    return provider, completions


def _table_counts(storage: LearningLogger) -> dict[str, int]:
    with storage._connect() as conn:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in _AUDIT_TABLES
        }


def _raw_run(storage: LearningLogger, run_id: str) -> dict:
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


@pytest.fixture
def unauthorized_http_context(tmp_path, monkeypatch):
    database = (tmp_path / "r1c3.db").resolve()
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
    monkeypatch.setenv("JWT_SECRET_KEY", "r1c3-isolated-jwt-secret-0123456789")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-r1c3-key")
    storage = LearningLogger(database)
    seeded = storage.seed_demo_identity()
    course_id = int(seeded["course_id"])
    class_id = int(seeded["class_id"])
    with storage._connect() as conn:
        conn.execute("UPDATE users SET must_change_password = 0")
    user = storage.user_by_username("student_demo")
    assert user is not None
    token, _ = create_access_token(storage.user(int(user["id"])) or user)
    headers = {"Authorization": f"Bearer {token}"}

    monkeypatch.setattr(main, "learning_logger", storage)
    client = TestClient(main.app)
    yield SimpleNamespace(
        storage=storage,
        user=user,
        course_id=course_id,
        class_id=class_id,
        headers=headers,
        client=client,
    )
    client.close()


def _create_requires_input_run(ctx) -> tuple[str, dict]:
    run_id = "r1c3-requires-input-run"
    state = AgentState(
        run_id=run_id,
        user_id=str(ctx.user["id"]),
        course_id=ctx.course_id,
        class_id=ctx.class_id,
        user_goal="解释课程认证流程",
        intent="course_qa",
        status=AgentRunStatus.REQUIRES_INPUT,
        current_step=1,
        pending_question="请补充期望输出",
        stop_reason="REQUIRES_INPUT",
    )
    state.add_user_message(state.user_goal)
    ctx.storage.create_agent_run(
        run_id=run_id,
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
        run_id=run_id,
        status=state.status.value,
        state=state.model_dump(mode="json"),
        stop_reason=state.stop_reason,
        final_answer=None,
        pending_question=state.pending_question,
    )
    return run_id, _raw_run(ctx.storage, run_id)


@pytest.mark.parametrize(
    ("endpoint", "allow_course", "use_course_references", "expected_status"),
    [
        ("chat", False, True, "blocked_by_policy"),
        ("chat", True, False, "disabled_by_user"),
        ("continue", False, True, "blocked_by_policy"),
        ("continue", True, False, "disabled_by_user"),
        ("practice", False, True, "blocked_by_policy"),
        ("practice", True, False, "disabled_by_user"),
    ],
)
def test_real_http_course_context_denial_preserves_policy_status_and_has_no_side_effects(
    unauthorized_http_context,
    monkeypatch,
    endpoint,
    allow_course,
    use_course_references,
    expected_status,
):
    ctx = unauthorized_http_context
    provider, completions = _provider(ctx.course_id, allow_course=allow_course)
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
    constructed: list[bool] = []
    real_orchestrator = main.AgentOrchestrator

    def orchestrator_spy(*args, **kwargs):
        constructed.append(True)
        return real_orchestrator(*args, **kwargs)

    monkeypatch.setattr(main, "AgentOrchestrator", orchestrator_spy)
    run_id = None
    run_before = None
    if endpoint == "continue":
        run_id, run_before = _create_requires_input_run(ctx)
    counts_before = _table_counts(ctx.storage)

    if endpoint == "chat":
        response = ctx.client.post(
            "/api/chat",
            headers=ctx.headers,
            json={
                "question": "请解释课程认证流程",
                "course_id": ctx.course_id,
                "class_id": ctx.class_id,
                "use_course_references": use_course_references,
            },
        )
    elif endpoint == "continue":
        response = ctx.client.post(
            f"/api/agent/runs/{run_id}/continue",
            headers=ctx.headers,
            json={
                "user_input": "继续解释课程认证流程",
                "use_course_references": use_course_references,
            },
        )
    else:
        response = ctx.client.post(
            "/api/practice/guide",
            headers=ctx.headers,
            json={
                "topic": "根据课程内容设计认证实验",
                "course_id": ctx.course_id,
                "class_id": ctx.class_id,
                "use_course_references": use_course_references,
            },
        )

    assert response.status_code == 403, response.text
    assert response.json() == {
        "error_code": "MODEL_CONTEXT_NOT_AUTHORIZED",
        "message": "课程资料未获本次外部模型处理授权，请确认课程许可并在本次请求中明确选择。",
        "run_id": None,
        "trace_id": None,
        "course_reference_status": expected_status,
    }
    assert completions.requests == []
    assert constructed == []
    assert _table_counts(ctx.storage) == counts_before
    if run_id is not None:
        assert _raw_run(ctx.storage, run_id) == run_before


def test_http_exception_projection_whitelists_course_reference_status():
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/test/http-exception-projection",
            "raw_path": b"/test/http-exception-projection",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "root_path": "",
        }
    )
    valid = asyncio.run(
        main.http_exception_handler(
            request,
            HTTPException(
                status_code=403,
                detail={
                    "error_code": "MODEL_CONTEXT_NOT_AUTHORIZED",
                    "message": "blocked",
                    "course_reference_status": "revoked",
                    "private_detail": "must-not-leak",
                },
            ),
        )
    )
    valid_body = json.loads(valid.body)
    assert valid_body["course_reference_status"] == "revoked"
    assert "private_detail" not in valid_body

    invalid = asyncio.run(
        main.http_exception_handler(
            request,
            HTTPException(
                status_code=403,
                detail={
                    "error_code": "MODEL_CONTEXT_NOT_AUTHORIZED",
                    "message": "blocked",
                    "course_reference_status": "client_forged_status",
                    "private_detail": "must-not-leak",
                },
            ),
        )
    )
    invalid_body = json.loads(invalid.body)
    assert "course_reference_status" not in invalid_body
    assert "private_detail" not in invalid_body


@pytest.mark.parametrize(
    ("allow_course", "use_course_references", "expected_status"),
    [
        (True, False, "disabled_by_user"),
        (False, True, "blocked_by_policy"),
    ],
)
def test_explicit_course_title_does_not_bypass_external_model_double_gate(
    unauthorized_http_context,
    monkeypatch,
    allow_course,
    use_course_references,
    expected_status,
):
    ctx = unauthorized_http_context
    provider, completions = _provider(ctx.course_id, allow_course=allow_course)
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
    counts_before = _table_counts(ctx.storage)

    response = ctx.client.post(
        "/api/chat",
        headers=ctx.headers,
        json={
            "question": "请只依据当前课程发布的《目标教材》解释认证流程",
            "course_id": ctx.course_id,
            "class_id": ctx.class_id,
            "use_course_references": use_course_references,
        },
    )

    assert response.status_code == 403, response.text
    assert response.json()["error_code"] == "MODEL_CONTEXT_NOT_AUTHORIZED"
    assert response.json()["course_reference_status"] == expected_status
    assert completions.requests == []
    assert _table_counts(ctx.storage) == counts_before


def test_api_error_schema_and_unrelated_http_error_remain_compatible(
    unauthorized_http_context,
):
    parsed = APIError(
        error_code="MODEL_CONTEXT_NOT_AUTHORIZED",
        message="blocked",
        course_reference_status="disabled_by_user",
    )
    assert parsed.course_reference_status == "disabled_by_user"
    with pytest.raises(ValidationError):
        APIError(
            error_code="MODEL_CONTEXT_NOT_AUTHORIZED",
            message="blocked",
            course_reference_status="client_forged_status",
        )

    response = unauthorized_http_context.client.get(
        "/api/agent/runs/r1c3-not-found",
        headers=unauthorized_http_context.headers,
    )
    assert response.status_code == 404
    assert response.json() == {
        "error_code": "RUN_NOT_FOUND",
        "message": "Agent run 不存在：r1c3-not-found",
        "run_id": "r1c3-not-found",
        "trace_id": None,
    }


def test_both_gates_allowed_still_send_one_minimal_course_span():
    provider, completions = _provider(42, allow_course=True)
    assert provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    ) == "allowed"
    marker = f"{ALLOWED_SPAN_MARKER} complete sentence."
    request = LLMRequest(
        question=f"Explain {ALLOWED_SPAN_MARKER}.",
        evidence=[
            EvidenceItem(
                source_id="S1",
                text=marker,
                resource_type="course",
                document_id=91,
                chunk_id=92,
            )
        ],
        metadata={
            "evidence_spans": [
                {
                    "span_id": "S1:text:1",
                    "source_id": "S1",
                    "field": "text",
                    "exact_text": marker,
                }
            ]
        },
    )

    provider.generate_answer(request)

    assert len(completions.requests) == 1
    outbound = json.dumps(completions.requests[0]["messages"], ensure_ascii=False)
    assert f"[S1:text:1] {marker}" in outbound

