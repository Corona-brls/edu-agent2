from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

from backend import main
from backend.analytics.events import LearningEventService, ServerLearningEventType
from backend.core.errors import LLMResponseError
from backend.learning.context import CourseReference
from backend.learning.errors import GeneralLearningError
from backend.learning import general_service as general_service_module
from backend.learning.general_service import GeneralLearningService
from backend.learning.runtime import LearningRuntime
from backend.learning.schemas import GeneralChatRequest, GeneralModelAnswer
from backend.llm.deepseek_provider import DeepSeekProvider
from backend.llm.data_policy import ExternalModelDataPolicy
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import RetrievalResponse, RetrievalResult
from backend.storage.logger import LearningLogger, sha256_bytes
from frontend import api_client


class _FakeGeneralProvider:
    provider_name = "deepseek"
    model_name = "deepseek-test"

    def __init__(self, answer: str = "这是一个模型生成的学习回答。") -> None:
        self.answer = answer
        self.calls = []

    def generate_general_answer(self, question, references, *, deadline, outbound_guard=None):
        if outbound_guard:
            outbound_guard("before_create")
        self.calls.append((question, list(references), deadline))
        if outbound_guard:
            outbound_guard("after")
        return GeneralModelAnswer(status="answered", answer=self.answer)


def _login(client: TestClient, username: str) -> dict[str, str]:
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": "demo123456"},
    )
    assert response.status_code == 200
    payload = response.json()
    if payload.get("must_change_password"):
        response = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {payload['access_token']}"},
            json={
                "current_password": "demo123456",
                "new_password": f"{username}-changed-123",
            },
        )
        assert response.status_code == 200
        payload = response.json()
    return {"Authorization": f"Bearer {payload['access_token']}"}


def test_general_chat_is_a_distinct_student_endpoint(tmp_path, monkeypatch):
    original = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "general-red.db")
    main.learning_logger.seed_demo_identity()
    monkeypatch.setattr(main, "_create_general_provider", lambda: _FakeGeneralProvider())
    try:
        client = TestClient(main.app)
        teacher = _login(client, "teacher_demo")
        student = _login(client, "student_demo")
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()
        teacher_response = client.post(
            "/api/learning/general-chat",
            headers=teacher,
            json={"question": "解释勾股定理", "course_id": course_id, "class_id": class_id},
        )
        student_response = client.post(
            "/api/learning/general-chat",
            headers=student,
            json={"question": "解释勾股定理", "course_id": course_id, "class_id": class_id},
        )

        unauthenticated = client.post(
            "/api/learning/general-chat",
            json={"question": "解释勾股定理", "course_id": course_id, "class_id": class_id},
        )
        assert unauthenticated.status_code == 401
        assert unauthenticated.json()["request_id"]
        assert teacher_response.status_code == 403
        assert teacher_response.json()["request_id"]
        assert student_response.status_code == 200
        payload = student_response.json()
        assert payload["answer_kind"] == "model_generated"
        assert payload["provider"] == "deepseek"
        assert payload["reference_status"] == "disabled"
        assert "citations" not in payload
    finally:
        main.learning_logger = original


def test_general_chat_records_server_event_but_client_cannot_forge_it(tmp_path, monkeypatch):
    original = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "general-event.db")
    main.learning_logger.seed_demo_identity()
    monkeypatch.setattr(main, "_create_general_provider", lambda: _FakeGeneralProvider())
    try:
        client = TestClient(main.app)
        headers = _login(client, "student_demo")
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()
        session = client.post(
            "/api/learning/sessions",
            headers=headers,
            json={"client_session_id": "general-session", "course_id": course_id, "class_id": class_id},
        )
        assert session.status_code == 200
        forged = client.post(
            "/api/learning/events",
            headers=headers,
            json={
                "client_session_id": "general-session",
                "event_type": "GENERAL_QUESTION_ASKED",
                "course_id": course_id,
                "class_id": class_id,
            },
        )
        assert forged.status_code == 400
        response = client.post(
            "/api/learning/general-chat",
            headers=headers,
            json={
                "question": "解释动量守恒",
                "course_id": course_id,
                "class_id": class_id,
                "client_session_id": "general-session",
            },
        )
        assert response.status_code == 200
        with main.learning_logger._connect() as conn:
            rows = conn.execute(
                "SELECT event_type, metadata_json FROM learning_events WHERE event_type='GENERAL_QUESTION_ASKED'"
            ).fetchall()
        assert len(rows) == 1
        assert "解释动量守恒" not in rows[0]["metadata_json"]
        assert "这是一个模型生成" not in rows[0]["metadata_json"]
        summary = main.learning_logger.learning_activity_summary(
            course_id=course_id,
            class_ids=[class_id],
        )
        assert summary["visit_count"] == 1
        assert "GENERAL_QUESTION_ASKED" not in summary["event_distribution"]
    finally:
        main.learning_logger = original


def test_runtime_keeps_slot_until_timed_out_worker_really_finishes():
    runtime = LearningRuntime(rate_limit=20)
    release = threading.Event()
    finished = threading.Event()

    def blocking():
        try:
            release.wait(2)
            return "done"
        finally:
            finished.set()

    lease = runtime.begin(7)
    with pytest.raises(GeneralLearningError) as captured:
        asyncio.run(lease.run(blocking, deadline=time.monotonic() + 0.03))
    assert captured.value.error_code == "MODEL_TIMEOUT"
    lease.close()
    with pytest.raises(GeneralLearningError) as busy:
        runtime.begin(7)
    assert busy.value.error_code == "MODEL_BUSY"
    release.set()
    assert finished.wait(1)
    for _ in range(100):
        try:
            next_lease = runtime.begin(7)
        except GeneralLearningError:
            time.sleep(0.001)
            continue
        next_lease.close()
        break
    else:
        pytest.fail("runtime slot was not released after the worker completed")
    runtime.shutdown()


def test_runtime_rate_limit_is_per_user_and_returns_retry_after():
    now = [100.0]
    runtime = LearningRuntime(rate_limit=2, rate_window_seconds=60, clock=lambda: now[0])
    for _ in range(2):
        lease = runtime.begin(7)
        lease.close()
    with pytest.raises(GeneralLearningError) as limited:
        runtime.begin(7)
    assert limited.value.error_code == "RATE_LIMITED"
    assert limited.value.retry_after == 60
    other = runtime.begin(8)
    other.close()
    now[0] = 161.0
    recovered = runtime.begin(7)
    recovered.close()
    runtime.shutdown()


def test_disconnect_returns_without_writing_but_keeps_process_slot_for_worker():
    started = threading.Event()
    release = threading.Event()
    event_calls = []

    class BlockingProvider(_FakeGeneralProvider):
        def generate_general_answer(self, question, references, *, deadline, outbound_guard=None):
            if outbound_guard:
                outbound_guard("before_create")
            started.set()
            release.wait(2)
            if outbound_guard:
                outbound_guard("after")
            return GeneralModelAnswer(status="answered", answer="late")

    runtime = LearningRuntime(max_process_requests=1, rate_limit=20)
    disconnected = threading.Event()
    service = GeneralLearningService(
        storage=object(),
        retriever=object(),
        provider_factory=BlockingProvider,
        data_policy=ExternalModelDataPolicy.from_ids(set()),
        runtime=runtime,
    )

    async def scenario():
        task = asyncio.create_task(
            service.answer(
                user={"id": 21},
                scope={"course_id": 3, "class_id": None},
                payload=GeneralChatRequest(question="late answer", course_id=3),
                request_id="disconnect-request",
                context_validator=lambda: None,
                session_validator=lambda: None,
                event_recorder=event_calls.append,
                disconnected=disconnected,
            )
        )
        while not started.is_set():
            await asyncio.sleep(0.001)
        disconnected.set()
        with pytest.raises(GeneralLearningError) as captured:
            await asyncio.wait_for(task, timeout=0.5)
        assert captured.value.error_code == "CLIENT_DISCONNECTED"

    asyncio.run(scenario())
    assert event_calls == []
    with pytest.raises(GeneralLearningError) as busy:
        runtime.begin(22)
    assert busy.value.error_code == "MODEL_BUSY"
    release.set()
    for _ in range(100):
        try:
            lease = runtime.begin(22)
        except GeneralLearningError:
            time.sleep(0.001)
            continue
        lease.close()
        break
    else:
        pytest.fail("process slot was released before or not released after worker completion")
    runtime.shutdown()


def test_reference_policy_blocks_rag_but_still_allows_pure_model():
    class TrapRetriever:
        def retrieve(self, *_args, **_kwargs):
            raise AssertionError("blocked policy must not read course references")

    provider = _FakeGeneralProvider()
    runtime = LearningRuntime(rate_limit=20)
    service = GeneralLearningService(
        storage=object(),
        retriever=TrapRetriever(),
        provider_factory=lambda: provider,
        data_policy=ExternalModelDataPolicy.from_ids(set()),
        runtime=runtime,
    )
    response = asyncio.run(
        service.answer(
            user={"id": 9},
            scope={"course_id": 3, "class_id": None},
            payload=GeneralChatRequest(
                question="解释牛顿第二定律",
                course_id=3,
                use_course_references=True,
            ),
            request_id="request-policy",
            context_validator=lambda: None,
            session_validator=lambda: None,
        )
    )
    assert response["reference_status"] == "blocked_by_policy"
    assert provider.calls[0][1] == []
    runtime.shutdown()


def test_allowed_reference_is_revalidated_and_exposed_as_context_only():
    class Storage:
        def __init__(self):
            self.document_row = {
                "id": 4,
                "course_id": 3,
                "index_status": "INDEXED",
                "title": "课程教材",
                "content_hash": "doc-hash",
                "metadata": {"published_to_students": True},
            }
            self.chunk_row = {
                "id": 8,
                "document_id": 4,
                "index_status": "INDEXED",
                "chunk_text": "课程中的最小参考片段。",
                "chunk_hash": "chunk-hash",
                "page": 1,
                "section": "第一节",
                "source_type": "markdown",
            }

        def document(self, document_id):
            return dict(self.document_row) if document_id == 4 else None

        def chunks_for_document(self, document_id):
            return [dict(self.chunk_row)] if document_id == 4 else []

    class Retriever:
        def retrieve(self, *_args, **_kwargs):
            result = type("Result", (), {"document_id": 4, "chunk_id": 8})()
            return type("Response", (), {"results": [result]})()

    storage = Storage()
    provider = _FakeGeneralProvider()
    runtime = LearningRuntime(rate_limit=20)
    service = GeneralLearningService(
        storage=storage,
        retriever=Retriever(),
        provider_factory=lambda: provider,
        data_policy=ExternalModelDataPolicy.from_ids({3}),
        runtime=runtime,
    )
    response = asyncio.run(
        service.answer(
            user={"id": 10},
            scope={"course_id": 3, "class_id": None},
            payload=GeneralChatRequest(question="课程问题", course_id=3, use_course_references=True),
            request_id="request-reference",
            context_validator=lambda: None,
            session_validator=lambda: None,
        )
    )
    assert response["reference_status"] == "used"
    assert response["reference_materials"] == [
        {
            "document_id": 4,
            "chunk_id": 8,
            "title": "课程教材",
            "excerpt": "课程中的最小参考片段。",
            "usage": "context_only",
        }
    ]
    assert provider.calls[0][1][0].content_hash == "doc-hash"
    assert provider.calls[0][1][0].chunk_hash == "chunk-hash"
    runtime.shutdown()


def test_reference_chunk_change_during_model_call_discards_result_and_event():
    class Storage:
        def __init__(self):
            self.document_row = {
                "id": 4,
                "course_id": 3,
                "index_status": "INDEXED",
                "title": "课程教材",
                "content_hash": "doc-hash",
                "metadata": {"published_to_students": True},
            }
            self.chunk_row = {
                "id": 8,
                "document_id": 4,
                "index_status": "INDEXED",
                "chunk_text": "课程中的最小参考片段。",
                "chunk_hash": "chunk-hash",
            }

        def document(self, document_id):
            return dict(self.document_row) if document_id == 4 else None

        def chunks_for_document(self, document_id):
            return [dict(self.chunk_row)] if document_id == 4 else []

    class Retriever:
        def retrieve(self, *_args, **_kwargs):
            result = type("Result", (), {"document_id": 4, "chunk_id": 8})()
            return type("Response", (), {"results": [result]})()

    storage = Storage()

    class MutatingProvider(_FakeGeneralProvider):
        def generate_general_answer(self, question, references, *, deadline, outbound_guard=None):
            outbound_guard("before_create")
            storage.chunk_row["chunk_hash"] = "changed-hash"
            outbound_guard("after")
            return GeneralModelAnswer(status="answered", answer="late")

    events = []
    runtime = LearningRuntime(rate_limit=20)
    service = GeneralLearningService(
        storage=storage,
        retriever=Retriever(),
        provider_factory=MutatingProvider,
        data_policy=ExternalModelDataPolicy.from_ids({3}),
        runtime=runtime,
    )
    with pytest.raises(GeneralLearningError) as changed:
        asyncio.run(
            service.answer(
                user={"id": 10},
                scope={"course_id": 3, "class_id": None},
                payload=GeneralChatRequest(question="课程问题", course_id=3, use_course_references=True),
                request_id="request-changed-reference",
                context_validator=lambda: None,
                session_validator=lambda: None,
                event_recorder=events.append,
            )
        )
    assert changed.value.error_code == "REFERENCE_CONTEXT_CHANGED"
    assert events == []
    runtime.shutdown()


def test_general_chat_http_error_contract_for_validation_unavailable_and_busy(tmp_path, monkeypatch):
    original_logger = main.learning_logger
    original_runtime = main.general_learning_runtime
    main.learning_logger = LearningLogger(tmp_path / "general-errors.db")
    main.learning_logger.seed_demo_identity()
    main.general_learning_runtime = LearningRuntime(rate_limit=20)
    try:
        client = TestClient(main.app)
        headers = _login(client, "student_demo")
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()
        invalid = client.post(
            "/api/learning/general-chat",
            headers=headers,
            json={"question": "   ", "course_id": course_id, "class_id": class_id},
        )
        assert invalid.status_code == 422
        assert invalid.json()["error_code"] == "VALIDATION_ERROR"
        assert invalid.json()["retryable"] is False

        session = client.post(
            "/api/learning/sessions",
            headers=headers,
            json={"client_session_id": "ended-general-session", "course_id": course_id, "class_id": class_id},
        )
        assert session.status_code == 200
        assert client.post(
            "/api/learning/sessions/ended-general-session/end",
            headers=headers,
        ).status_code == 200
        ended = client.post(
            "/api/learning/general-chat",
            headers=headers,
            json={
                "question": "解释概率",
                "course_id": course_id,
                "class_id": class_id,
                "client_session_id": "ended-general-session",
            },
        )
        assert ended.status_code == 409
        assert ended.json()["error_code"] == "SESSION_ENDED"

        unavailable = client.post(
            "/api/learning/general-chat",
            headers=headers,
            json={"question": "解释概率", "course_id": course_id, "class_id": class_id},
        )
        assert unavailable.status_code == 503
        assert unavailable.json()["error_code"] == "MODEL_UNAVAILABLE"
        assert unavailable.json()["request_id"]

        user_id = int(main.learning_logger.user_by_username("student_demo")["id"])
        occupied = main.general_learning_runtime.begin(user_id)
        monkeypatch.setattr(main, "_create_general_provider", lambda: _FakeGeneralProvider())
        busy = client.post(
            "/api/learning/general-chat",
            headers=headers,
            json={"question": "解释概率", "course_id": course_id, "class_id": class_id},
        )
        assert busy.status_code == 429
        assert busy.json()["error_code"] == "MODEL_BUSY"
        assert int(busy.headers["Retry-After"]) >= 1
        occupied.close()

        class InvalidProvider(_FakeGeneralProvider):
            def generate_general_answer(self, question, references, *, deadline, outbound_guard=None):
                if outbound_guard:
                    outbound_guard("before_create")
                time.sleep(0.01)
                raise LLMResponseError("invalid response containing upstream details")

        monkeypatch.setattr(main, "_create_general_provider", InvalidProvider)
        invalid_model = client.post(
            "/api/learning/general-chat",
            headers=headers,
            json={"question": "解释概率", "course_id": course_id, "class_id": class_id},
        )
        assert invalid_model.status_code == 502
        assert invalid_model.json()["error_code"] == "MODEL_FAILED"
        assert "upstream" not in invalid_model.json()["message"]

        finished = threading.Event()

        class SlowProvider(_FakeGeneralProvider):
            def generate_general_answer(self, question, references, *, deadline, outbound_guard=None):
                try:
                    time.sleep(0.4)
                    return GeneralModelAnswer(status="answered", answer="too late")
                finally:
                    finished.set()

        monkeypatch.setattr(main, "_create_general_provider", SlowProvider)
        # Leave enough setup time for the worker to start on slower Windows hosts;
        # the fake still exceeds the request budget deterministically.
        monkeypatch.setattr(general_service_module, "GENERAL_REQUEST_SECONDS", 0.2)
        timed_out = client.post(
            "/api/learning/general-chat",
            headers=headers,
            json={"question": "解释概率", "course_id": course_id, "class_id": class_id},
        )
        assert timed_out.status_code == 504
        assert timed_out.json()["error_code"] == "MODEL_TIMEOUT"
        assert timed_out.json()["retryable"] is True
        assert finished.wait(1)
    finally:
        main.general_learning_runtime.shutdown()
        main.general_learning_runtime = original_runtime
        main.learning_logger = original_logger


def test_frontend_general_chat_preserves_structured_error_and_timeout(monkeypatch):
    response = requests.Response()
    response.status_code = 429
    response._content = b'{"error_code":"MODEL_BUSY","message":"busy","request_id":"req-1","retryable":true}'
    response.headers["Retry-After"] = "2"
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update({"method": method, "url": url, **kwargs})
        return response

    monkeypatch.setattr(api_client.requests, "request", fake_request)
    with pytest.raises(api_client.APIClientError) as raised:
        api_client.general_chat("token", "question", course_id=3)

    assert captured["timeout"] == 105
    assert raised.value.status_code == 429
    assert raised.value.error_code == "MODEL_BUSY"
    assert raised.value.request_id == "req-1"
    assert raised.value.retryable is True
    assert raised.value.retry_after == 2


class _CountingDeepSeekCompletions:
    def __init__(self, *, before_return=None) -> None:
        self.calls: list[dict] = []
        self.before_return = before_return

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.before_return is not None:
            self.before_return()
        content = json.dumps(
            {"status": "answered", "answer": "这是由计数 fake SDK 返回的学习回答。"},
            ensure_ascii=False,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _CountingDeepSeekClient:
    def __init__(self, *, before_return=None) -> None:
        self.completions = _CountingDeepSeekCompletions(before_return=before_return)
        self.chat = SimpleNamespace(completions=self.completions)

    def with_options(self, **_kwargs):
        return self


class _SingleReferenceRetriever:
    def __init__(self, document_id: int, chunk_id: int, marker: str) -> None:
        self.result = RetrievalResult(
            document_id=document_id,
            chunk_id=chunk_id,
            source=f"{marker}.md",
            title=f"{marker} title",
            snippet=f"{marker} snippet",
            page=1,
            section="reference",
            source_type="markdown",
            keyword_score=1.0,
            vector_score=0.0,
            final_score=1.0,
        )

    def retrieve(self, *_args, **_kwargs):
        return RetrievalResponse(results=[self.result], abstained=False, threshold=0.0)


def _install_general_reference(storage: LearningLogger, course_id: int, marker: str) -> tuple[int, int]:
    text = f"{marker} canonical course reference text."
    document = Document(
        source_path=Path(f"course-{course_id}/{marker}.md"),
        filename=f"{marker}.md",
        file_type=".md",
        title=f"{marker} title",
        text=text,
        pages=[Page(text=text, page=1, section="reference", source_type="markdown")],
    )
    saved = storage.save_document(
        document=document,
        content_hash=sha256_bytes(text.encode("utf-8")),
        chunks=[{"chunk_text": text, "page": 1, "section": "reference", "source_type": "markdown"}],
        byte_size=len(text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
    )
    document_id = int(saved["id"])
    actor_id = int(storage.user_by_username("teacher_demo")["id"])
    storage.set_document_publication(document_id, published=True, actor_id=actor_id)
    chunk_id = int(storage.chunks_for_document(document_id)[0]["id"])
    return document_id, chunk_id


def _general_event_count(storage: LearningLogger) -> int:
    with storage._connect() as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM learning_events WHERE event_type = 'GENERAL_QUESTION_ASKED'"
            ).fetchone()[0]
        )


def _deepseek_settings(*, course_id: int | None = None):
    return replace(
        main.get_settings(),
        deepseek_api_key="offline-test-key",
        llm_timeout_seconds=2.0,
        llm_max_retries=0,
        external_model_course_allowlist=(frozenset({course_id}) if course_id else frozenset()),
    )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [("disable", "USER_DISABLED"), ("revoke", "TOKEN_REVOKED")],
)
def test_general_chat_reloads_account_and_token_before_physical_create(
    tmp_path, monkeypatch, mutation, expected_code
):
    original_logger = main.learning_logger
    original_runtime = main.general_learning_runtime
    main.learning_logger = LearningLogger(tmp_path / f"general-dynamic-{mutation}.db")
    main.learning_logger.seed_demo_identity()
    main.general_learning_runtime = LearningRuntime(rate_limit=20)
    sdk = _CountingDeepSeekClient()
    try:
        client = TestClient(main.app, raise_server_exceptions=False)
        headers = _login(client, "student_demo")
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()
        assert client.post(
            "/api/learning/sessions",
            headers=headers,
            json={
                "client_session_id": f"dynamic-{mutation}",
                "course_id": course_id,
                "class_id": class_id,
            },
        ).status_code == 200
        user_id = int(main.learning_logger.user_by_username("student_demo")["id"])

        def provider_factory():
            with main.learning_logger._connect() as conn:
                if mutation == "disable":
                    conn.execute("UPDATE users SET enabled = 0 WHERE id = ?", (user_id,))
                else:
                    conn.execute("UPDATE users SET token_version = token_version + 1 WHERE id = ?", (user_id,))
            return DeepSeekProvider(settings=_deepseek_settings(), client=sdk)

        monkeypatch.setattr(main, "_create_general_provider", provider_factory)
        response = client.post(
            "/api/learning/general-chat",
            headers=headers,
            json={
                "question": "解释一个公开数学概念",
                "course_id": course_id,
                "class_id": class_id,
                "client_session_id": f"dynamic-{mutation}",
            },
        )

        assert response.status_code == 401
        assert response.json()["error_code"] == expected_code
        assert sdk.completions.calls == []
        assert _general_event_count(main.learning_logger) == 0
    finally:
        main.general_learning_runtime.shutdown()
        main.general_learning_runtime = original_runtime
        main.learning_logger = original_logger


@pytest.mark.parametrize("mutation", ["role_change", "membership_revoke"])
def test_general_chat_reloads_role_and_membership_before_physical_create(
    tmp_path, monkeypatch, mutation
):
    original_logger = main.learning_logger
    original_runtime = main.general_learning_runtime
    main.learning_logger = LearningLogger(tmp_path / f"general-scope-{mutation}.db")
    main.learning_logger.seed_demo_identity()
    main.general_learning_runtime = LearningRuntime(rate_limit=20)
    sdk = _CountingDeepSeekClient()
    try:
        client = TestClient(main.app, raise_server_exceptions=False)
        headers = _login(client, "student_demo")
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()
        user_id = int(main.learning_logger.user_by_username("student_demo")["id"])

        def provider_factory():
            with main.learning_logger._connect() as conn:
                if mutation == "role_change":
                    conn.execute("UPDATE users SET role = 'teacher' WHERE id = ?", (user_id,))
                else:
                    conn.execute(
                        "DELETE FROM class_members WHERE class_id = ? AND user_id = ?",
                        (class_id, user_id),
                    )
            return DeepSeekProvider(settings=_deepseek_settings(), client=sdk)

        monkeypatch.setattr(main, "_create_general_provider", provider_factory)
        response = client.post(
            "/api/learning/general-chat",
            headers=headers,
            json={"question": "解释动态课程权限", "course_id": course_id, "class_id": class_id},
        )

        assert response.status_code == 403
        assert response.json()["error_code"] == "REFERENCE_CONTEXT_CHANGED"
        assert sdk.completions.calls == []
        assert _general_event_count(main.learning_logger) == 0
    finally:
        main.general_learning_runtime.shutdown()
        main.general_learning_runtime = original_runtime
        main.learning_logger = original_logger


@pytest.mark.parametrize(
    ("window", "expected_sdk_calls"),
    [("after_retrieval", 0), ("before_create", 0), ("after_candidate", 1)],
)
def test_general_chat_maps_reference_changes_across_real_deepseek_windows(
    tmp_path, monkeypatch, window, expected_sdk_calls
):
    original_logger = main.learning_logger
    original_runtime = main.general_learning_runtime
    main.learning_logger = LearningLogger(tmp_path / f"general-reference-{window}.db")
    main.learning_logger.seed_demo_identity()
    main.general_learning_runtime = LearningRuntime(rate_limit=20)
    course_id = main.learning_logger.default_course_id()
    class_id = main.learning_logger.default_class_id()
    document_id, chunk_id = _install_general_reference(main.learning_logger, course_id, f"REF_{window}")
    retriever = _SingleReferenceRetriever(document_id, chunk_id, f"REF_{window}")
    teacher_id = int(main.learning_logger.user_by_username("teacher_demo")["id"])
    changed = False

    def invalidate_reference():
        nonlocal changed
        if not changed:
            main.learning_logger.set_document_publication(document_id, published=False, actor_id=teacher_id)
            changed = True

    sdk = _CountingDeepSeekClient(
        before_return=(invalidate_reference if window == "after_candidate" else None)
    )
    settings = _deepseek_settings(course_id=course_id)
    try:
        monkeypatch.setattr(main, "HybridRetriever", lambda storage: retriever)
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        if window == "after_retrieval":
            original_load = general_service_module.load_course_references

            def load_then_invalidate(*args, **kwargs):
                result = original_load(*args, **kwargs)
                invalidate_reference()
                return result

            monkeypatch.setattr(general_service_module, "load_course_references", load_then_invalidate)

        def provider_factory():
            if window == "before_create":
                invalidate_reference()
            return DeepSeekProvider(settings=settings, client=sdk)

        monkeypatch.setattr(main, "_create_general_provider", provider_factory)
        client = TestClient(main.app, raise_server_exceptions=False)
        headers = _login(client, "student_demo")
        response = client.post(
            "/api/learning/general-chat",
            headers=headers,
            json={
                "question": "请结合课程参考解释测试概念",
                "course_id": course_id,
                "class_id": class_id,
                "use_course_references": True,
            },
        )

        assert response.status_code == 409
        assert response.json()["error_code"] == "REFERENCE_CONTEXT_CHANGED"
        assert len(sdk.completions.calls) == expected_sdk_calls
        assert _general_event_count(main.learning_logger) == 0
    finally:
        main.general_learning_runtime.shutdown()
        main.general_learning_runtime = original_runtime
        main.learning_logger = original_logger


@pytest.mark.parametrize(
    ("terminal_change", "expected_code"),
    [
        ("session_end", "SESSION_ENDED"),
        ("token_revoke", "TOKEN_REVOKED"),
        ("membership_revoke", "REFERENCE_CONTEXT_CHANGED"),
    ],
)
def test_general_chat_rechecks_terminal_state_immediately_before_event(
    tmp_path, monkeypatch, terminal_change, expected_code
):
    original_logger = main.learning_logger
    original_runtime = main.general_learning_runtime
    main.learning_logger = LearningLogger(tmp_path / f"general-event-terminal-{terminal_change}.db")
    main.learning_logger.seed_demo_identity()
    main.general_learning_runtime = LearningRuntime(rate_limit=20)
    sdk = _CountingDeepSeekClient()
    settings = _deepseek_settings()
    session_checks = 0
    mutated = False
    original_validate = LearningEventService.validate_session
    try:
        client = TestClient(main.app, raise_server_exceptions=False)
        headers = _login(client, "student_demo")
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()
        session_id = f"terminal-{terminal_change}"
        assert client.post(
            "/api/learning/sessions",
            headers=headers,
            json={"client_session_id": session_id, "course_id": course_id, "class_id": class_id},
        ).status_code == 200
        user_id = int(main.learning_logger.user_by_username("student_demo")["id"])

        def validate_then_mutate(self, **kwargs):
            nonlocal session_checks, mutated
            result = original_validate(self, **kwargs)
            session_checks += 1
            if session_checks >= 5 and not mutated:
                if terminal_change == "session_end":
                    row = main.learning_logger.learning_session_by_client_id(session_id)
                    main.learning_logger.end_learning_session(int(row["id"]))
                else:
                    with main.learning_logger._connect() as conn:
                        if terminal_change == "token_revoke":
                            conn.execute(
                                "UPDATE users SET token_version = token_version + 1 WHERE id = ?",
                                (user_id,),
                            )
                        else:
                            conn.execute(
                                "DELETE FROM class_members WHERE class_id = ? AND user_id = ?",
                                (class_id, user_id),
                            )
                mutated = True
            return result

        monkeypatch.setattr(LearningEventService, "validate_session", validate_then_mutate)
        monkeypatch.setattr(
            main,
            "_create_general_provider",
            lambda: DeepSeekProvider(settings=settings, client=sdk),
        )
        response = client.post(
            "/api/learning/general-chat",
            headers=headers,
            json={
                "question": "解释事件终态测试",
                "course_id": course_id,
                "class_id": class_id,
                "client_session_id": session_id,
            },
        )

        assert mutated is True
        assert response.status_code in {401, 403, 409}
        assert response.json()["error_code"] == expected_code
        assert len(sdk.completions.calls) == 1
        assert _general_event_count(main.learning_logger) == 0
    finally:
        main.general_learning_runtime.shutdown()
        main.general_learning_runtime = original_runtime
        main.learning_logger = original_logger


def test_general_chat_keeps_answer_when_only_event_storage_fails(tmp_path, monkeypatch):
    original_logger = main.learning_logger
    original_runtime = main.general_learning_runtime
    main.learning_logger = LearningLogger(tmp_path / "general-event-storage.db")
    main.learning_logger.seed_demo_identity()
    main.general_learning_runtime = LearningRuntime(rate_limit=20)
    sdk = _CountingDeepSeekClient()
    settings = _deepseek_settings()
    try:
        client = TestClient(main.app, raise_server_exceptions=False)
        headers = _login(client, "student_demo")
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()
        session_id = "event-storage-failure"
        assert client.post(
            "/api/learning/sessions",
            headers=headers,
            json={"client_session_id": session_id, "course_id": course_id, "class_id": class_id},
        ).status_code == 200
        monkeypatch.setattr(
            main,
            "_create_general_provider",
            lambda: DeepSeekProvider(settings=settings, client=sdk),
        )

        def fail_storage(*_args, **_kwargs):
            raise RuntimeError("isolated event storage unavailable")

        monkeypatch.setattr(LearningEventService, "record_server_event", fail_storage)
        response = client.post(
            "/api/learning/general-chat",
            headers=headers,
            json={
                "question": "解释普通事件存储故障",
                "course_id": course_id,
                "class_id": class_id,
                "client_session_id": session_id,
            },
        )

        assert response.status_code == 200
        assert response.json()["warnings"] == ["学习事件暂未记录，不影响本次回答。"]
        assert len(sdk.completions.calls) == 1
        assert _general_event_count(main.learning_logger) == 0
    finally:
        main.general_learning_runtime.shutdown()
        main.general_learning_runtime = original_runtime
        main.learning_logger = original_logger
