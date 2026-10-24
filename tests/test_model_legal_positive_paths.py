from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.agent.observation import tool_result_to_observation
from backend.agent.state import AgentRunStatus
from backend.analytics.events import LearningEventService
from backend.core.config import get_settings
from backend.llm.data_policy import COURSE_SPAN_PROFILE_PRACTICE, build_course_text_spans
from backend.llm.deepseek_provider import DeepSeekProvider
from backend.llm.schemas import EvidenceItem
from backend.llm.template_provider import TemplateProvider
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import RetrievalResponse, RetrievalResult
from backend.storage.logger import LearningLogger, sha256_bytes
from mcp_server.schemas import RealPaperSearchInput, ToolResult
from mcp_server.tools import TOOL_DEFINITIONS


PRACTICE_MARKER = "R1C6B1_TEMPLATE_COURSE_MARKER_7E42"
PRACTICE_PUBLIC_URL = "https://www.rfc-editor.org/info/rfc9000"
PRACTICE_TEXT = (
    f"{PRACTICE_MARKER} 的研究目标是比较正常 5G NAS 认证与重放检测在课程隔离日志中的差异。"
    f"{PRACTICE_MARKER} 的方法设计是使用同一字段集合分别标注正常认证样本与重放检测样本，再比较事件序列和告警结果。"
    f"{PRACTICE_MARKER} 的工具选择是离线日志解析器、表格统计工具和本地可视化工具。"
    f"{PRACTICE_MARKER} 的实验步骤一是准备课程隔离日志；步骤二是标注正常认证；"
    "步骤三是标注重放检测；步骤四是运行离线解析；步骤五是比较结果；步骤六是保存可复核记录。"
    f"{PRACTICE_MARKER} 的结果分析只比较认证事件数量、检测命中情况和误报情况，不推断资料外结论。"
    f"{PRACTICE_MARKER} 的创新点是把正常认证与重放检测放入同一受控字段框架中进行可复核比较。"
    f"{PRACTICE_MARKER} 的风险与注意事项是仅使用课程隔离样本，不接触公网目标，不收集真实用户数据。"
)


class _NoLiveGateway:
    is_available = False

    @staticmethod
    def get_agent_tool_catalog(*, role: str):
        del role
        return []


class _StoredDocumentRetriever:
    def __init__(self, *, document_id: int, chunk_id: int) -> None:
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.calls: list[dict[str, int | str | None]] = []

    def retrieve(self, query: str, *, course_id: int | None = None, top_k: int | None = None):
        self.calls.append({"query": query, "course_id": course_id, "top_k": top_k})
        return RetrievalResponse(
            results=[
                RetrievalResult(
                    document_id=self.document_id,
                    chunk_id=self.chunk_id,
                    source="template-positive.md",
                    title="Template positive course evidence",
                    snippet=PRACTICE_TEXT,
                    page=1,
                    section="practice",
                    source_type="md",
                    keyword_score=1.0,
                    vector_score=0.0,
                    final_score=1.0,
                    metadata={"content_hash": "template-positive-content"},
                )
            ],
            abstained=False,
            threshold=0.18,
            fallback_used=False,
        )


def _login_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/auth/login",
        json={"username": "student_demo", "password": "demo123456"},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _count_rows(storage: LearningLogger, table: str, where: str = "", params: tuple = ()) -> int:
    with storage._connect() as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table} {where}", params).fetchone()[0])


def _assert_template_identity(body: dict) -> None:
    assert body["run_status"] == AgentRunStatus.COMPLETED
    assert body["abstained"] is False
    assert body["provider"] == "template"
    assert body["model"] == "local-template"
    assert body["fallback_used"] is True


def _assert_completed_run_identity(storage: LearningLogger, run_id: str) -> dict:
    run = storage.agent_run(run_id)
    assert run is not None
    assert run["status"] == AgentRunStatus.COMPLETED
    assert run["state"]["response_provider"] == "template"
    assert run["state"]["response_model"] == "local-template"
    assert run["state"]["fallback_used"] is True
    return run


def _assert_model_identity(storage: LearningLogger, run_id: str, *, answer_purpose: str) -> list[dict]:
    rows = storage.model_call_rows(run_id)
    assert rows
    assert all(row["provider"] == "template" for row in rows)
    assert all(row["model"] == "local-template" for row in rows)
    planner_rows = [row for row in rows if row["purpose"] == "decide_next_action" and row["success"] == 1]
    assert planner_rows
    assert all(row["fallback_used"] == 0 for row in planner_rows)
    answer_rows = [row for row in rows if row["purpose"] == answer_purpose and row["success"] == 1]
    assert len(answer_rows) == 1
    assert answer_rows[0]["fallback_used"] == 1
    return answer_rows


def _assert_scoped_question_and_event(
    storage: LearningLogger,
    *,
    question_id: int,
    course_id: int,
    class_id: int,
    client_session_id: str,
    event_type: str,
) -> None:
    with storage._connect() as conn:
        question = conn.execute(
            "SELECT course_id, class_id, run_id FROM questions WHERE id = ?",
            (question_id,),
        ).fetchone()
        assert question is not None
        assert int(question["course_id"]) == course_id
        assert int(question["class_id"]) == class_id
        assert question["run_id"]
        event = conn.execute(
            """
            SELECT le.course_id, le.class_id, le.event_type
            FROM learning_events AS le
            JOIN learning_sessions AS ls ON ls.id = le.session_id
            WHERE ls.client_session_id = ? AND le.event_type = ?
            ORDER BY le.id DESC LIMIT 1
            """,
            (client_session_id, event_type),
        ).fetchone()
    assert event is not None
    assert int(event["course_id"]) == course_id
    assert int(event["class_id"]) == class_id
    assert event["event_type"] == event_type


@pytest.fixture
def template_http_context(tmp_path, monkeypatch):
    database = (tmp_path / "template-positive.db").resolve()
    knowledge = (tmp_path / "knowledge").resolve()
    vectors = (tmp_path / "vectors").resolve()
    uploads = (tmp_path / "uploads").resolve()
    for directory in (knowledge, vectors, uploads):
        directory.mkdir()

    monkeypatch.setenv("SQLITE_DB_PATH", str(database))
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(knowledge))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(vectors))
    monkeypatch.setenv("UPLOAD_DIR", str(uploads))
    monkeypatch.setenv("JWT_SECRET_KEY", "r1c6b1-template-jwt-secret-0123456789")
    monkeypatch.setenv("LLM_PROVIDER", "template")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "template")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("EXTERNAL_MODEL_COURSE_ALLOWLIST", "")
    storage = LearningLogger(database)
    seeded = storage.seed_demo_identity()
    course_id = int(seeded["course_id"])
    class_id = int(seeded["class_id"])
    with storage._connect() as conn:
        conn.execute("UPDATE users SET must_change_password = 0")
    user = storage.user_by_username("student_demo")
    assert user is not None

    document = Document(
        source_path=knowledge / "template-positive.md",
        filename="template-positive.md",
        file_type=".md",
        title="Template positive course evidence",
        text=PRACTICE_TEXT,
        pages=[Page(text=PRACTICE_TEXT, page=1, section="practice", source_type="markdown")],
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
    client_session_id = f"r1c6b1-session-{tmp_path.name}"
    LearningEventService(storage).create_session(
        user=user,
        client_session_id=client_session_id,
        course_id=course_id,
        class_id=class_id,
    )

    retriever = _StoredDocumentRetriever(document_id=document_id, chunk_id=chunk_id)
    sdk_trap = {"constructions": 0}

    class _ForbiddenDeepSeekSDK:
        def __init__(self, *_args, **_kwargs) -> None:
            sdk_trap["constructions"] += 1
            raise AssertionError("DeepSeek SDK must not be constructed on the explicit Template path")

    monkeypatch.setattr("backend.llm.factory.DeepSeekProvider", _ForbiddenDeepSeekSDK)
    monkeypatch.setattr(main, "learning_logger", storage)
    monkeypatch.setattr(main, "mcp_gateway", _NoLiveGateway())
    monkeypatch.setattr("backend.agent.orchestrator.HybridRetriever", lambda **_kwargs: retriever)

    selected = main.create_llm_provider()
    assert type(selected) is TemplateProvider
    client = TestClient(main.app)
    headers = _login_headers(client)
    try:
        yield SimpleNamespace(
            storage=storage,
            client=client,
            headers=headers,
            course_id=course_id,
            class_id=class_id,
            document_id=document_id,
            client_session_id=client_session_id,
            retriever=retriever,
            sdk_trap=sdk_trap,
        )
    finally:
        client.close()


def test_template_chat_http_reports_local_fallback_identity(template_http_context):
    ctx = template_http_context
    before_questions = _count_rows(ctx.storage, "questions")
    before_events = _count_rows(ctx.storage, "learning_events", "WHERE event_type = ?", ("QUESTION_ASKED",))
    response = ctx.client.post(
        "/api/chat",
        headers=ctx.headers,
        json={
            "question": f"请根据课程资料解释 {PRACTICE_MARKER} 的 NAS 认证风险",
            "course_id": ctx.course_id,
            "class_id": ctx.class_id,
            "client_session_id": ctx.client_session_id,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    _assert_template_identity(body)
    assert body["citations"]
    assert {item["document_id"] for item in body["citations"]} == {ctx.document_id}
    _assert_completed_run_identity(ctx.storage, body["run_id"])
    _assert_model_identity(ctx.storage, body["run_id"], answer_purpose="generate_answer")
    assert _count_rows(ctx.storage, "questions") == before_questions + 1
    assert _count_rows(ctx.storage, "learning_events", "WHERE event_type = ?", ("QUESTION_ASKED",)) == before_events + 1
    _assert_scoped_question_and_event(
        ctx.storage,
        question_id=body["question_id"],
        course_id=ctx.course_id,
        class_id=ctx.class_id,
        client_session_id=ctx.client_session_id,
        event_type="QUESTION_ASKED",
    )
    assert ctx.sdk_trap["constructions"] == 0


def test_template_continue_http_preserves_run_and_reports_local_fallback_identity(template_http_context):
    ctx = template_http_context
    first = ctx.client.post(
        "/api/chat",
        headers=ctx.headers,
        json={
            "question": "需要补充 NAS 认证目标和约束条件",
            "course_id": ctx.course_id,
            "class_id": ctx.class_id,
            "client_session_id": ctx.client_session_id,
        },
    )
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["run_status"] == AgentRunStatus.REQUIRES_INPUT
    assert first_body["requires_user_input"] is True
    assert first_body["fallback_used"] is False
    run_id = first_body["run_id"]

    supplemental = "使用课程隔离日志分析 NAS 认证失败原因和重放风险"
    continued = ctx.client.post(
        f"/api/agent/runs/{run_id}/continue",
        headers=ctx.headers,
        json={
            "user_input": supplemental,
            "conversation_id": "r1c6b1-template-continue",
            "client_session_id": ctx.client_session_id,
        },
    )

    assert continued.status_code == 200, continued.text
    body = continued.json()
    _assert_template_identity(body)
    assert body["run_id"] == run_id
    assert body["conversation_id"] == "r1c6b1-template-continue"
    assert body["citations"]
    run = _assert_completed_run_identity(ctx.storage, run_id)
    assert supplemental in run["state"]["user_goal"]
    assert run["state"]["messages"][-1]["content"] == supplemental
    _assert_model_identity(ctx.storage, run_id, answer_purpose="generate_answer")
    with ctx.storage._connect() as conn:
        questions = conn.execute(
            "SELECT question, course_id, class_id FROM questions WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
    assert len(questions) == 2
    assert questions[-1]["question"] == supplemental
    assert all(int(row["course_id"]) == ctx.course_id for row in questions)
    assert all(int(row["class_id"]) == ctx.class_id for row in questions)
    assert _count_rows(ctx.storage, "learning_events", "WHERE event_type = ?", ("QUESTION_ASKED",)) == 2
    assert ctx.sdk_trap["constructions"] == 0


def test_template_strict_practice_http_reports_local_fallback_identity(template_http_context):
    ctx = template_http_context
    before_questions = _count_rows(ctx.storage, "questions")
    before_events = _count_rows(
        ctx.storage,
        "learning_events",
        "WHERE event_type = ?",
        ("PRACTICE_GUIDE_GENERATED",),
    )
    response = ctx.client.post(
        "/api/practice/guide",
        headers=ctx.headers,
        json={
            "topic": (
                f"请仅根据包含 {PRACTICE_MARKER} 的课程资料生成六步实践指导；"
                "七个部分和每一步都必须由当前课程引用支持。"
            ),
            "course_id": ctx.course_id,
            "class_id": ctx.class_id,
            "client_session_id": ctx.client_session_id,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    _assert_template_identity(body)
    assert body["intent"] == "practice_guide"
    assert body["citations"]
    assert {item["document_id"] for item in body["citations"]} == {ctx.document_id}
    for label in ("研究目标", "方法设计", "工具选择", "实验步骤", "结果分析", "创新点", "风险与注意事项"):
        assert f"## {label}" in body["answer"]
    assert all(f"{index}." in body["answer"] for index in range(1, 7))
    assert body["answer"].count("evidence quote:") == 12
    assert body["answer"].count("[S1]") == 12

    _assert_completed_run_identity(ctx.storage, body["run_id"])
    answer_rows = _assert_model_identity(
        ctx.storage,
        body["run_id"],
        answer_purpose="generate_practice_guide",
    )
    metadata = answer_rows[0]["metadata"]
    assert metadata["candidate_accepted"] is True
    assert metadata["claim_count"] == 12
    assert len(metadata["resolved_evidence_span_ids"]) == 12
    assert _count_rows(ctx.storage, "questions") == before_questions + 1
    assert _count_rows(
        ctx.storage,
        "learning_events",
        "WHERE event_type = ?",
        ("PRACTICE_GUIDE_GENERATED",),
    ) == before_events + 1
    _assert_scoped_question_and_event(
        ctx.storage,
        question_id=body["question_id"],
        course_id=ctx.course_id,
        class_id=ctx.class_id,
        client_session_id=ctx.client_session_id,
        event_type="PRACTICE_GUIDE_GENERATED",
    )
    assert ctx.sdk_trap["constructions"] == 0


_B2_RAW_OUTSIDE_CATALOG = "B2_RAW_OUTSIDE_CATALOG_7F3C91"
_B2_FILENAME_SENTINEL = "B2_FILENAME_SENTINEL_93A1F4.md"
_B2_PATH_SENTINEL = "B2_PATH_SENTINEL_6D42AA"
_B2_SECTION_SENTINEL = "B2_SECTION_SENTINEL_17E5C2"
_B2_REAL_USER_IDENTITY_SENTINEL = "B2_REAL_USER_IDENTITY_SENTINEL_5C8E21"
_B2_FORGED_METADATA_SENTINEL = "B2_FORGED_METADATA_SENTINEL_A8B30D"


def _b2_practice_groups() -> dict[str, list[str]]:
    groups = {
        "research_goal": [
            f"研究目标{i}是验证课程认证流程的目标条件{i}" for i in range(1, 10)
        ],
        "method_design": [
            f"方法设计{i}采用课程认证对照实验方法{i}" for i in range(1, 10)
        ],
        "tools": [
            f"工具选择{i}使用离线认证记录工具{i}" for i in range(1, 10)
        ],
        "experiment_steps": [
            f"实验步骤{i}执行课程认证检查动作{i}" for i in range(1, 7)
        ],
        "result_analysis": [
            f"结果分析{i}比较认证前后记录差异{i}" for i in range(1, 10)
        ],
        "innovation_points": [
            f"创新点{i}采用双重课程认证观察法{i}" for i in range(1, 10)
        ],
        "risks": [
            f"风险控制{i}要求认证失败后立即停止操作{i}" for i in range(1, 10)
        ],
    }
    groups["research_goal"][0] += f"，公开标准参考 {PRACTICE_PUBLIC_URL}"
    return groups


def _b2_practice_material() -> tuple[str, dict[str, list[str]], list[str]]:
    groups = _b2_practice_groups()
    ordered = [
        *groups["research_goal"],
        *groups["method_design"],
        *groups["tools"],
        *groups["experiment_steps"],
        *groups["result_analysis"],
        *groups["innovation_points"],
        *groups["risks"],
    ]
    assert len(ordered) == 60
    return "。".join([*ordered, _B2_RAW_OUTSIDE_CATALOG]) + "。", groups, ordered


def _b2_practice_candidate(groups: dict[str, list[str]]) -> str:
    ordinal = 0

    def claims(texts: list[str]) -> list[dict[str, str]]:
        nonlocal ordinal
        result: list[dict[str, str]] = []
        for text in texts:
            ordinal += 1
            result.append(
                {
                    "claim_text": text,
                    "source_id": "S1",
                    "evidence_span_id": f"S1:text:{ordinal}",
                }
            )
        return result

    payload = {
        "research_goal": claims(groups["research_goal"]),
        "method_design": claims(groups["method_design"]),
        "tools": claims(groups["tools"]),
        "experiment_steps": claims(groups["experiment_steps"]),
        "result_analysis": claims(groups["result_analysis"]),
        "innovation_points": claims(groups["innovation_points"]),
        "risks": claims(groups["risks"]),
    }
    assert ordinal == 60
    return json.dumps(payload, ensure_ascii=False)


def _b2_planner(action: str, reason: str) -> str:
    return json.dumps(
        {
            "action": action,
            "arguments": {"query": "课程认证实践资料"} if action == "retrieve" else {},
            "reason": reason,
            "task_complete": action == "answer",
        },
        ensure_ascii=False,
    )


class _B2ScriptedCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def create(self, **request):
        self.requests.append(deepcopy(request))
        assert self.responses, "unexpected extra physical SDK create"
        return {"choices": [{"message": {"content": self.responses.pop(0)}}]}


class _B2RecordingDeepSeekProvider(DeepSeekProvider):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logical_stages: list[str] = []

    def decide_next_action(self, *args, **kwargs):
        self.logical_stages.append("planner")
        return super().decide_next_action(*args, **kwargs)

    def generate_answer(self, *args, **kwargs):
        self.logical_stages.append("answer")
        return super().generate_answer(*args, **kwargs)

    def generate_structured(self, *args, **kwargs):
        self.logical_stages.append("structured")
        return super().generate_structured(*args, **kwargs)


class _B2SentinelRetriever:
    def __init__(self, *, document_id: int, chunk_id: int, text: str) -> None:
        self.calls: list[dict[str, int | str | None]] = []
        self.result = RetrievalResult(
            document_id=document_id,
            chunk_id=chunk_id,
            source=f"C:/isolated/{_B2_PATH_SENTINEL}/course-material.md",
            title=_B2_FILENAME_SENTINEL,
            snippet=text,
            page=1,
            section=_B2_SECTION_SENTINEL,
            source_type="md",
            keyword_score=1.0,
            vector_score=0.0,
            final_score=1.0,
            metadata={
                "content_hash": "b2-course-content",
                "forged": _B2_FORGED_METADATA_SENTINEL,
            },
        )

    def retrieve(self, query: str, *, course_id: int | None = None, top_k: int | None = None):
        self.calls.append({"query": query, "course_id": course_id, "top_k": top_k})
        return RetrievalResponse(
            results=[self.result],
            abstained=False,
            threshold=0.18,
            fallback_used=False,
        )


@pytest.fixture
def deepseek_course_practice_context(tmp_path, monkeypatch):
    database = (tmp_path / "deepseek-course-positive.db").resolve()
    knowledge = (tmp_path / "knowledge").resolve()
    vectors = (tmp_path / "vectors").resolve()
    uploads = (tmp_path / "uploads").resolve()
    for directory in (knowledge, vectors, uploads):
        directory.mkdir()

    monkeypatch.setenv("SQLITE_DB_PATH", str(database))
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(knowledge))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(vectors))
    monkeypatch.setenv("UPLOAD_DIR", str(uploads))
    monkeypatch.setenv("JWT_SECRET_KEY", "r1c6b2-deepseek-jwt-secret-0123456789")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "template")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-fake-deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("QWEN_API_KEY", "")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "")
    monkeypatch.setenv("NVD_API_KEY", "")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    for flag in (
        "ENABLE_MCP_SERVER",
        "ENABLE_REAL_PAPER_SEARCH",
        "ENABLE_REAL_CVE_LOOKUP",
        "EXTERNAL_RESOURCE_INDEX_ENABLED",
        "ENABLE_SEMANTIC_SCHOLAR",
        "ENABLE_ARXIV",
        "ENABLE_NVD",
        "ENABLE_GITHUB_SEARCH",
        "ENABLE_STANDARD_REMOTE_METADATA",
        "CONNECTOR_LIVE_SMOKE_ENABLED",
    ):
        monkeypatch.setenv(flag, "false")

    storage = LearningLogger(database)
    seeded = storage.seed_demo_identity()
    course_id = int(seeded["course_id"])
    class_id = int(seeded["class_id"])
    monkeypatch.setenv("EXTERNAL_MODEL_COURSE_ALLOWLIST", str(course_id))
    with storage._connect() as conn:
        conn.execute("UPDATE users SET must_change_password = 0")
        conn.execute(
            "UPDATE users SET display_name = ? WHERE username = ?",
            (_B2_REAL_USER_IDENTITY_SENTINEL, "student_demo"),
        )
    user = storage.user_by_username("student_demo")
    assert user is not None
    assert user["display_name"] == _B2_REAL_USER_IDENTITY_SENTINEL

    canonical_text, groups, ordered_spans = _b2_practice_material()
    document = Document(
        source_path=knowledge / _B2_PATH_SENTINEL / _B2_FILENAME_SENTINEL,
        filename=_B2_FILENAME_SENTINEL,
        file_type=".md",
        title=_B2_FILENAME_SENTINEL,
        text=canonical_text,
        pages=[
            Page(
                text=canonical_text,
                page=1,
                section=_B2_SECTION_SENTINEL,
                source_type="markdown",
            )
        ],
        metadata={"forged": _B2_FORGED_METADATA_SENTINEL},
    )
    saved = storage.save_document(
        document=document,
        content_hash=sha256_bytes(canonical_text.encode("utf-8")),
        chunks=[
            {
                "chunk_text": canonical_text,
                "page": 1,
                "section": _B2_SECTION_SENTINEL,
                "source_type": "md",
            }
        ],
        byte_size=len(canonical_text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
        class_id=class_id,
        uploaded_by_user_id=int(user["id"]),
    )
    document_id = int(saved["id"])
    storage.set_document_publication(document_id, published=True, actor_id=int(user["id"]))
    chunk_id = int(storage.chunks_for_document(document_id)[0]["id"])
    client_session_id = f"r1c6b2-session-{tmp_path.name}"
    LearningEventService(storage).create_session(
        user=user,
        client_session_id=client_session_id,
        course_id=course_id,
        class_id=class_id,
    )

    retriever = _B2SentinelRetriever(
        document_id=document_id,
        chunk_id=chunk_id,
        text=canonical_text,
    )
    completions = _B2ScriptedCompletions(
        [
            _b2_planner("retrieve", "先读取当前课程资料"),
            _b2_planner("answer", "课程资料足以生成实践指南"),
            _b2_practice_candidate(groups),
        ]
    )
    provider_settings = replace(
        get_settings(),
        deepseek_api_key="test-only-fake-deepseek-key",
        external_model_course_allowlist=frozenset({course_id}),
        llm_max_retries=0,
        agent_max_steps=5,
        llm_timeout_seconds=5,
        agent_timeout_seconds=20,
    )
    assert provider_settings.llm_max_retries == 0
    assert provider_settings.agent_max_steps == 5
    provider = _B2RecordingDeepSeekProvider(
        settings=provider_settings,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    assert provider.settings.llm_max_retries == 0
    assert provider.settings.agent_max_steps >= 3

    monkeypatch.setattr(main, "learning_logger", storage)
    monkeypatch.setattr(main, "mcp_gateway", _NoLiveGateway())
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
    monkeypatch.setattr("backend.agent.orchestrator.HybridRetriever", lambda **_kwargs: retriever)

    client = TestClient(main.app)
    headers = _login_headers(client)
    auth_me = client.get("/api/auth/me", headers=headers)
    assert auth_me.status_code == 200, auth_me.text
    assert auth_me.json()["user"]["id"] == int(user["id"])
    assert auth_me.json()["user"]["display_name"] == _B2_REAL_USER_IDENTITY_SENTINEL
    try:
        yield SimpleNamespace(
            storage=storage,
            client=client,
            headers=headers,
            provider=provider,
            completions=completions,
            retriever=retriever,
            canonical_text=canonical_text,
            ordered_spans=ordered_spans,
            user_id=int(user["id"]),
            course_id=course_id,
            class_id=class_id,
            document_id=document_id,
            client_session_id=client_session_id,
        )
    finally:
        client.close()


def test_deepseek_course_strict_practice_http_uses_authoritative_span_catalog(
    deepseek_course_practice_context,
):
    ctx = deepseek_course_practice_context
    catalog = build_course_text_spans(
        [
            EvidenceItem(
                source_id="S1",
                text=ctx.canonical_text,
                document_id=ctx.document_id,
                resource_type="course",
                origin_kind="local_document",
                delivery_mode="local_rag",
            )
        ],
        profile=COURSE_SPAN_PROFILE_PRACTICE,
    )
    assert len(catalog) == 60
    assert [span.exact_text for span in catalog] == ctx.ordered_spans
    assert _B2_RAW_OUTSIDE_CATALOG in ctx.canonical_text
    assert _B2_RAW_OUTSIDE_CATALOG not in [span.exact_text for span in catalog]
    assert ctx.retriever.result.title == _B2_FILENAME_SENTINEL
    assert _B2_PATH_SENTINEL in ctx.retriever.result.source
    assert ctx.retriever.result.section == _B2_SECTION_SENTINEL
    assert ctx.retriever.result.metadata["forged"] == _B2_FORGED_METADATA_SENTINEL
    stored_document = ctx.storage.document(ctx.document_id)
    assert stored_document is not None
    assert stored_document["filename"] == _B2_FILENAME_SENTINEL
    assert _B2_PATH_SENTINEL in stored_document["source_path"]
    assert stored_document["metadata"]["forged"] == _B2_FORGED_METADATA_SENTINEL
    assert _B2_RAW_OUTSIDE_CATALOG in ctx.storage.chunks_for_document(ctx.document_id)[0][
        "chunk_text"
    ]

    topic = "请根据当前已发布课程认证资料生成六步实践指南"
    storage_user = ctx.storage.user(ctx.user_id)
    assert storage_user is not None
    assert storage_user["display_name"] == _B2_REAL_USER_IDENTITY_SENTINEL
    auth_me = ctx.client.get("/api/auth/me", headers=ctx.headers)
    assert auth_me.status_code == 200, auth_me.text
    assert auth_me.json()["user"]["id"] == ctx.user_id
    assert auth_me.json()["user"]["display_name"] == _B2_REAL_USER_IDENTITY_SENTINEL
    assert _B2_REAL_USER_IDENTITY_SENTINEL not in topic
    assert _B2_REAL_USER_IDENTITY_SENTINEL not in ctx.canonical_text
    assert len(ctx.completions.responses) == 3
    assert all(
        _B2_REAL_USER_IDENTITY_SENTINEL not in scripted_response
        for scripted_response in ctx.completions.responses
    )
    assert all(
        _B2_REAL_USER_IDENTITY_SENTINEL not in non_identity_sentinel
        for non_identity_sentinel in (
            _B2_RAW_OUTSIDE_CATALOG,
            _B2_FILENAME_SENTINEL,
            _B2_PATH_SENTINEL,
            _B2_SECTION_SENTINEL,
            _B2_FORGED_METADATA_SENTINEL,
        )
    )
    forbidden = (
        _B2_RAW_OUTSIDE_CATALOG,
        _B2_FILENAME_SENTINEL,
        _B2_PATH_SENTINEL,
        _B2_SECTION_SENTINEL,
        _B2_FORGED_METADATA_SENTINEL,
        _B2_REAL_USER_IDENTITY_SENTINEL,
    )
    assert all(marker not in topic for marker in forbidden)
    before_questions = _count_rows(ctx.storage, "questions")
    before_events = _count_rows(
        ctx.storage,
        "learning_events",
        "WHERE event_type = ?",
        ("PRACTICE_GUIDE_GENERATED",),
    )

    response = ctx.client.post(
        "/api/practice/guide",
        headers=ctx.headers,
        json={
            "topic": topic,
            "course_id": ctx.course_id,
            "class_id": ctx.class_id,
            "client_session_id": ctx.client_session_id,
            "use_course_references": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_status"] == AgentRunStatus.COMPLETED
    assert body["abstained"] is False
    assert body["provider"] == "deepseek"
    assert body["model"] == ctx.provider.model_name
    assert body["fallback_used"] is False
    assert body["course_reference_status"] == "allowed"
    assert all(
        field not in body
        for field in (
            "research_goal",
            "method_design",
            "tools",
            "experiment_steps",
            "result_analysis",
            "innovation_points",
            "risks",
        )
    )

    for heading in (
        "## 研究目标",
        "## 方法设计",
        "## 工具选择",
        "## 实验步骤",
        "## 结果分析",
        "## 创新点",
        "## 风险与注意事项",
    ):
        assert heading in body["answer"]
    assert all(f"{index}. " in body["answer"] for index in range(1, 7))
    assert body["answer"].count("evidence quote:") == 60
    assert len(body["citations"]) == 1
    assert {citation["document_id"] for citation in body["citations"]} == {ctx.document_id}

    assert len(ctx.retriever.calls) == 1
    assert ctx.retriever.calls[0]["course_id"] == ctx.course_id
    assert ctx.provider.logical_stages == ["planner", "planner", "structured"]
    assert len(ctx.completions.requests) == 3
    assert ctx.completions.responses == []
    outbound_messages = [
        json.dumps(request["messages"], ensure_ascii=False)
        for request in ctx.completions.requests
    ]
    assert len(outbound_messages) == 3
    for marker in forbidden:
        assert all(marker not in messages for messages in outbound_messages)
    for ordinal, exact_text in enumerate(ctx.ordered_spans, start=1):
        assert f"[S1:text:{ordinal}] {exact_text}" in outbound_messages[2]
    assert PRACTICE_PUBLIC_URL in outbound_messages[2]

    model_rows = ctx.storage.model_call_rows(body["run_id"])
    assert model_rows
    assert len(model_rows) == 3
    assert [row["purpose"] for row in model_rows] == [
        "decide_next_action",
        "decide_next_action",
        "generate_practice_guide",
    ]
    for row in model_rows:
        assert row["success"] == 1
        assert row["provider"] == "deepseek"
        assert row["model"] == ctx.provider.model_name
        assert row["fallback_used"] == 0
    answer_metadata = model_rows[-1]["metadata"]
    assert answer_metadata["candidate_accepted"] is True
    assert answer_metadata["claim_count"] == 60
    assert len(answer_metadata["resolved_evidence_span_ids"]) == 60
    assert ctx.storage.tool_call_rows(body["run_id"]) == []

    run = ctx.storage.agent_run(body["run_id"])
    assert run is not None
    assert run["status"] == AgentRunStatus.COMPLETED
    assert run["final_answer"] == body["answer"]
    assert run["state"]["response_provider"] == "deepseek"
    assert run["state"]["response_model"] == ctx.provider.model_name
    assert run["state"]["fallback_used"] is False
    assert _count_rows(ctx.storage, "questions") == before_questions + 1
    assert _count_rows(
        ctx.storage,
        "learning_events",
        "WHERE event_type = ?",
        ("PRACTICE_GUIDE_GENERATED",),
    ) == before_events + 1
    _assert_scoped_question_and_event(
        ctx.storage,
        question_id=body["question_id"],
        course_id=ctx.course_id,
        class_id=ctx.class_id,
        client_session_id=ctx.client_session_id,
        event_type="PRACTICE_GUIDE_GENERATED",
    )


_B3_PRIVATE_BODY_SENTINEL = "B3_PRIVATE_BODY_SENTINEL_6F34A1"
_B3_PRIVATE_TITLE_SENTINEL = "B3_PRIVATE_TITLE_SENTINEL_9C2E74"
_B3_PRIVATE_PATH_SENTINEL = "B3_PRIVATE_PATH_SENTINEL_47D8B3"
_B3_PRIVATE_METADATA_SENTINEL = "B3_PRIVATE_METADATA_SENTINEL_A13C69"
_B3_REAL_USER_IDENTITY_SENTINEL = "B3_REAL_USER_IDENTITY_SENTINEL_D9025E"
_B3_PUBLIC_TITLE = "5G Core Network Attack Detection with Verified Telemetry"
_B3_PUBLIC_ABSTRACT = (
    "5G core network attack detection method identifies anomalous control-plane "
    "behavior from verified public telemetry."
)
_B3_PUBLIC_EXTERNAL_ID = "2610.54321"
_B3_PUBLIC_URL = "https://arxiv.org/abs/2610.54321"
_B3_PUBLIC_CONTENT_HASH = "b3-public-paper-content-hash-8e21"


def _b3_call_tool_planner() -> str:
    return json.dumps(
        {
            "action": "call_tool",
            "tool_name": "paper_search_real",
            "arguments": {
                "query": "5G核心网攻击检测",
                "top_k": 1,
                "source": "auto",
            },
            "reason": "检索一条可信公共论文证据",
            "task_complete": False,
        },
        ensure_ascii=False,
    )


def _b3_answer_planner() -> str:
    return json.dumps(
        {
            "action": "answer",
            "arguments": {},
            "reason": "可信公共论文证据已经满足回答条件",
            "task_complete": True,
        },
        ensure_ascii=False,
    )


def _b3_grounded_candidate() -> str:
    return json.dumps(
        {
            "claims": [
                {
                    "claim_text": f"Abstract: {_B3_PUBLIC_ABSTRACT}",
                    "source_id": "S1",
                    "evidence_span_id": "S1:text:1",
                }
            ]
        },
        ensure_ascii=False,
    )


class _B3CourseRetrieverTrap:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("course HybridRetriever must not run on the public-paper path")


class _B3PublicPaperGateway:
    def __init__(self) -> None:
        self.catalog_calls: list[str] = []
        self.calls: list[dict] = []
        self.validated_inputs: list[RealPaperSearchInput] = []
        self.results: list[ToolResult] = []
        self.observations = []
        self.resource = {
            "external_id": _B3_PUBLIC_EXTERNAL_ID,
            "resource_type": "paper",
            "source_name": "arxiv",
            "title": _B3_PUBLIC_TITLE,
            "url": _B3_PUBLIC_URL,
            "abstract": _B3_PUBLIC_ABSTRACT,
            "fetched_at": "2026-10-15T00:00:00Z",
            "origin_kind": "remote",
            "delivery_mode": "fresh_cache",
            "content_hash": _B3_PUBLIC_CONTENT_HASH,
            "provenance_verified": True,
        }

    def get_agent_tool_catalog(self, role="student"):
        self.catalog_calls.append(role)
        definition = TOOL_DEFINITIONS["paper_search_real"]
        return [
            {
                "name": definition.name,
                "description": definition.agent_description,
                "input_schema": definition.input_schema,
                "risk": definition.risk,
                "network": definition.network,
                "read_only": definition.read_only,
                "enabled": definition.enabled,
                "mcp_available": True,
            }
        ]

    def call_tool(
        self,
        name,
        arguments,
        *,
        run_id,
        step_id,
        step_number=1,
        **_kwargs,
    ):
        assert name == "paper_search_real"
        validated = RealPaperSearchInput.model_validate(arguments)
        normalized = validated.model_dump(mode="json")
        self.calls.append({"name": name, "arguments": normalized})
        self.validated_inputs.append(validated)
        result = ToolResult(
            success=True,
            data={"resources": [deepcopy(self.resource)]},
            metadata={"tool_name": name, "cache_status": "fresh_cache"},
        )
        observation = tool_result_to_observation(
            result,
            step_number=step_number,
            tool_name=name,
            fallback_used=False,
            run_id=run_id,
            step_id=step_id,
            arguments=normalized,
            latency_ms=1,
        )
        self.results.append(result)
        self.observations.append(observation)
        return SimpleNamespace(
            result=result,
            observation=observation,
            fallback_used=False,
        )


@pytest.fixture
def public_evidence_http_context(tmp_path, monkeypatch):
    database = (tmp_path / "public-evidence-positive.db").resolve()
    knowledge = (tmp_path / "knowledge").resolve()
    vectors = (tmp_path / "vectors").resolve()
    uploads = (tmp_path / "uploads").resolve()
    for directory in (knowledge, vectors, uploads):
        directory.mkdir()

    monkeypatch.setenv("SQLITE_DB_PATH", str(database))
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(knowledge))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(vectors))
    monkeypatch.setenv("UPLOAD_DIR", str(uploads))
    monkeypatch.setenv("JWT_SECRET_KEY", "r1c6b3-public-jwt-secret-0123456789")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "template")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    monkeypatch.setenv("EXTERNAL_MODEL_COURSE_ALLOWLIST", "")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-fake-key")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("QWEN_API_KEY", "")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "")
    monkeypatch.setenv("NVD_API_KEY", "")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    for flag in (
        "ENABLE_MCP_SERVER",
        "ENABLE_REAL_PAPER_SEARCH",
        "ENABLE_REAL_CVE_LOOKUP",
        "EXTERNAL_RESOURCE_INDEX_ENABLED",
        "ENABLE_SEMANTIC_SCHOLAR",
        "ENABLE_ARXIV",
        "ENABLE_NVD",
        "ENABLE_GITHUB_SEARCH",
        "ENABLE_STANDARD_REMOTE_METADATA",
        "CONNECTOR_LIVE_SMOKE_ENABLED",
    ):
        monkeypatch.setenv(flag, "false")

    storage = LearningLogger(database)
    seeded = storage.seed_demo_identity()
    course_id = int(seeded["course_id"])
    class_id = int(seeded["class_id"])
    with storage._connect() as conn:
        conn.execute("UPDATE users SET must_change_password = 0")
        conn.execute(
            "UPDATE users SET display_name = ? WHERE username = ?",
            (_B3_REAL_USER_IDENTITY_SENTINEL, "student_demo"),
        )
    user = storage.user_by_username("student_demo")
    assert user is not None
    assert user["display_name"] == _B3_REAL_USER_IDENTITY_SENTINEL

    private_document = Document(
        source_path=knowledge / _B3_PRIVATE_PATH_SENTINEL / "private-course.md",
        filename="private-course.md",
        file_type=".md",
        title=_B3_PRIVATE_TITLE_SENTINEL,
        text=_B3_PRIVATE_BODY_SENTINEL,
        pages=[
            Page(
                text=_B3_PRIVATE_BODY_SENTINEL,
                page=1,
                section="private-course",
                source_type="markdown",
            )
        ],
        metadata={"private_marker": _B3_PRIVATE_METADATA_SENTINEL},
    )
    saved = storage.save_document(
        document=private_document,
        content_hash=sha256_bytes(_B3_PRIVATE_BODY_SENTINEL.encode("utf-8")),
        chunks=[
            {
                "chunk_text": _B3_PRIVATE_BODY_SENTINEL,
                "page": 1,
                "section": "private-course",
                "source_type": "md",
            }
        ],
        byte_size=len(_B3_PRIVATE_BODY_SENTINEL.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
        class_id=class_id,
        uploaded_by_user_id=int(user["id"]),
    )
    document_id = int(saved["id"])
    document_id_token = f"B3_PRIVATE_DOCUMENT_ID_TOKEN_{document_id}_END"
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT metadata_json FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        assert row is not None
        metadata = json.loads(row["metadata_json"] or "{}")
        metadata["wrapped_document_id"] = document_id_token
        conn.execute(
            "UPDATE documents SET metadata_json = ? WHERE id = ?",
            (json.dumps(metadata, ensure_ascii=False), document_id),
        )
    storage.set_document_publication(document_id, published=True, actor_id=int(user["id"]))
    client_session_id = f"r1c6b3-session-{tmp_path.name}"
    LearningEventService(storage).create_session(
        user=user,
        client_session_id=client_session_id,
        course_id=course_id,
        class_id=class_id,
    )

    gateway = _B3PublicPaperGateway()
    retriever_trap = _B3CourseRetrieverTrap()
    completions = _B2ScriptedCompletions(
        [
            _b3_call_tool_planner(),
            _b3_answer_planner(),
            _b3_grounded_candidate(),
        ]
    )
    provider_settings = replace(
        get_settings(),
        deepseek_api_key="test-only-fake-key",
        external_model_course_allowlist=frozenset(),
        llm_max_retries=0,
        agent_max_steps=5,
        llm_timeout_seconds=5,
        agent_timeout_seconds=20,
    )
    assert provider_settings.external_model_course_allowlist == frozenset()
    assert provider_settings.llm_max_retries == 0
    assert provider_settings.agent_max_steps == 5
    provider = _B2RecordingDeepSeekProvider(
        settings=provider_settings,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    assert provider.settings.external_model_course_allowlist == frozenset()
    assert provider.settings.llm_max_retries == 0
    assert provider.settings.agent_max_steps >= 3

    original_tool_definition = TOOL_DEFINITIONS["paper_search_real"]
    enabled_definition = original_tool_definition.model_copy(update={"enabled": True})
    assert enabled_definition.network == original_tool_definition.network
    assert enabled_definition.risk == original_tool_definition.risk
    assert enabled_definition.read_only == original_tool_definition.read_only
    assert enabled_definition.input_schema == original_tool_definition.input_schema
    monkeypatch.setitem(TOOL_DEFINITIONS, "paper_search_real", enabled_definition)
    original_gateway = main.mcp_gateway
    monkeypatch.setattr(main, "learning_logger", storage)
    monkeypatch.setattr(main, "mcp_gateway", gateway)
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "backend.agent.orchestrator.HybridRetriever",
        lambda **_kwargs: retriever_trap,
    )

    client = TestClient(main.app)
    headers = _login_headers(client)
    auth_me = client.get("/api/auth/me", headers=headers)
    assert auth_me.status_code == 200, auth_me.text
    assert auth_me.json()["user"]["id"] == int(user["id"])
    assert auth_me.json()["user"]["display_name"] == _B3_REAL_USER_IDENTITY_SENTINEL
    try:
        yield SimpleNamespace(
            storage=storage,
            client=client,
            headers=headers,
            provider=provider,
            completions=completions,
            gateway=gateway,
            retriever_trap=retriever_trap,
            user_id=int(user["id"]),
            course_id=course_id,
            class_id=class_id,
            document_id=document_id,
            document_id_token=document_id_token,
            client_session_id=client_session_id,
            original_tool_definition=original_tool_definition,
            original_gateway=original_gateway,
        )
    finally:
        client.close()
        TOOL_DEFINITIONS["paper_search_real"] = original_tool_definition
        main.mcp_gateway = original_gateway
        assert TOOL_DEFINITIONS["paper_search_real"] is original_tool_definition
        assert main.mcp_gateway is original_gateway


def test_public_evidence_http_succeeds_with_empty_course_allowlist(
    public_evidence_http_context,
):
    ctx = public_evidence_http_context
    assert ctx.provider.settings.external_model_course_allowlist == frozenset()
    assert TOOL_DEFINITIONS["paper_search_real"].enabled is True
    assert ctx.original_tool_definition.enabled is False
    enabled_definition = TOOL_DEFINITIONS["paper_search_real"]
    assert enabled_definition.model_copy(update={"enabled": False}) == ctx.original_tool_definition

    storage_user = ctx.storage.user(ctx.user_id)
    assert storage_user is not None
    assert storage_user["display_name"] == _B3_REAL_USER_IDENTITY_SENTINEL
    auth_me = ctx.client.get("/api/auth/me", headers=ctx.headers)
    assert auth_me.status_code == 200, auth_me.text
    assert auth_me.json()["user"]["id"] == ctx.user_id
    assert auth_me.json()["user"]["display_name"] == _B3_REAL_USER_IDENTITY_SENTINEL

    stored_document = ctx.storage.document(ctx.document_id)
    assert stored_document is not None
    assert stored_document["title"] == _B3_PRIVATE_TITLE_SENTINEL
    assert _B3_PRIVATE_PATH_SENTINEL in stored_document["source_path"]
    assert stored_document["metadata"]["private_marker"] == _B3_PRIVATE_METADATA_SENTINEL
    assert stored_document["metadata"]["wrapped_document_id"] == ctx.document_id_token
    assert stored_document["index_status"] == "INDEXED"
    assert stored_document["published_to_students"] is True
    chunks = ctx.storage.chunks_for_document(ctx.document_id)
    assert len(chunks) == 1
    assert chunks[0]["chunk_text"] == _B3_PRIVATE_BODY_SENTINEL

    question = "请检索1篇关于5G核心网攻击检测的论文，并依据公开资料给出可核验结论"
    private_sentinels = (
        _B3_PRIVATE_BODY_SENTINEL,
        _B3_PRIVATE_TITLE_SENTINEL,
        _B3_PRIVATE_PATH_SENTINEL,
        _B3_PRIVATE_METADATA_SENTINEL,
        _B3_REAL_USER_IDENTITY_SENTINEL,
        ctx.document_id_token,
    )
    assert all(sentinel not in question for sentinel in private_sentinels)
    gateway_payload = json.dumps(ctx.gateway.resource, ensure_ascii=False)
    assert all(sentinel not in gateway_payload for sentinel in private_sentinels)
    assert len(ctx.completions.responses) == 3
    assert all(
        sentinel not in scripted_response
        for sentinel in private_sentinels
        for scripted_response in ctx.completions.responses
    )

    catalog = ctx.gateway.get_agent_tool_catalog(role="student")
    assert len(catalog) == 1
    assert set(catalog[0]) == {
        "name",
        "description",
        "input_schema",
        "risk",
        "network",
        "read_only",
        "enabled",
        "mcp_available",
    }
    assert catalog[0]["name"] == "paper_search_real"
    assert catalog[0]["input_schema"] == RealPaperSearchInput.model_json_schema()
    assert catalog[0]["enabled"] is True
    assert catalog[0]["network"] is True
    before_questions = _count_rows(ctx.storage, "questions")
    before_events = _count_rows(
        ctx.storage,
        "learning_events",
        "WHERE event_type = ?",
        ("QUESTION_ASKED",),
    )

    response = ctx.client.post(
        "/api/chat",
        headers=ctx.headers,
        json={
            "question": question,
            "course_id": ctx.course_id,
            "class_id": ctx.class_id,
            "client_session_id": ctx.client_session_id,
            "use_course_references": False,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_status"] == AgentRunStatus.COMPLETED
    assert body["abstained"] is False
    assert body["provider"] == "deepseek"
    assert body["model"] == ctx.provider.model_name
    assert body["fallback_used"] is False
    assert body["course_reference_status"] == "disabled_by_user"
    assert body["intent"] == "paper_search"

    assert ctx.retriever_trap.calls == 0
    assert ctx.gateway.catalog_calls == ["student", "student"]
    assert len(ctx.gateway.calls) == 1
    assert len(ctx.gateway.validated_inputs) == 1
    normalized_input = ctx.gateway.validated_inputs[0]
    assert normalized_input.top_k == 6
    assert normalized_input.source == "auto"
    assert normalized_input.query
    assert ctx.gateway.calls[0] == {
        "name": "paper_search_real",
        "arguments": normalized_input.model_dump(mode="json"),
    }
    assert len(ctx.gateway.results) == 1
    assert isinstance(ctx.gateway.results[0], ToolResult)
    assert ctx.gateway.results[0].success is True
    assert len(ctx.gateway.observations) == 1
    assert ctx.gateway.observations[0].tool_name == "paper_search_real"
    assert ctx.gateway.observations[0].success is True
    assert _B3_PUBLIC_TITLE in ctx.gateway.observations[0].content

    assert ctx.provider.logical_stages == ["planner", "planner", "structured"]
    assert len(ctx.completions.requests) == 3
    assert ctx.completions.responses == []
    outbound_messages = [
        json.dumps(request["messages"], ensure_ascii=False)
        for request in ctx.completions.requests
    ]
    assert len(outbound_messages) == 3
    for sentinel in private_sentinels:
        assert all(sentinel not in messages for messages in outbound_messages)
    structured_request_messages = ctx.completions.requests[2]["messages"]
    assert isinstance(structured_request_messages, list)
    structured_user_contents = [
        message["content"]
        for message in structured_request_messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and "GroundedSpanAnswer contract:" in message["content"]
        and "Evidence catalog:" in message["content"]
        and "Evidence span catalog:" in message["content"]
    ]
    assert len(structured_user_contents) == 1
    structured_user_content = structured_user_contents[0]

    decoder = json.JSONDecoder()

    def _decode_labeled_json(label: str):
        assert structured_user_content.count(label) == 1
        encoded = structured_user_content.split(label, 1)[1].lstrip()
        decoded, _ = decoder.raw_decode(encoded)
        return decoded

    evidence_catalog = _decode_labeled_json("Evidence catalog: ")
    assert evidence_catalog == [{"resource_type": "paper", "source_id": "S1"}]

    evidence_span_catalog = _decode_labeled_json("Evidence span catalog: ")
    expected_text_span = {
        "span_id": "S1:text:1",
        "source_id": "S1",
        "field": "text",
        "resource_type": "paper",
        "exact_text": f"Abstract: {_B3_PUBLIC_ABSTRACT}",
    }
    assert evidence_span_catalog.count(expected_text_span) == 1
    assert not any(
        isinstance(span, dict) and span.get("span_id") == "S1:source_url"
        for span in evidence_span_catalog
    )

    for public_value in (
        _B3_PUBLIC_TITLE,
        _B3_PUBLIC_ABSTRACT,
        _B3_PUBLIC_EXTERNAL_ID,
        _B3_PUBLIC_URL,
        _B3_PUBLIC_CONTENT_HASH,
    ):
        assert public_value in structured_user_content
    evidence_lines = [
        line for line in structured_user_content.splitlines() if line.startswith("[S1] ")
    ]
    assert len(evidence_lines) == 1
    assert (
        f"paper | {_B3_PUBLIC_EXTERNAL_ID} | {_B3_PUBLIC_URL} | "
        "remote | fresh_cache |"
    ) in evidence_lines[0]
    assert _B3_PUBLIC_CONTENT_HASH in evidence_lines[0]
    assert "S1:source_url" not in _b3_grounded_candidate()

    assert _B3_PUBLIC_ABSTRACT.rstrip(".") in body["answer"]
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["title"] == _B3_PUBLIC_TITLE
    assert citation["external_id"] == _B3_PUBLIC_EXTERNAL_ID
    assert citation["source_url"] == _B3_PUBLIC_URL
    assert citation["resource_type"] == "paper"
    public_result = json.dumps(
        {"answer": body["answer"], "citations": body["citations"]},
        ensure_ascii=False,
    )
    for sentinel in private_sentinels:
        assert sentinel not in public_result

    model_rows = ctx.storage.model_call_rows(body["run_id"])
    assert model_rows
    assert len(model_rows) == 3
    assert [row["purpose"] for row in model_rows] == [
        "decide_next_action",
        "decide_next_action",
        "generate_answer",
    ]
    for row in model_rows:
        assert row["success"] == 1
        assert row["provider"] == "deepseek"
        assert row["model"] == ctx.provider.model_name
        assert row["fallback_used"] == 0
    tool_rows = ctx.storage.tool_call_rows(body["run_id"])
    assert len(tool_rows) == 1
    assert tool_rows[0]["tool_name"] == "paper_search_real"
    assert tool_rows[0]["success"] == 1
    assert tool_rows[0]["fallback_used"] == 0

    assert _count_rows(ctx.storage, "questions") == before_questions + 1
    assert _count_rows(
        ctx.storage,
        "learning_events",
        "WHERE event_type = ?",
        ("QUESTION_ASKED",),
    ) == before_events + 1
    _assert_scoped_question_and_event(
        ctx.storage,
        question_id=body["question_id"],
        course_id=ctx.course_id,
        class_id=ctx.class_id,
        client_session_id=ctx.client_session_id,
        event_type="QUESTION_ASKED",
    )
    assert ctx.provider.settings.external_model_course_allowlist == frozenset()
