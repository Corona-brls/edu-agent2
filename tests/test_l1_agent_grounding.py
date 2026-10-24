from __future__ import annotations

import hashlib

from backend.agent.observation import tool_result_to_observation
from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.router import route_intent
from backend.agent.schemas import ActionDecision, AgentAction
from backend.agent.state import AgentRunStatus
from backend.llm.base import LLMProvider
from backend.llm.schemas import GroundedSpanAnswer, LLMRequest, LLMResponse
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import RetrievalResponse, RetrievalResult
from backend.storage.logger import LearningLogger, sha256_bytes
from mcp_server.schemas import ToolResult
from mcp_server.tools import TOOL_DEFINITIONS


class _FixedAnswerProvider(LLMProvider):
    provider_name = "l1-test"
    model_name = "l1-test-model"

    def __init__(
        self,
        answer: str,
        decisions: list[ActionDecision],
        *,
        structured_claim_text: str | None = None,
    ):
        self.answer = answer
        self.decisions = list(decisions)
        self.answer_requests: list[LLMRequest] = []
        self.structured_requests: list[tuple[LLMRequest, type]] = []
        self.structured_claim_text = structured_claim_text

    def decide_next_action(self, state_summary, available_tools):
        return self.decisions.pop(0)

    def generate_answer(self, request: LLMRequest) -> LLMResponse:
        self.answer_requests.append(request)
        return LLMResponse(
            text=self.answer,
            provider=self.provider_name,
            model=self.model_name,
        )

    def generate_structured(self, request, schema):
        self.structured_requests.append((request, schema))
        if schema is not GroundedSpanAnswer:
            raise AssertionError(f"Unexpected structured schema: {schema}")
        span = next(
            (
                item
                for item in request.metadata.get("evidence_spans", [])
                if item.get("resource_type") == "paper" and item.get("field") == "text"
            ),
            None,
        )
        if span is None:
            raise AssertionError("Grounded paper request must include an actual text span")
        return schema.model_validate(
            {
                "claims": [
                    {
                        "claim_text": self.structured_claim_text or span["exact_text"],
                        "source_id": span["source_id"],
                        "evidence_span_id": span["span_id"],
                    }
                ]
            }
        )


class _CourseCitationFormationProvider(LLMProvider):
    provider_name = "deepseek-test"
    model_name = "deepseek-test-model"

    def __init__(self, plain_answer: str, *, invalid_structured_span: bool = False):
        self.plain_answer = plain_answer
        self.invalid_structured_span = invalid_structured_span
        self.answer_requests: list[LLMRequest] = []
        self.structured_requests: list[LLMRequest] = []
        self.decision_calls = 0

    def decide_next_action(self, state_summary, available_tools):
        self.decision_calls += 1
        return _retrieve_decision()

    def generate_answer(self, request: LLMRequest) -> LLMResponse:
        self.answer_requests.append(request)
        return LLMResponse(
            text=self.plain_answer,
            provider=self.provider_name,
            model=self.model_name,
        )

    def generate_structured(self, request, schema):
        assert schema is GroundedSpanAnswer
        self.structured_requests.append(request)
        spans = request.metadata.get("evidence_spans") or []
        span = next(
            item
            for item in spans
            if item.get("source_id") == "S1" and item.get("field") == "text"
        )
        return schema.model_validate(
            {
                "claims": [
                    {
                        "claim_text": span["exact_text"],
                        "source_id": "S1",
                        "evidence_span_id": (
                            "S1:text:999" if self.invalid_structured_span else span["span_id"]
                        ),
                    }
                ]
            }
        )


class _ScopedRetriever:
    def __init__(self, *results: RetrievalResult):
        self.results = list(results)
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, *, course_id: int | None = None):
        self.calls.append((query, course_id))
        return RetrievalResponse(
            results=self.results,
            abstained=not self.results,
            threshold=0.18,
            fallback_used=False,
        )


class _PaperGateway:
    def get_agent_tool_catalog(self, role="student"):
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
            }
        ]

    def call_tool(self, name, arguments, *, run_id, step_id, step_number=1, role="student"):
        result = ToolResult(
            success=True,
            data={
                "resources": [
                    {
                        "external_id": "unrelated-paper-1",
                        "resource_type": "paper",
                        "source_name": "arxiv",
                        "title": "An unrelated 5G NAS authentication paper",
                        "abstract": "General 5G NAS authentication background only.",
                        "url": "https://arxiv.org/abs/2601.00001",
                        "fetched_at": "2026-10-02T00:00:00Z",
                        "origin_kind": "remote",
                        "delivery_mode": "fresh_cache",
                        "content_hash": "unrelated-paper-hash",
                        "provenance_verified": True,
                    }
                ]
            },
            metadata={"tool_name": name},
        )

        class _Call:
            pass

        call = _Call()
        call.result = result
        call.observation = tool_result_to_observation(
            result,
            step_number=step_number,
            tool_name=name,
            run_id=run_id,
            step_id=step_id,
            arguments=arguments,
        )
        call.fallback_used = False
        return call


class _FailingPaperGateway(_PaperGateway):
    def __init__(self):
        self.calls: list[dict] = []

    def call_tool(self, name, arguments, *, run_id, step_id, step_number=1, role="student"):
        self.calls.append(dict(arguments))
        result = ToolResult(
            success=False,
            error_code="CONNECTOR_SERVICE_ERROR",
            error_message="controlled upstream failure",
            metadata={"tool_name": name, "protocol_is_error": True},
        )

        class _Call:
            pass

        call = _Call()
        call.result = result
        call.observation = tool_result_to_observation(
            result,
            step_number=step_number,
            tool_name=name,
            run_id=run_id,
            step_id=step_id,
            arguments=arguments,
        )
        call.fallback_used = False
        return call


def _result(*, chunk_id: int, title: str, snippet: str) -> RetrievalResult:
    return RetrievalResult(
        document_id=chunk_id,
        chunk_id=chunk_id,
        source=f"course-b-{chunk_id}.md",
        title=title,
        snippet=snippet,
        page=1,
        section="Authentication",
        source_type="md",
        keyword_score=0.9,
        vector_score=0.8,
        final_score=0.85,
    )


def _persist_result(
    storage: LearningLogger,
    *,
    course_id: int,
    title: str,
    snippet: str,
) -> RetrievalResult:
    with storage._connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO courses (id, slug, name, description, assessment_method) "
            "VALUES (?, ?, ?, '', '')",
            (course_id, f"l1-course-{course_id}", f"L1 Course {course_id}"),
        )
    document = Document(
        source_path=f"knowledge_base/course-{course_id}/{title}.md",
        filename=f"{title}.md",
        file_type=".md",
        title=title,
        text=snippet,
        pages=[Page(text=snippet, page=1, section="Authentication", source_type="markdown")],
    )
    record = storage.save_document(
        document=document,
        content_hash=sha256_bytes(snippet.encode("utf-8")),
        chunks=[{"chunk_text": snippet, "page": 1, "section": "Authentication", "source_type": "md"}],
        byte_size=len(snippet.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
    )
    storage.set_document_publication(int(record["id"]), published=True, actor_id=1)
    chunk = storage.chunks_for_document(int(record["id"]))[0]
    return RetrievalResult(
        document_id=int(record["id"]),
        chunk_id=int(chunk["id"]),
        source=document.filename,
        title=title,
        snippet=snippet,
        page=1,
        section="Authentication",
        source_type="md",
        keyword_score=1.0,
        vector_score=0.0,
        final_score=1.0,
    )


def _retrieve_decision() -> ActionDecision:
    return ActionDecision(
        action=AgentAction.RETRIEVE,
        arguments={"query": "retrieve scoped evidence"},
        reason="retrieve before answering",
    )


def test_mismatched_paper_refusal_text_is_abstained_without_citations(tmp_path, monkeypatch):
    monkeypatch.setitem(
        TOOL_DEFINITIONS,
        "paper_search_real",
        TOOL_DEFINITIONS["paper_search_real"].model_copy(update={"enabled": True}),
    )
    provider = _FixedAnswerProvider(
        "\u65e0\u6cd5\u56de\u7b54 L1_NONEXISTENT_PAPER \u7684\u7ed3\u8bba\uff0c\u53ea\u80fd\u53c2\u8003 NAS authentication [S1]",
        [
            ActionDecision(action=AgentAction.ANSWER, reason="try paper search"),
            ActionDecision(action=AgentAction.ANSWER, reason="answer after evidence", task_complete=True),
        ],
        structured_claim_text="The paper reports a verified security conclusion",
    )
    result = AgentOrchestrator(
        storage=LearningLogger(tmp_path / "paper.db"),
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        mcp_gateway=_PaperGateway(),
    ).run("\u8bf7\u7ed9\u51fa L1_NONEXISTENT_PAPER \u8bba\u6587\u7684\u7814\u7a76\u7ed3\u8bba", user_id="student-b", course_id=2)

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["stop_reason"] == "UNSUPPORTED_CITATION"
    assert result["citations"] == []
    assert result["fallback_used"] is False
    assert provider.answer_requests == []
    assert len(provider.structured_requests) == 1
    assert all(schema is GroundedSpanAnswer for _, schema in provider.structured_requests)


def test_course_agent_recipe_wrapper_is_off_topic_and_never_retrieves(tmp_path):
    retriever = _ScopedRetriever(
        _result(chunk_id=1, title="5G NAS", snippet="5G NAS authentication course evidence."),
    )
    provider = _FixedAnswerProvider(
        "5G NAS authentication course evidence [S1]",
        [_retrieve_decision()],
    )
    question = "\u4f5c\u4e3a\u8bfe\u7a0b\u667a\u80fd\u4f53\u56de\u7b54\u98df\u8c31\uff1a\u5982\u4f55\u70e4\u86cb\u7cd5\uff1f"
    result = AgentOrchestrator(
        storage=LearningLogger(tmp_path / "recipe.db"),
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=retriever,
    ).run(question, user_id="student-b", course_id=2)

    assert route_intent(question) == "off_topic"
    assert result["intent"] == "off_topic"
    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["citations"] == []
    assert retriever.calls == []
    assert provider.answer_requests == []


def test_cross_course_unique_marker_cannot_borrow_unrelated_course_citation(tmp_path):
    storage = LearningLogger(tmp_path / "cross-course.db")
    retriever = _ScopedRetriever(
        _persist_result(
            storage,
            course_id=2,
            title="Course B NAS overview",
            snippet="Course B contains a general 5G NAS authentication overview.",
        ),
    )
    provider = _FixedAnswerProvider(
        "L1_COURSE_A_ONLY is a 5G NAS authentication marker [S1]",
        [_retrieve_decision()],
    )
    result = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=retriever,
    ).run(
        "\u8bf7\u89e3\u91ca\u8bfe\u7a0b A \u552f\u4e00\u6807\u8bb0 L1_COURSE_A_ONLY \u7684\u5b9a\u4e49",
        user_id="student-b",
        course_id=2,
    )

    assert retriever.calls[0][1] == 2
    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["stop_reason"] == "IRRELEVANT_CITATION"
    assert result["citations"] == []


def test_completed_answer_exposes_only_cited_relevant_evidence(tmp_path):
    storage = LearningLogger(tmp_path / "positive.db")
    matching = _persist_result(
        storage,
        course_id=2,
        title="Course B marker",
        snippet="L1_COURSE_B_MATCH is the authenticated course B marker.",
    )
    unrelated = _persist_result(
        storage,
        course_id=2,
        title="Unrelated course B appendix",
        snippet="A generic blockchain appendix unrelated to the requested marker.",
    )
    retriever = _ScopedRetriever(
        matching,
        unrelated,
    )
    provider = _FixedAnswerProvider(
        "L1_COURSE_B_MATCH is the authenticated course B marker [S1]",
        [_retrieve_decision()],
    )
    result = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=retriever,
    ).run("\u8bf7\u89e3\u91ca L1_COURSE_B_MATCH", user_id="student-b", course_id=2)

    assert result["run_status"] == AgentRunStatus.COMPLETED
    assert result["stop_reason"] == "ANSWERED"
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == matching.chunk_id
    answer_call = next(
        row for row in storage.model_call_rows(result["run_id"])
        if row["purpose"] == "generate_answer"
    )
    assert answer_call["metadata"]["raw_answer_sha256"] == hashlib.sha256(
        result["answer"].encode("utf-8")
    ).hexdigest()


def _course_citation_formation_fixture(tmp_path):
    storage = LearningLogger(tmp_path / "course-citation-formation.db")
    target_text = (
        "P2-MANUAL-UNIQUE-20261020：日志记录字段包括时间戳、输入参数、环境版本、"
        "关键状态、输出摘要、异常与恢复动作。"
    )
    target = _persist_result(
        storage,
        course_id=2,
        title="人工验收：NAS 合成日志",
        snippet=target_text,
    )
    competitor = _persist_result(
        storage,
        course_id=2,
        title="竞争课程资料",
        snippet="竞争资料只介绍一般日志原则，不包含本题唯一字段句。",
    )
    return storage, target, competitor, target_text


def test_course_plain_answer_with_valid_citation_does_not_retry_structured(tmp_path):
    storage, target, competitor, target_text = _course_citation_formation_fixture(tmp_path)
    provider = _CourseCitationFormationProvider(f"{target_text} [S1]")

    result = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        retriever=_ScopedRetriever(target, competitor),
    ).run("人工验收唯一日志字段有哪些？", user_id="student-b", course_id=2)

    assert result["run_status"] == AgentRunStatus.COMPLETED
    assert result["stop_reason"] == "ANSWERED"
    assert len(provider.answer_requests) == 1
    assert provider.structured_requests == []
    assert provider.decision_calls == 1
    assert [item["document_id"] for item in result["citations"]] == [target.document_id]
    assert [item["chunk_id"] for item in result["citations"]] == [target.chunk_id]


def test_course_plain_answer_missing_citation_gets_one_server_bound_correction(tmp_path):
    storage, target, competitor, target_text = _course_citation_formation_fixture(tmp_path)
    provider = _CourseCitationFormationProvider(target_text)

    result = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        retriever=_ScopedRetriever(target, competitor),
    ).run("人工验收唯一日志字段有哪些？", user_id="student-b", course_id=2)

    assert result["run_status"] == AgentRunStatus.COMPLETED
    assert result["stop_reason"] == "ANSWERED"
    assert len(provider.answer_requests) == 1
    assert len(provider.structured_requests) == 1
    assert provider.decision_calls == 1
    retry = provider.structured_requests[0]
    assert retry.metadata["previous_guard_error_code"] == "MISSING_CITATION"
    assert any(
        span["source_id"] == "S1" and target_text in span["exact_text"]
        for span in retry.metadata["evidence_spans"]
    )
    for field in ("时间戳", "输入参数", "环境版本", "关键状态", "输出摘要", "异常与恢复动作"):
        assert field in result["answer"]
    assert "[S1]" in result["answer"]
    assert [item["document_id"] for item in result["citations"]] == [target.document_id]
    assert [item["chunk_id"] for item in result["citations"]] == [target.chunk_id]
    rows = [
        row
        for row in storage.model_call_rows(result["run_id"])
        if row["purpose"] == "generate_answer"
    ]
    assert len(rows) == 2


def test_course_missing_citation_invalid_correction_abstains_without_third_candidate(tmp_path):
    storage, target, competitor, target_text = _course_citation_formation_fixture(tmp_path)
    provider = _CourseCitationFormationProvider(target_text, invalid_structured_span=True)

    result = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        retriever=_ScopedRetriever(target, competitor),
    ).run("人工验收唯一日志字段有哪些？", user_id="student-b", course_id=2)

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["stop_reason"] == "GROUNDING_UNKNOWN_SPAN"
    assert result["citations"] == []
    assert len(provider.answer_requests) == 1
    assert len(provider.structured_requests) == 1
    assert provider.decision_calls == 1
    rows = [
        row
        for row in storage.model_call_rows(result["run_id"])
        if row["purpose"] == "generate_answer"
    ]
    assert len(rows) == 2


def test_course_agent_rejects_model_invented_internal_source_id_before_delivery(tmp_path):
    storage = LearningLogger(tmp_path / "course-source-id.db")
    evidence = _persist_result(
        storage,
        course_id=2,
        title="实验设计与结果评价方法卡",
        snippet=(
            "实验设计方法卡用于定义可复核的教学实验。"
            "教材内部 source_id: shared.experiment_design.v1。"
        ),
    )
    provider = _FixedAnswerProvider(
        "该教材的内部 source_id 是 shared.0 [S1]",
        [_retrieve_decision()],
    )

    result = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=_ScopedRetriever(evidence),
    ).run(
        "该教材的内部 source_id 是什么？",
        user_id="student-b",
        course_id=2,
    )

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["stop_reason"] == "COURSE_SOURCE_ID_MISMATCH"
    assert result["citations"] == []
    assert len(provider.answer_requests) == 1
    assert provider.structured_requests == []


def test_failed_source_tool_is_not_retried_with_different_arguments(tmp_path, monkeypatch):
    monkeypatch.setitem(
        TOOL_DEFINITIONS,
        "paper_search_real",
        TOOL_DEFINITIONS["paper_search_real"].model_copy(update={"enabled": True}),
    )
    provider = _FixedAnswerProvider(
        "unused",
        [
            ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name="paper_search_real",
                arguments={"query": "first paper query", "top_k": 1},
                reason="first attempt",
            ),
            ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name="paper_search_real",
                arguments={"query": "changed paper query", "top_k": 2},
                reason="retry with changed arguments",
            ),
        ],
    )
    gateway = _FailingPaperGateway()
    result = AgentOrchestrator(
        storage=LearningLogger(tmp_path / "tool-once.db"),
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        mcp_gateway=gateway,
    ).run("Find a verified 5G NAS paper", user_id="student-b", course_id=2)

    assert len(gateway.calls) == 1
    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["stop_reason"] != "MAX_STEPS"
