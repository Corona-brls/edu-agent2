from __future__ import annotations

import json

from backend.agent.observation import tool_result_to_observation
from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.schemas import ActionDecision, AgentAction
from backend.agent.state import AgentRunStatus
from backend.core.errors import LLMProviderError, LLMResponseError
from backend.llm.template_provider import TemplateProvider
from backend.rag.hybrid_retriever import RetrievalResponse, RetrievalResult
from backend.storage.logger import LearningLogger
from backend.tools.mcp_client import MCPClientGateway
from mcp_server.schemas import ToolResult
from mcp_server.tools import TOOL_DEFINITIONS

from tests.test_agent_orchestration import FakeRetriever, SequenceProvider


def _storage(tmp_path, name: str = "fallback-semantics.db") -> LearningLogger:
    return LearningLogger(tmp_path / name)


def _course_result(snippet: str = "5G NAS authentication evidence") -> RetrievalResult:
    return RetrievalResult(
        document_id=1,
        chunk_id=1,
        source="course.md",
        title="5G NAS",
        snippet=snippet,
        page=1,
        section="auth",
        source_type="md",
        keyword_score=1.0,
        vector_score=0.8,
        final_score=0.9,
    )


def _retrieve_provider() -> SequenceProvider:
    return SequenceProvider(
        [
            ActionDecision(
                action=AgentAction.RETRIEVE,
                arguments={"query": "NAS"},
                reason="Retrieve course evidence.",
            )
        ]
    )


def test_planner_failure_deterministic_policy_marks_run_and_model_call_fallback(tmp_path):
    storage = _storage(tmp_path, "planner-fallback.db")
    planner = SequenceProvider(fail_decision=True)
    answer_provider = SequenceProvider()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=planner,
        answer_provider=answer_provider,
        fallback_provider=answer_provider,
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.run("Explain 5G NAS authentication evidence.", user_id="u1")
    model_calls = storage.model_call_rows(result["run_id"])

    assert result["run_status"] == AgentRunStatus.COMPLETED
    assert result["fallback_used"] is True
    assert model_calls[0]["purpose"] == "decide_next_action"
    assert model_calls[0]["success"] == 0
    assert model_calls[0]["fallback_used"] == 1
    assert model_calls[0]["metadata"]["fallback_path"] == "deterministic_evidence_policy"
    assert model_calls[0]["metadata"]["fallback_trigger"] == "MODEL_DECISION_FAILED"
    assert model_calls[0]["metadata"]["failure_category"] == "unknown"
    assert model_calls[0]["metadata"]["root_error_type"] == "RuntimeError"
    assert model_calls[0]["metadata"]["http_status"] is None
    assert model_calls[0]["metadata"]["retryable"] is False
    assert model_calls[0]["metadata"]["phase"] == "decide_next_action"


def test_retriever_failure_without_replacement_is_not_fallback(tmp_path):
    storage = _storage(tmp_path, "retriever-failure.db")
    provider = _retrieve_provider()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=FakeRetriever(error=RuntimeError("chroma down")),
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.run("Explain 5G NAS authentication evidence.", user_id="u1")
    steps = storage.agent_steps(result["run_id"])
    persisted_state = storage.agent_run(result["run_id"])["state"]

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["stop_reason"] == "RETRIEVER_FAILED"
    assert result["fallback_used"] is False
    assert persisted_state["fallback_used"] is False
    assert steps[0]["observation"]["fallback_used"] is False


def test_scope_mismatch_without_replacement_is_not_fallback(tmp_path):
    storage = _storage(tmp_path, "scope-mismatch.db")
    provider = _retrieve_provider()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=FakeRetriever(
            response=RetrievalResponse(
                results=[_course_result()],
                abstained=False,
                threshold=0.18,
                fallback_used=False,
            )
        ),
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.run("Explain 5G NAS authentication evidence.", user_id="u1", course_id=42)
    steps = storage.agent_steps(result["run_id"])

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["stop_reason"] == "RETRIEVER_SCOPE_MISMATCH"
    assert result["citations"] == []
    assert result["fallback_used"] is False
    assert steps[0]["observation"]["fallback_used"] is False


def test_empty_vector_results_use_local_keyword_fallback(tmp_path, monkeypatch):
    class ControlledLocalRetriever:
        def __init__(self, *_args, **_kwargs):
            pass

        def search(self, *_args, **_kwargs):
            return [
                {
                    "source": "local.md",
                    "title": "Local fallback",
                    "snippet": "Local keyword evidence for NAS authentication.",
                    "module": "fallback",
                    "score": 0.8,
                }
            ]

    monkeypatch.setattr("backend.agent.orchestrator.LocalRetriever", ControlledLocalRetriever)
    storage = _storage(tmp_path, "local-fallback.db")
    provider = _retrieve_provider()
    retriever = FakeRetriever(
        response=RetrievalResponse(results=[], abstained=True, threshold=0.18, fallback_used=False)
    )
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=retriever,
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.run("Explain 5G NAS authentication evidence.", user_id="u1")

    assert result["run_status"] == AgentRunStatus.COMPLETED
    assert result["fallback_used"] is True
    assert result["delivery_mode"] == "local_rag"
    assert result["citations"][0]["source"] == "local.md"


def test_successful_planner_and_answer_do_not_mark_fallback(tmp_path):
    storage = _storage(tmp_path, "successful-run.db")
    provider = _retrieve_provider()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.run("Explain 5G NAS authentication evidence.", user_id="u1")

    assert result["run_status"] == AgentRunStatus.COMPLETED
    assert result["fallback_used"] is False
    assert all("failure_category" not in row["metadata"] for row in storage.model_call_rows(result["run_id"]))


def test_answer_generation_failure_template_takeover_marks_fallback(tmp_path):
    storage = _storage(tmp_path, "answer-template-fallback.db")
    planner = _retrieve_provider()
    answer_provider = SequenceProvider(fail_answer=True)
    fallback_provider = SequenceProvider()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=planner,
        answer_provider=answer_provider,
        fallback_provider=fallback_provider,
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.run("Explain 5G NAS authentication evidence.", user_id="u1")

    assert result["run_status"] == AgentRunStatus.COMPLETED
    assert result["fallback_used"] is True
    assert result["provider"] == fallback_provider.provider_name


def test_explicit_template_provider_answer_marks_fallback(tmp_path):
    provider = TemplateProvider()
    orchestrator = AgentOrchestrator(
        storage=_storage(tmp_path, "explicit-template.db"),
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.run("Explain 5G NAS authentication evidence.", user_id="u1")

    assert result["run_status"] == AgentRunStatus.COMPLETED
    assert result["provider"] == "template"
    assert result["fallback_used"] is True


def test_tool_failure_without_alternative_is_not_fallback(tmp_path, monkeypatch):
    original = TOOL_DEFINITIONS["paper_search_real"]
    monkeypatch.setitem(TOOL_DEFINITIONS, "paper_search_real", original.model_copy(update={"enabled": True}))
    provider = SequenceProvider(
        [
            ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name="paper_search_real",
                arguments={"query": "nonexistent paper", "top_k": 1},
                reason="Try a paper tool.",
            ),
            ActionDecision(
                action=AgentAction.ABSTAIN,
                arguments={"message": "No verified paper evidence is available."},
                reason="No replacement evidence is available after the tool failed.",
                task_complete=True,
            ),
        ]
    )
    gateway = _FailedNoFallbackGateway()
    orchestrator = AgentOrchestrator(
        storage=_storage(tmp_path, "tool-failure.db"),
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=FakeRetriever(),
        mcp_gateway=gateway,
    )

    result = orchestrator.run("Find a verified paper about a nonexistent topic.", user_id="u1")

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["fallback_used"] is False
    assert gateway.calls == 1


def test_continue_preserves_prior_fallback_history(tmp_path, monkeypatch):
    original = TOOL_DEFINITIONS["paper_search_real"]
    monkeypatch.setitem(TOOL_DEFINITIONS, "paper_search_real", original.model_copy(update={"enabled": True}))
    storage = _storage(tmp_path, "continue-fallback.db")
    provider = SequenceProvider(
        [
            ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name="paper_search_real",
                arguments={"query": "temporary cached paper outage", "top_k": 1},
                reason="Try tool before asking.",
            ),
            ActionDecision(
                action=AgentAction.ASK_USER,
                arguments={"question": "Please clarify the course context."},
                reason="Need clarification.",
            ),
            ActionDecision(
                action=AgentAction.RETRIEVE,
                arguments={"query": "NAS clarified"},
                reason="Continue after clarification.",
            ),
        ]
    )
    gateway = _FallbackThenAskGateway()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=FakeRetriever(),
        mcp_gateway=gateway,
    )

    first = orchestrator.run("Need help with a NAS lab.", user_id="u1")
    second = orchestrator.continue_run(first["run_id"], "Use the local NAS lab notes.", user_id="u1")

    assert first["run_status"] == AgentRunStatus.REQUIRES_INPUT
    assert first["fallback_used"] is True
    assert second["run_status"] == AgentRunStatus.COMPLETED
    assert second["run_id"] == first["run_id"]
    assert second["fallback_used"] is True


def test_planner_transport_failure_metadata_is_sanitized(tmp_path):
    class SecretConnectError(RuntimeError):
        pass

    class SecretPlanner(SequenceProvider):
        def decide_next_action(self, state_summary, available_tools):
            self.decision_calls += 1
            try:
                raise SecretConnectError("connect failed sk-secret Authorization Token Headers full prompt")
            except SecretConnectError as exc:
                raise LLMProviderError("DeepSeek planner failed safely") from exc

    storage = _storage(tmp_path, "planner-transport-audit.db")
    planner = SecretPlanner()
    answer_provider = SequenceProvider()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=planner,
        answer_provider=answer_provider,
        fallback_provider=answer_provider,
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.run("Explain 5G NAS authentication evidence.", user_id="u1")
    model_call = storage.model_call_rows(result["run_id"])[0]

    assert result["fallback_used"] is True
    assert model_call["error_code"] == "MODEL_DECISION_FAILED"
    assert model_call["metadata"]["failure_category"] == "transport"
    assert model_call["metadata"]["root_error_type"] == "SecretConnectError"
    assert model_call["metadata"]["phase"] == "decide_next_action"
    assert model_call["metadata"]["retryable"] is True
    _assert_secret_absent(storage, result)


def test_invalid_action_decision_json_metadata_is_response_parse(tmp_path):
    class InvalidJsonPlanner(SequenceProvider):
        def decide_next_action(self, state_summary, available_tools):
            self.decision_calls += 1
            try:
                raise ValueError("invalid json sk-secret Authorization Token Headers")
            except ValueError as exc:
                raise LLMResponseError("ActionDecision parse failed") from exc

    storage = _storage(tmp_path, "planner-parse-audit.db")
    planner = InvalidJsonPlanner()
    answer_provider = SequenceProvider()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=planner,
        answer_provider=answer_provider,
        fallback_provider=answer_provider,
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.run("Explain 5G NAS authentication evidence.", user_id="u1")
    model_call = storage.model_call_rows(result["run_id"])[0]

    assert result["fallback_used"] is True
    assert model_call["error_code"] == "MODEL_DECISION_FAILED"
    assert model_call["metadata"]["failure_category"] == "response_parse"
    assert model_call["metadata"]["root_error_type"] == "ValueError"
    assert model_call["metadata"]["phase"] == "decide_next_action"
    assert model_call["metadata"]["retryable"] is False
    _assert_secret_absent(storage, result)


def test_answer_timeout_metadata_is_sanitized_when_template_takes_over(tmp_path):
    class TimeoutAnswerProvider(SequenceProvider):
        def generate_answer(self, request):
            self.answer_calls += 1
            raise TimeoutError("timed out sk-secret Authorization Token Headers full response")

    storage = _storage(tmp_path, "answer-timeout-audit.db")
    planner = _retrieve_provider()
    answer_provider = TimeoutAnswerProvider()
    fallback_provider = SequenceProvider()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=planner,
        answer_provider=answer_provider,
        fallback_provider=fallback_provider,
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.run("Explain 5G NAS authentication evidence.", user_id="u1")
    failed_call = next(row for row in storage.model_call_rows(result["run_id"]) if row["success"] == 0)

    assert result["run_status"] == AgentRunStatus.COMPLETED
    assert result["fallback_used"] is True
    assert failed_call["error_code"] == "MODEL_ANSWER_FAILED"
    assert failed_call["metadata"]["failure_category"] == "timeout"
    assert failed_call["metadata"]["root_error_type"] == "TimeoutError"
    assert failed_call["metadata"]["phase"] == "generate_answer"
    assert failed_call["metadata"]["retryable"] is True
    _assert_secret_absent(storage, result)


class _FailedNoFallbackGateway:
    def __init__(self):
        self.calls = 0

    def get_agent_tool_catalog(self, role="student"):
        return [_tool_catalog_entry("paper_search_real")]

    def call_tool(self, name, arguments, *, run_id, step_id, step_number=1, role="student", **_kwargs):
        self.calls += 1
        result = ToolResult(
            success=False,
            error_code="CONNECTOR_SERVICE_ERROR",
            error_message="controlled connector failure",
            metadata={"tool_name": name},
        )
        return _call(result, name, arguments, run_id, step_id, step_number, fallback_used=False)


class _FallbackThenAskGateway:
    def __init__(self):
        self.calls = 0

    def get_agent_tool_catalog(self, role="student"):
        return [_tool_catalog_entry("paper_search_real")]

    def call_tool(self, name, arguments, *, run_id, step_id, step_number=1, role="student", **_kwargs):
        self.calls += 1
        result = ToolResult(
            success=False,
            error_code="CONNECTOR_SERVICE_ERROR",
            error_message="controlled connector failure with trusted stale cache path",
            metadata={"tool_name": name, "fallback": True, "delivery_mode": "stale_cache"},
        )
        return _call(result, name, arguments, run_id, step_id, step_number, fallback_used=True)


def _assert_secret_absent(storage: LearningLogger, result: dict) -> None:
    payload = json.dumps(
        {
            "api_response": result,
            "model_calls": storage.model_call_rows(result["run_id"]),
            "trace": storage.agent_trace_rows(result["run_id"]),
        },
        ensure_ascii=False,
        default=str,
    )
    for forbidden in ("sk-secret", "Authorization", "Token", "Headers", "full prompt", "full response"):
        assert forbidden not in payload


def _tool_catalog_entry(name: str) -> dict:
    definition = TOOL_DEFINITIONS[name]
    return {
        "name": name,
        "description": definition.agent_description,
        "input_schema": definition.input_schema,
        "risk": definition.risk,
        "network": definition.network,
        "read_only": definition.read_only,
        "enabled": True,
    }


def _call(
    result: ToolResult,
    name: str,
    arguments: dict,
    run_id: str,
    step_id: str,
    step_number: int,
    *,
    fallback_used: bool,
):
    class Call:
        pass

    call = Call()
    call.result = result
    call.observation = tool_result_to_observation(
        result,
        step_number=step_number,
        tool_name=name,
        fallback_used=fallback_used,
        run_id=run_id,
        step_id=step_id,
        arguments=arguments,
        latency_ms=1,
    )
    call.fallback_used = fallback_used
    return call
