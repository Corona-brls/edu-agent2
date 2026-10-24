from backend.agent.observation import (
    retrieval_results_to_observation,
    sanitize_payload,
    tool_result_to_observation,
)
from backend.rag.hybrid_retriever import RetrievalResult
from mcp_server.schemas import ToolResult


def test_tool_result_converts_to_untrusted_observation_and_redacts_secrets():
    result = ToolResult(
        success=True,
        data={
            "answer": "ok",
            "nested": {"api_key": "sk-secret123456789"},
            "text": "Bearer token123456789",
        },
        metadata={"latency_ms": 12.3},
    )

    observation = tool_result_to_observation(
        result,
        step_number=2,
        tool_name="course_retriever",
        run_id="run-1",
        step_id="step-2",
        arguments={"query": "NAS", "password": "secret"},
    )

    assert observation.step_number == 2
    assert observation.action == "call_tool"
    assert observation.success is True
    assert observation.metadata["content_trust"] == "untrusted"
    assert observation.metadata["run_id"] == "run-1"
    assert observation.metadata["step_id"] == "step-2"
    assert "***REDACTED***" in observation.content
    assert "sk-secret" not in observation.content
    assert observation.metadata["arguments"]["password"] == "***REDACTED***"


def test_failed_tool_result_keeps_error_code():
    result = ToolResult(
        success=False,
        error_code="TOOL_TIMEOUT",
        error_message="timeout",
        metadata={"tool_name": "paper_search"},
    )

    observation = tool_result_to_observation(
        result,
        step_number=1,
        tool_name="paper_search",
    )

    assert observation.success is False
    assert observation.error_code == "TOOL_TIMEOUT"
    assert "TOOL_TIMEOUT" in observation.summary


def test_retrieval_results_convert_to_observation_with_location_fields():
    result = RetrievalResult(
        document_id=1,
        chunk_id=10,
        source="demo.pdf",
        title="Demo",
        snippet="5G NAS evidence",
        page=3,
        section="Auth",
        source_type="pdf",
        keyword_score=0.4,
        vector_score=0.8,
        final_score=0.62,
    )

    observation = retrieval_results_to_observation(
        [result],
        step_number=1,
        fallback_used=True,
        run_id="run-2",
        step_id="step-1",
    )

    assert observation.action == "retrieve"
    assert observation.tool_name == "hybrid_retriever"
    assert observation.fallback_used is True
    assert observation.metadata["result_count"] == 1
    assert observation.metadata["content_trust"] == "untrusted"
    assert "demo.pdf" in observation.content
    assert '"page": 3' in observation.content


def test_sanitize_payload_masks_sensitive_keys_and_values():
    payload = {
        "DEEPSEEK_API_KEY": "abc",
        "plain": "sk-123456789abcdef",
        "items": [{"authorization": "Bearer abcdefghijklmnop"}],
    }

    sanitized = sanitize_payload(payload)

    assert sanitized["DEEPSEEK_API_KEY"] == "***REDACTED***"
    assert sanitized["plain"] == "***REDACTED***"
    assert sanitized["items"][0]["authorization"] == "***REDACTED***"
