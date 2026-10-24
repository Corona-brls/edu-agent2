import time
import asyncio
from pathlib import Path

import pytest

from backend.connectors.registry import ConnectorRegistry
from backend.connectors.standard_source import StandardSourceConnector
from backend.connectors.schemas import ResourceQuery
from backend.core.config import get_settings
from backend.resources.service import ExternalResourceService
from backend.storage.logger import LearningLogger
from backend.tools.mcp_client import MCPClientGateway
from mcp_server import tools as tool_module
from mcp_server.schemas import ToolResult
from mcp_server.tools import TOOL_DEFINITIONS


def _gateway_mcp_error(gateway: MCPClientGateway) -> str | None:
    return next(
        (
            str(tool["mcp_error"])
            for tool in gateway.list_tools()
            if tool.get("mcp_error")
        ),
        None,
    )


def test_mcp_tool_timeout_setting_is_loaded(monkeypatch):
    monkeypatch.setenv("MCP_TOOL_TIMEOUT_SECONDS", "17.5")

    settings = get_settings()
    gateway = MCPClientGateway()

    assert settings.mcp_tool_timeout_seconds == 17.5
    assert gateway._mcp_timeout_seconds == 17.5


@pytest.mark.parametrize("value", ["0", "-1", "301"])
def test_mcp_tool_timeout_setting_rejects_unsafe_values(monkeypatch, value):
    monkeypatch.setenv("MCP_TOOL_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="MCP_TOOL_TIMEOUT_SECONDS"):
        get_settings()


def test_live_start_script_enables_supported_gates_before_single_worker_start():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "start_live.ps1").read_text(encoding="utf-8")
    example = (root / ".env.example").read_text(encoding="utf-8")

    for name in (
        "ENABLE_REAL_PAPER_SEARCH",
        "ENABLE_REAL_CVE_LOOKUP",
        "ENABLE_GITHUB_SEARCH",
        "ENABLE_REAL_RFC_LOOKUP",
        "ENABLE_SEMANTIC_SCHOLAR",
        "ENABLE_ARXIV",
        "ENABLE_NVD",
    ):
        assert f'{name} = "true"' in script
        assert f"{name}=false" in example
    assert "backend.main:app" in script
    assert "--workers 1" in script
    assert "--reload" not in script
    assert 'SetEnvironmentVariable($name, $previous[$name], "Process")' in script
    assert "ENABLE_MCP_SERVER=false" in example and "no-op" in example
    assert "ENABLE_REAL_STANDARD_FETCH=false" in example


def test_mcp_child_inherits_the_same_explicit_tool_and_source_gates(monkeypatch):
    gates = (
        "ENABLE_REAL_PAPER_SEARCH",
        "ENABLE_REAL_CVE_LOOKUP",
        "ENABLE_GITHUB_SEARCH",
        "ENABLE_REAL_RFC_LOOKUP",
        "ENABLE_SEMANTIC_SCHOLAR",
        "ENABLE_ARXIV",
        "ENABLE_NVD",
    )
    for name in gates:
        monkeypatch.setenv(name, "false")

    params = MCPClientGateway()._make_server_params()

    assert params.env is not None
    assert {name: params.env[name] for name in gates} == {name: "false" for name in gates}


def test_gateway_start_is_reusable_session():
    gateway = MCPClientGateway()

    gateway.start()
    first_params = gateway._server_params
    gateway.start()

    assert gateway.is_available is True
    assert gateway._server_params is first_params
    assert gateway.start_count == 1


def test_gateway_lists_tools_with_mcp_status():
    gateway = MCPClientGateway()
    gateway.start()

    tools = gateway.list_tools()
    names = {tool["name"] for tool in tools}

    assert {
        "course_retriever",
        "paper_search",
        "vuln_lookup",
        "standard_lookup",
        "mermaid_generator",
    }.issubset(names)
    assert {
        "paper_search_real",
        "vuln_lookup_real",
        "standard_metadata_lookup",
        "github_project_search",
    }.issubset(names)
    assert all(tool["mcp_available"] is True for tool in tools)
    assert all(tool["status"] == "mcp" for tool in tools)


def test_learning_logger_uses_absolute_sqlite_db_path_from_environment(tmp_path, monkeypatch):
    db_path = (tmp_path / "configured.db").resolve()
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))

    logger = LearningLogger()

    assert logger.db_path == db_path
    assert db_path.exists()


def test_api_then_mcp_share_configured_sqlite_cache(tmp_path, monkeypatch):
    db_path = (tmp_path / "shared-api-first.db").resolve()
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    query = ResourceQuery(query="3GPP TS 24.501", resource_type="standard", page_size=1)
    api_service = ExternalResourceService(
        storage=LearningLogger(),
        registry=ConnectorRegistry([StandardSourceConnector()]),
    )

    api_first = asyncio.run(api_service.search(query, connector_name="standard_source"))
    runs_after_api = len(LearningLogger().connector_sync_runs(limit=100))
    gateway = MCPClientGateway()
    gateway.start()
    mcp_call = gateway.call_tool(
        "standard_metadata_lookup",
        {"query": "3GPP TS 24.501", "top_k": 1},
        run_id="s3-api-first",
        step_id="step-1",
    )
    runs_after_mcp = len(LearningLogger().connector_sync_runs(limit=100))

    assert api_first["cache_hit"] is False
    assert api_first["delivery_mode"] == "local_metadata"
    assert runs_after_api == 1
    assert gateway.is_available is True, _gateway_mcp_error(gateway)
    assert mcp_call.result.success is True
    assert mcp_call.result.data["cache_hit"] is True
    assert mcp_call.result.data["delivery_mode"] == "fresh_cache"
    assert mcp_call.result.data["origin_kind"] == "local_catalog"
    assert runs_after_mcp == runs_after_api


def test_mcp_then_api_share_configured_sqlite_cache(tmp_path, monkeypatch):
    db_path = (tmp_path / "shared-mcp-first.db").resolve()
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    gateway = MCPClientGateway()
    gateway.start()

    mcp_first = gateway.call_tool(
        "standard_metadata_lookup",
        {"query": "3GPP TS 33.501", "top_k": 1},
        run_id="s3-mcp-first",
        step_id="step-1",
    )
    runs_after_mcp = len(LearningLogger().connector_sync_runs(limit=100))
    api_service = ExternalResourceService(
        storage=LearningLogger(),
        registry=ConnectorRegistry([StandardSourceConnector()]),
    )
    api_second = asyncio.run(
        api_service.search(
            ResourceQuery(query="3GPP TS 33.501", resource_type="standard", page_size=1),
            connector_name="standard_source",
        )
    )
    runs_after_api = len(LearningLogger().connector_sync_runs(limit=100))

    assert gateway.is_available is True, _gateway_mcp_error(gateway)
    assert mcp_first.result.success is True
    assert mcp_first.result.data["cache_hit"] is False
    assert mcp_first.result.data["delivery_mode"] == "local_metadata"
    assert runs_after_mcp == 1
    assert api_second["cache_hit"] is True
    assert api_second["delivery_mode"] == "fresh_cache"
    assert api_second["origin_kind"] == "local_catalog"
    assert runs_after_api == runs_after_mcp


def test_gateway_calls_real_stdio_mcp_server_successfully():
    gateway = MCPClientGateway()
    gateway.start()

    call = gateway.call_tool(
        "mermaid_generator",
        {"topic": "MCP RAG", "diagram_type": "flowchart"},
        run_id="run-real-mcp",
        step_id="step-1",
    )

    assert gateway.is_available is True, _gateway_mcp_error(gateway)
    assert call.result.success is True
    assert call.fallback_used is False
    assert call.result.metadata["protocol_is_error"] is False
    assert "mermaid" in call.result.data


def test_gateway_real_mcp_parameter_error_is_controlled():
    gateway = MCPClientGateway()
    gateway.start()

    call = gateway.call_tool(
        "mermaid_generator",
        {"diagram_type": "flowchart"},
        run_id="run-param-error",
        step_id="step-1",
    )

    assert gateway.is_available is True, _gateway_mcp_error(gateway)
    assert call.fallback_used is False
    assert call.result.success is False
    assert call.result.error_code == "INVALID_ARGUMENTS"
    assert call.result.metadata["protocol_is_error"] is True


def test_gateway_unstartable_server_reports_fallback_status():
    gateway = MCPClientGateway(mcp_command="definitely-missing-mcp-command", mcp_timeout_seconds=0.5)
    gateway.start()

    tools = gateway.list_tools()
    call = gateway.call_tool(
        "mermaid_generator",
        {"topic": "fallback", "diagram_type": "flowchart"},
        run_id="run-no-server",
        step_id="step-1",
    )

    assert gateway.is_available is False
    assert all(tool["status"] == "local_fallback" for tool in tools)
    assert all(tool["mcp_available"] is False for tool in tools)
    assert any(tool.get("mcp_error") for tool in tools)
    assert call.result.success is True
    assert call.fallback_used is True


def test_gateway_start_timeout_reports_fallback_status():
    gateway = MCPClientGateway(
        mcp_args=["-c", "import time; time.sleep(10)"],
        mcp_timeout_seconds=0.2,
    )
    gateway.start()

    tools = gateway.list_tools()

    assert gateway.is_available is False
    assert all(tool["status"] == "local_fallback" for tool in tools)
    assert any(tool.get("mcp_error") for tool in tools)


def test_stopped_gateway_falls_back_for_low_risk_read_only_tool():
    gateway = MCPClientGateway()

    call = gateway.call_tool(
        "mermaid_generator",
        {"topic": "MCP RAG", "diagram_type": "flowchart"},
        run_id="run-1",
        step_id="step-1",
    )

    assert call.result.success is True
    assert call.fallback_used is True
    assert call.observation.fallback_used is True
    assert call.observation.metadata["content_trust"] == "untrusted"


def test_high_risk_or_network_tool_does_not_fallback():
    original = TOOL_DEFINITIONS["paper_search"]
    TOOL_DEFINITIONS["paper_search"] = original.model_copy(update={"risk": "HIGH", "network": True})
    gateway = MCPClientGateway()
    try:
        call = gateway.call_tool(
            "paper_search",
            {"query": "LLM", "top_k": 1},
            run_id="run-2",
            step_id="step-1",
        )
    finally:
        TOOL_DEFINITIONS["paper_search"] = original

    assert call.result.success is False
    assert call.result.error_code == "TOOL_POLICY_DENIED"
    assert call.fallback_used is False


def test_disabled_or_unknown_tool_is_not_executed():
    gateway = MCPClientGateway()
    unknown = gateway.call_tool(
        "delete_file",
        {"path": ".env"},
        run_id="run-3",
        step_id="step-1",
    )

    assert unknown.result.success is False
    assert unknown.result.error_code == "TOOL_NOT_FOUND"


def test_tool_call_records_run_step_and_redacts_arguments():
    gateway = MCPClientGateway()

    gateway.call_tool(
        "mermaid_generator",
        {"topic": "sk-secret123456789", "diagram_type": "flowchart", "api_key": "abc"},
        run_id="run-4",
        step_id="step-7",
        step_number=7,
    )

    record = gateway.tool_call_history()[0]
    assert record["run_id"] == "run-4"
    assert record["step_id"] == "step-7"
    assert record["arguments"]["api_key"] == "***REDACTED***"
    assert "sk-secret" not in str(record)


def test_gateway_retries_mcp_call(monkeypatch):
    gateway = MCPClientGateway(max_retries=2)
    gateway.start()
    attempts = {"count": 0}

    def flaky_call(name, arguments):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary")
        return ToolResult(success=True, data={"ok": True}, metadata={"tool_name": name})

    monkeypatch.setattr(gateway, "_call_mcp_tool_once", flaky_call)

    call = gateway.call_tool(
        "mermaid_generator",
        {"topic": "retry", "diagram_type": "flowchart"},
        run_id="run-5",
        step_id="step-1",
    )

    assert call.result.success is True
    assert attempts["count"] == 2
    assert call.fallback_used is False


def test_gateway_does_not_replay_uncertain_remote_timeout(monkeypatch):
    original = TOOL_DEFINITIONS["paper_search_real"]
    TOOL_DEFINITIONS["paper_search_real"] = original.model_copy(update={"enabled": True})
    gateway = MCPClientGateway(max_retries=3)
    gateway._started = True
    gateway._server_params = object()
    attempts = {"count": 0}

    def timed_out_call(name, arguments):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return ToolResult(
                success=False,
                error_code="MCP_CALL_TIMEOUT",
                error_message="timed out after dispatch",
                metadata={"tool_name": name},
            )
        return ToolResult(success=True, data={"status": "SUCCESS"}, metadata={"tool_name": name})

    monkeypatch.setattr(gateway, "_call_mcp_tool_once", timed_out_call)
    try:
        call = gateway.call_tool(
            "paper_search_real",
            {"query": "5G NAS security", "top_k": 1, "source": "arxiv"},
            run_id="l4-no-replay",
            step_id="step-1",
        )
    finally:
        TOOL_DEFINITIONS["paper_search_real"] = original

    assert attempts["count"] == 1
    assert call.result.success is False
    assert call.result.error_code == "MCP_CALL_TIMEOUT"
    assert call.result.metadata["attempt_count"] == 1
    assert call.result.metadata["outcome_unknown"] is True
    assert call.result.metadata["retry_suppressed"] is True


def test_gateway_rejects_dispatch_when_agent_budget_cannot_cover_server_deadline(monkeypatch):
    original = TOOL_DEFINITIONS["paper_search_real"]
    TOOL_DEFINITIONS["paper_search_real"] = original.model_copy(update={"enabled": True})
    gateway = MCPClientGateway(mcp_timeout_seconds=30)
    gateway._started = True
    gateway._server_params = object()
    dispatched = {"count": 0}

    def must_not_dispatch(name, arguments):
        dispatched["count"] += 1
        raise AssertionError("tool must not be dispatched")

    monkeypatch.setattr(gateway, "_call_mcp_tool_once", must_not_dispatch)
    try:
        call = gateway.call_tool(
            "paper_search_real",
            {"query": "5G NAS security", "top_k": 1, "source": "arxiv"},
            run_id="l4-budget",
            step_id="step-1",
            remaining_timeout_seconds=5,
        )
    finally:
        TOOL_DEFINITIONS["paper_search_real"] = original

    assert dispatched["count"] == 0
    assert call.result.success is False
    assert call.result.error_code == "AGENT_TOOL_BUDGET_EXHAUSTED"
    assert call.result.metadata["attempt_count"] == 0
    assert call.result.metadata["outcome"] == "not_dispatched"


def test_gateway_concurrency_limit_returns_error():
    gateway = MCPClientGateway(max_concurrent_calls=1, acquire_timeout_seconds=0.01)
    assert gateway._semaphore.acquire(timeout=0.1)
    try:
        started = time.perf_counter()
        call = gateway.call_tool(
            "mermaid_generator",
            {"topic": "busy", "diagram_type": "flowchart"},
            run_id="run-6",
            step_id="step-1",
        )
    finally:
        gateway._semaphore.release()

    assert time.perf_counter() - started < 2
    assert call.result.success is False
    assert call.result.error_code == "TOOL_CONCURRENCY_LIMIT"


def test_mcp_failure_does_not_fallback_while_session_is_available(monkeypatch):
    gateway = MCPClientGateway(max_retries=1)
    gateway.start()

    def broken_call(name, arguments):
        raise RuntimeError("down")

    monkeypatch.setattr(gateway, "_call_mcp_tool_once", broken_call)

    call = gateway.call_tool(
        "mermaid_generator",
        {"topic": "failure", "diagram_type": "flowchart"},
        run_id="run-7",
        step_id="step-1",
    )

    assert call.result.success is False
    assert call.result.error_code == "MCP_CALL_FAILED"
    assert call.fallback_used is False


def test_replacing_session_with_non_callable_object_no_longer_succeeds():
    gateway = MCPClientGateway(max_retries=1)
    gateway.start()
    gateway._server_params = object()

    call = gateway.call_tool(
        "mermaid_generator",
        {"topic": "must fail through MCP path", "diagram_type": "flowchart"},
        run_id="run-bad-session",
        step_id="step-1",
    )

    assert call.result.success is False
    assert call.result.error_code == "MCP_CALL_FAILED"
    assert call.fallback_used is False


def test_agent_catalog_only_exposes_allowed_tools():
    original = TOOL_DEFINITIONS["vuln_lookup"]
    TOOL_DEFINITIONS["vuln_lookup"] = original.model_copy(update={"network": True})
    gateway = MCPClientGateway()
    try:
        catalog = gateway.get_agent_tool_catalog(role="student")
    finally:
        TOOL_DEFINITIONS["vuln_lookup"] = original

    names = {tool["name"] for tool in catalog}
    assert "vuln_lookup" not in names
    assert "mermaid_generator" in names
