import time

import pytest

from backend.agent.schemas import ActionDecision, AgentAction
from backend.agent.state import AgentState
from backend.guards.action_guard import ActionGuard, ActionGuardError
from backend.tools.tool_policy import get_agent_tool_catalog
from mcp_server.tools import TOOL_DEFINITIONS, get_tool_input_models


def _state() -> AgentState:
    return AgentState(user_goal="demo")


def test_action_guard_allows_safe_tool_call():
    decision = ActionDecision(
        action=AgentAction.CALL_TOOL,
        tool_name="mermaid_generator",
        arguments={"topic": "Agent", "diagram_type": "flowchart"},
        reason="need diagram",
    )

    ActionGuard().validate(
        state=_state(),
        decision=decision,
        started_at=time.perf_counter(),
        timeout_seconds=120,
    )


def test_action_guard_blocks_unknown_tool():
    decision = ActionDecision(
        action=AgentAction.CALL_TOOL,
        tool_name="delete_file",
        arguments={"path": ".env"},
        reason="bad",
    )

    with pytest.raises(ActionGuardError) as exc_info:
        ActionGuard().validate(
            state=_state(),
            decision=decision,
            started_at=time.perf_counter(),
            timeout_seconds=120,
        )

    assert exc_info.value.error_code == "TOOL_NOT_FOUND"
    assert exc_info.value.run_id
    assert exc_info.value.trace_id


def test_action_guard_blocks_disabled_high_risk_or_network_tool():
    original = TOOL_DEFINITIONS["paper_search"]
    TOOL_DEFINITIONS["paper_search"] = original.model_copy(update={"risk": "HIGH", "network": True})
    decision = ActionDecision(
        action=AgentAction.CALL_TOOL,
        tool_name="paper_search",
        arguments={"query": "LLM", "top_k": 1},
        reason="policy check",
    )
    try:
        with pytest.raises(ActionGuardError) as exc_info:
            ActionGuard().validate(
                state=_state(),
                decision=decision,
                started_at=time.perf_counter(),
                timeout_seconds=120,
            )
    finally:
        TOOL_DEFINITIONS["paper_search"] = original

    assert exc_info.value.error_code == "TOOL_POLICY_DENIED"


def test_action_guard_blocks_invalid_arguments_and_limits():
    state = _state()
    state.tool_call_count = state.max_tool_calls
    decision = ActionDecision(
        action=AgentAction.CALL_TOOL,
        tool_name="mermaid_generator",
        arguments={"topic": "Agent", "unexpected": True},
        reason="bad args",
    )

    with pytest.raises(ActionGuardError) as exc_info:
        ActionGuard().validate(
            state=state,
            decision=decision,
            started_at=time.perf_counter(),
            timeout_seconds=120,
        )

    assert exc_info.value.error_code == "MAX_TOOL_CALLS"


def test_action_guard_blocks_invalid_tool_schema_arguments():
    decision = ActionDecision(
        action=AgentAction.CALL_TOOL,
        tool_name="mermaid_generator",
        arguments={"topic": "", "diagram_type": "bad"},
        reason="bad args",
    )

    with pytest.raises(ActionGuardError) as exc_info:
        ActionGuard().validate(
            state=_state(),
            decision=decision,
            started_at=time.perf_counter(),
            timeout_seconds=120,
        )

    assert exc_info.value.error_code == "INVALID_ARGUMENTS"


def test_action_guard_blocks_timeout():
    with pytest.raises(ActionGuardError) as exc_info:
        ActionGuard().validate(
            state=_state(),
            decision=ActionDecision(action=AgentAction.RETRIEVE, arguments={}, reason="retrieve"),
            started_at=time.perf_counter() - 10,
            timeout_seconds=0.01,
        )

    assert exc_info.value.error_code == "AGENT_TIMEOUT"


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("standard_metadata_lookup", {"query": "3GPP TS 24.501", "top_k": 1}),
        ("github_project_search", {"query": "open5gs", "top_k": 1}),
    ],
)
def test_action_guard_has_schema_for_registered_standard_and_github_tools(
    monkeypatch,
    tool_name,
    arguments,
):
    if tool_name == "github_project_search":
        original = TOOL_DEFINITIONS[tool_name]
        monkeypatch.setitem(TOOL_DEFINITIONS, tool_name, original.model_copy(update={"enabled": True}))

    ActionGuard().validate(
        state=_state(),
        decision=ActionDecision(
            action=AgentAction.CALL_TOOL,
            tool_name=tool_name,
            arguments=arguments,
            reason="J2 schema parity",
        ),
        started_at=time.perf_counter(),
        timeout_seconds=120,
    )


def test_action_guard_schema_covers_every_agent_visible_tool(monkeypatch):
    original = TOOL_DEFINITIONS["github_project_search"]
    monkeypatch.setitem(
        TOOL_DEFINITIONS,
        "github_project_search",
        original.model_copy(update={"enabled": True}),
    )
    catalog = get_agent_tool_catalog(role="student")

    for tool in catalog:
        schema = tool["input_schema"]
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        arguments = {}
        for name, field in properties.items():
            if name not in required:
                continue
            if field.get("type") == "string":
                arguments[name] = "J2 validation"
            elif field.get("type") == "integer":
                arguments[name] = int(field.get("minimum", 1))
        ActionGuard().validate(
            state=_state(),
            decision=ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name=tool["name"],
                arguments=arguments,
                reason="catalog schema parity",
            ),
            started_at=time.perf_counter(),
            timeout_seconds=120,
        )

    assert set(get_tool_input_models()) == set(TOOL_DEFINITIONS)


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("standard_metadata_lookup", {"top_k": 1}),
        ("github_project_search", {"query": "open5gs", "top_k": 11}),
    ],
)
def test_action_guard_keeps_strict_parameter_validation(monkeypatch, tool_name, arguments):
    if tool_name == "github_project_search":
        original = TOOL_DEFINITIONS[tool_name]
        monkeypatch.setitem(TOOL_DEFINITIONS, tool_name, original.model_copy(update={"enabled": True}))

    with pytest.raises(ActionGuardError) as exc_info:
        ActionGuard().validate(
            state=_state(),
            decision=ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name=tool_name,
                arguments=arguments,
                reason="J2 invalid arguments",
            ),
            started_at=time.perf_counter(),
            timeout_seconds=120,
        )

    assert exc_info.value.error_code == "INVALID_ARGUMENTS"


@pytest.mark.parametrize("tool_name", ["standard_metadata_lookup", "github_project_search"])
def test_action_guard_directly_rejects_missing_required_query(monkeypatch, tool_name):
    original = TOOL_DEFINITIONS[tool_name]
    monkeypatch.setitem(TOOL_DEFINITIONS, tool_name, original.model_copy(update={"enabled": True}))

    with pytest.raises(ActionGuardError) as exc_info:
        ActionGuard().validate(
            state=_state(),
            decision=ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name=tool_name,
                arguments={"top_k": 1},
                reason="0R2B direct missing-query contract",
            ),
            started_at=time.perf_counter(),
            timeout_seconds=120,
        )

    assert exc_info.value.error_code == "INVALID_ARGUMENTS"
