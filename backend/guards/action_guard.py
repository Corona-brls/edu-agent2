"""Action validation before the backend executes planner decisions."""

from __future__ import annotations

import time

from pydantic import ValidationError

from backend.agent.schemas import AgentAction, ActionDecision
from backend.agent.state import AgentState
from backend.tools.tool_policy import can_agent_execute_tool, get_tool_definition
from mcp_server.tools import get_tool_input_models


INPUT_MODELS = get_tool_input_models()


class ActionGuardError(ValueError):
    def __init__(self, message: str, *, error_code: str, run_id: str, trace_id: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.run_id = run_id
        self.trace_id = trace_id


class ActionGuard:
    def validate(
        self,
        *,
        state: AgentState,
        decision: ActionDecision,
        started_at: float,
        timeout_seconds: float,
        role: str = "student",
    ) -> None:
        if not isinstance(decision.action, AgentAction):
            self._deny("非法动作。", "INVALID_ACTION", state)
        if (time.perf_counter() - started_at) >= timeout_seconds:
            self._deny("Agent 总超时。", "AGENT_TIMEOUT", state)
        if state.current_step >= state.max_steps:
            self._deny("达到步骤上限。", "MAX_STEPS", state)
        if decision.action == AgentAction.CALL_TOOL:
            self._validate_tool_call(state, decision, role)

    def _validate_tool_call(self, state: AgentState, decision: ActionDecision, role: str) -> None:
        if not decision.tool_name:
            self._deny("工具调用缺少 tool_name。", "MISSING_TOOL_NAME", state)
        if state.tool_call_count >= state.max_tool_calls:
            self._deny("达到工具调用上限。", "MAX_TOOL_CALLS", state)

        definition = get_tool_definition(decision.tool_name)
        if definition is None:
            self._deny(f"工具不存在：{decision.tool_name}", "TOOL_NOT_FOUND", state)
        if not can_agent_execute_tool(decision.tool_name, role=role):
            self._deny(f"当前角色无权自动执行工具：{decision.tool_name}", "TOOL_POLICY_DENIED", state)
        if definition.risk != "LOW" or not definition.read_only:
            self._deny(f"工具风险策略不允许自动执行：{decision.tool_name}", "TOOL_POLICY_DENIED", state)

        model = INPUT_MODELS.get(decision.tool_name)
        if model is None:
            self._deny(f"工具缺少参数模型：{decision.tool_name}", "TOOL_SCHEMA_MISSING", state)
        try:
            model.model_validate(decision.arguments)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            message = first.get("msg") or "工具参数不符合 schema。"
            self._deny(f"工具参数不符合 schema：{message}", "INVALID_ARGUMENTS", state)

    def _deny(self, message: str, error_code: str, state: AgentState) -> None:
        raise ActionGuardError(message, error_code=error_code, run_id=state.run_id, trace_id=state.trace_id)
