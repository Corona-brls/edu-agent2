"""Planner wrapper around an LLM provider."""

from __future__ import annotations

from backend.agent.schemas import ActionDecision, ToolSpec
from backend.agent.state import AgentState
from backend.llm.base import LLMProvider


class AgentPlanner:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def decide(self, state: AgentState) -> ActionDecision:
        tools = [
            tool if isinstance(tool, ToolSpec) else ToolSpec.model_validate(tool)
            for tool in state.available_tools
        ]
        state_decider = getattr(self.provider, "decide_from_state", None)
        if callable(state_decider):
            return state_decider(state, tools)
        return self.provider.decide_next_action(state.to_summary(), tools)
