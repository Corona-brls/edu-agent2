"""Schemas shared by the agent planner and orchestrator."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentAction(StrEnum):
    RETRIEVE = "retrieve"
    CALL_TOOL = "call_tool"
    ASK_USER = "ask_user"
    ANSWER = "answer"
    ABSTAIN = "abstain"


class AgentIntent(StrEnum):
    COURSE_QA = "course_qa"
    PAPER_SEARCH = "paper_search"
    VULN_LOOKUP = "vuln_lookup"
    STANDARD_LOOKUP = "standard_lookup"
    PROJECT_SEARCH = "project_search"
    PRACTICE_GUIDE = "practice_guide"
    INNOVATION = "innovation"
    MULTI_SOURCE = "multi_source"
    OFF_TOPIC = "off_topic"


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    risk: str = "LOW"
    network: bool = False
    read_only: bool = True
    mcp_available: bool = False
    enabled: bool = True


class ActionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: AgentAction
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(..., min_length=1)
    task_complete: bool = False
    intent: AgentIntent | None = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "ActionDecision":
        if self.action == AgentAction.CALL_TOOL and not self.tool_name:
            raise ValueError("call_tool action requires tool_name")
        if self.action != AgentAction.CALL_TOOL and self.tool_name:
            raise ValueError("tool_name is only allowed for call_tool")
        if self.action in {AgentAction.RETRIEVE, AgentAction.CALL_TOOL} and self.task_complete:
            raise ValueError("retrieve and call_tool cannot mark task_complete")
        if self.action == AgentAction.ASK_USER and "question" not in self.arguments:
            raise ValueError("ask_user action requires arguments.question")
        return self
