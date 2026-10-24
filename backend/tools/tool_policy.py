"""Tool policy for exposing safe tools to the agent planner."""

from __future__ import annotations

from typing import Any

from mcp_server.schemas import ToolDefinition
from mcp_server.tools import TOOL_DEFINITIONS, call_tool, list_tool_definitions

_AGENT_NETWORK_TOOLS = {
    "paper_search_real",
    "vuln_lookup_real",
    "github_project_search",
    "standard_lookup_real",
}
_FORMAL_AGENT_FORBIDDEN_LOCAL_TOOLS = {"paper_search", "vuln_lookup", "standard_lookup"}


def get_tool_definition(name: str) -> ToolDefinition | None:
    return TOOL_DEFINITIONS.get(name)


def get_agent_tool_catalog(role: str = "student") -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for tool in list_tool_definitions():
        if not _is_visible_to_agent(tool, role):
            continue
        catalog.append(
            {
                "name": tool.name,
                "description": tool.agent_description,
                "input_schema": tool.input_schema,
                "risk": tool.risk,
                "network": tool.network,
                "read_only": tool.read_only,
            }
        )
    return catalog


def list_policy_tools() -> list[dict[str, Any]]:
    return [tool.model_dump() for tool in list_tool_definitions()]


def call_policy_tool(name: str, arguments: dict[str, Any]):
    return call_tool(name, arguments)


def can_agent_execute_tool(name: str, role: str = "student") -> bool:
    tool = get_tool_definition(name)
    return tool is not None and _is_visible_to_agent(tool, role)


def _is_visible_to_agent(tool: ToolDefinition, role: str) -> bool:
    effective_role = "student" if role == "demo" else role
    return (
        (role == "demo" or tool.name not in _FORMAL_AGENT_FORBIDDEN_LOCAL_TOOLS)
        and
        tool.enabled
        and tool.read_only
        and tool.risk == "LOW"
        and effective_role in tool.roles
        and (not tool.network or tool.name in _AGENT_NETWORK_TOOLS)
    )
