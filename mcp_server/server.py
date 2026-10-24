"""Standard MCP server exposing course-agent tools."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from mcp import types
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from mcp_server.tools import TOOL_DEFINITIONS, call_tool, get_tool_input_model


def build_server() -> MCPServer:
    server = MCPServer(
        name="course-agent-mcp",
        title="Course Agent MCP Server",
        description="Read-only local tools for the network security course agent.",
        version="0.1.0",
    )

    for name, definition in TOOL_DEFINITIONS.items():
        server.add_tool(
            _make_tool_handler(name),
            name=name,
            title=definition.name,
            description=definition.agent_description,
            structured_output=True,
            meta={
                "risk": definition.risk,
                "network": definition.network,
                "read_only": definition.read_only,
                "enabled": definition.enabled,
            },
        )
        registered_tool = server._tool_manager.get_tool(name)
        if registered_tool is not None:
            registered_tool.parameters = definition.input_schema
    return server


def _make_tool_handler(name: str):
    input_model = get_tool_input_model(name)
    if input_model is None:
        raise RuntimeError(f"MCP tool is missing an input model: {name}")

    def handler(**arguments: Any) -> types.CallToolResult:
        result = call_tool(name, arguments)
        payload = result.model_dump(mode="json")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
            structured_content=payload,
            is_error=not result.success,
        )

    handler.__name__ = name
    handler.__signature__ = _signature_for_model(input_model)  # type: ignore[attr-defined]
    return handler


def _signature_for_model(model: type[BaseModel]) -> inspect.Signature:
    parameters = []
    for field_name, field in model.model_fields.items():
        default = None if field.is_required() else field.default
        parameters.append(
            inspect.Parameter(
                field.alias or field_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=field.annotation,
            )
        )
    return inspect.Signature(parameters=parameters, return_annotation=dict[str, Any])


async def main() -> None:
    server = build_server()
    await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
