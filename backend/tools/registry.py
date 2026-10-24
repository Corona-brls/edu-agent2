"""Local MCP-like tool registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: str = "LOW"
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


_TOOLS: dict[str, Tool] = {}
_INTENT_TOOL_MAP = {
    "course_qa": "course_retriever",
    "practice_guide": "course_retriever",
    "paper_search": "paper_search",
    "vuln_lookup": "vuln_lookup",
    "standard_lookup": "standard_lookup",
}


def register_tool(tool: Tool) -> None:
    _TOOLS[tool.name] = tool


def get_tool(name: str) -> dict | None:
    tool = _TOOLS.get(name)
    return tool.to_dict() if tool else None


def select_tool_for_intent(intent: str) -> dict:
    tool_name = _INTENT_TOOL_MAP.get(intent, "course_retriever")
    tool = _TOOLS.get(tool_name)
    if tool is None or not tool.enabled:
        return _TOOLS["course_retriever"].to_dict()
    return tool.to_dict()


def list_tools() -> list[dict]:
    return [tool.to_dict() for tool in _TOOLS.values()]


register_tool(Tool(name="course_retriever", description="检索课程知识库"))
register_tool(Tool(name="paper_search", description="检索本地论文样例"))
register_tool(Tool(name="vuln_lookup", description="检索本地漏洞样例"))
register_tool(Tool(name="standard_lookup", description="检索标准片段样例"))
register_tool(Tool(name="mermaid_generator", description="生成 Mermaid 架构图或流程图"))
