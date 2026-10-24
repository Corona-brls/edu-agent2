"""Conservative intent routing used before the planner makes a decision."""

import re

from backend.agent.schemas import AgentIntent
from backend.rag.course_document_constraint import (
    has_explicit_course_document_constraint_syntax,
)


_PRACTICE_TERMS = (
    "\u5b9e\u8df5\u65b9\u6848", "\u5b9e\u9a8c\u65b9\u6848", "\u5b9e\u9a8c\u8bbe\u8ba1",
    "\u7814\u7a76\u76ee\u6807", "\u65b9\u6cd5\u8bbe\u8ba1", "\u5b9e\u9a8c\u6b65\u9aa4",
    "\u7ed3\u679c\u5206\u6790", "\u516d\u6b65", "practice guide", "experiment plan",
)
_INNOVATION_TERMS = (
    "\u521b\u65b0\u542f\u53d1", "\u521b\u65b0\u95ee\u9898", "\u542f\u53d1\u5f0f",
    "\u521b\u65b0\u65b9\u5411", "\u63d0\u51fa\u521b\u65b0", "innovation",
)
_SOURCE_TERMS = {
    "paper": ("\u8bba\u6587", "\u6587\u732e", "arxiv", "semantic scholar", "paper"),
    "vulnerability": ("\u6f0f\u6d1e", "cve", "cvss", "cwe", "poc"),
    "standard": ("\u6807\u51c6", "3gpp", "ts 24.501", "specification"),
    "project": ("github", "\u5f00\u6e90\u9879\u76ee", "\u4ee3\u7801\u4ed3\u5e93", "repository", "project"),
}
_SUBSTANTIVE_COURSE_TERMS = (
    "5g", "nas", "\u533a\u5757\u94fe", "\u9690\u853d\u901a\u4fe1",
    "\u7f51\u7edc\u5b89\u5168", "rag", "mcp", "cve", "3gpp", "\u6f0f\u6d1e", "\u8ba4\u8bc1", "\u9274\u6743",
)
_OFF_TOPIC_TERMS = (
    "\u751c\u70b9", "\u8d2d\u7269", "\u65c5\u6e38", "\u5929\u6c14", "\u5f69\u7968",
    "\u661f\u5ea7", "\u98df\u8c31", "\u70f9\u996a", "\u86cb\u7cd5", "dessert", "shopping", "weather", "travel",
    "recipe", "cooking",
)
_RFC_IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9_])RFC(?:[\s-]*)\d+(?![A-Za-z0-9_])", re.IGNORECASE)


def _mentions_source(normalized: str, source: str, terms: tuple[str, ...]) -> bool:
    if source == "standard" and _RFC_IDENTIFIER_RE.search(normalized):
        return True
    return any(term in normalized for term in terms)


def route_intent(text: str) -> str:
    normalized = " ".join(str(text or "").lower().split())
    if has_explicit_course_document_constraint_syntax(text):
        return AgentIntent.COURSE_QA.value
    if any(term in normalized for term in _PRACTICE_TERMS):
        return AgentIntent.PRACTICE_GUIDE.value
    if any(term in normalized for term in _INNOVATION_TERMS):
        return AgentIntent.INNOVATION.value
    if any(term in normalized for term in _OFF_TOPIC_TERMS) and not any(
        term in normalized for term in _SUBSTANTIVE_COURSE_TERMS
    ):
        return AgentIntent.OFF_TOPIC.value
    matched_sources = [
        source for source, terms in _SOURCE_TERMS.items()
        if _mentions_source(normalized, source, terms)
    ]
    if len(matched_sources) >= 2:
        return AgentIntent.MULTI_SOURCE.value
    if matched_sources == ["paper"]:
        return AgentIntent.PAPER_SEARCH.value
    if matched_sources == ["vulnerability"]:
        return AgentIntent.VULN_LOOKUP.value
    if matched_sources == ["standard"]:
        return AgentIntent.STANDARD_LOOKUP.value
    if matched_sources == ["project"]:
        return AgentIntent.PROJECT_SEARCH.value
    return AgentIntent.COURSE_QA.value


def requested_source_slots(text: str, intent: str) -> list[str]:
    normalized = " ".join(str(text or "").lower().split())
    if has_explicit_course_document_constraint_syntax(text):
        return ["course"]
    explicit = [
        source for source, terms in _SOURCE_TERMS.items()
        if _mentions_source(normalized, source, terms)
    ]
    if intent == AgentIntent.OFF_TOPIC.value:
        return []
    if intent == AgentIntent.PRACTICE_GUIDE.value:
        return explicit or ["course"]
    if intent == AgentIntent.INNOVATION.value:
        return explicit or ["paper", "project"]
    if intent == AgentIntent.MULTI_SOURCE.value:
        return explicit
    defaults = {
        AgentIntent.PAPER_SEARCH.value: ["paper"],
        AgentIntent.VULN_LOOKUP.value: ["vulnerability"],
        AgentIntent.STANDARD_LOOKUP.value: ["standard"],
        AgentIntent.PROJECT_SEARCH.value: ["project"],
    }
    return defaults.get(intent, ["course"])
