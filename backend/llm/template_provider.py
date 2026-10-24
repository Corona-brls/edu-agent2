"""Local deterministic fallback provider."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from backend.agent.schemas import ActionDecision, AgentAction, ToolSpec
from backend.llm.base import LLMProvider, StructuredModel
from backend.llm.schemas import (
    GroundedAnswer,
    GroundedPracticeGuide,
    GroundedSpanAnswer,
    LLMRequest,
    LLMResponse,
    PracticeGuide,
)

if TYPE_CHECKING:
    from backend.agent.state import AgentState


class TemplateProvider(LLMProvider):
    provider_name = "template"
    model_name = "local-template"

    def generate_answer(self, request: LLMRequest) -> LLMResponse:
        if not request.evidence:
            text = "当前没有足够证据支持回答，建议补充课程资料后再查询。"
        else:
            snippets = "；".join(item.text[:120] for item in request.evidence[:3])
            citations = " ".join(f"[{item.source_id}]" for item in request.evidence[:3])
            text = f"根据已检索证据，{snippets} {citations}"
        return LLMResponse(
            text=text,
            provider=self.provider_name,
            model=self.model_name,
            fallback_used=True,
        )

    def generate_structured(
        self,
        request: LLMRequest,
        schema: type[StructuredModel],
    ) -> StructuredModel:
        if schema is GroundedAnswer:
            claims = []
            for item in request.evidence[:5]:
                quote = _short_grounded_quote(item.text) or _short_grounded_quote(item.title) or _short_grounded_quote(item.external_id)
                if not quote:
                    continue
                label = item.external_id or item.title or item.resource_type or item.source
                claims.append(
                    {
                        "claim_text": f"{item.resource_type or 'source'} evidence is available for {label}.",
                        "source_id": item.source_id,
                        "evidence_quote": quote,
                    }
                )
            return schema.model_validate({"claims": claims})
        if schema is GroundedSpanAnswer:
            spans = request.metadata.get("evidence_spans")
            if not isinstance(spans, list):
                spans = []
            claims = []
            for item in request.evidence:
                preferred_fields = (
                    ("title", "external_id", "text")
                    if str(item.resource_type or "").casefold() == "project"
                    else ("external_id", "title", "text")
                )
                field_order = {field: index for index, field in enumerate(preferred_fields)}
                candidates = sorted(
                    (
                        span
                        for span in spans
                        if isinstance(span, dict)
                        and str(span.get("source_id") or "") == item.source_id
                        and str(span.get("span_id") or "")
                    ),
                    key=lambda span: field_order.get(str(span.get("field") or ""), 3),
                )
                selected = next(
                    (
                        span
                        for span in candidates
                        if 1 <= len(str(span.get("exact_text") or "")) <= 360
                    ),
                    None,
                )
                if selected is None:
                    continue
                claims.append(
                    {
                        "claim_text": str(selected["exact_text"]),
                        "source_id": item.source_id,
                        "evidence_span_id": str(selected["span_id"]),
                    }
                )
                if len(claims) >= 5:
                    break
            return schema.model_validate({"claims": claims})
        if schema is GroundedPracticeGuide:
            return schema.model_validate(_extract_grounded_practice_payload(request))
        if schema is PracticeGuide:
            return schema.model_validate(
                {
                    "research_goal": f"围绕“{request.question}”形成可验证的课程实践目标。",
                    "method_design": "基于本地证据设计实验变量、观察指标和对照流程。",
                    "tools": ["本地知识库", "课程样例数据", "实验记录表"],
                    "experiment_steps": ["明确问题", "检索证据", "设计实验", "记录结果", "分析风险"],
                    "result_analysis": "对比实验现象与证据片段，说明结论边界。",
                    "innovation_points": ["结合课程资料形成可复现实验闭环"],
                    "risks": ["证据不足时不扩展事实", "避免编造论文、CVE 或标准条款"],
                }
            )
        if issubclass(schema, BaseModel):
            return schema.model_validate({})
        raise TypeError("schema must be a Pydantic model")

    def decide_next_action(
        self,
        state_summary: str,
        available_tools: list[ToolSpec],
    ) -> ActionDecision:
        lowered = state_summary.lower()
        if "证据不足" in state_summary or "无证据" in state_summary:
            return ActionDecision(
                action=AgentAction.ABSTAIN,
                reason="当前状态显示证据不足，选择拒答。",
                task_complete=True,
            )
        if "需要补充" in state_summary or "请补充" in state_summary:
            return ActionDecision(
                action=AgentAction.ASK_USER,
                arguments={"question": "请补充实验目标、数据来源或约束条件。"},
                reason="任务缺少必要上下文，需要学生补充信息。",
            )
        if "evidence" in lowered or "已检索" in state_summary:
            return ActionDecision(
                action=AgentAction.ANSWER,
                reason="已有证据，可以生成最终回答。",
                task_complete=True,
            )
        return ActionDecision(
            action=AgentAction.RETRIEVE,
            arguments={"query": state_summary[:120]},
            reason="先检索课程知识库以获取证据。",
        )

    def decide_from_state(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> ActionDecision:
        if _has_completed_trusted_evidence(state):
            return ActionDecision(
                action=AgentAction.ANSWER,
                reason="必需的可信证据来源已经完成，可以生成最终回答。",
                task_complete=True,
            )
        if _is_resuming_after_user_clarification(state):
            return ActionDecision(
                action=AgentAction.RETRIEVE,
                arguments={"query": _resume_query(state)},
                reason="用户已补充信息，继续检索同一个 AgentRun 的证据。",
            )
        return self.decide_next_action(state.to_summary(), available_tools)


_SOURCE_ALIASES = {
    "cve": "vulnerability",
    "github": "project",
    "repo": "project",
    "repository": "project",
    "rfc": "standard",
}


def _source_slot(value: str) -> str:
    normalized = str(value or "").strip().casefold()
    return _SOURCE_ALIASES.get(normalized, normalized)


def _has_completed_trusted_evidence(state: AgentState) -> bool:
    required = {_source_slot(value) for value in state.required_sources if _source_slot(value)}
    if not required:
        return False
    completed = {_source_slot(value) for value in state.completed_sources if _source_slot(value)}
    if not required.issubset(completed):
        return False
    trusted_sources = {
        _source_slot(item.resource_type or item.source_type)
        for item in state.evidence
        if item.trusted and _source_slot(item.resource_type or item.source_type)
    }
    return required.issubset(trusted_sources)


def _is_resuming_after_user_clarification(state: AgentState) -> bool:
    if state.pending_question:
        return False
    if len(state.messages) < 2:
        return False
    latest_user_message = str(state.messages[-1].get("content") or "").strip()
    if not latest_user_message:
        return False
    return any(item.action == "ask_user" and item.success for item in state.observations)


def _resume_query(state: AgentState) -> str:
    latest_user_message = str(state.messages[-1].get("content") or "").strip()
    if latest_user_message and latest_user_message not in state.user_goal:
        return f"{state.user_goal}\n{latest_user_message}"[:500]
    return state.user_goal[:500]


def _short_grounded_quote(value: str, *, max_chars: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    return text[:max_chars].rstrip(" ,.;:")


_PRACTICE_FIELD_CUES: dict[str, tuple[str, ...]] = {
    "research_goal": ("研究目标", "研究目的", "目标是", "objective", "research goal"),
    "method_design": ("方法设计", "研究方法", "方法采用", "method design", "methodology"),
    "tools": ("工具选择", "实验工具", "使用工具", "工具只需", "工具限", "tool selection", "tools"),
    "result_analysis": ("结果分析", "分析结果", "result analysis", "analyze the result"),
    "innovation_points": ("创新点", "创新之处", "innovation point", "innovation"),
    "risks": ("风险与注意事项", "注意事项", "风险", "risk and caution", "risks"),
}
_PRACTICE_STEP_RE = re.compile(
    r"(?:实验)?步骤\s*(?:[1-9]|1[0-2]|[一二三四五六七八九十]{1,2})|"
    r"\bstep\s*(?:[1-9]|1[0-2])\b",
    flags=re.IGNORECASE,
)
_PRACTICE_PROCEDURAL_RE = re.compile(
    r"(?:实验(?:应当|应|需|需要)?记录|实验记录|记录实验)(?!\s*(?:无需|不需要|不是|并非))",
    flags=re.IGNORECASE,
)


def _extract_grounded_practice_payload(request: LLMRequest) -> dict[str, list[dict[str, str]]]:
    raw_spans = request.metadata.get("evidence_spans")
    spans = [span for span in raw_spans if isinstance(span, dict)] if isinstance(raw_spans, list) else []
    text_spans = [span for span in spans if str(span.get("field") or "") == "text"]
    used_span_ids: set[str] = set()

    def claim_for(span: dict[str, Any]) -> dict[str, str]:
        source_id = str(span.get("source_id") or "")
        span_id = str(span.get("span_id") or "")
        exact_text = " ".join(str(span.get("exact_text") or "").split())
        return {
            "claim_text": exact_text[:360].rstrip(" ,.;:，。；："),
            "source_id": source_id,
            "evidence_span_id": span_id,
        }

    def select_by_cues(cues: tuple[str, ...]) -> list[dict[str, str]]:
        for span in text_spans:
            span_id = str(span.get("span_id") or "")
            text = str(span.get("exact_text") or "")
            lowered = text.casefold()
            if span_id and span_id not in used_span_ids and any(cue.casefold() in lowered for cue in cues):
                used_span_ids.add(span_id)
                return [claim_for(span)]
        return []

    required_steps = request.metadata.get("required_practice_step_count")
    step_spans = [
        span
        for span in text_spans
        if _PRACTICE_STEP_RE.search(str(span.get("exact_text") or ""))
        or (
            not isinstance(required_steps, int)
            and _PRACTICE_PROCEDURAL_RE.search(str(span.get("exact_text") or ""))
        )
    ]
    step_limit = required_steps if isinstance(required_steps, int) and 1 <= required_steps <= 12 else len(step_spans)
    selected_steps = step_spans[:step_limit]
    for span in selected_steps:
        used_span_ids.add(str(span.get("span_id") or ""))

    return {
        "research_goal": select_by_cues(_PRACTICE_FIELD_CUES["research_goal"]),
        "method_design": select_by_cues(_PRACTICE_FIELD_CUES["method_design"]),
        "tools": select_by_cues(_PRACTICE_FIELD_CUES["tools"]),
        "experiment_steps": [claim_for(span) for span in selected_steps],
        "result_analysis": select_by_cues(_PRACTICE_FIELD_CUES["result_analysis"]),
        "innovation_points": select_by_cues(_PRACTICE_FIELD_CUES["innovation_points"]),
        "risks": select_by_cues(_PRACTICE_FIELD_CUES["risks"]),
    }
