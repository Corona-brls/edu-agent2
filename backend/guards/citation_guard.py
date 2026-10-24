"""Citation checks for grounded final answers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from backend.llm.schemas import EvidenceItem
from backend.rag.retriever import tokenize


CITATION_RE = re.compile(r"\[(S\d+)\]")
IDENTIFIER_RE = re.compile(
    r"\b(?:CVE-\d{4}-\d{4,8}|[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+){2,})\b",
    flags=re.IGNORECASE,
)
REFUSAL_RE = re.compile(
    r"(?:\u65e0\u6cd5|\u4e0d\u80fd|\u62d2\u7edd|\u4e0d\u4e88)(?:\u76f4\u63a5|\u53ef\u9760\u5730|\u51c6\u786e\u5730)?(?:\u56de\u7b54|\u786e\u8ba4|\u63d0\u4f9b)"
    r"|\u8bc1\u636e\u4e0d\u8db3|\u4e0d\u751f\u6210\u7ed3\u8bba|insufficient evidence|(?:cannot|can't|unable to) answer",
    flags=re.IGNORECASE,
)
PAPER_ASSERTION_RE = re.compile(
    r"\b(?:reduce[sd]?|lower(?:s|ed)?|improve[sd]?|increase[sd]?|cause[sd]?|"
    r"show[sn]?|demonstrate[sd]?|prove[sd]?|conclude[sd]?|finds?|found|"
    r"achieve[sd]?|outperform(?:s|ed)?|prevent(?:s|ed)?|mitigate[sd]?)\b"
    r"|(?:\u964d\u4f4e|\u63d0\u9ad8|\u63d0\u5347|\u589e\u52a0|\u5bfc\u81f4|\u56e0\u6b64|\u56e0\u4e3a|"
    r"\u8868\u660e|\u8bc1\u660e|\u7ed3\u8bba|\u7ed3\u679c|\u53d1\u73b0|\u8fbe\u5230|\u4f18\u4e8e|\u9632\u6b62|\u7f13\u89e3)",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%?(?![A-Za-z0-9])")
IDENTITY_CUE_RE = re.compile(
    r"\b(?:paper|article|title|named|called|exists?|arxiv|record)\b"
    r"|(?:\u8bba\u6587|\u6587\u7ae0|\u9898\u540d|\u6807\u9898|\u540d\u79f0|\u540d\u4e3a|\u5b58\u5728|\u6536\u5f55|\u8bb0\u5f55)",
    flags=re.IGNORECASE,
)
URL_RE = re.compile(r"https://[^\s<>\]\[)]+", flags=re.IGNORECASE)
GITHUB_REPOSITORY_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?![A-Za-z0-9_.-])"
)
ARXIV_ID_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5}(?:v\d+)?)(?!\d)", flags=re.IGNORECASE)
COURSE_INTERNAL_SOURCE_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(shared(?:\.[A-Za-z0-9_-]+)+)(?![A-Za-z0-9_.-])",
    flags=re.IGNORECASE,
)
NONFACTUAL_LEAD_RE = re.compile(
    r"^(?:based on (?:the )?(?:available )?evidence|according to (?:the )?evidence|"
    r"the following|summary|comparison|\u6839\u636e(?:\u5df2\u6709|\u63d0\u4f9b\u7684)?\u8bc1\u636e|"
    r"\u4ee5\u4e0b|\u7efc\u5408(?:\u6bd4\u8f83|\u7ed3\u679c)|\u603b\u7ed3)",
    flags=re.IGNORECASE,
)
GROUNDED_RENDER_RE = re.compile(
    r"^\s*(?:[-*]\s+|\d+[.)]\s+)?"
    r"(?P<claim>.+?)"
    r"\s+\|\s*evidence quote:\s*"
    r"(?P<quote>.*?)"
    r"\s+\[(?P<source_id>S\d+)\]\s*$",
    flags=re.IGNORECASE,
)
MULTI_SOURCE_TEACHING_BACKGROUND_PROFILE = "multi_source_teaching_background_v1"
_MULTI_SOURCE_PARAGRAPH_LABELS = {
    "paper": "论文教学背景",
    "vulnerability": "NVD漏洞教学背景",
    "project": "GitHub项目教学背景",
}
_MULTI_SOURCE_PARAGRAPH_ORDER = ("paper", "vulnerability", "project")


@dataclass(frozen=True)
class CitationCheck:
    ok: bool
    answer: str
    error_code: str | None = None
    error_message: str | None = None
    failure_detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class _RenderedGroundedUnit:
    claim_text: str
    evidence_quote: str
    source_id: str


class CitationGuard:
    def enforce(
        self,
        answer: str,
        evidence: list[EvidenceItem],
        *,
        question: str = "",
        required_source_types: set[str] | list[str] | tuple[str, ...] | None = None,
        verified_spans: Mapping[str, Any] | None = None,
        verified_bindings: Sequence[Any] | None = None,
        verified_presentation_profile: str | None = None,
        verified_practice_bindings: Sequence[Any] | None = None,
        verified_practice_context: str | None = None,
    ) -> CitationCheck:
        if not evidence and verified_practice_bindings is not None:
            return _practice_grounding_failure(
                "PRACTICE_GROUNDING_BINDING_MISMATCH",
                "Practice renderer cannot be verified without request-scoped evidence.",
            )
        if not evidence:
            return CitationCheck(ok=True, answer=answer)

        if _is_refusal_answer(answer):
            return CitationCheck(
                ok=False,
                answer="\u5f53\u524d\u8bc1\u636e\u4e0d\u8db3\uff0c\u65e0\u6cd5\u7ed9\u51fa\u53ef\u9760\u56de\u7b54\u3002",
                error_code="REFUSAL_ANSWER",
                error_message="The generated answer explicitly declined the requested conclusion.",
            )

        allowed = {item.source_id for item in evidence}
        cited = set(CITATION_RE.findall(answer))
        forged = cited - allowed
        if forged:
            return CitationCheck(
                ok=False,
                answer="\u5f53\u524d\u56de\u7b54\u5305\u542b\u65e0\u6cd5\u5b9a\u4f4d\u7684\u5f15\u7528\uff0c\u5df2\u62d2\u7edd\u8f93\u51fa\u3002",
                error_code="FORGED_CITATION",
                error_message=f"Citation does not exist: {sorted(forged)[0]}",
            )

        if not cited:
            return CitationCheck(
                ok=False,
                answer="\u5f53\u524d\u8bc1\u636e\u4e0d\u8db3\u4ee5\u751f\u6210\u5e26\u53ef\u5b9a\u4f4d\u5f15\u7528\u7684\u56de\u7b54\uff0c\u5df2\u62d2\u7b54\u3002",
                error_code="MISSING_CITATION",
                error_message="The answer is missing a locatable citation.",
            )

        by_id = {item.source_id: item for item in evidence}
        cited_types = {
            _normalized_source_type(by_id[source_id].resource_type)
            for source_id in cited
            if source_id in by_id
        }
        required_types = {
            _normalized_source_type(value)
            for value in (required_source_types or ())
            if str(value).strip()
        }
        strict_claim_units = (
            len(cited_types - {"course"}) > 1
            or len(required_types - {"course"}) > 1
        )

        grounded_renderer_mode = (
            verified_spans is not None
            or verified_bindings is not None
            or verified_presentation_profile is not None
            or (strict_claim_units and _contains_grounded_renderer_metadata(answer))
        )

        if verified_practice_bindings is not None:
            failure = _validate_practice_renderer_units(
                question=question,
                answer=answer,
                by_id=by_id,
                verified_spans=verified_spans,
                verified_bindings=verified_practice_bindings,
                verified_context=verified_practice_context,
            )
            if failure:
                return failure
        elif grounded_renderer_mode:
            failure = _validate_grounded_renderer_units(
                question=question,
                answer=answer,
                by_id=by_id,
                verified_spans=verified_spans,
                verified_bindings=verified_bindings,
                presentation_profile=verified_presentation_profile,
            )
            if failure:
                return failure
        elif strict_claim_units:
            for claim in _claim_units(answer):
                claim_citations = list(dict.fromkeys(CITATION_RE.findall(claim)))
                factual = _is_factual_claim(claim)
                if factual and not claim_citations:
                    return CitationCheck(
                        ok=False,
                        answer="\u5f53\u524d\u56de\u7b54\u5305\u542b\u672a\u6807\u6ce8\u6765\u6e90\u7684\u4e8b\u5b9e\uff0c\u5df2\u62d2\u7b54\u3002",
                        error_code="MISSING_CLAIM_CITATION",
                        error_message="A factual answer unit has no adjacent citation.",
                    )
                if factual and len(claim_citations) > 1:
                    return CitationCheck(
                        ok=False,
                        answer="\u5f53\u524d\u56de\u7b54\u4e2d\u7684\u590d\u5408\u4e8b\u5b9e\u65e0\u6cd5\u4e0e\u5f15\u7528\u4e00\u4e00\u5bf9\u5e94\uff0c\u5df2\u62d2\u7b54\u3002",
                        error_code="AMBIGUOUS_CITATION",
                        error_message="A factual answer unit contains multiple ambiguous citations.",
                    )
                for source_id in claim_citations:
                    failure = _validate_citation(question, claim, by_id[source_id])
                    if failure:
                        return failure
        else:
            # Preserve the original citation window for single-source-category
            # answers. Strict fact-to-source binding is required for mixed
            # external evidence, where a shared window would permit misbinding.
            for source_id in cited:
                claim = _claim_for_citation(answer, source_id)
                failure = _validate_citation(question, claim, by_id[source_id])
                if failure:
                    return failure

        missing_types = required_types - cited_types
        if missing_types:
            return CitationCheck(
                ok=False,
                answer="\u6700\u7ec8\u56de\u7b54\u672a\u8986\u76d6\u6240\u6709\u5fc5\u9700\u8bc1\u636e\u6765\u6e90\uff0c\u5df2\u62d2\u7b54\u3002",
                error_code="INCOMPLETE_CITATION_COVERAGE",
                error_message=f"Missing required citation types: {sorted(missing_types)}",
            )
        return CitationCheck(ok=True, answer=answer)


_PRACTICE_FIELD_ORDER = (
    "research_goal",
    "method_design",
    "tools",
    "experiment_steps",
    "result_analysis",
    "innovation_points",
    "risks",
)
_PRACTICE_FIELD_LABELS = {
    "research_goal": "研究目标",
    "method_design": "方法设计",
    "tools": "工具选择",
    "experiment_steps": "实验步骤",
    "result_analysis": "结果分析",
    "innovation_points": "创新点",
    "risks": "风险与注意事项",
}


def _validate_practice_renderer_units(
    *,
    question: str,
    answer: str,
    by_id: dict[str, EvidenceItem],
    verified_spans: Mapping[str, Any] | None,
    verified_bindings: Sequence[Any],
    verified_context: str | None = None,
) -> CitationCheck | None:
    if verified_spans is None or not verified_bindings:
        return _practice_grounding_failure(
            "PRACTICE_GROUNDING_BINDING_MISMATCH",
            "Practice renderer requires request-scoped spans and bindings.",
        )

    grouped: dict[str, list[tuple[Any, Any]]] = {field: [] for field in _PRACTICE_FIELD_ORDER}
    previous_field_index = -1
    seen_claims: set[str] = set()
    seen_bindings: set[tuple[str, str, str]] = set()
    for binding in verified_bindings:
        practice_field = _binding_value(binding, "practice_field")
        ordinal = _binding_int(binding, "ordinal")
        source_id = _binding_value(binding, "source_id")
        span_id = _binding_value(binding, "evidence_span_id")
        claim_text = _normalize_space(_binding_value(binding, "claim_text"))
        detail = _practice_failure_detail(
            error_code="PRACTICE_GROUNDING_BINDING_MISMATCH",
            practice_field=practice_field,
            ordinal=ordinal,
            source_id=source_id,
            span_id=span_id,
            claim_text=claim_text,
        )
        if practice_field not in grouped:
            return _practice_grounding_failure(
                "PRACTICE_GROUNDING_BINDING_MISMATCH",
                "Practice binding contains an unknown field.",
                detail,
            )
        field_index = _PRACTICE_FIELD_ORDER.index(practice_field)
        if field_index < previous_field_index or ordinal != len(grouped[practice_field]) + 1:
            return _practice_grounding_failure(
                "PRACTICE_GROUNDING_BINDING_MISMATCH",
                "Practice bindings are not in canonical field and ordinal order.",
                detail,
            )
        previous_field_index = field_index
        if not source_id or not span_id or not claim_text:
            return _practice_grounding_failure(
                "PRACTICE_GROUNDING_BINDING_MISMATCH",
                "Practice binding is missing a required value.",
                detail,
            )
        if CITATION_RE.search(claim_text) or re.search(r"\|\s*evidence\s+quote\s*:", claim_text, re.I):
            return _practice_grounding_failure(
                "PRACTICE_GROUNDING_RENDER_FORMAT",
                "Practice claim contains reserved renderer syntax.",
                {**detail, "error_code": "PRACTICE_GROUNDING_RENDER_FORMAT"},
            )
        normalized_claim = claim_text.casefold()
        binding_key = (source_id, span_id, normalized_claim)
        if normalized_claim in seen_claims or binding_key in seen_bindings:
            return _practice_grounding_failure(
                "PRACTICE_GROUNDING_BINDING_MISMATCH",
                "Practice bindings contain a duplicate claim or binding.",
                detail,
            )
        item = by_id.get(source_id)
        span = verified_spans.get(span_id)
        if item is None or span is None:
            return _practice_grounding_failure(
                "PRACTICE_GROUNDING_BINDING_MISMATCH",
                "Practice binding references an unknown source or span.",
                detail,
            )
        if _binding_value(span, "source_id") != source_id:
            return _practice_grounding_failure(
                "PRACTICE_GROUNDING_BINDING_MISMATCH",
                "Practice span does not belong to the bound source.",
                detail,
            )
        exact_text = _normalize_space(_binding_value(span, "exact_text"))
        binding_quote = _normalize_space(_binding_value(binding, "evidence_quote"))
        if not exact_text or binding_quote != exact_text:
            return _practice_grounding_failure(
                "PRACTICE_GROUNDING_BINDING_MISMATCH",
                "Practice quote does not match the request-scoped span.",
                detail,
            )
        failure = _validate_grounded_citation(question, claim_text, item, span)
        if failure:
            return CitationCheck(
                ok=False,
                answer=failure.answer,
                error_code=failure.error_code,
                error_message=failure.error_message,
                failure_detail={**detail, "error_code": failure.error_code or detail["error_code"]},
            )
        seen_claims.add(normalized_claim)
        seen_bindings.add(binding_key)
        grouped[practice_field].append((binding, span))

    if any(not grouped[field] for field in _PRACTICE_FIELD_ORDER):
        missing_field = next(field for field in _PRACTICE_FIELD_ORDER if not grouped[field])
        return _practice_grounding_failure(
            "PRACTICE_GROUNDING_BINDING_MISMATCH",
            "Practice bindings do not cover every required field.",
            _practice_failure_detail(
                error_code="PRACTICE_GROUNDING_BINDING_MISMATCH",
                practice_field=missing_field,
                ordinal=0,
                source_id="",
                span_id="",
                claim_text="",
            ),
        )

    expected = _render_verified_practice(grouped)
    if verified_context:
        expected = f"{expected}\n\n{verified_context}"
    if str(answer or "").replace("\r\n", "\n").strip() != expected:
        first = verified_bindings[0]
        return _practice_grounding_failure(
            "PRACTICE_GROUNDING_BINDING_MISMATCH",
            "Practice renderer output differs from the canonical bound rendering.",
            _practice_failure_detail(
                error_code="PRACTICE_GROUNDING_BINDING_MISMATCH",
                practice_field=_binding_value(first, "practice_field"),
                ordinal=_binding_int(first, "ordinal"),
                source_id=_binding_value(first, "source_id"),
                span_id=_binding_value(first, "evidence_span_id"),
                claim_text=_binding_value(first, "claim_text"),
            ),
        )
    return None


def _render_verified_practice(grouped: dict[str, list[tuple[Any, Any]]]) -> str:
    sections: list[str] = []
    for field in _PRACTICE_FIELD_ORDER:
        lines = [f"## {_PRACTICE_FIELD_LABELS[field]}"]
        for binding, span in grouped[field]:
            ordinal = _binding_int(binding, "ordinal")
            prefix = f"{ordinal}." if field == "experiment_steps" else "-"
            lines.append(
                f"{prefix} {_normalize_space(_binding_value(binding, 'claim_text'))} "
                f"| evidence quote: {_normalize_space(_binding_value(span, 'exact_text'))} "
                f"[{_binding_value(binding, 'source_id')}]"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections).strip()


def _binding_int(binding: Any, field: str) -> int:
    value = binding.get(field) if isinstance(binding, Mapping) else getattr(binding, field, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _practice_failure_detail(
    *,
    error_code: str,
    practice_field: str,
    ordinal: int,
    source_id: str,
    span_id: str,
    claim_text: str,
) -> dict[str, Any]:
    normalized = _normalize_space(claim_text).casefold()
    return {
        "error_code": error_code,
        "practice_field": practice_field if practice_field in _PRACTICE_FIELD_ORDER else "",
        "ordinal": max(0, int(ordinal or 0)),
        "source_id": source_id,
        "span_id": span_id,
        "claim_sha256": hashlib.sha256(str(claim_text or "").encode("utf-8")).hexdigest(),
        "normalized_claim_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "claim_length": len(str(claim_text or "")),
        "normalization_flags": {
            "space_normalized": str(claim_text or "") != _normalize_space(claim_text),
            "casefolded": _normalize_space(claim_text) != normalized,
        },
    }


def _practice_grounding_failure(
    error_code: str,
    error_message: str,
    failure_detail: dict[str, Any] | None = None,
) -> CitationCheck:
    return CitationCheck(
        ok=False,
        answer="当前实践指导的逐项证据绑定未通过校验，已拒答。",
        error_code=error_code,
        error_message=error_message,
        failure_detail=failure_detail,
    )


def _validate_grounded_renderer_units(
    *,
    question: str,
    answer: str,
    by_id: dict[str, EvidenceItem],
    verified_spans: Mapping[str, Any] | None,
    verified_bindings: Sequence[Any] | None,
    presentation_profile: str | None,
) -> CitationCheck | None:
    # The strict grounded renderer emits exactly one bound unit per physical
    # line.  Reusing the free-text sentence splitter here is asymmetric with
    # that renderer and can split a valid line at author initials such as
    # ``Aidan N. Gomez``.  Keep this path line-oriented; every resulting line
    # is still parsed and checked against the request-scoped binding below.
    if verified_spans is None or verified_bindings is None:
        return _grounding_failure(
            "GROUNDING_BINDING_REQUIRED",
            "Grounded renderer output requires request-scoped span bindings.",
        )
    profiled = presentation_profile is not None
    parsed: list[_RenderedGroundedUnit] = []
    if not profiled:
        units = [line.strip() for line in str(answer or "").splitlines() if line.strip()]
        for unit in units:
            rendered = _parse_grounded_renderer_unit(unit)
            if rendered is None:
                return _grounding_failure(
                    "GROUNDING_RENDER_FORMAT",
                    "Grounded renderer output contains an unbound or malformed fact unit.",
                )
            if CITATION_RE.search(rendered.claim_text) or CITATION_RE.search(rendered.evidence_quote):
                return _grounding_failure(
                    "GROUNDING_RENDER_FORMAT",
                    "Grounded renderer claim or quote contains an embedded citation marker.",
                )
            if not rendered.evidence_quote:
                return _grounding_failure(
                    "GROUNDING_RENDER_FORMAT",
                    "Grounded renderer output contains an empty evidence quote.",
                )
            parsed.append(rendered)

    if not profiled and len(parsed) != len(verified_bindings):
        return _grounding_failure(
            "GROUNDING_BINDING_COUNT_MISMATCH",
            "Grounded renderer unit count does not match verified bindings.",
        )

    canonical_units: list[_RenderedGroundedUnit] = []
    for ordinal, binding in enumerate(verified_bindings, 1):
        binding_source = _binding_value(binding, "source_id")
        binding_claim = _binding_value(binding, "claim_text")
        binding_quote = _binding_value(binding, "evidence_quote")
        span_id = _binding_value(binding, "evidence_span_id")
        rendered = (
            _RenderedGroundedUnit(
                claim_text=_normalize_space(binding_claim),
                evidence_quote=_normalize_space(binding_quote),
                source_id=binding_source,
            )
            if profiled
            else parsed[ordinal - 1]
        )
        failure_detail = _grounded_failure_detail(
            error_code="GROUNDING_BINDING_REQUIRED",
            ordinal=ordinal,
            source_id=binding_source or rendered.source_id,
            span_id=span_id,
            claim_text=binding_claim or rendered.claim_text,
        )
        if not binding_source or not binding_claim or not binding_quote or not span_id:
            return _grounding_failure(
                "GROUNDING_BINDING_REQUIRED",
                "Grounded citation binding is missing source, claim, or span.",
                failure_detail,
            )
        if (
            binding_source != rendered.source_id
            or _normalize_space(binding_claim).casefold()
            != _normalize_space(rendered.claim_text).casefold()
        ):
            return _grounding_failure(
                "GROUNDING_BINDING_MISMATCH",
                "Grounded renderer unit does not match the verified binding order.",
                {**failure_detail, "error_code": "GROUNDING_BINDING_MISMATCH"},
            )
        span = verified_spans.get(span_id)
        if span is None:
            return _grounding_failure(
                "GROUNDING_UNKNOWN_SPAN",
                f"Grounded citation binding references an unknown span: {span_id}",
                {**failure_detail, "error_code": "GROUNDING_UNKNOWN_SPAN"},
            )
        span_source = _binding_value(span, "source_id")
        if span_source != rendered.source_id:
            return _grounding_failure(
                "GROUNDING_SPAN_SOURCE_MISMATCH",
                f"Grounded span {span_id} does not belong to source {rendered.source_id}.",
                {**failure_detail, "error_code": "GROUNDING_SPAN_SOURCE_MISMATCH"},
            )
        if rendered.source_id not in by_id:
            return _grounding_failure(
                "GROUNDING_UNKNOWN_SOURCE",
                f"Grounded citation binding references an unknown source: {rendered.source_id}",
                {**failure_detail, "error_code": "GROUNDING_UNKNOWN_SOURCE"},
            )
        exact_text = _binding_value(span, "exact_text")
        if not exact_text or _normalize_rendered_quote(exact_text) != _normalize_rendered_quote(rendered.evidence_quote):
            return _grounding_failure(
                "GROUNDING_QUOTE_MISMATCH",
                f"Grounded renderer quote does not match span {span_id}.",
                {**failure_detail, "error_code": "GROUNDING_QUOTE_MISMATCH"},
            )
        failure = _validate_grounded_citation(
            question,
            rendered.claim_text,
            by_id[rendered.source_id],
            span,
        )
        if failure:
            return CitationCheck(
                ok=failure.ok,
                answer=failure.answer,
                error_code=failure.error_code,
                error_message=failure.error_message,
                failure_detail=_grounded_failure_detail(
                    error_code=failure.error_code or "GROUNDING_FAILED",
                    ordinal=ordinal,
                    source_id=rendered.source_id,
                    span_id=span_id,
                    claim_text=rendered.claim_text,
                ),
            )
        canonical_units.append(rendered)
    if profiled:
        try:
            expected = render_grounded_presentation(
                canonical_units,
                list(by_id.values()),
                presentation_profile=presentation_profile,
            )
        except ValueError:
            return _grounding_failure(
                "GROUNDING_RENDER_FORMAT",
                "Grounded presentation profile cannot be reconstructed from trusted bindings.",
            )
        if str(answer or "") != expected:
            return _grounding_failure(
                "GROUNDING_RENDER_FORMAT",
                "Grounded presentation does not match the canonical trusted rendering.",
            )
    return None


def render_grounded_presentation(
    units: Sequence[Any],
    evidence: Sequence[EvidenceItem],
    *,
    presentation_profile: str | None = None,
) -> str:
    """Render only server-verified units into one canonical visible shape."""
    if presentation_profile is None:
        return "\n".join(
            f"- {_binding_value(unit, 'claim_text')} | evidence quote: "
            f"{_binding_value(unit, 'evidence_quote')} [{_binding_value(unit, 'source_id')}]"
            for unit in units
        )
    if presentation_profile != MULTI_SOURCE_TEACHING_BACKGROUND_PROFILE:
        raise ValueError("Unknown grounded presentation profile")

    source_types = {
        item.source_id: _normalized_source_type(item.resource_type)
        for item in evidence
    }
    grouped: dict[str, list[str]] = {key: [] for key in _MULTI_SOURCE_PARAGRAPH_ORDER}
    for unit in units:
        source_id = _binding_value(unit, "source_id")
        claim_text = _binding_value(unit, "claim_text")
        source_type = source_types.get(source_id, "")
        if source_type not in grouped or not claim_text:
            raise ValueError("Presentation unit is not bound to an expected source type")
        grouped[source_type].append(f"{claim_text} [{source_id}]")
    if any(not grouped[source_type] for source_type in _MULTI_SOURCE_PARAGRAPH_ORDER):
        raise ValueError("Presentation profile requires every source type")
    return "\n".join(
        f"{_MULTI_SOURCE_PARAGRAPH_LABELS[source_type]}：{'；'.join(grouped[source_type])}"
        for source_type in _MULTI_SOURCE_PARAGRAPH_ORDER
    )


def _parse_grounded_renderer_unit(unit: str) -> _RenderedGroundedUnit | None:
    match = GROUNDED_RENDER_RE.match(str(unit or "").strip())
    if not match:
        return None
    claim_text = _normalize_space(match.group("claim"))
    evidence_quote = _normalize_space(match.group("quote"))
    source_id = match.group("source_id")
    if not claim_text or not evidence_quote:
        return None
    return _RenderedGroundedUnit(
        claim_text=claim_text,
        evidence_quote=evidence_quote,
        source_id=source_id,
    )


def _contains_grounded_renderer_metadata(answer: str) -> bool:
    return bool(re.search(r"\|\s*evidence quote\s*:", str(answer or ""), flags=re.IGNORECASE))


def _binding_value(binding: Any, field: str) -> str:
    if isinstance(binding, Mapping):
        return str(binding.get(field) or "").strip()
    return str(getattr(binding, field, "") or "").strip()


def _normalize_rendered_quote(value: str) -> str:
    text = _normalize_space(value)
    return text.strip(" \t\r\n.,;:!?。，；：！？")


def _grounded_failure_detail(
    *,
    error_code: str,
    ordinal: int,
    source_id: str,
    span_id: str,
    claim_text: str,
) -> dict[str, Any]:
    normalized = _normalize_space(claim_text).casefold()
    return {
        "error_code": error_code,
        "ordinal": max(0, int(ordinal or 0)),
        "source_id": source_id if re.fullmatch(r"S\d+", source_id or "") else "",
        "span_id": span_id if re.fullmatch(r"S\d+:(?:text|title|external_id):\d+", span_id or "") else "",
        "claim_sha256": hashlib.sha256(str(claim_text or "").encode("utf-8")).hexdigest(),
        "normalized_claim_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "claim_length": len(str(claim_text or "")),
    }


def _grounding_failure(
    error_code: str,
    error_message: str,
    failure_detail: dict[str, Any] | None = None,
) -> CitationCheck:
    return CitationCheck(
        ok=False,
        answer="\u5f53\u524d\u56de\u7b54\u7684\u8bc1\u636e\u7ed1\u5b9a\u672a\u901a\u8fc7\u6821\u9a8c\uff0c\u5df2\u62d2\u7b54\u3002",
        error_code=error_code,
        error_message=error_message,
        failure_detail=failure_detail,
    )


def _validate_citation(
    question: str,
    claim: str,
    item: EvidenceItem,
) -> CitationCheck | None:
    if not _has_location(item):
        return CitationCheck(
            ok=False,
            answer="\u5f53\u524d\u56de\u7b54\u7684\u5f15\u7528\u7f3a\u5c11\u53ef\u5b9a\u4f4d\u6765\u6e90\uff0c\u5df2\u62d2\u7edd\u8f93\u51fa\u3002",
            error_code="UNLOCATABLE_CITATION",
            error_message=f"Citation lacks location fields: {item.source_id}",
        )
    if question and not _is_relevant_to_question(question, item, claim=claim):
        return CitationCheck(
            ok=False,
            answer="\u5f53\u524d\u8bc1\u636e\u4e0e\u95ee\u9898\u4e0d\u5339\u914d\uff0c\u5df2\u62d2\u7b54\u3002",
            error_code="IRRELEVANT_CITATION",
            error_message=f"Citation is not relevant to the question: {item.source_id}",
        )
    identity_mismatch = _course_internal_source_id_mismatch(claim, item)
    if identity_mismatch:
        return CitationCheck(
            ok=False,
            answer="当前回答中的教材内部来源身份与引用证据不一致，已拒答。",
            error_code="COURSE_SOURCE_ID_MISMATCH",
            error_message=f"Course source identity does not match bound evidence: {item.source_id}",
        )
    if not _supports_answer(claim, item):
        return CitationCheck(
            ok=False,
            answer="\u5f53\u524d\u56de\u7b54\u5305\u542b\u8bc1\u636e\u7247\u6bb5\u65e0\u6cd5\u652f\u6301\u7684\u5173\u952e\u4e8b\u5b9e\uff0c\u5df2\u62d2\u7b54\u3002",
            error_code="UNSUPPORTED_CITATION",
            error_message=f"Citation does not support the answer: {item.source_id}",
        )
    return None


def _course_internal_source_id_mismatch(claim: str, item: EvidenceItem) -> bool:
    """Reject model-authored course catalog IDs that are not bound to this evidence."""

    if _normalized_source_type(item.resource_type) != "course":
        return False
    claimed = {
        value.casefold()
        for value in COURSE_INTERNAL_SOURCE_ID_RE.findall(CITATION_RE.sub(" ", claim))
    }
    if not claimed:
        return False
    canonical = {
        value.casefold()
        for value in COURSE_INTERNAL_SOURCE_ID_RE.findall(str(item.text or ""))
    }
    return not claimed.issubset(canonical)


def _validate_grounded_citation(
    question: str,
    claim: str,
    item: EvidenceItem,
    span: Any,
) -> CitationCheck | None:
    if not _has_location(item):
        return CitationCheck(
            ok=False,
            answer="\u5f53\u524d\u56de\u7b54\u7684\u5f15\u7528\u7f3a\u5c11\u53ef\u5b9a\u4f4d\u6765\u6e90\uff0c\u5df2\u62d2\u7edd\u8f93\u51fa\u3002",
            error_code="UNLOCATABLE_CITATION",
            error_message=f"Citation lacks location fields: {item.source_id}",
        )
    if question and not _is_relevant_to_question(question, item, claim=claim):
        return CitationCheck(
            ok=False,
            answer="\u5f53\u524d\u8bc1\u636e\u4e0e\u95ee\u9898\u4e0d\u5339\u914d\uff0c\u5df2\u62d2\u7b54\u3002",
            error_code="IRRELEVANT_CITATION",
            error_message=f"Citation is not relevant to the question: {item.source_id}",
        )
    if not _supports_grounded_span_claim(claim, item, span):
        return CitationCheck(
            ok=False,
            answer="\u5f53\u524d\u56de\u7b54\u5305\u542b\u8bc1\u636e\u7247\u6bb5\u65e0\u6cd5\u652f\u6301\u7684\u5173\u952e\u4e8b\u5b9e\uff0c\u5df2\u62d2\u7b54\u3002",
            error_code="UNSUPPORTED_CITATION",
            error_message=f"Citation does not support the selected evidence span: {item.source_id}",
        )
    return None


def _has_location(item: EvidenceItem) -> bool:
    local_location = bool(
        item.source and (item.page is not None or item.section or item.chunk_id is not None)
    )
    external_location = bool(
        item.trusted
        and item.source_url.startswith("https://")
        and (item.external_id or item.title)
        and item.content_hash
    )
    return local_location or external_location


def _supports_answer(answer: str, item: EvidenceItem) -> bool:
    plain_answer = CITATION_RE.sub(" ", answer)
    if _is_external_identity_fact(plain_answer, item):
        return True
    verified_title = _verified_paper_title(item)
    answer_for_text = plain_answer
    if verified_title:
        normalized_answer = _normalize_space(plain_answer)
        normalized_title = _normalize_space(verified_title)
        if normalized_title and normalized_title.casefold() in normalized_answer.casefold():
            title_start = normalized_answer.casefold().find(normalized_title.casefold())
            residual = (
                normalized_answer[:title_start]
                + " "
                + normalized_answer[title_start + len(normalized_title) :]
            )
            if _is_paper_identity_fact(residual, item):
                return True
            answer_for_text = residual

    if _has_unsupported_specifics(answer_for_text, item.text):
        return False
    answer_tokens = _support_tokens(answer_for_text)
    evidence_tokens = _support_tokens(item.text)
    if not answer_tokens:
        return True
    if not evidence_tokens:
        return False
    overlap = answer_tokens & evidence_tokens
    return len(overlap) >= min(2, len(answer_tokens))


def _supports_grounded_span_claim(answer: str, item: EvidenceItem, span: Any) -> bool:
    span_text = _binding_value(span, "exact_text")
    if not span_text:
        return False

    plain_answer = _normalize_space(CITATION_RE.sub(" ", answer))
    if not plain_answer:
        return True

    span_field = _binding_value(span, "field")
    answer_for_span = _remove_grounded_canonical_identity_text(plain_answer, item)
    if _has_unsupported_specifics_for_span(answer_for_span, span_text, item):
        return False

    if span_field in {"title", "external_id"}:
        return _supports_grounded_identity_span_claim(
            plain_answer,
            item,
            field=span_field,
            span_text=span_text,
        )

    answer_tokens = _strict_grounded_tokens(answer_for_span, item)
    if not answer_tokens:
        return True

    span_tokens = _grounded_support_tokens(span_text)
    if not span_tokens:
        return False
    return answer_tokens.issubset(span_tokens)


def _supports_grounded_identity_span_claim(
    claim: str,
    item: EvidenceItem,
    *,
    field: str,
    span_text: str,
) -> bool:
    value = _normalize_grounded_identity_value(span_text)
    if not value:
        return False
    source_type = _normalized_source_type(item.resource_type)
    if field == "external_id" and not _external_id_span_is_compatible(item, span_text, source_type):
        return False
    normalized_claim = _normalize_grounded_identity_sentence(claim)
    if normalized_claim == _normalize_grounded_identity_sentence(span_text):
        return True

    return normalized_claim in {
        _normalize_grounded_identity_sentence(template)
        for template in _grounded_identity_templates(
            source_type=source_type,
            field=field,
            value=value,
        )
    } or _supports_grounded_vulnerability_title_identity(
        normalized_claim,
        source_type=source_type,
        field=field,
        value=value,
    )


def _supports_grounded_vulnerability_title_identity(
    normalized_claim: str,
    *,
    source_type: str,
    field: str,
    value: str,
) -> bool:
    if source_type != "vulnerability" or field != "title" or not _cve_from_external_id(value):
        return False
    normalized_value = _normalize_grounded_identity_sentence(value)
    if normalized_value not in normalized_claim:
        return False
    residual = _normalize_space(
        re.sub(re.escape(normalized_value), " ", normalized_claim, flags=re.IGNORECASE)
    )
    tokens = _grounded_support_tokens(residual)
    return tokens.issubset({"nvd", "record", "vulnerability", "cve"})


def _external_id_span_is_compatible(item: EvidenceItem, span_text: str, source_type: str) -> bool:
    value = _normalize_grounded_identity_value(span_text)
    if source_type == "paper":
        canonical = _doi_from_external_id(item.external_id)
        return bool(_arxiv_id_from_external_id(value)) or bool(
            canonical and _doi_from_external_id(value) == canonical
        )
    if source_type == "vulnerability":
        return bool(_cve_from_external_id(value))
    if source_type == "project":
        return _is_repository_literal(value)
    return bool(value)


def _grounded_identity_templates(*, source_type: str, field: str, value: str) -> tuple[str, ...]:
    variants = _grounded_identity_value_variants(value)
    templates: list[str] = []
    for identity in variants:
        if field == "title":
            if source_type == "paper":
                templates.extend(
                    [
                        f"The paper is titled {identity}",
                        f'The paper is titled "{value}"',
                        f"The paper title is {identity}",
                        f"The paper is named {identity}",
                        f"\u8bba\u6587\u6807\u9898\u4e3a{identity}",
                        f"\u8bba\u6587\u6807\u9898\u662f{identity}",
                        f"\u8bba\u6587\u9898\u540d\u4e3a{identity}",
                    ]
                )
            elif source_type == "vulnerability":
                templates.extend(
                    [
                        f"The NVD record is titled {identity}",
                        f"The NVD record is named {identity}",
                        f"The vulnerability title is {identity}",
                        f"{identity} is an NVD record",
                        f"{identity} is the NVD record",
                        f"\u6f0f\u6d1e\u6807\u9898\u4e3a{identity}",
                        f"NVD\u8bb0\u5f55\u6807\u9898\u4e3a{identity}",
                    ]
                )
            elif source_type == "project":
                templates.extend(
                    [
                        f"The GitHub repository is {identity}",
                        f"The GitHub project is named {identity}",
                        f"The GitHub repository is named {identity}",
                        f"The project title is {identity}",
                        f"The project name is {identity}",
                        f"\u5f00\u6e90\u9879\u76ee\u540d\u79f0\u4e3a{identity}",
                        f"GitHub\u9879\u76ee\u540d\u79f0\u4e3a{identity}",
                    ]
                )
            elif source_type == "standard":
                templates.extend(
                    [
                        f"The standard is titled {identity}",
                        f"The standard title is {identity}",
                        f"标准标题为{identity}",
                    ]
                )
        elif field == "external_id":
            if source_type == "paper":
                if _doi_from_external_id(value):
                    templates.extend(
                        [
                            f"The paper DOI is {identity}",
                            f"The paper DOI identifier is {identity}",
                            f"The paper identifier is {identity}",
                            f"论文DOI为{identity}",
                        ]
                    )
                else:
                    templates.extend(
                        [
                            f"The paper arXiv ID is {identity}",
                            f"The paper arXiv identifier is {identity}",
                            f"The paper is identified by arXiv ID {identity}",
                            f"The paper identifier is {identity}",
                            f"{identity} is the paper arXiv ID",
                            f"\u8bba\u6587arXiv\u6807\u8bc6\u4e3a{identity}",
                            f"\u8bba\u6587\u7f16\u53f7\u4e3a{identity}",
                        ]
                    )
            elif source_type == "vulnerability":
                templates.extend(
                    [
                        f"The NVD record is identified as {identity}",
                        f"The NVD identifier is {identity}",
                        f"The CVE identifier is {identity}",
                        f"The vulnerability identifier is {identity}",
                        f"{identity} is the NVD identifier",
                        f"\u6f0f\u6d1eCVE\u7f16\u53f7\u4e3a{identity}",
                        f"NVD\u8bb0\u5f55\u6807\u8bc6\u4e3a{identity}",
                    ]
                )
            elif source_type == "project":
                templates.extend(
                    [
                        f"The GitHub repository is {identity}",
                        f"The GitHub repository is named {identity}",
                        f"The GitHub project is {identity}",
                        f"The project repository is {identity}",
                        f"The project identifier is {identity}",
                        f"{identity} is the GitHub repository",
                        f"GitHub\u4ed3\u5e93\u4e3a{identity}",
                        f"\u5f00\u6e90\u9879\u76ee\u4ed3\u5e93\u4e3a{identity}",
                    ]
                )
            elif source_type == "standard":
                templates.extend(
                    [
                        f"The standard identifier is {identity}",
                        f"The standard is identified as {identity}",
                        f"标准编号为{identity}",
                    ]
                )
    return tuple(dict.fromkeys(templates))


def _grounded_identity_value_variants(value: str) -> tuple[str, ...]:
    cleaned = _normalize_grounded_identity_value(value)
    if not cleaned:
        return ()
    return (
        cleaned,
        f'"{cleaned}"',
        f"'{cleaned}'",
        f"\u201c{cleaned}\u201d",
        f"\u300a{cleaned}\u300b",
    )


def _normalize_grounded_identity_value(value: str) -> str:
    return _normalize_space(value).strip(" \t\r\n.,;:!?\u3002\uff0c\uff1b\uff1a\uff01\uff1f")


def _normalize_grounded_identity_sentence(value: str) -> str:
    normalized = _normalize_grounded_identity_value(value)
    normalized = re.sub(r"\s*([:：])\s*", r"\1", normalized)
    return normalized.casefold()


def _has_unsupported_specifics_for_span(claim: str, span_text: str, item: EvidenceItem) -> bool:
    normalized_claim = _normalize_space(claim)
    normalized_span = _normalize_space(span_text)
    claim_numbers = set(NUMBER_RE.findall(normalized_claim))
    span_numbers = set(NUMBER_RE.findall(normalized_span))
    canonical_numbers = _grounded_canonical_numbers(item)
    if claim_numbers - span_numbers - canonical_numbers:
        return True

    claim_assertions = {
        match.group(0).casefold() for match in PAPER_ASSERTION_RE.finditer(normalized_claim)
    }
    span_assertions = {
        match.group(0).casefold() for match in PAPER_ASSERTION_RE.finditer(normalized_span)
    }
    return bool(claim_assertions - span_assertions)


def _strict_grounded_tokens(claim: str, item: EvidenceItem) -> set[str]:
    generic = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "with",
        "by",
        "from",
        "as",
        "this",
        "that",
        "it",
        "its",
        "paper",
        "article",
        "title",
        "identifier",
        "id",
        "record",
        "source",
        "url",
        "official",
        "link",
        "available",
        "nvd",
        "github",
        "repository",
        "repo",
        "project",
        "vulnerability",
        "cve",
        "\u8bb0\u5f55",
        "\u6761\u76ee",
        "\u6f0f\u6d1e",
        "\u7f16\u53f7",
        "\u6807\u8bc6",
        "\u4e3a",
        "\u662f",
        "\u7684",
        "\u53ef\u7528",
        _normalized_source_type(item.resource_type),
        str(item.source or "").casefold(),
    }
    return {token for token in _grounded_support_tokens(claim) if token not in generic}


def _grounded_support_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _support_tokens(text):
        normalized = _normalize_grounded_support_token(token)
        if normalized:
            tokens.add(normalized)
    return tokens


def _normalize_grounded_support_token(token: str) -> str:
    value = str(token or "").casefold()
    if value.startswith(("http://", "https://")):
        return value.rstrip(".,;:!?\u3002\uff0c\uff1b\uff1a\uff01\uff1f")
    if re.fullmatch(r"cve-\d{4}-\d{4,8}", value, flags=re.IGNORECASE):
        return value.upper()
    if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", value, flags=re.IGNORECASE):
        return value
    if _is_repository_literal(value):
        return value
    return value.strip(".,;:!?\u3002\uff0c\uff1b\uff1a\uff01\uff1f")


def _remove_grounded_canonical_identity_text(claim: str, item: EvidenceItem) -> str:
    reduced = claim
    for value in _grounded_canonical_identity_values(item):
        if not value:
            continue
        reduced = re.sub(re.escape(value), " ", reduced, flags=re.IGNORECASE)
        if value.endswith("/"):
            reduced = re.sub(re.escape(value.rstrip("/")), " ", reduced, flags=re.IGNORECASE)
        else:
            reduced = re.sub(re.escape(value + "/"), " ", reduced, flags=re.IGNORECASE)
    return _normalize_space(reduced)


def _grounded_canonical_numbers(item: EvidenceItem) -> set[str]:
    values = " ".join(_grounded_canonical_identity_values(item))
    return set(NUMBER_RE.findall(values))


def _grounded_canonical_identity_values(item: EvidenceItem) -> tuple[str, ...]:
    if not _is_groundable_external_identity(item):
        return ()
    item_type = _normalized_source_type(item.resource_type)
    values: set[str] = set()
    source_url = str(item.source_url or "").strip()
    normalized_source_url = _normalize_source_url_identity(source_url)
    if source_url.startswith("https://"):
        values.add(source_url.rstrip("/"))
        values.add(normalized_source_url)

    if item_type == "paper":
        for value in (_arxiv_id_from_external_id(item.external_id), _arxiv_id_from_url(source_url)):
            if value:
                values.add(value)
        doi = _doi_from_external_id(item.external_id)
        if doi:
            values.add(doi)
    elif item_type == "vulnerability":
        for value in (_cve_from_external_id(item.external_id), _cve_from_nvd_url(source_url)):
            if value:
                values.add(value)
    elif item_type == "project":
        external_id = str(item.external_id or "").strip()
        if _is_repository_literal(external_id):
            values.add(external_id)
        repo_from_url = _github_repository_from_url(source_url)
        if repo_from_url:
            values.add(repo_from_url)
    return tuple(sorted(values, key=len, reverse=True))


def _is_groundable_external_identity(item: EvidenceItem) -> bool:
    return bool(
        item.trusted
        and item.source_url.startswith("https://")
        and item.content_hash
        and (item.external_id or item.source_url)
    )


def _normalize_source_url_identity(value: str) -> str:
    parsed = urlparse(str(value or "").strip().rstrip(".,;:!?)]}"))
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return _normalize_space(value)
    path = parsed.path or ""
    if path != "/":
        path = path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme.casefold()}://{parsed.netloc.casefold()}{path}{query}"


def _arxiv_id_from_external_id(value: str) -> str:
    text = str(value or "").strip().strip(".,;:!?)]}")
    return text if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", text, flags=re.IGNORECASE) else ""


def _doi_from_external_id(value: str) -> str:
    text = str(value or "").strip().strip(".,;!?)]}")
    text = re.sub(r"^doi\s*:\s*", "", text, flags=re.IGNORECASE)
    if len(text) > 255:
        return ""
    return text.casefold() if re.fullmatch(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, flags=re.IGNORECASE) else ""


def _arxiv_id_from_url(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme != "https" or parsed.netloc.casefold() not in {"arxiv.org", "www.arxiv.org"}:
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0].casefold() != "abs":
        return ""
    return _arxiv_id_from_external_id(parts[1])


def _cve_from_external_id(value: str) -> str:
    text = str(value or "").strip().strip(".,;:!?)]}").upper()
    return text if re.fullmatch(r"CVE-\d{4}-\d{4,8}", text) else ""


def _cve_from_nvd_url(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme != "https" or parsed.netloc.casefold() != "nvd.nist.gov":
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 3 or [part.casefold() for part in parts[:2]] != ["vuln", "detail"]:
        return ""
    return _cve_from_external_id(parts[2])


def _is_repository_literal(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}/[A-Za-z0-9][A-Za-z0-9_.-]{0,100}",
            str(value or "").strip(),
        )
    )


def _github_repository_from_url(value: str) -> str:
    parsed = urlparse(str(value or ""))
    if parsed.scheme != "https" or parsed.netloc.casefold() != "github.com":
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return ""
    return f"{parts[0]}/{parts[1]}"


def _is_external_identity_fact(claim: str, item: EvidenceItem) -> bool:
    if not (
        item.trusted
        and item.source_url.startswith("https://")
        and (item.external_id or item.title)
        and item.content_hash
    ):
        return False

    normalized = _normalize_space(claim)
    locators = tuple(
        sorted(
            (
                str(value).strip()
                for value in (item.title, item.external_id, item.source_url)
                if str(value or "").strip()
            ),
            key=len,
            reverse=True,
        )
    )
    if not any(locator.casefold() in normalized.casefold() for locator in locators):
        return False

    remainder = normalized
    for value in (*locators, item.source, item.resource_type):
        if value:
            remainder = re.sub(re.escape(str(value)), " ", remainder, flags=re.IGNORECASE)
    if PAPER_ASSERTION_RE.search(remainder) or NUMBER_RE.search(remainder):
        return False

    english_tokens = set(re.findall(r"[A-Za-z][A-Za-z-]*", remainder.casefold()))
    identity_tokens = {
        "a",
        "an",
        "the",
        "paper",
        "article",
        "title",
        "is",
        "named",
        "called",
        "exists",
        "exist",
        "available",
        "arxiv",
        "nvd",
        "cve",
        "github",
        "record",
        "repository",
        "repo",
        "project",
        "source",
        "official",
        "link",
        "url",
        "of",
        "at",
    }
    if english_tokens - identity_tokens:
        return False

    chinese_remainder = re.sub(r"[A-Za-z\d\s\W_]", "", remainder)
    for phrase in (
        "\u8fd9\u7bc7",
        "\u8be5",
        "\u4e00\u7bc7",
        "\u8bba\u6587",
        "\u6587\u7ae0",
        "\u9898\u540d",
        "\u6807\u9898",
        "\u540d\u79f0",
        "\u7f16\u53f7",
        "\u6807\u8bc6",
        "\u540d\u4e3a",
        "\u6761\u76ee",
        "\u8bb0\u5f55",
        "\u4ed3\u5e93",
        "\u9879\u76ee",
        "\u6f0f\u6d1e",
        "\u6765\u6e90",
        "\u5b98\u65b9",
        "\u94fe\u63a5",
        "\u5730\u5740",
        "\u5b58\u5728",
        "\u6536\u5f55",
        "\u53ef\u7528",
        "\u4e3a",
        "\u662f",
        "\u7684",
    ):
        chinese_remainder = chinese_remainder.replace(phrase, "")
    return not chinese_remainder


def _verified_paper_title(item: EvidenceItem) -> str:
    if not (
        item.trusted
        and item.resource_type == "paper"
        and item.title
        and item.source_url.startswith("https://")
        and item.external_id
        and item.content_hash
    ):
        return ""
    return str(item.title).strip()


def _is_paper_identity_fact(residual: str, item: EvidenceItem) -> bool:
    normalized = _normalize_space(residual)
    if not IDENTITY_CUE_RE.search(normalized):
        return False
    if PAPER_ASSERTION_RE.search(normalized):
        return False

    allowed_values = (item.external_id, item.source_url, item.source)
    remainder = normalized
    for value in allowed_values:
        if value:
            remainder = re.sub(re.escape(str(value)), " ", remainder, flags=re.IGNORECASE)
    if NUMBER_RE.search(remainder):
        return False

    english_tokens = set(re.findall(r"[A-Za-z][A-Za-z-]*", remainder.casefold()))
    identity_tokens = {
        "a",
        "an",
        "the",
        "paper",
        "article",
        "title",
        "is",
        "named",
        "called",
        "exists",
        "exist",
        "arxiv",
        "record",
        "of",
    }
    if english_tokens - identity_tokens:
        return False

    chinese_remainder = re.sub(r"[A-Za-z\d\s\W_]", "", remainder)
    for phrase in (
        "\u8fd9\u7bc7",
        "\u8be5",
        "\u4e00\u7bc7",
        "\u8bba\u6587",
        "\u6587\u7ae0",
        "\u9898\u540d",
        "\u6807\u9898",
        "\u540d\u79f0",
        "\u540d\u4e3a",
        "\u4e3a",
        "\u662f",
        "\u5b58\u5728",
        "\u6536\u5f55",
        "\u8bb0\u5f55",
        "\u7684",
    ):
        chinese_remainder = chinese_remainder.replace(phrase, "")
    return not chinese_remainder


def _has_unsupported_specifics(claim: str, evidence_text: str) -> bool:
    normalized_claim = _normalize_space(claim)
    normalized_evidence = _normalize_space(evidence_text)
    claim_numbers = set(NUMBER_RE.findall(normalized_claim))
    evidence_numbers = set(NUMBER_RE.findall(normalized_evidence))
    if claim_numbers - evidence_numbers:
        return True

    claim_assertions = {
        match.group(0).casefold() for match in PAPER_ASSERTION_RE.finditer(normalized_claim)
    }
    evidence_assertions = {
        match.group(0).casefold() for match in PAPER_ASSERTION_RE.finditer(normalized_evidence)
    }
    return bool(claim_assertions - evidence_assertions)


def _normalize_space(value: str) -> str:
    return " ".join(str(value or "").split())


def _is_refusal_answer(answer: str) -> bool:
    plain = CITATION_RE.sub(" ", str(answer or "")).strip()
    lead = re.split(r"(?:\n\s*\n)|(?<=[.!?\u3002\uff01\uff1f])", plain, maxsplit=1)[0]
    return bool(REFUSAL_RE.search(lead[:240]))


def _claim_for_citation(answer: str, source_id: str) -> str:
    marker = f"[{source_id}]"
    parts = re.split(
        r"(?<=[!?\u3002\uff01\uff1f])|(?<=\.)(?=\s|$)|\n+",
        answer,
    )
    matched: list[str] = []
    for index, part in enumerate(parts):
        if marker not in part:
            continue
        context = part
        if not CITATION_RE.sub(" ", part).strip() and index > 0:
            context = parts[index - 1] + part
        matched.append(context)
    return " ".join(matched) if matched else answer


def _claim_units(answer: str) -> list[str]:
    return [
        unit.strip()
        for unit in re.split(
            r"(?:\r?\n)+|(?<=[.!?\u3002\uff01\uff1f;\uff1b])\s+(?!\[S\d+\])",
            str(answer or ""),
        )
        if unit.strip()
    ]


def _is_factual_claim(claim: str) -> bool:
    plain = CITATION_RE.sub(" ", claim)
    plain = re.sub(r"^[\s>*#\-+\d.)]+", "", plain).strip()
    plain = plain.strip("*_` ")
    if not plain or plain.endswith((":", "\uff1a")):
        return False
    if NONFACTUAL_LEAD_RE.match(plain) and len(_support_tokens(plain)) <= 4:
        return False
    return bool(_support_tokens(plain) or re.search(r"[\u4e00-\u9fff]{4,}", plain))


def _is_relevant_to_question(question: str, item: EvidenceItem, *, claim: str = "") -> bool:
    evidence_text = " ".join(
        part
        for part in (
            item.text,
            item.title,
            item.source,
            item.section,
            item.external_id,
            item.resource_type,
            item.source_url,
        )
        if part
    )
    normalized_evidence = _normalized_locator_text(evidence_text)
    item_type = _normalized_source_type(item.resource_type)

    claim_identifiers = _typed_identifiers(CITATION_RE.sub(" ", claim))
    if any(
        _normalized_locator_text(identifier) not in normalized_evidence
        for identifier, _ in claim_identifiers
    ):
        return False

    question_identifiers = {
        identifier
        for identifier, identifier_type in _typed_identifiers(question)
        if identifier_type == item_type or (identifier_type == "course" and item_type == "course")
    }
    if any(_normalized_locator_text(identifier) not in normalized_evidence for identifier in question_identifiers):
        return False

    question_tokens = _support_tokens(question)
    if not question_tokens:
        return True
    evidence_tokens = _support_tokens(evidence_text)
    if question_tokens & evidence_tokens:
        return True
    return item_type in _requested_resource_types(question)


def _typed_identifiers(text: str) -> set[tuple[str, str]]:
    identifiers: set[tuple[str, str]] = set()
    for match in re.finditer(r"\bCVE-\d{4}-\d{4,8}\b", text, flags=re.IGNORECASE):
        identifiers.add((match.group(0), "vulnerability"))
    for match in URL_RE.finditer(text):
        value = match.group(0).rstrip(".,;:\uff0c\u3002\uff1b\uff1a")
        lowered = value.lower()
        kind = (
            "project"
            if "github.com/" in lowered
            else "vulnerability"
            if "nvd.nist.gov/" in lowered
            else "paper"
            if "arxiv.org/" in lowered
            else "course"
        )
        identifiers.add((value, kind))
    for match in GITHUB_REPOSITORY_RE.finditer(text):
        value = match.group(1).rstrip(".,;:\uff0c\u3002\uff1b\uff1a")
        if not value.lower().startswith(("http/", "https/")):
            identifiers.add((value, "project"))
    for match in ARXIV_ID_RE.finditer(text):
        identifiers.add((match.group(1), "paper"))
    for match in IDENTIFIER_RE.finditer(text):
        value = match.group(0)
        if not value.upper().startswith("CVE-"):
            identifiers.add((value, "course"))
    return identifiers


def _normalized_locator_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _normalized_source_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "vuln": "vulnerability",
        "cve": "vulnerability",
        "github": "project",
        "repository": "project",
        "document": "course",
    }
    return aliases.get(normalized, normalized)


def _requested_resource_types(question: str) -> set[str]:
    normalized = question.lower()
    categories = {
        "paper": ("paper", "\u8bba\u6587", "\u6587\u732e", "arxiv"),
        "vulnerability": ("cve", "cvss", "cwe", "\u6f0f\u6d1e"),
        "standard": ("3gpp", "specification", "\u6807\u51c6"),
        "project": ("github", "repository", "project", "\u5f00\u6e90\u9879\u76ee"),
    }
    return {
        resource_type
        for resource_type, terms in categories.items()
        if any(term in normalized for term in terms)
    }


def _requested_resource_type(question: str) -> str:
    requested = _requested_resource_types(question)
    return next((item for item in ("paper", "vulnerability", "standard", "project") if item in requested), "")


def _support_tokens(text: str) -> set[str]:
    normalized = text.lower()
    tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,}", normalized))
    tokens.update(token.lower() for token in tokenize(text) if len(token.strip()) > 1)
    aliases = {
        "blockchain": ("blockchain", "\u533a\u5757\u94fe"),
        "covert_communication": ("covert communication", "covert channel", "\u9690\u853d\u901a\u4fe1", "\u9690\u853d\u4fe1\u9053"),
        "authentication": ("authentication", "\u8ba4\u8bc1", "\u9274\u6743"),
        "vulnerability": ("vulnerability", "\u6f0f\u6d1e"),
    }
    tokens.update(
        canonical
        for canonical, terms in aliases.items()
        if any(term in normalized for term in terms)
    )
    stopwords = {
        "\u6839\u636e",
        "\u8bc1\u636e",
        "\u56de\u7b54",
        "\u5f53\u524d",
        "\u8bfe\u7a0b",
        "\u53ef\u4ee5",
        "\u8fdb\u884c",
        "\u652f\u6301",
        "\u8bf7",
        "\u89e3\u91ca",
        "\u4ec0\u4e48",
        "\u5982\u4f55",
        "\u5b9a\u4e49",
        "\u552f\u4e00",
        "\u6807\u8bb0",
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "mock",
        "answer",
        "please",
        "explain",
        "compare",
        "research",
    }
    return {token for token in tokens if token not in stopwords}
