"""Request-scoped evidence catalog and fail-closed selection validation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable
from urllib.parse import urlsplit

from backend.learning.context import CourseReference
from backend.practice.schemas import DraftBackgroundQuote, PracticeDraftCandidate


_PLAN_REFERENCE_RE = re.compile(r"\[S[1-9][0-9]*(?::text:[1-9][0-9]*)?\]", re.IGNORECASE)


class DraftEvidenceError(ValueError):
    def __init__(self, error_code: str, field: str, index: int | None = None) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.field = field
        self.index = index

    def feedback(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error_code": self.error_code, "field": self.field}
        if self.index is not None:
            payload["index"] = self.index
        return payload


@dataclass(frozen=True)
class DraftEvidenceSource:
    source_id: str
    source_kind: str
    title: str
    url: str | None
    delivery_mode: str
    fetched_at: str | None
    identity: tuple[Any, ...]
    content_hash: str


@dataclass(frozen=True)
class DraftEvidenceSpan:
    source_id: str
    evidence_span_id: str
    exact_text: str


@dataclass(frozen=True)
class DraftEvidenceCatalog:
    sources: tuple[DraftEvidenceSource, ...]
    spans: tuple[DraftEvidenceSpan, ...]

    def model_payload(self) -> list[dict[str, str]]:
        return [
            {
                "source_id": span.source_id,
                "evidence_span_id": span.evidence_span_id,
                "exact_text": span.exact_text,
            }
            for span in self.spans
        ]


def build_draft_evidence(
    *,
    course_references: Iterable[CourseReference] = (),
    external_resources: Iterable[dict[str, Any]] = (),
    max_sources: int = 8,
    max_total_chars: int = 8000,
) -> DraftEvidenceCatalog:
    sources: list[DraftEvidenceSource] = []
    spans: list[DraftEvidenceSpan] = []
    seen: set[tuple[Any, ...]] = set()
    remaining = max(0, int(max_total_chars))

    items: list[tuple[str, dict[str, Any]]] = []
    for reference in course_references:
        items.append(
            (
                "course",
                {
                    "identity": (int(reference.document_id), int(reference.chunk_id)),
                    "title": reference.title,
                    "url": None,
                    "delivery_mode": "course_reference",
                    "fetched_at": None,
                    "content_hash": f"{reference.content_hash}:{reference.chunk_hash}",
                    "text": reference.text,
                },
            )
        )
    for resource in external_resources:
        if not bool(resource.get("provenance_verified")):
            continue
        if str(resource.get("origin_kind") or "") not in {"remote", "local_catalog"}:
            continue
        if not str(resource.get("source_name") or "") or not str(resource.get("external_id") or ""):
            continue
        url = _safe_http_url(resource.get("url"))
        items.append(
            (
                "external",
                {
                    "identity": (str(resource["source_name"]), str(resource["external_id"])),
                    "title": str(resource.get("title") or resource.get("external_id") or "公开资料")[:300],
                    "url": url,
                    "delivery_mode": str(resource.get("delivery_mode") or "unknown")[:60],
                    "fetched_at": str(resource.get("fetched_at") or "")[:80] or None,
                    "content_hash": str(resource.get("content_hash") or ""),
                    "text": str(resource.get("abstract") or ""),
                },
            )
        )

    for kind, item in items:
        identity = (kind, *item["identity"])
        if identity in seen or len(sources) >= max_sources or remaining <= 0:
            continue
        sentence = _complete_text_span(item["text"], min(360, remaining))
        if not sentence:
            continue
        seen.add(identity)
        source_id = f"S{len(sources) + 1}"
        sources.append(
            DraftEvidenceSource(
                source_id=source_id,
                source_kind=kind,
                title=str(item["title"]),
                url=item["url"],
                delivery_mode=str(item["delivery_mode"]),
                fetched_at=item["fetched_at"],
                identity=tuple(item["identity"]),
                content_hash=str(item["content_hash"]),
            )
        )
        spans.append(
            DraftEvidenceSpan(
                source_id=source_id,
                evidence_span_id=f"{source_id}:text:1",
                exact_text=sentence,
            )
        )
        remaining -= len(sentence)
    return DraftEvidenceCatalog(sources=tuple(sources), spans=tuple(spans))


def validate_draft_candidate(
    candidate: PracticeDraftCandidate,
    catalog: DraftEvidenceCatalog,
) -> tuple[DraftEvidenceSpan, ...]:
    span_map = {span.evidence_span_id: span for span in catalog.spans}
    seen: set[tuple[str, str]] = set()
    selected: list[DraftEvidenceSpan] = []
    if candidate.selected_spans and not span_map:
        raise DraftEvidenceError("DRAFT_EVIDENCE_EMPTY", "selected_spans")
    for index, selection in enumerate(candidate.selected_spans):
        key = (selection.source_id, selection.evidence_span_id)
        if key in seen:
            raise DraftEvidenceError("DRAFT_SELECTION_DUPLICATE", "selected_spans", index)
        seen.add(key)
        span = span_map.get(selection.evidence_span_id)
        if span is None:
            raise DraftEvidenceError("DRAFT_SPAN_UNKNOWN", "selected_spans", index)
        if span.source_id != selection.source_id:
            raise DraftEvidenceError("DRAFT_SOURCE_SPAN_MISMATCH", "selected_spans", index)
        selected.append(span)
    if candidate.proposed_plan is not None:
        text = candidate.proposed_plan.model_dump_json()
        if _PLAN_REFERENCE_RE.search(text):
            raise DraftEvidenceError("DRAFT_PLAN_CONTAINS_SOURCE_MARKER", "proposed_plan")
    return tuple(selected)


def render_background_quotes(
    selections: Iterable[DraftEvidenceSpan],
    catalog: DraftEvidenceCatalog,
) -> list[DraftBackgroundQuote]:
    source_map = {source.source_id: source for source in catalog.sources}
    result: list[DraftBackgroundQuote] = []
    for span in selections:
        source = source_map.get(span.source_id)
        if source is None:
            raise DraftEvidenceError("DRAFT_SOURCE_UNKNOWN", "selected_spans")
        result.append(
            DraftBackgroundQuote(
                source_id=source.source_id,
                evidence_span_id=span.evidence_span_id,
                quote_text=span.exact_text,
                source_title=source.title,
                source_url=source.url,
                delivery_mode=source.delivery_mode,
                fetched_at=source.fetched_at,
            )
        )
    return result


def _complete_text_span(value: Any, limit: int) -> str:
    text = str(value or "")
    if not text.strip() or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    boundary = max(text.rfind(mark, 0, limit + 1) for mark in ("。", "！", "？", "!", "?", "\n"))
    if boundary < 0:
        return ""
    return text[: boundary + 1].strip()


def _safe_http_url(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = urlsplit(text)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return None
    return text[:2000]
