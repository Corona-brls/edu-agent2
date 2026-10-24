"""Strict reader for the review-gated shared knowledge-source catalog.

The catalog describes source drafts only.  Loading it never imports a document
into a course, writes to SQLite/Chroma, or grants student visibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.parse import urlparse

from backend.rag.document_parser import DocumentParseError, parse_document


_ENTRY_FIELDS = {
    "source_id",
    "title",
    "topic",
    "path",
    "revision",
    "review_state",
    "reviewer",
    "reviewed_at",
    "reference_urls",
    "licence_note",
}
_REVIEW_STATES = {"draft", "pending_review", "approved", "retired"}
_SOURCE_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_REVISION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.-]+)?$")


class CatalogValidationError(ValueError):
    """Raised when catalog metadata or a referenced draft is invalid."""


@dataclass(frozen=True)
class KnowledgeSourceEntry:
    source_id: str
    title: str
    topic: str
    path: str
    revision: str
    review_state: str
    reviewer: str | None
    reviewed_at: str | None
    reference_urls: tuple[str, ...]
    licence_note: str
    char_count: int
    section_count: int


@dataclass(frozen=True)
class KnowledgeSourceCatalog:
    schema_version: int
    catalog_state: str
    entries: tuple[KnowledgeSourceEntry, ...]


def load_source_catalog(catalog_path: str | Path) -> KnowledgeSourceCatalog:
    """Load and validate a catalog and every Markdown draft it references."""

    path = Path(catalog_path)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogValidationError(f"无法读取知识源清单：{path.name}") from exc

    if not isinstance(payload, dict) or set(payload) != {"schema_version", "catalog_state", "entries"}:
        raise CatalogValidationError("知识源清单顶层字段不符合 schema")
    if payload["schema_version"] != 1:
        raise CatalogValidationError("不支持的知识源清单 schema_version")
    if payload["catalog_state"] != "draft":
        raise CatalogValidationError("当前知识源清单必须保持 draft")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise CatalogValidationError("知识源清单 entries 必须是非空数组")

    root = path.parent.resolve()
    source_ids: set[str] = set()
    relative_paths: set[str] = set()
    entries: list[KnowledgeSourceEntry] = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict) or set(raw) != _ENTRY_FIELDS:
            raise CatalogValidationError(f"第 {index + 1} 条知识源字段不符合 schema")
        entry = _validate_entry(raw, root=root, index=index)
        if entry.source_id in source_ids:
            raise CatalogValidationError(f"重复 source_id：{entry.source_id}")
        if entry.path.casefold() in relative_paths:
            raise CatalogValidationError(f"重复 path：{entry.path}")
        source_ids.add(entry.source_id)
        relative_paths.add(entry.path.casefold())
        entries.append(entry)

    return KnowledgeSourceCatalog(
        schema_version=1,
        catalog_state="draft",
        entries=tuple(entries),
    )


def _validate_entry(raw: dict[str, Any], *, root: Path, index: int) -> KnowledgeSourceEntry:
    label = f"第 {index + 1} 条知识源"
    source_id = _required_text(raw.get("source_id"), f"{label}.source_id", max_length=120)
    if not _SOURCE_ID_RE.fullmatch(source_id):
        raise CatalogValidationError(f"{label}.source_id 格式不合法")
    title = _required_text(raw.get("title"), f"{label}.title", max_length=160)
    topic = _required_text(raw.get("topic"), f"{label}.topic", max_length=80)
    revision = _required_text(raw.get("revision"), f"{label}.revision", max_length=48)
    if not _REVISION_RE.fullmatch(revision):
        raise CatalogValidationError(f"{label}.revision 格式不合法")

    review_state = _required_text(raw.get("review_state"), f"{label}.review_state", max_length=32)
    if review_state not in _REVIEW_STATES:
        raise CatalogValidationError(f"{label}.review_state 不受支持")
    reviewer = _optional_text(raw.get("reviewer"), f"{label}.reviewer", max_length=120)
    reviewed_at = _optional_text(raw.get("reviewed_at"), f"{label}.reviewed_at", max_length=64)
    if review_state in {"draft", "pending_review"} and (reviewer is not None or reviewed_at is not None):
        raise CatalogValidationError(f"{label}尚未审核，不得填写 reviewer/reviewed_at")
    if review_state == "approved" and (reviewer is None or reviewed_at is None):
        raise CatalogValidationError(f"{label}标为 approved 时必须有真实审核记录")

    relative_path = _required_text(raw.get("path"), f"{label}.path", max_length=180)
    pure_path = PurePosixPath(relative_path)
    if (
        pure_path.is_absolute()
        or Path(relative_path).is_absolute()
        or ":" in relative_path
        or "\\" in relative_path
        or relative_path != pure_path.as_posix()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or pure_path.suffix.lower() != ".md"
    ):
        raise CatalogValidationError(f"{label}.path 必须是安全的相对 Markdown 路径")
    resolved = (root / Path(*pure_path.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CatalogValidationError(f"{label}.path 越出共享知识源目录") from exc

    raw_urls = raw.get("reference_urls")
    if not isinstance(raw_urls, list) or not raw_urls:
        raise CatalogValidationError(f"{label}.reference_urls 必须是非空数组")
    urls: list[str] = []
    for url_index, raw_url in enumerate(raw_urls):
        url = _required_text(raw_url, f"{label}.reference_urls[{url_index}]", max_length=500)
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise CatalogValidationError(f"{label}.reference_urls 只允许无凭据 HTTPS URL")
        urls.append(url)
    if len(urls) != len(set(urls)):
        raise CatalogValidationError(f"{label}.reference_urls 存在重复项")

    licence_note = _required_text(raw.get("licence_note"), f"{label}.licence_note", max_length=500)
    try:
        document = parse_document(resolved)
    except DocumentParseError as exc:
        raise CatalogValidationError(f"{label}引用的草稿不可解析") from exc
    if document.title != title:
        raise CatalogValidationError(f"{label}.title 与 Markdown 一级标题不一致")
    sections = [line for line in document.text.splitlines() if line.startswith("## ")]
    if len(sections) < 6:
        raise CatalogValidationError(f"{label}缺少统一教学骨架")

    return KnowledgeSourceEntry(
        source_id=source_id,
        title=title,
        topic=topic,
        path=relative_path,
        revision=revision,
        review_state=review_state,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        reference_urls=tuple(urls),
        licence_note=licence_note,
        char_count=len(document.text),
        section_count=len(sections),
    )


def _required_text(value: Any, field: str, *, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogValidationError(f"{field} 必须是非空字符串")
    normalized = value.strip()
    if len(normalized) > max_length:
        raise CatalogValidationError(f"{field} 超过长度上限")
    return normalized


def _optional_text(value: Any, field: str, *, max_length: int) -> str | None:
    if value is None:
        return None
    return _required_text(value, field, max_length=max_length)


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CatalogValidationError(f"知识源清单存在重复 JSON 键：{key}")
        result[key] = value
    return result
