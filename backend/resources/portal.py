"""Unified course resource portal service."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.connectors.schemas import ResourceQuery
from backend.connectors.standard_source import (
    STANDARD_METADATA_SOURCE_NOTE,
    normalize_standard_resource_view,
)
from backend.resources.repository import ExternalResourceRepository
from backend.rag.visibility import is_student_visible_document
from backend.storage.logger import LearningLogger


RESOURCE_TYPES = {
    "course_document",
    "paper",
    "standard",
    "vulnerability",
    "project",
    "competition",
}


@dataclass(frozen=True)
class ResourcePortalQuery:
    keyword: str = ""
    topic: str = ""
    resource_type: str = ""
    year_from: int | None = None
    year_to: int | None = None
    source: str = ""
    page: int = 1
    page_size: int = 10


class ResourcePortalService:
    def __init__(
        self,
        storage: LearningLogger,
        *,
        competitions_dir: str | Path = "datasets/competitions",
        course_id: int | None = None,
    ) -> None:
        self.storage = storage
        self.repository = ExternalResourceRepository(storage)
        self.competitions_dir = Path(competitions_dir)
        self.course_id = course_id

    def search(self, query: ResourcePortalQuery) -> dict[str, Any]:
        page = max(1, query.page)
        page_size = max(1, min(50, query.page_size))
        all_items = self._dedupe(
            [
                *self._course_documents(),
                *self._external_resources(),
                *self._competitions(),
            ]
        )
        filtered = [item for item in all_items if self._matches(item, query)]
        filtered = [_present_item(item) for item in filtered]
        filtered.sort(key=lambda item: (item.get("updated_at") or item.get("fetched_at") or ""), reverse=True)
        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "items": filtered[start:end],
            "total": total,
            "page": page,
            "page_size": page_size,
            "types": sorted(RESOURCE_TYPES),
            "cache_mode": "cache_first",
        }

    def resource_exists(self, resource_id: str) -> bool:
        return any(item["resource_id"] == resource_id for item in self._dedupe([*self._course_documents(), *self._external_resources(), *self._competitions()]))

    def _course_documents(self) -> list[dict[str, Any]]:
        items = []
        if self.course_id is None:
            return items
        for document in self.storage.documents(course_id=self.course_id):
            if not is_student_visible_document(document):
                continue
            chunks = [
                item
                for item in self.storage.chunks_for_document(int(document["id"]))
                if str(item.get("index_status") or "").upper() == "INDEXED"
            ]
            if not chunks:
                continue
            snippet = ""
            if chunks:
                snippet = str(chunks[0].get("chunk_text") or "")[:500]
            tags = [str(document.get("file_type") or "").lstrip(".").upper(), "已索引", "已发布"]
            items.append(
                {
                    "resource_id": f"document:{document['id']}",
                    "type": "course_document",
                    "title": document.get("title") or document.get("filename") or f"Document {document['id']}",
                    "summary": snippet or f"Uploaded course document: {document.get('filename', '')}",
                    "raw_summary": snippet or "",
                    "tags": [tag for tag in tags if tag],
                    "source": "course_document",
                    "source_name": "local",
                    "external_id": str(document["id"]),
                    "url": "",
                    "module": "",
                    "year": _year(document.get("created_at")),
                    "cache_status": "local",
                    "origin_kind": "course_document",
                    "delivery_mode": "course_published",
                    "fetched_at": document.get("created_at"),
                    "updated_at": document.get("updated_at"),
                    "metadata": {
                        "filename": document.get("filename"),
                        "chunk_count": document.get("chunk_count"),
                        "index_status": document.get("index_status"),
                        "index_status_label": "已索引",
                        "course_id": document.get("course_id"),
                        "published_to_students": True,
                        "publication_status_label": "已发布",
                        "document_id": int(document["id"]),
                        "chunk_id": int(chunks[0]["id"]),
                    },
                }
            )
        return items

    def _external_resources(self) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        page = 1
        while page <= 20:
            batch = self.repository.search_cache(
                ResourceQuery(query="", page=page, page_size=50),
                include_expired=True,
            )
            resources.extend(batch)
            if len(batch) < 50:
                break
            page += 1
        items = []
        now = datetime.now(UTC)
        for stored_resource in resources:
            resource = normalize_standard_resource_view(stored_resource)
            if not resource.get("provenance_verified"):
                continue
            if resource.get("source_name") == "github" and not is_public_github_record(resource):
                continue
            expires_at = _parse_time(resource.get("expires_at"))
            cache_status = "expired" if expires_at and expires_at <= now else "fresh"
            source_name = resource.get("source_name") or ""
            origin_kind = resource.get("origin_kind") or "unknown"
            delivery_mode = "fresh_cache" if cache_status == "fresh" else "stale_cache"
            items.append(
                {
                    "resource_id": f"external:{resource['id']}",
                    "type": resource.get("resource_type"),
                    "title": resource.get("title") or "",
                    "summary": resource.get("abstract") or "",
                    "raw_summary": resource.get("abstract") or "",
                    "tags": resource.get("tags") or [],
                    "source": resource.get("source_name") or "",
                    "source_name": resource.get("source_name") or "",
                    "external_id": resource.get("external_id") or "",
                    "url": _safe_url(resource.get("url") or ""),
                    "module": resource.get("module") or "",
                    "year": _year(resource.get("published_at")),
                    "cache_status": cache_status,
                    "origin_kind": origin_kind,
                    "delivery_mode": delivery_mode,
                    "fetched_at": resource.get("fetched_at"),
                    "updated_at": resource.get("updated_at"),
                    "metadata": {
                        "authors": resource.get("authors") or [],
                        "rights_note": resource.get("rights_note") or "",
                        "indexed": resource.get("indexed"),
                        "vector_id": resource.get("vector_id"),
                        "raw_metadata": resource.get("raw_metadata") or {},
                        "provenance_verified": True,
                        "provenance_sync_run_id": resource.get("provenance_sync_run_id"),
                    },
                }
            )
        return items

    def _competitions(self) -> list[dict[str, Any]]:
        items = []
        if not self.competitions_dir.exists():
            return items
        for path in sorted(self.competitions_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            records = payload if isinstance(payload, list) else payload.get("items", [])
            for record in records:
                if not isinstance(record, dict):
                    continue
                external_id = str(record.get("id") or path.stem)
                items.append(
                    {
                        "resource_id": f"competition:{external_id}",
                        "type": "competition",
                        "title": str(record.get("title") or "Competition topic"),
                        "summary": str(record.get("summary") or record.get("description") or ""),
                        "raw_summary": str(record.get("summary") or record.get("description") or ""),
                        "tags": list(record.get("tags") or []),
                        "source": str(record.get("source") or "teacher_dataset"),
                        "source_name": "teacher_dataset",
                        "external_id": external_id,
                        "url": _safe_url(str(record.get("url") or "")),
                        "module": str(record.get("module") or ""),
                        "year": record.get("year"),
                        "cache_status": "teacher_maintained",
                        "fetched_at": record.get("updated_at") or "",
                        "updated_at": record.get("updated_at") or "",
                        "metadata": {"dataset": str(path.name)},
                    }
                )
        return items

    @staticmethod
    def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, str]] = set()
        deduped = []
        for item in items:
            key = (
                str(item.get("type") or ""),
                str(item.get("source_name") or item.get("source") or ""),
                str(item.get("external_id") or item.get("url") or item.get("title") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _matches(item: dict[str, Any], query: ResourcePortalQuery) -> bool:
        if query.resource_type and item.get("type") != query.resource_type:
            return False
        if query.source and query.source.lower() not in str(item.get("source") or "").lower():
            return False
        if query.topic and query.topic.lower() not in " ".join([str(item.get("module") or ""), " ".join(item.get("tags") or [])]).lower():
            return False
        if query.keyword:
            haystack = " ".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("summary") or ""),
                    " ".join(item.get("tags") or []),
                    str(item.get("module") or ""),
                    str(item.get("source") or ""),
                ]
            ).lower()
            if query.keyword.lower() not in haystack:
                return False
        year = item.get("year")
        if query.year_from is not None and (year is None or int(year) < int(query.year_from)):
            return False
        if query.year_to is not None and (year is None or int(year) > int(query.year_to)):
            return False
        return True


def _year(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value)
    for index in range(max(0, len(text) - 3)):
        candidate = text[index : index + 4]
        if candidate.isdigit():
            return int(candidate)
    return None


TYPE_LABELS = {
    "course_document": "课程文档",
    "paper": "论文",
    "standard": "标准元数据",
    "vulnerability": "漏洞",
    "project": "开源项目",
    "competition": "竞赛题目",
}


def _present_item(item: dict[str, Any]) -> dict[str, Any]:
    presented = dict(item)
    raw_summary = str(presented.get("raw_summary") or presented.get("summary") or "")
    presented["raw_summary"] = raw_summary
    presented["type_label"] = TYPE_LABELS.get(str(presented.get("type") or ""), str(presented.get("type") or "unknown"))
    topic_category, topic_basis = _classify_topic(presented)
    presented["topic_category"] = topic_category
    presented["date_label"] = str(presented["year"]) if presented.get("year") is not None else "日期未知"
    presented["summary"] = _source_bound_summary(presented, raw_summary)
    metadata = dict(presented.get("metadata") or {})
    metadata.setdefault("topic_basis", topic_basis)
    metadata.setdefault("topic_basis_source", "module/title/tags")
    if presented.get("type") == "standard":
        metadata["metadata_only"] = True
        metadata["local_full_text_available"] = False
        metadata["source_note"] = STANDARD_METADATA_SOURCE_NOTE
        presented["tags"] = _dedupe_tags([*(presented.get("tags") or []), "官方元数据", "日期未知" if presented.get("year") is None else ""])
    presented["metadata"] = metadata
    return presented


def _source_bound_summary(item: dict[str, Any], raw_summary: str, max_chars: int = 180) -> str:
    resource_type = str(item.get("type") or "")
    title = str(item.get("title") or "Untitled resource")
    source = str(item.get("source_name") or item.get("source") or "local")
    text = " ".join(raw_summary.split())
    if resource_type == "standard":
        basis = text or "标准编号、版本、官方链接和本地上传状态元数据。"
        return _trim(f"标准元数据：{title}，来源 {source}。{basis} 仅展示官方元数据链接和本地正文状态，不提供受版权保护全文。", max_chars)
    if resource_type == "paper":
        return _trim(f"论文资料：{title}，来源 {source}。{text or '暂无摘要，需打开来源核对。'}", max_chars)
    if resource_type == "vulnerability":
        return _trim(f"漏洞资料：{title}，来源 {source}。{text or '暂无描述，需打开来源核对。'}", max_chars)
    if resource_type == "project":
        return _trim(f"开源项目：{title}，来源 {source}。{text or '暂无项目说明，需打开来源核对。'}", max_chars)
    if resource_type == "competition":
        return _trim(f"竞赛题目：{title}，来源 {source}。{text or '教师维护题目，无外部抓取。'}", max_chars)
    return _trim(f"课程资料：{title}，来源 {source}。{text or '暂无摘要。'}", max_chars)


def _topic_category(item: dict[str, Any]) -> str:
    values = " ".join(
        [
            str(item.get("module") or ""),
            str(item.get("title") or ""),
            str(item.get("summary") or ""),
            " ".join(str(tag) for tag in item.get("tags") or []),
        ]
    ).lower()
    if any(token in values for token in ("5g nas", "3gpp", "ts 24.501", "nas")):
        return "5G NAS"
    if any(token in values for token in ("blockchain", "区块链", "covert", "隐蔽")):
        return "区块链隐蔽通信"
    if any(token in values for token in ("cve", "vulnerability", "漏洞", "cvss", "cwe")):
        return "漏洞分析"
    if any(token in values for token in ("llm", "prompt", "模型")):
        return "LLM 安全"
    if item.get("type") == "project":
        return "开源项目"
    return str(item.get("module") or "课程通用")


_legacy_source_bound_summary = _source_bound_summary
_legacy_topic_category = _topic_category


def _source_bound_summary(item: dict[str, Any], raw_summary: str, max_chars: int = 180) -> str:
    resource_type = str(item.get("type") or "")
    text = " ".join(str(raw_summary or "").split())
    if resource_type == "standard":
        # Reuse the existing source/type wording, but only pass verified metadata.
        return _legacy_source_bound_summary(
            item,
            _standard_metadata_basis(item, text),
            max_chars,
        )
    selected = _select_relevant_sentence(item, text)
    return _legacy_source_bound_summary(item, selected, max_chars)


def _classify_topic(item: dict[str, Any]) -> tuple[str, list[str]]:
    values = " ".join(
        [
            str(item.get("module") or ""),
            str(item.get("title") or ""),
            str(item.get("summary") or ""),
            " ".join(str(tag) for tag in item.get("tags") or []),
        ]
    ).lower()
    rules = (
        (("5g nas", "3gpp", "ts 24.501", "nas"), "5G NAS"),
        (("blockchain", "\u533a\u5757\u94fe", "covert", "\u9690\u853d"), "blockchain"),
        (("cve", "vulnerability", "\u6f0f\u6d1e", "cvss", "cwe"), "vulnerability"),
        (("llm", "prompt", "\u6a21\u578b"), "llm"),
    )
    for tokens, category in rules:
        matched = [token for token in tokens if token in values]
        if matched:
            return _legacy_topic_category(item), matched
    return "\u5f85\u5206\u7c7b", []


def _select_relevant_sentence(item: dict[str, Any], text: str) -> str:
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s*|\r?\n+", text)
        if part.strip()
    ]
    if len(sentences) <= 1:
        return text
    terms = _summary_terms(item)
    if not terms:
        return sentences[0]
    scored = [
        (sum(1 for term in terms if term in sentence.lower()), index, sentence)
        for index, sentence in enumerate(sentences)
    ]
    relevant = [entry for entry in scored if entry[0] > 0]
    if not relevant:
        return sentences[0]
    return max(relevant, key=lambda entry: (entry[0], -entry[1]))[2]


def _summary_terms(item: dict[str, Any]) -> list[str]:
    values = " ".join(
        [
            str(item.get("module") or ""),
            str(item.get("title") or ""),
            " ".join(str(tag) for tag in item.get("tags") or []),
        ]
    ).lower()
    category, basis = _classify_topic(item)
    terms = list(basis)
    if category == "5G NAS":
        terms.extend(["authentication", "security", "key agreement", "replay"])
    elif category == "vulnerability":
        terms.extend(["cvss", "cwe", "patch", "exploit"])
    elif category == "llm":
        terms.extend(["injection", "model", "prompt"])
    terms.extend(
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", values)
        if token not in terms
    )
    return list(dict.fromkeys(terms))


def _standard_metadata_basis(item: dict[str, Any], text: str) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    version = str(metadata.get("version") or "").strip()
    if not version:
        match = re.search(
            r"\bversion\s+([0-9][0-9A-Za-z._-]*)",
            text,
            flags=re.IGNORECASE,
        )
        version = match.group(1) if match else ""
    official_url = _safe_url(str(item.get("url") or metadata.get("official_url") or ""))
    local_text = (
        "local full text available"
        if metadata.get("local_full_text_available")
        else "local full text unavailable"
    )
    parts = [
        part
        for part in (
            f"version {version}" if version else "",
            "official link available" if official_url else "official link unavailable",
            local_text,
        )
        if part
    ]
    return "; ".join(parts) or "standard number, version, official link, and local text status are unavailable"


def _trim(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}..."


def _dedupe_tags(tags: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        text = str(tag or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        return None


def _safe_url(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith(("https://", "http://")):
        return text
    return ""


def is_public_github_record(resource: dict[str, Any]) -> bool:
    metadata = resource.get("raw_metadata")
    if not isinstance(metadata, dict):
        return False
    visibility = metadata.get("visibility")
    return (
        isinstance(metadata.get("full_name"), str)
        and bool(metadata["full_name"].strip())
        and metadata.get("private") is False
        and isinstance(visibility, str)
        and bool(visibility.strip())
        and visibility.lower() not in {"private", "internal"}
    )
