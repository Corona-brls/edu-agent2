"""Normalize external connector resources for cache and indexing."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from backend.connectors.schemas import ExternalResource


MAX_RESOURCE_AUTHORS = 20
MAX_RESOURCE_AUTHOR_CHARS = 160


def canonical_authors(value: Any) -> list[str]:
    """Return the bounded author list used by storage, indexing, and hashing."""

    if isinstance(value, dict):
        value = value.get("authors")
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            continue
        cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", raw)
        cleaned = " ".join(cleaned.split())[:MAX_RESOURCE_AUTHOR_CHARS].strip()
        if not cleaned or cleaned in seen:
            continue
        authors.append(cleaned)
        seen.add(cleaned)
        if len(authors) >= MAX_RESOURCE_AUTHORS:
            break
    return authors


def resource_content(resource: ExternalResource) -> str:
    authors = canonical_authors(resource.authors)
    parts = [
        resource.title,
        "Authors: " + "; ".join(authors) if authors else "",
        resource.abstract,
        " ".join(resource.tags),
        resource.module,
        resource.url,
    ]
    return "\n".join(part for part in parts if part)


def content_hash(resource: ExternalResource) -> str:
    payload = {
        "source_name": resource.source_name,
        "external_id": resource.external_id,
        "resource_type": resource.resource_type,
        "title": resource.title,
        "authors": canonical_authors(resource.authors),
        "abstract": resource.abstract,
        "url": resource.url,
        "published_at": resource.published_at,
        "tags": resource.tags,
        "module": resource.module,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

