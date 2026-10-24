"""Server-owned publication metadata and student visibility helpers."""

from __future__ import annotations

from typing import Any, Mapping


PUBLICATION_KEYS = {
    "published_to_students",
    "published_by",
    "published_at",
    "publication_last_action",
}


def strict_true(value: Any) -> bool:
    """Only a real JSON boolean true opens the publication gate."""

    return value is True


def sanitized_new_document_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Strip server-owned keys and force every new document to unpublished."""

    clean = {
        str(key): value
        for key, value in dict(metadata or {}).items()
        if str(key) not in PUBLICATION_KEYS
    }
    clean.update(
        {
            "published_to_students": False,
            "published_by": None,
            "published_at": None,
            "publication_last_action": None,
        }
    )
    return clean


def publication_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    action = metadata.get("publication_last_action")
    return {
        "published_to_students": strict_true(metadata.get("published_to_students")),
        "published_by": metadata.get("published_by") if isinstance(metadata.get("published_by"), int) else None,
        "published_at": metadata.get("published_at") if isinstance(metadata.get("published_at"), str) else None,
        "publication_last_action": dict(action) if isinstance(action, Mapping) else None,
    }


def is_student_visible_document(document: Mapping[str, Any]) -> bool:
    return (
        str(document.get("index_status") or "").upper() == "INDEXED"
        and publication_projection(document)["published_to_students"] is True
    )


def is_student_visible_chunk(chunk: Mapping[str, Any]) -> bool:
    return (
        str(chunk.get("index_status") or "").upper() == "INDEXED"
        and str(chunk.get("document_index_status") or "").upper() == "INDEXED"
        and strict_true(chunk.get("published_to_students"))
    )
