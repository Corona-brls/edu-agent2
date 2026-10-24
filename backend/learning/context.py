"""Read-only, course-scoped reference projection for learning features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.rag.visibility import is_student_visible_document
from backend.storage.logger import LearningLogger


@dataclass(frozen=True)
class CourseReference:
    """A current, server-owned view of one visible course chunk."""

    source_id: str
    document_id: int
    chunk_id: int
    title: str
    text: str
    page: int | None
    section: str
    source_type: str
    content_hash: str
    chunk_hash: str


class CourseReferenceError(RuntimeError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(f"{error_code}: {message}")
        self.error_code = error_code


def load_course_references(
    storage: LearningLogger,
    retriever: Any,
    *,
    course_id: int,
    query: str,
    top_k: int,
    max_chars: int,
) -> list[CourseReference]:
    """Retrieve then revalidate current course references without writing state."""

    if course_id is None or isinstance(course_id, bool) or int(course_id) <= 0:
        raise CourseReferenceError("COURSE_REQUIRED", "课程参考必须绑定非空课程。")
    if top_k <= 0 or max_chars <= 0 or not str(query or "").strip():
        return []

    selected_course_id = int(course_id)
    response = retriever.retrieve(
        str(query),
        course_id=selected_course_id,
        top_k=int(top_k),
    )
    results = list(getattr(response, "results", ()) or ())
    references: list[CourseReference] = []
    seen: set[tuple[int, int]] = set()
    remaining = int(max_chars)

    for result in results:
        try:
            document_id = int(result.document_id)
            chunk_id = int(result.chunk_id)
        except (AttributeError, TypeError, ValueError):
            continue
        identity = (document_id, chunk_id)
        if identity in seen:
            continue
        seen.add(identity)

        document = storage.document(document_id)
        if document is None:
            continue
        stored_course_id = document.get("course_id")
        if stored_course_id is None or int(stored_course_id) != selected_course_id:
            raise CourseReferenceError(
                "REFERENCE_SCOPE_VIOLATION",
                "检索结果不属于当前课程。",
            )
        if not is_student_visible_document(document):
            continue

        chunk = next(
            (
                item
                for item in storage.chunks_for_document(document_id)
                if int(item.get("id") or 0) == chunk_id
            ),
            None,
        )
        if chunk is None or str(chunk.get("index_status") or "").upper() != "INDEXED":
            continue

        text = _clean_reference_text(chunk.get("chunk_text"))
        if not text or remaining <= 0:
            continue
        bounded_text = text[:remaining].rstrip()
        if not bounded_text:
            continue
        remaining -= len(bounded_text)
        references.append(
            CourseReference(
                source_id=f"rag:{document_id}:{chunk_id}",
                document_id=document_id,
                chunk_id=chunk_id,
                title=_clean_reference_text(document.get("title") or document.get("filename"))[:300],
                text=bounded_text,
                page=chunk.get("page") if isinstance(chunk.get("page"), int) else None,
                section=_clean_reference_text(chunk.get("section"))[:300],
                source_type=_clean_reference_text(chunk.get("source_type") or document.get("file_type"))[:100],
                content_hash=str(document.get("content_hash") or ""),
                chunk_hash=str(chunk.get("chunk_hash") or ""),
            )
        )
        if remaining <= 0:
            break
    return references


def revalidate_course_references(
    storage: LearningLogger,
    references: list[CourseReference],
    *,
    course_id: int,
) -> list[CourseReference]:
    """Return a fresh projection only when every selected reference is unchanged."""

    current: list[CourseReference] = []
    for reference in references:
        document = storage.document(reference.document_id)
        if document is None or int(document.get("course_id") or 0) != int(course_id):
            raise CourseReferenceError("REFERENCE_CONTEXT_CHANGED", "课程参考已变化。")
        if not is_student_visible_document(document):
            raise CourseReferenceError("REFERENCE_CONTEXT_CHANGED", "课程参考已撤回或正在重建索引。")
        chunk = next(
            (
                item
                for item in storage.chunks_for_document(reference.document_id)
                if int(item.get("id") or 0) == int(reference.chunk_id)
            ),
            None,
        )
        if chunk is None or str(chunk.get("index_status") or "").upper() != "INDEXED":
            raise CourseReferenceError("REFERENCE_CONTEXT_CHANGED", "课程参考切片已变化。")
        if str(document.get("content_hash") or "") != reference.content_hash:
            raise CourseReferenceError("REFERENCE_CONTEXT_CHANGED", "课程参考版本已变化。")
        if str(chunk.get("chunk_hash") or "") != reference.chunk_hash:
            raise CourseReferenceError("REFERENCE_CONTEXT_CHANGED", "课程参考切片版本已变化。")
        text = _clean_reference_text(chunk.get("chunk_text"))
        if text[: len(reference.text)] != reference.text:
            raise CourseReferenceError("REFERENCE_CONTEXT_CHANGED", "课程参考正文已变化。")
        current.append(
            CourseReference(
                source_id=reference.source_id,
                document_id=reference.document_id,
                chunk_id=reference.chunk_id,
                title=_clean_reference_text(document.get("title") or document.get("filename"))[:300],
                text=reference.text,
                page=chunk.get("page") if isinstance(chunk.get("page"), int) else None,
                section=_clean_reference_text(chunk.get("section"))[:300],
                source_type=_clean_reference_text(chunk.get("source_type") or document.get("file_type"))[:100],
                content_hash=reference.content_hash,
                chunk_hash=reference.chunk_hash,
            )
        )
    return current


def _clean_reference_text(value: Any) -> str:
    text = str(value or "")
    safe = "".join(character if character >= " " or character in "\t\r\n" else " " for character in text)
    return " ".join(safe.split())
