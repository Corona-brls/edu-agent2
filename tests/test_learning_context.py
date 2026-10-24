from __future__ import annotations

from pathlib import Path

import pytest

from backend.learning.context import CourseReferenceError, load_course_references
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import RetrievalResponse, RetrievalResult
from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.vector_store import ChromaVectorStore
from backend.storage.logger import LearningLogger, sha256_bytes


def _course(storage: LearningLogger, slug: str) -> int:
    with storage._connect() as conn:
        return int(
            conn.execute(
                "INSERT INTO courses (slug, name, description, assessment_method) VALUES (?, ?, '', '')",
                (slug, slug),
            ).lastrowid
        )


def _document(
    storage: LearningLogger,
    course_id: int,
    marker: str,
    *,
    status: str = "INDEXED",
    published: bool = False,
) -> tuple[int, int]:
    text = f"{marker} full canonical chunk text with details beyond a retrieval snippet."
    document = Document(
        source_path=Path(f"course-{course_id}/{marker}.md"),
        filename=f"{marker}.md",
        file_type=".md",
        title=f"{marker} title",
        text=text,
        pages=[Page(text=text, page=2, section="reference", source_type="markdown")],
    )
    saved = storage.save_document(
        document=document,
        content_hash=sha256_bytes(text.encode("utf-8")),
        chunks=[{"chunk_text": text, "page": 2, "section": "reference", "source_type": "markdown"}],
        byte_size=len(text.encode("utf-8")),
        index_status=status,
        course_id=course_id,
    )
    document_id = int(saved["id"])
    if published:
        storage.set_document_publication(document_id, published=True, actor_id=1)
    chunk_id = int(storage.chunks_for_document(document_id)[0]["id"])
    return document_id, chunk_id


def _result(document_id: int, chunk_id: int, marker: str) -> RetrievalResult:
    return RetrievalResult(
        document_id=document_id,
        chunk_id=chunk_id,
        source="untrusted-retriever-name.md",
        title="untrusted retriever title",
        snippet=f"{marker} short snippet",
        page=99,
        section="untrusted",
        source_type="markdown",
        keyword_score=1.0,
        vector_score=0.0,
        final_score=1.0,
    )


class _Retriever:
    def __init__(self, *results: RetrievalResult) -> None:
        self.results = list(results)
        self.calls: list[dict] = []

    def retrieve(self, query: str, *, course_id: int, top_k: int) -> RetrievalResponse:
        self.calls.append({"query": query, "course_id": course_id, "top_k": top_k})
        return RetrievalResponse(results=self.results, abstained=not self.results, threshold=0.0)


class _ConstantEmbedding:
    def name(self) -> str:
        return "default"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in input]

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)


def test_course_context_revalidates_visibility_and_projects_stable_database_identity(tmp_path):
    storage = LearningLogger(tmp_path / "context.db")
    course_id = _course(storage, "context-course")
    visible_doc, visible_chunk = _document(storage, course_id, "VISIBLE_MARKER", published=True)
    hidden_doc, hidden_chunk = _document(storage, course_id, "HIDDEN_MARKER")
    retriever = _Retriever(
        _result(hidden_doc, hidden_chunk, "HIDDEN_MARKER"),
        _result(visible_doc, visible_chunk, "VISIBLE_MARKER"),
        _result(visible_doc, visible_chunk, "VISIBLE_MARKER"),
    )

    first = load_course_references(
        storage,
        retriever,
        course_id=course_id,
        query="VISIBLE_MARKER",
        top_k=5,
        max_chars=500,
    )
    second = load_course_references(
        storage,
        retriever,
        course_id=course_id,
        query="VISIBLE_MARKER",
        top_k=5,
        max_chars=500,
    )

    assert retriever.calls == [
        {"query": "VISIBLE_MARKER", "course_id": course_id, "top_k": 5},
        {"query": "VISIBLE_MARKER", "course_id": course_id, "top_k": 5},
    ]
    assert len(first) == 1
    assert first == second
    assert first[0].source_id == f"rag:{visible_doc}:{visible_chunk}"
    assert first[0].document_id == visible_doc
    assert first[0].chunk_id == visible_chunk
    assert first[0].title == "VISIBLE_MARKER title"
    assert "full canonical chunk text" in first[0].text
    assert "untrusted" not in first[0].title


def test_course_context_fails_closed_for_cross_course_result(tmp_path):
    storage = LearningLogger(tmp_path / "cross-course.db")
    selected_course = _course(storage, "selected-course")
    other_course = _course(storage, "other-course")
    document_id, chunk_id = _document(storage, other_course, "FOREIGN_MARKER", published=True)

    with pytest.raises(CourseReferenceError, match="REFERENCE_SCOPE_VIOLATION") as captured:
        load_course_references(
            storage,
            _Retriever(_result(document_id, chunk_id, "FOREIGN_MARKER")),
            course_id=selected_course,
            query="FOREIGN_MARKER",
            top_k=3,
            max_chars=500,
        )

    assert captured.value.error_code == "REFERENCE_SCOPE_VIOLATION"


def test_course_context_rejects_missing_scope_and_applies_total_text_budget(tmp_path):
    storage = LearningLogger(tmp_path / "budget.db")
    course_id = _course(storage, "budget-course")
    first_doc, first_chunk = _document(storage, course_id, "FIRST_MARKER", published=True)
    second_doc, second_chunk = _document(storage, course_id, "SECOND_MARKER", published=True)
    retriever = _Retriever(
        _result(first_doc, first_chunk, "FIRST_MARKER"),
        _result(second_doc, second_chunk, "SECOND_MARKER"),
    )

    with pytest.raises(CourseReferenceError, match="COURSE_REQUIRED"):
        load_course_references(
            storage,
            retriever,
            course_id=None,
            query="marker",
            top_k=2,
            max_chars=80,
        )

    references = load_course_references(
        storage,
        retriever,
        course_id=course_id,
        query="marker",
        top_k=2,
        max_chars=80,
    )

    assert references
    assert sum(len(item.text) for item in references) <= 80
    assert references[0].source_id == f"rag:{first_doc}:{first_chunk}"


def test_real_hybrid_context_is_not_starved_by_hidden_similar_chroma_rows(tmp_path):
    storage = LearningLogger(tmp_path / "real-context.db")
    selected_course = _course(storage, "real-selected")
    other_course = _course(storage, "real-other")
    for index in range(20):
        _document(storage, selected_course, f"UNPUBLISHED_{index}")
        _document(storage, other_course, f"FOREIGN_{index}", published=True)
    visible_doc, visible_chunk = _document(
        storage,
        selected_course,
        "AUTHORIZED_CANONICAL_BODY",
        published=True,
    )
    vector_store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="course_context",
        embedding_function=_ConstantEmbedding(),
    )
    vector_store.add_chunks(storage.document_chunks())
    retriever = HybridRetriever(
        storage=storage,
        vector_store=vector_store,
        keyword_weight=0.0,
        vector_weight=1.0,
        min_score=0.0,
    )

    references = load_course_references(
        storage,
        retriever,
        course_id=selected_course,
        query="semantic-only-query",
        top_k=1,
        max_chars=500,
    )

    assert [item.source_id for item in references] == [f"rag:{visible_doc}:{visible_chunk}"]
