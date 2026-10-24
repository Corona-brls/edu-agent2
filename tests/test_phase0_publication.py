from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from backend.rag.document_locks import DocumentBusyError, document_locks
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import HybridRetriever
from backend.resources.portal import ResourcePortalQuery, ResourcePortalService
from backend.storage.logger import LearningLogger, sha256_bytes


def _course(storage: LearningLogger, slug: str) -> int:
    with storage._connect() as conn:
        return int(
            conn.execute(
                "INSERT INTO courses (slug, name, description, assessment_method) VALUES (?, ?, '', '')",
                (slug, slug),
            ).lastrowid
        )


def _document(storage: LearningLogger, course_id: int, marker: str, *, metadata=None, status="INDEXED") -> dict:
    parsed = Document(
        source_path=Path(f"course-{course_id}/{marker}.md"),
        filename=f"{marker}.md",
        file_type=".md",
        title=marker,
        text=marker,
        pages=[Page(text=marker, page=1, section="phase0", source_type="markdown")],
        metadata=metadata or {},
    )
    return storage.save_document(
        document=parsed,
        content_hash=sha256_bytes(marker.encode()),
        chunks=[{"chunk_text": marker, "page": 1, "section": "phase0", "source_type": "markdown"}],
        byte_size=len(marker.encode()),
        index_status=status,
        course_id=course_id,
    )


def test_reserved_metadata_cannot_self_publish_and_lists_return_projection(tmp_path):
    storage = LearningLogger(tmp_path / "publication.db")
    course_id = _course(storage, "publication-reserved")
    saved = _document(
        storage,
        course_id,
        "MALICIOUS_METADATA_MARKER",
        metadata={
            "published_to_students": True,
            "published_by": 999,
            "published_at": "2099-01-01T00:00:00Z",
            "last_action": "publish",
        },
    )

    listed = next(item for item in storage.documents(course_id=course_id) if item["id"] == saved["id"])

    assert listed["published_to_students"] is False
    assert listed["published_by"] is None
    assert listed["published_at"] is None
    assert listed["metadata"]["published_to_students"] is False


def test_publish_unpublish_and_reindex_clear_student_visibility(tmp_path, monkeypatch):
    storage = LearningLogger(tmp_path / "visibility.db")
    course_id = _course(storage, "publication-flow")
    saved = _document(storage, course_id, "PUBLICATION_FLOW_MARKER")
    portal = ResourcePortalService(storage, course_id=course_id)
    query = ResourcePortalQuery(keyword="PUBLICATION_FLOW_MARKER", resource_type="course_document")

    assert portal.search(query)["items"] == []

    published = storage.set_document_publication(int(saved["id"]), published=True, actor_id=17)
    assert published["published_to_students"] is True
    assert published["published_by"] == 17
    assert portal.search(query)["items"][0]["resource_id"] == f"document:{saved['id']}"

    storage.set_document_publication(int(saved["id"]), published=False, actor_id=17)
    assert portal.search(query)["items"] == []

    storage.set_document_publication(int(saved["id"]), published=True, actor_id=17)
    storage.update_document_index_status(int(saved["id"]), "EMBEDDING")
    reindexed = storage.document(int(saved["id"]))
    assert reindexed["published_to_students"] is False
    assert reindexed["publication_last_action"]["action"] == "reindex"
    assert portal.search(query)["items"] == []


def test_reindex_start_never_commits_active_job_with_old_published_content(tmp_path, monkeypatch):
    from backend import main as backend_main
    from backend.storage import logger as logger_module

    db_path = tmp_path / "atomic-reindex.db"
    storage = LearningLogger(db_path)
    course_id = _course(storage, "atomic-reindex")
    saved = _document(storage, course_id, "ATOMIC_REINDEX_MARKER")
    document_id = int(saved["id"])
    storage.set_document_publication(document_id, published=True, actor_id=17)

    snapshots: list[dict] = []

    def snapshot(label: str) -> None:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            document = conn.execute(
                "SELECT index_status, metadata_json FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            active_jobs = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM index_jobs
                    WHERE document_id = ?
                      AND status IN ('PENDING', 'PARSING', 'CHUNKING', 'EMBEDDING')
                    """,
                    (document_id,),
                ).fetchone()[0]
            )
            indexed_chunks = int(
                conn.execute(
                    "SELECT COUNT(*) FROM document_chunks WHERE document_id = ? AND index_status = 'INDEXED'",
                    (document_id,),
                ).fetchone()[0]
            )
        metadata = json.loads(document["metadata_json"] or "{}")
        snapshots.append(
            {
                "label": label,
                "active_jobs": active_jobs,
                "document_status": document["index_status"],
                "published": metadata.get("published_to_students") is True,
                "indexed_chunks": indexed_chunks,
            }
        )

    original_update_job = storage.update_index_job

    def observed_update_job(job_id, status, document_id=None, error_message=None):
        original_update_job(job_id, status, document_id=document_id, error_message=error_message)
        if status == "EMBEDDING":
            snapshot("after-separate-job-commit")

    original_publication_metadata = logger_module._publication_metadata_json

    def observed_publication_metadata(*args, **kwargs):
        if kwargs.get("action") == "reindex":
            snapshot("while-reindex-state-is-being-published")
        return original_publication_metadata(*args, **kwargs)

    class FakeVectorStore:
        def delete_by_document(self, target_document_id):
            assert target_document_id == document_id

        def add_chunks(self, chunks):
            return len(chunks)

    monkeypatch.setattr(storage, "update_index_job", observed_update_job)
    monkeypatch.setattr(logger_module, "_publication_metadata_json", observed_publication_metadata)
    monkeypatch.setattr(backend_main, "learning_logger", storage)
    monkeypatch.setattr(backend_main, "ChromaVectorStore", FakeVectorStore)
    monkeypatch.setattr(backend_main, "_require_document_management_access", lambda *_: None)

    result = backend_main._reindex_knowledge_document_locked(document_id, {"id": 17, "role": "teacher"})

    assert result["status"] == "INDEXED"
    assert snapshots
    assert not any(
        item["active_jobs"]
        and item["document_status"] == "INDEXED"
        and item["published"]
        and item["indexed_chunks"] > 0
        for item in snapshots
    )


def test_unpublished_and_cross_course_chunks_cannot_starve_legal_keyword_result(tmp_path, monkeypatch):
    storage = LearningLogger(tmp_path / "topk.db")
    selected = _course(storage, "selected")
    foreign = _course(storage, "foreign")
    marker = "DENSE_SIMILAR_MARKER"
    for index in range(30):
        _document(storage, selected, f"{marker}_UNPUBLISHED_{index}")
        _document(storage, foreign, f"{marker}_FOREIGN_{index}")
    legal = _document(storage, selected, f"{marker}_LEGAL")
    storage.set_document_publication(int(legal["id"]), published=True, actor_id=1)
    monkeypatch.setattr(
        "backend.rag.hybrid_retriever._bounded_vector_store_init",
        lambda: (_ for _ in ()).throw(RuntimeError("vector unavailable")),
    )

    response = HybridRetriever(storage=storage).retrieve(marker, top_k=1, course_id=selected)

    assert len(response.results) == 1
    assert response.results[0].document_id == int(legal["id"])
    assert "LEGAL" in response.results[0].snippet


def test_same_document_operations_are_nonblocking_and_release_after_error():
    document_id = 77123
    with document_locks.acquire(document_id):
        with pytest.raises(DocumentBusyError):
            with document_locks.acquire(document_id):
                pass
    with document_locks.acquire(document_id):
        pass


def test_version_switch_preserves_ids_and_old_version_until_explicit_delete(tmp_path):
    storage = LearningLogger(tmp_path / "versions.db")
    course_id = _course(storage, "version-course")
    version_1 = _document(storage, course_id, "network_basics_v1")
    version_2 = _document(storage, course_id, "network_basics_v2")
    version_1_id = int(version_1["id"])
    version_2_id = int(version_2["id"])
    version_1_chunk = int(storage.chunks_for_document(version_1_id)[0]["id"])
    version_2_chunk = int(storage.chunks_for_document(version_2_id)[0]["id"])
    portal = ResourcePortalService(storage, course_id=course_id)

    storage.set_document_publication(version_1_id, published=True, actor_id=7)
    course_documents = ResourcePortalQuery(resource_type="course_document", page_size=50)
    assert {item["resource_id"] for item in portal.search(course_documents)["items"]} == {
        f"document:{version_1_id}"
    }

    storage.set_document_publication(version_2_id, published=True, actor_id=7)
    storage.set_document_publication(version_1_id, published=False, actor_id=7)
    assert {item["resource_id"] for item in portal.search(course_documents)["items"]} == {
        f"document:{version_2_id}"
    }
    assert int(storage.chunks_for_document(version_1_id)[0]["id"]) == version_1_chunk
    assert int(storage.chunks_for_document(version_2_id)[0]["id"]) == version_2_chunk

    storage.delete_document(version_1_id)
    assert storage.document(version_1_id) is None
    assert storage.document(version_2_id) is not None
    assert int(storage.chunks_for_document(version_2_id)[0]["id"]) == version_2_chunk
    assert portal.resource_exists(f"document:{version_2_id}") is True
