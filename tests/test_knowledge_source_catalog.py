from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from backend.auth.seed import seed_shared_textbooks
from backend.learning.source_catalog import CatalogValidationError, load_source_catalog
from backend.rag.document_parser import parse_document
from backend.rag.loader import load_knowledge_chunks
from backend.storage.logger import LearningLogger, sha256_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = REPOSITORY_ROOT / "knowledge_sources" / "shared"
CATALOG_PATH = SHARED_ROOT / "catalog.json"


class _RecordingVectorStore:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.records: dict[str, tuple[int, int]] = {}
        self.add_calls = 0

    def add_chunks(self, chunks):
        self.add_calls += 1
        if self.fail:
            raise RuntimeError("vector unavailable")
        for chunk in chunks:
            self.records[str(int(chunk["id"]))] = (int(chunk["document_id"]), int(chunk["id"]))
        return len(chunks)

    def contains_chunks(self, chunks):
        expected = {
            str(int(chunk["id"])): (int(chunk["document_id"]), int(chunk["id"]))
            for chunk in chunks
        }
        return all(self.records.get(key) == value for key, value in expected.items())


def test_shared_catalog_validates_nine_independent_approved_sources():
    catalog = load_source_catalog(CATALOG_PATH)

    assert catalog.schema_version == 1
    assert catalog.catalog_state == "draft"
    assert len(catalog.entries) == 9
    assert len({entry.source_id for entry in catalog.entries}) == 9
    assert len({entry.path.casefold() for entry in catalog.entries}) == 9
    assert {entry.topic for entry in catalog.entries} == {
        "协议通信",
        "密码身份",
        "漏洞CVSS",
        "系统应用防护",
        "日志响应",
        "5G NAS",
        "区块链与隐蔽通信",
        "LLM安全",
        "实验设计",
    }
    assert all(entry.revision == "1.0.0" for entry in catalog.entries)
    assert all(entry.review_state == "approved" for entry in catalog.entries)
    assert all(entry.reviewer == "当前对话用户（2026-10-18明确确认）" for entry in catalog.entries)
    assert all(entry.reviewed_at == "2026-10-18T09:57:52Z" for entry in catalog.entries)
    assert all(entry.reference_urls and entry.licence_note for entry in catalog.entries)
    assert len({sha256_bytes((SHARED_ROOT / entry.path).read_bytes()) for entry in catalog.entries}) == 9


def test_each_catalog_source_uses_existing_parser_and_has_bounded_structure():
    catalog = load_source_catalog(CATALOG_PATH)

    for entry in catalog.entries:
        document = parse_document(SHARED_ROOT / entry.path)
        assert document.title == entry.title
        assert 700 <= entry.char_count <= 12_000
        assert entry.section_count >= 6
        assert "review_state: `approved`" in document.text
        assert "reviewer: `当前对话用户（2026-10-18明确确认）`" in document.text
        assert "动态事实" in document.text or "动态漏洞事实" in document.text
        assert "仍须官方复核" in document.text


def test_existing_loader_reads_each_approved_source_without_importing_or_indexing():
    catalog = load_source_catalog(CATALOG_PATH)
    chunks = load_knowledge_chunks(SHARED_ROOT)

    assert {chunk.source for chunk in chunks} == {entry.path for entry in catalog.entries}
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(len(chunk.text) <= 700 for chunk in chunks)
    assert SHARED_ROOT.parent.name == "knowledge_sources"
    assert SHARED_ROOT.name == "shared"


def test_catalog_rejects_duplicate_json_keys(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        '{"schema_version":1,"schema_version":1,"catalog_state":"draft","entries":[]}',
        encoding="utf-8",
    )

    with pytest.raises(CatalogValidationError, match="重复 JSON 键"):
        load_source_catalog(catalog_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["entries"].append(dict(data["entries"][0])), "重复 source_id"),
        (lambda data: data["entries"][0].update(path="../escape.md"), "安全的相对 Markdown 路径"),
        (lambda data: data["entries"][0].update(path="C:/escape.md"), "安全的相对 Markdown 路径"),
        (lambda data: data["entries"][0].update(path=r"folder\escape.md"), "安全的相对 Markdown 路径"),
        (lambda data: data["entries"][0].update(reviewer=None), "真实审核记录"),
        (lambda data: data["entries"][0].update(reference_urls=[]), "非空数组"),
    ],
)
def test_catalog_rejects_duplicate_unsafe_unreviewed_or_unlicensed_entries(tmp_path, mutation, message):
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    for entry in data["entries"]:
        shutil.copy2(SHARED_ROOT / entry["path"], tmp_path / entry["path"])
    mutation(data)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(CatalogValidationError, match=message):
        load_source_catalog(catalog_path)


def test_shared_textbook_bootstrap_indexes_publishes_and_is_idempotent(tmp_path):
    logger = LearningLogger(tmp_path / "app.db")
    vector = _RecordingVectorStore()
    knowledge_root = tmp_path / "knowledge"

    first = seed_shared_textbooks(
        logger,
        vector_store=vector,
        catalog_path=CATALOG_PATH,
        knowledge_base_dir=knowledge_root,
    )
    before_counts = (
        len(logger.documents(course_id=int(first["course"]["id"]))),
        sum(len(logger.chunks_for_document(item["document_id"])) for item in first["materials"]),
        len(logger.index_jobs()),
        len(vector.records),
        vector.add_calls,
    )
    second = seed_shared_textbooks(
        logger,
        vector_store=vector,
        catalog_path=CATALOG_PATH,
        knowledge_base_dir=knowledge_root,
    )
    after_counts = (
        len(logger.documents(course_id=int(second["course"]["id"]))),
        sum(len(logger.chunks_for_document(item["document_id"])) for item in second["materials"]),
        len(logger.index_jobs()),
        len(vector.records),
        vector.add_calls,
    )

    assert first["course"]["slug"] == "demo-course"
    assert len(first["materials"]) == len(second["materials"]) == 9
    assert before_counts == after_counts
    assert before_counts[0] == before_counts[2] == 9
    for material in second["materials"]:
        document = logger.document(material["document_id"])
        assert document is not None
        source_path = Path(document["source_path"]).resolve()
        assert source_path.parent == (knowledge_root / f"course-{first['course']['id']}").resolve()
        assert SHARED_ROOT.resolve() not in source_path.parents
        assert document["index_status"] == "INDEXED"
        assert document["published_to_students"] is True
        assert document["published_by"] is None
        assert document["publication_last_action"]["action"] == "bootstrap_publish"
        assert document["publication_last_action"]["actor"] is None

    with pytest.raises(ValueError, match="DOCUMENT_ACTOR_REQUIRED"):
        logger.set_document_publication(first["materials"][0]["document_id"], published=True, actor_id=None)


def test_shared_textbook_bootstrap_rejects_unapproved_and_filename_conflicts(tmp_path):
    copied_root = tmp_path / "shared"
    copied_root.mkdir()
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    for entry in data["entries"]:
        shutil.copy2(SHARED_ROOT / entry["path"], copied_root / entry["path"])
    data["entries"][0].update(review_state="pending_review", reviewer=None, reviewed_at=None)
    copied_catalog = copied_root / "catalog.json"
    copied_catalog.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    logger = LearningLogger(tmp_path / "unapproved.db")

    with pytest.raises(RuntimeError, match="NINE_APPROVED"):
        seed_shared_textbooks(
            logger,
            vector_store=_RecordingVectorStore(),
            catalog_path=copied_catalog,
            knowledge_base_dir=tmp_path / "unapproved-knowledge",
        )

    conflict_logger = LearningLogger(tmp_path / "conflict.db")
    course_id = conflict_logger.default_course_id()
    conflict_root = tmp_path / "conflict-knowledge"
    target = conflict_root / f"course-{course_id}" / data["entries"][0]["path"]
    target.parent.mkdir(parents=True)
    target.write_text("different content", encoding="utf-8")
    with pytest.raises(RuntimeError, match="FILE_CONFLICT"):
        seed_shared_textbooks(
            conflict_logger,
            vector_store=_RecordingVectorStore(),
            catalog_path=CATALOG_PATH,
            knowledge_base_dir=conflict_root,
        )


def test_shared_textbook_vector_failure_stays_failed_and_unpublished(tmp_path):
    logger = LearningLogger(tmp_path / "app.db")
    with pytest.raises(RuntimeError, match="vector unavailable"):
        seed_shared_textbooks(
            logger,
            vector_store=_RecordingVectorStore(fail=True),
            catalog_path=CATALOG_PATH,
            knowledge_base_dir=tmp_path / "knowledge",
        )

    documents = logger.documents(course_id=logger.default_course_id())
    assert len(documents) == 1
    assert documents[0]["index_status"] == "FAILED"
    assert documents[0]["published_to_students"] is False
    jobs = logger.index_jobs(int(documents[0]["id"]))
    assert len(jobs) == 1
    assert jobs[0]["status"] == "FAILED"
