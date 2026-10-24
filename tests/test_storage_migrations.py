import sqlite3
from pathlib import Path

import pytest

from backend.rag.document_parser import Document, Page
from backend.storage.logger import LearningLogger, _run_learning_event_generation_migration, sha256_text


def test_migration_preserves_first_phase_question_logs(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer_summary TEXT NOT NULL,
                module TEXT NOT NULL,
                intent TEXT NOT NULL,
                confidence REAL NOT NULL,
                abstained INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO questions (
                user_id, question, answer_summary, module, intent, confidence, abstained
            )
            VALUES ('u1', 'legacy question', 'legacy answer', '5G NAS', 'course_qa', 0.8, 0)
            """
        )

    logger = LearningLogger(db_path)
    analytics = logger.analytics()

    assert analytics["total_questions"] == 1
    assert analytics["module_distribution"]["5G NAS"] == 1
    assert logger.index_jobs() == []


def test_document_chunks_keep_location_and_delete_cascades(tmp_path):
    logger = LearningLogger(tmp_path / "test.db")
    logger.seed_demo_identity()
    course_id = logger.default_course_id()
    document = Document(
        source_path=Path("knowledge_base/sample.md"),
        filename="sample.md",
        file_type="md",
        title="Sample",
        text="Section A\nChunk body",
        pages=[
            Page(
                text="Section A\nChunk body",
                page=1,
                section="Section A",
                source_type="md",
            )
        ],
    )
    job_id = logger.create_index_job(
        filename=document.filename,
        content_hash=sha256_text(document.text),
        status="PENDING",
    )
    record = logger.save_document(
        document=document,
        content_hash=sha256_text(document.text),
        chunks=None,
        byte_size=len(document.text.encode("utf-8")),
        job_id=job_id,
        course_id=course_id,
    )
    chunks = logger.chunks_for_document(record["id"])
    jobs = logger.index_jobs(record["id"])

    assert len(chunks) == 1
    assert chunks[0]["page"] == 1
    assert chunks[0]["section"] == "Section A"
    assert chunks[0]["source_type"] == "md"
    assert jobs[0]["status"] == "PENDING"

    logger.delete_document(record["id"])

    assert logger.chunks_for_document(record["id"]) == []
    assert logger.index_jobs(record["id"]) == []
    assert logger.documents() == []


def test_failed_index_job_records_error_message(tmp_path):
    logger = LearningLogger(tmp_path / "test.db")
    job_id = logger.create_index_job(
        filename="bad.pdf",
        content_hash="deadbeef",
        status="PENDING",
    )

    logger.fail_index_job(job_id, "解析 PDF 失败：bad.pdf")
    jobs = logger.index_jobs()

    assert jobs[0]["status"] == "FAILED"
    assert jobs[0]["error_message"] == "解析 PDF 失败：bad.pdf"


def test_learning_event_generation_migration_preserves_old_rows_indexes_and_dedupe(tmp_path):
    db_path = tmp_path / "events-v16.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        conn.executemany("INSERT INTO schema_migrations(version,name) VALUES (?,?)", [(i, f"v{i}") for i in range(1, 17)])
        conn.execute("CREATE TABLE users(id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE courses(id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE classes(id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE learning_sessions(id INTEGER PRIMARY KEY, user_id INTEGER, course_id INTEGER, class_id INTEGER)")
        conn.execute("INSERT INTO users VALUES (1)")
        conn.execute("INSERT INTO courses VALUES (1)")
        conn.execute("INSERT INTO classes VALUES (1)")
        conn.execute("INSERT INTO learning_sessions VALUES (1,1,1,1)")
        conn.execute(
            """
            CREATE TABLE learning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL, course_id INTEGER NOT NULL, class_id INTEGER,
                event_type TEXT NOT NULL CHECK(event_type IN ('LOGIN','PAGE_VIEW','RESOURCE_SEARCH','RESOURCE_OPEN','QUESTION_ASKED','PRACTICE_GUIDE_GENERATED','INNOVATION_SESSION_STARTED','INNOVATION_QUESTION_ANSWERED','ASSIGNMENT_PROGRESS_UPDATED','STUCK_REPORTED','FEEDBACK_SUBMITTED')),
                page_name TEXT NOT NULL DEFAULT '', module TEXT NOT NULL DEFAULT '', resource_id TEXT,
                assignment_id TEXT, metadata_json TEXT NOT NULL DEFAULT '{}', dedupe_key TEXT UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES learning_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(course_id) REFERENCES courses(id) ON DELETE CASCADE,
                FOREIGN KEY(class_id) REFERENCES classes(id) ON DELETE SET NULL
            )
            """
        )
        conn.execute("CREATE INDEX idx_learning_events_user_time ON learning_events(user_id, created_at)")
        conn.execute("CREATE INDEX idx_learning_events_course_class ON learning_events(course_id, class_id)")
        conn.execute("INSERT INTO learning_events(session_id,user_id,course_id,class_id,event_type,dedupe_key) VALUES (1,1,1,1,'LOGIN','old-login')")
        conn.execute("UPDATE sqlite_sequence SET seq=7 WHERE name='learning_events'")

        _run_learning_event_generation_migration(conn)

        assert conn.execute("SELECT id,event_type,dedupe_key FROM learning_events").fetchall() == [(1, "LOGIN", "old-login")]
        conn.execute("INSERT INTO learning_events(session_id,user_id,course_id,class_id,event_type,dedupe_key) VALUES (1,1,1,1,'GENERAL_QUESTION_ASKED','generated')")
        assert conn.execute("SELECT id FROM learning_events WHERE dedupe_key='generated'").fetchone()[0] == 8
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO learning_events(session_id,user_id,course_id,class_id,event_type,dedupe_key) VALUES (1,1,1,1,'GENERAL_QUESTION_ASKED','generated')")
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(learning_events)")}
        assert {"idx_learning_events_user_time", "idx_learning_events_course_class"} <= indexes
        assert conn.execute("SELECT name FROM schema_migrations WHERE version=17").fetchone()[0] == "learning_event_generation_types"


def test_learning_event_generation_migration_rolls_back_on_foreign_key_verification_failure(tmp_path):
    db_path = tmp_path / "events-invalid-v16.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.execute("CREATE TABLE users(id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE courses(id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE classes(id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE learning_sessions(id INTEGER PRIMARY KEY)")
        conn.execute(
            """
            CREATE TABLE learning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL, course_id INTEGER NOT NULL, class_id INTEGER,
                event_type TEXT NOT NULL, page_name TEXT NOT NULL DEFAULT '', module TEXT NOT NULL DEFAULT '',
                resource_id TEXT, assignment_id TEXT, metadata_json TEXT NOT NULL DEFAULT '{}',
                dedupe_key TEXT UNIQUE, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES learning_sessions(id),
                FOREIGN KEY(user_id) REFERENCES users(id), FOREIGN KEY(course_id) REFERENCES courses(id),
                FOREIGN KEY(class_id) REFERENCES classes(id)
            )
            """
        )
        conn.execute("INSERT INTO learning_events(session_id,user_id,course_id,event_type) VALUES (99,99,99,'LOGIN')")

        with pytest.raises(RuntimeError, match="foreign key"):
            _run_learning_event_generation_migration(conn)

        assert conn.execute("SELECT event_type FROM learning_events").fetchone()[0] == "LOGIN"
        assert conn.execute("SELECT 1 FROM schema_migrations WHERE version=17").fetchone() is None
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learning_events_v17'").fetchone() is None
