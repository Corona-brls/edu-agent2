"""SQLite storage for learning logs and uploaded documents."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.connectors.schemas import ExternalResource
from backend.rag.document_parser import Document
from backend.rag.visibility import publication_projection, sanitized_new_document_metadata
from backend.resources.normalizer import canonical_authors, content_hash as current_resource_content_hash

DEFAULT_DB_PATH = Path("backend/data/app.db")
VALID_INDEX_STATUSES = {"PENDING", "PARSING", "CHUNKING", "EMBEDDING", "INDEXED", "FAILED"}


class LearningLogger:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        configured = os.getenv("SQLITE_DB_PATH", "").strip()
        if db_path == DEFAULT_DB_PATH and configured:
            self.db_path = Path(configured).expanduser().resolve()
        else:
            self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer_summary TEXT NOT NULL,
                    module TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    tool_name TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL,
                    abstained INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            _ensure_column(conn, "questions", "tool_name", "TEXT NOT NULL DEFAULT ''")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    score REAL,
                    snippet TEXT NOT NULL,
                    FOREIGN KEY(question_id) REFERENCES questions(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS faq_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_label TEXT NOT NULL,
                    representative_question TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    module TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._apply_migrations(conn)

    def _apply_migrations(self, conn: sqlite3.Connection) -> None:
        _run_migration(conn, 1, "document_metadata_tables", _migration_document_metadata_tables)
        _run_migration(conn, 2, "document_chunk_location_fields", _migration_document_chunk_location_fields)
        _run_migration(conn, 3, "agent_run_trace_tables", _migration_agent_run_trace_tables)
        _run_migration(conn, 4, "feedback_and_model_metrics", _migration_feedback_and_model_metrics)
        _run_migration(conn, 5, "phase3_identity_course_context", _migration_phase3_identity_course_context)
        _run_migration(conn, 6, "learning_sessions_events", _migration_learning_sessions_events)
        _run_migration(conn, 7, "external_resource_connector_cache", _migration_external_resource_connector_cache)
        _run_migration(conn, 8, "innovation_sessions", _migration_innovation_sessions)
        _run_migration(conn, 9, "question_semantic_clusters", _migration_question_semantic_clusters)
        _run_migration(conn, 10, "assignments_progress_stuck", _migration_assignments_progress_stuck)
        _run_migration(conn, 11, "course_scoped_documents", _migration_course_scoped_documents)
        _run_migration(conn, 12, "user_token_version", _migration_user_token_version)
        _run_migration(conn, 13, "question_cluster_class_scope_repair", _migration_question_cluster_class_scope_repair)
        _run_migration(conn, 14, "resource_query_cache_mapping", _migration_resource_query_cache_mapping)
        _run_migration(conn, 15, "persistent_question_cluster_jobs", _migration_persistent_question_cluster_jobs)
        _run_migration(conn, 16, "external_resource_provenance", _migration_external_resource_provenance)
        _run_learning_event_generation_migration(conn)

    def log_question(
        self,
        user_id: str,
        question: str,
        result: dict[str, Any],
        *,
        allow_demo_defaults: bool = False,
    ) -> int:
        citations = result.get("citations", [])
        course_id = result.get("course_id")
        class_id = result.get("class_id")
        if course_id is None:
            if not allow_demo_defaults:
                raise ValueError("course_id is required for a protected question")
            course_id = self.default_course_id()
            class_id = self.default_class_id() if class_id is None else class_id
        if class_id is not None and not self.class_belongs_to_course(int(class_id), int(course_id)):
            raise ValueError("class_id does not belong to course_id")
        session_id = result.get("session_id")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO questions (
                    user_id, question, answer_summary, module, intent, tool_name, confidence, abstained,
                    course_id, class_id, session_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    question,
                    _summary(result.get("answer", "")),
                    result.get("module", "unknown"),
                    result.get("intent", "unknown"),
                    (result.get("tool") or {}).get("name", ""),
                    float(result.get("confidence", 0.0)),
                    1 if result.get("abstained") else 0,
                    course_id,
                    class_id,
                    session_id,
                ),
            )
            question_id = int(cursor.lastrowid)
            run_id = result.get("run_id")
            if run_id:
                conn.execute("UPDATE questions SET run_id = ? WHERE id = ?", (run_id, question_id))
            conn.executemany(
                """
                INSERT INTO retrieval_logs (question_id, source, title, score, snippet)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        question_id,
                        item.get("source", ""),
                        item.get("title", ""),
                        item.get("score"),
                        item.get("snippet", ""),
                    )
                    for item in citations
                ],
            )
        return question_id

    def analytics(self, limit: int = 10) -> dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
            abstained_count = conn.execute(
                "SELECT COUNT(*) FROM questions WHERE abstained = 1"
            ).fetchone()[0]
            average_confidence = conn.execute(
                "SELECT AVG(confidence) FROM questions"
            ).fetchone()[0]
            module_rows = conn.execute(
                """
                SELECT module, COUNT(*) AS count
                FROM questions
                GROUP BY module
                ORDER BY count DESC, module ASC
                """
            ).fetchall()
            recent_rows = conn.execute(
                """
                SELECT user_id, question, module, intent, confidence, abstained
                FROM questions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return {
            "total_questions": int(total),
            "abstained_count": int(abstained_count),
            "average_confidence": round(float(average_confidence or 0.0), 2),
            "abstain_rate": round((int(abstained_count) / int(total)) if total else 0.0, 4),
            "module_distribution": {row["module"]: int(row["count"]) for row in module_rows},
            "recent_questions": [
                {
                    "user_id": row["user_id"],
                    "question": _preview(row["question"]),
                    "module": row["module"],
                    "intent": row["intent"],
                    "confidence": float(row["confidence"]),
                    "abstained": bool(row["abstained"]),
                }
                for row in recent_rows
            ],
            **self._analytics_v2(),
        }

    def faq(self, limit: int = 20) -> dict[str, list[dict[str, Any]]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT question, COUNT(*) AS count, module, intent
                FROM questions
                GROUP BY question
                ORDER BY count DESC, MAX(id) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return {
            "items": [
                {
                    "representative_question": row["question"],
                    "count": int(row["count"]),
                    "module": row["module"],
                    "intent": row["intent"],
                }
                for row in rows
            ]
        }

    def retrieval_logs_for_question(self, question_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source, title, score, snippet
                FROM retrieval_logs
                WHERE question_id = ?
                ORDER BY id ASC
                """,
                (question_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_agent_run(
        self,
        *,
        run_id: str,
        user_id: str,
        goal: str,
        intent: str,
        status: str,
        max_steps: int,
        max_tool_calls: int,
        state: dict[str, Any],
        question_id: int | None = None,
        course_id: int | None = None,
        class_id: int | None = None,
        session_id: str | None = None,
    ) -> None:
        if course_id is None:
            if not str(user_id).isdigit():
                course_id = self.default_course_id()
                class_id = self.default_class_id() if class_id is None else class_id
                state = dict(state)
                state["course_id"] = course_id
                state["class_id"] = class_id
            else:
                raise ValueError("course_id is required for a protected Agent run")
        state_course_id = state.get("course_id")
        state_class_id = state.get("class_id")
        if state_course_id is None or int(state_course_id) != int(course_id):
            raise ValueError("Agent state course_id must match the persisted course_id")
        if state_class_id is not None:
            if class_id is None or int(state_class_id) != int(class_id):
                raise ValueError("Agent state class_id must match the persisted class_id")
        elif class_id is not None:
            raise ValueError("Agent state must explicitly contain the persisted class_id")
        if class_id is not None and not self.class_belongs_to_course(int(class_id), int(course_id)):
            raise ValueError("class_id does not belong to course_id")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_runs (
                    run_id, question_id, user_id, goal, intent, status,
                    current_step, max_steps, tool_call_count, max_tool_calls,
                    state_json, stop_reason, course_id, class_id, session_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    question_id,
                    user_id,
                    goal,
                    intent,
                    status,
                    int(state.get("current_step", 0)),
                    max_steps,
                    int(state.get("tool_call_count", 0)),
                    max_tool_calls,
                    _json_dumps(state),
                    state.get("stop_reason"),
                    course_id,
                    class_id,
                    session_id,
                ),
            )

    def update_agent_run(
        self,
        *,
        run_id: str,
        status: str,
        state: dict[str, Any],
        stop_reason: str | None = None,
        final_answer: str | None = None,
        pending_question: str | None = None,
        question_id: int | None = None,
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT course_id, class_id FROM agent_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Agent run does not exist")
            state_course_id = state.get("course_id")
            state_class_id = state.get("class_id")
            if state_course_id is None or int(state_course_id) != int(row["course_id"] or 0):
                raise ValueError("Agent state course_id does not match persisted context")
            if row["class_id"] is None:
                if state_class_id is not None:
                    raise ValueError("Agent state class_id does not match persisted context")
            elif state_class_id is None or int(state_class_id) != int(row["class_id"]):
                raise ValueError("Agent state class_id does not match persisted context")
            conn.execute(
                """
                UPDATE agent_runs
                SET question_id = COALESCE(?, question_id),
                    status = ?,
                    current_step = ?,
                    tool_call_count = ?,
                    state_json = ?,
                    pending_question = ?,
                    stop_reason = ?,
                    final_answer = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
                """,
                (
                    question_id,
                    status,
                    int(state.get("current_step", 0)),
                    int(state.get("tool_call_count", 0)),
                    _json_dumps(state),
                    pending_question,
                    stop_reason,
                    final_answer,
                    run_id,
                ),
            )

    def agent_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, question_id, user_id, goal, intent, status, current_step,
                       max_steps, tool_call_count, max_tool_calls, state_json,
                       course_id, class_id, session_id,
                       pending_question, final_answer, stop_reason, created_at, updated_at
                FROM agent_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        raw_state = result.pop("state_json")
        context_status = "valid"
        try:
            state = json.loads(raw_state or "{}")
        except (TypeError, json.JSONDecodeError):
            state = {}
            context_status = "invalid"
        if not isinstance(state, dict):
            state = {}
            context_status = "invalid"
        if context_status == "valid":
            raw_course_id = state.get("course_id")
            if raw_course_id is None:
                context_status = "missing"
            else:
                try:
                    if int(raw_course_id) <= 0 or int(raw_course_id) != int(result.get("course_id") or 0):
                        context_status = "invalid"
                except (TypeError, ValueError):
                    context_status = "invalid"
            column_class_id = result.get("class_id")
            state_has_class = "class_id" in state and state.get("class_id") is not None
            if context_status == "valid" and column_class_id is not None and not state_has_class:
                context_status = "missing"
            if context_status == "valid" and state_has_class:
                try:
                    if column_class_id is None or int(state["class_id"]) != int(column_class_id):
                        context_status = "invalid"
                    elif not self.class_belongs_to_course(int(column_class_id), int(result["course_id"])):
                        context_status = "invalid"
                except (TypeError, ValueError):
                    context_status = "invalid"
        result["state"] = state
        result["context_status"] = context_status
        return result

    def agent_attempt_checkpoint(self, run_id: str) -> dict[str, Any]:
        """Capture the exact persisted run and audit watermarks before one attempt."""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, question_id, status, current_step, tool_call_count,
                       state_json, pending_question, final_answer, stop_reason
                FROM agent_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Agent run does not exist")
            checkpoint = dict(row)
            for key, table in (
                ("step_watermark", "agent_steps"),
                ("trace_watermark", "agent_traces"),
                ("model_call_watermark", "model_calls"),
                ("tool_call_watermark", "tool_calls"),
            ):
                value = conn.execute(
                    f"SELECT COALESCE(MAX(id), 0) FROM {table} WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0]
                checkpoint[key] = int(value or 0)
        return checkpoint

    def close_rejected_agent_attempt(
        self,
        *,
        run_id: str,
        checkpoint: dict[str, Any],
        error_code: str,
        trace_id: str,
        restore_run: bool,
    ) -> None:
        """Atomically fail a new run or restore a continue attempt and redact its audit."""

        if str(checkpoint.get("run_id") or "") != str(run_id):
            raise ValueError("Agent attempt checkpoint does not match run_id")
        raw_state = checkpoint.get("state_json")
        try:
            checkpoint_state = json.loads(raw_state or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Agent attempt checkpoint state is invalid") from exc
        if not isinstance(checkpoint_state, dict):
            raise ValueError("Agent attempt checkpoint state is invalid")

        if restore_run:
            target_status = str(checkpoint["status"])
            target_state_json = str(raw_state)
            target_pending = checkpoint.get("pending_question")
            target_stop_reason = checkpoint.get("stop_reason")
            target_final_answer = checkpoint.get("final_answer")
        else:
            checkpoint_state["status"] = "FAILED"
            checkpoint_state["stop_reason"] = str(error_code)
            checkpoint_state["final_answer"] = None
            target_status = "FAILED"
            target_state_json = _json_dumps(checkpoint_state)
            target_pending = checkpoint_state.get("pending_question")
            target_stop_reason = str(error_code)
            target_final_answer = None

        safe_observation = _json_dumps(
            {"error_code": str(error_code), "rejected": True, "content_redacted": True}
        )
        safe_metadata = _json_dumps(
            {"error_code": str(error_code), "rejected": True, "content_redacted": True}
        )
        safe_message = "Model context rejected; attempt closed without accepting generated content."
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT 1 FROM agent_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if current is None:
                raise ValueError("Agent run does not exist")
            conn.execute(
                """
                UPDATE agent_runs
                SET question_id = ?, status = ?, current_step = ?, tool_call_count = ?,
                    state_json = ?, pending_question = ?, stop_reason = ?, final_answer = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
                """,
                (
                    checkpoint.get("question_id"),
                    target_status,
                    int(checkpoint.get("current_step") or 0),
                    int(checkpoint.get("tool_call_count") or 0),
                    target_state_json,
                    target_pending,
                    target_stop_reason,
                    target_final_answer,
                    run_id,
                ),
            )
            conn.execute(
                """
                UPDATE agent_steps
                SET action = 'security_preflight', status = 'FAILED', decision_json = '{}',
                    observation_json = ?, error_code = ?, error_message = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = ? AND id > ?
                """,
                (
                    safe_observation,
                    str(error_code),
                    "Model context rejected",
                    run_id,
                    int(checkpoint.get("step_watermark") or 0),
                ),
            )
            conn.execute(
                """
                UPDATE model_calls
                SET success = 0, error_code = ?, error_message = ?, metadata_json = ?,
                    fallback_used = 0
                WHERE run_id = ? AND id > ?
                """,
                (
                    str(error_code),
                    "Model context rejected",
                    safe_metadata,
                    run_id,
                    int(checkpoint.get("model_call_watermark") or 0),
                ),
            )
            conn.execute(
                """
                UPDATE tool_calls
                SET arguments_json = '{}', result_json = ?, success = 0, fallback_used = 0
                WHERE run_id = ? AND id > ?
                """,
                (
                    safe_observation,
                    run_id,
                    int(checkpoint.get("tool_call_watermark") or 0),
                ),
            )
            trace_cursor = conn.execute(
                """
                UPDATE agent_traces
                SET event_type = 'model_context_rejected', message = ?, metadata_json = ?
                WHERE run_id = ? AND id > ?
                """,
                (
                    safe_message,
                    safe_metadata,
                    run_id,
                    int(checkpoint.get("trace_watermark") or 0),
                ),
            )
            if trace_cursor.rowcount == 0:
                conn.execute(
                    """
                    INSERT INTO agent_traces (
                        trace_id, question_id, run_id, step_id, event_type, message, metadata_json
                    )
                    VALUES (?, ?, ?, NULL, 'model_context_rejected', ?, ?)
                    """,
                    (
                        trace_id,
                        checkpoint.get("question_id"),
                        run_id,
                        safe_message,
                        safe_metadata,
                    ),
                )

    def create_agent_step(
        self,
        *,
        run_id: str,
        step_id: str,
        step_number: int,
        action: str,
        status: str,
        decision: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        trace_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_steps (
                    step_id, run_id, step_number, action, status, decision_json,
                    observation_json, trace_id, error_code, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step_id,
                    run_id,
                    step_number,
                    action,
                    status,
                    _json_dumps(decision or {}),
                    _json_dumps(observation or {}),
                    trace_id,
                    error_code,
                    error_message,
                ),
            )

    def update_agent_step(
        self,
        *,
        step_id: str,
        status: str,
        action: str | None = None,
        decision: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        trace_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_steps
                SET action = COALESCE(?, action),
                    status = ?,
                    decision_json = ?,
                    observation_json = ?,
                    trace_id = COALESCE(?, trace_id),
                    error_code = ?,
                    error_message = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE step_id = ?
                """,
                (
                    action,
                    status,
                    _json_dumps(decision or {}),
                    _json_dumps(observation or {}),
                    trace_id,
                    error_code,
                    error_message,
                    step_id,
                ),
            )

    def agent_steps(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT step_id, run_id, step_number, action, status, decision_json,
                       observation_json, trace_id, error_code, error_message,
                       created_at, updated_at
                FROM agent_steps
                WHERE run_id = ?
                ORDER BY step_number ASC, id ASC
                """,
                (run_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["decision"] = json.loads(item.pop("decision_json") or "{}")
            item["observation"] = json.loads(item.pop("observation_json") or "{}")
            result.append(item)
        return result

    def log_model_call(
        self,
        *,
        run_id: str,
        step_id: str,
        question_id: int | None,
        provider: str,
        model: str,
        purpose: str,
        success: bool,
        error_code: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        latency_ms: float = 0,
        fallback_used: bool = False,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_calls (
                    question_id, run_id, step_id, provider, model, purpose,
                    success, error_code, error_message, metadata_json, latency_ms, fallback_used
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    run_id,
                    step_id,
                    provider,
                    model,
                    purpose,
                    1 if success else 0,
                    error_code,
                    error_message,
                    _json_dumps(metadata or {}),
                    round(float(latency_ms), 2),
                    1 if fallback_used else 0,
                ),
            )

    def log_tool_call(
        self,
        *,
        run_id: str,
        step_id: str,
        question_id: int | None,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        success: bool,
        latency_ms: float,
        fallback_used: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_calls (
                    question_id, run_id, step_id, tool_name, arguments_json,
                    result_json, success, latency_ms, fallback_used
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    run_id,
                    step_id,
                    tool_name,
                    _json_dumps(arguments),
                    _json_dumps(result),
                    1 if success else 0,
                    latency_ms,
                    1 if fallback_used else 0,
                ),
            )

    def log_agent_trace(
        self,
        *,
        trace_id: str,
        run_id: str,
        step_id: str | None,
        question_id: int | None,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_traces (
                    trace_id, question_id, run_id, step_id, event_type, message, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    question_id,
                    run_id,
                    step_id,
                    event_type,
                    message,
                    _json_dumps(metadata or {}),
                ),
            )

    def agent_trace_rows(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT trace_id, question_id, run_id, step_id, event_type, message, metadata_json, created_at
                FROM agent_traces
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            result.append(item)
        return result

    def model_call_rows(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM model_calls WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.get("metadata_json") or "{}")
            result.append(item)
        return result

    def tool_call_rows(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def log_feedback(
        self,
        *,
        question_id: int,
        user_id: str,
        rating: int,
        helpful: bool | None = None,
        comment: str = "",
    ) -> int:
        with self._connect() as conn:
            exists = conn.execute("SELECT course_id FROM questions WHERE id = ?", (question_id,)).fetchone()
            if not exists:
                raise ValueError(f"question_id 不存在：{question_id}")
            cursor = conn.execute(
                """
                INSERT INTO student_feedback (question_id, user_id, rating, helpful, comment, course_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    user_id,
                    int(rating),
                    None if helpful is None else int(helpful),
                    comment.strip(),
                    exists["course_id"],
                ),
            )
        return int(cursor.lastrowid)

    def _analytics_v2(self) -> dict[str, Any]:
        with self._connect() as conn:
            model_row = conn.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS success_count,
                       AVG(latency_ms) AS avg_latency_ms,
                       SUM(CASE WHEN fallback_used = 1 OR purpose LIKE '%fallback%' THEN 1 ELSE 0 END) AS fallback_count
                FROM model_calls
                """
            ).fetchone()
            run_row = conn.execute(
                """
                SELECT AVG(current_step) AS avg_steps,
                       AVG(tool_call_count) AS avg_tool_calls,
                       SUM(CASE WHEN status = 'REQUIRES_INPUT' THEN 1 ELSE 0 END) AS requires_input_count,
                       SUM(CASE WHEN status = 'LIMIT_REACHED' OR stop_reason = 'MAX_STEPS' THEN 1 ELSE 0 END) AS step_limit_count
                FROM agent_runs
                """
            ).fetchone()
            action_rows = conn.execute(
                """
                SELECT action, COUNT(*) AS count
                FROM agent_steps
                GROUP BY action
                ORDER BY count DESC, action ASC
                LIMIT 8
                """
            ).fetchall()
            tool_rows = conn.execute(
                """
                SELECT tool_name, COUNT(*) AS count
                FROM tool_calls
                GROUP BY tool_name
                ORDER BY count DESC, tool_name ASC
                LIMIT 8
                """
            ).fetchall()
            stop_rows = conn.execute(
                """
                SELECT COALESCE(stop_reason, status) AS reason, COUNT(*) AS count
                FROM agent_runs
                GROUP BY reason
                ORDER BY count DESC, reason ASC
                """
            ).fetchall()
            hit_row = conn.execute(
                """
                SELECT COUNT(DISTINCT q.id) AS total_questions,
                       COUNT(DISTINCT r.question_id) AS hit_questions
                FROM questions q
                LEFT JOIN retrieval_logs r ON r.question_id = q.id
                """
            ).fetchone()
            module_rows = conn.execute(
                """
                SELECT module,
                       COUNT(*) AS total,
                       SUM(CASE WHEN abstained = 1 OR confidence < 0.35 THEN 1 ELSE 0 END) AS stuck_count,
                       AVG(confidence) AS avg_confidence
                FROM questions
                GROUP BY module
                HAVING stuck_count > 0
                ORDER BY stuck_count DESC, avg_confidence ASC
                LIMIT 8
                """
            ).fetchall()
            feedback_row = conn.execute(
                """
                SELECT COUNT(*) AS count,
                       AVG(rating) AS average_rating,
                       SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) AS helpful_count
                FROM student_feedback
                """
            ).fetchone()
            feedback_recent = conn.execute(
                """
                SELECT question_id, rating, helpful, comment, created_at
                FROM student_feedback
                ORDER BY id DESC
                LIMIT 10
                """
            ).fetchall()

        model_total = int(model_row["total"] or 0)
        total_questions = int(hit_row["total_questions"] or 0)
        hit_questions = int(hit_row["hit_questions"] or 0)
        feedback_count = int(feedback_row["count"] or 0)
        helpful_count = int(feedback_row["helpful_count"] or 0)
        return {
            "model_metrics": {
                "total_calls": model_total,
                "success_rate": round((int(model_row["success_count"] or 0) / model_total) if model_total else 0.0, 4),
                "average_latency_ms": round(float(model_row["avg_latency_ms"] or 0.0), 2),
                "fallback_count": int(model_row["fallback_count"] or 0),
            },
            "agent_metrics": {
                "average_steps": round(float(run_row["avg_steps"] or 0.0), 2),
                "average_tool_calls": round(float(run_row["avg_tool_calls"] or 0.0), 2),
                "requires_input_count": int(run_row["requires_input_count"] or 0),
                "step_limit_reached_count": int(run_row["step_limit_count"] or 0),
                "top_actions": [dict(row) for row in action_rows],
                "stop_reasons": [dict(row) for row in stop_rows],
            },
            "tool_metrics": {"top_tools": [dict(row) for row in tool_rows]},
            "retrieval_metrics": {
                "knowledge_hit_rate": round((hit_questions / total_questions) if total_questions else 0.0, 4),
                "hit_questions": hit_questions,
                "total_questions": total_questions,
            },
            "module_stuck_points": [
                {
                    "module": row["module"],
                    "total": int(row["total"]),
                    "stuck_count": int(row["stuck_count"] or 0),
                    "average_confidence": round(float(row["avg_confidence"] or 0.0), 2),
                }
                for row in module_rows
            ],
            "feedback_metrics": {
                "count": feedback_count,
                "average_rating": round(float(feedback_row["average_rating"] or 0.0), 2),
                "helpful_rate": round((helpful_count / feedback_count) if feedback_count else 0.0, 4),
                "recent": [
                    {
                        "question_id": int(row["question_id"]),
                        "rating": int(row["rating"]),
                        "helpful": None if row["helpful"] is None else bool(row["helpful"]),
                        "comment": _preview(row["comment"], max_chars=80),
                        "created_at": row["created_at"],
                    }
                    for row in feedback_recent
                ],
            },
        }

    def attach_question_to_agent_run(self, run_id: str, question_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE agent_runs SET question_id = ?, updated_at = CURRENT_TIMESTAMP WHERE run_id = ?",
                (question_id, run_id),
            )
            conn.execute("UPDATE questions SET run_id = ? WHERE id = ?", (run_id, question_id))
            conn.execute(
                "UPDATE model_calls SET question_id = ? WHERE run_id = ?",
                (question_id, run_id),
            )
            conn.execute(
                "UPDATE tool_calls SET question_id = ? WHERE run_id = ?",
                (question_id, run_id),
            )
            conn.execute(
                "UPDATE agent_traces SET question_id = ? WHERE run_id = ?",
                (question_id, run_id),
            )

    def save_document(
        self,
        document: Document,
        content_hash: str,
        chunks: list[str | dict[str, Any]] | None,
        byte_size: int,
        index_status: str = "PENDING",
        job_id: int | None = None,
        course_id: int | None = None,
        class_id: int | None = None,
        uploaded_by_user_id: int | None = None,
        visibility: str = "course",
    ) -> dict[str, Any]:
        _validate_index_status(index_status)
        prepared_chunks = _prepare_chunks(document, chunks)
        if course_id is None:
            raise ValueError("course_id is required when saving a protected document")
        metadata = sanitized_new_document_metadata(document.metadata)
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT id, filename, content_hash, chunk_count, index_status
                FROM documents
                WHERE content_hash = ? AND course_id = ?
                """,
                (content_hash, course_id),
            ).fetchone()
            if existing:
                return {**dict(existing), "duplicate": True}

            if job_id is None:
                job_id = self.create_index_job(
                    filename=document.filename,
                    content_hash=content_hash,
                    status="CHUNKING",
                )
            else:
                self.update_index_job(job_id, status="CHUNKING")

            cursor = conn.execute(
                """
                INSERT INTO documents (
                    filename, source_path, file_type, title, content_hash, byte_size,
                    text_length, chunk_count, index_status, metadata_json,
                    course_id, class_id, uploaded_by_user_id, visibility
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.filename,
                    str(document.source_path),
                    document.file_type,
                    document.title,
                    content_hash,
                    byte_size,
                    len(document.text),
                    len(prepared_chunks),
                    index_status,
                    _json_dumps(metadata),
                    course_id,
                    class_id,
                    uploaded_by_user_id,
                    visibility,
                ),
            )
            document_id = int(cursor.lastrowid)
            conn.executemany(
                """
                INSERT INTO document_chunks (
                    document_id, chunk_index, chunk_text, chunk_hash, token_count,
                    page, section, source_type, index_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        chunk["chunk_index"],
                        chunk["chunk_text"],
                        chunk["chunk_hash"],
                        chunk["token_count"],
                        chunk["page"],
                        chunk["section"],
                        chunk["source_type"],
                        index_status,
                    )
                    for chunk in prepared_chunks
                ],
            )
            conn.execute(
                "UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (document_id,),
            )
        self.update_index_job(job_id, status=index_status, document_id=document_id)

        return {
            "id": document_id,
            "filename": document.filename,
            "content_hash": content_hash,
            "chunk_count": len(prepared_chunks),
            "index_status": index_status,
            "duplicate": False,
            "job_id": job_id,
        }

    def document_by_hash(self, content_hash: str, course_id: int | None = None) -> dict[str, Any] | None:
        where = "WHERE content_hash = ?"
        params: tuple[Any, ...] = (content_hash,)
        if course_id is not None:
            where += " AND course_id = ?"
            params = (content_hash, course_id)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT id, filename, source_path, file_type, title, content_hash, byte_size,
                       text_length, chunk_count, index_status, metadata_json, course_id, class_id,
                       uploaded_by_user_id, visibility, created_at, updated_at
                FROM documents
                {where}
                ORDER BY id DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        return _document_row(dict(row)) if row else None

    def document(self, document_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, filename, source_path, file_type, title, content_hash, byte_size,
                       text_length, chunk_count, index_status, metadata_json, course_id, class_id,
                       uploaded_by_user_id, visibility, created_at, updated_at
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        return _document_row(dict(row)) if row else None

    def documents(self, course_id: int | None = None) -> list[dict[str, Any]]:
        where = "WHERE course_id = ?" if course_id is not None else ""
        params: tuple[Any, ...] = (course_id,) if course_id is not None else ()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, filename, source_path, file_type, title, content_hash, byte_size,
                       text_length, chunk_count, index_status, metadata_json, course_id, class_id,
                       uploaded_by_user_id, visibility, created_at, updated_at
                FROM documents
                {where}
                ORDER BY id DESC
                """,
                params,
            ).fetchall()
        return [_document_row(dict(row)) for row in rows]

    def user(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, display_name, password_hash, role, enabled,
                       must_change_password, token_version, created_at, last_login_at
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def user_by_username(self, username: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, display_name, password_hash, role, enabled,
                       must_change_password, token_version, created_at, last_login_at
                FROM users
                WHERE username = ?
                """,
                (username.strip(),),
            ).fetchone()
        return dict(row) if row else None

    def upsert_user(
        self,
        *,
        username: str,
        display_name: str,
        password_hash: str,
        role: str,
        enabled: bool = True,
        must_change_password: bool = True,
    ) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if row:
                return int(row["id"])
            cursor = conn.execute(
                """
                INSERT INTO users (
                    username, display_name, password_hash, role, enabled, must_change_password
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    display_name,
                    password_hash,
                    role,
                    int(enabled),
                    int(must_change_password),
                ),
            )
        return int(cursor.lastrowid)

    def mark_user_login(self, user_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))

    def update_user_password(self, user_id: int, password_hash: str, *, must_change_password: bool = False) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, must_change_password = ?, token_version = token_version + 1
                WHERE id = ?
                """,
                (password_hash, int(must_change_password), user_id),
            )

    def revoke_user_tokens(self, user_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET token_version = token_version + 1 WHERE id = ?",
                (user_id,),
            )

    def default_course_id(self) -> int:
        with self._connect() as conn:
            return _ensure_demo_course(conn)

    def default_class_id(self) -> int:
        with self._connect() as conn:
            course_id = _ensure_demo_course(conn)
            return _ensure_demo_class(conn, course_id)

    def seed_demo_identity(self) -> dict[str, Any]:
        from backend.auth.password import hash_password

        default_hash = hash_password("demo123456")
        with self._connect() as conn:
            course_id = _ensure_demo_course(conn)
            teacher_id = _upsert_user_raw(conn, "teacher_demo", "Demo Teacher", default_hash, "teacher")
            class_id = _ensure_demo_class(conn, course_id, teacher_id=teacher_id)
            student_1 = _upsert_user_raw(conn, "student_demo", "Demo Student 1", default_hash, "student")
            student_2 = _upsert_user_raw(conn, "student_demo2", "Demo Student 2", default_hash, "student")
            for user_id in (teacher_id, student_1, student_2):
                _ensure_class_member(conn, class_id, user_id)
            _backfill_phase3_context(conn, course_id, class_id)
        return {
            "course_id": course_id,
            "class_id": class_id,
            "teacher_id": teacher_id,
            "student_ids": [student_1, student_2],
        }

    def courses(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, description, assessment_method, created_at FROM courses ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def classes(self, course_id: int | None = None) -> list[dict[str, Any]]:
        where = "WHERE course_id = ?" if course_id is not None else ""
        params: tuple[Any, ...] = (course_id,) if course_id is not None else ()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, course_id, name, term, teacher_id, created_at
                FROM classes
                {where}
                ORDER BY id
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def class_belongs_to_course(self, class_id: int, course_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM classes WHERE id = ? AND course_id = ? LIMIT 1",
                (int(class_id), int(course_id)),
            ).fetchone()
        return row is not None

    def user_in_class(self, user_id: int, class_id: int, course_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM class_members cm
                JOIN classes cls ON cls.id = cm.class_id
                WHERE cm.user_id = ? AND cm.class_id = ? AND cls.course_id = ?
                LIMIT 1
                """,
                (int(user_id), int(class_id), int(course_id)),
            ).fetchone()
        return row is not None

    def teacher_in_class(self, teacher_id: int, class_id: int, course_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM classes WHERE id = ? AND course_id = ? AND teacher_id = ? LIMIT 1",
                (int(class_id), int(course_id), int(teacher_id)),
            ).fetchone()
        return row is not None

    def courses_for_user(self, user_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT c.id, c.name, c.description, c.assessment_method, c.created_at
                FROM courses c
                JOIN classes cls ON cls.course_id = c.id
                JOIN class_members cm ON cm.class_id = cls.id
                WHERE cm.user_id = ?
                ORDER BY c.id
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def courses_for_teacher(self, teacher_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT c.id, c.name, c.description, c.assessment_method, c.created_at
                FROM courses c
                JOIN classes cls ON cls.course_id = c.id
                WHERE cls.teacher_id = ?
                ORDER BY c.id
                """,
                (teacher_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def classes_for_user(self, user_id: int, course_id: int | None = None) -> list[dict[str, Any]]:
        course_filter = "AND cls.course_id = ?" if course_id is not None else ""
        params: tuple[Any, ...] = (user_id, course_id) if course_id is not None else (user_id,)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT cls.id, cls.course_id, cls.name, cls.term, cls.teacher_id, cls.created_at
                FROM classes cls
                JOIN class_members cm ON cm.class_id = cls.id
                WHERE cm.user_id = ? {course_filter}
                ORDER BY cls.id
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def classes_for_teacher(self, teacher_id: int, course_id: int | None = None) -> list[dict[str, Any]]:
        course_filter = "AND course_id = ?" if course_id is not None else ""
        params: tuple[Any, ...] = (teacher_id, course_id) if course_id is not None else (teacher_id,)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, course_id, name, term, teacher_id, created_at
                FROM classes
                WHERE teacher_id = ? {course_filter}
                ORDER BY id
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def teacher_can_access_user(self, teacher_id: int, target_user_id: str | int) -> bool:
        with self._connect() as conn:
            target = _resolve_user_id(conn, target_user_id)
            if target is None:
                return False
            row = conn.execute(
                """
                SELECT 1
                FROM classes cls
                JOIN class_members cm ON cm.class_id = cls.id
                WHERE cls.teacher_id = ? AND cm.user_id = ?
                LIMIT 1
                """,
                (teacher_id, target),
            ).fetchone()
        return row is not None

    def create_learning_session(
        self,
        *,
        user_id: int,
        course_id: int,
        class_id: int | None,
        client_session_id: str,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM learning_sessions WHERE client_session_id = ?",
                (client_session_id,),
            ).fetchone()
            if existing:
                return dict(existing)
            cursor = conn.execute(
                """
                INSERT INTO learning_sessions (user_id, course_id, class_id, client_session_id)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, course_id, class_id, client_session_id),
            )
            row = conn.execute("SELECT * FROM learning_sessions WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)

    def create_learning_session_with_login(
        self,
        *,
        user_id: int,
        course_id: int,
        class_id: int | None,
        client_session_id: str,
        metadata: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM learning_sessions WHERE client_session_id = ?",
                (client_session_id,),
            ).fetchone()
            if existing:
                return dict(existing)
            cursor = conn.execute(
                """
                INSERT INTO learning_sessions (user_id, course_id, class_id, client_session_id)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, course_id, class_id, client_session_id),
            )
            session_id = int(cursor.lastrowid)
            if dedupe_key:
                existing_event = conn.execute(
                    "SELECT * FROM learning_events WHERE dedupe_key = ?",
                    (dedupe_key,),
                ).fetchone()
                if existing_event:
                    row = conn.execute("SELECT * FROM learning_sessions WHERE id = ?", (session_id,)).fetchone()
                    return dict(row)
            conn.execute(
                """
                INSERT INTO learning_events (
                    session_id, user_id, course_id, class_id, event_type, page_name,
                    module, resource_id, assignment_id, metadata_json, dedupe_key
                )
                VALUES (?, ?, ?, ?, 'LOGIN', '', '', NULL, NULL, ?, ?)
                """,
                (
                    session_id,
                    user_id,
                    course_id,
                    class_id,
                    _json_dumps(metadata or {}),
                    dedupe_key,
                ),
            )
            row = conn.execute("SELECT * FROM learning_sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row)

    def learning_session_by_client_id(self, client_session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM learning_sessions WHERE client_session_id = ?",
                (client_session_id,),
            ).fetchone()
        return dict(row) if row else None

    def heartbeat_learning_session(
        self,
        *,
        session_id: int,
        idle_timeout_seconds: int,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM learning_sessions WHERE id = ?", (session_id,)).fetchone()
            if not row:
                raise ValueError(f"learning session 不存在：{session_id}")
            delta_row = conn.execute(
                "SELECT MAX(0, CAST((julianday(CURRENT_TIMESTAMP) - julianday(?)) * 86400 AS INTEGER)) AS delta",
                (row["last_active_at"],),
            ).fetchone()
            delta = int(delta_row["delta"] or 0)
            increment = delta if 0 < delta <= idle_timeout_seconds else 0
            conn.execute(
                """
                UPDATE learning_sessions
                SET active_seconds = active_seconds + ?,
                    last_active_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (increment, session_id),
            )
            updated = conn.execute("SELECT * FROM learning_sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(updated)

    def end_learning_session(self, session_id: int) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute(
                "UPDATE learning_sessions SET ended_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )
            row = conn.execute("SELECT * FROM learning_sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            raise ValueError(f"learning session 不存在：{session_id}")
        return dict(row)

    def log_learning_event(
        self,
        *,
        session_id: int,
        user_id: int,
        course_id: int,
        class_id: int | None,
        event_type: str,
        page_name: str = "",
        module: str = "",
        resource_id: str | None = None,
        assignment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            if dedupe_key:
                existing = conn.execute(
                    "SELECT * FROM learning_events WHERE dedupe_key = ?",
                    (dedupe_key,),
                ).fetchone()
                if existing:
                    return {**dict(existing), "duplicate": True}
            cursor = conn.execute(
                """
                INSERT INTO learning_events (
                    session_id, user_id, course_id, class_id, event_type, page_name,
                    module, resource_id, assignment_id, metadata_json, dedupe_key
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_id,
                    course_id,
                    class_id,
                    event_type,
                    page_name,
                    module,
                    resource_id,
                    assignment_id,
                    _json_dumps(metadata or {}),
                    dedupe_key,
                ),
            )
            row = conn.execute("SELECT * FROM learning_events WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return {**dict(row), "duplicate": False}

    def learning_timeline(self, *, user_id: int, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, user_id, course_id, class_id, event_type, page_name,
                       module, resource_id, assignment_id, metadata_json, created_at
                FROM learning_events
                WHERE user_id = ?
                ORDER BY datetime(created_at) ASC, id ASC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        events = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            events.append(item)
        return events

    def learning_activity_summary(
        self,
        *,
        course_id: int | None = None,
        class_ids: list[int | None] | None = None,
    ) -> dict[str, Any]:
        clauses = []
        params: list[Any] = []
        if course_id is not None:
            clauses.append("course_id = ?")
            params.append(course_id)
        if class_ids is not None:
            if not class_ids:
                return _empty_activity_summary()
            placeholders = ",".join("?" for _ in class_ids)
            clauses.append(f"class_id IN ({placeholders})")
            params.extend(class_ids)
        session_where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        event_clauses = list(clauses)
        event_clauses.append(
            "event_type IN (" + ",".join("?" for _ in _CLIENT_LEARNING_EVENT_TYPES) + ")"
        )
        event_params = [*params, *_CLIENT_LEARNING_EVENT_TYPES]
        where = f"WHERE {' AND '.join(event_clauses)}"
        with self._connect() as conn:
            event_row = conn.execute(
                f"""
                SELECT COUNT(*) AS visit_count,
                       COUNT(DISTINCT user_id) AS unique_active_students,
                       COUNT(DISTINCT date(created_at)) AS active_days,
                       MAX(created_at) AS recent_active_at
                FROM learning_events
                {where}
                """,
                event_params,
            ).fetchone()
            page_rows = conn.execute(
                f"""
                SELECT COALESCE(NULLIF(page_name, ''), '(unknown)') AS page_name, COUNT(*) AS count
                FROM learning_events
                {where}
                GROUP BY COALESCE(NULLIF(page_name, ''), '(unknown)')
                ORDER BY count DESC, page_name ASC
                """,
                event_params,
            ).fetchall()
            event_rows = conn.execute(
                f"""
                SELECT event_type, COUNT(*) AS count
                FROM learning_events
                {where}
                GROUP BY event_type
                ORDER BY count DESC, event_type ASC
                """,
                event_params,
            ).fetchall()
            session_row = conn.execute(
                f"SELECT SUM(active_seconds) AS active_seconds FROM learning_sessions {session_where}",
                params,
            ).fetchone()
        return {
            "unique_active_students": int(event_row["unique_active_students"] or 0),
            "visit_count": int(event_row["visit_count"] or 0),
            "active_days": int(event_row["active_days"] or 0),
            "total_active_seconds": int(session_row["active_seconds"] or 0),
            "recent_active_at": event_row["recent_active_at"],
            "page_distribution": {row["page_name"]: int(row["count"]) for row in page_rows},
            "event_distribution": {row["event_type"]: int(row["count"]) for row in event_rows},
        }

    def upsert_external_resource(
        self,
        *,
        resource: dict[str, Any],
        ttl_hours: int,
        content_hash: str,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            row = self._upsert_external_resource_in_conn(
                conn,
                resource=resource,
                ttl_hours=ttl_hours,
                content_hash=content_hash,
            )
        return row

    def _upsert_external_resource_in_conn(
        self,
        conn: sqlite3.Connection,
        *,
        resource: dict[str, Any],
        ttl_hours: int,
        content_hash: str,
    ) -> dict[str, Any]:
        existing = conn.execute(
                """
                SELECT id, content_hash, indexed, vector_id
                FROM external_resources
                WHERE source_name = ? AND external_id = ?
                """,
                (resource["source_name"], resource["external_id"]),
            ).fetchone()
        params = (
                resource["external_id"],
                resource["resource_type"],
                resource["source_name"],
                resource["title"],
                _json_dumps(canonical_authors(resource.get("authors", []))),
                resource.get("abstract", ""),
                resource.get("url", ""),
                resource.get("published_at"),
                _json_dumps(resource.get("tags", [])),
                resource.get("module", ""),
                _json_dumps(resource.get("raw_metadata", {})),
                content_hash,
                resource.get("rights_note", ""),
                ttl_hours,
        )
        if existing:
            conn.execute(
                    """
                    UPDATE external_resources
                    SET resource_type = ?, title = ?, authors_json = ?, abstract = ?,
                        url = ?, published_at = ?, tags_json = ?, module = ?,
                        raw_metadata_json = ?, content_hash = ?, rights_note = ?,
                        fetched_at = CURRENT_TIMESTAMP,
                        expires_at = datetime(CURRENT_TIMESTAMP, '+' || ? || ' hours'),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        params[1],
                        params[3],
                        params[4],
                        params[5],
                        params[6],
                        params[7],
                        params[8],
                        params[9],
                        params[10],
                        params[11],
                        params[12],
                        ttl_hours,
                        existing["id"],
                    ),
            )
            row_id = int(existing["id"])
        else:
            cursor = conn.execute(
                    """
                    INSERT INTO external_resources (
                        external_id, resource_type, source_name, title, authors_json,
                        abstract, url, published_at, tags_json, module, raw_metadata_json,
                        content_hash, rights_note, fetched_at, expires_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP,
                            datetime(CURRENT_TIMESTAMP, '+' || ? || ' hours'))
                    """,
                    params,
            )
            row_id = int(cursor.lastrowid)
        row = conn.execute("SELECT * FROM external_resources WHERE id = ?", (row_id,)).fetchone()
        return _external_resource_row(dict(row))

    def external_resource_by_key(self, source_name: str, external_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM external_resources
                WHERE source_name = ? AND external_id = ?
                """,
                (source_name, external_id),
            ).fetchone()
        return _external_resource_row(dict(row)) if row else None

    def verified_external_resource_by_key(self, source_name: str, external_id: str) -> dict[str, Any] | None:
        """Read one resource only through its current successful provenance binding."""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT er.*, rp.sync_run_id AS provenance_sync_run_id,
                       rp.origin_kind, rp.verified_at, 1 AS provenance_verified
                FROM external_resources er
                JOIN resource_provenance rp
                  ON rp.resource_id = er.id
                 AND rp.connector_name = er.source_name
                 AND rp.content_hash = er.content_hash
                 AND rp.verified_url = er.url
                 AND rp.origin_kind IN ('remote', 'local_catalog')
                JOIN connector_sync_runs csr
                  ON csr.id = rp.sync_run_id
                 AND csr.connector_name = er.source_name
                 AND csr.status = 'SUCCESS'
                WHERE er.source_name = ? AND er.external_id = ?
                ORDER BY datetime(rp.verified_at) DESC, rp.sync_run_id DESC
                LIMIT 1
                """,
                (source_name, external_id),
            ).fetchone()
        return _external_resource_row(dict(row)) if row else None

    def search_external_resources(
        self,
        query: dict[str, Any],
        *,
        include_expired: bool = False,
        source_name: str | None = None,
        exact_only: bool = False,
    ) -> list[dict[str, Any]]:
        mapped = self._search_resource_query_cache(query, include_expired=include_expired, source_name=source_name)
        if mapped:
            return mapped
        if exact_only:
            return []
        clauses = [
            "csr.status = 'SUCCESS'",
            "csr.connector_name = er.source_name",
            "rp.connector_name = er.source_name",
            "rp.content_hash = er.content_hash",
            "rp.verified_url = er.url",
            "rp.origin_kind IN ('remote', 'local_catalog')",
        ]
        params: list[Any] = []
        q = str(query.get("query") or "").strip().lower()
        if q:
            clauses.append("(lower(title) LIKE ? OR lower(abstract) LIKE ? OR lower(tags_json) LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like, like])
        if query.get("resource_type"):
            clauses.append("resource_type = ?")
            params.append(query["resource_type"])
        if query.get("module"):
            clauses.append("module = ?")
            params.append(query["module"])
        if source_name:
            clauses.append("source_name = ?")
            params.append(source_name)
        if not include_expired:
            clauses.append("datetime(expires_at) > datetime(CURRENT_TIMESTAMP)")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        page = max(1, int(query.get("page") or 1))
        page_size = max(1, min(50, int(query.get("page_size") or 10)))
        offset = (page - 1) * page_size
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT er.*, rp.sync_run_id AS provenance_sync_run_id,
                       rp.origin_kind, rp.verified_at, 1 AS provenance_verified
                FROM external_resources er
                JOIN resource_provenance rp ON rp.resource_id = er.id
                JOIN connector_sync_runs csr ON csr.id = rp.sync_run_id
                {where}
                ORDER BY datetime(fetched_at) DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (*params, page_size, offset),
            ).fetchall()
        return [_external_resource_row(dict(row)) for row in rows]

    def store_resource_query_cache(
        self,
        query: dict[str, Any],
        *,
        source_name: str,
        resource_ids: list[int],
        ttl_hours: int,
    ) -> None:
        with self._connect() as conn:
            self._store_resource_query_cache_in_conn(
                conn,
                query,
                source_name=source_name,
                resource_ids=resource_ids,
                ttl_hours=ttl_hours,
            )

    def _store_resource_query_cache_in_conn(
        self,
        conn: sqlite3.Connection,
        query: dict[str, Any],
        *,
        source_name: str,
        resource_ids: list[int],
        ttl_hours: int,
    ) -> None:
        query_key = _resource_query_key(query, source_name=source_name)
        query_json = _json_dumps(_normalized_resource_query(query, source_name=source_name))
        cursor = conn.execute(
                """
                INSERT INTO resource_query_cache (
                    query_key, source_name, query_json, expires_at, updated_at
                )
                VALUES (?, ?, ?, datetime(CURRENT_TIMESTAMP, '+' || ? || ' hours'), CURRENT_TIMESTAMP)
                ON CONFLICT(query_key) DO UPDATE SET
                    source_name = excluded.source_name,
                    query_json = excluded.query_json,
                    expires_at = excluded.expires_at,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (query_key, source_name, query_json, ttl_hours),
        )
        row = conn.execute("SELECT id FROM resource_query_cache WHERE query_key = ?", (query_key,)).fetchone()
        cache_id = int(row["id"]) if row else int(cursor.lastrowid)
        conn.execute("DELETE FROM resource_query_cache_items WHERE cache_id = ?", (cache_id,))
        unique_resource_ids = list(dict.fromkeys(int(resource_id) for resource_id in resource_ids))
        conn.executemany(
                """
                INSERT INTO resource_query_cache_items (cache_id, resource_id, position)
                VALUES (?, ?, ?)
                """,
            [(cache_id, resource_id, index) for index, resource_id in enumerate(unique_resource_ids)],
        )

    def commit_verified_sync(
        self,
        *,
        run_id: int,
        connector_name: str,
        origin_kind: str,
        query: dict[str, Any],
        resources: list[dict[str, Any]],
        ttl_hours: int,
        latency_ms: float,
    ) -> list[dict[str, Any]]:
        if origin_kind not in {"remote", "local_catalog"}:
            raise ValueError(f"Untrusted origin kind: {origin_kind}")
        with self._connect() as conn:
            run = conn.execute(
                "SELECT connector_name, status FROM connector_sync_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if not run or run["connector_name"] != connector_name or run["status"] != "RUNNING":
                raise ValueError("Connector sync run is not writable.")
            saved: list[dict[str, Any]] = []
            for item in resources:
                resource = dict(item["resource"])
                digest = str(item["content_hash"])
                if resource.get("source_name") != connector_name:
                    raise ValueError("Resource source does not match connector.")
                if not digest or not str(resource.get("url") or "").strip().lower().startswith("https://"):
                    raise ValueError("Verified resources require content hash and source URL.")
                record = self._upsert_external_resource_in_conn(
                    conn,
                    resource=resource,
                    ttl_hours=ttl_hours,
                    content_hash=digest,
                )
                conn.execute(
                    """
                    INSERT INTO resource_provenance (
                        resource_id, sync_run_id, connector_name, origin_kind,
                        content_hash, verified_url, verified_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(resource_id) DO UPDATE SET
                        sync_run_id = excluded.sync_run_id,
                        connector_name = excluded.connector_name,
                        origin_kind = excluded.origin_kind,
                        content_hash = excluded.content_hash,
                        verified_url = excluded.verified_url,
                        verified_at = CURRENT_TIMESTAMP
                    """,
                    (
                        int(record["id"]),
                        run_id,
                        connector_name,
                        origin_kind,
                        digest,
                        str(resource["url"]),
                    ),
                )
                saved.append(record)
            if saved:
                self._store_resource_query_cache_in_conn(
                    conn,
                    query,
                    source_name=connector_name,
                    resource_ids=[int(item["id"]) for item in saved],
                    ttl_hours=ttl_hours,
                )
            conn.execute(
                """
                UPDATE connector_sync_runs
                SET status = 'SUCCESS', result_count = ?, cache_hit = 0,
                    error_code = NULL, error_message = NULL, latency_ms = ?,
                    finished_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (len(saved), latency_ms, run_id),
            )
        return saved

    def resource_provenance(self, resource_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT rp.*, csr.status AS sync_status
                FROM resource_provenance rp
                JOIN connector_sync_runs csr ON csr.id = rp.sync_run_id
                WHERE rp.resource_id = ?
                """,
                (resource_id,),
            ).fetchone()
        return dict(row) if row else None

    def _search_resource_query_cache(
        self,
        query: dict[str, Any],
        *,
        include_expired: bool,
        source_name: str | None,
    ) -> list[dict[str, Any]]:
        normalized = _normalized_resource_query(query, source_name=source_name or "")
        clauses = [
            "json_extract(qc.query_json, '$.query') = ?",
            "COALESCE(json_extract(qc.query_json, '$.resource_type'), '') = ?",
            "COALESCE(json_extract(qc.query_json, '$.module'), '') = ?",
            "COALESCE(json_extract(qc.query_json, '$.year_from'), '') = ?",
            "COALESCE(json_extract(qc.query_json, '$.year_to'), '') = ?",
            "json_extract(qc.query_json, '$.page') = ?",
            "json_extract(qc.query_json, '$.page_size') = ?",
            "COALESCE(json_extract(qc.query_json, '$.lookup_mode'), 'SEARCH') = ?",
            "COALESCE(json_extract(qc.query_json, '$.canonical_identifier'), '') = ?",
            "qc.source_name = er.source_name",
            "csr.status = 'SUCCESS'",
            "csr.connector_name = er.source_name",
            "rp.connector_name = er.source_name",
            "rp.content_hash = er.content_hash",
            "rp.verified_url = er.url",
            "rp.origin_kind IN ('remote', 'local_catalog')",
        ]
        params: list[Any] = [
            normalized["query"],
            normalized["resource_type"] or "",
            normalized["module"],
            normalized["year_from"] if normalized["year_from"] is not None else "",
            normalized["year_to"] if normalized["year_to"] is not None else "",
            normalized["page"],
            normalized["page_size"],
            normalized["lookup_mode"],
            normalized["canonical_identifier"] or "",
        ]
        if source_name:
            clauses.append("qc.source_name = ?")
            params.append(source_name)
            clauses.append("er.source_name = ?")
            params.append(source_name)
        if not include_expired:
            clauses.append("datetime(qc.expires_at) > datetime(CURRENT_TIMESTAMP)")
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT er.*, rp.sync_run_id AS provenance_sync_run_id,
                       rp.origin_kind, rp.verified_at, 1 AS provenance_verified,
                       json_extract(qc.query_json, '$.incomplete_results') AS query_incomplete_results
                FROM resource_query_cache qc
                JOIN resource_query_cache_items qi ON qi.cache_id = qc.id
                JOIN external_resources er ON er.id = qi.resource_id
                JOIN resource_provenance rp ON rp.resource_id = er.id
                JOIN connector_sync_runs csr ON csr.id = rp.sync_run_id
                WHERE {' AND '.join(clauses)}
                ORDER BY datetime(qc.updated_at) DESC, qi.position ASC, er.id DESC
                """,
                params,
            ).fetchall()
        seen: set[int] = set()
        result: list[dict[str, Any]] = []
        for row in rows:
            row_id = int(row["id"])
            if row_id in seen:
                continue
            seen.add(row_id)
            result.append(_external_resource_row(dict(row)))
        return result

    def mark_external_resource_indexed(self, resource_id: int, vector_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE external_resources
                SET indexed = 1, vector_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (vector_id, resource_id),
            )

    def create_connector_sync_run(
        self,
        *,
        connector_name: str,
        query: dict[str, Any],
        status: str,
        cache_hit: bool,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO connector_sync_runs (
                    connector_name, query, status, cache_hit, started_at
                )
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (connector_name, _json_dumps(query), status, int(cache_hit)),
            )
        return int(cursor.lastrowid)

    def finish_connector_sync_run(
        self,
        run_id: int,
        *,
        status: str,
        result_count: int,
        cache_hit: bool,
        error_code: str | None = None,
        error_message: str | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE connector_sync_runs
                SET status = ?, result_count = ?, cache_hit = ?, error_code = ?,
                    error_message = ?, latency_ms = ?, finished_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, result_count, int(cache_hit), error_code, error_message, latency_ms, run_id),
            )

    def connector_sync_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, connector_name, query, status, result_count, cache_hit,
                       error_code, error_message, latency_ms, started_at, finished_at
                FROM connector_sync_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_document_index_status(
        self,
        document_id: int,
        status: str,
        *,
        include_chunks: bool = True,
    ) -> None:
        _validate_index_status(status)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata_json, content_hash FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            metadata_json = None
            if row is not None and status != "INDEXED":
                metadata_json = _publication_metadata_json(
                    row["metadata_json"],
                    published=False,
                    actor_id=None,
                    action="reindex" if status in {"PENDING", "PARSING", "CHUNKING", "EMBEDDING"} else "index_failed",
                    content_hash=str(row["content_hash"] or ""),
                )
            conn.execute(
                """
                UPDATE documents
                SET index_status = ?,
                    metadata_json = COALESCE(?, metadata_json),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, metadata_json, document_id),
            )
            if include_chunks:
                conn.execute(
                    """
                    UPDATE document_chunks
                    SET index_status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE document_id = ?
                    """,
                    (status, document_id),
                )

    def has_active_index_job(self, document_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM index_jobs
                WHERE document_id = ?
                  AND status IN ('PENDING', 'PARSING', 'CHUNKING', 'EMBEDDING')
                LIMIT 1
                """,
                (int(document_id),),
            ).fetchone()
        return row is not None

    def begin_document_reindex(self, document_id: int) -> int:
        """Atomically hide a document and put its sole index job in EMBEDDING."""

        document_id = int(document_id)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            document = conn.execute(
                """
                SELECT id, filename, content_hash, metadata_json
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
            if document is None:
                raise ValueError("DOCUMENT_NOT_FOUND")

            active_jobs = conn.execute(
                """
                SELECT id, document_id, filename, content_hash, status
                FROM index_jobs
                WHERE document_id = ?
                  AND status IN ('PENDING', 'PARSING', 'CHUNKING', 'EMBEDDING')
                ORDER BY id
                """,
                (document_id,),
            ).fetchall()
            if len(active_jobs) > 1 or any(str(row["status"]) != "PENDING" for row in active_jobs):
                raise ValueError("DOCUMENT_BUSY")

            if active_jobs:
                job = active_jobs[0]
                if (
                    int(job["document_id"]) != document_id
                    or str(job["filename"] or "") != str(document["filename"] or "")
                    or str(job["content_hash"] or "") != str(document["content_hash"] or "")
                ):
                    raise ValueError("DOCUMENT_BUSY")
                job_id = int(job["id"])
                conn.execute(
                    """
                    UPDATE index_jobs
                    SET status = 'EMBEDDING', error_message = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND document_id = ? AND status = 'PENDING'
                    """,
                    (job_id, document_id),
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO index_jobs (
                        document_id, filename, content_hash, status
                    ) VALUES (?, ?, ?, 'EMBEDDING')
                    """,
                    (
                        document_id,
                        str(document["filename"] or ""),
                        str(document["content_hash"] or ""),
                    ),
                )
                job_id = int(cursor.lastrowid)

            metadata_json = _publication_metadata_json(
                document["metadata_json"],
                published=False,
                actor_id=None,
                action="reindex",
                content_hash=str(document["content_hash"] or ""),
            )
            conn.execute(
                """
                UPDATE documents
                SET index_status = 'EMBEDDING', metadata_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (metadata_json, document_id),
            )
            conn.execute(
                """
                UPDATE document_chunks
                SET index_status = 'EMBEDDING', updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (document_id,),
            )
            conn.commit()
        return job_id

    def set_document_publication(
        self,
        document_id: int,
        *,
        published: bool,
        actor_id: int,
    ) -> dict[str, Any]:
        if actor_id is None or isinstance(actor_id, bool) or int(actor_id) <= 0:
            raise ValueError("DOCUMENT_ACTOR_REQUIRED")
        action = "publish" if published else "unpublish"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id, index_status, metadata_json, content_hash FROM documents WHERE id = ?",
                (int(document_id),),
            ).fetchone()
            if row is None:
                raise ValueError("DOCUMENT_NOT_FOUND")
            if published:
                if str(row["index_status"] or "").upper() != "INDEXED":
                    raise ValueError("DOCUMENT_NOT_INDEXED")
                active = conn.execute(
                    """
                    SELECT 1 FROM index_jobs
                    WHERE document_id = ?
                      AND status IN ('PENDING', 'PARSING', 'CHUNKING', 'EMBEDDING')
                    LIMIT 1
                    """,
                    (int(document_id),),
                ).fetchone()
                if active is not None:
                    raise ValueError("DOCUMENT_BUSY")
                chunk = conn.execute(
                    "SELECT 1 FROM document_chunks WHERE document_id = ? AND index_status = 'INDEXED' LIMIT 1",
                    (int(document_id),),
                ).fetchone()
                if chunk is None:
                    raise ValueError("DOCUMENT_HAS_NO_INDEXED_CHUNKS")
            metadata_json = _publication_metadata_json(
                row["metadata_json"],
                published=published,
                actor_id=int(actor_id),
                action=action,
                content_hash=str(row["content_hash"] or ""),
            )
            conn.execute(
                "UPDATE documents SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (metadata_json, int(document_id)),
            )
            conn.commit()
        document = self.document(int(document_id))
        if document is None:
            raise ValueError("DOCUMENT_NOT_FOUND")
        return document

    def set_bootstrap_document_publication(
        self,
        document_id: int,
        *,
        source_id: str,
        revision: str,
    ) -> dict[str, Any]:
        """Publish an indexed built-in textbook without impersonating a user."""

        normalized_source_id = str(source_id or "").strip()
        normalized_revision = str(revision or "").strip()
        if not normalized_source_id or not normalized_revision:
            raise ValueError("BOOTSTRAP_SOURCE_REQUIRED")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id, index_status, metadata_json, content_hash FROM documents WHERE id = ?",
                (int(document_id),),
            ).fetchone()
            if row is None:
                raise ValueError("DOCUMENT_NOT_FOUND")
            if str(row["index_status"] or "").upper() != "INDEXED":
                raise ValueError("DOCUMENT_NOT_INDEXED")
            active = conn.execute(
                """
                SELECT 1 FROM index_jobs
                WHERE document_id = ?
                  AND status IN ('PENDING', 'PARSING', 'CHUNKING', 'EMBEDDING')
                LIMIT 1
                """,
                (int(document_id),),
            ).fetchone()
            if active is not None:
                raise ValueError("DOCUMENT_BUSY")
            chunk = conn.execute(
                "SELECT 1 FROM document_chunks WHERE document_id = ? AND index_status = 'INDEXED' LIMIT 1",
                (int(document_id),),
            ).fetchone()
            if chunk is None:
                raise ValueError("DOCUMENT_HAS_NO_INDEXED_CHUNKS")
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["source_id"] = normalized_source_id
            metadata["revision"] = normalized_revision
            metadata_json = _publication_metadata_json(
                _json_dumps(metadata),
                published=True,
                actor_id=None,
                action="bootstrap_publish",
                content_hash=str(row["content_hash"] or ""),
            )
            conn.execute(
                "UPDATE documents SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (metadata_json, int(document_id)),
            )
            conn.commit()
        document = self.document(int(document_id))
        if document is None:
            raise ValueError("DOCUMENT_NOT_FOUND")
        return document

    def create_index_job(
        self,
        filename: str,
        content_hash: str,
        status: str = "PENDING",
        document_id: int | None = None,
    ) -> int:
        _validate_index_status(status)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO index_jobs (document_id, filename, content_hash, status)
                VALUES (?, ?, ?, ?)
                """,
                (document_id, filename, content_hash, status),
            )
        return int(cursor.lastrowid)

    def update_index_job(
        self,
        job_id: int,
        status: str,
        document_id: int | None = None,
        error_message: str | None = None,
    ) -> None:
        _validate_index_status(status)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE index_jobs
                SET status = ?,
                    document_id = COALESCE(?, document_id),
                    error_message = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, document_id, error_message, job_id),
            )

    def fail_index_job(
        self,
        job_id: int,
        error_message: str,
        document_id: int | None = None,
    ) -> None:
        self.update_index_job(
            job_id=job_id,
            status="FAILED",
            document_id=document_id,
            error_message=error_message,
        )

    def complete_pending_index_jobs_for_document(
        self,
        document_id: int,
        *,
        status: str = "INDEXED",
    ) -> None:
        _validate_index_status(status)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE index_jobs
                SET status = ?,
                    error_message = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                  AND status IN ('PENDING', 'PARSING', 'CHUNKING', 'EMBEDDING')
                """,
                (status, document_id),
            )

    def index_jobs(self, document_id: int | None = None) -> list[dict[str, Any]]:
        if document_id is None:
            sql = """
                SELECT id, document_id, filename, content_hash, status, error_message,
                       created_at, updated_at
                FROM index_jobs
                ORDER BY id DESC
            """
            params: tuple[Any, ...] = ()
        else:
            sql = """
                SELECT id, document_id, filename, content_hash, status, error_message,
                       created_at, updated_at
                FROM index_jobs
                WHERE document_id = ?
                ORDER BY id DESC
            """
            params = (document_id,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def chunks_for_document(self, document_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, document_id, chunk_index, chunk_text, chunk_hash, token_count,
                       page, section, source_type, index_status, created_at, updated_at
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def document_chunks(self, only_pending: bool = False, course_id: int | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if only_pending:
            clauses.append("c.index_status != 'INDEXED'")
        if course_id is not None:
            clauses.append("d.course_id = ?")
            params.append(course_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id, c.document_id, c.chunk_index, c.chunk_text, c.chunk_hash,
                       c.token_count, c.page, c.section, c.source_type, c.index_status,
                        d.filename, d.title, d.source_path, d.file_type, d.content_hash,
                        d.course_id, d.class_id, d.index_status AS document_index_status,
                        d.metadata_json
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                {where}
                ORDER BY c.document_id ASC, c.chunk_index ASC
                """,
                tuple(params),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw = item.pop("metadata_json", None)
            try:
                metadata = json.loads(raw or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            item["published_to_students"] = (
                isinstance(metadata, dict) and metadata.get("published_to_students") is True
            )
            result.append(item)
        return result

    def mark_document_chunks_indexed(self, chunk_ids: list[int]) -> None:
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE document_chunks
                SET index_status = 'INDEXED', updated_at = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
                """,
                chunk_ids,
            )
            document_ids = [
                int(row["document_id"])
                for row in conn.execute(
                    f"""
                    SELECT DISTINCT document_id
                    FROM document_chunks
                    WHERE id IN ({placeholders})
                    """,
                    chunk_ids,
                ).fetchall()
            ]
            if document_ids:
                document_placeholders = ",".join("?" for _ in document_ids)
                conn.execute(
                    f"""
                    UPDATE documents
                    SET index_status = 'INDEXED', updated_at = CURRENT_TIMESTAMP
                    WHERE id IN ({document_placeholders})
                    """,
                    document_ids,
                )

    def delete_document(self, document_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def latest_question_id(self) -> int | None:
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(id) AS id FROM questions").fetchone()
        return int(row["id"]) if row and row["id"] is not None else None

    def create_innovation_session(
        self,
        *,
        user_id: int,
        topic: str,
        current_plan: str,
        cases: list[dict[str, Any]],
        limitations: list[str],
        questions: list[str],
        course_id: int | None = None,
        class_id: int | None = None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO innovation_sessions (
                    user_id, course_id, class_id, topic, current_plan,
                    status, cases_json, limitations_json, questions_json
                )
                VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)
                """,
                (
                    user_id,
                    course_id or _ensure_demo_course(conn),
                    class_id,
                    topic,
                    current_plan,
                    _json_dumps(cases),
                    _json_dumps(limitations),
                    _json_dumps(questions),
                ),
            )
            session_id = int(cursor.lastrowid)
            for index, question in enumerate(questions, 1):
                conn.execute(
                    """
                    INSERT INTO innovation_questions (session_id, question_index, question)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, index, question),
                )
            row = conn.execute("SELECT * FROM innovation_sessions WHERE id = ?", (session_id,)).fetchone()
        return _innovation_session_row(dict(row))

    def innovation_session(self, session_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM innovation_sessions WHERE id = ?", (session_id,)).fetchone()
        return _innovation_session_row(dict(row)) if row else None

    def innovation_session_summaries(
        self, *, user_id: int, course_id: int, class_id: int | None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Only scoped selection metadata; never return stored case or answer JSON."""
        bounded_limit = max(1, min(int(limit), 20))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, topic, status, created_at, updated_at
                FROM innovation_sessions
                WHERE user_id = ? AND course_id = ? AND class_id IS ?
                  AND status IN ('ACTIVE', 'FINALIZED')
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (user_id, course_id, class_id, bounded_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def innovation_questions(self, session_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM innovation_questions
                WHERE session_id = ?
                ORDER BY question_index ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def answer_innovation_question(
        self,
        *,
        session_id: int,
        question_id: int,
        answer: str,
        follow_up_question: str | None = None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE innovation_questions
                SET answer = ?, answered_at = CURRENT_TIMESTAMP
                WHERE id = ? AND session_id = ?
                """,
                (answer, question_id, session_id),
            )
            updated = conn.execute("SELECT changes() AS count").fetchone()
            if int(updated["count"]) == 0:
                raise ValueError("innovation question does not belong to this session")
            if follow_up_question:
                row = conn.execute(
                    "SELECT COALESCE(MAX(question_index), 0) + 1 AS next_index FROM innovation_questions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                conn.execute(
                    """
                    INSERT INTO innovation_questions (session_id, question_index, question)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, int(row["next_index"]), follow_up_question),
                )
                questions = [
                    item["question"]
                    for item in conn.execute(
                        "SELECT question FROM innovation_questions WHERE session_id = ? ORDER BY question_index",
                        (session_id,),
                    ).fetchall()
                ]
                conn.execute(
                    """
                    UPDATE innovation_sessions
                    SET questions_json = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (_json_dumps(questions), session_id),
                )
            row = conn.execute(
                "SELECT * FROM innovation_questions WHERE id = ? AND session_id = ?",
                (question_id, session_id),
            ).fetchone()
        return dict(row)

    def finalize_innovation_session(self, session_id: int, ideas: list[dict[str, Any]]) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE innovation_sessions
                SET status = 'FINALIZED', final_ideas_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (_json_dumps(ideas), session_id),
            )
            conn.execute("DELETE FROM innovation_ideas WHERE session_id = ?", (session_id,))
            conn.executemany(
                """
                INSERT INTO innovation_ideas (
                    session_id, title, hypothesis, experiment_method,
                    risks_json, evidence_json, model_suggestion
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        idea.get("title", ""),
                        idea.get("hypothesis", ""),
                        idea.get("experiment_method", ""),
                        _json_dumps(idea.get("risks", [])),
                        _json_dumps(idea.get("evidence", [])),
                        1 if idea.get("model_suggestion") else 0,
                    )
                    for idea in ideas
                ],
            )
            row = conn.execute("SELECT * FROM innovation_sessions WHERE id = ?", (session_id,)).fetchone()
        return _innovation_session_row(dict(row))

    def register_question_cluster_for_processing(
        self,
        *,
        question_id: int,
        course_id: int,
        class_id: int | None,
        threshold: int,
    ) -> bool:
        normalized_class_id = _normalize_cluster_class_id(class_id)
        threshold = max(1, int(threshold))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT status, pending_question_count
                FROM question_cluster_jobs
                WHERE course_id = ? AND class_id = ?
                """,
                (int(course_id), normalized_class_id),
            ).fetchone()
            is_new_state = row is None
            if is_new_state:
                conn.execute(
                    """
                    INSERT INTO question_cluster_jobs (
                        course_id, class_id, pending_question_count, status
                    )
                    VALUES (?, ?, 1, 'IDLE')
                    """,
                    (int(course_id), normalized_class_id),
                )
                pending_count = 1
                current_status = "IDLE"
            else:
                pending_count = _unprocessed_question_count(
                    conn,
                    course_id=int(course_id),
                    class_id=normalized_class_id,
                )
                current_status = str(row["status"])
            ready = pending_count >= threshold
            if current_status == "FAILED" and pending_count > 0:
                ready = True
            if ready:
                next_status = "PENDING"
            elif pending_count == 0 and current_status not in {"RUNNING", "FAILED"}:
                next_status = "SUCCEEDED"
            elif current_status == "SUCCEEDED":
                next_status = "IDLE"
            else:
                next_status = current_status
            conn.execute(
                """
                UPDATE question_cluster_jobs
                SET pending_question_count = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE course_id = ? AND class_id = ?
                """,
                (pending_count, next_status, int(course_id), normalized_class_id),
            )
        return ready

    def claim_question_cluster_job(
        self,
        course_id: int,
        class_id: int | None,
        *,
        worker_id: str,
    ) -> dict[str, Any] | None:
        normalized_class_id = _normalize_cluster_class_id(class_id)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT *
                FROM question_cluster_jobs
                WHERE course_id = ? AND class_id = ? AND status = 'PENDING'
                """,
                (int(course_id), normalized_class_id),
            ).fetchone()
            if row is None:
                return None
            updated = conn.execute(
                """
                UPDATE question_cluster_jobs
                SET status = 'RUNNING',
                    claimed_by = ?,
                    claimed_at = CURRENT_TIMESTAMP,
                    started_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'PENDING'
                """,
                (worker_id, int(row["id"])),
            )
            if updated.rowcount != 1:
                return None
            claimed = conn.execute(
                "SELECT * FROM question_cluster_jobs WHERE id = ?",
                (int(row["id"]),),
            ).fetchone()
        return _question_cluster_job_row(dict(claimed)) if claimed else None

    def complete_question_cluster_job(
        self,
        course_id: int,
        class_id: int | None,
        *,
        worker_id: str,
    ) -> None:
        normalized_class_id = _normalize_cluster_class_id(class_id)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT id, claimed_by
                FROM question_cluster_jobs
                WHERE course_id = ? AND class_id = ? AND status = 'RUNNING'
                """,
                (int(course_id), normalized_class_id),
            ).fetchone()
            if row is None or str(row["claimed_by"] or "") != str(worker_id):
                return
            max_question_id = _max_question_id_for_scope(
                conn,
                course_id=int(course_id),
                class_id=normalized_class_id,
            )
            conn.execute(
                """
                UPDATE question_cluster_jobs
                SET last_successful_question_id = ?,
                    pending_question_count = 0,
                    status = 'SUCCEEDED',
                    error_code = NULL,
                    error_message = NULL,
                    claimed_by = NULL,
                    claimed_at = NULL,
                    finished_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'RUNNING' AND claimed_by = ?
                """,
                (
                    max_question_id,
                    int(row["id"]),
                    worker_id,
                ),
            )

    def fail_question_cluster_job(
        self,
        course_id: int,
        class_id: int | None,
        *,
        worker_id: str,
        error_code: str,
        error_message: str,
    ) -> None:
        normalized_class_id = _normalize_cluster_class_id(class_id)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE question_cluster_jobs
                SET status = 'FAILED',
                    error_code = ?,
                    error_message = ?,
                    claimed_by = NULL,
                    claimed_at = NULL,
                    finished_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE course_id = ? AND class_id = ?
                  AND status = 'RUNNING' AND claimed_by = ?
                """,
                (
                    str(error_code)[:80],
                    str(error_message)[:240],
                    int(course_id),
                    normalized_class_id,
                    worker_id,
                ),
            )

    def recover_question_cluster_jobs(self, *, threshold: int) -> list[tuple[int, int | None]]:
        threshold = max(1, int(threshold))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            scope_rows = conn.execute(
                """
                SELECT course_id, class_id
                FROM question_cluster_jobs
                UNION
                SELECT DISTINCT course_id, COALESCE(class_id, 0)
                FROM questions
                WHERE course_id IS NOT NULL
                """
            ).fetchall()
            ready: list[tuple[int, int | None]] = []
            for scope in scope_rows:
                course_id = int(scope["course_id"])
                normalized_class_id = int(scope["class_id"] or 0)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO question_cluster_jobs (
                        course_id, class_id, status
                    )
                    VALUES (?, ?, 'IDLE')
                    """,
                    (course_id, normalized_class_id),
                )
                pending_count = _unprocessed_question_count(
                    conn,
                    course_id=course_id,
                    class_id=normalized_class_id,
                )
                row = conn.execute(
                    """
                    SELECT status
                    FROM question_cluster_jobs
                    WHERE course_id = ? AND class_id = ?
                    """,
                    (course_id, normalized_class_id),
                ).fetchone()
                current_status = str(row["status"]) if row else "IDLE"
                if current_status in {"RUNNING", "FAILED"} and pending_count > 0:
                    next_status = "PENDING"
                elif pending_count >= threshold:
                    next_status = "PENDING"
                elif pending_count == 0 and current_status != "FAILED":
                    next_status = "SUCCEEDED"
                else:
                    next_status = current_status
                conn.execute(
                    """
                    UPDATE question_cluster_jobs
                    SET pending_question_count = ?, status = ?,
                        claimed_by = NULL, claimed_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE course_id = ? AND class_id = ?
                    """,
                    (pending_count, next_status, course_id, normalized_class_id),
                )
                if next_status == "PENDING" and (
                    pending_count >= threshold or current_status in {"RUNNING", "FAILED"}
                ):
                    ready.append((course_id, None if normalized_class_id == 0 else normalized_class_id))
        return ready

    def question_cluster_jobs(
        self,
        *,
        course_id: int | None = None,
        class_id: int | None = None,
        class_ids: tuple[int, ...] | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if course_id is not None:
            clauses.append("course_id = ?")
            params.append(int(course_id))
        if class_id is not None:
            clauses.append("class_id = ?")
            params.append(_normalize_cluster_class_id(class_id))
        elif class_ids is not None:
            if not class_ids:
                return []
            placeholders = ",".join("?" for _ in class_ids)
            clauses.append(f"class_id IN ({placeholders})")
            params.extend(_normalize_cluster_class_id(item) for item in class_ids)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM question_cluster_jobs
                {where}
                ORDER BY course_id ASC, class_id ASC
                """,
                params,
            ).fetchall()
        return [_question_cluster_job_row(dict(row)) for row in rows]

    def questions_for_clustering(
        self,
        *,
        course_id: int | None = None,
        class_id: int | None = None,
        class_ids: tuple[int, ...] | None = None,
        start_at: str | None = None,
        end_at: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if course_id is not None:
            clauses.append("course_id = ?")
            params.append(course_id)
        if class_id is not None:
            clauses.append("class_id = ?")
            params.append(class_id)
        elif class_ids is not None:
            if not class_ids:
                return []
            placeholders = ",".join("?" for _ in class_ids)
            clauses.append(f"class_id IN ({placeholders})")
            params.extend(int(item) for item in class_ids)
        if start_at:
            clauses.append("datetime(created_at) >= datetime(?)")
            params.append(start_at)
        if end_at:
            clauses.append("datetime(created_at) <= datetime(?)")
            params.append(end_at)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, user_id, question, module, intent, course_id, class_id, created_at
                FROM questions
                {where}
                ORDER BY id ASC
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def protected_cluster_question_ids(
        self,
        course_id: int | None = None,
        *,
        class_id: int | None = None,
        class_ids: tuple[int, ...] | None = None,
    ) -> set[int]:
        clauses = ["(c.approved = 1 OR c.source = 'teacher' OR m.manual_adjusted = 1)"]
        params: list[Any] = []
        if course_id is not None:
            clauses.append("c.course_id = ?")
            params.append(course_id)
        if class_id is not None:
            clauses.append("c.class_id = ?")
            params.append(class_id)
        elif class_ids is not None:
            if not class_ids:
                return set()
            placeholders = ",".join("?" for _ in class_ids)
            clauses.append(f"c.class_id IN ({placeholders})")
            params.extend(int(item) for item in class_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT m.question_id
                FROM question_cluster_members m
                JOIN question_clusters c ON c.id = m.cluster_id
                WHERE {' AND '.join(clauses)}
                """,
                params,
            ).fetchall()
        return {int(row["question_id"]) for row in rows}

    def approved_cluster_question_ids(self, course_id: int | None = None) -> set[int]:
        return self.protected_cluster_question_ids(course_id=course_id)

    def delete_unapproved_question_clusters(
        self,
        course_id: int | None = None,
        *,
        class_id: int | None = None,
        class_ids: tuple[int, ...] | None = None,
    ) -> None:
        clauses = ["approved = 0", "source = 'auto'"]
        params: list[Any] = []
        if course_id is not None:
            clauses.append("course_id = ?")
            params.append(course_id)
        if class_id is not None:
            clauses.append("class_id = ?")
            params.append(class_id)
        elif class_ids is not None:
            if not class_ids:
                return
            placeholders = ",".join("?" for _ in class_ids)
            clauses.append(f"class_id IN ({placeholders})")
            params.extend(int(item) for item in class_ids)
        where = f"WHERE {' AND '.join(clauses)}"
        with self._connect() as conn:
            conn.execute(f"DELETE FROM question_clusters {where}", params)

    def replace_auto_question_clusters(
        self,
        clusters: list[dict[str, Any]],
        *,
        course_id: int | None = None,
        class_id: int | None = None,
        class_ids: tuple[int, ...] | None = None,
    ) -> list[int]:
        clauses = ["approved = 0", "source = 'auto'"]
        params: list[Any] = []
        if course_id is not None:
            clauses.append("course_id = ?")
            params.append(course_id)
        if class_id is not None:
            clauses.append("class_id = ?")
            params.append(class_id)
        elif class_ids is not None:
            if not class_ids:
                return []
            placeholders = ",".join("?" for _ in class_ids)
            clauses.append(f"class_id IN ({placeholders})")
            params.extend(int(item) for item in class_ids)
        where = f"WHERE {' AND '.join(clauses)}"
        created_ids: list[int] = []
        with self._connect() as conn:
            conn.execute(f"DELETE FROM question_clusters {where}", params)
            for cluster in clusters:
                cursor = conn.execute(
                    """
                    INSERT INTO question_clusters (
                        course_id, class_id, label, representative_question,
                        approved, is_noise, source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cluster.get("course_id"),
                        cluster.get("class_id"),
                        cluster["label"],
                        cluster["representative_question"],
                        int(bool(cluster.get("approved", False))),
                        int(bool(cluster.get("is_noise", False))),
                        cluster.get("source", "auto"),
                    ),
                )
                cluster_id = int(cursor.lastrowid)
                created_ids.append(cluster_id)
                for member in cluster.get("members", []):
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO question_cluster_members (
                            cluster_id, question_id, distance, is_representative, manual_adjusted
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            cluster_id,
                            int(member["question_id"]),
                            float(member.get("distance", 0.0)),
                            int(bool(member.get("is_representative", False))),
                            int(bool(member.get("manual_adjusted", False))),
                        ),
                    )
        return created_ids

    def create_question_cluster(
        self,
        *,
        label: str,
        representative_question: str,
        course_id: int | None,
        class_id: int | None = None,
        approved: bool = False,
        is_noise: bool = False,
        source: str = "auto",
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO question_clusters (
                    course_id, class_id, label, representative_question,
                    approved, is_noise, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (course_id, class_id, label, representative_question, int(approved), int(is_noise), source),
            )
        return int(cursor.lastrowid)

    def add_question_cluster_member(
        self,
        *,
        cluster_id: int,
        question_id: int,
        distance: float = 0.0,
        is_representative: bool = False,
        manual_adjusted: bool = False,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO question_cluster_members (
                    cluster_id, question_id, distance, is_representative, manual_adjusted
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (cluster_id, question_id, distance, int(is_representative), int(manual_adjusted)),
            )
            conn.execute("UPDATE question_clusters SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (cluster_id,))

    def question_clusters(
        self,
        course_id: int | None = None,
        *,
        class_id: int | None = None,
        class_ids: tuple[int, ...] | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if course_id is not None:
            clauses.append("c.course_id = ?")
            params.append(course_id)
        if class_id is not None:
            clauses.append("c.class_id = ?")
            params.append(class_id)
        elif class_ids is not None:
            if not class_ids:
                return []
            placeholders = ",".join("?" for _ in class_ids)
            clauses.append(f"c.class_id IN ({placeholders})")
            params.extend(int(item) for item in class_ids)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            clusters = conn.execute(
                f"""
                SELECT c.*
                FROM question_clusters c
                {where}
                ORDER BY c.approved DESC, c.id ASC
                """,
                params,
            ).fetchall()
            result = []
            for row in clusters:
                cluster = _question_cluster_row(dict(row))
                members = conn.execute(
                    """
                    SELECT m.*, q.question, q.module, q.intent, q.course_id, q.class_id, q.created_at
                    FROM question_cluster_members m
                    JOIN questions q ON q.id = m.question_id
                    WHERE m.cluster_id = ?
                    ORDER BY m.is_representative DESC, m.distance ASC, m.question_id ASC
                    """,
                    (cluster["id"],),
                ).fetchall()
                cluster["members"] = [dict(member) for member in members]
                cluster["count"] = len(cluster["members"])
                result.append(cluster)
        return result

    def rename_question_cluster(self, cluster_id: int, label: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE question_clusters
                SET label = ?, source = 'teacher', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (label, cluster_id),
            )

    def approve_question_cluster(self, cluster_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE question_clusters
                SET approved = 1, source = CASE WHEN source = 'auto' THEN 'teacher' ELSE source END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (cluster_id,),
            )

    def merge_question_clusters(self, target_cluster_id: int, source_cluster_ids: list[int], label: str | None = None) -> None:
        with self._connect() as conn:
            for source_id in source_cluster_ids:
                if int(source_id) == int(target_cluster_id):
                    continue
                rows = conn.execute(
                    "SELECT question_id, distance, is_representative FROM question_cluster_members WHERE cluster_id = ?",
                    (source_id,),
                ).fetchall()
                for row in rows:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO question_cluster_members (
                            cluster_id, question_id, distance, is_representative, manual_adjusted
                        )
                        VALUES (?, ?, ?, ?, 1)
                        """,
                        (target_cluster_id, row["question_id"], row["distance"], row["is_representative"]),
                    )
                conn.execute("DELETE FROM question_clusters WHERE id = ?", (source_id,))
            if label:
                conn.execute("UPDATE question_clusters SET label = ? WHERE id = ?", (label, target_cluster_id))
            conn.execute(
                """
                UPDATE question_clusters
                SET source = 'teacher', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (target_cluster_id,),
            )

    def split_question_cluster(self, cluster_id: int, question_ids: list[int], label: str) -> int:
        with self._connect() as conn:
            parent = conn.execute("SELECT * FROM question_clusters WHERE id = ?", (cluster_id,)).fetchone()
            if parent is None:
                raise ValueError("cluster not found")
            cursor = conn.execute(
                """
                INSERT INTO question_clusters (
                    course_id, class_id, label, representative_question, approved, is_noise, source
                )
                VALUES (?, ?, ?, ?, 0, 0, 'teacher')
                """,
                (parent["course_id"], parent["class_id"], label, label),
            )
            new_id = int(cursor.lastrowid)
            for question_id in question_ids:
                row = conn.execute(
                    "SELECT distance FROM question_cluster_members WHERE cluster_id = ? AND question_id = ?",
                    (cluster_id, question_id),
                ).fetchone()
                if row is None:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO question_cluster_members (
                        cluster_id, question_id, distance, is_representative, manual_adjusted
                    )
                    VALUES (?, ?, ?, 0, 1)
                    """,
                    (new_id, question_id, row["distance"]),
                )
                conn.execute(
                    "DELETE FROM question_cluster_members WHERE cluster_id = ? AND question_id = ?",
                    (cluster_id, question_id),
                )
                conn.execute("UPDATE question_clusters SET source = 'teacher', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (cluster_id,))
        return new_id

    def ensure_demo_assignments(self) -> None:
        with self._connect() as conn:
            course_id = _ensure_demo_course(conn)
            existing = conn.execute("SELECT id FROM assignments LIMIT 1").fetchone()
            if existing:
                return
            cursor = conn.execute(
                """
                INSERT INTO assignments (course_id, module, title, description, visibility)
                VALUES (?, '5G NAS', '5G NAS 安全实验', '围绕 NAS 认证、重放风险和实验评测完成课程实践。', 'course')
                """,
                (course_id,),
            )
            assignment_id = int(cursor.lastrowid)
            steps = [
                ("选题与目标", "明确研究问题、评价指标和安全边界。"),
                ("资料检索", "检索论文、标准、漏洞和课程资料，形成证据表。"),
                ("实验设计", "设计基线组、改进组、变量和记录方式。"),
                ("结果分析", "分析实验结果、失败案例和创新点。"),
            ]
            conn.executemany(
                """
                INSERT INTO assignment_steps (assignment_id, step_index, title, description, module)
                VALUES (?, ?, ?, ?, '5G NAS')
                """,
                [(assignment_id, index, title, description) for index, (title, description) in enumerate(steps, 1)],
            )

    def assignments(self) -> list[dict[str, Any]]:
        self.ensure_demo_assignments()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM assignments
                ORDER BY id ASC
                """
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                steps = conn.execute(
                    "SELECT * FROM assignment_steps WHERE assignment_id = ? ORDER BY step_index ASC",
                    (item["id"],),
                ).fetchall()
                item["steps"] = [dict(step) for step in steps]
                result.append(item)
        return result

    def assignment(self, assignment_id: int) -> dict[str, Any] | None:
        matches = [item for item in self.assignments() if int(item["id"]) == int(assignment_id)]
        return matches[0] if matches else None

    def assignment_step(self, step_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM assignment_steps WHERE id = ?", (step_id,)).fetchone()
        return dict(row) if row else None

    def upsert_assignment_progress(
        self,
        *,
        user_id: int,
        assignment_id: int,
        step_id: int,
        status: str,
        progress_percent: int,
        note: str = "",
    ) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO student_assignment_progress (
                    user_id, assignment_id, step_id, status, progress_percent, note
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, assignment_id)
                DO UPDATE SET step_id = excluded.step_id, status = excluded.status,
                    progress_percent = excluded.progress_percent, note = excluded.note,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, assignment_id, step_id, status, progress_percent, note),
            )
            row = conn.execute(
                "SELECT * FROM student_assignment_progress WHERE user_id = ? AND assignment_id = ?",
                (user_id, assignment_id),
            ).fetchone()
        return dict(row)

    def assignment_progress(self, *, user_id: int, assignment_id: int | None = None) -> list[dict[str, Any]]:
        clauses = ["p.user_id = ?"]
        params: list[Any] = [user_id]
        if assignment_id is not None:
            clauses.append("p.assignment_id = ?")
            params.append(assignment_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT p.*, a.title AS assignment_title, s.title AS step_title, s.module
                FROM student_assignment_progress p
                JOIN assignments a ON a.id = p.assignment_id
                JOIN assignment_steps s ON s.id = p.step_id
                WHERE {' AND '.join(clauses)}
                ORDER BY p.updated_at DESC
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def create_stuck_report(
        self,
        *,
        user_id: int,
        assignment_id: int,
        step_id: int,
        module: str,
        reason: str,
        source: str = "active",
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO stuck_reports (
                    user_id, assignment_id, step_id, module, reason, source, status, evidence_json
                )
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (user_id, assignment_id, step_id, module, reason, source, _json_dumps(evidence or {})),
            )
            row = conn.execute("SELECT * FROM stuck_reports WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _stuck_report_row(dict(row))

    def stuck_reports(self, status: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE r.status = ?" if status else ""
        params = (status,) if status else ()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT r.*, a.title AS assignment_title, s.title AS step_title
                FROM stuck_reports r
                JOIN assignments a ON a.id = r.assignment_id
                JOIN assignment_steps s ON s.id = r.step_id
                {where}
                ORDER BY r.created_at DESC
                """,
                params,
            ).fetchall()
        return [_stuck_report_row(dict(row)) for row in rows]

    def update_stuck_report_status(self, report_id: int, status: str) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute(
                "UPDATE stuck_reports SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, report_id),
            )
            row = conn.execute("SELECT * FROM stuck_reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            raise ValueError("stuck report not found")
        return _stuck_report_row(dict(row))

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM stuck_reports")
            conn.execute("DELETE FROM student_assignment_progress")
            conn.execute("DELETE FROM assignment_steps")
            conn.execute("DELETE FROM assignments")
            conn.execute("DELETE FROM question_cluster_members")
            conn.execute("DELETE FROM question_clusters")
            conn.execute("DELETE FROM innovation_ideas")
            conn.execute("DELETE FROM innovation_questions")
            conn.execute("DELETE FROM innovation_sessions")
            conn.execute("DELETE FROM connector_sync_runs")
            conn.execute("DELETE FROM external_resources")
            conn.execute("DELETE FROM learning_events")
            conn.execute("DELETE FROM learning_sessions")
            conn.execute("DELETE FROM agent_traces")
            conn.execute("DELETE FROM tool_calls")
            conn.execute("DELETE FROM model_calls")
            conn.execute("DELETE FROM agent_steps")
            conn.execute("DELETE FROM agent_runs")
            conn.execute("DELETE FROM student_feedback")
            conn.execute("DELETE FROM retrieval_logs")
            conn.execute("DELETE FROM faq_items")
            conn.execute("DELETE FROM questions")
            conn.execute("DELETE FROM index_jobs")
            conn.execute("DELETE FROM documents")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def _summary(text: str, max_chars: int = 160) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _preview(text: str, max_chars: int = 60) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized_resource_query(query: dict[str, Any], *, source_name: str) -> dict[str, Any]:
    normalized = {
        "source_name": str(source_name or ""),
        "query": " ".join(str(query.get("query") or "").lower().split()),
        "resource_type": query.get("resource_type") or None,
        "module": str(query.get("module") or "").strip(),
        "year_from": query.get("year_from"),
        "year_to": query.get("year_to"),
        "page": max(1, int(query.get("page") or 1)),
        "page_size": max(1, min(50, int(query.get("page_size") or 10))),
        "lookup_mode": str(query.get("lookup_mode") or "SEARCH").upper(),
        "canonical_identifier": str(query.get("canonical_identifier") or "") or None,
    }
    if "incomplete_results" in query:
        normalized["incomplete_results"] = query.get("incomplete_results")
    return normalized


def _resource_query_key(query: dict[str, Any], *, source_name: str) -> str:
    normalized = _normalized_resource_query(query, source_name=source_name)
    normalized.pop("incomplete_results", None)
    return sha256_text(_json_dumps(normalized))


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _prepare_chunks(document: Document, chunks: list[str | dict[str, Any]] | None) -> list[dict[str, Any]]:
    if chunks is None:
        raw_chunks: list[dict[str, Any]] = []
        for page in document.pages:
            for chunk_text in _split_text(page.text):
                raw_chunks.append(
                    {
                        "chunk_text": chunk_text,
                        "page": page.page,
                        "section": page.section,
                        "source_type": page.source_type,
                    }
                )
    else:
        raw_chunks = [
            item if isinstance(item, dict) else {"chunk_text": item}
            for item in chunks
        ]

    prepared: list[dict[str, Any]] = []
    for index, chunk in enumerate(raw_chunks):
        chunk_text = " ".join(str(chunk.get("chunk_text", "")).split())
        if not chunk_text:
            continue
        prepared.append(
            {
                "chunk_index": index,
                "chunk_text": chunk_text,
                "chunk_hash": sha256_text(chunk_text),
                "token_count": len(chunk_text.split()),
                "page": chunk.get("page"),
                "section": chunk.get("section", ""),
                "source_type": chunk.get("source_type") or document.file_type,
            }
        )
    return prepared


def _split_text(text: str, max_chars: int = 800) -> list[str]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph
    if current:
        chunks.append(current)
    return chunks


def _validate_index_status(status: str) -> None:
    if status not in VALID_INDEX_STATUSES:
        raise ValueError(f"非法索引状态：{status}")


def _run_migration(
    conn: sqlite3.Connection,
    version: int,
    name: str,
    migration: Any,
) -> None:
    exists = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE version = ?",
        (version,),
    ).fetchone()
    if exists:
        return
    migration(conn)
    conn.execute(
        "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
        (version, name),
    )


def _migration_document_metadata_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            source_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            byte_size INTEGER NOT NULL,
            text_length INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            index_status TEXT NOT NULL DEFAULT 'PENDING',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_hash TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            index_status TEXT NOT NULL DEFAULT 'PENDING',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(document_id, chunk_index),
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS index_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            filename TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_index_jobs_document_id ON index_jobs(document_id)")


def _migration_document_chunk_location_fields(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "document_chunks", "page", "INTEGER")
    _ensure_column(conn, "document_chunks", "section", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "document_chunks", "source_type", "TEXT NOT NULL DEFAULT ''")


def _migration_agent_run_trace_tables(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "questions", "run_id", "TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL UNIQUE,
            question_id INTEGER,
            user_id TEXT NOT NULL,
            goal TEXT NOT NULL,
            intent TEXT NOT NULL,
            status TEXT NOT NULL,
            current_step INTEGER NOT NULL DEFAULT 0,
            max_steps INTEGER NOT NULL,
            tool_call_count INTEGER NOT NULL DEFAULT 0,
            max_tool_calls INTEGER NOT NULL,
            state_json TEXT NOT NULL DEFAULT '{}',
            pending_question TEXT,
            final_answer TEXT,
            stop_reason TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step_id TEXT NOT NULL UNIQUE,
            run_id TEXT NOT NULL,
            step_number INTEGER NOT NULL,
            action TEXT NOT NULL,
            status TEXT NOT NULL,
            decision_json TEXT NOT NULL DEFAULT '{}',
            observation_json TEXT NOT NULL DEFAULT '{}',
            trace_id TEXT,
            error_code TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER,
            run_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            purpose TEXT NOT NULL,
            success INTEGER NOT NULL,
            error_code TEXT,
            error_message TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE SET NULL,
            FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE,
            FOREIGN KEY(step_id) REFERENCES agent_steps(step_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER,
            run_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            arguments_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT NOT NULL DEFAULT '{}',
            success INTEGER NOT NULL,
            latency_ms REAL NOT NULL DEFAULT 0,
            fallback_used INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE SET NULL,
            FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE,
            FOREIGN KEY(step_id) REFERENCES agent_steps(step_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            question_id INTEGER,
            run_id TEXT NOT NULL,
            step_id TEXT,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE SET NULL,
            FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_run_id ON questions(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_steps_run_id ON agent_steps(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model_calls_run_id ON model_calls(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_run_id ON tool_calls(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_traces_run_id ON agent_traces(run_id)")


def _migration_feedback_and_model_metrics(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "model_calls", "latency_ms", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "model_calls", "fallback_used", "INTEGER NOT NULL DEFAULT 0")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS student_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            helpful INTEGER,
            comment TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_student_feedback_question_id ON student_feedback(question_id)")


def _migration_phase3_identity_course_context(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('student', 'teacher', 'admin')),
            enabled INTEGER NOT NULL DEFAULT 1,
            must_change_password INTEGER NOT NULL DEFAULT 1,
            token_version INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_login_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            assessment_method TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            slug TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            term TEXT NOT NULL DEFAULT '',
            teacher_id INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id) ON DELETE CASCADE,
            FOREIGN KEY(teacher_id) REFERENCES users(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS class_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            joined_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(class_id, user_id),
            FOREIGN KEY(class_id) REFERENCES classes(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL UNIQUE,
            user_id TEXT NOT NULL DEFAULT 'demo_student',
            course_id INTEGER,
            class_id INTEGER,
            session_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id) ON DELETE SET NULL,
            FOREIGN KEY(class_id) REFERENCES classes(id) ON DELETE SET NULL
        )
        """
    )
    course_id = _ensure_demo_course(conn)
    class_id = _ensure_demo_class(conn, course_id)
    _ensure_column(conn, "questions", "course_id", "INTEGER")
    _ensure_column(conn, "questions", "class_id", "INTEGER")
    _ensure_column(conn, "questions", "session_id", "TEXT")
    _ensure_column(conn, "agent_runs", "course_id", "INTEGER")
    _ensure_column(conn, "agent_runs", "class_id", "INTEGER")
    _ensure_column(conn, "agent_runs", "session_id", "TEXT")
    _ensure_column(conn, "student_feedback", "course_id", "INTEGER")
    _ensure_column(conn, "documents", "course_id", "INTEGER")
    _ensure_column(conn, "documents", "class_id", "INTEGER")
    _ensure_column(conn, "documents", "uploaded_by_user_id", "INTEGER")
    _ensure_column(conn, "documents", "visibility", "TEXT NOT NULL DEFAULT 'course'")
    _backfill_phase3_context(conn, course_id, class_id)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_course_class ON questions(course_id, class_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_runs_course_class ON agent_runs(course_id, class_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_course_visibility ON documents(course_id, visibility)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_class_members_user ON class_members(user_id)")


def _ensure_demo_course(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT id FROM courses WHERE slug = 'demo-course'").fetchone()
    if row:
        return int(row["id"])
    cursor = conn.execute(
        """
        INSERT INTO courses (slug, name, description, assessment_method)
        VALUES ('demo-course', '网络安全实践课程', '第三阶段默认演示课程。', '过程表现、实验报告和创新方案综合评价。')
        """
    )
    return int(cursor.lastrowid)


def _ensure_demo_class(
    conn: sqlite3.Connection,
    course_id: int,
    *,
    teacher_id: int | None = None,
) -> int:
    row = conn.execute("SELECT id, teacher_id FROM classes WHERE slug = 'demo-class'").fetchone()
    if row:
        if teacher_id is not None and row["teacher_id"] is None:
            conn.execute("UPDATE classes SET teacher_id = ? WHERE id = ?", (teacher_id, row["id"]))
        return int(row["id"])
    cursor = conn.execute(
        """
        INSERT INTO classes (course_id, slug, name, term, teacher_id)
        VALUES (?, 'demo-class', '演示班级', '2026 秋季', ?)
        """,
        (course_id, teacher_id),
    )
    return int(cursor.lastrowid)


def _upsert_user_raw(
    conn: sqlite3.Connection,
    username: str,
    display_name: str,
    password_hash: str,
    role: str,
) -> int:
    row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if row:
        return int(row["id"])
    cursor = conn.execute(
        """
        INSERT INTO users (
            username, display_name, password_hash, role, enabled, must_change_password
        )
        VALUES (?, ?, ?, ?, 1, 1)
        """,
        (username, display_name, password_hash, role),
    )
    return int(cursor.lastrowid)


def _ensure_class_member(conn: sqlite3.Connection, class_id: int, user_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO class_members (class_id, user_id) VALUES (?, ?)",
        (class_id, user_id),
    )


def _resolve_user_id(conn: sqlite3.Connection, user_id: str | int) -> int | None:
    if isinstance(user_id, int) or str(user_id).isdigit():
        row = conn.execute("SELECT id FROM users WHERE id = ?", (int(user_id),)).fetchone()
        return int(row["id"]) if row else None
    row = conn.execute("SELECT id FROM users WHERE username = ?", (str(user_id),)).fetchone()
    return int(row["id"]) if row else None


def _backfill_phase3_context(conn: sqlite3.Connection, course_id: int, class_id: int) -> None:
    conn.execute("UPDATE questions SET course_id = COALESCE(course_id, ?)", (course_id,))
    conn.execute("UPDATE questions SET class_id = COALESCE(class_id, ?)", (class_id,))
    conn.execute("UPDATE agent_runs SET course_id = COALESCE(course_id, ?)", (course_id,))
    conn.execute("UPDATE agent_runs SET class_id = COALESCE(class_id, ?)", (class_id,))
    conn.execute(
        """
        UPDATE student_feedback
        SET course_id = COALESCE(
            course_id,
            (SELECT q.course_id FROM questions q WHERE q.id = student_feedback.question_id),
            ?
        )
        """,
        (course_id,),
    )
    conn.execute("UPDATE documents SET course_id = COALESCE(course_id, ?)", (course_id,))
    conn.execute("UPDATE documents SET class_id = COALESCE(class_id, ?)", (class_id,))
    conn.execute("UPDATE documents SET visibility = COALESCE(NULLIF(visibility, ''), 'course')")


def _migration_learning_sessions_events(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS learning_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            class_id INTEGER,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_active_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            ended_at TEXT,
            active_seconds INTEGER NOT NULL DEFAULT 0,
            client_session_id TEXT NOT NULL UNIQUE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(course_id) REFERENCES courses(id) ON DELETE CASCADE,
            FOREIGN KEY(class_id) REFERENCES classes(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS learning_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            class_id INTEGER,
            event_type TEXT NOT NULL CHECK(event_type IN (
                'LOGIN',
                'PAGE_VIEW',
                'RESOURCE_SEARCH',
                'RESOURCE_OPEN',
                'QUESTION_ASKED',
                'PRACTICE_GUIDE_GENERATED',
                'INNOVATION_SESSION_STARTED',
                'INNOVATION_QUESTION_ANSWERED',
                'ASSIGNMENT_PROGRESS_UPDATED',
                'STUCK_REPORTED',
                'FEEDBACK_SUBMITTED'
            )),
            page_name TEXT NOT NULL DEFAULT '',
            module TEXT NOT NULL DEFAULT '',
            resource_id TEXT,
            assignment_id TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            dedupe_key TEXT UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES learning_sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(course_id) REFERENCES courses(id) ON DELETE CASCADE,
            FOREIGN KEY(class_id) REFERENCES classes(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_sessions_user ON learning_sessions(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_sessions_course_class ON learning_sessions(course_id, class_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_events_user_time ON learning_events(user_id, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_events_course_class ON learning_events(course_id, class_id)")


_CLIENT_LEARNING_EVENT_TYPES = (
    "LOGIN",
    "PAGE_VIEW",
    "RESOURCE_SEARCH",
    "RESOURCE_OPEN",
    "QUESTION_ASKED",
    "PRACTICE_GUIDE_GENERATED",
    "INNOVATION_SESSION_STARTED",
    "INNOVATION_QUESTION_ANSWERED",
    "ASSIGNMENT_PROGRESS_UPDATED",
    "STUCK_REPORTED",
    "FEEDBACK_SUBMITTED",
)

_SERVER_GENERATED_EVENT_TYPES = (
    "GENERAL_QUESTION_ASKED",
    "PRACTICE_DRAFT_GENERATED",
    "INNOVATION_SUGGESTION_GENERATED",
)


def _run_learning_event_generation_migration(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT 1 FROM schema_migrations WHERE version = 17").fetchone():
        return
    conn.commit()
    try:
        conn.execute("BEGIN IMMEDIATE")
        old_count, old_max = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM learning_events"
        ).fetchone()
        old_sequence_row = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'learning_events'"
        ).fetchone()
        old_sequence = int(old_sequence_row[0]) if old_sequence_row is not None else int(old_max)
        conn.execute(
            """
            CREATE TABLE learning_events_v17 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                class_id INTEGER,
                event_type TEXT NOT NULL CHECK(event_type IN (
                    'LOGIN','PAGE_VIEW','RESOURCE_SEARCH','RESOURCE_OPEN','QUESTION_ASKED',
                    'PRACTICE_GUIDE_GENERATED','INNOVATION_SESSION_STARTED',
                    'INNOVATION_QUESTION_ANSWERED','ASSIGNMENT_PROGRESS_UPDATED',
                    'STUCK_REPORTED','FEEDBACK_SUBMITTED','GENERAL_QUESTION_ASKED',
                    'PRACTICE_DRAFT_GENERATED','INNOVATION_SUGGESTION_GENERATED'
                )),
                page_name TEXT NOT NULL DEFAULT '',
                module TEXT NOT NULL DEFAULT '',
                resource_id TEXT,
                assignment_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                dedupe_key TEXT UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES learning_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(course_id) REFERENCES courses(id) ON DELETE CASCADE,
                FOREIGN KEY(class_id) REFERENCES classes(id) ON DELETE SET NULL
            )
            """
        )
        columns = (
            "id,session_id,user_id,course_id,class_id,event_type,page_name,module,"
            "resource_id,assignment_id,metadata_json,dedupe_key,created_at"
        )
        conn.execute(f"INSERT INTO learning_events_v17 ({columns}) SELECT {columns} FROM learning_events")
        new_count, new_max = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM learning_events_v17"
        ).fetchone()
        if (old_count, old_max) != (new_count, new_max):
            raise RuntimeError("learning event migration row verification failed")
        conn.execute("DROP TABLE learning_events")
        conn.execute("ALTER TABLE learning_events_v17 RENAME TO learning_events")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'learning_events_v17'")
        sequence = max(int(old_sequence), int(old_max))
        updated = conn.execute(
            "UPDATE sqlite_sequence SET seq = ? WHERE name = 'learning_events'",
            (sequence,),
        )
        if updated.rowcount == 0:
            conn.execute(
                "INSERT INTO sqlite_sequence(name, seq) VALUES ('learning_events', ?)",
                (sequence,),
            )
        conn.execute("CREATE INDEX idx_learning_events_user_time ON learning_events(user_id, created_at)")
        conn.execute("CREATE INDEX idx_learning_events_course_class ON learning_events(course_id, class_id)")
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("learning event migration foreign key check failed")
        migrated_sequence = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'learning_events'"
        ).fetchone()
        if migrated_sequence is None or int(migrated_sequence[0]) != sequence:
            raise RuntimeError("learning event migration sequence verification failed")
        conn.execute(
            "INSERT INTO schema_migrations (version, name) VALUES (17, 'learning_event_generation_types')"
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _migration_external_resource_connector_cache(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS external_resources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            source_name TEXT NOT NULL,
            title TEXT NOT NULL,
            authors_json TEXT NOT NULL DEFAULT '{}',
            abstract TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '',
            published_at TEXT,
            tags_json TEXT NOT NULL DEFAULT '[]',
            module TEXT NOT NULL DEFAULT '',
            raw_metadata_json TEXT NOT NULL DEFAULT '{}',
            fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            rights_note TEXT NOT NULL DEFAULT '',
            indexed INTEGER NOT NULL DEFAULT 0,
            vector_id TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_name, external_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS connector_sync_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            connector_name TEXT NOT NULL,
            query TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL,
            result_count INTEGER NOT NULL DEFAULT 0,
            cache_hit INTEGER NOT NULL DEFAULT 0,
            error_code TEXT,
            error_message TEXT,
            latency_ms REAL NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT
        )
        """
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_external_resources_source_id ON external_resources(source_name, external_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_external_resources_type_module ON external_resources(resource_type, module)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_external_resources_expires_at ON external_resources(expires_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_connector_sync_runs_name ON connector_sync_runs(connector_name, started_at)")


def _migration_innovation_sessions(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS innovation_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            course_id INTEGER,
            class_id INTEGER,
            topic TEXT NOT NULL,
            current_plan TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            cases_json TEXT NOT NULL DEFAULT '[]',
            limitations_json TEXT NOT NULL DEFAULT '[]',
            questions_json TEXT NOT NULL DEFAULT '[]',
            final_ideas_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(course_id) REFERENCES courses(id),
            FOREIGN KEY(class_id) REFERENCES classes(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS innovation_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            question_index INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL DEFAULT '',
            answered_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES innovation_sessions(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS innovation_ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            experiment_method TEXT NOT NULL,
            risks_json TEXT NOT NULL DEFAULT '[]',
            evidence_json TEXT NOT NULL DEFAULT '[]',
            model_suggestion INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES innovation_sessions(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_innovation_sessions_user ON innovation_sessions(user_id, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_innovation_questions_session ON innovation_questions(session_id, question_index)")


def _migration_question_semantic_clusters(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS question_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER,
            class_id INTEGER,
            label TEXT NOT NULL,
            representative_question TEXT NOT NULL,
            approved INTEGER NOT NULL DEFAULT 0,
            is_noise INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT 'auto',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id),
            FOREIGN KEY(class_id) REFERENCES classes(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS question_cluster_members (
            cluster_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            distance REAL NOT NULL DEFAULT 0,
            is_representative INTEGER NOT NULL DEFAULT 0,
            manual_adjusted INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(cluster_id, question_id),
            FOREIGN KEY(cluster_id) REFERENCES question_clusters(id) ON DELETE CASCADE,
            FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_question_clusters_course ON question_clusters(course_id, approved)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_question_cluster_members_question ON question_cluster_members(question_id)")


def _migration_question_cluster_class_scope_repair(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_question_clusters_course_class ON question_clusters(course_id, class_id, approved)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS question_cluster_repair_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_cluster_id INTEGER NOT NULL,
            new_cluster_id INTEGER NOT NULL,
            course_id INTEGER,
            class_id INTEGER,
            question_ids_json TEXT NOT NULL DEFAULT '[]',
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    clusters = conn.execute(
        """
        SELECT id, course_id, class_id, label, representative_question, approved,
               is_noise, source, metadata_json
        FROM question_clusters
        ORDER BY id ASC
        """
    ).fetchall()
    for cluster in clusters:
        members = conn.execute(
            """
            SELECT m.question_id, m.distance, m.is_representative, m.manual_adjusted,
                   q.course_id, q.class_id, q.question
            FROM question_cluster_members m
            JOIN questions q ON q.id = m.question_id
            WHERE m.cluster_id = ?
            ORDER BY m.is_representative DESC, m.distance ASC, m.question_id ASC
            """,
            (cluster["id"],),
        ).fetchall()
        if not members:
            continue
        grouped: dict[tuple[int | None, int | None], list[sqlite3.Row]] = {}
        for member in members:
            grouped.setdefault((member["course_id"], member["class_id"]), []).append(member)
        current_key = (cluster["course_id"], cluster["class_id"])
        if len(grouped) == 1 and current_key in grouped and cluster["class_id"] is not None:
            continue
        ordered_groups = sorted(grouped.items(), key=lambda item: ((item[0][0] or 0), (item[0][1] or 0)))
        reason = "split_mixed_class_cluster" if len(ordered_groups) > 1 else "repair_cluster_scope"
        for index, ((course_id, class_id), rows) in enumerate(ordered_groups):
            representative = next((row for row in rows if row["is_representative"]), rows[0])
            question_ids = [int(row["question_id"]) for row in rows]
            if index == 0:
                new_cluster_id = int(cluster["id"])
                conn.execute(
                    """
                    UPDATE question_clusters
                    SET course_id = ?, class_id = ?, representative_question = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (course_id, class_id, representative["question"], cluster["id"]),
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO question_clusters (
                        course_id, class_id, label, representative_question,
                        approved, is_noise, source, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        course_id,
                        class_id,
                        cluster["label"],
                        representative["question"],
                        cluster["approved"],
                        cluster["is_noise"],
                        cluster["source"],
                        cluster["metadata_json"],
                    ),
                )
                new_cluster_id = int(cursor.lastrowid)
                placeholders = ",".join("?" for _ in question_ids)
                conn.execute(
                    f"""
                    UPDATE question_cluster_members
                    SET cluster_id = ?
                    WHERE cluster_id = ? AND question_id IN ({placeholders})
                    """,
                    [new_cluster_id, cluster["id"], *question_ids],
                )
            conn.execute(
                """
                INSERT INTO question_cluster_repair_log (
                    original_cluster_id, new_cluster_id, course_id, class_id,
                    question_ids_json, reason
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cluster["id"], new_cluster_id, course_id, class_id, json.dumps(question_ids), reason),
            )


def _migration_resource_query_cache_mapping(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resource_query_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_key TEXT NOT NULL UNIQUE,
            source_name TEXT NOT NULL DEFAULT '',
            query_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resource_query_cache_items (
            cache_id INTEGER NOT NULL,
            resource_id INTEGER NOT NULL,
            position INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(cache_id, resource_id),
            FOREIGN KEY(cache_id) REFERENCES resource_query_cache(id) ON DELETE CASCADE,
            FOREIGN KEY(resource_id) REFERENCES external_resources(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_resource_query_cache_source ON resource_query_cache(source_name, expires_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_resource_query_cache_items_resource ON resource_query_cache_items(resource_id)")


def _migration_external_resource_provenance(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resource_provenance (
            resource_id INTEGER PRIMARY KEY,
            sync_run_id INTEGER NOT NULL,
            connector_name TEXT NOT NULL,
            origin_kind TEXT NOT NULL CHECK(origin_kind IN ('remote', 'local_catalog')),
            content_hash TEXT NOT NULL,
            verified_url TEXT NOT NULL,
            verified_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(resource_id) REFERENCES external_resources(id) ON DELETE CASCADE,
            FOREIGN KEY(sync_run_id) REFERENCES connector_sync_runs(id) ON DELETE RESTRICT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_resource_provenance_sync_run ON resource_provenance(sync_run_id)"
    )


def _migration_persistent_question_cluster_jobs(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS question_cluster_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            class_id INTEGER NOT NULL DEFAULT 0,
            last_successful_question_id INTEGER NOT NULL DEFAULT 0,
            pending_question_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'IDLE'
                CHECK(status IN ('IDLE', 'PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED')),
            error_code TEXT,
            error_message TEXT,
            claimed_by TEXT,
            claimed_at TEXT,
            started_at TEXT,
            finished_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(course_id, class_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_question_cluster_jobs_status
        ON question_cluster_jobs(status, updated_at)
        """
    )
    scopes = conn.execute(
        """
        SELECT DISTINCT course_id, COALESCE(class_id, 0) AS class_id
        FROM questions
        WHERE course_id IS NOT NULL
        UNION
        SELECT DISTINCT course_id, COALESCE(class_id, 0) AS class_id
        FROM question_clusters
        WHERE course_id IS NOT NULL
        """
    ).fetchall()
    for scope in scopes:
        course_id = int(scope["course_id"])
        class_id = int(scope["class_id"] or 0)
        pending_count = _unprocessed_question_count(
            conn,
            course_id=course_id,
            class_id=class_id,
        )
        max_processed_id = _max_processed_question_id(
            conn,
            course_id=course_id,
            class_id=class_id,
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO question_cluster_jobs (
                course_id, class_id, last_successful_question_id,
                pending_question_count, status
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                course_id,
                class_id,
                max_processed_id,
                pending_count,
                "PENDING" if pending_count else "SUCCEEDED",
            ),
        )


def _migration_assignments_progress_stuck(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER,
            module TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            visibility TEXT NOT NULL DEFAULT 'course',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assignment_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER NOT NULL,
            step_index INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            module TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(assignment_id) REFERENCES assignments(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS student_assignment_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            assignment_id INTEGER NOT NULL,
            step_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'not_started',
            progress_percent INTEGER NOT NULL DEFAULT 0,
            note TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, assignment_id),
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(assignment_id) REFERENCES assignments(id) ON DELETE CASCADE,
            FOREIGN KEY(step_id) REFERENCES assignment_steps(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stuck_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            assignment_id INTEGER NOT NULL,
            step_id INTEGER NOT NULL,
            module TEXT NOT NULL DEFAULT '',
            reason TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'active',
            status TEXT NOT NULL DEFAULT 'active',
            evidence_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(assignment_id) REFERENCES assignments(id) ON DELETE CASCADE,
            FOREIGN KEY(step_id) REFERENCES assignment_steps(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_assignment_steps_assignment ON assignment_steps(assignment_id, step_index)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_progress_assignment_step ON student_assignment_progress(assignment_id, step_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stuck_reports_status ON stuck_reports(status, assignment_id, step_id)")


def _migration_course_scoped_documents(conn: sqlite3.Connection) -> None:
    """Scope document hash uniqueness to a course without dropping existing rows."""
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("ALTER TABLE document_chunks RENAME TO document_chunks_v11_old")
    conn.execute("ALTER TABLE index_jobs RENAME TO index_jobs_v11_old")
    conn.execute("ALTER TABLE documents RENAME TO documents_v11_old")
    conn.execute(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            source_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            text_length INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            index_status TEXT NOT NULL DEFAULT 'PENDING',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            course_id INTEGER,
            class_id INTEGER,
            uploaded_by_user_id INTEGER,
            visibility TEXT NOT NULL DEFAULT 'course',
            UNIQUE(course_id, content_hash)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO documents (
            id, filename, source_path, file_type, title, content_hash, byte_size,
            text_length, chunk_count, index_status, metadata_json, created_at, updated_at,
            course_id, class_id, uploaded_by_user_id, visibility
        )
        SELECT id, filename, source_path, file_type, title, content_hash, byte_size,
               text_length, chunk_count, index_status, metadata_json, created_at, updated_at,
               course_id, class_id, uploaded_by_user_id, visibility
        FROM documents_v11_old
        """
    )
    conn.execute(
        """
        CREATE TABLE document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_hash TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            index_status TEXT NOT NULL DEFAULT 'PENDING',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            page INTEGER,
            section TEXT NOT NULL DEFAULT '',
            source_type TEXT NOT NULL DEFAULT '',
            UNIQUE(document_id, chunk_index),
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO document_chunks (
            id, document_id, chunk_index, chunk_text, chunk_hash, token_count,
            index_status, created_at, updated_at, page, section, source_type
        )
        SELECT id, document_id, chunk_index, chunk_text, chunk_hash, token_count,
               index_status, created_at, updated_at, page, section, source_type
        FROM document_chunks_v11_old
        """
    )
    conn.execute(
        """
        CREATE TABLE index_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            filename TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO index_jobs (
            id, document_id, filename, content_hash, status, error_message, created_at, updated_at
        )
        SELECT id, document_id, filename, content_hash, status, error_message, created_at, updated_at
        FROM index_jobs_v11_old
        """
    )
    conn.execute("DROP TABLE document_chunks_v11_old")
    conn.execute("DROP TABLE index_jobs_v11_old")
    conn.execute("DROP TABLE documents_v11_old")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_course_hash ON documents(course_id, content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_course_visibility ON documents(course_id, visibility)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_index_jobs_document_id ON index_jobs(document_id)")
    conn.execute("PRAGMA foreign_keys = ON")


def _migration_user_token_version(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "users", "token_version", "INTEGER NOT NULL DEFAULT 0")


def _empty_activity_summary() -> dict[str, Any]:
    return {
        "unique_active_students": 0,
        "visit_count": 0,
        "active_days": 0,
        "total_active_seconds": 0,
        "recent_active_at": None,
        "page_distribution": {},
        "event_distribution": {},
    }


def _external_resource_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_authors, legacy_authors = _decoded_external_authors(row.pop("authors_json", "[]"))
    row["authors"] = canonical_authors(raw_authors)
    row["tags"] = _decoded_json_list(row.pop("tags_json", "[]"))
    row["raw_metadata"] = _decoded_json_dict(row.pop("raw_metadata_json", "{}"))
    row["indexed"] = bool(row["indexed"])
    if "provenance_verified" in row:
        row["provenance_verified"] = bool(row["provenance_verified"])
    row["authors_hash_covered"] = _authors_hash_covered(row, legacy_authors=legacy_authors)
    return row


def _decoded_external_authors(value: Any) -> tuple[list[Any], bool]:
    try:
        decoded = json.loads(value or "[]") if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return [], True
    if isinstance(decoded, list):
        return decoded, False
    if isinstance(decoded, dict) and isinstance(decoded.get("authors"), list):
        return decoded["authors"], True
    return [], True


def _decoded_json_list(value: Any) -> list[Any]:
    try:
        decoded = json.loads(value or "[]") if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return decoded if isinstance(decoded, list) else []


def _decoded_json_dict(value: Any) -> dict[str, Any]:
    try:
        decoded = json.loads(value or "{}") if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _authors_hash_covered(row: dict[str, Any], *, legacy_authors: bool) -> bool:
    if legacy_authors or not row.get("authors") or not row.get("provenance_verified"):
        return False
    try:
        resource = ExternalResource(
            external_id=str(row.get("external_id") or ""),
            resource_type=str(row.get("resource_type") or ""),
            source_name=str(row.get("source_name") or ""),
            title=str(row.get("title") or ""),
            authors=list(row["authors"]),
            abstract=str(row.get("abstract") or ""),
            url=str(row.get("url") or ""),
            published_at=row.get("published_at"),
            tags=list(row.get("tags") or []),
            module=str(row.get("module") or ""),
            raw_metadata=dict(row.get("raw_metadata") or {}),
            rights_note=str(row.get("rights_note") or ""),
        )
    except Exception:
        return False
    return current_resource_content_hash(resource) == str(row.get("content_hash") or "")


def _innovation_session_row(row: dict[str, Any]) -> dict[str, Any]:
    row["cases"] = json.loads(row.pop("cases_json") or "[]")
    row["limitations"] = json.loads(row.pop("limitations_json") or "[]")
    row["questions"] = json.loads(row.pop("questions_json") or "[]")
    row["final_ideas"] = json.loads(row.pop("final_ideas_json") or "[]")
    return row


def _document_row(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.pop("metadata_json", None)
    try:
        metadata = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    row["metadata"] = metadata
    row.update(publication_projection(row))
    return row


def _publication_metadata_json(
    raw: Any,
    *,
    published: bool,
    actor_id: int | None,
    action: str,
    content_hash: str,
) -> str:
    try:
        metadata = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    timestamp = datetime.now(UTC).isoformat()
    metadata["published_to_students"] = bool(published)
    metadata["published_by"] = int(actor_id) if published and actor_id is not None else None
    metadata["published_at"] = timestamp if published else None
    metadata["publication_last_action"] = {
        "action": action,
        "actor": int(actor_id) if actor_id is not None else None,
        "at": timestamp,
        "content_version": content_hash,
    }
    return _json_dumps(metadata)


def _question_cluster_row(row: dict[str, Any]) -> dict[str, Any]:
    row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
    row["approved"] = bool(row["approved"])
    row["is_noise"] = bool(row["is_noise"])
    return row


def _question_cluster_job_row(row: dict[str, Any]) -> dict[str, Any]:
    row["class_id"] = int(row["class_id"]) if int(row["class_id"] or 0) else None
    row["last_successful_question_id"] = int(row["last_successful_question_id"] or 0)
    row["pending_question_count"] = int(row["pending_question_count"] or 0)
    return row


def _normalize_cluster_class_id(class_id: int | None) -> int:
    return int(class_id) if class_id is not None else 0


def _scope_question_clause(
    *,
    course_id: int,
    class_id: int,
    prefix: str = "q",
) -> tuple[str, list[Any]]:
    if class_id:
        return f"{prefix}.course_id = ? AND {prefix}.class_id = ?", [course_id, class_id]
    return f"{prefix}.course_id = ?", [course_id]


def _unprocessed_question_count(
    conn: sqlite3.Connection,
    *,
    course_id: int,
    class_id: int,
) -> int:
    question_clause, question_params = _scope_question_clause(
        course_id=course_id,
        class_id=class_id,
        prefix="q",
    )
    member_scope = "c.course_id = q.course_id"
    if class_id:
        member_scope += " AND c.class_id = q.class_id"
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM questions q
        WHERE {question_clause}
          AND NOT EXISTS (
              SELECT 1
              FROM question_cluster_members m
              JOIN question_clusters c ON c.id = m.cluster_id
              WHERE m.question_id = q.id
                AND {member_scope}
          )
        """,
        question_params,
    ).fetchone()
    return int(row["count"] or 0)


def _max_processed_question_id(
    conn: sqlite3.Connection,
    *,
    course_id: int,
    class_id: int,
) -> int:
    question_clause, question_params = _scope_question_clause(
        course_id=course_id,
        class_id=class_id,
        prefix="q",
    )
    member_scope = "c.course_id = q.course_id"
    if class_id:
        member_scope += " AND c.class_id = q.class_id"
    row = conn.execute(
        f"""
        SELECT COALESCE(MAX(q.id), 0) AS max_id
        FROM questions q
        WHERE {question_clause}
          AND EXISTS (
              SELECT 1
              FROM question_cluster_members m
              JOIN question_clusters c ON c.id = m.cluster_id
              WHERE m.question_id = q.id
                AND {member_scope}
          )
        """,
        question_params,
    ).fetchone()
    return int(row["max_id"] or 0)


def _max_question_id_for_scope(
    conn: sqlite3.Connection,
    *,
    course_id: int,
    class_id: int,
) -> int:
    question_clause, question_params = _scope_question_clause(
        course_id=course_id,
        class_id=class_id,
        prefix="q",
    )
    row = conn.execute(
        f"SELECT COALESCE(MAX(q.id), 0) AS max_id FROM questions q WHERE {question_clause}",
        question_params,
    ).fetchone()
    return int(row["max_id"] or 0)


def _stuck_report_row(row: dict[str, Any]) -> dict[str, Any]:
    row["evidence"] = json.loads(row.pop("evidence_json") or "{}")
    return row


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
