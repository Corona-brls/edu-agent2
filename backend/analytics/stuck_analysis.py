"""Assignment stuck-point analysis."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from backend.storage.logger import LearningLogger


class StuckAnalysisService:
    def __init__(self, storage: LearningLogger) -> None:
        self.storage = storage

    def analyze(
        self,
        *,
        assignment_id: int | None = None,
        step_id: int | None = None,
        module: str = "",
        class_id: int | None = None,
        class_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        reports = [
            item
            for item in self.storage.stuck_reports()
            if item["status"] in {"active", "confirmed"}
            and (assignment_id is None or int(item["assignment_id"]) == int(assignment_id))
            and (step_id is None or int(item["step_id"]) == int(step_id))
            and (not module or item.get("module") == module)
            and self._matches_class(int(item["user_id"]), class_id, class_ids)
        ]
        items = [_from_report(item) for item in reports]
        items.extend(
            self._repeated_question_items(
                assignment_id=assignment_id,
                step_id=step_id,
                module=module,
                class_id=class_id,
                class_ids=class_ids,
            )
        )
        items.extend(self._high_abstain_items(module=module, class_id=class_id, class_ids=class_ids))
        items.extend(
            self._stale_progress_items(
                assignment_id=assignment_id,
                step_id=step_id,
                class_id=class_id,
                class_ids=class_ids,
            )
        )
        deduped = _dedupe(items)
        return {"items": deduped, "total": len(deduped)}

    def update_status(self, report_id: int, action: str) -> dict[str, Any]:
        mapping = {"confirm": "confirmed", "ignore": "ignored", "resolve": "resolved"}
        if action not in mapping:
            raise ValueError("unsupported stuck action")
        return self.storage.update_stuck_report_status(report_id, mapping[action])

    def _repeated_question_items(
        self,
        *,
        assignment_id: int | None,
        step_id: int | None,
        module: str,
        class_id: int | None,
        class_ids: list[int] | None,
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        with self.storage._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_id, assignment_id, class_id, module, metadata_json, created_at
                FROM learning_events
                WHERE event_type = 'QUESTION_ASKED' AND assignment_id IS NOT NULL
                """
            ).fetchall()
        for row in rows:
            metadata = _json(row["metadata_json"])
            row_step_id = metadata.get("step_id")
            if not self._row_matches_class(row["class_id"], class_id, class_ids):
                continue
            if assignment_id is not None and str(row["assignment_id"]) != str(assignment_id):
                continue
            if step_id is not None and str(row_step_id) != str(step_id):
                continue
            if module and row["module"] != module:
                continue
            grouped[(str(row["assignment_id"]), str(row_step_id or ""))].append(dict(row))
        items = []
        for (assignment, step), rows in grouped.items():
            if len(rows) < 2:
                continue
            items.append(
                {
                    "source": "suspected",
                    "status": "active",
                    "assignment_id": int(assignment),
                    "step_id": int(step) if step.isdigit() else None,
                    "module": rows[0].get("module", ""),
                    "reason": "同一步骤出现重复提问",
                    "evidence": {"question_event_count": len(rows)},
                }
            )
        return items

    def _high_abstain_items(self, *, module: str, class_id: int | None, class_ids: list[int] | None) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if class_id is not None:
            clauses.append("class_id = ?")
            params.append(class_id)
        elif class_ids is not None:
            if not class_ids:
                return []
            placeholders = ",".join("?" for _ in class_ids)
            clauses.append(f"class_id IN ({placeholders})")
            params.extend(class_ids)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.storage._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT module, COUNT(*) AS total,
                       SUM(CASE WHEN abstained = 1 THEN 1 ELSE 0 END) AS abstained_count
                FROM questions
                {where}
                GROUP BY module
                HAVING total >= 2
                """,
                params,
            ).fetchall()
        items = []
        for row in rows:
            if module and row["module"] != module:
                continue
            total = int(row["total"] or 0)
            abstained = int(row["abstained_count"] or 0)
            if total and abstained / total >= 0.5:
                items.append(
                    {
                        "source": "suspected",
                        "status": "active",
                        "assignment_id": None,
                        "step_id": None,
                        "module": row["module"],
                        "reason": "该模块拒答率偏高",
                        "evidence": {"total_questions": total, "abstained_count": abstained},
                    }
                )
        return items

    def _stale_progress_items(
        self,
        *,
        assignment_id: int | None,
        step_id: int | None,
        class_id: int | None,
        class_ids: list[int] | None,
    ) -> list[dict[str, Any]]:
        threshold = datetime.now(UTC) - timedelta(days=7)
        with self.storage._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*, s.module
                FROM student_assignment_progress p
                JOIN assignment_steps s ON s.id = p.step_id
                WHERE p.status != 'completed'
                """
            ).fetchall()
        items = []
        for row in rows:
            if not self._matches_class(int(row["user_id"]), class_id, class_ids):
                continue
            if assignment_id is not None and int(row["assignment_id"]) != int(assignment_id):
                continue
            if step_id is not None and int(row["step_id"]) != int(step_id):
                continue
            updated = _parse_time(row["updated_at"])
            if updated and updated < threshold:
                items.append(
                    {
                        "source": "suspected",
                        "status": "active",
                        "assignment_id": int(row["assignment_id"]),
                        "step_id": int(row["step_id"]),
                        "module": row["module"],
                        "reason": "学生长期未推进该步骤",
                        "evidence": {"updated_at": row["updated_at"], "progress_percent": row["progress_percent"]},
                    }
                )
        return items

    def _matches_class(self, user_id: int, class_id: int | None, class_ids: list[int] | None = None) -> bool:
        if class_id is None and class_ids is None:
            return True
        allowed = {int(class_id)} if class_id is not None else {int(item) for item in (class_ids or [])}
        return any(int(item["id"]) in allowed for item in self.storage.classes_for_user(user_id))

    def _row_matches_class(self, row_class_id: Any, class_id: int | None, class_ids: list[int] | None = None) -> bool:
        if class_id is None and class_ids is None:
            return True
        if row_class_id is None:
            return False
        allowed = {int(class_id)} if class_id is not None else {int(item) for item in (class_ids or [])}
        return int(row_class_id) in allowed


def _from_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": report["id"],
        "source": report["source"],
        "status": report["status"],
        "assignment_id": report["assignment_id"],
        "assignment_title": report.get("assignment_title"),
        "step_id": report["step_id"],
        "step_title": report.get("step_title"),
        "module": report.get("module", ""),
        "reason": report["reason"],
        "evidence": report.get("evidence", {}),
        "created_at": report.get("created_at"),
    }


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = (item.get("source"), item.get("id"), item.get("assignment_id"), item.get("step_id"), item.get("reason"))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _json(text: str) -> dict[str, Any]:
    import json

    try:
        return json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}


def _parse_time(text: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
