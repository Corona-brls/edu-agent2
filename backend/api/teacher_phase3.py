"""Phase 3 teacher analytics APIs."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.analytics.activity import ActivityService
from backend.analytics.clustering import QuestionClusteringService
from backend.analytics.stuck_analysis import StuckAnalysisService
from backend.auth.dependencies import get_current_user, require_roles

router = APIRouter(prefix="/api/teacher/phase3", tags=["teacher-phase3"])

STUDENT_EVENT_TYPES = {
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
}


@router.get("/summary")
def teacher_phase3_summary(
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
    course_id: int | None = None,
    class_id: int | None = None,
    start_at: str | None = None,
    end_at: str | None = None,
) -> dict[str, Any]:
    storage = _storage()
    scope = _resolve_scope(storage, current_user, course_id=course_id, class_id=class_id)
    _validate_time_range(start_at, end_at)
    event_summary = _event_summary(
        storage,
        course_id=scope["course_id"],
        class_ids=scope["class_ids"],
        start_at=start_at,
        end_at=end_at,
    )
    question_summary = _question_summary(
        storage,
        course_id=scope["course_id"],
        class_ids=scope["class_ids"],
        start_at=start_at,
        end_at=end_at,
    )
    practice_summary = _event_type_summary(event_summary, "PRACTICE_GUIDE_GENERATED")
    innovation_summary = _innovation_summary(
        storage,
        course_id=scope["course_id"],
        class_ids=scope["class_ids"],
        start_at=start_at,
        end_at=end_at,
    )
    assignment_summary = _assignment_summary(storage, class_ids=scope["class_ids"])
    stuck_summary = _stuck_summary(storage, class_ids=scope["class_ids"], class_id=class_id)
    class_ids = tuple(int(item) for item in scope["class_ids"])
    clusters = QuestionClusteringService(storage).list_clusters(
        course_id=scope["course_id"],
        class_id=class_id,
        class_ids=class_ids,
    )["clusters"]
    faq = QuestionClusteringService(storage).export_faq(
        course_id=scope["course_id"],
        class_id=class_id,
        class_ids=class_ids,
    )["items"]

    return {
        "filters": {
            "course_id": scope["course_id"],
            "class_id": class_id,
            "class_ids": scope["class_ids"],
            "start_at": start_at,
            "end_at": end_at,
            "scope_note": "visit_count counts PAGE_VIEW learning events only; Agent internal steps are excluded.",
        },
        "courses": scope["courses"],
        "classes": scope["classes"],
        "access": {
            "unique_active_students": event_summary["unique_active_students"],
            "visit_count": event_summary["visit_count"],
            "active_days": event_summary["active_days"],
            "recent_active_at": event_summary["recent_active_at"],
            "total_active_seconds": event_summary["total_active_seconds"],
            "page_distribution": event_summary["page_distribution"],
        },
        "resources": {
            "search_count": _event_type_summary(event_summary, "RESOURCE_SEARCH"),
            "open_count": _event_type_summary(event_summary, "RESOURCE_OPEN"),
            "top_resources": event_summary["top_resources"],
        },
        "qa": question_summary,
        "practice": {"guide_count": practice_summary},
        "innovation": innovation_summary,
        "assignments": assignment_summary,
        "stuck": stuck_summary,
        "clusters": {
            "total": len(clusters),
            "approved": sum(1 for item in clusters if item.get("approved")),
            "noise": sum(1 for item in clusters if item.get("is_noise")),
            "items": clusters,
        },
        "faq": {
            "items": faq,
            "ready_count": sum(1 for item in faq if item.get("citations")),
            "needs_review_count": sum(1 for item in faq if not item.get("citations")),
        },
        "event_distribution": event_summary["event_distribution"],
    }


@router.get("/students/{student_id}/timeline")
def teacher_phase3_student_timeline(
    student_id: int,
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
    limit: int = 100,
) -> dict[str, Any]:
    storage = _storage()
    try:
        events = ActivityService(storage).student_timeline(
            requester=current_user,
            student_id=student_id,
            limit=limit,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail={"error_code": "FORBIDDEN", "message": str(exc)}) from exc
    return {
        "student_id": student_id,
        "events": events,
        "summary_only_by_default": True,
    }


@router.get("/faq-export.csv")
def teacher_phase3_faq_export_csv(
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
    course_id: int | None = None,
    class_id: int | None = None,
) -> StreamingResponse:
    storage = _storage()
    scope = _resolve_scope(storage, current_user, course_id=course_id, class_id=class_id)
    items = QuestionClusteringService(storage).export_faq(
        course_id=scope["course_id"],
        class_id=class_id,
        class_ids=tuple(int(item) for item in scope["class_ids"]),
    )["items"]
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["cluster_id", "label", "representative_question", "count", "approved", "answer", "citations"],
    )
    writer.writeheader()
    for item in items:
        writer.writerow(
            {
                "cluster_id": item.get("cluster_id"),
                "label": _escape_csv_formula(item.get("label", "")),
                "representative_question": _escape_csv_formula(item.get("representative_question", "")),
                "count": item.get("count", 0),
                "approved": item.get("approved", False),
                "answer": _escape_csv_formula(item.get("answer", "")),
                "citations": _escape_csv_formula(json.dumps(item.get("citations", []), ensure_ascii=False)),
            }
        )
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue().encode("utf-8-sig")]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="phase3_faq.csv"'},
    )


def _storage():
    from backend.main import learning_logger

    return learning_logger


def _resolve_scope(
    storage,
    user: dict[str, Any],
    *,
    course_id: int | None,
    class_id: int | None,
) -> dict[str, Any]:
    role = str(user["role"]).lower()
    if role == "teacher":
        classes = storage.classes_for_teacher(int(user["id"]), course_id=course_id)
        if not classes:
            raise HTTPException(
                status_code=403,
                detail={"error_code": "COURSE_FORBIDDEN", "message": "教师没有可访问的任课课程或班级。"},
            )
        if class_id is not None:
            classes = [item for item in classes if int(item["id"]) == int(class_id)]
            if not classes:
                raise HTTPException(status_code=403, detail={"error_code": "CLASS_FORBIDDEN", "message": "教师只能查看自己班级。"})
        visible_courses = storage.courses_for_teacher(int(user["id"]))
    else:
        classes = storage.classes(course_id=course_id)
        if class_id is not None:
            classes = [item for item in classes if int(item["id"]) == int(class_id)]
        visible_courses = storage.courses()

    selected_course_id = course_id
    if selected_course_id is None and classes:
        selected_course_id = int(classes[0]["course_id"])
    class_ids = [int(item["id"]) for item in classes]
    return {
        "course_id": selected_course_id,
        "class_ids": class_ids,
        "classes": classes,
        "courses": visible_courses,
    }


def _validate_time_range(start_at: str | None, end_at: str | None) -> None:
    for label, value in {"start_at": start_at, "end_at": end_at}.items():
        if not value:
            continue
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error_code": "INVALID_TIME_RANGE", "message": f"{label} 不是合法时间。"}) from exc
    if start_at and end_at and start_at > end_at:
        raise HTTPException(status_code=400, detail={"error_code": "INVALID_TIME_RANGE", "message": "start_at 不能晚于 end_at。"})


def _event_summary(
    storage,
    *,
    course_id: int | None,
    class_ids: list[int],
    start_at: str | None,
    end_at: str | None,
) -> dict[str, Any]:
    clauses, params = _scope_clauses("learning_events", course_id=course_id, class_ids=class_ids, start_at=start_at, end_at=end_at)
    clauses.append(f"learning_events.event_type IN ({','.join('?' for _ in STUDENT_EVENT_TYPES)})")
    params.extend(sorted(STUDENT_EVENT_TYPES))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with storage._connect() as conn:
        row = conn.execute(
            f"""
            SELECT
                COUNT(DISTINCT user_id) AS unique_active_students,
                SUM(CASE WHEN event_type = 'PAGE_VIEW' THEN 1 ELSE 0 END) AS visit_count,
                COUNT(DISTINCT date(created_at)) AS active_days,
                MAX(created_at) AS recent_active_at
            FROM learning_events
            {where}
            """,
            params,
        ).fetchone()
        page_rows = conn.execute(
            f"""
            SELECT COALESCE(NULLIF(page_name, ''), '(unknown)') AS page_name, COUNT(*) AS count
            FROM learning_events
            {where} AND event_type = 'PAGE_VIEW'
            GROUP BY COALESCE(NULLIF(page_name, ''), '(unknown)')
            ORDER BY count DESC, page_name ASC
            """,
            params,
        ).fetchall()
        event_rows = conn.execute(
            f"""
            SELECT event_type, COUNT(*) AS count
            FROM learning_events
            {where}
            GROUP BY event_type
            ORDER BY count DESC, event_type ASC
            """,
            params,
        ).fetchall()
        resource_rows = conn.execute(
            f"""
            SELECT resource_id, COUNT(*) AS count
            FROM learning_events
            {where} AND event_type = 'RESOURCE_OPEN' AND resource_id IS NOT NULL
            GROUP BY resource_id
            ORDER BY count DESC, resource_id ASC
            LIMIT 10
            """,
            params,
        ).fetchall()
        session_clauses, session_params = _scope_clauses(
            "learning_sessions",
            course_id=course_id,
            class_ids=class_ids,
            start_at=start_at,
            end_at=end_at,
            time_column="started_at",
        )
        session_where = f"WHERE {' AND '.join(session_clauses)}" if session_clauses else ""
        session_row = conn.execute(
            f"SELECT SUM(active_seconds) AS active_seconds FROM learning_sessions {session_where}",
            session_params,
        ).fetchone()
    return {
        "unique_active_students": int(row["unique_active_students"] or 0),
        "visit_count": int(row["visit_count"] or 0),
        "active_days": int(row["active_days"] or 0),
        "recent_active_at": row["recent_active_at"],
        "total_active_seconds": int(session_row["active_seconds"] or 0),
        "page_distribution": {item["page_name"]: int(item["count"]) for item in page_rows},
        "event_distribution": {item["event_type"]: int(item["count"]) for item in event_rows},
        "top_resources": [{"resource_id": item["resource_id"], "count": int(item["count"])} for item in resource_rows],
    }


def _question_summary(
    storage,
    *,
    course_id: int | None,
    class_ids: list[int],
    start_at: str | None,
    end_at: str | None,
) -> dict[str, Any]:
    clauses, params = _scope_clauses("questions", course_id=course_id, class_ids=class_ids, start_at=start_at, end_at=end_at)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with storage._connect() as conn:
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS question_count,
                   SUM(CASE WHEN abstained = 1 THEN 1 ELSE 0 END) AS abstained_count,
                   AVG(confidence) AS average_confidence
            FROM questions
            {where}
            """,
            params,
        ).fetchone()
        module_rows = conn.execute(
            f"""
            SELECT module, COUNT(*) AS count
            FROM questions
            {where}
            GROUP BY module
            ORDER BY count DESC, module ASC
            """,
            params,
        ).fetchall()
    total = int(row["question_count"] or 0)
    abstained = int(row["abstained_count"] or 0)
    return {
        "question_count": total,
        "abstained_count": abstained,
        "abstain_rate": round(abstained / total, 4) if total else 0.0,
        "average_confidence": round(float(row["average_confidence"] or 0.0), 4),
        "module_distribution": {item["module"]: int(item["count"]) for item in module_rows},
    }


def _innovation_summary(storage, *, course_id: int | None, class_ids: list[int], start_at: str | None, end_at: str | None) -> dict[str, Any]:
    clauses, params = _scope_clauses("innovation_sessions", course_id=course_id, class_ids=class_ids, start_at=start_at, end_at=end_at)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with storage._connect() as conn:
        row = conn.execute(f"SELECT COUNT(*) AS count FROM innovation_sessions {where}", params).fetchone()
    return {"session_count": int(row["count"] or 0)}


def _assignment_summary(storage, *, class_ids: list[int]) -> dict[str, Any]:
    if not class_ids:
        return {"progress_count": 0, "by_status": {}, "recent": []}
    placeholders = ",".join("?" for _ in class_ids)
    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT p.status, COUNT(*) AS count
            FROM student_assignment_progress p
            JOIN class_members cm ON cm.user_id = p.user_id
            WHERE cm.class_id IN ({placeholders})
            GROUP BY p.status
            ORDER BY count DESC, p.status ASC
            """,
            class_ids,
        ).fetchall()
        recent = conn.execute(
            f"""
            SELECT p.*, u.username, a.title AS assignment_title, s.title AS step_title
            FROM student_assignment_progress p
            JOIN users u ON u.id = p.user_id
            JOIN assignments a ON a.id = p.assignment_id
            JOIN assignment_steps s ON s.id = p.step_id
            JOIN class_members cm ON cm.user_id = p.user_id
            WHERE cm.class_id IN ({placeholders})
            ORDER BY datetime(p.updated_at) DESC
            LIMIT 20
            """,
            class_ids,
        ).fetchall()
    by_status = {item["status"]: int(item["count"]) for item in rows}
    return {
        "progress_count": sum(by_status.values()),
        "by_status": by_status,
        "recent": [dict(item) for item in recent],
    }


def _stuck_summary(storage, *, class_ids: list[int], class_id: int | None) -> dict[str, Any]:
    service = StuckAnalysisService(storage)
    if class_id is not None:
        return service.analyze(class_id=class_id)
    items: list[dict[str, Any]] = []
    seen = set()
    for visible_class_id in class_ids:
        for item in service.analyze(class_id=visible_class_id)["items"]:
            key = (item.get("source"), item.get("id"), item.get("assignment_id"), item.get("step_id"), item.get("reason"))
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
    return {"items": items, "total": len(items)}


def _event_type_summary(event_summary: dict[str, Any], event_type: str) -> int:
    return int(event_summary.get("event_distribution", {}).get(event_type, 0))


def _scope_clauses(
    table: str,
    *,
    course_id: int | None,
    class_ids: list[int],
    start_at: str | None,
    end_at: str | None,
    time_column: str = "created_at",
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if course_id is not None:
        clauses.append(f"{table}.course_id = ?")
        params.append(course_id)
    if class_ids:
        placeholders = ",".join("?" for _ in class_ids)
        clauses.append(f"{table}.class_id IN ({placeholders})")
        params.extend(class_ids)
    else:
        clauses.append("1 = 0")
    if start_at:
        clauses.append(f"datetime({table}.{time_column}) >= datetime(?)")
        params.append(start_at)
    if end_at:
        clauses.append(f"datetime({table}.{time_column}) <= datetime(?)")
        params.append(end_at)
    return clauses, params


def _escape_csv_formula(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value
