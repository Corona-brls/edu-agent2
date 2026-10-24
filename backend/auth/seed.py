"""Idempotent local demo seed command."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
import tempfile
from typing import Any

from backend.connectors.schemas import ExternalResource
from backend.core.config import get_settings
from backend.learning.source_catalog import load_source_catalog
from backend.rag.document_parser import Document, Page, parse_document
from backend.rag.vector_store import ChromaVectorStore
from backend.resources.normalizer import content_hash
from backend.storage.logger import LearningLogger, sha256_bytes


def seed_demo_data(storage: LearningLogger | None = None, *, include_materials: bool = True) -> dict[str, Any]:
    logger = storage or LearningLogger()
    backup_path = _backup_database(logger) if include_materials else None
    result = logger.seed_demo_identity()
    if include_materials:
        logger.ensure_demo_assignments()
        materials = _seed_demo_materials(logger)
        result = {**result, "backup_path": str(backup_path) if backup_path else None, "materials": materials}
    return result


def seed_shared_textbooks(
    storage: LearningLogger | None = None,
    *,
    vector_store: Any | None = None,
    catalog_path: str | Path | None = None,
    knowledge_base_dir: str | Path | None = None,
    target_slug: str = "demo-course",
) -> dict[str, Any]:
    """Install the approved shared textbooks into one explicit course."""

    if target_slug != "demo-course":
        raise RuntimeError("SHARED_MATERIAL_TARGET_MUST_BE_DEMO_COURSE")
    logger = storage or LearningLogger()
    repository_root = Path(__file__).resolve().parents[2]
    resolved_catalog = Path(catalog_path or repository_root / "knowledge_sources" / "shared" / "catalog.json").resolve()
    catalog = load_source_catalog(resolved_catalog)
    if len(catalog.entries) != 9 or any(entry.review_state != "approved" for entry in catalog.entries):
        raise RuntimeError("SHARED_MATERIALS_REQUIRE_NINE_APPROVED_ENTRIES")

    source_root = resolved_catalog.parent.resolve()
    settings = get_settings()
    knowledge_root = Path(knowledge_base_dir or settings.knowledge_base_dir).resolve()
    course = _unique_course_by_slug(logger, target_slug)
    course_id = int(course["id"])
    course_dir = (knowledge_root / f"course-{course_id}").resolve()
    if source_root == course_dir or source_root in course_dir.parents or course_dir in source_root.parents:
        raise RuntimeError("SHARED_MATERIAL_SOURCE_TARGET_OVERLAP")

    all_documents = logger.documents()
    plans: list[dict[str, Any]] = []
    digests: set[str] = set()
    for entry in catalog.entries:
        source = (source_root / entry.path).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as exc:
            raise RuntimeError("SHARED_MATERIAL_SOURCE_OUTSIDE_CATALOG") from exc
        payload = source.read_bytes()
        digest = sha256_bytes(payload)
        if digest in digests:
            raise RuntimeError("SHARED_MATERIAL_CONTENT_HASH_COLLISION")
        digests.add(digest)
        target = (course_dir / entry.path).resolve()
        if target.parent != course_dir:
            raise RuntimeError("SHARED_MATERIAL_TARGET_PATH_INVALID")
        same_source_id = [
            row for row in all_documents if (row.get("metadata") or {}).get("source_id") == entry.source_id
        ]
        cross_course_hash = [
            row
            for row in all_documents
            if row.get("content_hash") == digest and int(row.get("course_id") or 0) != course_id
        ]
        filename_conflicts = [
            row
            for row in all_documents
            if int(row.get("course_id") or 0) == course_id
            and row.get("filename") == entry.path
            and row.get("content_hash") != digest
        ]
        if cross_course_hash or filename_conflicts:
            raise RuntimeError(f"SHARED_MATERIAL_CONFLICT:{entry.source_id}")
        exact = [
            row
            for row in all_documents
            if int(row.get("course_id") or 0) == course_id and row.get("content_hash") == digest
        ]
        if len(exact) > 1 or len(same_source_id) > 1:
            raise RuntimeError(f"SHARED_MATERIAL_IDENTITY_CONFLICT:{entry.source_id}")
        existing = exact[0] if exact else None
        if same_source_id and (existing is None or int(same_source_id[0]["id"]) != int(existing["id"])):
            raise RuntimeError(f"SHARED_MATERIAL_SOURCE_ID_CONFLICT:{entry.source_id}")
        if target.exists() and sha256_bytes(target.read_bytes()) != digest:
            raise RuntimeError(f"SHARED_MATERIAL_FILE_CONFLICT:{entry.source_id}")
        plans.append(
            {
                "entry": entry,
                "target": target,
                "payload": payload,
                "content_hash": digest,
                "existing": existing,
            }
        )

    store = vector_store or ChromaVectorStore()
    installed: list[dict[str, Any]] = []
    for plan in plans:
        entry = plan["entry"]
        target = plan["target"]
        digest = plan["content_hash"]
        existing = plan["existing"]
        if existing is not None:
            installed.append(
                _validate_installed_textbook(
                    logger,
                    store,
                    existing,
                    target=target,
                    entry=entry,
                    content_hash=digest,
                )
            )
            continue

        _write_course_copy_atomic(target, plan["payload"], digest)
        job_id = logger.create_index_job(filename=entry.path, content_hash=digest, status="PENDING")
        document_id: int | None = None
        try:
            parsed = parse_document(target)
            document = replace(
                parsed,
                metadata={
                    **parsed.metadata,
                    "source_id": entry.source_id,
                    "revision": entry.revision,
                    "review_state": entry.review_state,
                    "reviewer": entry.reviewer,
                    "reviewed_at": entry.reviewed_at,
                },
            )
            record = logger.save_document(
                document=document,
                content_hash=digest,
                chunks=None,
                byte_size=len(plan["payload"]),
                index_status="PENDING",
                job_id=job_id,
                course_id=course_id,
                class_id=None,
                uploaded_by_user_id=None,
                visibility="course",
            )
            if record.get("duplicate"):
                raise RuntimeError(f"SHARED_MATERIAL_UNEXPECTED_DUPLICATE:{entry.source_id}")
            document_id = int(record["id"])
            logger.update_document_index_status(document_id, "EMBEDDING")
            logger.update_index_job(job_id, status="EMBEDDING", document_id=document_id)
            chunks = logger.chunks_for_document(document_id)
            if not chunks:
                raise RuntimeError(f"SHARED_MATERIAL_HAS_NO_CHUNKS:{entry.source_id}")
            inserted = int(store.add_chunks(chunks))
            if inserted != len(chunks) or not store.contains_chunks(chunks):
                raise RuntimeError(f"SHARED_MATERIAL_VECTOR_INCOMPLETE:{entry.source_id}")
            logger.mark_document_chunks_indexed([int(chunk["id"]) for chunk in chunks])
            logger.update_index_job(job_id, status="INDEXED", document_id=document_id)
            published = logger.set_bootstrap_document_publication(
                document_id,
                source_id=entry.source_id,
                revision=entry.revision,
            )
            installed.append(
                _validate_installed_textbook(
                    logger,
                    store,
                    published,
                    target=target,
                    entry=entry,
                    content_hash=digest,
                )
            )
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            if document_id is not None:
                logger.update_document_index_status(document_id, "FAILED")
            logger.fail_index_job(job_id, message, document_id=document_id)
            raise

    return {"course": course, "materials": installed}


def _unique_course_by_slug(logger: LearningLogger, slug: str) -> dict[str, Any]:
    with logger._connect() as conn:
        rows = conn.execute(
            "SELECT id, slug, name FROM courses WHERE slug = ? ORDER BY id",
            (slug,),
        ).fetchall()
    if len(rows) != 1 or int(rows[0]["id"] or 0) <= 0:
        raise RuntimeError("SHARED_MATERIAL_TARGET_COURSE_NOT_UNIQUE")
    return dict(rows[0])


def _write_course_copy_atomic(target: Path, payload: bytes, digest: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_bytes(target.read_bytes()) != digest:
            raise RuntimeError("SHARED_MATERIAL_FILE_CONFLICT")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        if sha256_bytes(temporary.read_bytes()) != digest:
            raise RuntimeError("SHARED_MATERIAL_TEMP_HASH_MISMATCH")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_installed_textbook(
    logger: LearningLogger,
    store: Any,
    document: dict[str, Any],
    *,
    target: Path,
    entry: Any,
    content_hash: str,
) -> dict[str, Any]:
    current = logger.document(int(document["id"]))
    if current is None:
        raise RuntimeError(f"SHARED_MATERIAL_DOCUMENT_MISSING:{entry.source_id}")
    metadata = current.get("metadata") or {}
    action = current.get("publication_last_action") or {}
    checks = (
        current.get("filename") == entry.path,
        str(Path(current.get("source_path") or "").resolve()) == str(target.resolve()),
        current.get("content_hash") == content_hash,
        str(current.get("index_status") or "").upper() == "INDEXED",
        current.get("class_id") is None,
        current.get("uploaded_by_user_id") is None,
        current.get("visibility") == "course",
        current.get("published_to_students") is True,
        current.get("published_by") is None,
        metadata.get("source_id") == entry.source_id,
        metadata.get("revision") == entry.revision,
        action.get("action") == "bootstrap_publish",
        action.get("actor") is None,
        action.get("content_version") == content_hash,
        target.is_file() and sha256_bytes(target.read_bytes()) == content_hash,
    )
    chunks = logger.chunks_for_document(int(current["id"]))
    jobs = logger.index_jobs(int(current["id"]))
    if (
        not all(checks)
        or not chunks
        or len(chunks) != int(current.get("chunk_count") or 0)
        or any(str(chunk.get("index_status") or "").upper() != "INDEXED" for chunk in chunks)
        or len(jobs) != 1
        or str(jobs[0].get("status") or "").upper() != "INDEXED"
        or jobs[0].get("content_hash") != content_hash
        or not store.contains_chunks(chunks)
    ):
        raise RuntimeError(f"SHARED_MATERIAL_PARTIAL_STATE:{entry.source_id}")
    return {
        "source_id": entry.source_id,
        "revision": entry.revision,
        "relative_path": target.name,
        "document_id": int(current["id"]),
        "content_hash": content_hash,
        "chunk_count": len(chunks),
        "index_status": "INDEXED",
        "published_to_students": True,
        "publication_action": "bootstrap_publish",
    }


def _backup_database(logger: LearningLogger) -> Path | None:
    db_path = logger.db_path
    if not db_path.exists():
        return None
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    backup_path = backup_dir / f"{db_path.stem}.demo-seed.{stamp}{db_path.suffix}"
    shutil.copy2(db_path, backup_path)
    return backup_path


def _seed_demo_materials(logger: LearningLogger) -> dict[str, Any]:
    course_id = logger.default_course_id()
    class_id = logger.default_class_id()
    student = logger.user_by_username("student_demo")
    if student is None:
        raise RuntimeError("student_demo was not created")
    student_id = int(student["id"])

    teacher = logger.user_by_username("teacher_demo")
    if teacher is None:
        raise RuntimeError("teacher_demo was not created")
    document_ids = _seed_documents(logger, course_id=course_id, actor_id=int(teacher["id"]))
    resource_ids = _seed_external_resources(logger)
    assignment = logger.assignments()[0]
    steps = assignment["steps"]
    progress = logger.upsert_assignment_progress(
        user_id=student_id,
        assignment_id=int(assignment["id"]),
        step_id=int(steps[-1]["id"]),
        status="IN_PROGRESS",
        progress_percent=75,
        note="Demo: result analysis needs guidance.",
    )
    stuck = _seed_stuck_report(logger, student_id, int(assignment["id"]), int(steps[-1]["id"]))
    session = logger.create_learning_session(
        user_id=student_id,
        course_id=course_id,
        class_id=class_id,
        client_session_id="demo-phase3-session",
    )
    _seed_learning_events(logger, int(session["id"]), student_id, course_id, class_id, int(assignment["id"]))
    question_ids = _seed_questions_and_faq(logger, student_id, course_id, class_id)
    return {
        "documents": document_ids,
        "external_resources": resource_ids,
        "assignment_id": int(assignment["id"]),
        "progress_id": int(progress["id"]),
        "stuck_report_id": int(stuck["id"]),
        "session_id": int(session["id"]),
        "question_ids": question_ids,
    }


def _seed_documents(logger: LearningLogger, *, course_id: int, actor_id: int) -> list[int]:
    knowledge_dir = get_settings().knowledge_base_dir
    documents = [
        Document(
            source_path=knowledge_dir / "5g_nas_demo_notes.md",
            filename="5g_nas_demo_notes.md",
            file_type=".md",
            title="5G NAS Demo Notes",
            text=(
                "5G NAS authentication risk analysis should cite replay protection, identity exposure, "
                "security context handling, and local trace evidence. Safe labs use local captures only."
            ),
            pages=[
                Page(
                    text=(
                        "5G NAS authentication risk analysis should cite replay protection, identity exposure, "
                        "security context handling, and local trace evidence. Safe labs use local captures only."
                    ),
                    page=1,
                    section="5G NAS risk analysis",
                    source_type="markdown",
                )
            ],
        ),
        Document(
            source_path=knowledge_dir / "3GPP_TS_24.501_teaching_notes.md",
            filename="3GPP_TS_24.501_teaching_notes.md",
            file_type=".md",
            title="3GPP TS 24.501 Teaching Notes",
            text=(
                "This local teaching note summarizes where 3GPP TS 24.501 metadata is used in the course. "
                "It is not the copyrighted standard full text; students should use the official 3GPP link for metadata."
            ),
            pages=[
                Page(
                    text=(
                        "This local teaching note summarizes where 3GPP TS 24.501 metadata is used in the course. "
                        "It is not the copyrighted standard full text; students should use the official 3GPP link for metadata."
                    ),
                    page=1,
                    section="standard metadata",
                    source_type="markdown",
                )
            ],
        ),
    ]
    ids: list[int] = []
    for document in documents:
        document.source_path.parent.mkdir(parents=True, exist_ok=True)
        if not document.source_path.exists():
            document.source_path.write_text(document.text, encoding="utf-8")
        record = logger.save_document(
            document=document,
            content_hash=sha256_bytes(document.text.encode("utf-8")),
            chunks=[page.text for page in document.pages],
            byte_size=len(document.text.encode("utf-8")),
            index_status="INDEXED",
            course_id=course_id,
        )
        logger.set_document_publication(int(record["id"]), published=True, actor_id=actor_id)
        ids.append(int(record["id"]))
    return ids


def _seed_external_resources(logger: LearningLogger) -> list[int]:
    resources = [
        ExternalResource(
            external_id="demo-paper-5g-nas",
            resource_type="paper",
            source_name="demo_cache",
            title="Demo Paper: 5G NAS Security Analysis",
            authors=["Course Demo"],
            abstract="Cached paper metadata for 5G NAS authentication and replay-risk teaching discussion.",
            url="https://example.edu/demo/5g-nas-paper",
            published_at="2026",
            tags=["5G NAS", "authentication", "paper"],
            module="5G NAS",
            rights_note="Demo metadata only.",
        ),
        ExternalResource(
            external_id="CVE-2026-DEMO",
            resource_type="vulnerability",
            source_name="demo_cache",
            title="CVE-2026-DEMO teaching sample",
            abstract="Cached vulnerability metadata for local defensive analysis. CVSS 7.5 HIGH; CWE-79.",
            url="https://nvd.nist.gov/vuln/detail/CVE-2026-DEMO",
            published_at="2026",
            tags=["CVE", "CVSS 7.5 HIGH", "CWE-79"],
            module="vulnerability analysis",
            rights_note="Demo cache for course teaching.",
        ),
        ExternalResource(
            external_id="3GPP-TS-24.501",
            resource_type="standard",
            source_name="standard_source",
            title="3GPP TS 24.501",
            abstract="Standard metadata only. Official 3GPP link plus local teaching-note upload status.",
            url="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3370",
            tags=["3GPP", "TS 24.501", "NAS", "standard"],
            module="5G NAS",
            raw_metadata={"official_metadata_only": True, "local_uploaded": True},
            rights_note="Metadata and official link only; copyrighted full text is not fetched.",
        ),
        ExternalResource(
            external_id="demo-project-nas-lab",
            resource_type="project",
            source_name="demo_cache",
            title="Demo Project: Local NAS Trace Lab",
            abstract="Open teaching project idea for local trace parsing, replay-risk metrics, and defensive logging.",
            url="https://example.edu/demo/nas-lab-project",
            published_at="2026",
            tags=["project", "local lab", "5G NAS"],
            module="5G NAS",
            rights_note="Demo project metadata.",
        ),
    ]
    ids: list[int] = []
    for resource in resources:
        record = logger.upsert_external_resource(
            resource=resource.model_dump(mode="json"),
            ttl_hours=24 * 30,
            content_hash=content_hash(resource),
        )
        ids.append(int(record["id"]))
    return ids


def _seed_stuck_report(logger: LearningLogger, user_id: int, assignment_id: int, step_id: int) -> dict[str, Any]:
    with logger._connect() as conn:
        existing = conn.execute(
            """
            SELECT * FROM stuck_reports
            WHERE user_id = ? AND assignment_id = ? AND step_id = ? AND reason = ?
            LIMIT 1
            """,
            (user_id, assignment_id, step_id, "Demo: result analysis step is unclear"),
        ).fetchone()
    if existing:
        return dict(existing)
    return logger.create_stuck_report(
        user_id=user_id,
        assignment_id=assignment_id,
        step_id=step_id,
        module="5G NAS",
        reason="Demo: result analysis step is unclear",
        source="active",
        evidence={"demo_seed": True, "reason": "student reported difficulty interpreting replay-risk metrics"},
    )


def _seed_learning_events(
    logger: LearningLogger,
    session_id: int,
    user_id: int,
    course_id: int,
    class_id: int,
    assignment_id: int,
) -> None:
    events = [
        ("PAGE_VIEW", "course_intro", "", None),
        ("RESOURCE_SEARCH", "resources", "5G NAS", None),
        ("RESOURCE_OPEN", "resources", "5G NAS", "external:standard_source:3GPP-TS-24.501"),
        ("QUESTION_ASKED", "student_chat", "5G NAS", None),
        ("PRACTICE_GUIDE_GENERATED", "practice_guide", "5G NAS", None),
        ("ASSIGNMENT_PROGRESS_UPDATED", "assignment", "5G NAS", None),
        ("STUCK_REPORTED", "assignment", "5G NAS", None),
    ]
    for event_type, page_name, module, resource_id in events:
        logger.log_learning_event(
            session_id=session_id,
            user_id=user_id,
            course_id=course_id,
            class_id=class_id,
            event_type=event_type,
            page_name=page_name,
            module=module,
            resource_id=resource_id,
            assignment_id=str(assignment_id) if event_type in {"ASSIGNMENT_PROGRESS_UPDATED", "STUCK_REPORTED"} else None,
            metadata={"demo_seed": True},
            dedupe_key=f"demo-phase3:{event_type}:{page_name}:{resource_id or ''}",
        )


def _seed_questions_and_faq(logger: LearningLogger, user_id: int, course_id: int, class_id: int) -> list[int]:
    questions = [
        "How should I evaluate 5G NAS replay risk in a safe local lab?",
        "5G NAS authentication risk analysis should cite which evidence?",
        "I am stuck on the result analysis step for NAS replay-risk metrics.",
    ]
    question_ids: list[int] = []
    for question in questions:
        question_ids.append(_seed_question_once(logger, user_id, course_id, class_id, question))
    label = "Demo approved 5G NAS FAQ"
    with logger._connect() as conn:
        existing = conn.execute("SELECT id FROM question_clusters WHERE label = ?", (label,)).fetchone()
    if existing:
        return question_ids
    cluster_id = logger.create_question_cluster(
        label=label,
        representative_question=questions[0],
        course_id=course_id,
        class_id=class_id,
        approved=True,
        is_noise=False,
        source="teacher",
    )
    for index, question_id in enumerate(question_ids):
        logger.add_question_cluster_member(
            cluster_id=cluster_id,
            question_id=question_id,
            distance=0.05 * index,
            is_representative=index == 0,
        )
    return question_ids


def _seed_question_once(logger: LearningLogger, user_id: int, course_id: int, class_id: int, question: str) -> int:
    with logger._connect() as conn:
        existing = conn.execute(
            """
            SELECT id FROM questions
            WHERE user_id = ? AND course_id = ? AND class_id = ? AND question = ?
            LIMIT 1
            """,
            (str(user_id), course_id, class_id, question),
        ).fetchone()
    if existing:
        return int(existing["id"])
    return logger.log_question(
        str(user_id),
        question,
        {
            "answer": "Use local traces, replay protection evidence, identity exposure evidence, and defensive metrics.",
            "module": "5G NAS",
            "intent": "course_qa",
            "confidence": 0.86,
            "abstained": False,
            "course_id": course_id,
            "class_id": class_id,
            "citations": [
                {
                    "source": "5g_nas_demo_notes.md",
                    "title": "5G NAS Demo Notes",
                    "score": 0.9,
                    "snippet": "5G NAS authentication risk analysis should cite replay protection and identity exposure evidence.",
                }
            ],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize explicit project data.")
    parser.add_argument(
        "--shared-materials-only",
        action="store_true",
        help="install only the approved built-in textbooks into demo-course",
    )
    args = parser.parse_args()
    if args.shared_materials_only:
        result = seed_shared_textbooks()
        print(json.dumps(result, ensure_ascii=False))
    else:
        result = seed_demo_data()
        print("Demo seed completed.")
        print(f"course_id={result['course_id']} class_id={result['class_id']}")
        print("accounts=teacher_demo, student_demo, student_demo2")
        print("default_password=demo123456")
        print(f"backup_path={result.get('backup_path')}")
        print("Please change demo passwords before non-local use.")
