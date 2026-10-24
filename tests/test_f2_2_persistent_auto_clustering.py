from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from backend import main
from backend.storage.logger import LearningLogger


def _question_result(course_id: int, class_id: int | None = None) -> dict:
    return {
        "answer": "demo",
        "module": "5G NAS",
        "intent": "course_qa",
        "confidence": 0.9,
        "abstained": False,
        "course_id": course_id,
        "class_id": class_id,
    }


def test_migration_backfills_unclustered_questions_instead_of_max_id(tmp_path) -> None:
    logger = LearningLogger(tmp_path / "backfill.db")
    course_id = logger.default_course_id()
    class_id = logger.default_class_id()
    question_ids = [
        logger.log_question("student_demo", f"migration question {index}", _question_result(course_id, class_id))
        for index in range(3)
    ]
    cluster_id = logger.create_question_cluster(
        label="Existing cluster",
        representative_question="migration question 0",
        course_id=course_id,
        class_id=class_id,
    )
    logger.add_question_cluster_member(cluster_id=cluster_id, question_id=question_ids[0], is_representative=True)
    logger.add_question_cluster_member(cluster_id=cluster_id, question_id=question_ids[1])

    with logger._connect() as conn:
        conn.execute("DELETE FROM schema_migrations WHERE version = 15")

    logger.init_db()

    with logger._connect() as conn:
        row = conn.execute(
            """
            SELECT course_id, class_id, last_successful_question_id,
                   pending_question_count, status
            FROM question_cluster_jobs
            WHERE course_id = ? AND class_id = ?
            """,
            (course_id, class_id),
        ).fetchone()

    assert row is not None
    assert int(row["pending_question_count"]) == 1
    assert int(row["last_successful_question_id"]) != max(question_ids)
    assert row["status"] == "PENDING"


def test_auto_cluster_threshold_survives_process_restart(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUESTION_CLUSTER_AUTO_THRESHOLD", "3")
    original_logger = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "restart.db")
    calls: list[tuple[int | None, int | None]] = []
    original_runner = main._run_auto_question_cluster_rebuild
    monkeypatch.setattr(
        main,
        "_run_auto_question_cluster_rebuild",
        lambda course_id, class_id=None: calls.append((course_id, class_id)),
    )
    try:
        main._auto_cluster_new_question_counts.clear()
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()
        for index in range(2):
            main._record_question(
                "student_demo",
                f"restart threshold NAS question {index}",
                _question_result(course_id, class_id),
            )

        main._auto_cluster_new_question_counts.clear()
        main._record_question(
            "student_demo",
            "restart threshold NAS question 2",
            _question_result(course_id, class_id),
        )

        assert calls == [(course_id, class_id)]
    finally:
        main._auto_cluster_new_question_counts.clear()
        main.learning_logger = original_logger
        main._run_auto_question_cluster_rebuild = original_runner


def test_failed_auto_cluster_job_is_visible_and_retryable(tmp_path) -> None:
    logger = LearningLogger(tmp_path / "retry.db")
    course_id = logger.default_course_id()
    class_id = logger.default_class_id()
    question_id = logger.log_question(
        "student_demo",
        "retryable NAS question",
        _question_result(course_id, class_id),
    )

    assert logger.register_question_cluster_for_processing(
        question_id=question_id,
        course_id=course_id,
        class_id=class_id,
        threshold=1,
    )
    first_claim = logger.claim_question_cluster_job(course_id, class_id, worker_id="worker-a")
    assert first_claim is not None
    logger.fail_question_cluster_job(
        course_id,
        class_id,
        worker_id="worker-a",
        error_code="CLUSTER_REBUILD_FAILED",
        error_message="embedding unavailable",
    )

    failed = logger.question_cluster_jobs(course_id=course_id, class_id=class_id)[0]
    assert failed["status"] == "FAILED"
    assert failed["error_code"] == "CLUSTER_REBUILD_FAILED"
    assert failed["error_message"] == "embedding unavailable"

    scopes = logger.recover_question_cluster_jobs(threshold=1)
    assert (course_id, class_id) in scopes
    second_claim = logger.claim_question_cluster_job(course_id, class_id, worker_id="worker-b")
    assert second_claim is not None
    logger.complete_question_cluster_job(course_id, class_id, worker_id="worker-b")
    recovered = logger.question_cluster_jobs(course_id=course_id, class_id=class_id)[0]
    assert recovered["status"] == "SUCCEEDED"
    assert recovered["error_code"] is None


def test_two_workers_claim_one_persistent_cluster_job(tmp_path) -> None:
    db_path = tmp_path / "claim.db"
    logger = LearningLogger(db_path)
    course_id = logger.default_course_id()
    class_id = logger.default_class_id()
    question_id = logger.log_question(
        "student_demo",
        "single claim NAS question",
        _question_result(course_id, class_id),
    )
    assert logger.register_question_cluster_for_processing(
        question_id=question_id,
        course_id=course_id,
        class_id=class_id,
        threshold=1,
    )

    def claim(worker_id: str):
        return LearningLogger(db_path).claim_question_cluster_job(
            course_id,
            class_id,
            worker_id=worker_id,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, ["worker-a", "worker-b"]))

    assert sum(claim is not None for claim in claims) == 1


def test_teacher_cluster_endpoint_exposes_failed_job_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-f2-2")
    original_logger = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "api-status.db")
    main.learning_logger.seed_demo_identity()
    try:
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()
        question_id = main.learning_logger.log_question(
            "student_demo",
            "visible failed cluster question",
            _question_result(course_id, class_id),
        )
        main.learning_logger.register_question_cluster_for_processing(
            question_id=question_id,
            course_id=course_id,
            class_id=class_id,
            threshold=1,
        )
        assert main.learning_logger.claim_question_cluster_job(
            course_id,
            class_id,
            worker_id="api-status-worker",
        )
        main.learning_logger.fail_question_cluster_job(
            course_id,
            class_id,
            worker_id="api-status-worker",
            error_code="CLUSTER_REBUILD_FAILED",
            error_message="retry later",
        )

        client = TestClient(main.app)
        login = client.post(
            "/api/auth/login",
            json={"username": "teacher_demo", "password": "demo123456"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]
        if login.json().get("must_change_password"):
            changed = client.post(
                "/api/auth/change-password",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "current_password": "demo123456",
                    "new_password": "f2-2-api-teacher-123",
                },
            )
            assert changed.status_code == 200
            token = changed.json()["access_token"]

        response = client.get(
            "/api/teacher/question-clusters",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        job = next(item for item in response.json()["jobs"] if item["course_id"] == course_id)
        assert job["status"] == "FAILED"
        assert job["error_code"] == "CLUSTER_REBUILD_FAILED"
    finally:
        main.learning_logger = original_logger
