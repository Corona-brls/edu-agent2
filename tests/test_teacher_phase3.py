from __future__ import annotations

from fastapi.testclient import TestClient

from backend import main
from backend.analytics.clustering import QuestionClusteringService
from backend.auth.password import hash_password
from backend.rag.document_parser import Document, Page
from backend.storage.logger import LearningLogger, sha256_bytes


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-teacher-phase3")
    original = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "teacher_phase3.db")
    main.learning_logger.seed_demo_identity()
    main.learning_logger.ensure_demo_assignments()
    _seed_knowledge(main.learning_logger, tmp_path)
    client = TestClient(main.app)
    client._original_logger = original  # type: ignore[attr-defined]
    return client


def _restore(client: TestClient) -> None:
    main.learning_logger = client._original_logger  # type: ignore[attr-defined]


def _login(client: TestClient, username: str, password: str = "demo123456") -> tuple[str, dict]:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    payload = response.json()
    if payload.get("must_change_password"):
        changed = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {payload['access_token']}"},
            json={"current_password": password, "new_password": f"{username}-changed-123"},
        )
        assert changed.status_code == 200
        payload = changed.json()
    return payload["access_token"], payload["user"]


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seed_knowledge(logger: LearningLogger, tmp_path) -> None:
    course_id = logger.default_course_id()
    class_id = logger.default_class_id()
    document = Document(
        source_path=tmp_path / "phase3-faq.md",
        filename="phase3-faq.md",
        file_type=".md",
        title="Phase 3 FAQ Evidence",
        text="5G NAS authentication risk analysis should cite replay protection and identity exposure evidence.",
        pages=[
            Page(
                text="5G NAS authentication risk analysis should cite replay protection and identity exposure evidence.",
                page=1,
                section="faq",
                source_type="markdown",
            )
        ],
    )
    saved = logger.save_document(
        document=document,
        content_hash=sha256_bytes(document.text.encode("utf-8")),
        chunks=[document.text],
        byte_size=len(document.text),
        index_status="INDEXED",
        course_id=course_id,
        class_id=class_id,
    )
    logger.set_document_publication(int(saved["id"]), published=True, actor_id=1)


def _create_session(client: TestClient, token: str, sid: str) -> None:
    assert client.post("/api/learning/sessions", headers=_headers(token), json={"client_session_id": sid}).status_code == 200


def _event(client: TestClient, token: str, sid: str, event_type: str, **payload):
    data = {"client_session_id": sid, "event_type": event_type, **payload}
    response = client.post("/api/learning/events", headers=_headers(token), json=data)
    assert response.status_code == 200
    return response.json()


def _first_assignment(token: str, client: TestClient) -> tuple[dict, dict]:
    assignment = client.get("/api/assignments", headers=_headers(token)).json()["assignments"][0]
    return assignment, assignment["steps"][0]


def test_phase3_summary_counts_events_without_agent_internal_steps(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        student_token, _ = _login(client, "student_demo")
        teacher_token, _ = _login(client, "teacher_demo")
        sid = "phase3-summary-session"
        _create_session(client, student_token, sid)
        _event(client, student_token, sid, "PAGE_VIEW", page_name="course_intro")
        _event(client, student_token, sid, "PAGE_VIEW", page_name="resources")
        _event(client, student_token, sid, "RESOURCE_SEARCH", page_name="resources", metadata={"query": "NAS"})
        _event(client, student_token, sid, "RESOURCE_OPEN", page_name="resources", resource_id="document:1")
        _event(client, student_token, sid, "QUESTION_ASKED", page_name="student_chat", module="5G NAS")
        _event(client, student_token, sid, "PRACTICE_GUIDE_GENERATED", page_name="practice_guide", module="5G NAS")
        main.learning_logger.log_question(
            "2",
            "5G NAS replay risk?",
            {
                "answer": "demo",
                "module": "5G NAS",
                "intent": "course_qa",
                "confidence": 0.9,
                "abstained": False,
                "course_id": main.learning_logger.default_course_id(),
                "class_id": main.learning_logger.default_class_id(),
            },
        )

        summary = client.get("/api/teacher/phase3/summary", headers=_headers(teacher_token)).json()

        assert summary["access"]["unique_active_students"] == 1
        assert summary["access"]["visit_count"] == 2
        assert summary["resources"]["search_count"] == 1
        assert summary["resources"]["open_count"] == 1
        assert summary["qa"]["question_count"] == 1
        assert summary["practice"]["guide_count"] == 1
        assert summary["filters"]["scope_note"].startswith("visit_count counts PAGE_VIEW")
    finally:
        _restore(client)


def test_phase3_course_class_and_time_filters(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        student_token, _ = _login(client, "student_demo")
        teacher_token, _ = _login(client, "teacher_demo")
        class_id = client.get("/api/classes", headers=_headers(teacher_token)).json()[0]["id"]
        sid = "phase3-time-session"
        _create_session(client, student_token, sid)
        old_event = _event(client, student_token, sid, "PAGE_VIEW", page_name="old")
        new_event = _event(client, student_token, sid, "PAGE_VIEW", page_name="new")
        with main.learning_logger._connect() as conn:
            conn.execute("UPDATE learning_events SET created_at = '2026-01-01T00:00:00' WHERE id = ?", (old_event["id"],))
            conn.execute("UPDATE learning_events SET created_at = '2026-09-20T00:00:00' WHERE id = ?", (new_event["id"],))

        filtered = client.get(
            "/api/teacher/phase3/summary",
            headers=_headers(teacher_token),
            params={"class_id": class_id, "start_at": "2026-09-01T00:00:00", "end_at": "2026-09-30T23:59:59"},
        )
        empty = client.get(
            "/api/teacher/phase3/summary",
            headers=_headers(teacher_token),
            params={"start_at": "2025-01-01T00:00:00", "end_at": "2025-12-31T23:59:59"},
        )

        assert filtered.status_code == 200
        assert filtered.json()["access"]["visit_count"] == 1
        assert filtered.json()["access"]["page_distribution"] == {"new": 1}
        assert empty.status_code == 200
        assert empty.json()["access"]["visit_count"] == 0
    finally:
        _restore(client)


def test_phase3_teacher_cannot_cross_course_or_student_scope(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        teacher_token, _ = _login(client, "teacher_demo")
        other_student_id = _seed_other_course(main.learning_logger)

        forbidden_course = client.get(
            "/api/teacher/phase3/summary",
            headers=_headers(teacher_token),
            params={"course_id": 999},
        )
        forbidden_timeline = client.get(
            f"/api/teacher/phase3/students/{other_student_id}/timeline",
            headers=_headers(teacher_token),
        )

        assert forbidden_course.status_code == 403
        assert forbidden_timeline.status_code == 403
    finally:
        _restore(client)


def test_legacy_teacher_scope_matches_phase3_summary_for_clusters_stuck_and_assignments(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        teacher_token, _ = _login(client, "teacher_demo")
        other = _seed_other_course_bundle(main.learning_logger)
        other_teacher_token, _ = _login(client, "other_teacher")
        admin_id = main.learning_logger.upsert_user(
            username="admin_scope",
            display_name="Admin Scope",
            password_hash=hash_password("admin-pass"),
            role="admin",
            must_change_password=False,
        )
        assert admin_id
        admin_token, _ = _login(client, "admin_scope", "admin-pass")

        forbidden_summary = client.get(
            "/api/teacher/phase3/summary",
            headers=_headers(teacher_token),
            params={"course_id": other["course_id"]},
        )
        forbidden_rebuild = client.post(
            "/api/teacher/question-clusters/rebuild",
            headers=_headers(teacher_token),
            json={"course_id": other["course_id"], "distance_threshold": 0.1, "min_cluster_size": 1},
        )
        other_rebuild = client.post(
            "/api/teacher/question-clusters/rebuild",
            headers=_headers(other_teacher_token),
            json={"course_id": other["course_id"], "distance_threshold": 0.1, "min_cluster_size": 1},
        )
        other_clusters = client.get(
            "/api/teacher/question-clusters",
            headers=_headers(other_teacher_token),
            params={"course_id": other["course_id"]},
        ).json()["clusters"]
        assert other_clusters
        other_cluster_id = other_clusters[0]["id"]

        assert forbidden_summary.status_code == 403
        assert forbidden_rebuild.status_code == 403
        assert other_rebuild.status_code == 200
        assert client.patch(
            f"/api/teacher/question-clusters/{other_cluster_id}",
            headers=_headers(teacher_token),
            json={"label": "cross-course rename"},
        ).status_code == 403
        assert client.post(
            f"/api/teacher/question-clusters/{other_cluster_id}/approve",
            headers=_headers(teacher_token),
        ).status_code == 403
        assert client.patch(
            f"/api/teacher/question-clusters/{other_cluster_id}",
            headers=_headers(admin_token),
            json={"label": "admin rename"},
        ).status_code == 200

        visible_clusters = client.get("/api/teacher/question-clusters", headers=_headers(teacher_token)).json()["clusters"]
        assert all(int(cluster["course_id"]) != other["course_id"] for cluster in visible_clusters)

        other_student_headers = _headers(_login(client, "other_student")[0])
        stuck = client.post(
            "/api/stuck-reports",
            headers=other_student_headers,
            json={
                "assignment_id": other["assignment_id"],
                "step_id": other["step_id"],
                "module": "Blockchain",
                "reason": "Need help with result analysis.",
            },
        )
        assert stuck.status_code == 200
        report_id = stuck.json()["report"]["id"]

        assert client.get(
            "/api/teacher/stuck-analysis",
            headers=_headers(teacher_token),
            params={"course_id": other["course_id"]},
        ).status_code == 403
        assert client.get(
            "/api/teacher/stuck-analysis",
            headers=_headers(teacher_token),
            params={"assignment_id": other["assignment_id"]},
        ).status_code == 403
        assert client.get(
            "/api/teacher/stuck-analysis",
            headers=_headers(teacher_token),
            params={"step_id": other["step_id"]},
        ).status_code == 403
        assert client.post(
            f"/api/teacher/stuck-reports/{report_id}/action",
            headers=_headers(teacher_token),
            json={"action": "confirm"},
        ).status_code == 403
        assert client.post(
            f"/api/teacher/stuck-reports/{report_id}/action",
            headers=_headers(other_teacher_token),
            json={"action": "confirm"},
        ).status_code == 200

        student_token, _ = _login(client, "student_demo")
        assignments = client.get("/api/assignments", headers=_headers(student_token)).json()["assignments"]
        assert all(int(item["course_id"]) != other["course_id"] for item in assignments)
        assert client.get(
            f"/api/assignments/{other['assignment_id']}/progress",
            headers=_headers(student_token),
        ).status_code == 403
        assert client.post(
            f"/api/assignments/{other['assignment_id']}/progress",
            headers=_headers(student_token),
            json={"step_id": other["step_id"], "status": "in_progress", "progress_percent": 10},
        ).status_code == 403
    finally:
        _restore(client)


def test_phase3_teacher_cannot_mix_course_and_class_scope(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        teacher_token, teacher = _login(client, "teacher_demo")
        with main.learning_logger._connect() as conn:
            base_class = conn.execute(
                "SELECT id, course_id FROM classes WHERE teacher_id = ? ORDER BY id LIMIT 1",
                (teacher["id"],),
            ).fetchone()
            cursor = conn.execute(
                """
                INSERT INTO courses (slug, name, description, assessment_method)
                VALUES ('teacher-second-course', 'Teacher Second Course', '', '')
                """
            )
            second_course_id = int(cursor.lastrowid)
            cursor = conn.execute(
                """
                INSERT INTO classes (course_id, slug, name, term, teacher_id)
                VALUES (?, 'teacher-second-class', 'Teacher Second Class', '2026', ?)
                """,
                (second_course_id, teacher["id"]),
            )
            second_class_id = int(cursor.lastrowid)
            conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (second_class_id, teacher["id"]))

        mixed = client.get(
            "/api/teacher/phase3/summary",
            headers=_headers(teacher_token),
            params={"course_id": int(base_class["course_id"]), "class_id": second_class_id},
        )

        assert mixed.status_code == 403
        assert mixed.json()["error_code"] == "CLASS_FORBIDDEN"
    finally:
        _restore(client)


def test_unassigned_teacher_is_rejected_from_phase3_and_cluster_endpoints(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        teacher_id = main.learning_logger.upsert_user(
            username="floating_teacher",
            display_name="Floating Teacher",
            password_hash=hash_password("floating-pass"),
            role="teacher",
            must_change_password=False,
        )
        assert teacher_id
        token, _ = _login(client, "floating_teacher", "floating-pass")
        headers = _headers(token)
        before_clusters = len(main.learning_logger.question_clusters())
        with main.learning_logger._connect() as conn:
            before_members = conn.execute("SELECT COUNT(*) FROM question_cluster_members").fetchone()[0]

        endpoints = [
            client.get("/api/teacher/phase3/summary", headers=headers),
            client.get("/api/teacher/phase3/faq-export.csv", headers=headers),
            client.get("/api/teacher/question-clusters", headers=headers),
            client.get("/api/teacher/question-clusters/faq-export", headers=headers),
            client.post(
                "/api/teacher/question-clusters/rebuild",
                headers=headers,
                json={"distance_threshold": 0.1, "min_cluster_size": 1},
            ),
        ]

        assert [response.status_code for response in endpoints] == [403, 403, 403, 403, 403]
        assert len(main.learning_logger.question_clusters()) == before_clusters
        with main.learning_logger._connect() as conn:
            after_members = conn.execute("SELECT COUNT(*) FROM question_cluster_members").fetchone()[0]
        assert after_members == before_members
        assert endpoints[-1].json()["error_code"] == "COURSE_FORBIDDEN"
    finally:
        _restore(client)


def test_unscoped_legacy_cluster_is_admin_only(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        teacher_token, _ = _login(client, "teacher_demo")
        admin_id = main.learning_logger.upsert_user(
            username="admin_cluster_scope",
            display_name="Admin Cluster Scope",
            password_hash=hash_password("admin-pass"),
            role="admin",
            must_change_password=False,
        )
        assert admin_id
        admin_token, _ = _login(client, "admin_cluster_scope", "admin-pass")
        cluster_id = main.learning_logger.create_question_cluster(
            label="Legacy unscoped cluster",
            representative_question="legacy question without course scope",
            course_id=None,
            class_id=None,
            approved=False,
            is_noise=False,
            source="auto",
        )

        forbidden_rename = client.patch(
            f"/api/teacher/question-clusters/{cluster_id}",
            headers=_headers(teacher_token),
            json={"label": "teacher should not rename"},
        )
        forbidden_approve = client.post(
            f"/api/teacher/question-clusters/{cluster_id}/approve",
            headers=_headers(teacher_token),
        )
        admin_approve = client.post(
            f"/api/teacher/question-clusters/{cluster_id}/approve",
            headers=_headers(admin_token),
        )

        assert forbidden_rename.status_code == 403
        assert forbidden_rename.json()["error_code"] == "COURSE_FORBIDDEN"
        assert forbidden_approve.status_code == 403
        assert admin_approve.status_code == 200
    finally:
        _restore(client)


def test_teacher_phase3_summary_filters_clusters_by_authorized_classes(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        teacher_token, _ = _login(client, "teacher_demo")
        same_course = _seed_same_course_other_class(main.learning_logger)
        hidden_cluster_id = main.learning_logger.create_question_cluster(
            label="Hidden same course class FAQ",
            representative_question="Other class NAS question",
            course_id=same_course["course_id"],
            class_id=same_course["class_id"],
            approved=False,
            is_noise=False,
            source="auto",
        )
        main.learning_logger.add_question_cluster_member(
            cluster_id=hidden_cluster_id,
            question_id=same_course["question_id"],
            is_representative=True,
        )

        summary = client.get("/api/teacher/phase3/summary", headers=_headers(teacher_token))
        clusters = client.get("/api/teacher/question-clusters", headers=_headers(teacher_token))

        assert summary.status_code == 200
        assert hidden_cluster_id not in [item["id"] for item in summary.json()["clusters"]["items"]]
        assert clusters.status_code == 200
        assert hidden_cluster_id not in [item["id"] for item in clusters.json()["clusters"]]
    finally:
        _restore(client)


def test_cluster_scope_validates_member_classes_not_only_cluster_class_id(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        teacher_token, _ = _login(client, "teacher_demo")
        same_course = _seed_same_course_other_class(main.learning_logger)
        forged_cluster_id = main.learning_logger.create_question_cluster(
            label="Forged class cluster",
            representative_question="Other class question with wrong cluster class",
            course_id=same_course["course_id"],
            class_id=main.learning_logger.default_class_id(),
            approved=False,
            is_noise=False,
            source="auto",
        )
        main.learning_logger.add_question_cluster_member(
            cluster_id=forged_cluster_id,
            question_id=same_course["question_id"],
            is_representative=True,
        )

        rename = client.patch(
            f"/api/teacher/question-clusters/{forged_cluster_id}",
            headers=_headers(teacher_token),
            json={"label": "should not rename cross-class members"},
        )
        approve = client.post(
            f"/api/teacher/question-clusters/{forged_cluster_id}/approve",
            headers=_headers(teacher_token),
        )

        assert rename.status_code == 403
        assert approve.status_code == 403
    finally:
        _restore(client)


def test_teacher_cannot_merge_clusters_across_classes_even_when_authorized_for_both(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        teacher_token, teacher = _login(client, "teacher_demo")
        with main.learning_logger._connect() as conn:
            default_course_id = main.learning_logger.default_course_id()
            cursor = conn.execute(
                """
                INSERT INTO classes (course_id, slug, name, term, teacher_id)
                VALUES (?, 'teacher-extra-same-course', 'Teacher Extra Same Course', '2026', ?)
                """,
                (default_course_id, teacher["id"]),
            )
            second_class_id = int(cursor.lastrowid)
            conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (second_class_id, teacher["id"]))
        qid_a = main.learning_logger.log_question(
            str(teacher["id"]),
            "Class A NAS authentication question",
            {
                "answer": "demo",
                "module": "5G NAS",
                "intent": "course_qa",
                "confidence": 0.8,
                "abstained": False,
                "course_id": default_course_id,
                "class_id": main.learning_logger.default_class_id(),
            },
        )
        qid_b = main.learning_logger.log_question(
            str(teacher["id"]),
            "Class B NAS authentication question",
            {
                "answer": "demo",
                "module": "5G NAS",
                "intent": "course_qa",
                "confidence": 0.8,
                "abstained": False,
                "course_id": default_course_id,
                "class_id": second_class_id,
            },
        )
        cluster_a = main.learning_logger.create_question_cluster(
            label="Class A FAQ",
            representative_question="Class A NAS authentication question",
            course_id=default_course_id,
            class_id=main.learning_logger.default_class_id(),
            source="teacher",
        )
        cluster_b = main.learning_logger.create_question_cluster(
            label="Class B FAQ",
            representative_question="Class B NAS authentication question",
            course_id=default_course_id,
            class_id=second_class_id,
            source="teacher",
        )
        main.learning_logger.add_question_cluster_member(cluster_id=cluster_a, question_id=qid_a, is_representative=True)
        main.learning_logger.add_question_cluster_member(cluster_id=cluster_b, question_id=qid_b, is_representative=True)

        response = client.post(
            "/api/teacher/question-clusters/merge",
            headers=_headers(teacher_token),
            json={"target_cluster_id": cluster_a, "source_cluster_ids": [cluster_b], "label": "merged"},
        )

        assert response.status_code == 403
        assert response.json()["error_code"] == "CLASS_FORBIDDEN"
        labels = {item["id"]: item["label"] for item in main.learning_logger.question_clusters(course_id=default_course_id)}
        assert labels[cluster_a] == "Class A FAQ"
        assert labels[cluster_b] == "Class B FAQ"
    finally:
        _restore(client)


def test_phase3_admin_can_view_all_and_empty_summary_is_stable(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        admin_id = main.learning_logger.upsert_user(
            username="admin_demo",
            display_name="Admin Demo",
            password_hash=hash_password("admin-pass"),
            role="admin",
            must_change_password=False,
        )
        assert admin_id
        admin_token, _ = _login(client, "admin_demo", "admin-pass")

        summary = client.get("/api/teacher/phase3/summary", headers=_headers(admin_token)).json()

        assert "access" in summary
        assert "clusters" in summary
        assert isinstance(summary["access"]["page_distribution"], dict)
    finally:
        _restore(client)


def test_phase3_faq_csv_escapes_formula_injection(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        teacher_token, _ = _login(client, "teacher_demo")
        qid = main.learning_logger.log_question(
            "2",
            "5G NAS FAQ formula test",
            {
                "answer": "demo",
                "module": "5G NAS",
                "intent": "course_qa",
                "confidence": 0.9,
                "abstained": False,
                "course_id": main.learning_logger.default_course_id(),
                "class_id": main.learning_logger.default_class_id(),
            },
        )
        cluster_id = main.learning_logger.create_question_cluster(
            label="=HYPERLINK(\"http://evil.test\")",
            representative_question="+5G NAS authentication risk analysis",
            course_id=main.learning_logger.default_course_id(),
            class_id=main.learning_logger.default_class_id(),
            approved=True,
            is_noise=False,
            source="teacher",
        )
        main.learning_logger.add_question_cluster_member(cluster_id=cluster_id, question_id=qid, is_representative=True)

        response = client.get("/api/teacher/phase3/faq-export.csv", headers=_headers(teacher_token))
        content = response.content.decode("utf-8-sig")

        assert response.status_code == 200
        assert "'=HYPERLINK" in content
        assert "'+5G NAS authentication risk analysis" in content
    finally:
        _restore(client)


def test_phase3_faq_csv_exports_only_approved_evidence_backed_clusters(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        teacher_token, _ = _login(client, "teacher_demo")
        headers = _headers(teacher_token)
        qid = main.learning_logger.log_question(
            "2",
            "5G NAS authentication risk analysis FAQ",
            {
                "answer": "demo",
                "module": "5G NAS",
                "intent": "course_qa",
                "confidence": 0.9,
                "abstained": False,
                "course_id": main.learning_logger.default_course_id(),
                "class_id": main.learning_logger.default_class_id(),
            },
        )
        cluster_id = main.learning_logger.create_question_cluster(
            label="Approved NAS FAQ",
            representative_question="5G NAS authentication risk analysis FAQ",
            course_id=main.learning_logger.default_course_id(),
            class_id=main.learning_logger.default_class_id(),
            approved=False,
            is_noise=False,
            source="auto",
        )
        main.learning_logger.add_question_cluster_member(cluster_id=cluster_id, question_id=qid, is_representative=True)

        draft_csv = client.get("/api/teacher/phase3/faq-export.csv", headers=headers).content.decode("utf-8-sig")
        assert "Approved NAS FAQ" not in draft_csv

        approved = client.post(f"/api/teacher/question-clusters/{cluster_id}/approve", headers=headers)
        final_csv = client.get("/api/teacher/phase3/faq-export.csv", headers=headers).content.decode("utf-8-sig")

        assert approved.status_code == 200
        assert "Approved NAS FAQ" in final_csv
    finally:
        _restore(client)


def test_phase3_student_timeline_requires_teacher_scope(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        student_token, user = _login(client, "student_demo")
        teacher_token, _ = _login(client, "teacher_demo")
        sid = "phase3-timeline-session"
        _create_session(client, student_token, sid)
        _event(client, student_token, sid, "PAGE_VIEW", page_name="course_intro")

        timeline = client.get(
            f"/api/teacher/phase3/students/{user['id']}/timeline",
            headers=_headers(teacher_token),
        )

        assert timeline.status_code == 200
        assert timeline.json()["summary_only_by_default"] is True
        assert timeline.json()["events"][0]["event_type"] == "LOGIN"
    finally:
        _restore(client)


def _seed_other_course(logger: LearningLogger) -> int:
    from backend.auth.password import hash_password

    teacher_id = logger.upsert_user(
        username="other_teacher",
        display_name="Other Teacher",
        password_hash=hash_password("demo123456"),
        role="teacher",
        must_change_password=False,
    )
    student_id = logger.upsert_user(
        username="other_student",
        display_name="Other Student",
        password_hash=hash_password("demo123456"),
        role="student",
        must_change_password=False,
    )
    with logger._connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO courses (slug, name, description, assessment_method)
            VALUES ('other-course', 'Other Course', '', '')
            """
        )
        course_id = int(cursor.lastrowid)
        cursor = conn.execute(
            """
            INSERT INTO classes (course_id, slug, name, term, teacher_id)
            VALUES (?, 'other-class', 'Other Class', '2026', ?)
            """,
            (course_id, teacher_id),
        )
        class_id = int(cursor.lastrowid)
        conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (class_id, teacher_id))
        conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (class_id, student_id))
    return int(student_id)


def _seed_other_course_bundle(logger: LearningLogger) -> dict[str, int]:
    teacher_id = logger.upsert_user(
        username="other_teacher",
        display_name="Other Teacher",
        password_hash=hash_password("demo123456"),
        role="teacher",
        must_change_password=False,
    )
    student_id = logger.upsert_user(
        username="other_student",
        display_name="Other Student",
        password_hash=hash_password("demo123456"),
        role="student",
        must_change_password=False,
    )
    with logger._connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO courses (slug, name, description, assessment_method)
            VALUES ('other-scope-course', 'Other Scope Course', '', '')
            """
        )
        course_id = int(cursor.lastrowid)
        cursor = conn.execute(
            """
            INSERT INTO classes (course_id, slug, name, term, teacher_id)
            VALUES (?, 'other-scope-class', 'Other Scope Class', '2026', ?)
            """,
            (course_id, teacher_id),
        )
        class_id = int(cursor.lastrowid)
        conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (class_id, teacher_id))
        conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (class_id, student_id))
        cursor = conn.execute(
            """
            INSERT INTO assignments (course_id, module, title, description, visibility)
            VALUES (?, 'Blockchain', 'Blockchain Covert Channel Lab', '', 'course')
            """,
            (course_id,),
        )
        assignment_id = int(cursor.lastrowid)
        cursor = conn.execute(
            """
            INSERT INTO assignment_steps (assignment_id, step_index, title, description, module)
            VALUES (?, 1, 'Result analysis', 'Analyze experiment results.', 'Blockchain')
            """,
            (assignment_id,),
        )
        step_id = int(cursor.lastrowid)
    logger.log_question(
        str(student_id),
        "How should blockchain covert communication be evaluated?",
        {
            "answer": "demo",
            "module": "Blockchain",
            "intent": "course_qa",
            "confidence": 0.8,
            "abstained": False,
            "course_id": course_id,
            "class_id": class_id,
        },
    )
    return {
        "course_id": course_id,
        "class_id": class_id,
        "assignment_id": assignment_id,
        "step_id": step_id,
        "student_id": int(student_id),
        "teacher_id": int(teacher_id),
    }


def _seed_same_course_other_class(logger: LearningLogger) -> dict[str, int]:
    teacher_id = logger.upsert_user(
        username="same_course_teacher",
        display_name="Same Course Teacher",
        password_hash=hash_password("demo123456"),
        role="teacher",
        must_change_password=False,
    )
    student_id = logger.upsert_user(
        username="same_course_student",
        display_name="Same Course Student",
        password_hash=hash_password("demo123456"),
        role="student",
        must_change_password=False,
    )
    course_id = logger.default_course_id()
    with logger._connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO classes (course_id, slug, name, term, teacher_id)
            VALUES (?, 'same-course-other-class', 'Same Course Other Class', '2026', ?)
            """,
            (course_id, teacher_id),
        )
        class_id = int(cursor.lastrowid)
        conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (class_id, teacher_id))
        conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (class_id, student_id))
    question_id = logger.log_question(
        str(student_id),
        "Other class NAS authentication risk?",
        {
            "answer": "demo",
            "module": "5G NAS",
            "intent": "course_qa",
            "confidence": 0.8,
            "abstained": False,
            "course_id": course_id,
            "class_id": class_id,
        },
    )
    return {"course_id": int(course_id), "class_id": class_id, "question_id": question_id}
