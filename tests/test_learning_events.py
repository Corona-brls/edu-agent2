from __future__ import annotations

from fastapi.testclient import TestClient

from backend import main
from backend.auth.password import hash_password
from backend.storage.logger import LearningLogger


def _client(tmp_path, monkeypatch, *, raise_server_exceptions: bool = True) -> TestClient:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-learning-events")
    monkeypatch.setenv("SESSION_IDLE_TIMEOUT_MINUTES", "30")
    original = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "events.db")
    main.learning_logger.seed_demo_identity()
    client = TestClient(main.app, raise_server_exceptions=raise_server_exceptions)
    client._original_logger = original  # type: ignore[attr-defined]
    return client


def _restore(client: TestClient) -> None:
    main.learning_logger = client._original_logger  # type: ignore[attr-defined]


def _login(client: TestClient, username: str) -> tuple[str, dict]:
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": "demo123456"},
    )
    assert response.status_code == 200
    payload = response.json()
    if payload.get("must_change_password"):
        changed = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {payload['access_token']}"},
            json={"current_password": "demo123456", "new_password": f"{username}-changed-123"},
        )
        assert changed.status_code == 200
        payload = changed.json()
    return payload["access_token"], payload["user"]


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _counts() -> dict[str, int]:
    with main.learning_logger._connect() as conn:
        return {
            "sessions": conn.execute("SELECT COUNT(*) FROM learning_sessions").fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0],
        }


def _seed_two_course_scope() -> dict[str, int]:
    password_hash = hash_password("demo123456")
    with main.learning_logger._connect() as conn:
        teacher_a = main.learning_logger.upsert_user(
            username="teacher_scope_a",
            display_name="Teacher Scope A",
            password_hash=password_hash,
            role="teacher",
            must_change_password=False,
        )
        teacher_b = main.learning_logger.upsert_user(
            username="teacher_scope_b",
            display_name="Teacher Scope B",
            password_hash=password_hash,
            role="teacher",
            must_change_password=False,
        )
        floating_teacher = main.learning_logger.upsert_user(
            username="teacher_scope_none",
            display_name="Teacher Scope None",
            password_hash=password_hash,
            role="teacher",
            must_change_password=False,
        )
        student_a = main.learning_logger.upsert_user(
            username="student_scope_a",
            display_name="Student Scope A",
            password_hash=password_hash,
            role="student",
            must_change_password=False,
        )
        student_b = main.learning_logger.upsert_user(
            username="student_scope_b",
            display_name="Student Scope B",
            password_hash=password_hash,
            role="student",
            must_change_password=False,
        )
        admin = main.learning_logger.upsert_user(
            username="admin_scope",
            display_name="Admin Scope",
            password_hash=password_hash,
            role="admin",
            must_change_password=False,
        )
        course_a = conn.execute(
            "INSERT INTO courses (slug, name, description, assessment_method) VALUES (?, ?, '', '')",
            ("scope-course-a", "Scope Course A"),
        ).lastrowid
        course_b = conn.execute(
            "INSERT INTO courses (slug, name, description, assessment_method) VALUES (?, ?, '', '')",
            ("scope-course-b", "Scope Course B"),
        ).lastrowid
        class_a = conn.execute(
            "INSERT INTO classes (course_id, slug, name, term, teacher_id) VALUES (?, ?, ?, '', ?)",
            (course_a, "scope-class-a", "Scope Class A", teacher_a),
        ).lastrowid
        class_b = conn.execute(
            "INSERT INTO classes (course_id, slug, name, term, teacher_id) VALUES (?, ?, ?, '', ?)",
            (course_b, "scope-class-b", "Scope Class B", teacher_b),
        ).lastrowid
        conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (class_a, student_a))
        conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (class_b, student_b))
    return {
        "teacher_a": teacher_a,
        "teacher_b": teacher_b,
        "floating_teacher": floating_teacher,
        "student_a": student_a,
        "student_b": student_b,
        "admin": admin,
        "course_a": course_a,
        "course_b": course_b,
        "class_a": class_a,
        "class_b": class_b,
    }


def test_student_workflow_forms_ordered_timeline(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, user = _login(client, "student_demo")
        headers = _headers(token)
        sid = "client-session-ordered"

        assert client.post("/api/learning/sessions", headers=headers, json={"client_session_id": sid}).status_code == 200
        assert client.post(
            "/api/learning/events",
            headers=headers,
            json={"client_session_id": sid, "event_type": "PAGE_VIEW", "page_name": "course_intro"},
        ).status_code == 200
        assert client.post(
            "/api/learning/events",
            headers=headers,
            json={
                "client_session_id": sid,
                "event_type": "RESOURCE_SEARCH",
                "page_name": "resources",
                "module": "5G NAS",
                "metadata": {"query": "NAS security"},
            },
        ).status_code == 200
        assert client.post(
            "/api/learning/events",
            headers=headers,
            json={
                "client_session_id": sid,
                "event_type": "RESOURCE_OPEN",
                "page_name": "resources",
                "resource_id": "doc-1",
            },
        ).status_code == 200
        chat = client.post(
            "/api/chat",
            headers=headers,
            json={"question": "5G NAS 认证流程有哪些关键风险点？", "client_session_id": sid},
        )
        assert chat.status_code == 200
        guide = client.post(
            "/api/practice/guide",
            headers=headers,
            json={"topic": "基于区块链的隐蔽通信", "client_session_id": sid},
        )
        assert guide.status_code == 200
        assert client.post(
            "/api/learning/events",
            headers=headers,
            json={"client_session_id": sid, "event_type": "INNOVATION_SESSION_STARTED", "page_name": "innovation"},
        ).status_code == 200
        assert client.post(
            "/api/learning/events",
            headers=headers,
            json={"client_session_id": sid, "event_type": "ASSIGNMENT_PROGRESS_UPDATED", "page_name": "assignment"},
        ).status_code == 200
        assert client.post(
            "/api/learning/events",
            headers=headers,
            json={"client_session_id": sid, "event_type": "STUCK_REPORTED", "page_name": "assignment"},
        ).status_code == 200
        feedback = client.post(
            "/api/feedback",
            headers=headers,
            json={
                "question_id": chat.json()["question_id"],
                "rating": 5,
                "helpful": True,
                "client_session_id": sid,
            },
        )
        assert feedback.status_code == 200

        timeline = client.get(f"/api/students/{user['id']}/timeline", headers=headers).json()["events"]
        event_types = [item["event_type"] for item in timeline]

        assert event_types[:4] == ["LOGIN", "PAGE_VIEW", "RESOURCE_SEARCH", "RESOURCE_OPEN"]
        assert "QUESTION_ASKED" in event_types
        assert "PRACTICE_GUIDE_GENERATED" in event_types
        assert event_types[-1] == "FEEDBACK_SUBMITTED"
        assert all(item["user_id"] == user["id"] for item in timeline)
    finally:
        _restore(client)


def test_teacher_activity_statistics_count_visits_pages_and_recent_active(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        student_token, _ = _login(client, "student_demo")
        teacher_token, _ = _login(client, "teacher_demo")
        sid = "client-session-stats"
        student_headers = _headers(student_token)
        teacher_headers = _headers(teacher_token)

        client.post("/api/learning/sessions", headers=student_headers, json={"client_session_id": sid})
        client.post(
            "/api/learning/events",
            headers=student_headers,
            json={"client_session_id": sid, "event_type": "PAGE_VIEW", "page_name": "home"},
        )
        client.post(
            "/api/learning/events",
            headers=student_headers,
            json={"client_session_id": sid, "event_type": "PAGE_VIEW", "page_name": "resources"},
        )
        client.post(
            "/api/learning/events",
            headers=student_headers,
            json={"client_session_id": sid, "event_type": "RESOURCE_OPEN", "page_name": "resources"},
        )

        activity = client.get("/api/teacher/activity", headers=teacher_headers)
        payload = activity.json()

        assert activity.status_code == 200
        assert payload["unique_active_students"] == 1
        assert payload["visit_count"] == 4
        assert payload["active_days"] == 1
        assert payload["recent_active_at"]
        assert payload["page_distribution"]["resources"] == 2
        assert payload["event_distribution"]["LOGIN"] == 1
    finally:
        _restore(client)


def test_duplicate_events_and_heartbeat_do_not_inflate_activity(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, _ = _login(client, "student_demo")
        headers = _headers(token)
        sid = "client-session-dedupe"

        session = client.post("/api/learning/sessions", headers=headers, json={"client_session_id": sid}).json()
        first = client.post(
            "/api/learning/events",
            headers=headers,
            json={
                "client_session_id": sid,
                "event_type": "PAGE_VIEW",
                "page_name": "home",
                "dedupe_key": "home-once",
            },
        ).json()
        second = client.post(
            "/api/learning/events",
            headers=headers,
            json={
                "client_session_id": sid,
                "event_type": "PAGE_VIEW",
                "page_name": "home",
                "dedupe_key": "home-once",
            },
        ).json()

        heartbeat_1 = client.post(f"/api/learning/sessions/{sid}/heartbeat", headers=headers).json()
        heartbeat_2 = client.post(f"/api/learning/sessions/{sid}/heartbeat", headers=headers).json()

        assert first["duplicate"] is False
        assert second["duplicate"] is True
        assert second["id"] == first["id"]
        assert heartbeat_1["active_seconds"] >= session["active_seconds"]
        assert heartbeat_2["active_seconds"] == heartbeat_1["active_seconds"]
    finally:
        _restore(client)


def test_idle_timeout_gap_is_not_counted(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, _ = _login(client, "student_demo")
        headers = _headers(token)
        sid = "client-session-idle"
        session = client.post("/api/learning/sessions", headers=headers, json={"client_session_id": sid}).json()
        with main.learning_logger._connect() as conn:
            conn.execute(
                """
                UPDATE learning_sessions
                SET last_active_at = datetime(CURRENT_TIMESTAMP, '-2 hours')
                WHERE id = ?
                """,
                (session["id"],),
            )

        heartbeat = client.post(f"/api/learning/sessions/{sid}/heartbeat", headers=headers)

        assert heartbeat.status_code == 200
        assert heartbeat.json()["active_seconds"] == 0
    finally:
        _restore(client)


def test_student_cannot_submit_or_read_other_users_events(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token_1, user_1 = _login(client, "student_demo")
        token_2, user_2 = _login(client, "student_demo2")
        sid = "client-session-owner"
        client.post("/api/learning/sessions", headers=_headers(token_1), json={"client_session_id": sid})

        forged = client.post(
            "/api/learning/events",
            headers=_headers(token_2),
            json={"client_session_id": sid, "event_type": "PAGE_VIEW", "page_name": "home"},
        )
        timeline = client.get(f"/api/students/{user_1['id']}/timeline", headers=_headers(token_2))

        assert forged.status_code in {403, 404}
        assert timeline.status_code == 403
        assert user_1["id"] != user_2["id"]
    finally:
        _restore(client)


def test_student_cannot_reuse_other_users_client_session_id(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token_1, _ = _login(client, "student_demo")
        token_2, _ = _login(client, "student_demo2")
        sid = "client-session-conflict"
        created = client.post("/api/learning/sessions", headers=_headers(token_1), json={"client_session_id": sid})
        reused = client.post("/api/learning/sessions", headers=_headers(token_2), json={"client_session_id": sid})

        assert created.status_code == 200
        assert reused.status_code == 409
        assert reused.json()["error_code"] == "SESSION_CONFLICT"
    finally:
        _restore(client)


def test_teacher_scope_and_invalid_event_type(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        teacher_token, _ = _login(client, "teacher_demo")
        student_token, _ = _login(client, "student_demo")
        sid = "client-session-invalid"
        client.post("/api/learning/sessions", headers=_headers(student_token), json={"client_session_id": sid})

        invalid = client.post(
            "/api/learning/events",
            headers=_headers(student_token),
            json={"client_session_id": sid, "event_type": "OTHER", "page_name": "home"},
        )
        forbidden_class = client.get(
            "/api/teacher/activity?class_id=9999",
            headers=_headers(teacher_token),
        )

        assert invalid.status_code == 400
        assert invalid.json()["error_code"] == "INVALID_EVENT_TYPE"
        assert forbidden_class.status_code == 403
    finally:
        _restore(client)


def test_s0_unassigned_teacher_and_cross_scope_learning_writes_are_forbidden(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        scope = _seed_two_course_scope()
        teacher_token, _ = _login(client, "teacher_scope_none")
        student_token, _ = _login(client, "student_scope_a")
        admin_token, _ = _login(client, "admin_scope")

        before = _counts()
        teacher_activity = client.get("/api/teacher/activity", headers=_headers(teacher_token))
        teacher_session = client.post(
            "/api/learning/sessions",
            headers=_headers(teacher_token),
            json={"client_session_id": "s0-unassigned-teacher"},
        )
        assert teacher_activity.status_code == 403
        assert teacher_session.status_code == 403
        assert _counts() == before

        before = _counts()
        cross_class = client.post(
            "/api/learning/sessions",
            headers=_headers(student_token),
            json={
                "client_session_id": "s0-student-cross-class",
                "course_id": scope["course_b"],
                "class_id": scope["class_b"],
            },
        )
        assert cross_class.status_code == 403
        assert _counts() == before

        before = _counts()
        mismatched_admin_context = client.post(
            "/api/learning/sessions",
            headers=_headers(admin_token),
            json={
                "client_session_id": "s0-admin-mismatch",
                "course_id": scope["course_a"],
                "class_id": scope["class_b"],
            },
        )
        assert mismatched_admin_context.status_code == 403
        assert _counts() == before

        admin_session = client.post(
            "/api/learning/sessions",
            headers=_headers(admin_token),
            json={
                "client_session_id": "s0-admin-ok",
                "course_id": scope["course_b"],
                "class_id": scope["class_b"],
            },
        )
        admin_event = client.post(
            "/api/learning/events",
            headers=_headers(admin_token),
            json={
                "client_session_id": "s0-admin-ok",
                "event_type": "PAGE_VIEW",
                "page_name": "teacher_dashboard",
            },
        )
        assert admin_session.status_code == 200
        assert admin_session.json()["course_id"] == scope["course_b"]
        assert admin_session.json()["class_id"] == scope["class_b"]
        assert admin_event.status_code == 200
        assert admin_event.json()["course_id"] == scope["course_b"]
        assert admin_event.json()["class_id"] == scope["class_b"]
    finally:
        _restore(client)


def test_s0_event_context_must_match_existing_session_context(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        scope = _seed_two_course_scope()
        admin_token, _ = _login(client, "admin_scope")
        headers = _headers(admin_token)

        created = client.post(
            "/api/learning/sessions",
            headers=headers,
            json={
                "client_session_id": "s0-admin-session-boundary",
                "course_id": scope["course_a"],
                "class_id": scope["class_a"],
            },
        )
        before = _counts()
        conflicting_event = client.post(
            "/api/learning/events",
            headers=headers,
            json={
                "client_session_id": "s0-admin-session-boundary",
                "event_type": "PAGE_VIEW",
                "page_name": "wrong-class",
                "course_id": scope["course_b"],
                "class_id": scope["class_b"],
            },
        )
        assert created.status_code == 200
        assert conflicting_event.status_code == 409
        assert _counts() == before

        valid_event = client.post(
            "/api/learning/events",
            headers=headers,
            json={
                "client_session_id": "s0-admin-session-boundary",
                "event_type": "PAGE_VIEW",
                "page_name": "right-class",
            },
        )

        assert valid_event.status_code == 200
        assert valid_event.json()["course_id"] == scope["course_a"]
        assert valid_event.json()["class_id"] == scope["class_a"]
    finally:
        _restore(client)


def test_s0_session_and_login_are_atomic_when_login_write_fails(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, raise_server_exceptions=False)
    try:
        _seed_two_course_scope()
        student_token, _ = _login(client, "student_scope_a")
        before = _counts()
        with main.learning_logger._connect() as conn:
            conn.execute(
                """
                CREATE TRIGGER fail_s0_login_insert
                BEFORE INSERT ON learning_events
                WHEN NEW.event_type = 'LOGIN'
                BEGIN
                    SELECT RAISE(ABORT, 's0 forced login failure');
                END
                """
            )

        failed = client.post(
            "/api/learning/sessions",
            headers=_headers(student_token),
            json={"client_session_id": "s0-atomic-login-failure"},
        )

        assert failed.status_code >= 500
        assert _counts() == before
        with main.learning_logger._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM learning_sessions WHERE client_session_id = ?",
                ("s0-atomic-login-failure",),
            ).fetchone()
        assert row is None
    finally:
        _restore(client)
