from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.auth.password import hash_password, verify_password
from backend.auth.service import AuthError, create_access_token, decode_access_token
from backend.storage.logger import LearningLogger


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-phase3-auth-rbac")
    original = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "auth.db")
    main.learning_logger.seed_demo_identity()
    client = TestClient(main.app)
    client._original_logger = original  # type: ignore[attr-defined]
    return client


def _restore(client: TestClient) -> None:
    main.learning_logger = client._original_logger  # type: ignore[attr-defined]


def _login(client: TestClient, username: str, password: str = "demo123456", *, activate: bool = True) -> str:
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    payload = response.json()
    token = payload["access_token"]
    if activate and payload.get("must_change_password"):
        new_password = f"{username}-changed-123"
        changed = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": password, "new_password": new_password},
        )
        assert changed.status_code == 200
        token = changed.json()["access_token"]
    return token


def test_phase3_migration_backfills_legacy_rows(tmp_path):
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
            VALUES ('legacy_user', 'legacy question', 'legacy answer', '5G NAS', 'course_qa', 0.8, 0)
            """
        )

    logger = LearningLogger(db_path)
    with logger._connect() as conn:
        question = conn.execute("SELECT * FROM questions").fetchone()
        course = conn.execute("SELECT * FROM courses WHERE id = ?", (question["course_id"],)).fetchone()
        klass = conn.execute("SELECT * FROM classes WHERE id = ?", (question["class_id"],)).fetchone()

    assert question["question"] == "legacy question"
    assert course["slug"] == "demo-course"
    assert klass["slug"] == "demo-class"
    assert logger.analytics()["total_questions"] == 1


def test_password_hash_uses_argon2_and_never_plaintext():
    password_hash = hash_password("demo123456")

    assert password_hash != "demo123456"
    assert password_hash.startswith("$argon2")
    assert verify_password("demo123456", password_hash) is True
    assert verify_password("wrong", password_hash) is False


def test_login_me_logout_and_role_courses(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        first_token = _login(client, "student_demo", activate=False)
        first_headers = {"Authorization": f"Bearer {first_token}"}
        blocked = client.get("/api/courses", headers=first_headers)
        assert blocked.status_code == 403
        assert blocked.json()["error_code"] == "PASSWORD_CHANGE_REQUIRED"

        changed = client.post(
            "/api/auth/change-password",
            headers=first_headers,
            json={"current_password": "demo123456", "new_password": "new-demo-pass-123"},
        )
        assert changed.status_code == 200
        token = changed.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        me = client.get("/api/auth/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["user"]["role"] == "student"

        assert client.get("/api/courses", headers=headers).status_code == 200
        classes = client.get("/api/classes", headers=headers).json()
        assert len(classes) == 1

        logout = client.post("/api/auth/logout", headers=headers)
        assert logout.status_code == 200
        assert logout.json()["logged_out"] is True
        assert client.get("/api/auth/me", headers=headers).status_code == 401
    finally:
        _restore(client)


def test_demo_user_can_change_initial_password_and_seed_does_not_reset_it(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token = _login(client, "student_demo", activate=False)
        headers = {"Authorization": f"Bearer {token}"}

        changed = client.post(
            "/api/auth/change-password",
            headers=headers,
            json={"current_password": "demo123456", "new_password": "new-demo-pass-123"},
        )
        old_login = client.post(
            "/api/auth/login",
            json={"username": "student_demo", "password": "demo123456"},
        )
        new_login = client.post(
            "/api/auth/login",
            json={"username": "student_demo", "password": "new-demo-pass-123"},
        )
        main.learning_logger.seed_demo_identity()
        old_after_seed = client.post(
            "/api/auth/login",
            json={"username": "student_demo", "password": "demo123456"},
        )
        new_after_seed = client.post(
            "/api/auth/login",
            json={"username": "student_demo", "password": "new-demo-pass-123"},
        )

        assert changed.status_code == 200
        assert changed.json()["changed"] is True
        assert changed.json()["must_change_password"] is False
        assert old_login.status_code == 401
        assert new_login.status_code == 200
        assert new_login.json()["must_change_password"] is False
        assert old_after_seed.status_code == 401
        assert new_after_seed.status_code == 200
    finally:
        _restore(client)


def test_change_password_rejects_wrong_current_password(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token = _login(client, "student_demo")
        response = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": "wrong-password", "new_password": "new-demo-pass-123"},
        )

        assert response.status_code == 401
        assert response.json()["error_code"] == "INVALID_CREDENTIALS"
    finally:
        _restore(client)


def test_demo_seed_can_be_disabled_for_login(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-demo-disabled")
    monkeypatch.setenv("ENABLE_DEMO_ACCOUNTS", "false")
    original = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "auth-disabled.db")
    client = TestClient(main.app)
    try:
        response = client.post(
            "/api/auth/login",
            json={"username": "student_demo", "password": "demo123456"},
        )

        assert response.status_code == 401
        assert main.learning_logger.user_by_username("student_demo") is None
    finally:
        main.learning_logger = original


def test_token_expiry_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-phase3-auth-rbac")
    logger = LearningLogger(tmp_path / "auth.db")
    logger.seed_demo_identity()
    user = logger.user_by_username("student_demo")
    token, _ = create_access_token(user, expires_delta=timedelta(seconds=-1))

    try:
        decode_access_token(token)
    except AuthError as exc:
        assert exc.error_code == "TOKEN_EXPIRED"
    else:
        raise AssertionError("expired token should be rejected")


def test_student_token_overrides_request_body_user_id(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token = _login(client, "student_demo")
        response = client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": "5G NAS 认证流程有哪些关键风险点？", "user_id": "teacher_demo"},
        )

        assert response.status_code == 200
        question_id = response.json()["question_id"]
        with main.learning_logger._connect() as conn:
            row = conn.execute("SELECT user_id FROM questions WHERE id = ?", (question_id,)).fetchone()
        assert row["user_id"] == "2"
    finally:
        _restore(client)


def test_student_cannot_access_other_student_run_or_teacher_endpoint(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        student_1 = _login(client, "student_demo")
        student_2 = _login(client, "student_demo2")
        created = client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {student_1}"},
            json={"question": "5G NAS 认证流程有哪些关键风险点？"},
        )
        run_id = created.json()["run_id"]

        other = client.get(
            f"/api/agent/runs/{run_id}",
            headers={"Authorization": f"Bearer {student_2}"},
        )
        teacher_endpoint = client.get(
            "/api/teacher/analytics",
            headers={"Authorization": f"Bearer {student_2}"},
        )

        assert other.status_code == 403
        assert teacher_endpoint.status_code == 403
    finally:
        _restore(client)


def test_stateful_agent_endpoints_require_login_and_owner_scope(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    class FakeOrchestrator:
        def __init__(self, *args, **kwargs):
            pass

        def continue_run(self, run_id: str, user_input: str, user_id: str, **_scope):
            return {
                "answer": "continued answer",
                "citations": [],
                "confidence": 0.8,
                "abstained": False,
                "module": "5G NAS",
                "intent": "course_qa",
                "tool": {"name": "course_retriever", "description": "local", "risk": "LOW", "enabled": True},
                "run_id": run_id,
                "run_status": "COMPLETED",
                "current_step": 2,
                "requires_user_input": False,
                "pending_question": None,
                "trace_id": "trace-owner",
                "provider": "mock",
                "model": "mock",
                "fallback_used": False,
                "tool_calls": [],
            }

    try:
        monkeypatch.setattr(main, "AgentOrchestrator", FakeOrchestrator)
        student_1 = _login(client, "student_demo")
        student_2 = _login(client, "student_demo2")
        run_id = "owner-run"
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()
        main.learning_logger.create_agent_run(
            run_id=run_id,
            user_id="2",
            goal="needs more input",
            intent="course_qa",
            status="REQUIRES_INPUT",
            max_steps=3,
            max_tool_calls=3,
            state={
                "current_step": 1,
                "tool_call_count": 0,
                "trace_id": "trace-owner",
                "pending_question": "补充实验条件",
                "course_id": course_id,
                "class_id": class_id,
            },
            course_id=course_id,
            class_id=class_id,
        )

        anonymous_get = client.get(f"/api/agent/runs/{run_id}")
        anonymous_continue = client.post(
            f"/api/agent/runs/{run_id}/continue",
            json={"user_input": "local lab"},
        )
        cross_continue = client.post(
            f"/api/agent/runs/{run_id}/continue",
            headers={"Authorization": f"Bearer {student_2}"},
            json={"user_input": "other student"},
        )
        owner_continue = client.post(
            f"/api/agent/runs/{run_id}/continue",
            headers={"Authorization": f"Bearer {student_1}"},
            json={"user_input": "local NAS lab", "conversation_id": "owner-conv"},
        )

        assert anonymous_get.status_code == 401
        assert anonymous_continue.status_code == 401
        assert cross_continue.status_code == 403
        assert owner_continue.status_code == 200
        assert owner_continue.json()["run_id"] == run_id
        assert owner_continue.json()["conversation_id"] == "owner-conv"
    finally:
        _restore(client)


@pytest.mark.parametrize(
    ("state_json", "context_status"),
    [("{}", "missing"), ("{malformed", "invalid"), ("[]", "invalid")],
)
def test_run_context_corruption_is_fail_closed_and_admin_get_is_safe(
    tmp_path,
    monkeypatch,
    state_json,
    context_status,
):
    client = _client(tmp_path, monkeypatch)
    try:
        main.learning_logger.upsert_user(
            username="admin_demo",
            display_name="Admin Demo",
            password_hash=hash_password("demo123456"),
            role="admin",
            must_change_password=False,
        )
        owner_token = _login(client, "student_demo")
        teacher_token = _login(client, "teacher_demo")
        admin_token = _login(client, "admin_demo")
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()
        run_id = f"corrupt-{context_status}-{len(state_json)}"
        main.learning_logger.create_agent_run(
            run_id=run_id,
            user_id="2",
            goal="protected run",
            intent="course_qa",
            status="REQUIRES_INPUT",
            max_steps=3,
            max_tool_calls=2,
            state={
                "course_id": course_id,
                "class_id": class_id,
                "current_step": 0,
                "tool_call_count": 0,
            },
            course_id=course_id,
            class_id=class_id,
        )
        with main.learning_logger._connect() as conn:
            conn.execute("UPDATE agent_runs SET state_json = ? WHERE run_id = ?", (state_json, run_id))
            before = {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("agent_steps", "model_calls", "questions", "learning_events")
            }

        owner_get = client.get(
            f"/api/agent/runs/{run_id}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        teacher_get = client.get(
            f"/api/agent/runs/{run_id}",
            headers={"Authorization": f"Bearer {teacher_token}"},
        )
        admin_get = client.get(
            f"/api/agent/runs/{run_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        continue_responses = [
            client.post(
                f"/api/agent/runs/{run_id}/continue",
                headers={"Authorization": f"Bearer {token}"},
                json={"user_input": "must not execute"},
            )
            for token in (owner_token, teacher_token, admin_token)
        ]

        assert owner_get.status_code == 403
        assert owner_get.json()["error_code"] == "RUN_CONTEXT_REQUIRED"
        assert teacher_get.status_code == 403
        assert teacher_get.json()["error_code"] == "RUN_CONTEXT_REQUIRED"
        assert admin_get.status_code == 200
        assert admin_get.json()["context_status"] == context_status
        assert admin_get.json()["trace_id"] is None
        assert "state_json" not in admin_get.text
        assert all(item.status_code == 403 for item in continue_responses)
        assert all(item.json()["error_code"] == "RUN_CONTEXT_REQUIRED" for item in continue_responses)
        with main.learning_logger._connect() as conn:
            after = {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in before
            }
        assert after == before
    finally:
        _restore(client)


def test_chat_and_feedback_require_login_and_feedback_question_owner(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        anonymous_chat = client.post("/api/chat", json={"question": "5G NAS?"})
        anonymous_feedback = client.post("/api/feedback", json={"question_id": 1, "rating": 5})

        student_1 = _login(client, "student_demo")
        student_2 = _login(client, "student_demo2")
        created = client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {student_1}"},
            json={"question": "5G NAS 认证流程有哪些关键风险点？", "user_id": "student_demo2"},
        )
        question_id = created.json()["question_id"]
        cross_feedback = client.post(
            "/api/feedback",
            headers={"Authorization": f"Bearer {student_2}"},
            json={"question_id": question_id, "rating": 1, "comment": "spoofed"},
        )
        owner_feedback = client.post(
            "/api/feedback",
            headers={"Authorization": f"Bearer {student_1}"},
            json={"question_id": question_id, "rating": 5, "helpful": True},
        )

        assert anonymous_chat.status_code == 401
        assert anonymous_feedback.status_code == 401
        assert created.status_code == 200
        assert cross_feedback.status_code == 403
        assert owner_feedback.status_code == 200
        with main.learning_logger._connect() as conn:
            row = conn.execute("SELECT user_id FROM questions WHERE id = ?", (question_id,)).fetchone()
        assert row["user_id"] == "2"
    finally:
        _restore(client)


def test_teacher_sees_only_owned_classes_and_can_read_class_student_run(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        teacher = _login(client, "teacher_demo")
        student = _login(client, "student_demo")
        created = client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {student}"},
            json={"question": "5G NAS 认证流程有哪些关键风险点？"},
        )

        classes = client.get("/api/classes", headers={"Authorization": f"Bearer {teacher}"}).json()
        run = client.get(
            f"/api/agent/runs/{created.json()['run_id']}",
            headers={"Authorization": f"Bearer {teacher}"},
        )

        assert len(classes) == 1
        assert classes[0]["teacher_id"] == 1
        assert run.status_code == 200
    finally:
        _restore(client)


def test_admin_can_access_all_courses_and_teacher_api(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        password_hash = hash_password("admin-pass")
        main.learning_logger.upsert_user(
            username="admin_demo",
            display_name="Admin Demo",
            password_hash=password_hash,
            role="admin",
            must_change_password=False,
        )
        response = client.post(
            "/api/auth/login",
            json={"username": "admin_demo", "password": "admin-pass"},
        )
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        assert client.get("/api/courses", headers=headers).status_code == 200
        assert client.get("/api/teacher/analytics", headers=headers).status_code == 200
    finally:
        _restore(client)


def test_demo_seed_is_idempotent(tmp_path):
    logger = LearningLogger(tmp_path / "auth.db")
    first = logger.seed_demo_identity()
    second = logger.seed_demo_identity()

    with logger._connect() as conn:
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        course_count = conn.execute("SELECT COUNT(*) FROM courses").fetchone()[0]
        class_count = conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0]
        member_count = conn.execute("SELECT COUNT(*) FROM class_members").fetchone()[0]

    assert first == second
    assert user_count == 3
    assert course_count == 1
    assert class_count == 1
    assert member_count == 3
