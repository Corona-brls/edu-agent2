from fastapi.testclient import TestClient
from pathlib import Path

from backend import main
from backend.auth.seed import seed_demo_data
from backend.connectors.schemas import ResourceQuery
from backend.connectors.standard_source import StandardSourceConnector
from backend.storage.logger import LearningLogger


def _login(client: TestClient, username: str = "teacher_demo") -> dict[str, str]:
    response = client.post("/api/auth/login", json={"username": username, "password": "demo123456"})
    assert response.status_code == 200
    payload = response.json()
    token = payload["access_token"]
    if payload.get("must_change_password"):
        changed = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": "demo123456", "new_password": f"{username}-changed-123"},
        )
        assert changed.status_code == 200
        token = changed.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_phase3_demo_seed_prepares_acceptance_materials_without_new_tables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-demo-seed")
    original = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "demo.db")
    try:
        result = seed_demo_data(main.learning_logger)
        client = TestClient(main.app)
        teacher_headers = _login(client, "teacher_demo")
        student_headers = _login(client, "student_demo")

        portal = client.get("/api/resources/portal", headers=student_headers, params={"keyword": "NAS"}).json()
        summary = client.get("/api/teacher/phase3/summary", headers=teacher_headers).json()
        faq = client.get("/api/teacher/question-clusters/faq-export", headers=teacher_headers).json()
        faq_csv = client.get("/api/teacher/phase3/faq-export.csv", headers=teacher_headers).content.decode("utf-8-sig")
        stuck = client.get("/api/teacher/stuck-analysis", headers=teacher_headers).json()

        assert result["backup_path"]
        assert Path(result["backup_path"]).exists()
        assert result["materials"]["documents"]
        assert result["materials"]["external_resources"]
        assert portal["items"]
        assert {"course_document", "standard", "paper", "vulnerability", "project"}.intersection(
            {item["type"] for item in portal["items"]}
        )
        assert summary["access"]["visit_count"] >= 1
        assert summary["resources"]["search_count"] >= 1
        assert stuck["items"]
        assert faq["items"]
        assert faq["items"][0]["approved"] is True
        assert faq["items"][0]["citations"]
        assert "Demo approved 5G NAS FAQ" in faq_csv
    finally:
        main.learning_logger = original


def test_phase3_demo_seed_materials_are_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = LearningLogger(tmp_path / "demo-idempotent.db")

    first = seed_demo_data(logger)
    second = seed_demo_data(logger)

    with logger._connect() as conn:
        question_count = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        cluster_count = conn.execute("SELECT COUNT(*) FROM question_clusters").fetchone()[0]
        event_count = conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0]
        document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    assert first["materials"]["question_ids"] == second["materials"]["question_ids"]
    assert question_count == 3
    assert cluster_count == 1
    assert event_count == 7
    assert document_count == 2
    assert Path(first["backup_path"]).exists()
    assert Path(second["backup_path"]).exists()
    assert first["backup_path"] != second["backup_path"]


def test_standard_source_demo_files_do_not_create_generic_standard_false_positive(tmp_path):
    (tmp_path / "generic_standard_notes.pdf").write_bytes(b"generic notes")
    connector = StandardSourceConnector(knowledge_dir=tmp_path)

    generic = __import__("asyncio").run(connector.search(ResourceQuery(query="3GPP TS 24.501")))
    (tmp_path / "3GPP_TS_24.501_teaching_notes.md").write_text("teaching notes", encoding="utf-8")
    matched = __import__("asyncio").run(connector.search(ResourceQuery(query="3GPP TS 24.501")))

    assert generic[0].raw_metadata["local_uploaded"] is False
    assert matched[0].raw_metadata["local_uploaded"] is False
    assert matched[0].raw_metadata["local_match"] is None
    assert "Local full text uploaded: True" not in matched[0].abstract
