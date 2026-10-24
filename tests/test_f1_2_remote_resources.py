from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from backend import main
from backend.connectors.registry import ConnectorRegistry, FakeConnector
from backend.connectors.schemas import ExternalResource
from backend.innovation.service import InnovationService
from backend.storage.logger import LearningLogger
from mcp_server import tools as mcp_tools


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _login(client: TestClient, username: str = "student_demo") -> str:
    response = client.post("/api/auth/login", json={"username": username, "password": "demo123456"})
    assert response.status_code == 200
    payload = response.json()
    token = payload["access_token"]
    if payload.get("must_change_password"):
        changed = client.post(
            "/api/auth/change-password",
            headers=_headers(token),
            json={"current_password": "demo123456", "new_password": f"{username}-f1-2-123"},
        )
        assert changed.status_code == 200
        token = changed.json()["access_token"]
    return token


def test_anonymous_remote_resource_search_requires_login(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-f1-2-anonymous")
    original_logger = main.learning_logger
    original_registry = main.connector_registry
    main.learning_logger = LearningLogger(tmp_path / "anonymous.db")
    main.learning_logger.seed_demo_identity()
    main.connector_registry = ConnectorRegistry([FakeConnector()])
    client = TestClient(main.app)
    try:
        response = client.post(
            "/api/resources/search?connector_name=fake",
            json={"query": "5G NAS", "resource_type": "paper", "allow_remote": True},
        )

        assert response.status_code == 401
    finally:
        main.learning_logger = original_logger
        main.connector_registry = original_registry


def test_student_remote_resource_search_rejects_explicit_fake(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-f1-2-fake")
    original_logger = main.learning_logger
    original_registry = main.connector_registry
    main.learning_logger = LearningLogger(tmp_path / "fake.db")
    main.learning_logger.seed_demo_identity()
    main.connector_registry = ConnectorRegistry([FakeConnector()])
    client = TestClient(main.app)
    try:
        token = _login(client)
        response = client.post(
            "/api/resources/search?connector_name=fake",
            headers=_headers(token),
            json={"query": "5G NAS", "resource_type": "paper", "allow_remote": True},
        )

        assert response.status_code == 400
        assert response.json()["error_code"] == "FAKE_CONNECTOR_FORBIDDEN"
    finally:
        main.learning_logger = original_logger
        main.connector_registry = original_registry


class ArxivFixtureConnector(FakeConnector):
    name = "arxiv"
    supports_remote = True


def test_student_remote_search_uses_real_source_and_query_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-f1-2-student")
    original_logger = main.learning_logger
    original_registry = main.connector_registry
    resource = ExternalResource(
        external_id="arxiv-http-f1-2",
        resource_type="paper",
        source_name="arxiv",
        title="F1-2 HTTP paper",
        abstract="A deterministic HTTP resource fixture.",
        url="https://arxiv.org/abs/http-f1-2",
    )
    main.learning_logger = LearningLogger(tmp_path / "student-search.db")
    main.learning_logger.seed_demo_identity()
    main.connector_registry = ConnectorRegistry([ArxivFixtureConnector(resources=[resource])])
    client = TestClient(main.app)
    try:
        token = _login(client)
        response = client.post(
            "/api/resources/search?connector_name=arxiv",
            headers=_headers(token),
            json={"query": "F1-2", "resource_type": "paper", "allow_remote": True},
        )
        cached = client.post(
            "/api/resources/search?connector_name=arxiv",
            headers=_headers(token),
            json={"query": "F1-2", "resource_type": "paper", "allow_remote": True},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "SUCCESS"
        assert response.json()["resources"][0]["source_name"] == "arxiv"
        assert cached.status_code == 200
        assert cached.json()["cache_hit"] is True
        assert cached.json()["status"] == "SUCCESS"
        assert cached.json()["connector"] == "arxiv"
        assert main.learning_logger.connector_sync_runs(limit=10)[0]["status"] == "SUCCESS"
    finally:
        main.learning_logger = original_logger
        main.connector_registry = original_registry


def test_paper_search_auto_chooses_enabled_real_source(tmp_path, monkeypatch):
    resource = ExternalResource(
        external_id="arxiv-f1-2",
        resource_type="paper",
        source_name="arxiv",
        title="F1-2 arXiv paper",
        abstract="A deterministic paper fixture.",
        url="https://arxiv.org/abs/f1-2",
    )
    monkeypatch.setattr(
        mcp_tools,
        "get_default_registry",
        lambda: ConnectorRegistry([ArxivFixtureConnector(resources=[resource])]),
    )
    monkeypatch.setattr(mcp_tools, "LearningLogger", lambda: LearningLogger(tmp_path / "mcp.db"))

    result = mcp_tools._paper_search_real(
        mcp_tools.RealPaperSearchInput(query="F1-2", source="auto", top_k=3)
    )

    assert result["connector"] == "arxiv"
    assert result["status"] == "SUCCESS"
    assert result["resources"][0]["source_name"] == "arxiv"


class OnDemandResourceService:
    def search(self, query, connector_name=None, **kwargs):
        if query.resource_type == "paper":
            before_remote = kwargs.get("before_remote")
            if before_remote:
                before_remote("arxiv")
            return {
                "status": "SUCCESS",
                "cache_hit": False,
                "delivery_mode": "live",
                "resources": [
                    {
                        "external_id": "paper-on-demand",
                        "resource_type": "paper",
                        "source_name": "arxiv",
                        "provenance_verified": True,
                        "origin_kind": "remote",
                        "delivery_mode": "live",
                        "title": "5G NAS on-demand paper",
                        "abstract": "A paper fetched for a NAS authentication innovation session.",
                        "url": "https://arxiv.org/abs/on-demand",
                    }
                ],
            }
        if query.resource_type == "project":
            return {
                "status": "SUCCESS",
                "cache_hit": True,
                "delivery_mode": "fresh_cache",
                "resources": [
                    {
                        "external_id": "project-on-demand",
                        "resource_type": "project",
                        "source_name": "github",
                        "provenance_verified": True,
                        "origin_kind": "remote",
                        "delivery_mode": "fresh_cache",
                        "title": "5G NAS on-demand project",
                        "abstract": "A cached NAS security project fixture.",
                        "url": "https://github.com/example/on-demand",
                    }
                ],
            }
        return {"status": "FAILED", "resources": []}


def test_innovation_session_fetches_cases_from_on_demand_resource_service(tmp_path):
    logger = LearningLogger(tmp_path / "innovation-on-demand.db")
    logger.seed_demo_identity()
    user = logger.user_by_username("student_demo")

    service = InnovationService(
        logger,
        resource_service=OnDemandResourceService(),
    )

    started = service.start_session(
        user=user,
        topic="5G NAS",
        current_plan="Compare authentication baselines and replay detection.",
    )

    types = {case["type"] for case in started["cases"]}
    assert {"paper", "project"}.issubset(types)
    assert any(case["url"].startswith("https://arxiv.org/") for case in started["cases"])
    assert any(case["cache_status"] == "fresh_cache" for case in started["cases"])
