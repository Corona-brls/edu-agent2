from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend import main
from backend.connectors.schemas import ExternalResource
from backend.rag.document_parser import Document, Page
from backend.resources.normalizer import content_hash
from backend.storage.logger import LearningLogger, sha256_bytes


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-resource-portal")
    original_logger = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "resources.db")
    main.learning_logger.seed_demo_identity()
    _seed_resources(main.learning_logger, tmp_path)
    client = TestClient(main.app)
    client._original_logger = original_logger  # type: ignore[attr-defined]
    return client


def _restore(client: TestClient) -> None:
    main.learning_logger = client._original_logger  # type: ignore[attr-defined]


def _login(client: TestClient, username: str) -> tuple[str, dict]:
    response = client.post("/api/auth/login", json={"username": username, "password": "demo123456"})
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


def _commit_verified_resources(
    logger: LearningLogger,
    resources: list[ExternalResource],
    *,
    ttl_hours: int = 24,
) -> list[dict]:
    saved: list[dict] = []
    by_source: dict[str, list[ExternalResource]] = {}
    for resource in resources:
        by_source.setdefault(resource.source_name, []).append(resource)
    for source_name, source_resources in by_source.items():
        query = {"query": "", "page": 1, "page_size": 50, "allow_remote": True}
        run_id = logger.create_connector_sync_run(
            connector_name=source_name,
            query=query,
            status="RUNNING",
            cache_hit=False,
        )
        saved.extend(
            logger.commit_verified_sync(
                run_id=run_id,
                connector_name=source_name,
                origin_kind="local_catalog" if source_name == "standard_source" else "remote",
                query=query,
                resources=[
                    {"resource": item.model_dump(mode="json"), "content_hash": content_hash(item)}
                    for item in source_resources
                ],
                ttl_hours=ttl_hours,
                latency_ms=0,
            )
        )
    return saved


def _seed_resources(logger: LearningLogger, tmp_path: Path) -> None:
    document = Document(
        source_path=tmp_path / "nas-guide.md",
        filename="nas-guide.md",
        file_type=".md",
        title="5G NAS course document",
        text="5G NAS authentication course material.",
        pages=[Page(text="5G NAS authentication course material.", page=1, section="intro", source_type="markdown")],
    )
    saved = logger.save_document(
        document=document,
        content_hash=sha256_bytes(document.text.encode("utf-8")),
        chunks=["5G NAS authentication course material."],
        byte_size=len(document.text),
        index_status="INDEXED",
        course_id=logger.default_course_id(),
    )
    logger.set_document_publication(int(saved["id"]), published=True, actor_id=1)
    resources = [
        ExternalResource(
            external_id="paper-1",
            resource_type="paper",
            source_name="semantic_scholar",
            title="5G NAS Security Paper",
            authors=["A. Researcher"],
            abstract="Paper about 5G NAS authentication security.",
            url="https://example.test/paper-1",
            published_at="2024",
            tags=["5G NAS", "paper"],
            module="5G NAS",
        ),
        ExternalResource(
            external_id="standard-24501",
            resource_type="standard",
            source_name="standard_source",
            title="3GPP TS 24.501 metadata",
            abstract="Standard metadata only.",
            url="https://example.test/standard",
            published_at="2023",
            tags=["3GPP", "NAS"],
            module="5G NAS",
        ),
        ExternalResource(
            external_id="standard-no-date",
            resource_type="standard",
            source_name="standard_source",
            title="3GPP TS 33.501 official metadata",
            abstract="Official metadata link only; local full text has not been uploaded.",
            url="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3169",
            published_at=None,
            tags=["3GPP", "security"],
            module="5G NAS",
        ),
        ExternalResource(
            external_id="CVE-2021-44228",
            resource_type="vulnerability",
            source_name="nvd",
            title="CVE-2021-44228",
            abstract="NVD vulnerability metadata.",
            url="https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
            published_at="2021",
            tags=["CVSS 10 CRITICAL", "CWE-20"],
            module="漏洞样例",
        ),
        ExternalResource(
            external_id="repo-1",
            resource_type="project",
            source_name="github",
            title="security/course-agent-demo",
            abstract="Open source project metadata.",
            url="https://github.com/example/course-agent-demo",
            published_at="2025",
            tags=["Python", "security"],
            module="开源项目",
            raw_metadata={
                "full_name": "example/course-agent-demo",
                "private": False,
                "visibility": "public",
            },
        ),
    ]
    duplicate = resources[0].model_copy(update={"abstract": "Updated duplicate paper metadata."})
    _commit_verified_resources(logger, [*resources[1:], duplicate])
    expired = ExternalResource(
        external_id="expired-paper",
        resource_type="paper",
        source_name="arxiv",
        title="Expired cached blockchain paper",
        abstract="Expired cache should still be visible in the resource portal.",
        url="https://arxiv.org/abs/0000.00000",
        published_at="2022",
        tags=["blockchain"],
        module="区块链隐蔽通信",
    )
    _commit_verified_resources(logger, [expired], ttl_hours=0)


def test_student_can_browse_six_resource_types_with_unified_shape(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, _ = _login(client, "student_demo")
        response = client.get("/api/resources/portal", headers=_headers(token), params={"page_size": 50})

        assert response.status_code == 200
        payload = response.json()
        types = {item["type"] for item in payload["items"]}
        assert {"course_document", "paper", "standard", "vulnerability", "project", "competition"}.issubset(types)
        course_document = next(item for item in response.json()["items"] if item["type"] == "course_document")
        assert course_document["resource_id"].startswith("document:")
        assert course_document["metadata"]["chunk_id"] > 0
        assert course_document["metadata"]["index_status_label"] == "已索引"
        assert course_document["metadata"]["publication_status_label"] == "已发布"
        assert course_document["delivery_mode"] == "course_published"
        assert payload["can_manage"] is False
        assert payload["cache_mode"] == "cache_first"
        for item in payload["items"]:
            assert {
                "title",
                "summary",
                "raw_summary",
                "type_label",
                "topic_category",
                "date_label",
                "tags",
                "source",
                "url",
                "cache_status",
                "updated_at",
            }.issubset(item)
    finally:
        _restore(client)


def test_standard_without_year_is_visible_by_default_and_marked_as_metadata(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, _ = _login(client, "student_demo")
        response = client.get(
            "/api/resources/portal",
            headers=_headers(token),
            params={"resource_type": "standard", "page_size": 50},
        )

        assert response.status_code == 200
        items = response.json()["items"]
        standard = next(item for item in items if item["external_id"] == "standard-no-date")
        assert standard["date_label"] == "日期未知"
        assert standard["topic_category"] == "5G NAS"
        assert standard["type_label"] == "标准元数据"
        assert "标准元数据" in standard["summary"]
        assert "全文" in standard["summary"]
        assert standard["raw_summary"] == (
            "Standard metadata only. Verified local full text: unavailable."
        )
        assert standard["metadata"]["metadata_only"] is True
        assert standard["metadata"]["local_full_text_available"] is False
        assert "官方元数据" in standard["metadata"]["source_note"]
    finally:
        _restore(client)


def test_resource_items_have_source_bound_summary_and_topic_classification(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, _ = _login(client, "student_demo")
        headers = _headers(token)
        expected_cases = {
            "paper": ("5G NAS Security Paper", "5G NAS"),
            "standard": ("3GPP TS 24.501", "5G NAS"),
            "vulnerability": ("CVE-2021-44228", "漏洞分析"),
        }
        for resource_type, (keyword, expected_topic) in expected_cases.items():
            response = client.get(
                "/api/resources/portal",
                headers=headers,
                params={"resource_type": resource_type, "keyword": keyword, "page_size": 50},
            )
            assert response.status_code == 200
            item = response.json()["items"][0]
            assert item["summary"]
            assert item["raw_summary"] is not None
            assert item["type_label"]
            assert item["topic_category"] == expected_topic
            assert item["source"] in item["summary"]
    finally:
        _restore(client)


def test_filters_pagination_and_dedupe(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, _ = _login(client, "student_demo")
        headers = _headers(token)
        papers = client.get(
            "/api/resources/portal",
            headers=headers,
            params={"resource_type": "paper", "keyword": "paper", "page_size": 50},
        ).json()
        ids = [item["external_id"] for item in papers["items"]]

        assert ids.count("paper-1") == 1
        assert all(item["type"] == "paper" for item in papers["items"])

        page_1 = client.get("/api/resources/portal", headers=headers, params={"page": 1, "page_size": 2}).json()
        page_2 = client.get("/api/resources/portal", headers=headers, params={"page": 2, "page_size": 2}).json()

        assert page_1["total"] >= 6
        assert len(page_1["items"]) == 2
        assert page_1["items"][0]["resource_id"] != page_2["items"][0]["resource_id"]
    finally:
        _restore(client)


def test_resource_portal_reads_external_cache_beyond_first_page(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, _ = _login(client, "student_demo")
        bulk_resources = []
        for index in range(60):
            bulk_resources.append(ExternalResource(
                external_id=f"bulk-{index}",
                resource_type="paper",
                source_name="semantic_scholar",
                title=f"Bulk cached paper {index}",
                abstract="Large cache pagination coverage.",
                url=f"https://example.test/bulk-{index}",
                published_at="2026",
                tags=["bulk"],
                module="bulk",
            ))
        _commit_verified_resources(main.learning_logger, bulk_resources)

        response = client.get(
            "/api/resources/portal",
            headers=_headers(token),
            params={"keyword": "Bulk cached paper 59", "resource_type": "paper", "page_size": 5},
        )

        assert response.status_code == 200
        assert response.json()["total"] == 1
        assert response.json()["items"][0]["external_id"] == "bulk-59"
    finally:
        _restore(client)


def test_cached_results_are_visible_when_remote_is_unavailable(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, _ = _login(client, "student_demo")
        response = client.get(
            "/api/resources/portal",
            headers=_headers(token),
            params={"keyword": "Expired cached", "resource_type": "paper"},
        )

        assert response.status_code == 200
        item = response.json()["items"][0]
        assert item["source"] == "arxiv"
        assert item["cache_status"] == "expired"
        assert item["fetched_at"]
    finally:
        _restore(client)


def test_resource_search_and_open_are_recorded_in_learning_timeline(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token, user = _login(client, "student_demo")
        headers = _headers(token)
        sid = "resource-portal-session"
        assert client.post("/api/learning/sessions", headers=headers, json={"client_session_id": sid}).status_code == 200

        search = client.get(
            "/api/resources/portal",
            headers=headers,
            params={"keyword": "NAS", "client_session_id": sid},
        )
        resource_id = search.json()["items"][0]["resource_id"]
        opened = client.post(
            "/api/resources/open",
            headers=headers,
            json={"resource_id": resource_id, "client_session_id": sid},
        )
        timeline = client.get(f"/api/students/{user['id']}/timeline", headers=headers).json()["events"]

        assert opened.status_code == 200
        event_types = [item["event_type"] for item in timeline]
        assert "RESOURCE_SEARCH" in event_types
        assert "RESOURCE_OPEN" in event_types
        assert any(item.get("resource_id") == resource_id for item in timeline)
    finally:
        _restore(client)


def test_permissions_student_read_only_teacher_can_manage(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        student_token, _ = _login(client, "student_demo")
        teacher_token, _ = _login(client, "teacher_demo")

        student_payload = client.get("/api/resources/portal", headers=_headers(student_token)).json()
        teacher_payload = client.get("/api/resources/portal", headers=_headers(teacher_token)).json()
        denied = client.post(
            "/api/connectors/fake/sync",
            headers=_headers(student_token),
            json={"query": "5G NAS", "resource_type": "paper"},
        )

        assert student_payload["can_manage"] is False
        assert teacher_payload["can_manage"] is True
        assert denied.status_code == 403
    finally:
        _restore(client)
