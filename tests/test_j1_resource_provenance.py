from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from backend.agent.observation import tool_result_to_observation
from backend.connectors.registry import ConnectorRegistry, FakeConnector
from backend.connectors.schemas import ExternalResource, ResourceQuery
from backend.innovation.service import InnovationService
from backend.resources.normalizer import content_hash
from backend.resources.portal import ResourcePortalQuery, ResourcePortalService
from backend.resources.service import ExternalResourceService
from backend.storage.logger import LearningLogger, _external_resource_row
from mcp_server import tools as mcp_tools
from mcp_server.schemas import ToolResult


class ArxivFixtureConnector(FakeConnector):
    name = "arxiv"
    supports_remote = True


class _NoopVectorStore:
    def upsert_external_resource(self, vector_id: str, text: str, record: dict) -> None:
        return None

    def delete_external_resource(self, vector_id: str) -> None:
        return None


class _EmptyRetriever:
    def retrieve(self, *args, **kwargs):
        return SimpleNamespace(results=[])


class _EmptyGateway:
    def call_tool(self, *args, **kwargs):
        return SimpleNamespace(
            result=SimpleNamespace(success=False, data={}, error_code="DISABLED"),
            fallback_used=False,
        )


def _resource(
    external_id: str,
    source_name: str,
    title: str,
    *,
    url: str,
    authors: list[str] | None = None,
    abstract: str = "unmapped-j1 mapped-j1 provenance isolation fixture",
) -> ExternalResource:
    return ExternalResource(
        external_id=external_id,
        resource_type="paper",
        source_name=source_name,
        title=title,
        authors=list(authors or []),
        abstract=abstract,
        url=url,
        published_at="2026",
        tags=["j1", "provenance"],
        module="5G NAS",
    )


def test_verified_paper_authors_roundtrip_and_current_hash_coverage(tmp_path):
    logger = LearningLogger(tmp_path / "author-roundtrip.db")
    paper = _resource(
        "2610.12345",
        "arxiv",
        "5G Core Attack Detection",
        url="https://arxiv.org/abs/2610.12345",
        authors=["Alice Example", "Bob Example"],
        abstract="A controlled paper abstract about 5G core attack detection.",
    )
    service = ExternalResourceService(
        storage=logger,
        registry=ConnectorRegistry([ArxivFixtureConnector(resources=[paper])]),
        vector_store=_NoopVectorStore(),
    )

    result = asyncio.run(
        service.search(
            ResourceQuery(query="5G core attack detection", resource_type="paper", page_size=3),
            connector_name="arxiv",
        )
    )

    assert result["status"] == "SUCCESS"
    assert len(result["resources"]) == 1
    stored = result["resources"][0]
    assert stored["authors"] == ["Alice Example", "Bob Example"]
    assert stored["authors_hash_covered"] is True
    changed = paper.model_copy(update={"authors": ["Alice Example", "Carol Example"]})
    assert content_hash(changed) != content_hash(paper)


def test_legacy_or_hash_mismatched_authors_never_gain_citeable_coverage(tmp_path):
    logger = LearningLogger(tmp_path / "author-compatibility.db")
    paper = _resource(
        "2610.54321",
        "arxiv",
        "Verified Paper",
        url="https://arxiv.org/abs/2610.54321",
        authors=["Alice Example", "Bob Example"],
    )
    service = ExternalResourceService(
        storage=logger,
        registry=ConnectorRegistry([ArxivFixtureConnector(resources=[paper])]),
        vector_store=_NoopVectorStore(),
    )
    query = ResourceQuery(query="verified", resource_type="paper", page_size=3)
    assert asyncio.run(service.search(query, connector_name="arxiv"))["status"] == "SUCCESS"

    with logger._connect() as conn:
        conn.execute(
            "UPDATE external_resources SET authors_json = ? WHERE external_id = ?",
            (json.dumps({"authors": ["Alice Example", "Bob Example"]}), paper.external_id),
        )
        raw = dict(conn.execute("SELECT * FROM external_resources WHERE external_id = ?", (paper.external_id,)).fetchone())
    raw["provenance_verified"] = 1
    legacy = _external_resource_row(raw)
    assert legacy["authors"] == ["Alice Example", "Bob Example"]
    assert legacy["authors_hash_covered"] is False

    with logger._connect() as conn:
        conn.execute(
            "UPDATE external_resources SET authors_json = ? WHERE external_id = ?",
            (json.dumps(["Mallory Example"]), paper.external_id),
        )
        raw = dict(conn.execute("SELECT * FROM external_resources WHERE external_id = ?", (paper.external_id,)).fetchone())
    raw["provenance_verified"] = 1
    mismatched = _external_resource_row(raw)
    assert mismatched["authors"] == ["Mallory Example"]
    assert mismatched["authors_hash_covered"] is False


def _seed_mixed_sources(logger: LearningLogger) -> tuple[ExternalResourceService, dict[str, dict]]:
    trusted = _resource(
        "trusted-arxiv",
        "arxiv",
        "bootstrap-j1 trusted paper",
        url="https://arxiv.org/abs/2609.00001",
    )
    service = ExternalResourceService(
        storage=logger,
        registry=ConnectorRegistry([ArxivFixtureConnector(resources=[trusted])]),
        vector_store=_NoopVectorStore(),
    )
    first = asyncio.run(
        service.search(
            ResourceQuery(query="bootstrap-j1", resource_type="paper", page_size=10, allow_remote=True),
            connector_name="arxiv",
        )
    )
    assert first["status"] == "SUCCESS"

    historical = {
        "fake": _resource(
            "historical-fake",
            "fake",
            "unmapped-j1 fake paper",
            url="https://example.test/fake",
        ),
        "unknown": _resource(
            "historical-unknown",
            "mystery_source",
            "unmapped-j1 unknown paper",
            url="https://example.test/unknown",
        ),
        "masquerade": _resource(
            "historical-arxiv-no-proof",
            "arxiv",
            "unmapped-j1 masquerading arxiv paper",
            url="https://arxiv.org/abs/2609.99999",
        ),
    }
    records = {"trusted": logger.external_resource_by_key("arxiv", trusted.external_id)}
    for key, resource in historical.items():
        records[key] = logger.upsert_external_resource(
            resource=resource.model_dump(mode="json"),
            ttl_hours=24,
            content_hash=content_hash(resource),
        )
    return service, records


def test_migration_16_is_idempotent_and_does_not_trust_history(tmp_path):
    db_path = tmp_path / "migration.db"
    logger = LearningLogger(db_path)
    historical = _resource(
        "legacy-arxiv",
        "arxiv",
        "legacy row without proof",
        url="https://arxiv.org/abs/legacy",
    )
    logger.upsert_external_resource(
        resource=historical.model_dump(mode="json"),
        ttl_hours=24,
        content_hash=content_hash(historical),
    )

    LearningLogger(db_path)
    with logger._connect() as conn:
        versions = conn.execute(
            "SELECT COUNT(*) AS count FROM schema_migrations WHERE version = 16"
        ).fetchone()["count"]
        proofs = conn.execute("SELECT COUNT(*) AS count FROM resource_provenance").fetchone()["count"]

    assert versions == 1
    assert proofs == 0


def test_unverified_history_is_filtered_with_and_without_query_mapping(tmp_path):
    logger = LearningLogger(tmp_path / "mixed.db")
    service, records = _seed_mixed_sources(logger)

    unmapped = asyncio.run(
        service.search(ResourceQuery(query="unmapped-j1", resource_type="paper", allow_remote=False))
    )
    assert [item["external_id"] for item in unmapped["resources"]] == ["trusted-arxiv"]

    logger.store_resource_query_cache(
        ResourceQuery(query="mapped-j1", resource_type="paper").model_dump(mode="json"),
        source_name="arxiv",
        resource_ids=[int(records["masquerade"]["id"]), int(records["trusted"]["id"])],
        ttl_hours=24,
    )
    mapped = asyncio.run(
        service.search(
            ResourceQuery(query="mapped-j1", resource_type="paper", allow_remote=False),
            connector_name="arxiv",
        )
    )
    assert [item["external_id"] for item in mapped["resources"]] == ["trusted-arxiv"]
    assert mapped["origin_kind"] == "remote"
    assert mapped["delivery_mode"] == "fresh_cache"


@pytest.mark.parametrize(
    ("statement", "params"),
    [
        ("UPDATE resource_provenance SET content_hash = ?", ("wrong-hash",)),
        ("UPDATE resource_provenance SET verified_url = ?", ("https://example.test/wrong",)),
        ("UPDATE resource_provenance SET connector_name = ?", ("semantic_scholar",)),
        (
            "UPDATE connector_sync_runs SET status = ? WHERE id = (SELECT sync_run_id FROM resource_provenance LIMIT 1)",
            ("FAILED",),
        ),
    ],
)
def test_provenance_mismatch_is_never_visible(tmp_path, statement, params):
    logger = LearningLogger(tmp_path / "mismatch.db")
    service, _records = _seed_mixed_sources(logger)
    with logger._connect() as conn:
        conn.execute(statement, params)

    result = asyncio.run(
        service.search(ResourceQuery(query="unmapped-j1", resource_type="paper", allow_remote=False))
    )

    assert result["resources"] == []
    assert result["origin_kind"] == "unknown"


def test_connector_source_mismatch_is_failed_without_student_visible_rows(tmp_path):
    logger = LearningLogger(tmp_path / "source-mismatch.db")
    mismatched = _resource(
        "wrong-source",
        "semantic_scholar",
        "source mismatch paper",
        url="https://example.test/source-mismatch",
    )
    service = ExternalResourceService(
        storage=logger,
        registry=ConnectorRegistry([ArxivFixtureConnector(resources=[mismatched])]),
        vector_store=_NoopVectorStore(),
    )

    result = asyncio.run(
        service.sync("arxiv", ResourceQuery(query="source mismatch", resource_type="paper"))
    )

    assert result.status == "FAILED"
    assert result.error_code == "CONNECTOR_SOURCE_MISMATCH"
    assert logger.search_external_resources({"query": "source mismatch"}, include_expired=True) == []
    with logger._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM external_resources").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM resource_provenance").fetchone()[0] == 0


def test_portal_innovation_and_mcp_observation_exclude_unverified_history(tmp_path, monkeypatch):
    logger = LearningLogger(tmp_path / "student-paths.db")
    _service, _records = _seed_mixed_sources(logger)

    portal = ResourcePortalService(logger)
    portal_items = portal.search(ResourcePortalQuery(keyword="unmapped-j1", page_size=50))["items"]
    portal_titles = {item["title"] for item in portal_items}
    assert "bootstrap-j1 trusted paper" in portal_titles
    assert not any("fake" in title or "unknown" in title or "masquerading" in title for title in portal_titles)

    innovation = InnovationService(
        logger,
        retriever=_EmptyRetriever(),
        mcp_gateway=_EmptyGateway(),
        resource_portal=portal,
    )
    collection = innovation._collect_cases(topic="unmapped-j1", current_plan="baseline", course_id=None)
    cases = collection.cases
    case_titles = {case["title"] for case in cases}
    assert "bootstrap-j1 trusted paper" in case_titles
    assert not any("fake" in title or "unknown" in title or "masquerading" in title for title in case_titles)

    trusted = logger.external_resource_by_key("arxiv", "trusted-arxiv")
    monkeypatch.setattr(mcp_tools, "LearningLogger", lambda: logger)
    monkeypatch.setattr(
        mcp_tools,
        "get_default_registry",
        lambda: ConnectorRegistry(
            [
                ArxivFixtureConnector(
                    resources=[
                        _resource(
                            "trusted-arxiv",
                            "arxiv",
                            "bootstrap-j1 trusted paper",
                            url="https://arxiv.org/abs/2609.00001",
                        )
                    ]
                )
            ]
        ),
    )
    tool_data = mcp_tools._paper_search_real(
        mcp_tools.RealPaperSearchInput(query="unmapped-j1", source="arxiv", top_k=10)
    )
    assert trusted is not None
    assert [item["external_id"] for item in tool_data["resources"]] == ["trusted-arxiv"]
    observation = tool_result_to_observation(
        ToolResult(success=True, data=tool_data),
        step_number=1,
        tool_name="paper_search_real",
    )
    assert "historical-fake" not in observation.content
    assert "historical-unknown" not in observation.content
    assert "historical-arxiv-no-proof" not in observation.content


def test_verified_sync_is_atomic_when_provenance_write_fails(tmp_path):
    logger = LearningLogger(tmp_path / "atomic.db")
    resource = _resource(
        "atomic-arxiv",
        "arxiv",
        "atomic-j1 paper",
        url="https://arxiv.org/abs/2609.00002",
    )
    service = ExternalResourceService(
        storage=logger,
        registry=ConnectorRegistry([ArxivFixtureConnector(resources=[resource])]),
        vector_store=_NoopVectorStore(),
    )
    with logger._connect() as conn:
        conn.execute(
            """
            CREATE TRIGGER fail_j1_provenance
            BEFORE INSERT ON resource_provenance
            BEGIN
                SELECT RAISE(ABORT, 'injected provenance failure');
            END
            """
        )

    result = asyncio.run(
        service.sync(
            "arxiv",
            ResourceQuery(query="atomic-j1", resource_type="paper", allow_remote=True),
        )
    )

    with logger._connect() as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
            for table in (
                "external_resources",
                "resource_query_cache",
                "resource_query_cache_items",
                "resource_provenance",
            )
        }
        latest = conn.execute(
            "SELECT status, result_count FROM connector_sync_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()

    assert result.status == "FAILED"
    assert counts == {
        "external_resources": 0,
        "resource_query_cache": 0,
        "resource_query_cache_items": 0,
        "resource_provenance": 0,
    }
    assert dict(latest) == {"status": "FAILED", "result_count": 0}
