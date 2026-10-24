from __future__ import annotations

import asyncio
import json
import os
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from backend.agent.observation import tool_result_to_observation
from backend.connectors.registry import ConnectorRegistry
from backend.connectors.schemas import ExternalResource, ResourceQuery
from backend.connectors.standard_source import StandardSourceConnector
from backend.resources.normalizer import content_hash
from backend.resources.portal import ResourcePortalQuery, ResourcePortalService
from backend.resources.service import ExternalResourceService
from backend.storage.logger import LearningLogger
from mcp_server.tools import call_tool


def _seed_misclassified_standard(logger: LearningLogger, query: ResourceQuery) -> int:
    resource = ExternalResource(
        external_id="3GPP-TS-24.501",
        resource_type="standard",
        source_name="standard_source",
        title="3GPP TS 24.501",
        abstract=(
            "Standard metadata only. Version: Release 18 metadata. "
            "Local full text uploaded: True."
        ),
        url=(
            "https://portal.3gpp.org/desktopmodules/Specifications/"
            "SpecificationDetails.aspx?specificationId=3370"
        ),
        tags=["3GPP", "5G", "NAS"],
        module="5G NAS",
        raw_metadata={
            "version": "Release 18 metadata",
            "local_uploaded": True,
            "local_match": "knowledge_base/3GPP_TS_24.501_teaching_notes.md",
            "official_metadata_only": True,
        },
        rights_note="Metadata and official link only; copyrighted full text is not fetched.",
    )
    run_id = logger.create_connector_sync_run(
        connector_name="standard_source",
        query=query.model_dump(mode="json"),
        status="RUNNING",
        cache_hit=False,
    )
    saved = logger.commit_verified_sync(
        run_id=run_id,
        connector_name="standard_source",
        origin_kind="local_catalog",
        query=query.model_dump(mode="json"),
        resources=[
            {
                "resource": resource.model_dump(mode="json"),
                "content_hash": content_hash(resource),
            }
        ],
        ttl_hours=24,
        latency_ms=0,
    )
    return int(saved[0]["id"])


def test_old_verified_standard_cache_is_corrected_in_every_student_view(tmp_path, monkeypatch):
    db_path = (tmp_path / "l2-standard.db").resolve()
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "3GPP_TS_24.501_teaching_notes.md").write_text(
        "Teaching notes only; this is not the copyrighted standard full text.",
        encoding="utf-8",
    )
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("VECTOR_STORE_DIR", str((tmp_path / "chroma").resolve()))
    logger = LearningLogger(db_path)
    query = ResourceQuery(
        query="3GPP TS 24.501",
        resource_type="standard",
        page_size=1,
    )
    resource_id = _seed_misclassified_standard(logger, query)
    connector = StandardSourceConnector(knowledge_dir=knowledge_dir)
    service = ExternalResourceService(
        storage=logger,
        registry=ConnectorRegistry([connector]),
    )

    service_payload = asyncio.run(service.search(query, connector_name="standard_source"))
    portal_payload = ResourcePortalService(logger).search(
        ResourcePortalQuery(resource_type="standard", page_size=50)
    )
    mcp_result = call_tool(
        "standard_metadata_lookup",
        {"query": "3GPP TS 24.501", "top_k": 1},
    )
    observation = tool_result_to_observation(
        mcp_result,
        step_number=1,
        tool_name="standard_metadata_lookup",
    )

    service_item = service_payload["resources"][0]
    portal_item = next(
        item
        for item in portal_payload["items"]
        if item["external_id"] == "3GPP-TS-24.501"
    )
    mcp_item = mcp_result.data["resources"][0]
    for item in (service_item, mcp_item):
        assert "Local full text uploaded: True" not in item["abstract"]
        assert item["raw_metadata"]["local_uploaded"] is False
        assert item["raw_metadata"]["local_match"] is None
        assert item["raw_metadata"]["official_metadata_only"] is True
        assert item["url"].startswith("https://portal.3gpp.org/")
    assert service_payload["cache_hit"] is True
    assert service_payload["delivery_mode"] == "fresh_cache"
    assert portal_item["type"] == "standard"
    assert "Local full text uploaded: True" not in portal_item["summary"]
    assert "Local full text uploaded: True" not in portal_item["raw_summary"]
    assert portal_item["metadata"]["local_full_text_available"] is False
    assert portal_item["metadata"]["raw_metadata"]["local_uploaded"] is False
    assert portal_item["metadata"]["raw_metadata"]["local_match"] is None
    assert "Local full text uploaded: True" not in observation.content
    assert '"local_uploaded": false' in observation.content

    with logger._connect() as conn:
        stored = conn.execute(
            "SELECT abstract, raw_metadata_json FROM external_resources WHERE id = ?",
            (resource_id,),
        ).fetchone()
    stored_metadata = json.loads(stored["raw_metadata_json"])
    assert "Local full text uploaded: True" in stored["abstract"]
    assert stored_metadata["local_uploaded"] is True
    assert logger.resource_provenance(resource_id)["sync_status"] == "SUCCESS"


def test_standard_stdio_protocol_corrects_verified_legacy_cache(tmp_path, monkeypatch):
    db_path = (tmp_path / "l2-standard-stdio.db").resolve()
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("VECTOR_STORE_DIR", str((tmp_path / "chroma").resolve()))
    logger = LearningLogger(db_path)
    query = ResourceQuery(
        query="3GPP TS 24.501",
        resource_type="standard",
        page_size=1,
    )
    resource_id = _seed_misclassified_standard(logger, query)

    async def run_protocol_check():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            cwd=".",
            env=dict(os.environ),
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed = await session.list_tools()
                result = await session.call_tool(
                    "standard_metadata_lookup",
                    {"query": "3GPP TS 24.501", "top_k": 1},
                )
                return {tool.name for tool in listed.tools}, result

    names, result = asyncio.run(run_protocol_check())

    assert "standard_metadata_lookup" in names
    assert result.is_error is False
    assert result.structured_content["success"] is True
    payload = result.structured_content["data"]
    assert payload["cache_hit"] is True
    assert payload["delivery_mode"] == "fresh_cache"
    item = payload["resources"][0]
    assert item["source_name"] == "standard_source"
    assert item["url"].startswith("https://portal.3gpp.org/")
    assert "Local full text uploaded: True" not in item["abstract"]
    assert item["raw_metadata"]["local_uploaded"] is False
    assert item["raw_metadata"]["local_match"] is None
    assert item["raw_metadata"]["official_metadata_only"] is True

    with logger._connect() as conn:
        stored = conn.execute(
            "SELECT abstract, raw_metadata_json FROM external_resources WHERE id = ?",
            (resource_id,),
        ).fetchone()
    assert "Local full text uploaded: True" in stored["abstract"]
    assert json.loads(stored["raw_metadata_json"])["local_uploaded"] is True
    assert logger.resource_provenance(resource_id)["sync_status"] == "SUCCESS"
