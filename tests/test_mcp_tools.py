import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from backend.connectors.schemas import ExternalResource, ResourceQuery
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import HybridRetriever
from backend.resources.normalizer import content_hash
from backend.storage.logger import LearningLogger, sha256_text
from backend.tools.tool_policy import get_agent_tool_catalog
from mcp_server import tools as tool_module
from mcp_server.server import build_server
from mcp_server.tools import TOOL_DEFINITIONS, call_tool, list_tool_definitions


def test_course_retriever_consumer_drops_chunk_reindexing_during_hybrid_query(tmp_path, monkeypatch):
    storage = LearningLogger(tmp_path / "mcp-course-recheck.db")
    course_id = storage.default_course_id()
    parsed = Document(
        source_path=Path("course/mcp-recheck.md"),
        filename="mcp-recheck.md",
        file_type="md",
        title="MCP Recheck",
        text="MCP_RECHECK_MARKER published course evidence",
        pages=[
            Page(
                text="MCP_RECHECK_MARKER published course evidence",
                page=1,
                section="mcp-recheck",
                source_type="markdown",
            )
        ],
    )
    saved = storage.save_document(
        document=parsed,
        content_hash=sha256_text(parsed.text),
        chunks=[parsed.text],
        byte_size=len(parsed.text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
    )
    document_id = int(saved["id"])
    chunk_id = int(storage.chunks_for_document(document_id)[0]["id"])
    storage.set_document_publication(document_id, published=True, actor_id=1)

    class ReindexOnQueryVectorStore:
        calls = 0

        def query(self, text: str, top_k: int = 5):
            self.calls += 1
            storage.begin_document_reindex(document_id)
            return [
                {
                    "id": str(chunk_id),
                    "metadata": {"chunk_id": chunk_id, "document_id": document_id},
                    "score": 0.99,
                }
            ]

    vector_store = ReindexOnQueryVectorStore()
    retriever = HybridRetriever(storage=storage, vector_store=vector_store, min_score=0.01)
    monkeypatch.setattr(tool_module, "HybridRetriever", lambda storage=None: retriever)

    payload = tool_module._course_retriever(
        tool_module.CourseRetrieverInput(query="MCP_RECHECK_MARKER", top_k=1, course_id=course_id)
    )

    assert vector_store.calls == 1
    assert storage.document(document_id)["index_status"] == "EMBEDDING"
    assert payload["results"] == []
    assert payload["abstained"] is True


def test_mcp_server_can_list_tools():
    async def run_check():
        server = build_server()
        tools = await server.list_tools()
        return {tool.name for tool in tools}

    names = asyncio.run(run_check())

    assert {
        "course_retriever",
        "paper_search",
        "vuln_lookup",
        "standard_lookup",
        "mermaid_generator",
    }.issubset(names)
    assert {"paper_search_real", "vuln_lookup_real", "standard_metadata_lookup", "github_project_search"}.issubset(names)


def test_stdio_mcp_protocol_exposes_strict_schema_and_calls_mermaid_generator():
    async def run_check():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            cwd=".",
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed = await session.list_tools()
                mermaid = next(tool for tool in listed.tools if tool.name == "mermaid_generator")
                valid = await session.call_tool(
                    "mermaid_generator",
                    {"topic": "MCP RAG 系统架构", "diagram_type": "flowchart"},
                )
                invalid = await session.call_tool(
                    "mermaid_generator",
                    {"diagram_type": "flowchart"},
                )
                return (
                    mermaid.input_schema,
                    valid.is_error,
                    valid.structured_content,
                    invalid.is_error,
                    invalid.structured_content,
                )

    input_schema, valid_is_error, valid, invalid_is_error, invalid = asyncio.run(run_check())

    assert input_schema == TOOL_DEFINITIONS["mermaid_generator"].input_schema
    assert set(input_schema["required"]) == {"topic"}
    assert set(input_schema["properties"]) == {"topic", "diagram_type"}
    assert valid_is_error is False
    assert valid["success"] is True
    assert "mermaid" in valid["data"]
    assert invalid_is_error is True
    assert invalid["success"] is False
    assert invalid["error_code"] == "INVALID_ARGUMENTS"
    assert invalid["metadata"]["errors"]


def test_stdio_mcp_protocol_passes_disabled_tool_status():
    async def run_check():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            cwd=".",
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "github_project_search",
                    {"query": "open5gs", "top_k": 1},
                )
                return result.structured_content

    result = asyncio.run(run_check())

    assert result["success"] is False
    assert result["error_code"] == "TOOL_DISABLED"
    assert result["data"] == {}


def test_stdio_mcp_protocol_aligns_connector_failure_and_stale_cache(tmp_path):
    db_path = tmp_path / "j6-failure.db"
    query = ResourceQuery(query="J6 stale evidence", resource_type="paper", page_size=1)
    resource = ExternalResource(
        external_id="j6-stale-paper",
        resource_type="paper",
        source_name="arxiv",
        title="J6 stale trusted paper",
        abstract="Previously verified stale evidence.",
        url="https://arxiv.org/abs/j6-stale-paper",
    )
    logger = LearningLogger(db_path)
    run_id = logger.create_connector_sync_run(
        connector_name="arxiv",
        query=query.model_dump(mode="json"),
        status="RUNNING",
        cache_hit=False,
    )
    logger.commit_verified_sync(
        run_id=run_id,
        connector_name="arxiv",
        origin_kind="remote",
        query=query.model_dump(mode="json"),
        resources=[{"resource": resource.model_dump(mode="json"), "content_hash": content_hash(resource)}],
        ttl_hours=0,
        latency_ms=1,
    )

    async def run_check():
        params = _j6_server_params(tmp_path, db_path, mode="fail")
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed = await session.list_tools()
                missing = await session.call_tool("paper_search_real", {"top_k": 1})
                failed = await session.call_tool(
                    "paper_search_real",
                    {"query": "J6 no cache", "top_k": 1, "source": "arxiv"},
                )
                stale = await session.call_tool(
                    "paper_search_real",
                    {"query": query.query, "top_k": 1, "source": "arxiv"},
                )
                return listed, missing, failed, stale

    listed, missing, failed, stale = asyncio.run(run_check())

    assert any(tool.name == "paper_search_real" for tool in listed.tools)
    assert missing.is_error is True
    assert missing.structured_content["success"] is False
    assert missing.structured_content["error_code"] == "INVALID_ARGUMENTS"
    assert failed.is_error is True
    assert failed.structured_content["success"] is False
    assert failed.structured_content["error_code"] == "CONNECTOR_SERVICE_ERROR"
    assert stale.is_error is False
    assert stale.structured_content["success"] is True
    assert stale.structured_content["data"]["delivery_mode"] == "stale_cache"
    assert stale.structured_content["data"]["fallback_used"] is True
    assert stale.structured_content["data"]["upstream_error_code"] == "CONNECTOR_SERVICE_ERROR"


def test_stdio_mcp_timeout_has_no_late_successful_cache_commit(tmp_path):
    db_path = tmp_path / "j6-slow.db"
    marker_path = tmp_path / "j6-worker.json"
    LearningLogger(db_path)
    before = _j6_db_counts(db_path)

    async def run_check():
        params = _j6_server_params(tmp_path, db_path, mode="slow", marker_path=marker_path)
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                started = time.time()
                result = await session.call_tool(
                    "paper_search_real",
                    {"query": "J6 delayed result", "top_k": 1, "source": "arxiv"},
                )
                returned_at = time.time()
                immediate = _j6_db_counts(db_path)
                await asyncio.sleep(0.8)
                after_wait = _j6_db_counts(db_path)
                return result, started, returned_at, immediate, after_wait

    result, started, returned_at, immediate, after_wait = asyncio.run(run_check())
    worker = json.loads(marker_path.read_text(encoding="utf-8"))

    assert result.is_error is True
    assert result.structured_content["success"] is False
    assert result.structured_content["error_code"] == "TOOL_TIMEOUT"
    assert returned_at - started < 0.4
    assert worker["ended_at"] > returned_at
    assert before["external_resources"] == 0
    assert immediate["external_resources"] == 0
    assert after_wait["external_resources"] == 0
    assert after_wait["resource_query_cache_items"] == 0
    assert after_wait["resource_provenance"] == 0
    assert after_wait["successful_sync_runs"] == 0
    assert after_wait["running_sync_runs"] == 0
    assert after_wait["failed_sync_runs"] == 1
    assert after_wait["last_error_code"] == "CONNECTOR_TIMEOUT"


def test_tool_definitions_include_policy_fields():
    definitions = {tool.name: tool for tool in list_tool_definitions()}

    assert definitions["course_retriever"].input_schema["type"] == "object"
    assert definitions["course_retriever"].risk == "LOW"
    assert definitions["course_retriever"].read_only is True
    assert definitions["course_retriever"].network is False
    assert "何时使用" in definitions["course_retriever"].agent_description
    assert "何时不应使用" in definitions["course_retriever"].agent_description


def test_five_tools_accept_valid_arguments():
    calls = {
        "course_retriever": {"query": "5G NAS 认证风险", "top_k": 2, "course_id": 1},
        "paper_search": {"query": "LLM 漏洞检测研究方向", "top_k": 2},
        "vuln_lookup": {"query": "CVE 样例", "top_k": 2},
        "standard_lookup": {"query": "3GPP TS 24.501 NAS", "top_k": 2},
        "mermaid_generator": {"topic": "MCP-RAG 系统架构", "diagram_type": "flowchart"},
    }

    for name, arguments in calls.items():
        result = call_tool(name, arguments)
        assert result.success is True, name
        assert result.error_code is None
        assert result.metadata["tool_name"] == name

    assert "mermaid" in call_tool("mermaid_generator", calls["mermaid_generator"]).data


def test_local_paper_search_returns_course_clues_not_verified_papers():
    result = call_tool("paper_search", {"query": "nonexistent paper title", "top_k": 2})

    assert result.success is True
    assert result.data["result_type"] == "local_course_clues"
    assert result.data["verified_papers"] == []
    for item in result.data["results"]:
        assert item["resource_type"] == "course_document"
        assert item["verification_status"] == "local_course_clue"


def test_real_tools_are_registered_but_network_tools_default_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_DB_PATH", str((tmp_path / "mcp-tools.db").resolve()))
    definitions = {tool.name: tool for tool in list_tool_definitions()}

    assert definitions["paper_search_real"].network is True
    assert definitions["vuln_lookup_real"].network is True
    assert definitions["github_project_search"].network is True
    assert definitions["standard_lookup_real"].network is True
    assert definitions["paper_search_real"].enabled is False
    assert definitions["vuln_lookup_real"].enabled is False
    assert definitions["github_project_search"].enabled is False
    assert definitions["standard_lookup_real"].enabled is False
    assert definitions["standard_metadata_lookup"].network is False
    assert definitions["standard_metadata_lookup"].enabled is True

    result = call_tool("standard_metadata_lookup", {"query": "3GPP TS 24.501", "top_k": 1})
    assert result.success is True
    assert result.data["resources"]
    assert result.data["resources"][0]["resource_type"] == "standard"


def test_real_standard_tool_routes_rfc_and_preserves_lookup_contract(monkeypatch):
    calls = []

    def fake_search(connector_name, query, **kwargs):
        calls.append((connector_name, query, kwargs))
        return {
            "connector": "rfc_editor",
            "status": "SUCCESS",
            "resources": [],
            "lookup_state": "FOUND",
            "source_attempts": [{"source_name": "rfc_editor", "status": "FOUND"}],
            "incomplete_results": False,
        }

    original = TOOL_DEFINITIONS["standard_lookup_real"]
    monkeypatch.setitem(
        TOOL_DEFINITIONS,
        "standard_lookup_real",
        original.model_copy(update={"enabled": True}),
    )
    monkeypatch.setattr(tool_module, "_run_connector_search", fake_search)

    result = call_tool("standard_lookup_real", {"query": "RFC 9000", "top_k": 1})

    assert result.success is True
    assert calls[0][0] is None
    assert calls[0][1].resource_type == "standard"
    assert result.data["lookup_state"] == "FOUND"
    assert result.data["source_attempts"][0]["source_name"] == "rfc_editor"
    assert result.data["incomplete_results"] is False


def test_real_vulnerability_tool_reports_safe_query_normalization(monkeypatch):
    original = TOOL_DEFINITIONS["vuln_lookup_real"]
    TOOL_DEFINITIONS["vuln_lookup_real"] = original.model_copy(update={"enabled": True})
    monkeypatch.setattr(
        tool_module,
        "_run_connector_sync",
        lambda connector_name, query, **kwargs: {
            "connector": connector_name,
            "status": "SUCCESS",
            "cache_hit": False,
            "resources": [],
        },
    )
    raw_query = "Explain CVE-2021-44228 api_key=sk-test-secret-value"
    try:
        result = call_tool("vuln_lookup_real", {"query": raw_query, "top_k": 1})
    finally:
        TOOL_DEFINITIONS["vuln_lookup_real"] = original

    assert result.success is True
    assert result.data["query_mode"] == "cve_id"
    assert result.data["normalized_cve_id"] == "CVE-2021-44228"
    assert result.data["query_summary"].startswith("sha256:")
    assert raw_query not in str(result.model_dump())
    assert "sk-test-secret-value" not in str(result.model_dump())


def test_invalid_arguments_return_unified_error():
    result = call_tool("course_retriever", {"query": "", "unexpected": True})

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENTS"
    assert result.error_message == "工具参数不符合 schema"
    assert result.metadata["errors"]


def test_resource_mcp_tools_share_connector_query_length_boundary(monkeypatch):
    calls = []

    def fake_run(connector_name, query, **kwargs):
        calls.append((connector_name, query.query))
        return {"connector": connector_name, "status": "SUCCESS", "resources": []}

    monkeypatch.setattr(tool_module, "_run_connector_search", fake_run)
    tool_names = (
        "paper_search_real",
        "vuln_lookup_real",
        "standard_metadata_lookup",
        "standard_lookup_real",
        "github_project_search",
    )
    originals = {name: TOOL_DEFINITIONS[name] for name in tool_names}
    try:
        for name in tool_names:
            TOOL_DEFINITIONS[name] = originals[name].model_copy(update={"enabled": True})
            arguments = {"query": "x" * 300, "top_k": 1}
            if name == "paper_search_real":
                arguments["source"] = "auto"
            accepted = call_tool(name, arguments)
            assert accepted.success is True

        calls_before_rejection = len(calls)
        for name in tool_names:
            arguments = {"query": "x" * 301, "top_k": 1}
            if name == "paper_search_real":
                arguments["source"] = "auto"
            rejected = call_tool(name, arguments)
            assert rejected.success is False
            assert rejected.error_code == "INVALID_ARGUMENTS"
        assert len(calls) == calls_before_rejection
    finally:
        for name, definition in originals.items():
            TOOL_DEFINITIONS[name] = definition


def test_disabled_tool_returns_unified_error():
    original = TOOL_DEFINITIONS["paper_search"]
    TOOL_DEFINITIONS["paper_search"] = original.model_copy(update={"enabled": False})
    try:
        result = call_tool("paper_search", {"query": "LLM", "top_k": 1})
    finally:
        TOOL_DEFINITIONS["paper_search"] = original

    assert result.success is False
    assert result.error_code == "TOOL_DISABLED"
    assert result.error_message == "工具已禁用：paper_search"


def test_unknown_tool_is_not_executed():
    result = call_tool("delete_file", {"path": ".env"})

    assert result.success is False
    assert result.error_code == "TOOL_NOT_FOUND"


def test_timeout_returns_unified_error(monkeypatch):
    original = TOOL_DEFINITIONS["course_retriever"]
    TOOL_DEFINITIONS["course_retriever"] = original.model_copy(update={"timeout_seconds": 0.01})

    def slow_tool(payload):
        time.sleep(0.2)
        return {"results": []}

    monkeypatch.setattr(tool_module, "_course_retriever", slow_tool)
    try:
        result = call_tool("course_retriever", {"query": "NAS", "top_k": 1, "course_id": 1})
    finally:
        TOOL_DEFINITIONS["course_retriever"] = original

    assert result.success is False
    assert result.error_code == "TOOL_TIMEOUT"


def test_agent_tool_catalog_filters_disabled_high_risk_network_and_roles():
    original_course = TOOL_DEFINITIONS["course_retriever"]
    original_paper = TOOL_DEFINITIONS["paper_search"]
    original_vuln = TOOL_DEFINITIONS["vuln_lookup"]
    original_standard = TOOL_DEFINITIONS["standard_lookup"]
    TOOL_DEFINITIONS["course_retriever"] = original_course.model_copy(update={"enabled": False})
    TOOL_DEFINITIONS["paper_search"] = original_paper.model_copy(update={"risk": "HIGH"})
    TOOL_DEFINITIONS["vuln_lookup"] = original_vuln.model_copy(update={"network": True})
    TOOL_DEFINITIONS["standard_lookup"] = original_standard.model_copy(update={"roles": ["teacher"]})
    try:
        catalog = get_agent_tool_catalog(role="student")
    finally:
        TOOL_DEFINITIONS["course_retriever"] = original_course
        TOOL_DEFINITIONS["paper_search"] = original_paper
        TOOL_DEFINITIONS["vuln_lookup"] = original_vuln
        TOOL_DEFINITIONS["standard_lookup"] = original_standard

    names = {tool["name"] for tool in catalog}
    assert names == {"mermaid_generator", "standard_metadata_lookup"}


def test_no_dangerous_write_or_path_tools_are_registered():
    names = {tool.name for tool in list_tool_definitions()}

    assert "delete_file" not in names
    assert "write_file" not in names
    assert "read_path" not in names
    assert "execute_code" not in names


def _j6_server_params(tmp_path, db_path, *, mode: str, marker_path=None):
    root = Path(__file__).resolve().parents[1]
    marker = marker_path or (tmp_path / "j6-unused-marker.json")
    env = dict(os.environ)
    env.update(
        {
            "SQLITE_DB_PATH": str(db_path.resolve()),
            "ENABLE_REAL_PAPER_SEARCH": "true",
            "EXTERNAL_RESOURCE_INDEX_ENABLED": "false",
            "EXTERNAL_HTTP_MAX_RETRIES": "0",
            "EXTERNAL_CACHE_TTL_HOURS": "0",
            "J6_CONNECTOR_MODE": mode,
            "J6_TOOL_TIMEOUT_SECONDS": "0.15",
            "J6_DELAY_SECONDS": "0.5",
            "J6_MARKER_PATH": str(marker.resolve()),
            "PYTHONPATH": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return StdioServerParameters(
        command=sys.executable,
        args=[str((root / "tests" / "j6_mcp_server_helper.py").resolve())],
        cwd=str(root),
        env=env,
    )


def _j6_db_counts(db_path):
    with sqlite3.connect(db_path) as conn:
        result = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "external_resources",
                "resource_query_cache_items",
                "resource_provenance",
                "connector_sync_runs",
            )
        }
        result["running_sync_runs"] = conn.execute(
            "SELECT COUNT(*) FROM connector_sync_runs WHERE status = 'RUNNING'"
        ).fetchone()[0]
        result["successful_sync_runs"] = conn.execute(
            "SELECT COUNT(*) FROM connector_sync_runs WHERE status = 'SUCCESS'"
        ).fetchone()[0]
        result["failed_sync_runs"] = conn.execute(
            "SELECT COUNT(*) FROM connector_sync_runs WHERE status = 'FAILED'"
        ).fetchone()[0]
        row = conn.execute(
            "SELECT error_code FROM connector_sync_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        result["last_error_code"] = row[0] if row else None
        return result
