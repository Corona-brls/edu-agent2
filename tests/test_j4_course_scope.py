from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from backend import main
from backend.agent.observation import tool_result_to_observation
from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.schemas import ActionDecision, AgentAction
from backend.auth.password import hash_password
from backend.llm.base import LLMProvider
from backend.llm.schemas import LLMRequest, LLMResponse
from backend.rag.document_parser import Document, Page
from backend.storage.logger import LearningLogger, sha256_bytes
from mcp_server.schemas import ToolResult
from mcp_server.tools import TOOL_DEFINITIONS, call_tool


OTHER_SEARCH_TOOLS = (
    "paper_search",
    "vuln_lookup",
    "standard_lookup",
    "paper_search_real",
    "vuln_lookup_real",
    "standard_metadata_lookup",
    "github_project_search",
)


def _course(logger: LearningLogger, slug: str) -> tuple[int, int]:
    with logger._connect() as conn:
        course = conn.execute(
            "INSERT INTO courses (slug, name, description, assessment_method) VALUES (?, ?, '', '')",
            (slug, slug),
        )
        course_id = int(course.lastrowid)
        klass = conn.execute(
            "INSERT INTO classes (course_id, slug, name, term) VALUES (?, ?, ?, '2026')",
            (course_id, f"{slug}-class", f"{slug} class"),
        )
        return course_id, int(klass.lastrowid)


def _indexed_document(
    logger: LearningLogger,
    *,
    course_id: int,
    class_id: int,
    filename: str,
    text: str,
) -> tuple[int, int]:
    document = Document(
        source_path=filename,
        filename=filename,
        file_type=".md",
        title=filename,
        text=text,
        pages=[Page(text=text, page=1, section="J4", source_type="markdown")],
    )
    saved = logger.save_document(
        document=document,
        content_hash=sha256_bytes(text.encode("utf-8")),
        chunks=[{"chunk_text": text, "page": 1, "section": "J4", "source_type": "md"}],
        byte_size=len(text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
        class_id=class_id,
    )
    logger.set_document_publication(int(saved["id"]), published=True, actor_id=1)
    chunks = [
        chunk
        for chunk in logger.document_chunks(course_id=course_id)
        if int(chunk["document_id"]) == int(saved["id"])
    ]
    return int(saved["id"]), int(chunks[0]["id"])


def test_only_course_retriever_schema_requires_course_id():
    course_schema = TOOL_DEFINITIONS["course_retriever"].input_schema

    assert set(course_schema["required"]) == {"query", "course_id"}
    assert course_schema["properties"]["course_id"]["minimum"] == 1
    for name in OTHER_SEARCH_TOOLS:
        schema = TOOL_DEFINITIONS[name].input_schema
        assert "course_id" not in schema["properties"], name
        assert set(schema.get("required", [])) == {"query"}, name
    mermaid_schema = TOOL_DEFINITIONS["mermaid_generator"].input_schema
    assert "course_id" not in mermaid_schema["properties"]
    assert set(mermaid_schema.get("required", [])) == {"topic"}


def test_course_retriever_uses_scoped_indexed_chunks_and_keyword_fallback(tmp_path, monkeypatch):
    db_path = (tmp_path / "j4-tools.db").resolve()
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("VECTOR_STORE_DIR", str((tmp_path / "chroma").resolve()))
    monkeypatch.setattr(
        "backend.rag.hybrid_retriever._bounded_vector_store_init",
        lambda: (_ for _ in ()).throw(RuntimeError("chroma unavailable")),
    )
    logger = LearningLogger(db_path)
    course_a, class_a = _course(logger, "j4-a")
    course_b, class_b = _course(logger, "j4-b")
    doc_a, chunk_a = _indexed_document(
        logger,
        course_id=course_a,
        class_id=class_a,
        filename="a.md",
        text="J4AlphaOnlyMarker authentication course evidence.",
    )
    _indexed_document(
        logger,
        course_id=course_b,
        class_id=class_b,
        filename="b.md",
        text="J4BetaOnlyMarker blockchain course evidence.",
    )

    own = call_tool(
        "course_retriever",
        {"query": "J4AlphaOnlyMarker", "top_k": 3, "course_id": course_a},
    )
    cross = call_tool(
        "course_retriever",
        {"query": "J4AlphaOnlyMarker", "top_k": 3, "course_id": course_b},
    )
    missing = call_tool("course_retriever", {"query": "J4AlphaOnlyMarker", "top_k": 3})

    assert own.success is True
    assert own.data["fallback_used"] is True
    assert own.data["origin_kind"] == "course_document"
    assert own.data["results"][0]["document_id"] == doc_a
    assert own.data["results"][0]["chunk_id"] == chunk_a
    assert own.data["results"][0]["title"]
    assert "J4AlphaOnlyMarker" in own.data["results"][0]["snippet"]
    assert cross.success is True
    assert cross.data["results"] == []
    assert missing.success is False
    assert missing.error_code == "INVALID_ARGUMENTS"


def test_course_retriever_stdio_protocol_is_course_scoped(tmp_path, monkeypatch):
    db_path = (tmp_path / "j4-stdio.db").resolve()
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("VECTOR_STORE_DIR", str((tmp_path / "stdio-chroma").resolve()))
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.01")
    logger = LearningLogger(db_path)
    course_a, class_a = _course(logger, "stdio-a")
    course_b, class_b = _course(logger, "stdio-b")
    doc_a, chunk_a = _indexed_document(
        logger,
        course_id=course_a,
        class_id=class_a,
        filename="stdio-a.md",
        text="J4StdioAlphaMarker protocol scoped evidence.",
    )
    _indexed_document(
        logger,
        course_id=course_b,
        class_id=class_b,
        filename="stdio-b.md",
        text="J4StdioBetaMarker separate course evidence.",
    )

    async def run_check():
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
                definition = next(tool for tool in listed.tools if tool.name == "course_retriever")
                missing = await session.call_tool(
                    "course_retriever",
                    {"query": "J4StdioAlphaMarker", "top_k": 3},
                )
                own = await session.call_tool(
                    "course_retriever",
                    {"query": "J4StdioAlphaMarker", "top_k": 3, "course_id": course_a},
                )
                cross = await session.call_tool(
                    "course_retriever",
                    {"query": "J4StdioAlphaMarker", "top_k": 3, "course_id": course_b},
                )
                return (
                    definition.input_schema,
                    missing.structured_content,
                    own.structured_content,
                    cross.structured_content,
                )

    schema, missing, own, cross = asyncio.run(run_check())

    assert set(schema["required"]) == {"query", "course_id"}
    assert missing["success"] is False
    assert missing["error_code"] == "INVALID_ARGUMENTS"
    assert own["success"] is True
    assert own["data"]["fallback_used"] is True, own
    assert own["data"]["results"], own
    assert own["data"]["results"][0]["document_id"] == doc_a
    assert own["data"]["results"][0]["chunk_id"] == chunk_a
    assert "J4StdioAlphaMarker" in own["data"]["results"][0]["snippet"]
    assert cross["success"] is True
    assert cross["data"]["results"] == []


class _Provider(LLMProvider):
    provider_name = "j4-provider"
    model_name = "j4-provider"

    def __init__(self, forged_course_id: int):
        self.forged_course_id = forged_course_id
        self.calls = 0

    def decide_next_action(self, state_summary, available_tools):
        self.calls += 1
        if self.calls == 1:
            return ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name="course_retriever",
                arguments={
                    "query": "J4AgentBoundMarker",
                    "top_k": 1,
                    "course_id": self.forged_course_id,
                },
                reason="retrieve course evidence",
            )
        return ActionDecision(action=AgentAction.ANSWER, reason="evidence ready", task_complete=True)

    def generate_answer(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=f"Authenticated course evidence [{request.evidence[0].source_id}]",
            provider=self.provider_name,
            model=self.model_name,
        )

    def generate_structured(self, request, schema):
        raise AssertionError("not used")


class _Gateway:
    def __init__(self, logger: LearningLogger, course_id: int):
        self.logger = logger
        self.course_id = course_id
        self.arguments: list[dict] = []

    def get_agent_tool_catalog(self, role="student"):
        definition = TOOL_DEFINITIONS["course_retriever"]
        return [
            {
                "name": definition.name,
                "description": definition.agent_description,
                "input_schema": definition.input_schema,
                "risk": definition.risk,
                "network": definition.network,
                "read_only": definition.read_only,
                "enabled": definition.enabled,
            }
        ]

    def call_tool(self, name, arguments, *, run_id, step_id, step_number=1, role="student"):
        self.arguments.append(dict(arguments))
        chunk = self.logger.document_chunks(course_id=self.course_id)[0]
        result = ToolResult(
            success=True,
            data={
                "course_id": self.course_id,
                "results": [
                    {
                        "document_id": int(chunk["document_id"]),
                        "chunk_id": int(chunk["id"]),
                        "source": chunk["filename"],
                        "title": chunk["title"],
                        "snippet": chunk["chunk_text"],
                        "page": chunk["page"],
                        "section": chunk["section"],
                        "source_type": chunk["source_type"],
                        "final_score": 1.0,
                        "metadata": {"content_hash": chunk["content_hash"]},
                        "origin_kind": "course_document",
                        "delivery_mode": "local_rag",
                    }
                ],
                "abstained": False,
                "fallback_used": True,
                "origin_kind": "course_document",
            },
        )
        observation = tool_result_to_observation(
            result,
            step_number=step_number,
            tool_name=name,
            run_id=run_id,
            step_id=step_id,
            arguments=arguments,
        )
        return SimpleNamespace(result=result, observation=observation, fallback_used=True)


def test_agent_overwrites_model_course_id_before_guard_and_accepts_scoped_evidence(tmp_path):
    logger = LearningLogger(tmp_path / "agent.db")
    course_a, class_a = _course(logger, "agent-a")
    course_b, _ = _course(logger, "agent-b")
    doc_a, chunk_a = _indexed_document(
        logger,
        course_id=course_a,
        class_id=class_a,
        filename="agent-a.md",
        text="J4AgentBoundMarker belongs to authenticated course A.",
    )
    provider = _Provider(forged_course_id=course_b)
    gateway = _Gateway(logger, course_a)

    result = AgentOrchestrator(
        storage=logger,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        mcp_gateway=gateway,
    ).run("J4AgentBoundMarker", user_id="student-a", course_id=course_a, class_id=class_a)

    assert result["run_status"] == "COMPLETED"
    assert gateway.arguments == [
        {"query": "J4AgentBoundMarker", "top_k": 1, "course_id": course_a}
    ]
    assert result["tool_calls"][0]["tool_name"] == "course_retriever"
    assert result["citations"][0]["document_id"] == doc_a
    assert result["citations"][0]["chunk_id"] == chunk_a


def test_agent_without_course_never_executes_course_retriever(tmp_path):
    logger = LearningLogger(tmp_path / "agent-no-course.db")
    provider = _Provider(forged_course_id=999)
    gateway = _Gateway(logger, 999)

    with pytest.raises(ValueError, match="course_id is required"):
        AgentOrchestrator(
            storage=logger,
            planner_provider=provider,
            answer_provider=provider,
            fallback_provider=provider,
            mcp_gateway=gateway,
        ).run("J4AgentBoundMarker", user_id="2")
    assert gateway.arguments == []


def test_multicourse_http_requires_selection_and_rejects_cross_course(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "j4-http-jwt-secret-at-least-32-bytes")
    monkeypatch.setenv("VECTOR_STORE_DIR", str((tmp_path / "http-chroma").resolve()))
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setattr(main, "learning_logger", LearningLogger(tmp_path / "http.db"))
    course_a, class_a = _course(main.learning_logger, "http-a")
    course_b, class_b = _course(main.learning_logger, "http-b")
    _indexed_document(
        main.learning_logger,
        course_id=course_a,
        class_id=class_a,
        filename="http-a.md",
        text="J4HttpAlphaOnlyMarker course A evidence.",
    )
    _indexed_document(
        main.learning_logger,
        course_id=course_b,
        class_id=class_b,
        filename="http-b.md",
        text="J4HttpBetaOnlyMarker course B evidence.",
    )
    multi_id = main.learning_logger.upsert_user(
        username="j4_multi",
        display_name="J4 Multi",
        password_hash=hash_password("j4-password-123"),
        role="student",
        must_change_password=False,
    )
    only_a_id = main.learning_logger.upsert_user(
        username="j4_only_a",
        display_name="J4 A",
        password_hash=hash_password("j4-password-123"),
        role="student",
        must_change_password=False,
    )
    with main.learning_logger._connect() as conn:
        conn.executemany(
            "INSERT INTO class_members (class_id, user_id) VALUES (?, ?)",
            [(class_a, multi_id), (class_b, multi_id), (class_a, only_a_id)],
        )

    with TestClient(main.app) as client:
        def headers(username: str) -> dict[str, str]:
            login = client.post(
                "/api/auth/login",
                json={"username": username, "password": "j4-password-123"},
            )
            assert login.status_code == 200
            return {"Authorization": f"Bearer {login.json()['access_token']}"}

        multi = headers("j4_multi")
        only_a = headers("j4_only_a")
        no_selection = client.post(
            "/api/chat",
            headers=multi,
            json={"question": "J4HttpAlphaOnlyMarker"},
        )
        selected_a = client.post(
            "/api/chat",
            headers=multi,
            json={"question": "J4HttpAlphaOnlyMarker", "course_id": course_a},
        )
        cross = client.post(
            "/api/chat",
            headers=only_a,
            json={"question": "J4HttpBetaOnlyMarker", "course_id": course_b},
        )
        visible_courses = client.get("/api/courses", headers=multi)
        portal_without_course = client.get("/api/resources/portal", headers=multi)
        portal_a = client.get(
            "/api/resources/portal",
            headers=multi,
            params={"course_id": course_a, "resource_type": "course_document"},
        )
        practice_without_course = client.post(
            "/api/practice/guide",
            headers=multi,
            json={"topic": "J4HttpAlphaOnlyMarker"},
        )
        practice_cross = client.post(
            "/api/practice/guide",
            headers=only_a,
            json={"topic": "J4HttpBetaOnlyMarker", "course_id": course_b},
        )
        innovation_without_course = client.post(
            "/api/innovation/sessions",
            headers=multi,
            json={"topic": "J4 topic", "current_plan": "J4 current plan"},
        )
        innovation_cross = client.post(
            "/api/innovation/sessions",
            headers=only_a,
            json={"topic": "J4 topic", "current_plan": "J4 current plan", "course_id": course_b},
        )

    assert no_selection.status_code == 400
    assert no_selection.json()["error_code"] == "COURSE_REQUIRED"
    assert selected_a.status_code == 200
    assert selected_a.json()["course_id"] == course_a
    assert "J4HttpAlphaOnlyMarker" in str(selected_a.json())
    assert cross.status_code == 403
    assert cross.json()["error_code"] == "COURSE_FORBIDDEN"
    assert visible_courses.status_code == 200
    assert {item["id"] for item in visible_courses.json()} == {course_a, course_b}
    assert portal_without_course.status_code == 400
    assert portal_without_course.json()["error_code"] == "COURSE_REQUIRED"
    assert portal_a.status_code == 200
    assert "J4HttpAlphaOnlyMarker" in str(portal_a.json())
    assert "J4HttpBetaOnlyMarker" not in str(portal_a.json())
    assert practice_without_course.status_code == 400
    assert practice_without_course.json()["error_code"] == "COURSE_REQUIRED"
    assert practice_cross.status_code == 403
    assert practice_cross.json()["error_code"] == "COURSE_FORBIDDEN"
    assert innovation_without_course.status_code == 400
    assert innovation_without_course.json()["error_code"] == "COURSE_REQUIRED"
    assert innovation_cross.status_code == 403
    assert innovation_cross.json()["error_code"] == "COURSE_FORBIDDEN"
