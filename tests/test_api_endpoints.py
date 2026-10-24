import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from backend import main
from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.schemas import ActionDecision, AgentAction
from backend.connectors.registry import ConnectorRegistry
from backend.rag.document_parser import Document, Page
from backend.schemas import AgentContinueRequest, ChatRequest, PracticeGuideRequest
from backend.storage.logger import LearningLogger, sha256_bytes
from backend.tools.mcp_client import MCPClientGateway
from mcp_server.tools import TOOL_DEFINITIONS
from tests.test_agent_orchestration import SequenceProvider
from tests.test_vector_store import KeywordEmbedding


@pytest.fixture(autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    original_logger = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "test.db")
    yield
    main.learning_logger = original_logger


def test_health_endpoint():
    assert main.health() == {"status": "ok", "service": "course-agent"}


def test_agent_remote_source_reservation_is_atomic_and_filters_nonremote(monkeypatch):
    class Connector:
        def __init__(self, name, *, enabled=True, supports_remote=True):
            self.name = name
            self.enabled = enabled
            self.supports_remote = supports_remote

    original_registry = main.connector_registry
    main.connector_registry = __import__(
        "backend.connectors.registry", fromlist=["ConnectorRegistry"]
    ).ConnectorRegistry(
        [
            Connector("semantic_scholar"),
            Connector("arxiv"),
            Connector("standard_source", supports_remote=False),
        ]
    )
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_MAX_CALLS", 1)
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_WINDOW_SECONDS", 60.0)
    main._RESOURCE_RATE_BUCKETS.clear()
    try:
        main._reserve_resource_rate_limits(7, ["arxiv"])
        with pytest.raises(HTTPException) as captured:
            main._reserve_agent_tool_rate_limit(7, "paper_search_real", {"source": "auto"})

        assert getattr(captured.value, "status_code", None) == 429
        assert len(main._RESOURCE_RATE_BUCKETS[(7, "arxiv")]) == 1
        assert len(main._RESOURCE_RATE_BUCKETS[(7, "semantic_scholar")]) == 0
        assert main._agent_remote_sources("standard_metadata_lookup", {}) == []
    finally:
        main._RESOURCE_RATE_BUCKETS.clear()
        main.connector_registry = original_registry


def test_chat_and_continue_surface_remote_rate_limit_before_gateway(tmp_path, monkeypatch):
    class EnabledGitHub:
        name = "github"
        enabled = True
        supports_remote = True

    class ZeroDispatchGateway:
        def __init__(self):
            self.calls = 0

        def start(self):
            return None

        def stop(self):
            return None

        def get_agent_tool_catalog(self, role="student"):
            definition = TOOL_DEFINITIONS["github_project_search"]
            return [
                {
                    "name": definition.name,
                    "description": definition.agent_description,
                    "input_schema": definition.input_schema,
                    "risk": definition.risk,
                    "network": True,
                    "read_only": True,
                    "enabled": True,
                }
            ]

        def call_tool(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("rate limit must reject before gateway dispatch")

    original_gateway = main.mcp_gateway
    original_registry = main.connector_registry
    original_definition = TOOL_DEFINITIONS["github_project_search"]
    gateway = ZeroDispatchGateway()
    main.mcp_gateway = gateway
    main.connector_registry = ConnectorRegistry([EnabledGitHub()])
    monkeypatch.setitem(
        TOOL_DEFINITIONS,
        "github_project_search",
        original_definition.model_copy(update={"enabled": True}),
    )
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_MAX_CALLS", 1)
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_WINDOW_SECONDS", 60.0)
    main._RESOURCE_RATE_BUCKETS.clear()
    user = _demo_student()
    main._reserve_resource_rate_limits(int(user["id"]), ["github"])
    decision = ActionDecision(
        action=AgentAction.CALL_TOOL,
        tool_name="github_project_search",
        arguments={"query": "open5gs", "top_k": 1},
        reason="public project lookup",
    )
    paused_provider = SequenceProvider(
        [ActionDecision(action=AgentAction.ASK_USER, arguments={"question": "which project"}, reason="clarify")]
    )
    paused = AgentOrchestrator(
        storage=main.learning_logger,
        planner_provider=paused_provider,
        answer_provider=paused_provider,
        fallback_provider=paused_provider,
        mcp_gateway=gateway,
    ).run(
        "请补充项目名",
        user_id=str(user["id"]),
        course_id=main.learning_logger.default_course_id(),
        class_id=main.learning_logger.default_class_id(),
    )
    providers = iter([SequenceProvider([decision]), SequenceProvider([decision])])
    monkeypatch.setattr(main, "create_llm_provider", lambda: next(providers))
    try:
        with TestClient(main.app) as client:
            headers = _login_headers(client, "student_demo")
            chat_response = client.post(
                "/api/chat",
                headers=headers,
                json={"question": "请查询 GitHub open5gs 公开仓库"},
            )
            continue_response = client.post(
                f"/api/agent/runs/{paused['run_id']}/continue",
                headers=headers,
                json={"user_input": "请查询 GitHub open5gs 公开仓库"},
            )

        for response in (chat_response, continue_response):
            assert response.status_code == 429
            assert response.json()["error_code"] == "RESOURCE_RATE_LIMIT"
            assert int(response.headers["Retry-After"]) >= 1
        assert gateway.calls == 0
    finally:
        main._RESOURCE_RATE_BUCKETS.clear()
        main.mcp_gateway = original_gateway
        main.connector_registry = original_registry


def _demo_student() -> dict:
    main.learning_logger.seed_demo_identity()
    course_id = main.learning_logger.default_course_id()
    if not main.learning_logger.documents(course_id=course_id):
        _save_indexed_document(
            course_id,
            main.learning_logger.default_class_id(),
            "demo-baseline.md",
            "5G NAS 认证流程包括注册、鉴权和安全模式；关键风险包括重放、降级与上下文绑定。",
        )
    return main.learning_logger.user_by_username("student_demo")


def _demo_teacher() -> dict:
    main.learning_logger.seed_demo_identity()
    return main.learning_logger.user_by_username("teacher_demo")


def _login_headers(client: TestClient, username: str, password: str = "demo123456") -> dict[str, str]:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    if response.status_code == 401 and password == "demo123456":
        password = f"{username}-changed-123"
        response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    payload = response.json()
    token = payload["access_token"]
    if payload.get("must_change_password"):
        changed = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": password, "new_password": f"{username}-changed-123"},
        )
        assert changed.status_code == 200
        token = changed.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_course_bundle(slug: str, *, teacher_username: str, student_username: str | None = None) -> dict[str, int]:
    from backend.auth.password import hash_password

    teacher_id = main.learning_logger.upsert_user(
        username=teacher_username,
        display_name=teacher_username,
        password_hash=hash_password("demo123456"),
        role="teacher",
        must_change_password=False,
    )
    student_id = None
    if student_username:
        student_id = main.learning_logger.upsert_user(
            username=student_username,
            display_name=student_username,
            password_hash=hash_password("demo123456"),
            role="student",
            must_change_password=False,
        )
    with main.learning_logger._connect() as conn:
        cursor = conn.execute(
            "INSERT INTO courses (slug, name, description, assessment_method) VALUES (?, ?, '', '')",
            (slug, slug),
        )
        course_id = int(cursor.lastrowid)
        cursor = conn.execute(
            "INSERT INTO classes (course_id, slug, name, term, teacher_id) VALUES (?, ?, ?, '2026', ?)",
            (course_id, f"{slug}-class", f"{slug} class", teacher_id),
        )
        class_id = int(cursor.lastrowid)
        conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (class_id, teacher_id))
        if student_id is not None:
            conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (class_id, student_id))
    return {"course_id": course_id, "class_id": class_id, "teacher_id": teacher_id, "student_id": student_id or 0}


def _save_indexed_document(course_id: int, class_id: int, filename: str, text: str) -> int:
    document = Document(
        source_path=main.KNOWLEDGE_BASE_DIR / f"course-{course_id}" / filename,
        filename=filename,
        file_type=".md",
        title=filename,
        text=text,
        pages=[Page(text=text, page=1, section="Scope", source_type="markdown")],
    )
    record = main.learning_logger.save_document(
        document=document,
        content_hash=sha256_bytes(text.encode("utf-8")),
        chunks=[{"chunk_text": text, "page": 1, "section": "Scope", "source_type": "md"}],
        byte_size=len(text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
        class_id=class_id,
    )
    main.learning_logger.set_document_publication(int(record["id"]), published=True, actor_id=1)
    return int(record["id"])


def test_chat_endpoint_records_question():
    main.learning_logger.clear()
    user = _demo_student()
    _save_indexed_document(
        main.learning_logger.default_course_id(),
        main.learning_logger.default_class_id(),
        "demo-chat.md",
        "5G NAS 认证流程的关键风险点包括重放、降级和上下文绑定。",
    )

    result = main.chat(
        ChatRequest(
            question="5G NAS 认证流程有哪些关键风险点？",
            conversation_id="conv-1",
        ),
        current_user=user,
    )

    assert result["abstained"] is False
    assert result["citations"]
    assert result["citations"][0]["source_id"].startswith("S")
    assert result["citations"][0]["chunk_id"] is not None
    assert result["citations"][0]["section"]
    assert result["tool"] is None
    assert result["run_id"]
    assert result["run_status"] == "COMPLETED"
    assert result["current_step"] >= 1
    assert result["requires_user_input"] is False
    assert result["conversation_id"] == "conv-1"
    assert result["question_id"] is not None
    assert result["trace_id"]
    assert result["provider"]
    assert result["model"]
    assert isinstance(result["fallback_used"], bool)
    assert isinstance(result["tool_calls"], list)
    analytics = main.teacher_analytics()
    assert analytics["total_questions"] == 1
    assert sum(analytics["module_distribution"].values()) == 1


def test_practice_guide_endpoint_records_question():
    main.learning_logger.clear()

    result = main.practice_guide(
        PracticeGuideRequest(topic="我想做基于区块链的隐蔽通信，怎么设计实验？")
    )

    assert result["intent"] == "practice_guide"
    assert "研究目标" in result["answer"]
    assert main.teacher_faq()["items"][0]["count"] == 1


def test_anonymous_practice_cannot_impersonate_a_student():
    main.learning_logger.clear()

    result = main.practice_guide(
        PracticeGuideRequest(topic="Public practice demo", user_id="2"),
        current_user=None,
    )

    assert result["provider"] == "local-template"
    assert result["delivery_mode"] == "offline_fallback"
    assert main.teacher_analytics()["recent_questions"][0]["user_id"] == "anonymous_demo"


def test_authenticated_practice_uses_agent_run():
    main.learning_logger.clear()

    result = main.practice_guide(
        PracticeGuideRequest(topic="Create a 5G NAS practice guide"),
        current_user=_demo_student(),
    )

    assert result["run_id"]
    assert result["trace_id"]
    assert result["current_step"] >= 1
    assert result["provider"]


def test_chat_routes_practice_intent_and_refuses_incomplete_evidence():
    main.learning_logger.clear()

    result = main.chat(
        ChatRequest(question="Create a practice guide for blockchain covert communication"),
        current_user=_demo_student(),
    )

    assert result["intent"] == "practice_guide"
    assert result["tool"] is None
    assert result["abstained"] is True
    assert result["stop_reason"] == "PRACTICE_EVIDENCE_INCOMPLETE"


def test_authenticated_practice_guide_binds_all_fields_and_six_steps():
    main.learning_logger.clear()
    bundle = _create_course_bundle(
        "practice-grounded",
        teacher_username="practice-teacher",
        student_username="practice-student",
    )
    material = (
        "F8-DIAG-PRACTICE-A 的研究目标是比较正常 5G NAS 认证与重放检测在课程隔离日志中的差异。"
        "F8-DIAG-PRACTICE-A 的方法设计是使用同一字段集合分别标注正常认证样本与重放检测样本，再比较事件序列和告警结果。"
        "F8-DIAG-PRACTICE-A 的工具选择是离线日志解析器、表格统计工具和本地可视化工具。"
        "F8-DIAG-PRACTICE-A 的实验步骤一是准备课程A隔离日志；步骤二是标注正常认证；"
        "步骤三是标注重放检测；步骤四是运行离线解析；步骤五是比较结果；步骤六是保存可复核记录。"
        "F8-DIAG-PRACTICE-A 的结果分析只比较认证事件数量、检测命中情况和误报情况，不推断资料外结论。"
        "F8-DIAG-PRACTICE-A 的创新点是把正常认证与重放检测放入同一受控字段框架中进行可复核比较。"
        "F8-DIAG-PRACTICE-A 的风险与注意事项是仅使用课程A隔离样本，不接触公网目标，不收集真实用户数据。"
    )
    document_id = _save_indexed_document(
        bundle["course_id"],
        bundle["class_id"],
        "practice-grounded.md",
        material,
    )
    user = main.learning_logger.user_by_username("practice-student")

    result = main.practice_guide(
        PracticeGuideRequest(
            topic=(
                "请仅根据课程A中包含 F8-DIAG-PRACTICE-A 的资料生成六步实践指导；"
                "每个事实均须能在资料中找到依据，不足之处请明确拒答，不要补充资料外事实。"
            ),
            course_id=bundle["course_id"],
            class_id=bundle["class_id"],
        ),
        current_user=user,
    )

    assert result["run_status"] == "COMPLETED"
    assert result["abstained"] is False
    assert result["answer"].count("evidence quote:") == 12
    assert all(f"{index}." in result["answer"] for index in range(1, 7))
    assert result["citations"]
    assert {citation["document_id"] for citation in result["citations"]} == {document_id}
    assert all(citation["chunk_id"] is not None for citation in result["citations"])
    assert all(len(citation["snippet"]) <= 243 for citation in result["citations"])
    assert "不收集真实用户数据" not in result["citations"][0]["snippet"]


def test_indexed_upload_exact_question_keeps_correct_document_and_chunk_binding():
    main.learning_logger.clear()
    bundle = _create_course_bundle(
        "upload-exact",
        teacher_username="upload-teacher",
        student_username="upload-student",
    )
    material = "F8-DIAG-UPLOAD-A：课程A的受控日志样本仅用于比较正常认证与重放检测，不得用于公网目标。"
    document_id = _save_indexed_document(
        bundle["course_id"],
        bundle["class_id"],
        "upload-exact.md",
        material,
    )
    user = main.learning_logger.user_by_username("upload-student")

    result = main.chat(
        ChatRequest(
            question="请仅根据刚上传的课程A资料，原样复述包含 F8-DIAG-UPLOAD-A 的完整一句，并显示引用和来源。",
            course_id=bundle["course_id"],
            class_id=bundle["class_id"],
        ),
        current_user=user,
    )

    assert result["run_status"] == "COMPLETED"
    assert material in result["answer"]
    assert result["citations"]
    assert result["citations"][0]["document_id"] == document_id
    assert result["citations"][0]["chunk_id"] is not None


def test_retrieval_logs_are_persisted():
    main.learning_logger.clear()

    result = main.chat(ChatRequest(question="5G NAS 认证流程有哪些关键风险点？"), current_user=_demo_student())
    analytics = main.teacher_analytics()
    question_id = main.learning_logger.latest_question_id()
    assert question_id is not None
    logs = main.learning_logger.retrieval_logs_for_question(question_id)

    assert analytics["total_questions"] == 1
    assert len(logs) == len(result["citations"])
    assert logs[0]["source"]


def test_tools_endpoint():
    tools = main.tools()
    names = {tool["name"] for tool in tools}

    base_tools = {
        "course_retriever",
        "paper_search",
        "vuln_lookup",
        "standard_lookup",
        "mermaid_generator",
    }
    real_tools = {
        "paper_search_real",
        "vuln_lookup_real",
        "standard_metadata_lookup",
        "github_project_search",
    }
    assert base_tools.issubset(names)
    assert real_tools.issubset(names)
    assert all("status" in tool for tool in tools)


def test_tools_http_endpoint_uses_lifespan_mcp_gateway():
    original_gateway = main.mcp_gateway
    original_gateway.stop()
    gateway = MCPClientGateway()
    main.mcp_gateway = gateway
    try:
        with TestClient(main.app) as client:
            response = client.get("/api/tools")
            assert response.status_code == 200
            payload = response.json()
            assert gateway.start_count == 1
            assert gateway.is_available is True
            names = {tool["name"] for tool in payload}
            assert {
                "course_retriever",
                "paper_search",
                "vuln_lookup",
                "standard_lookup",
                "mermaid_generator",
            }.issubset(names)
            assert {
                "paper_search_real",
                "vuln_lookup_real",
                "standard_metadata_lookup",
                "github_project_search",
            }.issubset(names)
            assert all(tool["status"] == "mcp" for tool in payload)
            assert all(tool["mcp_available"] is True for tool in payload)
        assert gateway.is_available is False
    finally:
        gateway.stop()
        original_gateway.stop()
        # This test owns and stops the process-global gateway.  Restore a fresh
        # restartable instance so later lifespan tests do not inherit the
        # deliberately stopped transport from this test's setup.
        main.mcp_gateway = MCPClientGateway()


def test_agent_run_get_endpoint_and_error_contract():
    user = _demo_student()
    result = main.chat(ChatRequest(question="5G NAS 认证流程有哪些关键风险点？"), current_user=user)

    run_payload = main.get_agent_run(result["run_id"], current_user=user)

    assert run_payload["run_id"] == result["run_id"]
    assert run_payload["status"] == "COMPLETED"
    assert run_payload["steps"]
    assert run_payload["traces"]

    with TestClient(main.app) as client:
        main.learning_logger.seed_demo_identity()
        response = client.get(
            "/api/agent/runs/not-found",
            headers=_login_headers(client, "student_demo"),
        )
    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "RUN_NOT_FOUND"
    assert payload["run_id"] == "not-found"


def test_continue_endpoint_resumes_requires_input_run():
    provider = SequenceProvider(
        [
            ActionDecision(
                action=AgentAction.ASK_USER,
                arguments={"question": "请补充实验环境"},
                reason="信息不足",
            )
        ]
    )
    orchestrator = AgentOrchestrator(
        storage=main.learning_logger,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        mcp_gateway=main.mcp_gateway,
    )
    demo_user = _demo_student()
    paused = orchestrator.run(
        "我想设计实验",
        user_id=str(demo_user["id"]),
        course_id=main.learning_logger.default_course_id(),
        class_id=main.learning_logger.default_class_id(),
    )

    resumed = main.continue_agent_run(
        paused["run_id"],
        AgentContinueRequest(user_input="使用本地日志和模拟 NAS 流程", conversation_id="conv-resume"),
        current_user=_demo_student(),
    )

    assert resumed["run_id"] == paused["run_id"]
    assert resumed["run_status"] in {"COMPLETED", "ABSTAINED"}
    assert resumed["conversation_id"] == "conv-resume"
    assert resumed["question_id"] is not None


def test_formal_chat_recognizes_local_sample_intent_without_executing_it():
    result = main.chat(ChatRequest(question="有没有可以作为漏洞检测样例的 CVE？"), current_user=_demo_student())

    assert result["intent"] == "vuln_lookup"
    assert result["tool"] is None
    assert all(call["tool_name"] != "vuln_lookup" for call in result["tool_calls"])


def test_template_http_diagram_request_makes_real_mcp_call():
    main.learning_logger.clear()
    main.learning_logger.seed_demo_identity()

    with TestClient(main.app) as client:
        headers = _login_headers(client, "student_demo")
        response = client.post(
            "/api/chat",
            headers=headers,
            json={"question": "请生成 MCP-RAG 课程智能体的 Mermaid 架构图"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["tool"]["name"] == "mermaid_generator"
    assert result["tool_calls"]
    assert result["tool_calls"][0]["tool_name"] == "mermaid_generator"
    assert bool(result["tool_calls"][0]["fallback_used"]) is False
    assert "```mermaid" in result["answer"]


def test_template_http_regular_nas_question_does_not_force_tool_call():
    main.learning_logger.clear()
    main.learning_logger.seed_demo_identity()
    _save_indexed_document(
        main.learning_logger.default_course_id(),
        main.learning_logger.default_class_id(),
        "nas-risk.md",
        "5G NAS 认证流程的关键风险点包括重放、降级和上下文绑定。",
    )

    with TestClient(main.app) as client:
        headers = _login_headers(client, "student_demo")
        response = client.post(
            "/api/chat",
            headers=headers,
            json={"question": "5G NAS 认证流程有哪些关键风险点？"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["run_status"] == "COMPLETED"
    assert result["tool"] is None
    assert result["tool_calls"] == []
    assert result["citations"]


def test_explicit_nonexistent_paper_request_abstains_without_local_course_citations():
    main.learning_logger.clear()
    main.learning_logger.seed_demo_identity()

    with TestClient(main.app) as client:
        headers = _login_headers(client, "student_demo")
        response = client.post(
            "/api/chat",
            headers=headers,
            json={"question": "paper DefinitelyNonexistentCourseAgentPaperTitleXYZ with verified DOI"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["run_status"] == "ABSTAINED"
    assert result["abstained"] is True
    assert result["stop_reason"] == "PAPER_EVIDENCE_REQUIRED"
    assert result["citations"] == []
    assert "缺少必需的可验证证据" in result["answer"]


def test_upload_rejects_bad_suffix():
    async def run_check():
        upload = UploadFile(filename="bad.xlsx", file=BytesIO(b"demo"))
        try:
            await main.upload_knowledge(upload)
        except Exception as exc:
            assert getattr(exc, "status_code", None) == 400
        else:
            raise AssertionError("bad upload should fail")

    asyncio.run(run_check())


def test_upload_records_document_chunks_and_duplicate_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    main.learning_logger.clear()
    content = "# 上传测试\n\n这是一段用于切片和 hash 去重的课程资料。".encode("utf-8")

    async def run_check():
        first = UploadFile(filename="sample.md", file=BytesIO(content))
        first_result = await main.upload_knowledge(first)

        second = UploadFile(filename="sample-copy.md", file=BytesIO(content))
        second_result = await main.upload_knowledge(second)
        return first_result, second_result

    first_result, second_result = asyncio.run(run_check())
    documents = main.learning_logger.documents()
    chunks = main.learning_logger.chunks_for_document(first_result.document_id)
    jobs = main.learning_logger.index_jobs(first_result.document_id)

    course_dir = tmp_path / "knowledge_base" / f"course-{main.learning_logger.default_course_id()}"
    assert (course_dir / "sample.md").exists()
    assert not (course_dir / "sample-copy.md").exists()

    assert first_result.saved is True
    assert first_result.duplicate is False
    assert first_result.content_hash
    assert first_result.chunk_count >= 1
    assert len(documents) == 1
    assert documents[0]["content_hash"] == first_result.content_hash
    assert len(chunks) == first_result.chunk_count
    assert "hash 去重" in chunks[0]["chunk_text"]
    assert chunks[0]["page"] == 1
    assert chunks[0]["section"] == "上传测试"
    assert chunks[0]["source_type"] == "md"
    assert jobs
    assert jobs[0]["status"] == "PENDING"
    assert jobs[0]["error_message"] is None

    assert second_result.saved is False
    assert second_result.duplicate is True
    assert second_result.document_id == first_result.document_id
    assert second_result.content_hash == first_result.content_hash


def test_system_status_does_not_expose_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-test-key")

    payload = main.system_status()

    assert payload["deepseek_configured"] is True
    assert "secret-test-key" not in str(payload)
    assert payload["sqlite"]["status"] in {"ok", "error"}
    assert payload["chroma"]["status"] in {"ok", "error"}
    assert payload["mcp"]["status"] in {"ok", "fallback"}


def test_knowledge_documents_reindex_and_delete_syncs_vector_store(tmp_path, monkeypatch):
    class FakeVectorStore:
        deleted: list[int] = []
        added: list[dict] = []

        def __init__(self):
            pass

        def delete_by_document(self, document_id: int) -> None:
            self.deleted.append(document_id)

        def add_chunks(self, chunks: list[dict]) -> int:
            self.added.extend(chunks)
            return len(chunks)

    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setattr(main, "ChromaVectorStore", FakeVectorStore)
    main.learning_logger.clear()
    content = "# 二阶段索引\n\nPDF 和 DOCX 上传后也应进入统一知识库索引。".encode("utf-8")

    async def run_upload():
        upload = UploadFile(filename="phase2.md", file=BytesIO(content))
        return await main.upload_knowledge(upload)

    upload_result = asyncio.run(run_upload())
    document_id = int(upload_result.document_id)

    listed = main.list_knowledge_documents()
    assert listed["documents"][0]["id"] == document_id
    assert listed["documents"][0]["jobs"]

    reindexed = main.reindex_knowledge_document(document_id, current_user=_demo_teacher())
    assert reindexed["status"] == "INDEXED"
    assert reindexed["chunk_count"] == upload_result.chunk_count
    assert FakeVectorStore.deleted == [document_id]
    assert {int(item["id"]) for item in FakeVectorStore.added} == {
        int(chunk["id"]) for chunk in main.learning_logger.chunks_for_document(document_id)
    }

    deleted = main.delete_knowledge_document(document_id, current_user=_demo_teacher())
    assert deleted["status"] == "DELETED"
    assert main.learning_logger.document(document_id) is None
    assert main.learning_logger.chunks_for_document(document_id) == []
    assert FakeVectorStore.deleted[-1] == document_id


def test_http_upload_pending_then_reindex_makes_document_retrievable(tmp_path, monkeypatch):
    from backend.rag import hybrid_retriever as hybrid_module
    from backend.rag.vector_store import ChromaVectorStore

    class TestChromaVectorStore(ChromaVectorStore):
        def __init__(self):
            super().__init__(
                persist_dir=tmp_path / "chroma",
                collection_name="p1_4_chunks",
                embedding_function=KeywordEmbedding(),
            )

    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setattr(main, "ChromaVectorStore", TestChromaVectorStore)
    monkeypatch.setattr(hybrid_module, "ChromaVectorStore", TestChromaVectorStore)
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-api-endpoints-32-bytes")
    main.learning_logger.clear()
    main.learning_logger.seed_demo_identity()
    content = (
        "# P1-4 Retrieval Gate\n\n"
        "UniqueP14Term NAS authentication indexing gate evidence. "
        "This uploaded document should be searchable only after INDEXED."
    ).encode("utf-8")

    with TestClient(main.app) as client:
        teacher_headers = _login_headers(client, "teacher_demo")
        student_headers = _login_headers(client, "student_demo")
        upload_response = client.post(
            "/api/knowledge/upload",
            headers=teacher_headers,
            files={"file": ("p1-4.md", content, "text/markdown")},
        )

        assert upload_response.status_code == 200
        uploaded = upload_response.json()
        document_id = int(uploaded["document_id"])
        assert uploaded["index_status"] == "PENDING"
        assert uploaded["job_id"]

        listed_before = client.get("/api/knowledge/documents", headers=teacher_headers).json()
        document_before = next(item for item in listed_before["documents"] if int(item["id"]) == document_id)
        assert document_before["index_status"] == "PENDING"
        assert document_before["jobs"][0]["status"] == "PENDING"

        before_query = client.post(
            "/api/chat",
            headers=student_headers,
            json={"question": "UniqueP14Term indexing gate evidence"},
        ).json()
        assert all(citation["document_id"] != document_id for citation in before_query["citations"])

        reindex_response = client.post(
            f"/api/knowledge/documents/{document_id}/reindex",
            headers=teacher_headers,
        )
        assert reindex_response.status_code == 200
        assert reindex_response.json()["status"] == "INDEXED"

        listed_after = client.get("/api/knowledge/documents", headers=teacher_headers).json()
        document_after = next(item for item in listed_after["documents"] if int(item["id"]) == document_id)
        assert document_after["index_status"] == "INDEXED"
        assert document_after["jobs"][0]["status"] == "INDEXED"

        after_query = client.post(
            "/api/chat",
            headers=student_headers,
            json={"question": "UniqueP14Term NAS authentication indexing gate evidence"},
        ).json()

        assert after_query["citations"] == []
        publish_response = client.post(
            f"/api/knowledge/documents/{document_id}/publish",
            headers=teacher_headers,
        )
        assert publish_response.status_code == 200
        assert publish_response.json()["status"] == "PUBLISHED"

        after_publish_query = client.post(
            "/api/chat",
            headers=student_headers,
            json={"question": "UniqueP14Term NAS authentication indexing gate evidence"},
        ).json()

    assert after_publish_query["citations"]
    assert after_publish_query["citations"][0]["document_id"] == document_id


def test_reindex_retry_success_closes_pending_jobs_and_keeps_failure_history(tmp_path, monkeypatch):
    class FlakyVectorStore:
        attempts = 0

        def __init__(self):
            pass

        def delete_by_document(self, document_id: int):
            return None

        def add_chunks(self, chunks: list[dict]):
            FlakyVectorStore.attempts += 1
            if FlakyVectorStore.attempts == 1:
                raise RuntimeError("temporary vector failure")
            return len(chunks)

    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setattr(main, "ChromaVectorStore", FlakyVectorStore)
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-api-endpoints-32-bytes")
    main.learning_logger.clear()
    main.learning_logger.seed_demo_identity()

    with TestClient(main.app) as client:
        teacher_headers = _login_headers(client, "teacher_demo")
        upload = client.post(
            "/api/knowledge/upload",
            headers=teacher_headers,
            files={"file": ("retry.md", b"# Retry\n\nretry indexing content", "text/markdown")},
        )
        document_id = int(upload.json()["document_id"])

        failed = client.post(f"/api/knowledge/documents/{document_id}/reindex", headers=teacher_headers)
        succeeded = client.post(f"/api/knowledge/documents/{document_id}/reindex", headers=teacher_headers)
        listed = client.get("/api/knowledge/documents", headers=teacher_headers).json()

    document = next(item for item in listed["documents"] if int(item["id"]) == document_id)
    jobs = document["jobs"]
    active_statuses = {"PENDING", "PARSING", "CHUNKING", "EMBEDDING"}

    assert failed.status_code == 500
    assert succeeded.status_code == 200
    assert document["index_status"] == "INDEXED"
    assert document["latest_job_status"] == "INDEXED"
    assert all(job["status"] not in active_statuses for job in jobs)
    assert any(job["status"] == "FAILED" and "temporary vector failure" in job["error_message"] for job in jobs)
    assert any(job["status"] == "INDEXED" for job in jobs)


def test_delete_document_keeps_sqlite_when_vector_delete_fails(tmp_path, monkeypatch):
    class FailingVectorStore:
        def __init__(self):
            pass

        def delete_by_document(self, document_id: int) -> None:
            raise RuntimeError("vector store locked")

    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setattr(main, "ChromaVectorStore", FailingVectorStore)
    main.learning_logger.clear()
    content = "# 删除保护\n\n向量删除失败时不应先删除 SQLite 文档。".encode("utf-8")

    async def run_upload():
        upload = UploadFile(filename="delete-guard.md", file=BytesIO(content))
        return await main.upload_knowledge(upload)

    upload_result = asyncio.run(run_upload())
    document_id = int(upload_result.document_id)

    with pytest.raises(Exception) as exc_info:
        main.delete_knowledge_document(document_id, current_user=_demo_teacher())

    assert getattr(exc_info.value, "status_code", None) == 500
    assert main.learning_logger.document(document_id) is not None
    assert main.learning_logger.chunks_for_document(document_id)


def test_feedback_and_analytics_metrics_are_aggregated():
    main.learning_logger.clear()
    user = _demo_student()
    result = main.chat(ChatRequest(question="5G NAS 认证流程有哪些关键风险点？"), current_user=user)

    feedback = main.feedback(
        main.FeedbackRequest(
            question_id=int(result["question_id"]),
            rating=5,
            helpful=True,
            comment="很有帮助",
        ),
        current_user=user,
    )
    analytics = main.teacher_analytics()

    assert feedback["saved"] is True
    assert feedback["question_id"] == result["question_id"]
    assert analytics["model_metrics"]["total_calls"] >= 1
    assert 0 <= analytics["model_metrics"]["success_rate"] <= 1
    assert analytics["agent_metrics"]["average_steps"] >= 1
    assert analytics["retrieval_metrics"]["knowledge_hit_rate"] > 0
    assert analytics["feedback_metrics"]["count"] == 1
    assert analytics["feedback_metrics"]["average_rating"] == 5
    assert all(len(item["question"]) <= 63 for item in analytics["recent_questions"])


def test_model_metric_aggregation_counts_latency_success_and_fallback():
    main.learning_logger.clear()
    run_id = "metric-run"
    step_id = "metric-step"
    main.learning_logger.create_agent_run(
        run_id=run_id,
        user_id="teacher_metric",
        goal="统计口径测试",
        intent="course_qa",
        status="RUNNING",
        max_steps=3,
        max_tool_calls=3,
        state={"current_step": 1, "tool_call_count": 0},
    )
    main.learning_logger.create_agent_step(
        run_id=run_id,
        step_id=step_id,
        step_number=1,
        action="planning",
        status="RUNNING",
    )
    main.learning_logger.log_model_call(
        run_id=run_id,
        step_id=step_id,
        question_id=None,
        provider="deepseek",
        model="deepseek-flash",
        purpose="decide_next_action",
        success=True,
        latency_ms=100,
    )
    main.learning_logger.log_model_call(
        run_id=run_id,
        step_id=step_id,
        question_id=None,
        provider="template",
        model="template",
        purpose="generate_answer_fallback",
        success=False,
        latency_ms=300,
        fallback_used=True,
    )

    metrics = main.teacher_analytics()["model_metrics"]

    assert metrics["total_calls"] == 2
    assert metrics["success_rate"] == 0.5
    assert metrics["average_latency_ms"] == 200
    assert metrics["fallback_count"] == 1


def test_feedback_rejects_unknown_question():
    with pytest.raises(Exception) as exc_info:
        main.feedback(main.FeedbackRequest(question_id=9999, rating=3), current_user=_demo_student())

    assert getattr(exc_info.value, "status_code", None) == 404


def test_management_endpoints_require_teacher_role(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-api-endpoints-32-bytes")
    main.learning_logger.clear()
    main.learning_logger.seed_demo_identity()
    content = "# 权限测试\n\n只有教师可以删除知识库文档。".encode("utf-8")

    async def run_upload():
        upload = UploadFile(filename="permission.md", file=BytesIO(content))
        return await main.upload_knowledge(upload)

    upload_result = asyncio.run(run_upload())
    document_id = int(upload_result.document_id)

    with TestClient(main.app) as client:
        list_response = client.get("/api/knowledge/documents")
        upload_response = client.post(
            "/api/knowledge/upload",
            files={"file": ("anonymous.md", b"# anonymous", "text/markdown")},
        )
        delete_response = client.delete(f"/api/knowledge/documents/{document_id}")
        reindex_response = client.post(f"/api/knowledge/documents/{document_id}/reindex")
        student_headers = _login_headers(client, "student_demo")
        student_list_response = client.get("/api/knowledge/documents", headers=student_headers)
        student_upload_response = client.post(
            "/api/knowledge/upload",
            headers=student_headers,
            files={"file": ("student.md", b"# student", "text/markdown")},
        )
        student_delete_response = client.delete(f"/api/knowledge/documents/{document_id}", headers=student_headers)
        student_reindex_response = client.post(f"/api/knowledge/documents/{document_id}/reindex", headers=student_headers)
        teacher_headers = _login_headers(client, "teacher_demo")
        teacher_upload_response = client.post(
            "/api/knowledge/upload",
            headers=teacher_headers,
            files={"file": ("teacher.md", b"# teacher\n\nallowed upload", "text/markdown")},
        )

    assert list_response.status_code == 401
    assert upload_response.status_code == 401
    assert delete_response.status_code == 401
    assert reindex_response.status_code == 401
    assert student_list_response.status_code == 403
    assert student_upload_response.status_code == 403
    assert student_delete_response.status_code == 403
    assert student_reindex_response.status_code == 403
    assert teacher_upload_response.status_code == 200


def test_knowledge_management_is_scoped_to_teacher_courses(tmp_path, monkeypatch):
    class FakeVectorStore:
        def __init__(self):
            pass

        def delete_by_document(self, document_id: int) -> None:
            return None

        def add_chunks(self, chunks: list[dict]) -> int:
            return len(chunks)

    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setattr(main, "ChromaVectorStore", FakeVectorStore)
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-api-endpoints-32-bytes")
    main.learning_logger.clear()
    first = _create_course_bundle("scope-course-a", teacher_username="teacher_a")
    second = _create_course_bundle("scope-course-b", teacher_username="teacher_b")

    with TestClient(main.app) as client:
        teacher_a = _login_headers(client, "teacher_a")
        teacher_b = _login_headers(client, "teacher_b")
        upload_a = client.post(
            "/api/knowledge/upload",
            headers=teacher_a,
            files={"file": ("shared.md", b"# Course A\n\nOnly teacher A should manage this.", "text/markdown")},
        )
        upload_b = client.post(
            "/api/knowledge/upload",
            headers=teacher_b,
            files={"file": ("shared.md", b"# Course B\n\nOnly teacher B should manage this.", "text/markdown")},
        )

        assert upload_a.status_code == 200
        assert upload_b.status_code == 200
        doc_a = int(upload_a.json()["document_id"])
        doc_b = int(upload_b.json()["document_id"])
        assert doc_a != doc_b
        assert (tmp_path / "knowledge_base" / f"course-{first['course_id']}" / "shared.md").exists()
        assert (tmp_path / "knowledge_base" / f"course-{second['course_id']}" / "shared.md").exists()

        list_a = client.get("/api/knowledge/documents", headers=teacher_a)
        list_b = client.get("/api/knowledge/documents", headers=teacher_b)
        cross_reindex = client.post(f"/api/knowledge/documents/{doc_b}/reindex", headers=teacher_a)
        own_reindex = client.post(f"/api/knowledge/documents/{doc_a}/reindex", headers=teacher_a)
        cross_delete = client.delete(f"/api/knowledge/documents/{doc_b}", headers=teacher_a)
        own_delete = client.delete(f"/api/knowledge/documents/{doc_b}", headers=teacher_b)

    assert list_a.status_code == 200
    assert {int(item["id"]) for item in list_a.json()["documents"]} == {doc_a}
    assert list_b.status_code == 200
    assert {int(item["id"]) for item in list_b.json()["documents"]} == {doc_b}
    assert cross_reindex.status_code == 403
    assert own_reindex.status_code == 200
    assert cross_delete.status_code == 403
    assert own_delete.status_code == 200
    assert main.learning_logger.document(doc_a) is not None
    assert main.learning_logger.document(doc_b) is None


def test_teacher_without_course_cannot_manage_knowledge(tmp_path, monkeypatch):
    from backend.auth.password import hash_password

    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-api-endpoints-32-bytes")
    main.learning_logger.clear()
    main.learning_logger.seed_demo_identity()
    main.learning_logger.upsert_user(
        username="teacher_no_course",
        display_name="Teacher No Course",
        password_hash=hash_password("demo123456"),
        role="teacher",
        must_change_password=False,
    )

    with TestClient(main.app) as client:
        headers = _login_headers(client, "teacher_no_course")
        list_response = client.get("/api/knowledge/documents", headers=headers)
        upload_response = client.post(
            "/api/knowledge/upload",
            headers=headers,
            files={"file": ("blocked.md", b"# blocked\n\nno course", "text/markdown")},
        )

    assert list_response.status_code == 403
    assert upload_response.status_code == 403


def test_same_hash_can_be_uploaded_to_different_courses(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-api-endpoints-32-bytes")
    main.learning_logger.clear()
    first = _create_course_bundle("hash-course-a", teacher_username="hash_teacher_a")
    second = _create_course_bundle("hash-course-b", teacher_username="hash_teacher_b")
    content = b"# Shared Content\n\nSame bytes should be course scoped."

    with TestClient(main.app) as client:
        teacher_a = _login_headers(client, "hash_teacher_a")
        teacher_b = _login_headers(client, "hash_teacher_b")
        upload_a = client.post(
            "/api/knowledge/upload",
            headers=teacher_a,
            files={"file": ("same-hash.md", content, "text/markdown")},
        )
        upload_b = client.post(
            "/api/knowledge/upload",
            headers=teacher_b,
            files={"file": ("same-hash.md", content, "text/markdown")},
        )

    assert upload_a.status_code == 200
    assert upload_b.status_code == 200
    assert upload_a.json()["duplicate"] is False
    assert upload_b.json()["duplicate"] is False
    assert upload_a.json()["document_id"] != upload_b.json()["document_id"]
    assert (tmp_path / "knowledge_base" / f"course-{first['course_id']}" / "same-hash.md").read_bytes() == content
    assert (tmp_path / "knowledge_base" / f"course-{second['course_id']}" / "same-hash.md").read_bytes() == content


def test_no_class_student_cannot_read_course_scoped_content(tmp_path, monkeypatch):
    from backend.auth.password import hash_password

    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-api-endpoints-32-bytes")
    main.learning_logger.clear()
    main.learning_logger.seed_demo_identity()
    main.learning_logger.upsert_user(
        username="student_no_class",
        display_name="Student No Class",
        password_hash=hash_password("demo123456"),
        role="student",
        must_change_password=False,
    )

    with TestClient(main.app) as client:
        headers = _login_headers(client, "student_no_class")
        chat = client.post(
            "/api/chat",
            headers=headers,
            json={"question": "5G NAS authentication"},
        )
        portal = client.get("/api/resources/portal", headers=headers)
        innovation = client.post(
            "/api/innovation/sessions",
            headers=headers,
            json={"topic": "5G NAS", "current_plan": "Evaluate authentication ideas."},
        )

    assert chat.status_code == 403
    assert portal.status_code == 403
    assert innovation.status_code == 403


def test_chat_and_resource_portal_are_limited_to_student_course(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-api-endpoints-32-bytes")
    main.learning_logger.clear()
    course_a = _create_course_bundle("student-scope-a", teacher_username="scope_teacher_a", student_username="scope_student_a")
    course_b = _create_course_bundle("student-scope-b", teacher_username="scope_teacher_b", student_username="scope_student_b")
    doc_a = _save_indexed_document(
        course_a["course_id"],
        course_a["class_id"],
        "course-a-scope.md",
        "CourseAUniqueMarker alpha NAS evidence belongs only to course A.",
    )
    doc_b = _save_indexed_document(
        course_b["course_id"],
        course_b["class_id"],
        "course-b-scope.md",
        "CourseBUniqueMarker beta NAS evidence belongs only to course B.",
    )

    with TestClient(main.app) as client:
        student_a = _login_headers(client, "scope_student_a")
        own_chat = client.post(
            "/api/chat",
            headers=student_a,
            json={"question": "CourseAUniqueMarker alpha NAS evidence"},
        )
        cross_chat = client.post(
            "/api/chat",
            headers=student_a,
            json={"question": "CourseBUniqueMarker beta NAS evidence"},
        )
        portal = client.get("/api/resources/portal", headers=student_a, params={"resource_type": "course_document", "page_size": 50})

    assert own_chat.status_code == 200
    own_payload = own_chat.json()
    assert any(int(citation["document_id"]) == doc_a for citation in own_payload["citations"])
    assert all(int(citation["document_id"]) != doc_b for citation in own_payload["citations"])

    assert cross_chat.status_code == 200
    cross_payload = cross_chat.json()
    assert "CourseBUniqueMarker" not in str(cross_payload)
    assert all(int(citation.get("document_id") or 0) != doc_b for citation in cross_payload["citations"])

    assert portal.status_code == 200
    portal_text = str(portal.json())
    assert "course-a-scope.md" in portal_text
    assert "course-b-scope.md" not in portal_text


def test_upload_rejects_oversize_and_same_filename_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setattr(main, "DEFAULT_MAX_FILE_BYTES", 16)
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-api-endpoints-32-bytes")
    main.learning_logger.clear()
    main.learning_logger.seed_demo_identity()

    with TestClient(main.app) as client:
        teacher_headers = _login_headers(client, "teacher_demo")
        oversize_response = client.post(
            "/api/knowledge/upload",
            headers=teacher_headers,
            files={"file": ("large.md", b"# title\n\nthis content is too large", "text/markdown")},
        )

    assert oversize_response.status_code == 413
    assert main.learning_logger.documents() == []

    monkeypatch.setattr(main, "DEFAULT_MAX_FILE_BYTES", 1024)
    first_content = b"# Same Name\n\nfirst version"
    second_content = b"# Same Name\n\nsecond version"

    with TestClient(main.app) as client:
        teacher_headers = _login_headers(client, "teacher_demo")
        first_response = client.post(
            "/api/knowledge/upload",
            headers=teacher_headers,
            files={"file": ("same-name.md", first_content, "text/markdown")},
        )
        conflict_response = client.post(
            "/api/knowledge/upload",
            headers=teacher_headers,
            files={"file": ("same-name.md", second_content, "text/markdown")},
        )

    course_id = main.learning_logger.default_course_id()
    saved_path = tmp_path / "knowledge_base" / f"course-{course_id}" / "same-name.md"
    documents = main.learning_logger.documents()

    assert first_response.status_code == 200
    assert conflict_response.status_code == 409
    assert saved_path.read_bytes() == first_content
    assert len([document for document in documents if document["filename"] == "same-name.md"]) == 1
    assert documents[0]["content_hash"] == first_response.json()["content_hash"]


def test_upload_failed_parse_records_failed_index_job(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    main.learning_logger.clear()

    async def run_check():
        upload = UploadFile(filename="bad.pdf", file=BytesIO(b"not a real pdf"))
        try:
            await main.upload_knowledge(upload)
        except Exception as exc:
            assert getattr(exc, "status_code", None) == 400
        else:
            raise AssertionError("bad PDF should fail")

    asyncio.run(run_check())
    jobs = main.learning_logger.index_jobs()

    assert len(jobs) == 1
    assert jobs[0]["status"] == "FAILED"
    assert "解析 PDF 失败" in jobs[0]["error_message"]
    assert main.learning_logger.documents() == []

