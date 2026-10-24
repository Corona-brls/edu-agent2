from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.observation import tool_result_to_observation
from backend.agent.schemas import ActionDecision, AgentAction, AgentIntent, ToolSpec
from backend.agent.state import AgentRunStatus, AgentState, state_from_record
from backend.llm.schemas import EvidenceItem, GroundedSpanAnswer, LLMRequest, LLMResponse
from backend.llm.template_provider import TemplateProvider
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import (
    HybridRetriever,
    RetrievalResponse,
    query_fact_fingerprint,
    retrieval_snippet,
)
from backend.resources.query_intent import LookupInputError
from backend.storage.logger import LearningLogger, sha256_bytes
from backend.tools.mcp_client import MCPClientGateway
from mcp_server.schemas import ToolResult
from mcp_server.tools import TOOL_DEFINITIONS

from tests.test_agent_orchestration import (
    ConstraintRecordingRetriever,
    FakeRetriever,
    PaperContractProvider,
    Project28PracticeProvider,
    SequenceProvider,
    VersionedExactPaperGateway,
    R7_PROJECT20_QUESTION,
    PROJECT28_PRACTICE_ANCHOR,
    _paper_item,
    _result,
    save_project28_practice_document,
)


def _template_ask_trigger() -> str:
    constants = TemplateProvider.decide_next_action.__code__.co_consts
    return next(
        item
        for item in constants
        if isinstance(item, str) and TemplateProvider().decide_next_action(item, []).action == AgentAction.ASK_USER
    )


def _login_headers(client: TestClient, username: str) -> dict[str, str]:
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


def _save_indexed_document(storage: LearningLogger, course_id: int, class_id: int, text: str) -> int:
    document = Document(
        source_path=f"knowledge_base/course-{course_id}/s1.md",
        filename="s1.md",
        file_type=".md",
        title="S1 NAS Guidance",
        text=text,
        pages=[Page(text=text, page=1, section="NAS Lab", source_type="markdown")],
    )
    record = storage.save_document(
        document=document,
        content_hash=sha256_bytes(text.encode("utf-8")),
        chunks=[{"chunk_text": text, "page": 1, "section": "NAS Lab", "source_type": "md"}],
        byte_size=len(text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
        class_id=class_id,
    )
    storage.set_document_publication(int(record["id"]), published=True, actor_id=1)
    return int(record["id"])


def _seed_default_published_document(storage: LearningLogger) -> None:
    _save_indexed_document(
        storage,
        storage.default_course_id(),
        storage.default_class_id(),
        "5G NAS authentication evidence",
    )


def _persist_paused_state(storage: LearningLogger, state: AgentState) -> None:
    storage.create_agent_run(
        run_id=state.run_id,
        user_id=state.user_id,
        goal=state.user_goal,
        intent=state.intent,
        status=state.status.value,
        max_steps=state.max_steps,
        max_tool_calls=state.max_tool_calls,
        state=state.model_dump(mode="json"),
        course_id=state.course_id,
        class_id=state.class_id,
    )


def test_continue_exact_arxiv_base_identifier_accepts_persisted_versioned_evidence(tmp_path):
    storage = LearningLogger(tmp_path / "exact-paper-resume.db")
    question = (
        "请查询 arXiv:1706.03762 的公开论文元数据，给出标题、作者线索和 "
        "arXiv 官方链接；不要用本地样例代替。"
    )
    state = AgentState(
        user_id="paper-resume-user",
        user_goal=question,
        intent=AgentIntent.PAPER_SEARCH.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        pending_question="是否继续？",
        required_sources=["paper"],
        completed_sources=["paper"],
        evidence=[
            _paper_item(
                "S1",
                title="Attention Is All You Need",
                abstract="A sequence transduction model based entirely on attention.",
                authors=["Ashish Vaswani", "Noam Shazeer"],
                external_id="1706.03762v7",
            )
        ],
    )
    _persist_paused_state(storage, state)
    provider = PaperContractProvider(complete_first=True)
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.continue_run(state.run_id, "继续查询", user_id="paper-resume-user")

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert len(provider.structured_requests) == 1
    assert len(result["citations"]) == 1
    assert result["citations"][0]["external_id"] == "1706.03762v7"
    assert result["citations"][0]["source_url"] == "https://arxiv.org/abs/1706.03762v7"
    assert storage.tool_call_rows(state.run_id) == []


def test_continue_exact_arxiv_renderer_keeps_author_initial_on_one_verified_line(tmp_path):
    storage = LearningLogger(tmp_path / "exact-paper-author-initial-resume.db")
    question = (
        "请查询 arXiv:1706.03762 的公开论文元数据，给出标题、作者线索和 "
        "arXiv 官方链接；不要用本地样例代替。"
    )
    state = AgentState(
        user_id="paper-author-initial-resume-user",
        user_goal=question,
        intent=AgentIntent.PAPER_SEARCH.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        pending_question="是否继续？",
        required_sources=["paper"],
        completed_sources=["paper"],
        evidence=[
            _paper_item(
                "S1",
                title="Attention Is All You Need",
                abstract="A sequence transduction model based entirely on attention.",
                authors=["Ashish Vaswani", "Aidan N. Gomez"],
                external_id="1706.03762v7",
            )
        ],
    )
    _persist_paused_state(storage, state)
    provider = PaperContractProvider(complete_first=True)
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.continue_run(
        state.run_id,
        "继续查询",
        user_id="paper-author-initial-resume-user",
    )

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert result["stop_reason"] == "ANSWERED"
    assert len(provider.structured_requests) == 1
    assert storage.tool_call_rows(state.run_id) == []


def test_continue_multisource_control_backtracks_caller_paper_target_once(tmp_path, monkeypatch):
    monkeypatch.setitem(
        TOOL_DEFINITIONS,
        "paper_search_real",
        TOOL_DEFINITIONS["paper_search_real"].model_copy(update={"enabled": True}),
    )
    storage = LearningLogger(tmp_path / "r7-project20-resume.db")
    state = AgentState(
        user_id="r7-project20-resume-user",
        user_goal=R7_PROJECT20_QUESTION,
        intent=AgentIntent.MULTI_SOURCE.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        pending_question="请确认授权边界。",
        required_sources=["paper"],
        available_tools=[
            ToolSpec(
                name="paper_search_real",
                description="controlled exact paper lookup",
                network=True,
                read_only=True,
                enabled=True,
            )
        ],
    )
    state.add_user_message(R7_PROJECT20_QUESTION)
    _persist_paused_state(storage, state)
    planner = SequenceProvider(
        [
            ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name="paper_search_real",
                arguments={"query": "planner composite", "source": "semantic_scholar"},
                reason="collect the required paper",
            ),
            ActionDecision(
                action=AgentAction.ANSWER,
                reason="paper evidence is ready",
                task_complete=True,
            ),
        ]
    )
    answer_provider = PaperContractProvider(complete_first=True)
    gateway = VersionedExactPaperGateway(authors=["Ashish Vaswani", "Aidan N. Gomez"])
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=planner,
        answer_provider=answer_provider,
        fallback_provider=answer_provider,
        retriever=FakeRetriever(),
        mcp_gateway=gateway,
    )

    result = orchestrator.continue_run(
        state.run_id,
        "仅做授权防御性教学资料梳理；分别列来源身份和可核对链接，不做真实系统攻击。",
        user_id=state.user_id,
    )

    assert len(gateway.calls) == 1
    assert gateway.calls[0][1]["query"] == "arXiv:1706.03762"
    assert gateway.calls[0][1]["source"] == "auto"
    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert result["citations"][0]["external_id"] == "1706.03762v7"


def test_continue_multisource_only_cve_does_not_call_old_paper_or_project(tmp_path, monkeypatch):
    for tool_name in ("paper_search_real", "vuln_lookup_real", "github_project_search"):
        monkeypatch.setitem(
            TOOL_DEFINITIONS,
            tool_name,
            TOOL_DEFINITIONS[tool_name].model_copy(update={"enabled": True}),
        )

    class CveOnlyGateway:
        def __init__(self):
            self.calls = []

        def get_agent_tool_catalog(self, role="student"):
            return [
                {
                    "name": name,
                    "description": "controlled public lookup",
                    "input_schema": TOOL_DEFINITIONS[name].input_schema,
                    "risk": "LOW",
                    "network": True,
                    "read_only": True,
                    "enabled": True,
                }
                for name in ("paper_search_real", "vuln_lookup_real", "github_project_search")
            ]

        def call_tool(self, name, arguments, *, run_id, step_id, step_number=1, role="student"):
            self.calls.append((name, dict(arguments)))
            assert name == "vuln_lookup_real"
            resource = {
                "external_id": "CVE-2021-44228",
                "resource_type": "vulnerability",
                "source_name": "nvd",
                "title": "CVE-2021-44228",
                "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
                "abstract": "The NVD record identifies CVE-2021-44228 for defensive review.",
                "content_hash": "r7a1-cve-hash",
                "fetched_at": "2026-10-23T00:00:00Z",
                "origin_kind": "remote",
                "delivery_mode": "fresh_cache",
                "provenance_verified": True,
            }
            result = ToolResult(success=True, data={"resources": [resource]}, metadata={"tool_name": name})

            class Call:
                pass

            call = Call()
            call.result = result
            call.observation = tool_result_to_observation(
                result,
                step_number=step_number,
                tool_name=name,
                fallback_used=False,
                run_id=run_id,
                step_id=step_id,
                arguments=arguments,
                latency_ms=1,
            )
            call.fallback_used = False
            return call

    storage = LearningLogger(tmp_path / "r7a1-cve-only-resume.db")
    state = AgentState(
        user_id="r7a1-cve-only-user",
        user_goal=R7_PROJECT20_QUESTION,
        intent=AgentIntent.MULTI_SOURCE.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        pending_question="请确认新的来源范围。",
        required_sources=["paper", "vulnerability", "project"],
        completed_sources=["paper", "project"],
        evidence=[
            _paper_item(
                "S1",
                title="Attention Is All You Need",
                abstract="A historical paper result that is no longer in the active scope.",
                authors=["Ashish Vaswani", "Aidan N. Gomez"],
                external_id="1706.03762v7",
            ),
            EvidenceItem(
                source_id="S2",
                text="Open5GS is an open source 5G core project.",
                source="github",
                title="open5gs/open5gs",
                source_url="https://github.com/open5gs/open5gs",
                external_id="80611216",
                resource_type="project",
                origin_kind="remote",
                delivery_mode="fresh_cache",
                content_hash="historical-project-hash",
                tool_name="github_project_search",
                trusted=True,
            ),
        ],
        available_tools=[
            ToolSpec(
                name=name,
                description="controlled public lookup",
                network=True,
                read_only=True,
                enabled=True,
            )
            for name in ("paper_search_real", "vuln_lookup_real", "github_project_search")
        ],
    )
    state.add_user_message(R7_PROJECT20_QUESTION)
    _persist_paused_state(storage, state)
    storage.create_agent_step(
        run_id=state.run_id,
        step_id="historical-paper-step",
        step_number=1,
        action="call_tool",
        status=AgentRunStatus.REQUIRES_INPUT.value,
        trace_id=state.trace_id,
    )
    storage.log_tool_call(
        run_id=state.run_id,
        step_id="historical-paper-step",
        question_id=None,
        tool_name="paper_search_real",
        arguments={"query": "arXiv:1706.03762"},
        result={"status": "historical_success"},
        success=True,
        latency_ms=1,
        fallback_used=False,
    )
    planner = SequenceProvider(
        [
            ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name="paper_search_real",
                arguments={"query": "planner stale paper target"},
                reason="stale planner request must be overridden by caller scope",
            ),
            ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name="github_project_search",
                arguments={"query": "planner stale project target"},
                reason="a second stale planner request must not escape",
            ),
        ]
    )
    gateway = CveOnlyGateway()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=planner,
        answer_provider=TemplateProvider(),
        fallback_provider=TemplateProvider(),
        retriever=FakeRetriever(),
        mcp_gateway=gateway,
    )

    result = orchestrator.continue_run(
        state.run_id,
        "请只查 CVE-2021-44228 并在论文背景中说明其教学意义。",
        user_id=state.user_id,
    )

    assert [name for name, _arguments in gateway.calls] == ["vuln_lookup_real"]
    assert gateway.calls[0][1]["query"] == "CVE-2021-44228"
    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert {item["resource_type"] for item in result["citations"]} == {"vulnerability"}
    persisted = state_from_record(storage.agent_run(state.run_id))
    assert persisted.required_sources == ["vulnerability"]
    assert persisted.intent == AgentIntent.VULN_LOOKUP.value
    assert {item.resource_type for item in persisted.evidence} == {"vulnerability"}
    assert persisted.completed_sources == ["vulnerability"]
    assert [tool.name for tool in persisted.available_tools] == ["vuln_lookup_real"]
    assert [row["tool_name"] for row in storage.tool_call_rows(state.run_id)] == [
        "paper_search_real",
        "vuln_lookup_real",
    ]


def test_continue_project20_safety_control_keeps_three_source_scope_once(tmp_path, monkeypatch):
    tool_names = ("paper_search_real", "vuln_lookup_real", "github_project_search")
    for tool_name in tool_names:
        monkeypatch.setitem(
            TOOL_DEFINITIONS,
            tool_name,
            TOOL_DEFINITIONS[tool_name].model_copy(update={"enabled": True}),
        )

    class ThreeSourceGateway:
        def __init__(self):
            self.calls = []

        def get_agent_tool_catalog(self, role="student"):
            return [
                {
                    "name": name,
                    "description": "controlled public lookup",
                    "input_schema": TOOL_DEFINITIONS[name].input_schema,
                    "risk": "LOW",
                    "network": True,
                    "read_only": True,
                    "enabled": True,
                }
                for name in tool_names
            ]

        def call_tool(self, name, arguments, *, run_id, step_id, step_number=1, role="student"):
            self.calls.append((name, dict(arguments)))
            resources = {
                "paper_search_real": {
                    "external_id": "1706.03762v7",
                    "resource_type": "paper",
                    "source_name": "arxiv",
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762v7",
                    "authors": ["Ashish Vaswani", "Aidan N. Gomez"],
                    "authors_hash_covered": True,
                    "abstract": "A sequence transduction model based entirely on attention.",
                    "content_hash": "resume-paper-hash",
                },
                "vuln_lookup_real": {
                    "external_id": "CVE-2021-44228",
                    "resource_type": "vulnerability",
                    "source_name": "nvd",
                    "title": "CVE-2021-44228",
                    "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
                    "abstract": "The NVD record identifies CVE-2021-44228 for defensive review.",
                    "content_hash": "resume-cve-hash",
                },
                "github_project_search": {
                    "external_id": "80611216",
                    "resource_type": "project",
                    "source_name": "github",
                    "title": "open5gs/open5gs",
                    "url": "https://github.com/open5gs/open5gs",
                    "abstract": "Open5GS is an open source 5G core project.",
                    "raw_metadata": {
                        "full_name": "open5gs/open5gs",
                        "private": False,
                        "visibility": "public",
                    },
                    "content_hash": "resume-project-hash",
                },
            }
            resource = {
                **resources[name],
                "fetched_at": "2026-10-23T00:00:00Z",
                "origin_kind": "remote",
                "delivery_mode": "fresh_cache",
                "provenance_verified": True,
            }
            result = ToolResult(
                success=True,
                data={"resources": [resource]},
                metadata={"tool_name": name},
            )

            class Call:
                pass

            call = Call()
            call.result = result
            call.observation = tool_result_to_observation(
                result,
                step_number=step_number,
                tool_name=name,
                fallback_used=False,
                run_id=run_id,
                step_id=step_id,
                arguments=arguments,
                latency_ms=1,
            )
            call.fallback_used = False
            return call

    storage = LearningLogger(tmp_path / "r7a2-three-source-resume.db")
    state = AgentState(
        user_id="r7a2-three-source-user",
        user_goal=R7_PROJECT20_QUESTION,
        intent=AgentIntent.MULTI_SOURCE.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        pending_question="请确认防御性授权边界。",
        required_sources=["paper", "vulnerability", "project"],
        available_tools=[
            ToolSpec(
                name=name,
                description="controlled public lookup",
                network=True,
                read_only=True,
                enabled=True,
            )
            for name in tool_names
        ],
    )
    state.add_user_message(R7_PROJECT20_QUESTION)
    _persist_paused_state(storage, state)
    planner = SequenceProvider(
        [
            ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name="paper_search_real",
                arguments={"query": "planner composite", "source": "semantic_scholar"},
                reason="paper",
            ),
            ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name="vuln_lookup_real",
                arguments={"query": "planner wrong CVE"},
                reason="vulnerability",
            ),
            ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name="github_project_search",
                arguments={"query": "planner wrong repo"},
                reason="project",
            ),
            ActionDecision(
                action=AgentAction.ANSWER,
                reason="all required sources are ready",
                task_complete=True,
            ),
        ]
    )
    gateway = ThreeSourceGateway()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=planner,
        answer_provider=TemplateProvider(),
        fallback_provider=TemplateProvider(),
        retriever=FakeRetriever(),
        mcp_gateway=gateway,
    )

    result = orchestrator.continue_run(
        state.run_id,
        "仅做授权防御性教学资料梳理；分别列来源身份和可核对链接，不做真实系统攻击或三来源错误关联。",
        user_id=state.user_id,
    )

    assert [name for name, _arguments in gateway.calls] == list(tool_names)
    assert gateway.calls[0][1]["query"] == "arXiv:1706.03762"
    assert gateway.calls[1][1]["query"] == "CVE-2021-44228"
    assert gateway.calls[2][1]["query"] == "open5gs/open5gs"
    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert {item["resource_type"] for item in result["citations"]} == {
        "paper",
        "vulnerability",
        "project",
    }
    paragraphs = [item for item in result["answer"].splitlines() if item.strip()]
    assert len(paragraphs) == 3
    assert [item.split("：", 1)[0] for item in paragraphs] == [
        "论文教学背景",
        "NVD漏洞教学背景",
        "GitHub项目教学背景",
    ]
    assert all("| evidence quote:" not in item for item in paragraphs)
    persisted = state_from_record(storage.agent_run(state.run_id))
    assert persisted.required_sources == ["paper", "vulnerability", "project"]


@pytest.mark.parametrize(
    "latest_query",
    [
        "改为查询 GitHub open5gs/open5gs/issues",
        "改为查询 https://github.com/a/b#",
        "改为查询 https://github.com/a/b?#",
        "改为查询 https://github.com:/a/b",
        "改为查询 github.com/a/b#",
        "改为查询 github.com/a/b?#",
        "改为查询 GitHub https://github.com/a/b#",
        "改为查询 GitHub https://github.com/a/b?#",
        "改为查询 GitHub github.com/a/b#",
        "改为查询 GitHub github.com/a/b?#",
        "改为查询 GitHub https://github.com:/a/b",
    ],
)
def test_continue_malformed_latest_github_repository_does_not_inherit_old_exact_or_call_gateway(
    tmp_path,
    monkeypatch,
    latest_query,
):
    tool_name = "github_project_search"
    monkeypatch.setitem(
        TOOL_DEFINITIONS,
        tool_name,
        TOOL_DEFINITIONS[tool_name].model_copy(update={"enabled": True}),
    )

    class ZeroCallGateway:
        def __init__(self):
            self.calls = []

        def get_agent_tool_catalog(self, role="student"):
            return [
                {
                    "name": tool_name,
                    "description": "controlled project lookup",
                    "input_schema": TOOL_DEFINITIONS[tool_name].input_schema,
                    "risk": "LOW",
                    "network": True,
                    "read_only": True,
                    "enabled": True,
                }
            ]

        def call_tool(self, name, arguments, **kwargs):
            self.calls.append((name, dict(arguments)))
            raise AssertionError("Malformed latest caller repository must fail before gateway")

    storage = LearningLogger(tmp_path / "malformed-project-resume.db")
    original_question = "请核对 GitHub open5gs/open5gs"
    state = AgentState(
        user_id="malformed-project-user",
        user_goal=original_question,
        intent=AgentIntent.PROJECT_SEARCH.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        pending_question="请确认要继续查询的仓库。",
        required_sources=["project"],
        available_tools=[
            ToolSpec(
                name=tool_name,
                description="controlled project lookup",
                network=True,
                read_only=True,
                enabled=True,
            )
        ],
    )
    state.add_user_message(original_question)
    _persist_paused_state(storage, state)
    planner = SequenceProvider(
        [
            ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name=tool_name,
                arguments={"query": "planner old safe target"},
                reason="latest caller target must replace the old target",
            )
        ]
    )
    gateway = ZeroCallGateway()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=planner,
        answer_provider=TemplateProvider(),
        fallback_provider=TemplateProvider(),
        retriever=FakeRetriever(),
        mcp_gateway=gateway,
    )

    with pytest.raises(LookupInputError) as rejected:
        orchestrator.continue_run(
            state.run_id,
            latest_query,
            user_id=state.user_id,
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert gateway.calls == []
    assert storage.tool_call_rows(state.run_id) == []


def test_continue_practice_adds_current_same_document_support_without_retrieval_or_tool(tmp_path):
    storage = LearningLogger(tmp_path / "project28-practice-resume.db")
    context = storage.seed_demo_identity()
    course_id = int(context["course_id"])
    class_id = int(context["class_id"])
    document_id, chunks = save_project28_practice_document(storage, tmp_path, course_id)
    state = AgentState(
        user_id=str(context["student_ids"][0]),
        course_id=course_id,
        class_id=class_id,
        user_goal="请基于课程资料设计5G NAS合成日志中的重放检测评估。",
        intent=AgentIntent.PRACTICE_GUIDE.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        pending_question="请确认继续生成实践指导。",
        required_sources=["course"],
        completed_sources=["course"],
        evidence=[
            EvidenceItem(
                source_id="S1",
                text=PROJECT28_PRACTICE_ANCHOR,
                source="project28-practice.md",
                title="5G NAS安全专题方法卡",
                document_id=document_id,
                chunk_id=int(chunks[0]["id"]),
                page=1,
                section="5G NAS实践",
                source_type="md",
                score=0.9,
                resource_type="course",
                origin_kind="local_document",
                delivery_mode="local_rag",
                content_hash=str(chunks[0]["chunk_hash"]),
                trusted=True,
            )
        ],
    )
    _persist_paused_state(storage, state)
    provider = Project28PracticeProvider(
        [
            ActionDecision(
                action=AgentAction.ANSWER,
                reason="current course evidence is ready",
                task_complete=True,
            )
        ]
    )
    retriever = FakeRetriever()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=retriever,
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.continue_run(
        state.run_id,
        "继续生成，不增加新资料。",
        user_id=str(context["student_ids"][0]),
        course_id=course_id,
        class_id=class_id,
    )

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert retriever.calls == []
    assert storage.tool_call_rows(state.run_id) == []
    assert len(provider.practice_requests) == 1
    request = provider.practice_requests[0]
    assert any(
        item.document_id == document_id
        and item.chunk_id == int(chunks[1]["id"])
        and "输入编号" in item.text
        for item in request.evidence
    )
    assert int(chunks[1]["id"]) in {item["chunk_id"] for item in result["citations"]}


def test_continue_preserves_existing_explicit_course_constraint_over_external_keywords(tmp_path):
    storage = LearningLogger(tmp_path / "constraint-resume.db")
    state = AgentState(
        user_id="2",
        course_id=1,
        class_id=1,
        user_goal="请只根据《目标教材》回答",
        intent=AgentIntent.COURSE_QA.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        pending_question="请补充",
        required_sources=["course"],
        course_document_constraint_title="目标教材",
        allowed_course_document_ids=[17],
        allowed_course_document_hashes={17: "document-hash"},
        allowed_course_chunk_hashes={23: "chunk-hash"},
    )
    _persist_paused_state(storage, state)
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=SequenceProvider(),
        answer_provider=SequenceProvider(),
        fallback_provider=SequenceProvider(),
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    resumed = orchestrator._load_or_create_state(
        user_goal="请查询 RFC 9000 和 CVE",
        user_id="2",
        course_id=1,
        class_id=1,
        run_id=state.run_id,
        supplemental_input="补充说明 RFC 9000 和 CVE 仅是教材术语",
        intent_override=None,
    )

    assert resumed.intent == AgentIntent.COURSE_QA.value
    assert resumed.required_sources == ["course"]
    assert resumed.course_document_constraint_title == "目标教材"
    assert resumed.allowed_course_document_ids == [17]


def _paused_kb9_constraint_state(storage: LearningLogger, *, question: str):
    course_id = storage.default_course_id()
    class_id = storage.default_class_id()
    text = (
        "# 网络安全实验设计与评价方法卡\n"
        + ("课程审阅边界与目录说明，不构成问题答案。" * 18)
        + "\n"
        "建立基线组与处理组，只改变预先声明的独立变量。\n"
        "固定软件版本、数据划分、随机种子和运行资源；至少重复运行三次并报告波动。\n"
        "分类任务至少报告混淆矩阵、精确率、召回率和样本量。"
    )
    document = Document(
        source_path=Path("knowledge_base/kb9.md"),
        filename="kb9.md",
        file_type="md",
        title="网络安全实验设计与评价方法卡",
        text=text,
        pages=[Page(text=text, page=1, section="方法设计", source_type="markdown")],
    )
    saved = storage.save_document(
        document=document,
        content_hash=sha256_bytes(text.encode("utf-8")),
        chunks=[{"chunk_text": text, "page": 1, "section": "方法设计", "source_type": "md"}],
        byte_size=len(text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
        class_id=class_id,
    )
    document_id = int(saved["id"])
    storage.set_document_publication(document_id, published=True, actor_id=1)
    document_row = storage.document(document_id)
    chunk = storage.chunks_for_document(document_id)[0]
    current_chunk = next(
        row
        for row in storage.document_chunks(course_id=course_id)
        if int(row["id"]) == int(chunk["id"])
    )
    snippet = retrieval_snippet(text, query=question)
    state = AgentState(
        user_id="2",
        course_id=course_id,
        class_id=class_id,
        user_goal=question,
        intent=AgentIntent.COURSE_QA.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        current_step=1,
        pending_question="请补充",
        required_sources=["course"],
        completed_sources=["course"],
        course_document_constraint_title="网络安全实验设计与评价方法卡",
        allowed_course_document_ids=[document_id],
        allowed_course_document_hashes={document_id: str(document_row["content_hash"])},
        allowed_course_chunk_hashes={int(chunk["id"]): str(chunk["chunk_hash"])},
        evidence=[
            EvidenceItem(
                source_id="S1",
                text=snippet,
                source="kb9.md",
                title="网络安全实验设计与评价方法卡",
                document_id=document_id,
                chunk_id=int(chunk["id"]),
                page=1,
                section="方法设计",
                source_type="md",
                score=0.9,
                resource_type="course",
                origin_kind="local_document",
                delivery_mode="local_rag",
                content_hash=str(current_chunk["content_hash"]),
                trusted=True,
            )
        ],
    )
    return state, text, document_id, int(chunk["id"])


def _paused_cross_chunk_constraint_state(storage: LearningLogger, *, question: str):
    course_id = storage.default_course_id()
    class_id = storage.default_class_id()
    chunks = [
        "# 跨块目标教材\n事实A：建立基线组与对照组，只改变一个独立变量。",
        (
            "背景边界用于说明隔离环境角色范围和安全约束但不构成本题答案" * 16
            + "，事实B：实验记录必须保存每轮输入和输出，评价方法采用误报率和样本量，"
            "实验设计保持同一批数据。 The experiment log preserves each run and the "
            "evaluation method compares false-positive rate and sample size."
        ),
        "背景附录：课堂术语索引，不包含本题的补充事实。",
    ]
    text = "\n\n".join(chunks)
    document = Document(
        source_path=Path("knowledge_base/cross-chunk-target.md"),
        filename="cross-chunk-target.md",
        file_type="md",
        title="跨块目标教材",
        text=text,
        pages=[Page(text=text, page=1, section="实验方法", source_type="markdown")],
    )
    saved = storage.save_document(
        document=document,
        content_hash=sha256_bytes(text.encode("utf-8")),
        chunks=[
            {
                "chunk_text": chunk_text,
                "page": 1,
                "section": f"实验方法{index}",
                "source_type": "md",
            }
            for index, chunk_text in enumerate(chunks, 1)
        ],
        byte_size=len(text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
        class_id=class_id,
    )
    document_id = int(saved["id"])
    storage.set_document_publication(document_id, published=True, actor_id=1)
    document_row = storage.document(document_id)
    stored_chunks = storage.chunks_for_document(document_id)
    current_chunks = {
        int(row["id"]): row
        for row in storage.document_chunks(course_id=course_id)
        if int(row["document_id"]) == document_id
    }
    chunk_a = stored_chunks[0]
    current_a = current_chunks[int(chunk_a["id"])]
    state = AgentState(
        user_id="2",
        course_id=course_id,
        class_id=class_id,
        user_goal=question,
        intent=AgentIntent.COURSE_QA.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        current_step=1,
        pending_question="请补充",
        required_sources=["course"],
        completed_sources=["course"],
        course_document_constraint_title="跨块目标教材",
        course_document_snippet_query=question,
        allowed_course_document_ids=[document_id],
        allowed_course_document_hashes={document_id: str(document_row["content_hash"])},
        allowed_course_chunk_hashes={
            int(chunk["id"]): str(chunk["chunk_hash"]) for chunk in stored_chunks
        },
        evidence=[
            EvidenceItem(
                source_id="S1",
                text=retrieval_snippet(chunks[0], query=question),
                source="cross-chunk-target.md",
                title="跨块目标教材",
                document_id=document_id,
                chunk_id=int(chunk_a["id"]),
                page=1,
                section="实验方法1",
                source_type="md",
                score=0.9,
                resource_type="course",
                origin_kind="local_document",
                delivery_mode="local_rag",
                content_hash=str(current_a["content_hash"]),
                trusted=True,
            )
        ],
    )
    return state, document_id, [int(chunk["id"]) for chunk in stored_chunks]


class _CrossChunkAnswerProvider(SequenceProvider):
    def __init__(self):
        super().__init__(
            [
                ActionDecision(
                    action=AgentAction.ANSWER,
                    reason="continue with constrained course evidence",
                    task_complete=True,
                ),
                ActionDecision(
                    action=AgentAction.ANSWER,
                    reason="answer with refreshed constrained course evidence",
                    task_complete=True,
                ),
            ]
        )
        self.answer_request = None

    def generate_answer(self, request):
        self.answer_calls += 1
        self.answer_request = request
        supporting = next(
            (
                item
                for item in request.evidence
                if any(
                    marker in item.text.casefold()
                    for marker in ("误报率", "实验记录", "评价方法", "experiment log", "evaluation method")
                )
            ),
            request.evidence[0],
        )
        return LLMResponse(
            text=f"{supporting.text} [{supporting.source_id}]",
            provider=self.provider_name,
            model=self.model_name,
        )


class _BrokenVector:
    def query(self, text, top_k=5):
        raise RuntimeError("offline fallback")


class _CountingHybrid(HybridRetriever):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0
        self.allowed_document_ids = []

    def retrieve(self, *args, **kwargs):
        self.calls += 1
        self.allowed_document_ids.append(kwargs.get("allowed_document_ids"))
        return super().retrieve(*args, **kwargs)


def _cross_chunk_orchestrator(storage: LearningLogger):
    provider = _CrossChunkAnswerProvider()
    retriever = _CountingHybrid(storage=storage, vector_store=_BrokenVector(), min_score=0.0)
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=retriever,
        mcp_gateway=MCPClientGateway(),
    )
    return orchestrator, provider, retriever


def test_continue_explicit_textbook_new_fact_in_unseen_chunk_retrieves_once(tmp_path):
    storage = LearningLogger(tmp_path / "cross-chunk-resume.db")
    storage.seed_demo_identity()
    question = "请只根据《跨块目标教材》解释基线与对照的关系。"
    state, document_id, chunk_ids = _paused_cross_chunk_constraint_state(
        storage,
        question=question,
    )
    long_support = next(
        row for row in storage.chunks_for_document(document_id) if int(row["id"]) == chunk_ids[1]
    )["chunk_text"]
    assert len(long_support) > 240
    assert long_support.index("实验记录") > 240
    _persist_paused_state(storage, state)
    orchestrator, provider, retriever = _cross_chunk_orchestrator(storage)

    result = orchestrator.continue_run(
        state.run_id,
        "再说明误报率和样本量。",
        user_id="2",
        course_id=state.course_id,
        class_id=state.class_id,
    )

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert retriever.calls == 1
    assert retriever.allowed_document_ids == [frozenset({document_id})]
    assert provider.answer_request is not None
    assert any(
        item.document_id == document_id
        and item.chunk_id == chunk_ids[1]
        and "误报率" in item.text
        and "样本量" in item.text
        for item in provider.answer_request.evidence
    )
    assert any(citation["chunk_id"] == chunk_ids[1] for citation in result["citations"])
    persisted = AgentState.model_validate(storage.agent_run(state.run_id)["state"])
    assert "误报率和样本量" in persisted.course_document_snippet_query
    assert storage.tool_call_rows(state.run_id) == []


def test_continue_explicit_textbook_unsupported_new_fact_retrieves_once_and_abstains(tmp_path):
    storage = LearningLogger(tmp_path / "unsupported-cross-chunk-resume.db")
    storage.seed_demo_identity()
    question = "请只根据《跨块目标教材》解释基线与对照的关系。"
    state, document_id, _chunk_ids = _paused_cross_chunk_constraint_state(
        storage,
        question=question,
    )
    _persist_paused_state(storage, state)
    orchestrator, provider, retriever = _cross_chunk_orchestrator(storage)

    result = orchestrator.continue_run(
        state.run_id,
        "再说明量子纠错预算。",
        user_id="2",
        course_id=state.course_id,
        class_id=state.class_id,
    )

    assert result["run_status"] == AgentRunStatus.ABSTAINED, result
    assert result["stop_reason"] == "INSUFFICIENT_EVIDENCE"
    assert retriever.calls == 1
    assert retriever.allowed_document_ids == [frozenset({document_id})]
    assert provider.answer_calls == 0
    assert result["citations"] == []
    assert storage.tool_call_rows(state.run_id) == []


@pytest.mark.parametrize(
    "supplement",
    [
        "请详细一点",
        "请展开说明",
        "换一种说法",
        "请简要总结",
        "说明得更具体一些",
        "please continue",
        "please elaborate",
        "请继续根据《跨块目标教材》回答。",
    ],
)
def test_continue_explicit_textbook_style_only_keeps_evidence_without_retrieval(
    tmp_path,
    supplement,
):
    storage = LearningLogger(tmp_path / "style-only-resume.db")
    storage.seed_demo_identity()
    question = "请只根据《跨块目标教材》解释基线与对照的关系。"
    state, document_id, chunk_ids = _paused_cross_chunk_constraint_state(
        storage,
        question=question,
    )
    _persist_paused_state(storage, state)
    orchestrator, provider, retriever = _cross_chunk_orchestrator(storage)

    result = orchestrator.continue_run(
        state.run_id,
        supplement,
        user_id="2",
        course_id=state.course_id,
        class_id=state.class_id,
    )

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert retriever.calls == 0
    assert provider.answer_calls == 1
    assert provider.answer_request is not None
    assert {(item.document_id, item.chunk_id) for item in provider.answer_request.evidence} == {
        (document_id, chunk_ids[0])
    }
    assert storage.tool_call_rows(state.run_id) == []


@pytest.mark.parametrize(
    "supplement",
    [
        "请详细说明评价方法",
        "请用另一种说法解释实验记录",
        "please elaborate on the evaluation method",
    ],
)
def test_continue_explicit_textbook_mixed_style_and_fact_retrieves_support_once(
    tmp_path,
    supplement,
):
    storage = LearningLogger(tmp_path / "mixed-style-fact-resume.db")
    storage.seed_demo_identity()
    question = "请只根据《跨块目标教材》解释基线与对照的关系。"
    state, document_id, chunk_ids = _paused_cross_chunk_constraint_state(
        storage,
        question=question,
    )
    _persist_paused_state(storage, state)
    orchestrator, provider, retriever = _cross_chunk_orchestrator(storage)

    result = orchestrator.continue_run(
        state.run_id,
        supplement,
        user_id="2",
        course_id=state.course_id,
        class_id=state.class_id,
    )

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert retriever.calls == 1
    assert retriever.allowed_document_ids == [frozenset({document_id})]
    assert provider.answer_request is not None
    support = next(
        item for item in provider.answer_request.evidence if item.chunk_id == chunk_ids[1]
    )
    assert support.document_id == document_id
    assert len(support.text) <= 240
    assert any(marker in support.text.casefold() for marker in ("实验记录", "评价方法", "evaluation method"))
    assert any(citation["chunk_id"] == chunk_ids[1] for citation in result["citations"])
    assert storage.tool_call_rows(state.run_id) == []


@pytest.mark.parametrize(
    ("supplement", "required_marker"),
    [
        ("不要再回答基线与对照，只说明误报率和样本量", "误报率"),
        ("改为只说明实验记录", "实验记录"),
        ("instead explain only the evaluation method", "evaluation method"),
    ],
)
def test_continue_explicit_textbook_replacement_drops_old_facts_and_retrieves_once(
    tmp_path,
    supplement,
    required_marker,
):
    storage = LearningLogger(tmp_path / "replacement-resume.db")
    storage.seed_demo_identity()
    question = "请只根据《跨块目标教材》解释基线与对照的关系。"
    state, document_id, chunk_ids = _paused_cross_chunk_constraint_state(
        storage,
        question=question,
    )
    old_facts = set(query_fact_fingerprint(question))
    _persist_paused_state(storage, state)
    orchestrator, provider, retriever = _cross_chunk_orchestrator(storage)

    result = orchestrator.continue_run(
        state.run_id,
        supplement,
        user_id="2",
        course_id=state.course_id,
        class_id=state.class_id,
    )

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert retriever.calls == 1
    assert retriever.allowed_document_ids == [frozenset({document_id})]
    assert provider.answer_request is not None
    assert {item.chunk_id for item in provider.answer_request.evidence} == {chunk_ids[1]}
    assert required_marker in provider.answer_request.evidence[0].text.casefold()
    assert {citation["chunk_id"] for citation in result["citations"]} == {chunk_ids[1]}
    persisted = AgentState.model_validate(storage.agent_run(state.run_id)["state"])
    replacement_facts = set(query_fact_fingerprint(persisted.course_document_snippet_query))
    assert replacement_facts
    assert replacement_facts.isdisjoint(old_facts)
    assert storage.tool_call_rows(state.run_id) == []


def test_continue_explicit_textbook_unsupported_replacement_does_not_reuse_old_evidence(tmp_path):
    storage = LearningLogger(tmp_path / "unsupported-replacement-resume.db")
    storage.seed_demo_identity()
    question = "请只根据《跨块目标教材》解释基线与对照的关系。"
    state, document_id, _chunk_ids = _paused_cross_chunk_constraint_state(
        storage,
        question=question,
    )
    _persist_paused_state(storage, state)
    orchestrator, provider, retriever = _cross_chunk_orchestrator(storage)

    result = orchestrator.continue_run(
        state.run_id,
        "改为只说明量子纠错预算",
        user_id="2",
        course_id=state.course_id,
        class_id=state.class_id,
    )

    assert result["run_status"] == AgentRunStatus.ABSTAINED, result
    assert result["stop_reason"] == "INSUFFICIENT_EVIDENCE"
    assert retriever.calls == 1
    assert retriever.allowed_document_ids == [frozenset({document_id})]
    assert provider.answer_calls == 0
    assert result["citations"] == []
    assert storage.tool_call_rows(state.run_id) == []


def test_continue_explicit_textbook_control_supplement_keeps_bound_snippet(tmp_path):
    storage = LearningLogger(tmp_path / "kb9-control-resume.db")
    storage.seed_demo_identity()
    question = "请只根据《网络安全实验设计与评价方法卡》解释基线与对照的关系。"
    state, _text, document_id, chunk_id = _paused_kb9_constraint_state(
        storage,
        question=question,
    )
    _persist_paused_state(storage, state)

    class CaptureProvider(SequenceProvider):
        def __init__(self):
            super().__init__(
                [
                    ActionDecision(
                        action=AgentAction.ANSWER,
                        reason="answer from the existing constrained evidence",
                        task_complete=True,
                    )
                ]
            )
            self.answer_request = None

        def generate_answer(self, request):
            self.answer_request = request
            return super().generate_answer(request)

    class CountingHybrid(HybridRetriever):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def retrieve(self, *args, **kwargs):
            self.calls += 1
            return super().retrieve(*args, **kwargs)

    provider = CaptureProvider()
    retriever = CountingHybrid(storage=storage, vector_store=FakeRetriever(), min_score=0.0)
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=retriever,
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.continue_run(
        state.run_id,
        "请继续根据《网络安全实验设计与评价方法卡》回答，不增加新的事实主题。",
        user_id="2",
        course_id=state.course_id,
        class_id=state.class_id,
    )

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert retriever.calls == 0
    assert provider.answer_request is not None
    assert {(item.document_id, item.chunk_id) for item in provider.answer_request.evidence} == {
        (document_id, chunk_id)
    }
    assert storage.tool_call_rows(state.run_id) == []


def test_continue_explicit_textbook_new_concepts_reprojects_once_inside_target(tmp_path):
    storage = LearningLogger(tmp_path / "kb9-changed-resume.db")
    storage.seed_demo_identity()
    question = "请只根据《网络安全实验设计与评价方法卡》解释基线与对照的关系。"
    state, _text, document_id, chunk_id = _paused_kb9_constraint_state(
        storage,
        question=question,
    )
    _persist_paused_state(storage, state)

    class CaptureProvider(SequenceProvider):
        def __init__(self):
            super().__init__(
                [
                    ActionDecision(
                        action=AgentAction.ANSWER,
                        reason="reproject the changed constrained question",
                        task_complete=True,
                    ),
                    ActionDecision(
                        action=AgentAction.ANSWER,
                        reason="answer from the refreshed constrained evidence",
                        task_complete=True,
                    ),
                ]
            )
            self.answer_request = None

        def generate_answer(self, request):
            self.answer_request = request
            return super().generate_answer(request)

    class BrokenVector:
        def query(self, text, top_k=5):
            raise RuntimeError("offline fallback")

    class CountingHybrid(HybridRetriever):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def retrieve(self, *args, **kwargs):
            self.calls += 1
            return super().retrieve(*args, **kwargs)

    provider = CaptureProvider()
    retriever = CountingHybrid(storage=storage, vector_store=BrokenVector(), min_score=0.0)
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=retriever,
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.continue_run(
        state.run_id,
        "再说明评价指标和复现实验记录。",
        user_id="2",
        course_id=state.course_id,
        class_id=state.class_id,
    )

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert retriever.calls == 1
    assert provider.answer_request is not None
    rendered = "".join(item.text for item in provider.answer_request.evidence)
    assert "混淆矩阵、精确率、召回率和样本量" in rendered
    assert "至少重复运行三次" in rendered
    assert {(item.document_id, item.chunk_id) for item in provider.answer_request.evidence} == {
        (document_id, chunk_id)
    }
    assert storage.tool_call_rows(state.run_id) == []


def test_continue_switches_external_run_to_explicit_course_and_drops_external_evidence(tmp_path):
    storage = LearningLogger(tmp_path / "constraint-switch.db")
    state = AgentState(
        user_id="2",
        course_id=1,
        class_id=1,
        user_goal="请查询 RFC 9000",
        intent=AgentIntent.STANDARD_LOOKUP.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        pending_question="请补充",
        required_sources=["standard"],
        completed_sources=["standard"],
        evidence=[
            EvidenceItem(
                source_id="S1",
                text="旧的 RFC 外部证据",
                source="rfc_editor",
                title="RFC 9000",
                external_id="RFC-9000",
                source_url="https://www.rfc-editor.org/info/rfc9000",
                resource_type="standard",
                origin_kind="remote",
                delivery_mode="live",
                content_hash="external-hash",
                tool_name="standard_lookup_real",
                trusted=True,
            )
        ],
    )
    _persist_paused_state(storage, state)
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=SequenceProvider(),
        answer_provider=SequenceProvider(),
        fallback_provider=SequenceProvider(),
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    resumed = orchestrator._load_or_create_state(
        user_goal="请只依据《漏洞、CVE与CVSS基础》回答",
        user_id="2",
        course_id=1,
        class_id=1,
        run_id=state.run_id,
        supplemental_input="请只依据《漏洞、CVE与CVSS基础》回答",
        intent_override=None,
    )

    assert resumed.intent == AgentIntent.COURSE_QA.value
    assert resumed.required_sources == ["course"]
    assert resumed.completed_sources == []
    assert resumed.evidence == []


def test_continue_external_run_retrieves_only_new_explicit_course_document(tmp_path):
    storage = LearningLogger(tmp_path / "constraint-switch-full.db")
    context = storage.seed_demo_identity()
    course_id = int(context["course_id"])
    class_id = int(context["class_id"])
    text = "CVE编号用于唯一标识漏洞，CVSS用于表达评分。"
    document = Document(
        source_path=tmp_path / "vulnerability-course.md",
        filename="vulnerability-course.md",
        file_type="md",
        title="漏洞、CVE与CVSS基础",
        text=text,
        pages=[Page(text=text, page=1, section="漏洞基础", source_type="markdown")],
    )
    saved = storage.save_document(
        document=document,
        content_hash=sha256_bytes(text.encode("utf-8")),
        chunks=[text],
        byte_size=len(text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
    )
    document_id = int(saved["id"])
    storage.set_document_publication(document_id, published=True, actor_id=1)
    chunk_id = int(storage.chunks_for_document(document_id)[0]["id"])
    state = AgentState(
        user_id="2",
        course_id=course_id,
        class_id=class_id,
        user_goal="请查询 RFC 9000",
        intent=AgentIntent.STANDARD_LOOKUP.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        pending_question="请补充",
        required_sources=["standard"],
        completed_sources=["standard"],
        evidence=[
            EvidenceItem(
                source_id="S1",
                text="旧外部证据",
                source="rfc_editor",
                title="RFC 9000",
                external_id="RFC-9000",
                source_url="https://www.rfc-editor.org/info/rfc9000",
                resource_type="standard",
                origin_kind="remote",
                delivery_mode="live",
                content_hash="external-hash",
                tool_name="standard_lookup_real",
                trusted=True,
            )
        ],
    )
    _persist_paused_state(storage, state)
    provider = SequenceProvider(
        [
            ActionDecision(
                action=AgentAction.RETRIEVE,
                arguments={"query": "CVE编号与CVSS评分"},
                reason="retrieve the newly constrained course document",
            )
        ]
    )
    retriever = ConstraintRecordingRetriever(
        [
            _result(
                document_id=document_id,
                chunk_id=chunk_id,
                snippet=text,
                title="漏洞、CVE与CVSS基础",
                source="vulnerability-course.md",
                metadata={"content_hash": storage.document(document_id)["content_hash"]},
            )
        ]
    )
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=retriever,
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.continue_run(
        state.run_id,
        "请只依据《漏洞、CVE与CVSS基础》说明CVE编号与CVSS评分的作用",
        user_id="2",
        course_id=course_id,
        class_id=class_id,
    )

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert result["intent"] == AgentIntent.COURSE_QA.value
    assert retriever.calls[0]["allowed_document_ids"] == {document_id}
    assert {item["document_id"] for item in result["citations"]} == {document_id}
    persisted = storage.agent_run(state.run_id)["state"]
    assert persisted["required_sources"] == ["course"]
    assert {item["resource_type"] for item in persisted["evidence"]} == {"course"}


def test_continue_explicit_course_keyword_fallback_reaches_support_text(tmp_path):
    storage = LearningLogger(tmp_path / "constraint-switch-support.db")
    context = storage.seed_demo_identity()
    course_id = int(context["course_id"])
    class_id = int(context["class_id"])
    query = "目标教材 合成消息序列 防御性实验 记录字段"
    anchor = (
        "# 目标教材\nsource_id target review_state approved 目标教材 当前课程发布 "
        "课程资料 引用 防御性实验 合成消息序列"
    ) * 5
    support = (
        "在隔离的合成消息序列防御性实验中，每条记录应保存输入编号、初始状态、"
        "预期转移、实际转移、拒绝原因和处理时延。"
        + " 补充说明用于离线教学分析，不包含攻击步骤。" * 12
    )
    noise = "天气烹饪诗歌与遥远主题完全无关。" * 20
    text = "\n".join([anchor, support, noise])
    document = Document(
        source_path=tmp_path / "target-long.md",
        filename="target-long.md",
        file_type="md",
        title="目标教材",
        text=text,
        pages=[Page(text=text, page=1, section="目标教材", source_type="markdown")],
    )
    saved = storage.save_document(
        document=document,
        content_hash=sha256_bytes(text.encode("utf-8")),
        chunks=[anchor, support, noise],
        byte_size=len(text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
    )
    document_id = int(saved["id"])
    storage.set_document_publication(document_id, published=True, actor_id=1)
    target_chunks = [int(item["id"]) for item in storage.chunks_for_document(document_id)]
    state = AgentState(
        user_id="2",
        course_id=course_id,
        class_id=class_id,
        user_goal="请查询 RFC 9000",
        intent=AgentIntent.STANDARD_LOOKUP.value,
        status=AgentRunStatus.REQUIRES_INPUT,
        pending_question="请补充",
        required_sources=["standard"],
        completed_sources=["standard"],
        evidence=[
            EvidenceItem(
                source_id="S1",
                text="旧外部证据",
                source="rfc_editor",
                title="RFC 9000",
                external_id="RFC-9000",
                source_url="https://www.rfc-editor.org/info/rfc9000",
                resource_type="standard",
                origin_kind="remote",
                delivery_mode="live",
                content_hash="external-hash",
                tool_name="standard_lookup_real",
                trusted=True,
            )
        ],
    )
    _persist_paused_state(storage, state)

    class BrokenVectorStore:
        def query(self, text, top_k=5):
            raise RuntimeError("chroma unavailable")

    class SupportSelectingProvider(SequenceProvider):
        def __init__(self):
            super().__init__(
                [
                    ActionDecision(
                        action=AgentAction.RETRIEVE,
                        arguments={"query": query},
                        reason="retrieve the newly constrained textbook",
                    )
                ]
            )
            self.answer_request = None

        def generate_answer(self, request):
            self.answer_request = request
            return super().generate_answer(request)

    provider = SupportSelectingProvider()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=HybridRetriever(
            storage=storage,
            vector_store=BrokenVectorStore(),
            min_score=0.18,
        ),
        mcp_gateway=MCPClientGateway(),
    )

    result = orchestrator.continue_run(
        state.run_id,
        "请只依据《目标教材》回答合成消息序列实验应记录哪些字段",
        user_id="2",
        course_id=course_id,
        class_id=class_id,
    )

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert result["fallback_used"] is True
    assert provider.answer_request is not None
    assert any(
        item.chunk_id == target_chunks[1] and "输入编号" in item.text
        for item in provider.answer_request.evidence
    )
    assert any(
        span.get("field") == "text" and "输入编号" in span.get("exact_text", "")
        for span in provider.answer_request.metadata.get("evidence_spans") or []
    )
    assert target_chunks[1] in {item["chunk_id"] for item in result["citations"]}
    persisted = storage.agent_run(state.run_id)["state"]
    assert {item["resource_type"] for item in persisted["evidence"]} == {"course"}


def test_ask_user_pauses_and_resume_keeps_state(tmp_path):
    storage = LearningLogger(tmp_path / "resume.db")
    _seed_default_published_document(storage)
    provider = SequenceProvider(
        [
            ActionDecision(
                action=AgentAction.ASK_USER,
                arguments={"question": "请补充实验环境"},
                reason="信息不足",
            ),
            ActionDecision(
                action=AgentAction.RETRIEVE,
                arguments={"query": "NAS 实验环境"},
                reason="补充后检索",
            ),
        ]
    )
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    first = orchestrator.run("我想设计实验")

    assert first["run_status"] == AgentRunStatus.REQUIRES_INPUT
    assert first["requires_user_input"] is True
    assert first["pending_question"] == "请补充实验环境"
    run_id = first["run_id"]
    before = storage.agent_run(run_id)["state"]
    assert len(before["observations"]) == 1

    second = orchestrator.continue_run(run_id, "使用本地日志和模拟 NAS 流程")

    assert second["run_id"] == run_id
    assert second["run_status"] == AgentRunStatus.COMPLETED
    after = storage.agent_run(run_id)["state"]
    assert after["pending_question"] is None
    assert len(after["observations"]) >= 2
    assert after["evidence"]
    assert "补充信息" in after["user_goal"]


def test_continue_course_answer_missing_citation_uses_one_server_bound_correction(tmp_path):
    class ResumeProvider(SequenceProvider):
        provider_name = "deepseek-test"

        def __init__(self):
            super().__init__(
                [
                    ActionDecision(
                        action=AgentAction.ASK_USER,
                        arguments={"question": "请补充实验环境"},
                        reason="信息不足",
                    ),
                    ActionDecision(
                        action=AgentAction.RETRIEVE,
                        arguments={"query": "NAS 实验日志字段"},
                        reason="补充后检索",
                    ),
                ]
            )
            self.structured_calls = 0

        def generate_answer(self, request: LLMRequest) -> LLMResponse:
            self.answer_calls += 1
            return LLMResponse(
                text="NAS 实验日志字段包括时间戳和输出摘要。",
                provider=self.provider_name,
                model=self.model_name,
            )

        def generate_structured(self, request, schema):
            assert schema is GroundedSpanAnswer
            self.structured_calls += 1
            return super().generate_structured(request, schema)

    storage = LearningLogger(tmp_path / "resume-citation-formation.db")
    _seed_default_published_document(storage)
    provider = ResumeProvider()
    retriever = FakeRetriever(
        RetrievalResponse(
            results=[
                _result(
                    document_id=1,
                    chunk_id=1,
                    snippet="NAS 实验日志字段包括时间戳和输出摘要。",
                    title="NAS 实验日志",
                )
            ],
            abstained=False,
            threshold=0.18,
            fallback_used=False,
        )
    )
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        retriever=retriever,
        mcp_gateway=MCPClientGateway(),
    )

    first = orchestrator.run("我想设计实验")
    second = orchestrator.continue_run(first["run_id"], "使用本地NAS实验日志")

    assert first["run_status"] == AgentRunStatus.REQUIRES_INPUT
    assert second["run_status"] == AgentRunStatus.COMPLETED
    assert second["stop_reason"] == "ANSWERED"
    assert "[S1]" in second["answer"]
    assert provider.answer_calls == 1
    assert provider.structured_calls == 1
    assert len(storage.tool_call_rows(first["run_id"])) == 0


def test_template_provider_continue_uses_supplemental_input_instead_of_asking_again(tmp_path):
    storage = LearningLogger(tmp_path / "template-resume.db")
    _seed_default_published_document(storage)
    provider = TemplateProvider()
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=FakeRetriever(),
        mcp_gateway=MCPClientGateway(),
    )

    first = orchestrator.run(f"{_template_ask_trigger()} NAS 实验目标不清楚")
    run_id = first["run_id"]
    first_record = storage.agent_run(run_id)

    assert first["run_status"] == AgentRunStatus.REQUIRES_INPUT
    assert first["requires_user_input"] is True
    assert first_record["status"] == AgentRunStatus.REQUIRES_INPUT
    assert [step["action"] for step in storage.agent_steps(run_id)] == ["ask_user"]

    second = orchestrator.continue_run(run_id, "使用本地 NAS 日志分析认证失败原因")
    second_record = storage.agent_run(run_id)
    actions = [step["action"] for step in storage.agent_steps(run_id)]

    assert second["run_id"] == run_id
    assert second["run_status"] == AgentRunStatus.COMPLETED
    assert second["requires_user_input"] is False
    assert second["citations"]
    assert second_record["status"] == AgentRunStatus.COMPLETED
    assert actions.count("ask_user") == 1
    assert "retrieve" in actions


def test_template_provider_http_continue_same_run_and_owner_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "template")
    monkeypatch.setenv("JWT_SECRET_KEY", "s1-template-http-secret-0123456789012345")
    monkeypatch.setenv("SQLITE_DB_PATH", str((tmp_path / "template-http.db").resolve()))
    original = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "template-http.db")
    context = main.learning_logger.seed_demo_identity()
    _save_indexed_document(
        main.learning_logger,
        int(context["course_id"]),
        int(context["class_id"]),
        "NAS 实验目标不清楚。使用本地 NAS 日志分析认证失败原因，并比较 security mode commands。",
    )
    client = TestClient(main.app)
    try:
        student_headers = _login_headers(client, "student_demo")
        other_headers = _login_headers(client, "student_demo2")
        first = client.post(
            "/api/chat",
            headers=student_headers,
            json={"question": f"{_template_ask_trigger()} NAS 实验目标不清楚"},
        )
        first_body = first.json()
        run_id = first_body["run_id"]
        first_run = client.get(f"/api/agent/runs/{run_id}", headers=student_headers).json()
        other_get = client.get(f"/api/agent/runs/{run_id}", headers=other_headers)
        other_continue = client.post(
            f"/api/agent/runs/{run_id}/continue",
            headers=other_headers,
            json={"user_input": "other user input"},
        )

        assert first.status_code == 200
        assert first_body["run_status"] == AgentRunStatus.REQUIRES_INPUT
        assert first_body["requires_user_input"] is True
        assert first_run["status"] == AgentRunStatus.REQUIRES_INPUT
        assert len(first_run["steps"]) == 1
        assert first_run["steps"][0]["action"] == "ask_user"
        assert other_get.status_code == 403
        assert other_continue.status_code == 403

        continued = client.post(
            f"/api/agent/runs/{run_id}/continue",
            headers=student_headers,
            json={"user_input": "使用本地 NAS 日志分析认证失败原因", "conversation_id": "s1-conv"},
        )
        continued_body = continued.json()
        final_run = client.get(f"/api/agent/runs/{run_id}", headers=student_headers).json()
        final_actions = [step["action"] for step in final_run["steps"]]

        assert continued.status_code == 200
        assert continued_body["run_id"] == run_id
        assert continued_body["conversation_id"] == "s1-conv"
        assert continued_body["run_status"] == AgentRunStatus.COMPLETED, continued_body
        assert continued_body["requires_user_input"] is False
        assert continued_body["citations"]
        assert final_run["status"] == AgentRunStatus.COMPLETED
        assert final_actions.count("ask_user") == 1
        assert "retrieve" in final_actions
    finally:
        main.learning_logger = original


def test_template_provider_resume_abstains_when_retrieval_has_no_evidence(tmp_path):
    storage = LearningLogger(tmp_path / "template-abstain.db")
    provider = TemplateProvider()

    retriever = FakeRetriever(
        response=RetrievalResponse(
            results=[],
            abstained=True,
            threshold=0.8,
            fallback_used=False,
        )
    )
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=retriever,
        mcp_gateway=MCPClientGateway(),
    )

    first = orchestrator.run(f"{_template_ask_trigger()} 缺少实验边界")
    second = orchestrator.continue_run(first["run_id"], "没有任何可用课程资料")
    actions = [step["action"] for step in storage.agent_steps(first["run_id"])]

    assert first["run_status"] == AgentRunStatus.REQUIRES_INPUT
    assert second["run_status"] == AgentRunStatus.ABSTAINED
    assert second["abstained"] is True
    assert second["citations"] == []
    assert actions == ["ask_user", "retrieve"]
