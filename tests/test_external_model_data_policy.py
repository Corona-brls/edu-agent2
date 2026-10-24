from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from backend import main
from backend.agent.observation import tool_result_to_observation
from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.schemas import ActionDecision, AgentAction, AgentIntent
from backend.agent.state import AgentRunStatus, AgentState
from backend.analytics.events import LearningEventService
from backend.core.config import get_settings
from backend.llm.data_policy import (
    COURSE_SPAN_PROFILE_GROUNDED,
    COURSE_SPAN_PROFILE_KEY,
    COURSE_SPAN_PROFILE_PLAIN,
    COURSE_SPAN_PROFILE_PLANNER,
    COURSE_SPAN_PROFILE_PRACTICE,
    ExternalModelDataPolicy,
    ModelCallPreflightError,
    build_course_text_spans,
    select_relevant_course_text_spans,
)
from backend.llm.deepseek_provider import DeepSeekProvider
from backend.llm.base import LLMProvider
from backend.llm.schemas import EvidenceItem, LLMRequest, LLMResponse
from backend.llm.template_provider import TemplateProvider
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import RetrievalResponse, RetrievalResult
from backend.schemas import AgentContinueRequest, ChatRequest, PracticeGuideRequest
from backend.storage.logger import LearningLogger, sha256_bytes
from backend.main import _project_run_steps, _project_run_traces, _project_tool_calls
from backend.agent.orchestrator import (
    _explicit_external_query_origin,
    _normalized_tool_arguments,
    _paper_search_query,
)
from mcp_server.schemas import ToolResult
from mcp_server.tools import TOOL_DEFINITIONS


PRIVATE_MARKER = "COURSE_PRIVATE_MARKER_7F2A"
SELECTED_MARKER = "R1C1_SELECTED_MARKER_8D91"
UNSELECTED_TAIL = "R1C1_UNSELECTED_TAIL_C4E7"
QUERY_TOKEN = "R1C1_QUERY_TOKEN_A62F"
IDENTITY_SENTINELS = {
    "user_id": "R1C1_USER_ID_91B3",
    "course_id": "R1C1_COURSE_ID_72C4",
    "class_id": "R1C1_CLASS_ID_53D5",
    "document_id": "R1C1_DOCUMENT_ID_34E6",
    "chunk_id": "R1C1_CHUNK_ID_15F7",
    "path": r"C:\R1C1_PATH_SENTINEL_86A8\private.md",
}
CONNECTOR_QUERY_PRIVATE_MARKER = "R2_CONNECTOR_PRIVATE_COURSE_MARKER_9A31"


class _CourseAwareQueryPlanner(LLMProvider):
    provider_name = "fake-course-aware-planner"
    model_name = "fake-course-aware-planner-model"

    def __init__(self, tool_name: str, *, ask_first: bool) -> None:
        self.tool_name = tool_name
        self.ask_first = ask_first
        self.calls = 0
        self.tool_emitted = False

    def decide_next_action(self, state_summary, available_tools):
        self.calls += 1
        if self.ask_first and self.calls == 1:
            return ActionDecision(
                action=AgentAction.ASK_USER,
                arguments={"question": "请补充公开查询范围"},
                reason="exercise the continue entry",
            )
        if CONNECTOR_QUERY_PRIVATE_MARKER not in state_summary:
            return ActionDecision(
                action=AgentAction.RETRIEVE,
                arguments={"query": "course context"},
                reason="read the authorized course span first",
            )
        if not self.tool_emitted:
            self.tool_emitted = True
            arguments = {
                "query": f"{CONNECTOR_QUERY_PRIVATE_MARKER} planner-derived outbound query",
                "top_k": 1,
            }
            if self.tool_name == "paper_search_real":
                arguments["source"] = "auto"
            return ActionDecision(
                action=AgentAction.CALL_TOOL,
                tool_name=self.tool_name,
                arguments=arguments,
                reason="attempt to reuse a private course span in connector arguments",
            )
        return ActionDecision(
            action=AgentAction.ABSTAIN,
            arguments={"message": "controlled test completion"},
            reason="connector attempt completed",
            task_complete=True,
        )

    def generate_answer(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("query-origin tests must not synthesize an answer")

    def generate_structured(self, request, schema):
        raise AssertionError("query-origin tests must not synthesize structured output")


class _PrivateCourseRetriever:
    def retrieve(self, query, top_k=None, course_id=None):
        return RetrievalResponse(
            results=[
                RetrievalResult(
                    document_id=1,
                    chunk_id=1,
                    source="server-owned-course-source",
                    title="Server-owned course context",
                    snippet=f"{CONNECTOR_QUERY_PRIVATE_MARKER} private course-only sentence",
                    page=1,
                    section="query origin",
                    source_type="md",
                    keyword_score=1.0,
                    vector_score=0.0,
                    final_score=1.0,
                )
            ],
            abstained=False,
            threshold=0.1,
            fallback_used=False,
        )


class _ConnectorDispatchGateway:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.calls: list[dict] = []

    def get_agent_tool_catalog(self, role="student"):
        definition = TOOL_DEFINITIONS[self.tool_name]
        return [
            {
                "name": definition.name,
                "description": definition.agent_description,
                "input_schema": definition.input_schema,
                "risk": definition.risk,
                "network": definition.network,
                "read_only": definition.read_only,
                "enabled": True,
            }
        ]

    def call_tool(self, name, arguments, *, run_id, step_id, step_number=1, **_kwargs):
        self.calls.append({"name": name, "arguments": dict(arguments), "run_id": run_id})
        result = ToolResult(
            success=False,
            error_code="CONTROLLED_CONNECTOR_EMPTY",
            error_message="controlled query-origin boundary probe",
            metadata={"tool_name": name},
        )
        return SimpleNamespace(
            result=result,
            observation=tool_result_to_observation(
                result,
                step_number=step_number,
                tool_name=name,
                fallback_used=False,
                run_id=run_id,
                step_id=step_id,
                arguments=arguments,
                latency_ms=1,
            ),
            fallback_used=False,
        )


_QUERY_ORIGIN_CASES = (
    ("paper_search_real", "请检索公开论文 Open5GS authentication security", "Open5GS"),
    ("vuln_lookup_real", "请查询公开漏洞目录中的 memory corruption", "memory corruption"),
    ("github_project_search", "请查询 GitHub srsRAN_Project 公开仓库", "srsRAN_Project"),
    ("standard_metadata_lookup", "请查询公开标准 3GPP TS 24.501 元数据", "3GPP TS 24.501"),
)

_MULTILINE_QUERY_ORIGIN_CASES = (
    ("paper_search_real", "请查询公开资料\n公开论文 Open5GS authentication security", "Open5GS"),
    ("vuln_lookup_real", "请查询公开资料\n公开漏洞 CVE-2024-12345", "CVE-2024-12345"),
    ("github_project_search", "请查询公开资料\nGitHub srsRAN_Project 公开仓库", "srsRAN_Project"),
    ("standard_metadata_lookup", "请查询公开资料\n公开标准 3GPP TS 24.501", "3GPP TS 24.501"),
)

_CONTINUE_QUERY_ORIGIN_CASES = (
    ("paper_search_real", "请补充公开查询范围", "公开论文 Open5GS authentication security", "Open5GS"),
    ("vuln_lookup_real", "请补充公开查询范围", "公开漏洞 CVE-2024-12345", "CVE-2024-12345"),
    ("github_project_search", "请补充公开查询范围", "GitHub srsRAN_Project 公开仓库", "srsRAN_Project"),
    ("standard_metadata_lookup", "请补充公开查询范围", "公开标准 3GPP TS 24.501", "3GPP TS 24.501"),
)

_REFERENTIAL_NEW_QUERY_CASES = (
    (
        "paper_search_real",
        "请使用上述公开查询补充公开论文 Open5GS authentication security",
        "Open5GS",
    ),
    ("vuln_lookup_real", "请使用上述公开查询核验 CVE-2024-12345", "CVE-2024-12345"),
    (
        "github_project_search",
        "请使用上述公开查询查看 GitHub srsRAN_Project 公开仓库",
        "srsRAN_Project",
    ),
    (
        "standard_metadata_lookup",
        "请使用上述公开查询核验公开标准 3GPP TS 24.501",
        "3GPP TS 24.501",
    ),
)

_GENERIC_CONTINUE_CASES = (
    ("paper_search_real", "请检索公开论文 Open5GS authentication security", "请继续查询", "Open5GS"),
    ("vuln_lookup_real", "请查询公开漏洞 CVE-2024-12345", "按前述继续", "CVE-2024-12345"),
    (
        "github_project_search",
        "请查询 GitHub srsRAN_Project 公开仓库",
        "请继续查询",
        "srsRAN_Project",
    ),
    (
        "standard_metadata_lookup",
        "请查询公开标准 3GPP TS 24.501",
        "按前述继续",
        "3GPP TS 24.501",
    ),
)

_SCHEMA_LENGTH_CASES = (
    ("paper_search_real", "Open5GS authentication security"),
    ("vuln_lookup_real", "CVE-2024-12345"),
    ("github_project_search", "srsRAN_Project"),
    ("standard_metadata_lookup", "3GPP TS 24.501"),
)


def _within_connector_query_with_middle_payload(payload):
    value = ("background " * 10) + payload + (" context" * 10)
    assert len(value) <= 300
    return value


def _over_connector_query_limit(payload):
    value = ("background " * 36) + payload + (" context" * 12)
    assert len(value) > 300
    return value


class _CaptureCompletions:
    def __init__(self, content: str = "safe response") -> None:
        self.requests: list[dict] = []
        self.content = content

    def create(self, **request):
        self.requests.append(request)
        return {"choices": [{"message": {"content": self.content}}]}


def _provider(
    allowlist: frozenset[int],
    *,
    content: str = "safe response",
) -> tuple[DeepSeekProvider, _CaptureCompletions]:
    completions = _CaptureCompletions(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    settings = replace(
        get_settings(),
        deepseek_api_key="fake-test-key",
        external_model_course_allowlist=allowlist,
        llm_max_retries=0,
        llm_timeout_seconds=2.0,
        agent_timeout_seconds=2.0,
    )
    return DeepSeekProvider(settings=settings, client=client), completions


def _request() -> LLMRequest:
    course = EvidenceItem(
        source_id="S1",
        text=f"{PRIVATE_MARKER} minimal approved sentence.",
        source=r"C:\private\course-42\secret.md",
        title="secret.md",
        document_id=91,
        chunk_id=92,
        resource_type="course",
        origin_kind="local_document",
        delivery_mode="local_rag",
    )
    public = EvidenceItem(
        source_id="S2",
        text="PUBLIC_EVIDENCE_MARKER",
        source="arxiv",
        title="Public paper",
        resource_type="paper",
        origin_kind="remote",
        delivery_mode="live_api",
    )
    return LLMRequest(
        question="Explain the selected evidence.",
        evidence=[course, public],
        metadata={
            "user_id": 7,
            "course_id": 42,
            "class_id": 9,
            "filename": "secret.md",
            "evidence_spans": [
                {
                    "span_id": "S1:text:1",
                    "source_id": "S1",
                    "field": "text",
                    "exact_text": f"{PRIVATE_MARKER} minimal approved sentence.",
                    "path": r"C:\private\course-42\secret.md",
                },
                {
                    "span_id": "S2:text:1",
                    "source_id": "S2",
                    "field": "text",
                    "exact_text": "PUBLIC_EVIDENCE_MARKER",
                },
            ],
        },
    )


def _outbound_text(completions: _CaptureCompletions) -> str:
    return str(completions.requests[-1]["messages"])


def _same_chunk_window_request(*, spans: list[dict] | None = None) -> LLMRequest:
    selected = f"{SELECTED_MARKER} {QUERY_TOKEN} is the approved complete sentence."
    tail = f"{UNSELECTED_TAIL} is an adjacent but unapproved complete sentence."
    course = EvidenceItem(
        source_id="S1",
        text=f"{selected} {tail}",
        source=IDENTITY_SENTINELS["path"],
        title="R1C1_TITLE_SENTINEL_D709.md",
        document_id=91,
        chunk_id=92,
        resource_type="course",
        origin_kind="local_document",
        delivery_mode="local_rag",
    )
    catalog = spans if spans is not None else [
        {
            "span_id": "S1:text:1",
            "source_id": "S1",
            "field": "text",
            "exact_text": selected,
        }
    ]
    return LLMRequest(
        question=f"Explain {QUERY_TOKEN}.",
        evidence=[course],
        metadata={**IDENTITY_SENTINELS, "evidence_spans": catalog},
    )


def _authoritative_binding_request(*, span_id: str, exact_text: str) -> LLMRequest:
    return LLMRequest(
        question="Explain the first complete sentence.",
        evidence=[
            EvidenceItem(
                source_id="S1",
                text="FIRST complete sentence. SECOND complete sentence.",
                resource_type="course",
                document_id=91,
                chunk_id=92,
            )
        ],
        metadata={
            "evidence_spans": [
                {
                    "span_id": span_id,
                    "source_id": "S1",
                    "field": "text",
                    "exact_text": exact_text,
                }
            ]
        },
    )


def test_default_empty_allowlist_and_request_flag_both_fail_closed():
    provider, completions = _provider(frozenset())
    assert provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    ) == "blocked_by_policy"

    provider.generate_answer(_request())
    outbound = _outbound_text(completions)

    assert PRIVATE_MARKER not in outbound
    assert "secret.md" not in outbound
    assert r"C:\private" not in outbound
    assert "PUBLIC_EVIDENCE_MARKER" in outbound


def test_allowlist_without_request_selection_still_omits_course_text():
    provider, completions = _provider(frozenset({42}))
    assert provider.configure_course_policy(
        course_id=42,
        use_course_references=False,
        currently_authorized=True,
    ) == "disabled_by_user"

    provider.generate_answer(_request())
    assert PRIVATE_MARKER not in _outbound_text(completions)


def test_double_gate_sends_only_bounded_deidentified_course_span():
    provider, completions = _provider(frozenset({42}))
    assert provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    ) == "allowed"

    provider.generate_answer(_request())
    outbound = _outbound_text(completions)

    assert PRIVATE_MARKER in outbound
    assert "secret.md" not in outbound
    assert r"C:\private" not in outbound
    assert "course_id" not in outbound
    assert "class_id" not in outbound
    assert "document_id" not in outbound
    assert "chunk_id" not in outbound


def test_answer_and_structured_messages_rebuild_course_text_only_from_approved_spans():
    request = _same_chunk_window_request()
    canonical_before = request.model_dump(mode="json")

    answer, answer_calls = _provider(frozenset({42}))
    answer.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )
    answer.generate_answer(request)

    structured, structured_calls = _provider(frozenset({42}), content='{"result":"ok"}')
    structured.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )
    parsed = structured.generate_structured(request, _StructuredProbe)
    assert parsed.result == "ok"

    for outbound in (_outbound_text(answer_calls), _outbound_text(structured_calls)):
        assert SELECTED_MARKER in outbound
        assert "[S1:text:1]" in outbound
        assert UNSELECTED_TAIL not in outbound
        assert "R1C1_TITLE_SENTINEL_D709.md" not in outbound
        assert all(value not in outbound for value in IDENTITY_SENTINELS.values())
    assert request.model_dump(mode="json") == canonical_before


def test_plain_course_answer_outbound_keeps_selected_s1_and_citation_instruction_only():
    request = _same_chunk_window_request().model_copy(
        update={
            "instructions": (
                "Use only the supplied evidence and cite the supporting evidence ID such as [S1]."
            ),
            "metadata": {
                **_same_chunk_window_request().metadata,
                COURSE_SPAN_PROFILE_KEY: COURSE_SPAN_PROFILE_PLAIN,
            },
        }
    )
    provider, completions = _provider(frozenset({42}))
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    provider.generate_answer(request)

    outbound = _outbound_text(completions)
    assert SELECTED_MARKER in outbound
    assert "[S1:text:1]" in outbound
    assert "such as [S1]" in outbound
    assert UNSELECTED_TAIL not in outbound
    assert "R1C1_TITLE_SENTINEL_D709.md" not in outbound
    assert all(value not in outbound for value in IDENTITY_SENTINELS.values())


def test_planner_selects_complete_relevant_sentence_without_adjacent_tail():
    request = _same_chunk_window_request()
    state = SimpleNamespace(
        user_goal=f"Explain {QUERY_TOKEN}.",
        intent="course_qa",
        current_step=1,
        max_steps=5,
        tool_call_count=0,
        max_tool_calls=3,
        required_sources=["course"],
        completed_sources=["course"],
        evidence=request.evidence,
        observations=[],
        pending_question=None,
        learning_context={
            "assignment_title": "Safe assignment title",
            "step_title": "Safe current step",
            "progress_percent": 60,
            "progress_note": "bounded current-user note",
        },
    )
    provider, completions = _provider(
        frozenset({42}),
        content='{"action":"answer","arguments":{},"reason":"ready","task_complete":true}',
    )
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    provider.decide_from_state(state, [])
    outbound = _outbound_text(completions)

    assert SELECTED_MARKER in outbound
    assert "[S1:text:1]" in outbound
    assert UNSELECTED_TAIL not in outbound
    assert "R1C1_TITLE_SENTINEL_D709.md" not in outbound
    assert all(value not in outbound for value in IDENTITY_SENTINELS.values())


def test_course_span_validation_uses_the_builder_whitespace_view_without_mutation():
    request = _same_chunk_window_request(
        spans=[
            {
                "span_id": "S1:text:1",
                "source_id": "S1",
                "field": "text",
                "exact_text": f"{SELECTED_MARKER}\n\t{QUERY_TOKEN} is the approved complete sentence.",
            }
        ]
    )
    canonical_before = request.model_dump(mode="json")
    projected = ExternalModelDataPolicy.from_ids({42}).project_request(request, status="allowed")

    assert projected.evidence[0].text == (
        f"[S1:text:1] {SELECTED_MARKER} {QUERY_TOKEN} is the approved complete sentence."
    )
    assert request.model_dump(mode="json") == canonical_before


def test_course_title_catalog_entry_is_filtered_while_all_text_spans_from_one_source_remain():
    request = _same_chunk_window_request(
        spans=[
            {
                "span_id": "S1:title:1",
                "source_id": "S1",
                "field": "title",
                "exact_text": "R1C1_TITLE_SENTINEL_D709.md",
            },
            {
                "span_id": "S1:text:1",
                "source_id": "S1",
                "field": "text",
                "exact_text": f"{SELECTED_MARKER} {QUERY_TOKEN} is the approved complete sentence.",
            },
            {
                "span_id": "S1:text:2",
                "source_id": "S1",
                "field": "text",
                "exact_text": f"{UNSELECTED_TAIL} is an adjacent but unapproved complete sentence.",
            },
        ]
    )
    policy = ExternalModelDataPolicy(frozenset({42}), max_course_spans=1)
    projected = policy.project_request(request, status="allowed")

    assert len(projected.evidence) == 1
    assert SELECTED_MARKER in projected.evidence[0].text
    assert UNSELECTED_TAIL in projected.evidence[0].text
    assert "R1C1_TITLE_SENTINEL_D709.md" not in projected.evidence[0].text
    assert [span["span_id"] for span in projected.metadata["evidence_spans"]] == [
        "S1:text:1",
        "S1:text:2",
    ]


@pytest.mark.parametrize(
    "exact_text",
    [
        "course_id=R1C1_EMBEDDED_ID_603A",
        r"C:\R1C1_EMBEDDED_PATH_711B\private.md",
    ],
)
def test_course_span_with_server_identity_or_path_fails_closed(exact_text):
    evidence = EvidenceItem(
        source_id="S1",
        text=exact_text,
        resource_type="course",
        document_id=91,
        chunk_id=92,
    )
    request = LLMRequest(
        question="Explain the approved span.",
        evidence=[evidence],
        metadata={
            "evidence_spans": [
                {
                    "span_id": "S1:text:1",
                    "source_id": "S1",
                    "field": "text",
                    "exact_text": exact_text,
                }
            ]
        },
    )

    with pytest.raises(ModelCallPreflightError) as caught:
        ExternalModelDataPolicy.from_ids({42}).project_request(request, status="allowed")

    assert caught.value.error_code == "MODEL_CONTEXT_PROJECTION_INVALID"


_PUBLIC_COURSE_URLS = (
    "https://public.example/home/guide",
    "HTTP://PUBLIC.EXAMPLE:8443/course/guide?view=full#section-2",
    "https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final",
    "https://www.rfc-editor.org/info/rfc9000",
    "https://www.3gpp.org/ftp/Specs/archive/24_series/24.501/",
    "[公开指南](https://public.example/guide?q=course#part-1)",
    "（https://public.example/guide?lang=zh）",
    "https://public.example/guide，",
    "https://public.example/one https://docs.example/two?mode=safe#part",
)

_PUBLIC_PORT_AND_IPV6_URLS = (
    "https://public.example/guide",
    "http://public.example:80/home/guide",
    "https://public.example:443/home/guide",
    "https://public.example:8443/home/guide?view=full#part",
    "https://[2001:db8::1]/home/guide",
    "https://[2001:db8::1]:8443/home/guide?view=full#part",
)

_INVALID_PORT_URLS = (
    "https://public.example:bad/home/private",
    "https://public.example:99999/workspace/x",
    "https://public.example:-1/home/private",
    "https://public.example:999999999999999999999/home/private",
)

_FORWARD_UNC_BYPASSES = (
    "//server/share/secret.md",
    "%2F%2Fserver%2Fshare%2Fsecret.md",
    "https://public.example/guide //server/share/secret.md",
    "https://public.example//server/share/secret.md",
    "https://public.example/%2F%2Fserver%2Fshare%2Fsecret.md",
    "https://public.example/guide?next=//server/share/secret.md",
    "https://public.example/guide?next=%2F%2Fserver%2Fshare%2Fsecret.md",
    "https://public.example/guide#next=//server/share/secret.md",
    "https://public.example/one https://docs.example/two?next=%2F%2Fserver%2Fshare/secret.md",
)

_INVALID_IPV6_URLS = (
    "https://[2001:db8::1/home/guide",
    "https://[2001:db8::zz]/home/guide",
    "https://[2001:db8::1]:bad/home/guide",
    "https://[2001:db8::1]:99999/home/guide",
    "https://user:secret@[2001:db8::1]/home/guide",
    "https://[2001:db8::1]/%2F%2Fserver%2Fshare/secret.md",
)


def _course_url_request(url_text: str, *, profile: str) -> tuple[LLMRequest, list]:
    canonical = f"防御记录的官方公开参考是 {url_text} 仅用于课程说明。"
    evidence = EvidenceItem(
        source_id="S1",
        text=canonical,
        resource_type="course",
        origin_kind="local_document",
        delivery_mode="local_rag",
        document_id=91,
        chunk_id=92,
    )
    selected = select_relevant_course_text_spans(
        [evidence],
        question="防御记录 官方公开参考",
        profile=profile,
        total_chars=1200,
        max_span_chars=1200,
        fallback_first=False,
    )
    request = LLMRequest(
        question="说明防御记录采用的官方公开参考。",
        evidence=[evidence],
        metadata={
            COURSE_SPAN_PROFILE_KEY: profile,
            "evidence_spans": [span.model_dump(mode="json") for span in selected],
        },
    )
    return request, selected


@pytest.mark.parametrize(
    "profile",
    [
        COURSE_SPAN_PROFILE_PLAIN,
        COURSE_SPAN_PROFILE_PLANNER,
        COURSE_SPAN_PROFILE_GROUNDED,
        COURSE_SPAN_PROFILE_PRACTICE,
    ],
)
@pytest.mark.parametrize("url_text", _PUBLIC_COURSE_URLS)
def test_public_http_url_survives_course_span_selection_and_outbound_projection(
    profile,
    url_text,
):
    request, selected = _course_url_request(url_text, profile=profile)
    provider, completions = _provider(frozenset({42}))
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    provider.generate_answer(request)

    assert selected
    assert url_text in " ".join(span.exact_text for span in selected)
    assert len(completions.requests) == 1
    assert url_text in _outbound_text(completions)


@pytest.mark.parametrize(
    "profile",
    [
        COURSE_SPAN_PROFILE_PLAIN,
        COURSE_SPAN_PROFILE_PLANNER,
        COURSE_SPAN_PROFILE_GROUNDED,
        COURSE_SPAN_PROFILE_PRACTICE,
    ],
)
@pytest.mark.parametrize("url_text", _PUBLIC_PORT_AND_IPV6_URLS)
def test_valid_ports_and_bracketed_ipv6_survive_all_public_projection_entries(
    profile,
    url_text,
):
    request, selected = _course_url_request(url_text, profile=profile)
    provider, completions = _provider(frozenset({42}))
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    provider.generate_answer(request)

    assert selected
    assert url_text in " ".join(span.exact_text for span in selected)
    assert len(completions.requests) == 1
    assert url_text in _outbound_text(completions)


@pytest.mark.parametrize(
    "profile",
    [
        COURSE_SPAN_PROFILE_PLAIN,
        COURSE_SPAN_PROFILE_PLANNER,
        COURSE_SPAN_PROFILE_GROUNDED,
        COURSE_SPAN_PROFILE_PRACTICE,
    ],
)
@pytest.mark.parametrize(
    "unsafe_text",
    _INVALID_PORT_URLS + _FORWARD_UNC_BYPASSES + _INVALID_IPV6_URLS,
)
def test_invalid_port_forward_unc_or_invalid_ipv6_fails_before_sdk(
    profile,
    unsafe_text,
):
    canonical = f"课程防御记录引用 {unsafe_text} 并要求保持服务器边界。"
    evidence = EvidenceItem(
        source_id="S1",
        text=canonical,
        resource_type="course",
        origin_kind="local_document",
        delivery_mode="local_rag",
        document_id=91,
        chunk_id=92,
    )
    built = build_course_text_spans([evidence], profile=profile)
    selected = select_relevant_course_text_spans(
        [evidence],
        question="课程防御记录 服务器边界",
        profile=profile,
        total_chars=1200,
        max_span_chars=1200,
        fallback_first=False,
    )
    request = LLMRequest(
        question="说明课程防御记录。",
        evidence=[evidence],
        metadata={
            COURSE_SPAN_PROFILE_KEY: profile,
            "evidence_spans": [span.model_dump(mode="json") for span in built],
        },
    )
    provider, completions = _provider(frozenset({42}))
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    with pytest.raises(ModelCallPreflightError) as caught:
        provider.generate_answer(request)

    assert built
    assert selected == []
    assert caught.value.error_code == "MODEL_CONTEXT_PROJECTION_INVALID"
    assert completions.requests == []


@pytest.mark.parametrize(
    "profile",
    [
        COURSE_SPAN_PROFILE_PLAIN,
        COURSE_SPAN_PROFILE_PLANNER,
        COURSE_SPAN_PROFILE_GROUNDED,
        COURSE_SPAN_PROFILE_PRACTICE,
    ],
)
def test_url_segmentation_round_trip_never_collides_with_course_marker_text(profile):
    canonical = (
        "防御记录保留 R4BURLTOKEN0R4B 并参考 "
        "https://public.example/R4BURLTOKEN1R4B/one。 "
        "防御记录保留 R4BURLTOKEN1R4B 并参考 https://docs.example/two。"
    )
    normalized = " ".join(canonical.split())
    evidence = EvidenceItem(
        source_id="S1",
        text=canonical,
        resource_type="course",
        origin_kind="local_document",
        delivery_mode="local_rag",
        document_id=91,
        chunk_id=92,
    )
    built = build_course_text_spans([evidence], profile=profile)
    selected = select_relevant_course_text_spans(
        [evidence],
        question="防御记录 保留 参考",
        profile=profile,
        total_chars=1200,
        max_span_chars=1200,
        fallback_first=False,
    )
    request = LLMRequest(
        question="说明防御记录保留的公开参考。",
        evidence=[evidence],
        metadata={
            COURSE_SPAN_PROFILE_KEY: profile,
            "evidence_spans": [span.model_dump(mode="json") for span in selected],
        },
    )
    provider, completions = _provider(frozenset({42}))
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    provider.generate_answer(request)

    assert built
    assert selected
    assert all(span.exact_text in normalized for span in built)
    rendered = " ".join(span.exact_text for span in built)
    assert rendered.count("R4BURLTOKEN0R4B") == normalized.count("R4BURLTOKEN0R4B")
    assert rendered.count("R4BURLTOKEN1R4B") == normalized.count("R4BURLTOKEN1R4B")
    assert rendered.count("https://public.example/R4BURLTOKEN1R4B/one") == 1
    assert rendered.count("https://docs.example/two") == 1
    assert len(completions.requests) == 1
    assert all(span.exact_text in _outbound_text(completions) for span in selected)


_LOCAL_PATH_AND_WRAPPED_BYPASSES = (
    r"C:\private\x",
    "C:/private/x",
    "path=C:/private/x",
    "'C:/private/x'",
    r"\\server\share\secret.md",
    "/home/private/secret.md",
    "/users/private/secret.md",
    "/var/private/secret.md",
    "/tmp/private/secret.md",
    "/workspace/private/secret.md",
    "file:///C:/private/secret.md",
    "file:///home/private/secret.md",
    "[本地文件](file:///C:/private/secret.md)",
    "x://private/secret.md",
    "https://public.example/C:/private/secret.md",
    "https://public.example/%43%3A%2Fprivate%2Fsecret.md",
    r"https://public.example/guide?next=\\server\share\secret.md",
    "https://public.example/guide?next=%5C%5Cserver%5Cshare%5Csecret.md",
    "https://public.example/guide?next=file:///C:/private/secret.md",
    "https://public.example/guide?next=%66%69%6c%65%3A%2F%2F%2FC%3A%2Fprivate",
    "https://public.example/guide?course_id=77",
    "https://public.example/one https://public.example/two C:/private/secret.md",
)


@pytest.mark.parametrize(
    "profile",
    [
        COURSE_SPAN_PROFILE_PLAIN,
        COURSE_SPAN_PROFILE_PLANNER,
        COURSE_SPAN_PROFILE_GROUNDED,
        COURSE_SPAN_PROFILE_PRACTICE,
    ],
)
@pytest.mark.parametrize("unsafe_text", _LOCAL_PATH_AND_WRAPPED_BYPASSES)
def test_local_path_or_identity_wrapped_in_course_text_fails_before_sdk(
    unsafe_text,
    profile,
):
    canonical = f"课程防御记录引用 {unsafe_text} 并要求保持服务器边界。"
    evidence = EvidenceItem(
        source_id="S1",
        text=canonical,
        resource_type="course",
        origin_kind="local_document",
        delivery_mode="local_rag",
        document_id=91,
        chunk_id=92,
    )
    spans = build_course_text_spans([evidence], profile=profile)
    request = LLMRequest(
        question="说明课程防御记录。",
        evidence=[evidence],
        metadata={
            COURSE_SPAN_PROFILE_KEY: profile,
            "evidence_spans": [span.model_dump(mode="json") for span in spans],
        },
    )
    provider, completions = _provider(frozenset({42}))
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    with pytest.raises(ModelCallPreflightError) as caught:
        provider.generate_answer(request)

    assert caught.value.error_code == "MODEL_CONTEXT_PROJECTION_INVALID"
    assert completions.requests == []


@pytest.mark.parametrize(
    "catalog",
    [
        [],
        [
            {
                "span_id": "S999:text:1",
                "source_id": "S999",
                "field": "text",
                "exact_text": SELECTED_MARKER,
            }
        ],
        [
            {
                "span_id": "S1:text:1",
                "source_id": "S1",
                "field": "text",
                "exact_text": "R1C1_NOT_CANONICAL_55C2",
            }
        ],
        [
            {
                "span_id": "S2:text:1",
                "source_id": "S1",
                "field": "text",
                "exact_text": f"{SELECTED_MARKER} {QUERY_TOKEN} is the approved complete sentence.",
            }
        ],
        [
            {
                "span_id": "S1:text:1",
                "source_id": "S1",
                "field": "title",
                "exact_text": "R1C1_TITLE_SENTINEL_D709.md",
            }
        ],
        [
            {
                "span_id": "S1:text:1",
                "source_id": "S1",
                "field": "text",
                "exact_text": f"{SELECTED_MARKER} {QUERY_TOKEN} is the approved complete sentence.",
            },
            {
                "span_id": "S1:text:1",
                "source_id": "S1",
                "field": "text",
                "exact_text": f"{SELECTED_MARKER} {QUERY_TOKEN} is the approved complete sentence.",
            },
        ],
    ],
    ids=["empty", "forged-source", "non-canonical", "unbound-id", "title-only", "duplicate-id"],
)
def test_invalid_course_span_catalog_fails_closed_before_sdk(catalog):
    provider, completions = _provider(frozenset({42}))
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    with pytest.raises(ModelCallPreflightError) as caught:
        provider.generate_answer(_same_chunk_window_request(spans=catalog))

    assert caught.value.error_code == "MODEL_CONTEXT_PROJECTION_INVALID"
    assert caught.value.status_code == 409
    assert caught.value.policy_status == "revoked"
    assert completions.requests == []


@pytest.mark.parametrize(
    ("span_id", "exact_text"),
    [
        ("S1:text:1", "FIRST"),
        ("S1:text:1", "SECOND complete sentence."),
        ("S1:text:999", "SECOND complete sentence."),
        (f"S1:text:{'9' * 80}", "SECOND complete sentence."),
    ],
    ids=["partial-text", "wrong-ordinal", "unknown-ordinal", "overlong-id"],
)
def test_course_span_tuple_must_equal_server_builder_entry_before_sdk(span_id, exact_text):
    provider, completions = _provider(frozenset({42}))
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    with pytest.raises(ModelCallPreflightError) as caught:
        provider.generate_answer(
            _authoritative_binding_request(span_id=span_id, exact_text=exact_text)
        )

    assert caught.value.error_code == "MODEL_CONTEXT_PROJECTION_INVALID"
    assert caught.value.status_code == 409
    assert caught.value.policy_status == "revoked"
    assert completions.requests == []


def test_repeated_course_sentences_remain_bound_to_both_server_ordinals():
    repeated = "REPEATED complete sentence."
    request = LLMRequest(
        question="Explain the repeated sentence.",
        evidence=[
            EvidenceItem(
                source_id="S1",
                text=f"{repeated} {repeated}",
                resource_type="course",
                document_id=91,
                chunk_id=92,
            )
        ],
        metadata={
            "evidence_spans": [
                {
                    "span_id": "S1:text:1",
                    "source_id": "S1",
                    "field": "text",
                    "exact_text": repeated,
                },
                {
                    "span_id": "S1:text:2",
                    "source_id": "S1",
                    "field": "text",
                    "exact_text": repeated,
                },
            ]
        },
    )
    canonical_before = request.model_dump(mode="json")

    projected = ExternalModelDataPolicy.from_ids({42}).project_request(
        request,
        status="allowed",
    )

    assert projected.evidence[0].text.splitlines() == [
        f"[S1:text:1] {repeated}",
        f"[S1:text:2] {repeated}",
    ]
    assert request.model_dump(mode="json") == canonical_before


@pytest.mark.parametrize(
    ("profile", "canonical", "approved"),
    [
        (
            COURSE_SPAN_PROFILE_PLAIN,
            "PLAIN complete sentence. Adjacent sentence.",
            "PLAIN complete sentence.",
        ),
        (
            COURSE_SPAN_PROFILE_PLANNER,
            "PLANNER complete sentence. Adjacent sentence.",
            "PLANNER complete sentence.",
        ),
        (
            COURSE_SPAN_PROFILE_GROUNDED,
            "GROUNDED complete sentence. Adjacent sentence.",
            "GROUNDED complete sentence.",
        ),
        (
            COURSE_SPAN_PROFILE_PRACTICE,
            "PRACTICE complete sentence. Adjacent sentence.",
            "PRACTICE complete sentence",
        ),
    ],
)
def test_each_server_course_span_profile_accepts_its_complete_authoritative_entry(
    profile,
    canonical,
    approved,
):
    request = LLMRequest(
        question="Explain the approved entry.",
        evidence=[
            EvidenceItem(
                source_id="S1",
                text=canonical,
                resource_type="course",
                document_id=91,
                chunk_id=92,
            )
        ],
        metadata={
            COURSE_SPAN_PROFILE_KEY: profile,
            "evidence_spans": [
                {
                    "span_id": "S1:text:1",
                    "source_id": "S1",
                    "field": "text",
                    "exact_text": approved,
                }
            ],
        },
    )
    canonical_before = request.model_dump(mode="json")

    projected = ExternalModelDataPolicy.from_ids({42}).project_request(
        request,
        status="allowed",
    )

    assert projected.evidence[0].text == f"[S1:text:1] {approved}"
    assert COURSE_SPAN_PROFILE_KEY not in projected.metadata
    assert request.model_dump(mode="json") == canonical_before


def test_plain_answer_selector_can_choose_later_relevant_source_without_approving_first_source():
    query_token = "R1C1R_SECOND_SOURCE_QUERY_64A2"
    evidence = [
        EvidenceItem(
            source_id="S1",
            text="FIRST_SOURCE_UNRELATED complete sentence.",
            resource_type="course",
            document_id=91,
            chunk_id=92,
        ),
        EvidenceItem(
            source_id="S2",
            text=f"SECOND_SOURCE_SELECTED {query_token} complete sentence.",
            resource_type="course",
            document_id=93,
            chunk_id=94,
        ),
    ]
    selected = select_relevant_course_text_spans(
        evidence,
        question=f"Explain {query_token}.",
    )
    assert [span.span_id for span in selected] == ["S2:text:1"]
    request = LLMRequest(
        question=f"Explain {query_token}.",
        evidence=evidence,
        metadata={
            COURSE_SPAN_PROFILE_KEY: COURSE_SPAN_PROFILE_PLAIN,
            "evidence_spans": [span.model_dump(mode="json") for span in selected],
        },
    )

    projected = ExternalModelDataPolicy.from_ids({42}).project_request(
        request,
        status="allowed",
    )

    assert [item.source_id for item in projected.evidence] == ["S2"]
    assert query_token in projected.evidence[0].text
    assert "FIRST_SOURCE_UNRELATED" not in str(projected.model_dump(mode="json"))


def test_course_span_outside_bounded_source_set_fails_closed_before_sdk():
    evidence = [
        EvidenceItem(
            source_id=f"S{index}",
            text=f"SOURCE_{index} complete sentence.",
            resource_type="course",
            document_id=100 + index,
            chunk_id=200 + index,
        )
        for index in range(1, 7)
    ]
    request = LLMRequest(
        question="Explain SOURCE_1.",
        evidence=evidence,
        metadata={
            COURSE_SPAN_PROFILE_KEY: COURSE_SPAN_PROFILE_PLAIN,
            "evidence_spans": [
                {
                    "span_id": "S1:text:1",
                    "source_id": "S1",
                    "field": "text",
                    "exact_text": "SOURCE_1 complete sentence.",
                },
                {
                    "span_id": "S6:text:999",
                    "source_id": "S6",
                    "field": "text",
                    "exact_text": "SOURCE_6",
                },
            ],
        },
    )
    provider, completions = _provider(frozenset({42}))
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    with pytest.raises(ModelCallPreflightError) as caught:
        provider.generate_answer(request)

    assert caught.value.error_code == "MODEL_CONTEXT_PROJECTION_INVALID"
    assert completions.requests == []


def test_plain_answer_total_budget_keeps_whole_spans_without_filling_or_truncating_tail():
    query_token = "R1C1R_BUDGET_QUERY_0F31"
    first = f"{query_token} FIRST_FULL_SPAN " + ("a" * 310) + "."
    second = f"{query_token} SECOND_FULL_SPAN " + ("b" * 310) + "."
    evidence = EvidenceItem(
        source_id="S1",
        text=f"{first} {second}",
        resource_type="course",
        document_id=91,
        chunk_id=92,
    )

    selected = select_relevant_course_text_spans(
        [evidence],
        question=f"Explain {query_token}.",
    )

    assert [span.span_id for span in selected] == ["S1:text:1"]
    assert selected[0].exact_text == first
    assert "SECOND_FULL_SPAN" not in selected[0].exact_text


def test_unknown_course_span_profile_fails_closed_before_sdk():
    request = _same_chunk_window_request()
    request = request.model_copy(
        update={
            "metadata": {
                **request.metadata,
                COURSE_SPAN_PROFILE_KEY: "client_forged_profile_v999",
            }
        }
    )
    provider, completions = _provider(frozenset({42}))
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    with pytest.raises(ModelCallPreflightError) as caught:
        provider.generate_answer(request)

    assert caught.value.error_code == "MODEL_CONTEXT_PROJECTION_INVALID"
    assert caught.value.status_code == 409
    assert completions.requests == []


def test_duplicate_source_id_across_course_and_public_evidence_fails_closed_before_sdk():
    request = _request()
    request = request.model_copy(
        update={
            "evidence": [
                request.evidence[0],
                request.evidence[1].model_copy(update={"source_id": "S1"}),
            ]
        }
    )
    provider, completions = _provider(frozenset({42}))
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    with pytest.raises(ModelCallPreflightError) as caught:
        provider.generate_answer(request)

    assert caught.value.error_code == "MODEL_CONTEXT_PROJECTION_INVALID"
    assert completions.requests == []


def test_overlong_course_span_is_rejected_whole_without_same_id_truncation():
    exact_text = "R1C1_OVERLONG_SPAN_37B9_" + ("x" * 700)
    evidence = EvidenceItem(
        source_id="S1",
        text=exact_text,
        resource_type="course",
        document_id=91,
        chunk_id=92,
    )
    request = LLMRequest(
        question="Explain the approved span.",
        evidence=[evidence],
        metadata={
            "evidence_spans": [
                {
                    "span_id": "S1:text:1",
                    "source_id": "S1",
                    "field": "text",
                    "exact_text": exact_text,
                }
            ]
        },
    )
    provider, completions = _provider(frozenset({42}))
    provider.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )

    with pytest.raises(ModelCallPreflightError) as caught:
        provider.generate_answer(request)

    assert caught.value.error_code == "MODEL_CONTEXT_PROJECTION_INVALID"
    assert completions.requests == []


def test_revocation_status_removes_course_span_before_outbound_call():
    policy = ExternalModelDataPolicy.from_ids({42})
    status = policy.status(course_id=42, use_course_references=True, currently_authorized=False)
    projected = policy.project_request(_request(), status=status)

    assert status == "revoked"
    assert [item.source_id for item in projected.evidence] == ["S2"]
    assert PRIVATE_MARKER not in str(projected.model_dump())


class _StructuredProbe(BaseModel):
    result: str


def _live_guard_fixture(tmp_path, monkeypatch, *, with_session: bool = False):
    storage = LearningLogger(tmp_path / "guard.db")
    context = storage.seed_demo_identity()
    monkeypatch.setattr(main, "learning_logger", storage)
    user = storage.user_by_username("student_demo")
    assert user is not None
    course_id = int(context["course_id"])
    class_id = int(context["class_id"])
    document = Document(
        source_path=tmp_path / "private-course.md",
        filename="private-course.md",
        file_type=".md",
        title="Private course",
        text=f"{PRIVATE_MARKER} minimal approved sentence.",
        pages=[
            Page(
                text=f"{PRIVATE_MARKER} minimal approved sentence.",
                page=1,
                section="scope",
                source_type="md",
            )
        ],
    )
    saved = storage.save_document(
        document=document,
        content_hash=sha256_bytes(document.text.encode("utf-8")),
        chunks=[{"chunk_text": document.text, "page": 1, "section": "scope", "source_type": "md"}],
        byte_size=len(document.text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
        class_id=class_id,
        uploaded_by_user_id=int(user["id"]),
    )
    document_id = int(saved["id"])
    storage.set_document_publication(document_id, published=True, actor_id=int(user["id"]))
    chunk_id = int(storage.chunks_for_document(document_id)[0]["id"])
    evidence = EvidenceItem(
        source_id="S1",
        text=document.text,
        source=str(document.source_path),
        title=document.filename,
        document_id=document_id,
        chunk_id=chunk_id,
        resource_type="course",
        origin_kind="local_document",
        delivery_mode="local_rag",
    )
    state = AgentState(
        user_id=str(user["id"]),
        course_id=course_id,
        class_id=class_id,
        user_goal="Explain the approved course span.",
        intent="course_qa",
        evidence=[evidence],
        required_sources=["course"],
        completed_sources=["course"],
    )
    client_session_id = None
    if with_session:
        client_session_id = "guard-session"
        LearningEventService(storage).create_session(
            user=user,
            client_session_id=client_session_id,
            course_id=course_id,
            class_id=class_id,
        )
    provider, completions = _provider(
        frozenset({course_id}),
        content='{"action":"answer","arguments":{},"reason":"ready","task_complete":true}',
    )
    guard = main._RequestModelCallGuard(
        provider=provider,
        user_id=int(user["id"]),
        course_id=course_id,
        class_id=class_id,
        client_session_id=client_session_id,
        use_course_references=True,
        require_course_context=True,
    )
    request = LLMRequest(
        question=state.user_goal,
        evidence=[evidence],
        metadata={
            "evidence_spans": [
                {
                    "span_id": "S1:text:1",
                    "source_id": "S1",
                    "field": "text",
                    "exact_text": document.text,
                }
            ]
        },
    )
    return storage, user, document_id, state, request, provider, completions, guard


def test_planner_and_structured_paths_use_the_same_outbound_projection():
    state = SimpleNamespace(
        user_goal="public student wording",
        intent="course_qa",
        current_step=0,
        max_steps=3,
        tool_call_count=0,
        max_tool_calls=2,
        required_sources=["course"],
        completed_sources=[],
        evidence=_request().evidence,
        observations=[],
        pending_question=None,
    )
    planner, planner_calls = _provider(
        frozenset(),
        content='{"action":"abstain","arguments":{},"reason":"blocked","task_complete":true}',
    )
    planner.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )
    planner.decide_from_state(state, [])
    assert PRIVATE_MARKER not in _outbound_text(planner_calls)

    practice, practice_calls = _provider(frozenset({42}), content='{"result":"ok"}')
    practice.configure_course_policy(
        course_id=42,
        use_course_references=True,
        currently_authorized=True,
    )
    parsed = practice.generate_structured(_request(), _StructuredProbe)
    assert parsed.result == "ok"
    assert PRIVATE_MARKER in _outbound_text(practice_calls)
    assert "secret.md" not in _outbound_text(practice_calls)


def test_planner_empty_relevance_omits_course_text_without_weakening_strict_projection():
    evidence = EvidenceItem(
        source_id="S1",
        text="private curriculum cryptography corpus",
        source="private.md",
        title="Private",
        document_id=7,
        chunk_id=9,
        resource_type="course",
        origin_kind="local_document",
        delivery_mode="local_rag",
        content_hash="hash-1",
        trusted=True,
    )
    state = SimpleNamespace(
        user_goal="weather sunshine forecast",
        intent="course_qa",
        current_step=0,
        max_steps=3,
        tool_call_count=0,
        max_tool_calls=2,
        required_sources=["course"],
        completed_sources=["course"],
        evidence=[evidence],
        observations=[],
        pending_question=None,
        learning_context={
            "assignment_title": "Safe assignment title",
            "step_title": "Safe current step",
            "progress_percent": 60,
            "progress_note": "bounded current-user note",
        },
    )
    policy = ExternalModelDataPolicy.from_ids({42})

    summary = policy.planner_summary(state, status="allowed")

    assert "private curriculum" not in summary
    assert "证据：\n暂无" in summary
    assert "Safe assignment title" in summary
    assert "Safe current step" in summary
    assert "60%" in summary
    with pytest.raises(ModelCallPreflightError) as caught:
        policy.project_request(
            LLMRequest(question=state.user_goal, evidence=[evidence], metadata={}),
            status="allowed",
        )
    assert caught.value.error_code == "MODEL_CONTEXT_PROJECTION_INVALID"


def test_historical_local_sample_tool_payloads_are_redacted_from_run_projections():
    tool_calls = _project_tool_calls(
        [{"tool_name": "paper_search", "arguments_json": "{}", "result_json": PRIVATE_MARKER}]
    )
    steps = _project_run_steps(
        [{
            "decision": {"tool_name": "vuln_lookup", "arguments": {"query": "public"}},
            "observation": {"tool_name": "vuln_lookup", "content": PRIVATE_MARKER},
        }]
    )
    traces = _project_run_traces(
        [{"event_type": "tool", "message": PRIVATE_MARKER, "metadata": {"tool_name": "standard_lookup"}}]
    )

    outbound = str({"tool_calls": tool_calls, "steps": steps, "traces": traces})
    assert PRIVATE_MARKER not in outbound
    assert "paper_search" not in outbound
    assert "vuln_lookup" not in outbound
    assert "standard_lookup" not in outbound
    assert "redacted_legacy_tool" in outbound


def test_connector_queries_never_receive_server_course_spans():
    projected = ExternalModelDataPolicy.from_ids(set()).project_request(
        _request(),
        status="blocked_by_policy",
    )
    explicit_public_query = "Open5GS NAS authentication security"
    captured: list[dict] = []

    for tool_name in (
        "paper_search_real",
        "github_project_search",
        "vuln_lookup_real",
        "standard_metadata_lookup",
    ):
        captured.append(
            {
                "tool_name": tool_name,
                "arguments": _normalized_tool_arguments(
                    tool_name,
                    {"query": explicit_public_query},
                    explicit_public_query,
                ),
            }
        )

    assert PRIVATE_MARKER not in str(projected.model_dump())
    assert PRIVATE_MARKER not in str(captured)
    assert all(item["arguments"].get("query") for item in captured)


def _run_query_origin_probe(
    tmp_path,
    monkeypatch,
    tool_name,
    user_goal,
    *,
    via_continue,
    continue_input="只使用上面的公开检索词",
):
    definition = TOOL_DEFINITIONS[tool_name]
    monkeypatch.setitem(
        TOOL_DEFINITIONS,
        tool_name,
        definition.model_copy(update={"enabled": True}),
    )
    planner = _CourseAwareQueryPlanner(tool_name, ask_first=via_continue)
    gateway = _ConnectorDispatchGateway(tool_name)
    storage = LearningLogger(tmp_path / f"query-origin-{tool_name}-{'continue' if via_continue else 'run'}.db")
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=planner,
        answer_provider=planner,
        fallback_provider=planner,
        retriever=_PrivateCourseRetriever(),
        mcp_gateway=gateway,
    )
    if via_continue:
        first = orchestrator.run(user_goal, intent_override=AgentIntent.MULTI_SOURCE.value)
        assert first["run_status"] == AgentRunStatus.REQUIRES_INPUT
        result = orchestrator.continue_run(first["run_id"], continue_input)
        assert result["run_id"] == first["run_id"]
    else:
        result = orchestrator.run(user_goal, intent_override=AgentIntent.MULTI_SOURCE.value)
    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert planner.tool_emitted is True
    assert len(gateway.calls) == 1
    return gateway.calls[0]


@pytest.mark.parametrize("tool_name,user_goal,expected_public_fragment", _QUERY_ORIGIN_CASES)
def test_real_agent_connector_dispatch_uses_only_explicit_public_query(
    tmp_path,
    monkeypatch,
    tool_name,
    user_goal,
    expected_public_fragment,
):
    dispatched = _run_query_origin_probe(
        tmp_path,
        monkeypatch,
        tool_name,
        user_goal,
        via_continue=False,
    )

    assert dispatched["name"] == tool_name
    assert CONNECTOR_QUERY_PRIVATE_MARKER not in str(dispatched["arguments"])
    assert expected_public_fragment.casefold() in dispatched["arguments"]["query"].casefold()


@pytest.mark.parametrize("tool_name,user_goal,expected_public_fragment", _QUERY_ORIGIN_CASES)
def test_real_agent_continue_connector_dispatch_uses_only_explicit_public_query(
    tmp_path,
    monkeypatch,
    tool_name,
    user_goal,
    expected_public_fragment,
):
    dispatched = _run_query_origin_probe(
        tmp_path,
        monkeypatch,
        tool_name,
        user_goal,
        via_continue=True,
    )

    assert dispatched["name"] == tool_name
    assert CONNECTOR_QUERY_PRIVATE_MARKER not in str(dispatched["arguments"])
    assert expected_public_fragment.casefold() in dispatched["arguments"]["query"].casefold()


@pytest.mark.parametrize(
    "tool_name,user_goal,expected_public_fragment",
    _MULTILINE_QUERY_ORIGIN_CASES,
)
def test_real_agent_multiline_caller_query_reaches_connector(
    tmp_path,
    monkeypatch,
    tool_name,
    user_goal,
    expected_public_fragment,
):
    dispatched = _run_query_origin_probe(
        tmp_path,
        monkeypatch,
        tool_name,
        user_goal,
        via_continue=False,
    )

    assert dispatched["name"] == tool_name
    assert CONNECTOR_QUERY_PRIVATE_MARKER not in str(dispatched["arguments"])
    assert expected_public_fragment.casefold() in dispatched["arguments"]["query"].casefold()


@pytest.mark.parametrize(
    "tool_name,user_goal,continue_input,expected_public_fragment",
    _CONTINUE_QUERY_ORIGIN_CASES,
)
def test_real_agent_continue_new_caller_query_reaches_connector(
    tmp_path,
    monkeypatch,
    tool_name,
    user_goal,
    continue_input,
    expected_public_fragment,
):
    dispatched = _run_query_origin_probe(
        tmp_path,
        monkeypatch,
        tool_name,
        user_goal,
        via_continue=True,
        continue_input=continue_input,
    )

    assert dispatched["name"] == tool_name
    assert CONNECTOR_QUERY_PRIVATE_MARKER not in str(dispatched["arguments"])
    assert expected_public_fragment.casefold() in dispatched["arguments"]["query"].casefold()


@pytest.mark.parametrize("caller_input", ["", " \r\n\t "])
def test_external_query_origin_empty_input_uses_safe_fallback(caller_input):
    assert _explicit_external_query_origin(
        caller_input,
        fallback="public metadata fallback",
    ) == "public metadata fallback"


@pytest.mark.parametrize(
    "tool_name,continue_input,expected_public_fragment",
    _REFERENTIAL_NEW_QUERY_CASES,
)
def test_real_agent_referential_continue_with_new_query_reaches_connector(
    tmp_path,
    monkeypatch,
    tool_name,
    continue_input,
    expected_public_fragment,
):
    dispatched = _run_query_origin_probe(
        tmp_path,
        monkeypatch,
        tool_name,
        "请补充公开查询范围",
        via_continue=True,
        continue_input=continue_input,
    )

    assert dispatched["name"] == tool_name
    assert CONNECTOR_QUERY_PRIVATE_MARKER not in str(dispatched["arguments"])
    assert expected_public_fragment.casefold() in dispatched["arguments"]["query"].casefold()


@pytest.mark.parametrize(
    "tool_name,user_goal,continue_input,expected_public_fragment",
    _GENERIC_CONTINUE_CASES,
)
def test_real_agent_generic_continue_preserves_previous_query(
    tmp_path,
    monkeypatch,
    tool_name,
    user_goal,
    continue_input,
    expected_public_fragment,
):
    dispatched = _run_query_origin_probe(
        tmp_path,
        monkeypatch,
        tool_name,
        user_goal,
        via_continue=True,
        continue_input=continue_input,
    )

    assert dispatched["name"] == tool_name
    assert CONNECTOR_QUERY_PRIVATE_MARKER not in str(dispatched["arguments"])
    assert expected_public_fragment.casefold() in dispatched["arguments"]["query"].casefold()


def test_external_query_origin_preserves_connector_sized_middle_payload():
    payload = "MID_PAYLOAD_PUBLIC_TOPIC_7B21"
    caller_input = _within_connector_query_with_middle_payload(payload)

    result = _explicit_external_query_origin(caller_input, fallback="public metadata fallback")

    assert result == caller_input
    assert payload in result
    assert len(result) <= 300


def test_paper_query_bounds_over_connector_limit_deterministically():
    payload = "LATE_PAPER_PUBLIC_TOPIC_5C18"
    caller_input = _over_connector_query_limit(payload)

    first = _paper_search_query(caller_input)
    second = _paper_search_query(caller_input)

    assert first
    assert first == second
    assert len(first) <= 300


@pytest.mark.parametrize("tool_name,expected_public_fragment", _SCHEMA_LENGTH_CASES)
def test_real_agent_connector_sized_middle_query_reaches_connector(
    tmp_path,
    monkeypatch,
    tool_name,
    expected_public_fragment,
):
    user_goal = _within_connector_query_with_middle_payload(expected_public_fragment)
    dispatched = _run_query_origin_probe(
        tmp_path,
        monkeypatch,
        tool_name,
        user_goal,
        via_continue=False,
    )

    assert dispatched["name"] == tool_name
    assert CONNECTOR_QUERY_PRIVATE_MARKER not in str(dispatched["arguments"])
    assert expected_public_fragment.casefold() in dispatched["arguments"]["query"].casefold()
    assert len(dispatched["arguments"]["query"]) <= 300


@pytest.mark.parametrize("tool_name,expected_public_fragment", _SCHEMA_LENGTH_CASES)
def test_real_agent_over_connector_limit_uses_safe_bounded_query(
    tmp_path,
    monkeypatch,
    tool_name,
    expected_public_fragment,
):
    user_goal = _over_connector_query_limit(expected_public_fragment)
    dispatched = _run_query_origin_probe(
        tmp_path,
        monkeypatch,
        tool_name,
        user_goal,
        via_continue=False,
    )

    assert dispatched["name"] == tool_name
    assert CONNECTOR_QUERY_PRIVATE_MARKER not in str(dispatched["arguments"])
    assert dispatched["arguments"]["query"].strip()
    assert len(dispatched["arguments"]["query"]) <= 300


def test_retrieval_then_unpublish_blocks_planner_before_any_outbound_message(tmp_path, monkeypatch):
    storage, user, document_id, state, _request_value, provider, completions, guard = _live_guard_fixture(
        tmp_path, monkeypatch
    )
    guard.check_request_start()
    storage.set_document_publication(document_id, published=False, actor_id=int(user["id"]))
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        model_call_guard=guard,
    )

    with pytest.raises(ModelCallPreflightError, match="撤回") as caught:
        orchestrator._guarded_model_call(
            provider=provider,
            state=state,
            request=None,
            phase="planner",
            call=lambda: provider.decide_from_state(state, []),
        )

    assert caught.value.error_code == "MODEL_CONTEXT_REVOKED"
    assert completions.requests == []
    assert PRIVATE_MARKER not in str(completions.requests)


def test_planner_return_then_unpublish_blocks_practice_before_second_outbound(tmp_path, monkeypatch):
    storage, user, document_id, state, request, provider, completions, base_guard = _live_guard_fixture(
        tmp_path, monkeypatch
    )

    def revoke_after_planner(**kwargs):
        status = base_guard(**kwargs)
        if kwargs["phase"] == "planner" and kwargs["stage"] == "after":
            storage.set_document_publication(document_id, published=False, actor_id=int(user["id"]))
        return status

    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        model_call_guard=revoke_after_planner,
    )
    orchestrator._guarded_model_call(
        provider=provider,
        state=state,
        request=None,
        phase="planner",
        call=lambda: provider.decide_from_state(state, []),
    )
    assert len(completions.requests) == 1

    with pytest.raises(ModelCallPreflightError) as caught:
        orchestrator._guarded_model_call(
            provider=provider,
            state=state,
            request=request,
            phase="practice_structured",
            call=lambda: provider.generate_structured(request, _StructuredProbe),
        )

    assert caught.value.error_code == "MODEL_CONTEXT_REVOKED"
    assert len(completions.requests) == 1


def test_model_return_after_unpublish_is_discarded_before_model_log_or_result(tmp_path, monkeypatch):
    storage, user, document_id, state, _request_value, provider, completions, guard = _live_guard_fixture(
        tmp_path, monkeypatch
    )
    original_create = completions.create

    def revoke_while_blocked(**request):
        storage.set_document_publication(document_id, published=False, actor_id=int(user["id"]))
        return original_create(**request)

    completions.create = revoke_while_blocked
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        model_call_guard=guard,
    )

    with pytest.raises(ModelCallPreflightError) as caught:
        orchestrator._decide(state, "late-step", None)

    assert caught.value.error_code == "MODEL_CONTEXT_REVOKED"
    assert len(completions.requests) == 1
    assert storage.model_call_rows(state.run_id) == []
    assert state.final_answer is None


def test_internal_http_retry_rechecks_and_does_not_send_stale_messages(tmp_path, monkeypatch):
    storage, user, document_id, state, request, provider, completions, guard = _live_guard_fixture(
        tmp_path, monkeypatch
    )
    provider.settings = replace(provider.settings, llm_max_retries=1)

    def fail_first_attempt(**outbound):
        completions.requests.append(outbound)
        storage.set_document_publication(document_id, published=False, actor_id=int(user["id"]))
        raise RuntimeError("synthetic transient failure")

    completions.create = fail_first_attempt
    orchestrator = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        model_call_guard=guard,
    )

    with pytest.raises(ModelCallPreflightError) as caught:
        orchestrator._guarded_model_call(
            provider=provider,
            state=state,
            request=request,
            phase="answer",
            call=lambda: provider.generate_answer(request),
        )

    assert caught.value.error_code == "MODEL_CONTEXT_REVOKED"
    assert len(completions.requests) == 1


def test_ended_learning_session_blocks_next_outbound_and_template_remains_local(tmp_path, monkeypatch):
    storage, user, _document_id, state, request, provider, completions, guard = _live_guard_fixture(
        tmp_path, monkeypatch, with_session=True
    )
    guard.check_request_start()
    LearningEventService(storage).end_session(user=user, client_session_id="guard-session")

    with pytest.raises(ModelCallPreflightError) as caught:
        guard(
            provider=provider,
            state=state,
            request=request,
            phase="answer",
            stage="before",
        )
    assert caught.value.error_code == "SESSION_ENDED"
    assert caught.value.status_code == 409
    assert completions.requests == []

    local_guard = main._RequestModelCallGuard(
        provider=TemplateProvider(),
        user_id=int(user["id"]),
        course_id=int(state.course_id),
        class_id=int(state.class_id),
        client_session_id=None,
        use_course_references=False,
        require_course_context=True,
    )
    local = TemplateProvider()
    local_result = AgentOrchestrator(
        storage=storage,
        planner_provider=local,
        answer_provider=local,
        model_call_guard=local_guard,
    )._guarded_model_call(
        provider=local,
        state=state,
        request=request,
        phase="answer",
        call=lambda: local.generate_answer(request),
    )
    assert local_result.provider == "template"
    assert local_guard.last_policy_status == "local"


@pytest.mark.parametrize("endpoint", ["chat", "continue", "practice"])
def test_late_revocation_reaches_stable_endpoint_error_without_question_event_or_result(
    tmp_path,
    monkeypatch,
    endpoint,
):
    storage, user, document_id, state, _request_value, provider, _completions, _guard = _live_guard_fixture(
        tmp_path, monkeypatch, with_session=True
    )
    monkeypatch.setattr(main, "create_llm_provider", lambda: provider)

    class _LateRejectOrchestrator:
        def __init__(self, **kwargs):
            self.guard = kwargs["model_call_guard"]
            self.provider = kwargs["answer_provider"]

        def _reject(self, *, user_id, course_id, class_id):
            storage.set_document_publication(document_id, published=False, actor_id=int(user["id"]))
            late_state = state.model_copy(
                update={
                    "user_id": str(user_id),
                    "course_id": int(course_id),
                    "class_id": class_id,
                }
            )
            self.guard(
                provider=self.provider,
                state=late_state,
                request=None,
                phase="answer",
                stage="after",
            )
            raise AssertionError("late model result must not be accepted")

        def run(self, _text, *, user_id, course_id, class_id, **_kwargs):
            return self._reject(user_id=user_id, course_id=course_id, class_id=class_id)

        def continue_run(self, _run_id, _text, *, user_id, course_id, class_id, **_kwargs):
            return self._reject(user_id=user_id, course_id=course_id, class_id=class_id)

    monkeypatch.setattr(main, "AgentOrchestrator", _LateRejectOrchestrator)
    with storage._connect() as conn:
        event_count_before = int(conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0])
        question_count_before = int(conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0])

    run_id = "late-continue-run"
    if endpoint == "continue":
        run_state = state.model_copy(
            update={
                "run_id": run_id,
                "status": AgentRunStatus.REQUIRES_INPUT,
                "pending_question": "补充条件",
            }
        )
        storage.create_agent_run(
            run_id=run_id,
            user_id=str(user["id"]),
            goal=state.user_goal,
            intent="course_qa",
            status=AgentRunStatus.REQUIRES_INPUT.value,
            max_steps=state.max_steps,
            max_tool_calls=state.max_tool_calls,
            state=run_state.model_dump(mode="json"),
            course_id=int(state.course_id),
            class_id=int(state.class_id),
        )

    with pytest.raises(HTTPException) as caught:
        if endpoint == "chat":
            main.chat(
                ChatRequest(
                    question="解释当前课程资料",
                    course_id=int(state.course_id),
                    class_id=int(state.class_id),
                    client_session_id="guard-session",
                    use_course_references=True,
                ),
                None,
                user,
            )
        elif endpoint == "continue":
            main.continue_agent_run(
                run_id,
                AgentContinueRequest(
                    user_input="继续",
                    client_session_id="guard-session",
                    use_course_references=True,
                ),
                None,
                user,
            )
        else:
            main.practice_guide(
                PracticeGuideRequest(
                    topic="当前课程实践",
                    course_id=int(state.course_id),
                    class_id=int(state.class_id),
                    client_session_id="guard-session",
                    use_course_references=True,
                ),
                None,
                user,
            )

    assert caught.value.status_code == 403
    assert caught.value.detail["error_code"] == "MODEL_CONTEXT_REVOKED"
    with storage._connect() as conn:
        assert int(conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0]) == event_count_before
        assert int(conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]) == question_count_before
    if endpoint == "continue":
        record = storage.agent_run(run_id)
        assert record is not None
        assert record["final_answer"] is None
