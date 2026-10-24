from __future__ import annotations

from backend.agent.observation import tool_result_to_observation
from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.router import route_intent
from backend.agent.schemas import ActionDecision, AgentAction
from backend.agent.state import AgentRunStatus
from backend.llm.base import LLMProvider
from backend.llm.schemas import GroundedAnswer, GroundedSpanAnswer, LLMRequest, LLMResponse
from backend.storage.logger import LearningLogger
from mcp_server.schemas import ToolResult
from mcp_server.tools import TOOL_DEFINITIONS


class J3Provider(LLMProvider):
    provider_name = "j3-mock"
    model_name = "j3-mock-model"

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.answer_requests: list[LLMRequest] = []
        self.structured_requests: list[tuple[LLMRequest, type]] = []

    def decide_next_action(self, state_summary, available_tools):
        return self.decisions.pop(0)

    def generate_answer(self, request: LLMRequest) -> LLMResponse:
        self.answer_requests.append(request)
        lines = [
            f"{item.title}: {item.text.rstrip(' .')} [{item.source_id}]."
            for item in request.evidence
        ]
        return LLMResponse(
            text="\n".join(lines),
            provider=self.provider_name,
            model=self.model_name,
        )

    def generate_structured(self, request, schema):
        self.structured_requests.append((request, schema))
        if schema is GroundedSpanAnswer:
            spans = request.metadata.get("evidence_spans", [])
            claims = []
            for item in request.evidence:
                span = next(
                    (
                        candidate
                        for candidate in spans
                        if candidate.get("source_id") == item.source_id
                        and candidate.get("field") == "text"
                    ),
                    None,
                )
                if span is None:
                    raise AssertionError(f"Missing text span for {item.source_id}: {spans}")
                claims.append(
                    {
                        "claim_text": span["exact_text"],
                        "source_id": span["source_id"],
                        "evidence_span_id": span["span_id"],
                    }
                )
            return schema.model_validate(
                {"claims": claims}
            )
        if schema is GroundedAnswer:
            return schema.model_validate(
                {
                    "claims": [
                        {
                            "claim_text": "该论文证据可用于5G NAS安全比较。",
                            "source_id": "S1",
                            "evidence_quote": "Verified paper evidence for 5G NAS security.",
                        },
                        {
                            "claim_text": "该漏洞证据可用于5G NAS安全比较。",
                            "source_id": "S2",
                            "evidence_quote": "Verified vulnerability evidence for 5G NAS security.",
                        },
                        {
                            "claim_text": "该开源项目证据可用于5G NAS安全比较。",
                            "source_id": "S3",
                            "evidence_quote": "Verified project evidence for 5G NAS security.",
                        },
                    ]
                }
            )
        raise AssertionError(f"Unexpected structured schema: {schema}")


class BadGroundedProvider(J3Provider):
    def __init__(self, decisions, claims):
        super().__init__(decisions)
        self.claims = claims

    def generate_structured(self, request, schema):
        self.structured_requests.append((request, schema))
        if schema is GroundedSpanAnswer:
            return schema.model_validate({"claims": self.claims})
        if schema is GroundedAnswer:
            return schema.model_validate({"claims": self.claims})
        raise AssertionError(f"Unexpected structured schema: {schema}")


class RetryGroundedProvider(J3Provider):
    provider_name = "deepseek"
    model_name = "deepseek-flash"

    def __init__(self, decisions):
        super().__init__(decisions)
        self.answer_calls = 0

    def generate_structured(self, request, schema):
        self.structured_requests.append((request, schema))
        if schema is not GroundedSpanAnswer:
            return super().generate_structured(request, schema)
        spans = request.metadata.get("evidence_spans", [])

        def span_id(source_id: str, field: str = "text") -> str:
            for span in spans:
                if span.get("source_id") == source_id and span.get("field") == field:
                    return span["span_id"]
            raise AssertionError(f"Missing span for {source_id}:{field}: {spans}")

        self.answer_calls += 1
        if self.answer_calls == 1:
            return schema.model_validate(
                {
                    "claims": [
                        {
                            "claim_text": "Verified paper evidence for 5G NAS security",
                            "source_id": "S1",
                            "evidence_span_id": span_id("S1"),
                        },
                        {
                            "claim_text": (
                                "Verified vulnerability evidence for 5G NAS security includes remote code execution"
                            ),
                            "source_id": "S2",
                            "evidence_span_id": span_id("S2"),
                        },
                        {
                            "claim_text": "Verified project evidence for 5G NAS security",
                            "source_id": "S3",
                            "evidence_span_id": span_id("S3"),
                        },
                    ]
                }
            )
        return schema.model_validate(
            {
                "claims": [
                    {
                        "claim_text": "Verified paper evidence for 5G NAS security",
                        "source_id": "S1",
                        "evidence_span_id": span_id("S1"),
                    },
                    {
                        "claim_text": "Verified vulnerability evidence for 5G NAS security",
                        "source_id": "S2",
                        "evidence_span_id": span_id("S2"),
                    },
                    {
                        "claim_text": "Verified project evidence for 5G NAS security",
                        "source_id": "S3",
                        "evidence_span_id": span_id("S3"),
                    },
                ]
            }
        )


class EmbeddedCitationRetryProvider(J3Provider):
    provider_name = "deepseek"
    model_name = "deepseek-flash"

    def __init__(self, decisions, *, second_attempt: str = "valid"):
        super().__init__(decisions)
        self.answer_calls = 0
        self.second_attempt = second_attempt

    def generate_structured(self, request, schema):
        self.structured_requests.append((request, schema))
        if schema is not GroundedSpanAnswer:
            return super().generate_structured(request, schema)
        spans = request.metadata.get("evidence_spans", [])

        def span_id(source_id: str, field: str = "text") -> str:
            for span in spans:
                if span.get("source_id") == source_id and span.get("field") == field:
                    return span["span_id"]
            raise AssertionError(f"Missing span for {source_id}:{field}: {spans}")

        self.answer_calls += 1
        if self.answer_calls == 1 or self.second_attempt == "embedded":
            marker = "[S1]" if self.answer_calls == 1 else "[S2]"
            return schema.model_validate(
                {
                    "claims": [
                        {
                            "claim_text": f"Verified paper evidence for 5G NAS security {marker}",
                            "source_id": "S1",
                            "evidence_span_id": span_id("S1"),
                        },
                        {
                            "claim_text": "Verified vulnerability evidence for 5G NAS security",
                            "source_id": "S2",
                            "evidence_span_id": span_id("S2"),
                        },
                        {
                            "claim_text": "Verified project evidence for 5G NAS security",
                            "source_id": "S3",
                            "evidence_span_id": span_id("S3"),
                        },
                    ]
                }
            )
        if self.second_attempt == "overreach":
            return schema.model_validate(
                {
                    "claims": [
                        {
                            "claim_text": "Verified paper evidence for 5G NAS security",
                            "source_id": "S1",
                            "evidence_span_id": span_id("S1"),
                        },
                        {
                            "claim_text": (
                                "Verified vulnerability evidence for 5G NAS security includes remote code execution"
                            ),
                            "source_id": "S2",
                            "evidence_span_id": span_id("S2"),
                        },
                        {
                            "claim_text": "Verified project evidence for 5G NAS security",
                            "source_id": "S3",
                            "evidence_span_id": span_id("S3"),
                        },
                    ]
                }
            )
        return schema.model_validate(
            {
                "claims": [
                    {
                        "claim_text": "Verified paper evidence for 5G NAS security",
                        "source_id": "S1",
                        "evidence_span_id": span_id("S1"),
                    },
                    {
                        "claim_text": "Verified vulnerability evidence for 5G NAS security",
                        "source_id": "S2",
                        "evidence_span_id": span_id("S2"),
                    },
                    {
                        "claim_text": "Verified project evidence for 5G NAS security",
                        "source_id": "S3",
                        "evidence_span_id": span_id("S3"),
                    },
                ]
            }
        )


class J3Gateway:
    def __init__(self, *, verified: bool = True):
        self.verified = verified
        self.calls: list[str] = []

    def get_agent_tool_catalog(self, role="student"):
        return [
            {
                "name": name,
                "description": name,
                "input_schema": {},
                "risk": "LOW",
                "network": name != "standard_metadata_lookup",
                "read_only": True,
                "enabled": True,
            }
            for name in (
                "paper_search_real",
                "vuln_lookup_real",
                "standard_metadata_lookup",
                "github_project_search",
            )
        ]

    def call_tool(self, name, arguments, *, run_id, step_id, step_number=1, role="student"):
        self.calls.append(name)
        resource_type = {
            "paper_search_real": "paper",
            "vuln_lookup_real": "vulnerability",
            "standard_metadata_lookup": "standard",
            "github_project_search": "project",
        }[name]
        resource = {
            "id": len(self.calls),
            "external_id": f"verified-{resource_type}",
            "resource_type": resource_type,
            "source_name": "standard_source" if resource_type == "standard" else resource_type,
            "title": f"Verified {resource_type} evidence",
            "abstract": f"Verified {resource_type} evidence for 5G NAS security.",
            "url": f"https://example.test/{resource_type}",
            "fetched_at": "2026-09-30T00:00:00Z",
            "origin_kind": "local_catalog" if resource_type == "standard" else "remote",
            "delivery_mode": "fresh_cache",
            "content_hash": f"hash-{resource_type}",
            "provenance_verified": self.verified,
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
            run_id=run_id,
            step_id=step_id,
            arguments=arguments,
        )
        call.fallback_used = False
        return call


def _decision(tool_name: str) -> ActionDecision:
    return ActionDecision(
        action=AgentAction.CALL_TOOL,
        tool_name=tool_name,
        arguments={"query": "5G NAS security", "top_k": 1},
        reason=f"need {tool_name}",
    )


def test_successful_duplicate_tool_signature_is_not_executed_twice(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    provider = J3Provider(
        [
            _decision("standard_metadata_lookup"),
            _decision("standard_metadata_lookup"),
            ActionDecision(action=AgentAction.ANSWER, arguments={}, reason="enough", task_complete=True),
        ]
    )
    gateway = J3Gateway()
    storage = LearningLogger(tmp_path / "agent.db")
    result = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        mcp_gateway=gateway,
    ).run("Check the official metadata for 3GPP TS 24.501", user_id="u1")

    assert result["run_status"] == AgentRunStatus.COMPLETED
    assert gateway.calls == ["standard_metadata_lookup"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["resource_type"] == "standard"
    assert provider.answer_requests == []
    assert len(provider.structured_requests) == 1
    request, schema = provider.structured_requests[0]
    assert schema is GroundedSpanAnswer
    assert request.evidence[0].source_url == "https://example.test/standard"


def test_three_trusted_tool_sources_reach_final_generation_once_each(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    for name in ("paper_search_real", "vuln_lookup_real", "github_project_search"):
        monkeypatch.setitem(
            TOOL_DEFINITIONS,
            name,
            TOOL_DEFINITIONS[name].model_copy(update={"enabled": True}),
        )
    provider = J3Provider(
        [
            _decision("paper_search_real"),
            _decision("vuln_lookup_real"),
            _decision("github_project_search"),
            ActionDecision(action=AgentAction.ANSWER, arguments={}, reason="all evidence ready", task_complete=True),
        ]
    )
    gateway = J3Gateway()
    storage = LearningLogger(tmp_path / "agent.db")
    result = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        mcp_gateway=gateway,
    ).run("Compare a research paper, CVE, and GitHub project for 5G NAS", user_id="u1")

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert gateway.calls == ["paper_search_real", "vuln_lookup_real", "github_project_search"]
    assert len(result["citations"]) == 3
    assert provider.answer_requests == []
    assert len(provider.structured_requests) == 1
    assert provider.structured_requests[0][1] is GroundedSpanAnswer
    metadata = storage.model_call_rows(result["run_id"])[-1]["metadata"]
    assert metadata["evidence_ids"] == ["S1", "S2", "S3"]
    assert metadata["output_contract"] == "grounded_claims_v2"
    assert metadata["claim_count"] == 3
    assert metadata["resolved_evidence_span_ids"] == ["S1:text:1", "S2:text:1", "S3:text:1"]
    assert {"S1:title:1", "S2:title:1", "S3:title:1"}.issubset(set(metadata["evidence_span_ids"]))
    assert metadata["resource_ids"] == [
        "verified-paper",
        "verified-vulnerability",
        "verified-project",
    ]
    assert metadata["source_urls"] == [
        "https://example.test/paper",
        "https://example.test/vulnerability",
        "https://example.test/project",
    ]
    assert len(metadata["content_hashes"]) == 3


def test_grounded_answer_retries_once_after_selected_span_overreach(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    for name in ("paper_search_real", "vuln_lookup_real", "github_project_search"):
        monkeypatch.setitem(
            TOOL_DEFINITIONS,
            name,
            TOOL_DEFINITIONS[name].model_copy(update={"enabled": True}),
        )
    provider = RetryGroundedProvider(
        [
            _decision("paper_search_real"),
            _decision("vuln_lookup_real"),
            _decision("github_project_search"),
            ActionDecision(action=AgentAction.ANSWER, arguments={}, reason="all evidence ready", task_complete=True),
        ]
    )
    gateway = J3Gateway()
    storage = LearningLogger(tmp_path / "agent-grounded-retry.db")
    result = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=J3Provider([]),
        mcp_gateway=gateway,
    ).run("Compare a research paper, CVE, and GitHub project for 5G NAS", user_id="u1")

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert result["fallback_used"] is False
    assert gateway.calls == ["paper_search_real", "vuln_lookup_real", "github_project_search"]
    assert provider.answer_calls == 2
    assert len(provider.structured_requests) == 2
    rows = [row for row in storage.model_call_rows(result["run_id"]) if row["purpose"] == "generate_answer"]
    attempts = [row["metadata"].get("grounding_retry_attempt") for row in rows]
    assert attempts == [0, 1]
    assert rows[1]["metadata"]["previous_guard_error_code"] == "UNSUPPORTED_CITATION"
    assert rows[1]["metadata"]["output_contract"] == "grounded_claims_v2"
    retry_request = provider.structured_requests[1][0]
    guidance = retry_request.metadata["grounded_retry_guidance"]
    assert retry_request.metadata["retry_reason"] == "unsupported_citation"
    assert "copy the selected exact_text" in guidance
    assert "keep the same language" in guidance
    assert "we/authors" in guidance
    assert "The paper is titled <exact title>" in guidance
    assert "The NVD record is identified as <exact CVE>" in guidance
    assert "The GitHub repository is <exact owner/repo>" in guidance
    assert "Bind title, external_id, and text facts to separate spans" in guidance
    assert len(result["citations"]) == 3


def test_grounded_answer_retries_once_after_embedded_citation_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    for name in ("paper_search_real", "vuln_lookup_real", "github_project_search"):
        monkeypatch.setitem(
            TOOL_DEFINITIONS,
            name,
            TOOL_DEFINITIONS[name].model_copy(update={"enabled": True}),
        )
    provider = EmbeddedCitationRetryProvider(
        [
            _decision("paper_search_real"),
            _decision("vuln_lookup_real"),
            _decision("github_project_search"),
            ActionDecision(action=AgentAction.ANSWER, arguments={}, reason="all evidence ready", task_complete=True),
        ]
    )
    gateway = J3Gateway()
    storage = LearningLogger(tmp_path / "agent-grounded-embedded-retry.db")
    result = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=J3Provider([]),
        mcp_gateway=gateway,
    ).run("Compare a research paper, CVE, and GitHub project for 5G NAS", user_id="u1")

    assert result["run_status"] == AgentRunStatus.COMPLETED, result
    assert result["fallback_used"] is False
    assert gateway.calls == ["paper_search_real", "vuln_lookup_real", "github_project_search"]
    assert provider.answer_calls == 2
    rows = [row for row in storage.model_call_rows(result["run_id"]) if row["purpose"] == "generate_answer"]
    assert [row["success"] for row in rows] == [False, True]
    assert [row["error_code"] for row in rows] == ["GROUNDING_EMBEDDED_CITATION", None]
    assert [row["metadata"].get("grounding_retry_attempt") for row in rows] == [0, 1]
    assert rows[0]["metadata"]["retry_reason"] == "embedded_citation"
    assert rows[0]["metadata"]["candidate_accepted"] is False
    assert rows[1]["metadata"]["candidate_accepted"] is True
    assert rows[1]["metadata"]["previous_error_code"] == "GROUNDING_EMBEDDED_CITATION"
    assert len(result["citations"]) == 3


def test_grounded_answer_embedded_retry_does_not_chain_support_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    for name in ("paper_search_real", "vuln_lookup_real", "github_project_search"):
        monkeypatch.setitem(
            TOOL_DEFINITIONS,
            name,
            TOOL_DEFINITIONS[name].model_copy(update={"enabled": True}),
        )
    provider = EmbeddedCitationRetryProvider(
        [
            _decision("paper_search_real"),
            _decision("vuln_lookup_real"),
            _decision("github_project_search"),
            ActionDecision(action=AgentAction.ANSWER, arguments={}, reason="all evidence ready", task_complete=True),
        ],
        second_attempt="overreach",
    )
    gateway = J3Gateway()
    storage = LearningLogger(tmp_path / "agent-grounded-no-third.db")
    result = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=J3Provider([]),
        mcp_gateway=gateway,
    ).run("Compare a research paper, CVE, and GitHub project for 5G NAS", user_id="u1")

    assert result["run_status"] == AgentRunStatus.ABSTAINED, result
    assert result["fallback_used"] is False
    assert result["stop_reason"] == "UNSUPPORTED_CITATION"
    assert gateway.calls == ["paper_search_real", "vuln_lookup_real", "github_project_search"]
    assert provider.answer_calls == 2
    rows = [row for row in storage.model_call_rows(result["run_id"]) if row["purpose"] == "generate_answer"]
    assert [row["metadata"].get("grounding_retry_attempt") for row in rows] == [0, 1]
    assert len(rows) == 2


def test_grounded_answer_rejects_repeated_embedded_citation_without_template(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    for name in ("paper_search_real", "vuln_lookup_real", "github_project_search"):
        monkeypatch.setitem(
            TOOL_DEFINITIONS,
            name,
            TOOL_DEFINITIONS[name].model_copy(update={"enabled": True}),
        )
    provider = EmbeddedCitationRetryProvider(
        [
            _decision("paper_search_real"),
            _decision("vuln_lookup_real"),
            _decision("github_project_search"),
            ActionDecision(action=AgentAction.ANSWER, arguments={}, reason="all evidence ready", task_complete=True),
        ],
        second_attempt="embedded",
    )
    storage = LearningLogger(tmp_path / "agent-grounded-embedded-twice.db")
    result = AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=J3Provider([]),
        mcp_gateway=J3Gateway(),
    ).run("Compare a research paper, CVE, and GitHub project for 5G NAS", user_id="u1")

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["fallback_used"] is False
    assert result["stop_reason"] == "GROUNDING_EMBEDDED_CITATION"
    rows = [row for row in storage.model_call_rows(result["run_id"]) if row["purpose"] == "generate_answer"]
    assert [row["success"] for row in rows] == [False, False]
    assert [row["error_code"] for row in rows] == [
        "GROUNDING_EMBEDDED_CITATION",
        "GROUNDING_EMBEDDED_CITATION",
    ]
    assert [row["metadata"].get("grounding_retry_attempt") for row in rows] == [0, 1]
    assert provider.answer_calls == 2


def test_grounded_answer_rejects_quote_from_wrong_source(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    for name in ("paper_search_real", "vuln_lookup_real", "github_project_search"):
        monkeypatch.setitem(
            TOOL_DEFINITIONS,
            name,
            TOOL_DEFINITIONS[name].model_copy(update={"enabled": True}),
        )
    provider = BadGroundedProvider(
        [
            _decision("paper_search_real"),
            _decision("vuln_lookup_real"),
            _decision("github_project_search"),
            ActionDecision(action=AgentAction.ANSWER, arguments={}, reason="all evidence ready", task_complete=True),
        ],
        [
            {
                "claim_text": "The paper claim is bound to the wrong quote.",
                "source_id": "S1",
                "evidence_span_id": "S2:text:1",
            },
            {
                "claim_text": "5G NAS vulnerability evidence is available.",
                "source_id": "S2",
                "evidence_span_id": "S2:text:1",
            },
            {
                "claim_text": "5G NAS project evidence is available.",
                "source_id": "S3",
                "evidence_span_id": "S3:text:1",
            },
        ],
    )
    result = AgentOrchestrator(
        storage=LearningLogger(tmp_path / "agent.db"),
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        mcp_gateway=J3Gateway(),
    ).run("Compare a research paper, CVE, and GitHub project for 5G NAS", user_id="u1")

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["citations"] == []
    assert result["stop_reason"] == "GROUNDING_SPAN_SOURCE_MISMATCH"


def test_grounded_answer_rejects_missing_required_source(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    for name in ("paper_search_real", "vuln_lookup_real", "github_project_search"):
        monkeypatch.setitem(
            TOOL_DEFINITIONS,
            name,
            TOOL_DEFINITIONS[name].model_copy(update={"enabled": True}),
        )
    provider = BadGroundedProvider(
        [
            _decision("paper_search_real"),
            _decision("vuln_lookup_real"),
            _decision("github_project_search"),
            ActionDecision(action=AgentAction.ANSWER, arguments={}, reason="all evidence ready", task_complete=True),
        ],
        [
            {
                "claim_text": "5G NAS paper evidence is available.",
                "source_id": "S1",
                "evidence_span_id": "S1:text:1",
            },
            {
                "claim_text": "5G NAS vulnerability evidence is available.",
                "source_id": "S2",
                "evidence_span_id": "S2:text:1",
            },
        ],
    )
    result = AgentOrchestrator(
        storage=LearningLogger(tmp_path / "agent.db"),
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        mcp_gateway=J3Gateway(),
    ).run("Compare a research paper, CVE, and GitHub project for 5G NAS", user_id="u1")

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["citations"] == []
    assert result["stop_reason"] == "GROUNDING_SOURCE_COVERAGE"


def test_unverified_tool_resource_never_becomes_answer_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    monkeypatch.setitem(
        TOOL_DEFINITIONS,
        "paper_search_real",
        TOOL_DEFINITIONS["paper_search_real"].model_copy(update={"enabled": True}),
    )
    provider = J3Provider([_decision("paper_search_real"), _decision("paper_search_real")])
    gateway = J3Gateway(verified=False)
    result = AgentOrchestrator(
        storage=LearningLogger(tmp_path / "agent.db"),
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        mcp_gateway=gateway,
    ).run("Find a nonexistent research paper", user_id="u1")

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["citations"] == []
    assert provider.answer_requests == []


def test_j3_intent_priority_and_multi_source_routing():
    assert route_intent("Create a six-step practice guide with paper evidence") == "practice_guide"
    assert route_intent("Recommend papers and GitHub projects with innovation questions") == "innovation"
    assert route_intent("Compare a paper, CVE, and GitHub project") == "multi_source"
    assert route_intent("Recommend a dessert shop for travel") == "off_topic"
