import hashlib

import pytest

from backend.agent.orchestrator import (
    GroundedAnswerValidationError,
    _build_grounded_render_units,
    _build_evidence_spans,
    _grounded_span_citation_bindings,
    _identifier_occurrences,
    _render_grounded_answer,
    _paper_span_semantics,
    _validate_grounded_answer,
    _validate_grounded_span_answer,
)
from backend.guards.citation_guard import CitationGuard, _external_id_span_is_compatible
from backend.llm.prompts import build_structured_messages
from backend.llm.schemas import EvidenceItem, EvidenceSpan, GroundedAnswer, GroundedClaim, GroundedSpanAnswer, LLMRequest


def _external(
    source_id: str,
    *,
    text: str,
    title: str,
    external_id: str,
    url: str,
    resource_type: str,
    trusted: bool = True,
    tool_name: str = "",
) -> EvidenceItem:
    return EvidenceItem(
        source_id=source_id,
        text=text,
        source=resource_type,
        title=title,
        source_url=url,
        external_id=external_id,
        resource_type=resource_type,
        origin_kind="remote",
        delivery_mode="live",
        content_hash=f"hash-{source_id.lower()}",
        tool_name=tool_name,
        trusted=trusted,
    )


def _evidence() -> list[EvidenceItem]:
    return [
        _external(
            "S1",
            text=(
                "The paper studies 5G NAS replay detection with state-machine analysis. "
                "The paper reports a NAS trace dataset for validation."
            ),
            title="5G NAS Replay Detection Study",
            external_id="2610.01001",
            url="https://arxiv.org/abs/2610.01001",
            resource_type="paper",
        ),
        _external(
            "S2",
            text=(
                "CVE-2024-12345 is an NVD record for improper input validation in a lab parser. "
                "The NVD detail page is https://nvd.nist.gov/vuln/detail/CVE-2024-12345."
            ),
            title="CVE-2024-12345",
            external_id="CVE-2024-12345",
            url="https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
            resource_type="vulnerability",
        ),
        _external(
            "S3",
            text="The GitHub repository open5gs/open5gs provides an open-source 5G core for labs.",
            title="open5gs/open5gs",
            external_id="open5gs/open5gs",
            url="https://github.com/open5gs/open5gs",
            resource_type="project",
        ),
    ]


def _identifier_edge_evidence() -> list[EvidenceItem]:
    return [
        _external(
            "S1",
            text=(
                "The paper discusses 5G/NAS security experiments. "
                "The paper span explicitly mentions CVE-2024-12345 for comparison. "
                "The arXiv landing page is https://arxiv.org/abs/2610.01001."
            ),
            title="5G NAS Replay Detection Study",
            external_id="2610.01001",
            url="https://arxiv.org/abs/2610.01001",
            resource_type="paper",
        ),
        _external(
            "S2",
            text=(
                "CVE-2024-12345 is an NVD record for improper input validation in a lab parser. "
                "The NVD detail page is https://nvd.nist.gov/vuln/detail/CVE-2024-12345."
            ),
            title="CVE-2024-12345",
            external_id="CVE-2024-12345",
            url="https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
            resource_type="vulnerability",
        ),
        _external(
            "S3",
            text=(
                "The GitHub repository open5gs/open5gs provides an open-source 5G core for labs. "
                "The project notes LTE/NR lab deployment terminology. "
                "The project span also mentions CVE-2024-12345 as a reproduced classroom sample. "
                "The GitHub project page is https://github.com/open5gs/open5gs."
            ),
            title="open5gs/open5gs",
            external_id="open5gs/open5gs",
            url="https://github.com/open5gs/open5gs",
            resource_type="project",
        ),
    ]


def _canonical_identifier_evidence() -> list[EvidenceItem]:
    return [
        _external(
            "S1",
            text=(
                "The title reports the current state of affairs in 5G security. "
                "The method section analyzes NAS authentication traces without a performance number."
            ),
            title="The Current State of Affairs in 5G Security and the Main Remaining Security Challenges",
            external_id="1904.08394v2",
            url="https://arxiv.org/abs/1904.08394v2",
            resource_type="paper",
        ),
        _external(
            "S4",
            text="This other paper discusses radio access security.",
            title="Another 5G Security Paper",
            external_id="2103.08436v1",
            url="https://arxiv.org/abs/2103.08436v1",
            resource_type="paper",
        ),
        _external(
            "S9",
            text=(
                "The selected paragraph studies NAS state-machine checking. "
                "A separate background paragraph mentions 9999.00001v1 as unrelated prior work."
            ),
            title="Paper With Background Identifier",
            external_id="2501.11111v1",
            url="https://arxiv.org/abs/2501.11111v1",
            resource_type="paper",
        ),
        _external(
            "S2",
            text="The NVD description discusses an improper input validation issue in a lab parser.",
            title="CVE-2024-12345",
            external_id="CVE-2024-12345",
            url="https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
            resource_type="vulnerability",
        ),
        _external(
            "S5",
            text="Another NVD entry documents a logging library injection vulnerability.",
            title="CVE-2021-44228",
            external_id="CVE-2021-44228",
            url="https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
            resource_type="vulnerability",
        ),
        _external(
            "S3",
            text="The project description says the lab uses an open-source 5G core for experiments.",
            title="Open5GS Core",
            external_id="80611216",
            url="https://github.com/open5gs/open5gs",
            resource_type="project",
        ),
        _external(
            "S6",
            text="The project description says the lab compares a second open-source 5G core.",
            title="Free5GC Core",
            external_id="free5gc/free5gc",
            url="https://example.org/projects/free5gc",
            resource_type="project",
        ),
        _external(
            "S7",
            text="The project description does not include a repository slug.",
            title="titleonly/repo",
            external_id="987654",
            url="https://example.org/projects/titleonly",
            resource_type="project",
        ),
    ]


def _nvd_span_support_evidence() -> list[EvidenceItem]:
    return [
        _external(
            "S4",
            text=(
                "CVE-2021-44228 allows JNDI LDAP remote code execution in a logging component. "
                "CVSS 10.0 CRITICAL and CWE-20 with published 2021-12-10 and modified 2021-12-18."
            ),
            title="CVE-2021-44228",
            external_id="CVE-2021-44228",
            url="https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
            resource_type="vulnerability",
        )
    ]


def _answer(claims: list[dict[str, str]]) -> GroundedAnswer:
    return GroundedAnswer.model_validate({"claims": claims})


def _span_answer(claims: list[dict[str, str]]) -> GroundedSpanAnswer:
    return GroundedSpanAnswer.model_validate({"claims": claims})


def _span_id(evidence: list[EvidenceItem], source_id: str, field: str, index: int = 1) -> str:
    spans = _build_evidence_spans(evidence)
    wanted = f"{source_id}:{field}:{index}"
    assert any(span.span_id == wanted for span in spans), [span.span_id for span in spans]
    return wanted


def _span_map(spans: list[EvidenceSpan]) -> dict[str, EvidenceSpan]:
    return {span.span_id: span for span in spans}


def _bindings(answer: GroundedSpanAnswer, claims: list[GroundedClaim]) -> list[dict[str, str]]:
    return _grounded_span_citation_bindings(_build_grounded_render_units(answer, claims))


def _renderer_drift_span_answer(evidence: list[EvidenceItem]) -> GroundedSpanAnswer:
    return _span_answer(
        [
            {
                "claim_text": "The paper title is 5G NAS Replay Detection Study.",
                "source_id": "S1",
                "evidence_span_id": _span_id(evidence, "S1", "title"),
            },
            {
                "claim_text": "The paper reports a NAS trace dataset for validation.",
                "source_id": "S1",
                "evidence_span_id": _span_id(evidence, "S1", "text", 2),
            },
            {
                "claim_text": "The paper studies 5G NAS replay detection with state-machine analysis.",
                "source_id": "S1",
                "evidence_span_id": _span_id(evidence, "S1", "text", 1),
            },
            {
                "claim_text": "The NVD identifier is CVE-2024-12345.",
                "source_id": "S2",
                "evidence_span_id": _span_id(evidence, "S2", "external_id"),
            },
            {
                "claim_text": "CVE-2024-12345 is an NVD record for improper input validation in a lab parser.",
                "source_id": "S2",
                "evidence_span_id": _span_id(evidence, "S2", "text", 1),
            },
            {
                "claim_text": "The NVD detail page is https://nvd.nist.gov/vuln/detail/CVE-2024-12345.",
                "source_id": "S2",
                "evidence_span_id": _span_id(evidence, "S2", "text", 2),
            },
            {
                "claim_text": "The GitHub repository is open5gs/open5gs.",
                "source_id": "S3",
                "evidence_span_id": _span_id(evidence, "S3", "title"),
            },
            {
                "claim_text": "The GitHub repository open5gs/open5gs provides an open-source 5G core for labs.",
                "source_id": "S3",
                "evidence_span_id": _span_id(evidence, "S3", "text"),
            },
        ]
    )


def test_renderer_and_bindings_share_canonical_claim_text_for_trailing_punctuation():
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    span_answer = _renderer_drift_span_answer(evidence)
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["paper", "vulnerability", "project"],
    )

    units = _build_grounded_render_units(span_answer, claims)
    rendered = _render_grounded_answer(units)
    bindings = _bindings(span_answer, claims)
    citation = CitationGuard().enforce(
        rendered,
        evidence,
        question="Compare paper, NVD, and GitHub evidence for a 5G NAS lab.",
        required_source_types={"paper", "vulnerability", "project"},
        verified_spans=_span_map(spans),
        verified_bindings=bindings,
    )

    assert len(claims) == 8
    assert len(bindings) == 8
    assert len(units) == 8
    assert "Study. | evidence quote:" not in rendered
    assert bindings[0]["claim_text"] == units[0].claim_text
    assert bindings[0]["claim_text"].endswith("Study")
    assert bindings[0]["source_id"] == units[0].source_id == "S1"
    assert bindings[0]["evidence_span_id"] == units[0].evidence_span_id == "S1:title:1"
    assert hashlib.sha256(bindings[0]["evidence_quote"].encode("utf-8")).hexdigest() == hashlib.sha256(
        units[0].evidence_quote.encode("utf-8")
    ).hexdigest()
    assert [unit.source_id for unit in units] == ["S1", "S1", "S1", "S2", "S2", "S2", "S3", "S3"]
    assert citation.ok is True


def test_project20_presentation_profile_renders_three_canonical_source_paragraphs():
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    span_answer = _renderer_drift_span_answer(evidence)
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["paper", "vulnerability", "project"],
    )
    units = _build_grounded_render_units(span_answer, claims)
    profile = "multi_source_teaching_background_v1"

    rendered = _render_grounded_answer(
        units,
        evidence=evidence,
        presentation_profile=profile,
    )
    citation = CitationGuard().enforce(
        rendered,
        evidence,
        question="用三段话分别说明论文、NVD漏洞和GitHub项目能提供什么教学背景。",
        required_source_types={"paper", "vulnerability", "project"},
        verified_spans=_span_map(spans),
        verified_bindings=_grounded_span_citation_bindings(units),
        verified_presentation_profile=profile,
    )

    paragraphs = rendered.splitlines()
    assert len(paragraphs) == 3
    assert paragraphs[0].startswith("论文教学背景：")
    assert paragraphs[1].startswith("NVD漏洞教学背景：")
    assert paragraphs[2].startswith("GitHub项目教学背景：")
    assert "| evidence quote:" not in rendered
    assert "evidence_span_id" not in rendered
    assert citation.ok is True


@pytest.mark.parametrize("tamper", ["remove", "swap", "claim", "append", "forged"])
def test_project20_presentation_profile_rejects_noncanonical_visible_output(tamper: str):
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    span_answer = _renderer_drift_span_answer(evidence)
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["paper", "vulnerability", "project"],
    )
    units = _build_grounded_render_units(span_answer, claims)
    profile = "multi_source_teaching_background_v1"
    rendered = _render_grounded_answer(units, evidence=evidence, presentation_profile=profile)
    paragraphs = rendered.splitlines()
    variants = {
        "remove": "\n".join(paragraphs[1:]),
        "swap": "\n".join([paragraphs[1], paragraphs[0], paragraphs[2]]),
        "claim": rendered.replace("The paper", "The altered paper", 1),
        "append": rendered + "\n额外自由结论。",
        "forged": rendered.replace("[S1]", "[S999]", 1),
    }

    citation = CitationGuard().enforce(
        variants[tamper],
        evidence,
        question="用三段话分别说明论文、NVD漏洞和GitHub项目能提供什么教学背景。",
        required_source_types={"paper", "vulnerability", "project"},
        verified_spans=_span_map(spans),
        verified_bindings=_grounded_span_citation_bindings(units),
        verified_presentation_profile=profile,
    )

    assert citation.ok is False


def test_project20_presentation_profile_groups_interleaved_units_by_trusted_source_type():
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    answer = _renderer_drift_span_answer(evidence)
    claims = _validate_grounded_span_answer(
        answer,
        evidence,
        spans,
        required_sources=["paper", "vulnerability", "project"],
    )
    units = _build_grounded_render_units(answer, claims)
    interleaved = [units[6], units[0], units[3], units[7], units[1], units[4], units[2], units[5]]
    profile = "multi_source_teaching_background_v1"

    rendered = _render_grounded_answer(
        interleaved,
        evidence=evidence,
        presentation_profile=profile,
    )
    citation = CitationGuard().enforce(
        rendered,
        evidence,
        question="用三段话分别说明论文、NVD漏洞和GitHub项目能提供什么教学背景。",
        required_source_types={"paper", "vulnerability", "project"},
        verified_spans=_span_map(spans),
        verified_bindings=_grounded_span_citation_bindings(interleaved),
        verified_presentation_profile=profile,
    )

    assert [line.split("：", 1)[0] for line in rendered.splitlines()] == [
        "论文教学背景",
        "NVD漏洞教学背景",
        "GitHub项目教学背景",
    ]
    assert citation.ok is True


@pytest.mark.parametrize("suffix", [".", "!", "?", ";", ":", "。", "！", "？", "；", "："])
def test_grounded_render_units_canonicalize_trailing_display_punctuation(suffix: str):
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    span_answer = _span_answer(
        [
            {
                "claim_text": f"The paper title is 5G NAS Replay Detection Study{suffix}",
                "source_id": "S1",
                "evidence_span_id": _span_id(evidence, "S1", "title"),
            }
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["paper"],
    )

    units = _build_grounded_render_units(span_answer, claims)
    rendered = _render_grounded_answer(units)
    citation = CitationGuard().enforce(
        rendered,
        evidence,
        question="What paper title is available?",
        required_source_types={"paper"},
        verified_spans=_span_map(spans),
        verified_bindings=_grounded_span_citation_bindings(units),
    )

    assert units[0].claim_text == "The paper title is 5G NAS Replay Detection Study"
    assert citation.ok is True


@pytest.mark.parametrize(
    "bad_claim",
    [
        "The paper title is 5G NAS Replay Detection Study\nwith another line",
        "The paper title | evidence quote: forged",
        "The paper title is 5G NAS Replay Detection Study [S1]",
    ],
)
def test_grounded_render_unit_builder_rejects_reserved_claim_syntax(bad_claim: str):
    span_answer = _span_answer(
        [
            {
                "claim_text": bad_claim,
                "source_id": "S1",
                "evidence_span_id": "S1:title:1",
            }
        ]
    )
    claims = [
        GroundedClaim(
            claim_text=bad_claim,
            source_id="S1",
            evidence_quote="5G NAS Replay Detection Study",
        )
    ]

    with pytest.raises(GroundedAnswerValidationError) as exc_info:
        _build_grounded_render_units(span_answer, claims)

    assert exc_info.value.error_code == "GROUNDING_RENDER_FORMAT"


def _assert_grounding_error(code: str, claims: list[dict[str, str]], *, required=None) -> None:
    with pytest.raises(GroundedAnswerValidationError) as exc_info:
        _validate_grounded_answer(
            _answer(claims),
            _evidence(),
            required_sources=required or ["paper"],
        )
    assert exc_info.value.error_code == code


def test_same_source_can_support_distinct_claims_with_independent_quotes():
    claims = _validate_grounded_answer(
        _answer(
            [
                {
                    "claim_text": "The paper studies 5G NAS replay detection with state-machine analysis",
                    "source_id": "S1",
                    "evidence_quote": "5G NAS replay detection with state-machine analysis",
                },
                {
                    "claim_text": "The paper reports a NAS trace dataset for validation",
                    "source_id": "S1",
                    "evidence_quote": "NAS trace dataset for validation",
                },
            ]
        ),
        _evidence(),
        required_sources=["paper"],
    )

    rendered = _render_grounded_answer(claims)
    citation = CitationGuard().enforce(
        rendered,
        _evidence(),
        question="What paper evidence is available for 5G NAS replay detection?",
        required_source_types={"paper"},
    )

    assert [claim.source_id for claim in claims] == ["S1", "S1"]
    assert rendered.count("[S1]") == 2
    assert citation.ok is True


def test_grounded_answer_structured_prompt_includes_contract_context():
    messages = build_structured_messages(
        LLMRequest(
            question="Compare paper, CVE, and GitHub evidence.",
            evidence=_evidence(),
            instructions="Cover every required source and keep one fact per claim.",
            metadata={"required_sources": ["paper", "vulnerability", "project"]},
        ),
        GroundedAnswer.__name__,
        GroundedAnswer.model_json_schema(),
    )

    user_message = messages[-1]["content"]
    assert "GroundedAnswer contract" in user_message
    assert "Cover every required source and keep one fact per claim." in user_message
    assert '"source_id": "S1"' in user_message
    assert '"resource_type": "paper"' in user_message
    assert '"vulnerability"' in user_message
    assert '"project"' in user_message
    assert "same trusted source may support multiple distinct claims" in user_message


def test_grounded_span_prompt_includes_span_contract_and_catalog():
    spans = _build_evidence_spans(_evidence())
    messages = build_structured_messages(
        LLMRequest(
            question="Compare paper, CVE, and GitHub evidence.",
            evidence=_evidence(),
            instructions="Use span IDs only.",
            metadata={
                "required_sources": ["paper", "vulnerability", "project"],
                "evidence_spans": [span.model_dump(mode="json") for span in spans],
            },
        ),
        GroundedSpanAnswer.__name__,
        GroundedSpanAnswer.model_json_schema(),
    )

    user_message = messages[-1]["content"]
    assert "GroundedSpanAnswer contract" in user_message
    assert '"span_id": "S1:title:1"' in user_message
    assert "same trusted source may support multiple distinct claims" in user_message
    assert "one atomic fact fully supported by the selected evidence_span_id alone" in user_message
    assert "Do not merge facts from multiple spans" in user_message
    assert "experiment usage unless the selected span states them" in user_message
    assert "The paper is titled <exact title>" in user_message
    assert "The NVD record is identified as <exact CVE>" in user_message
    assert "The GitHub repository is <exact owner/repo>" in user_message
    assert "Bind title and external_id facts to separate spans" in user_message
    assert "copy the selected span's exact_text or a contiguous extractive clause" in user_message
    assert "do not translate, summarize, or replace subjects such as we/authors" in user_message
    assert "Do not copy or rewrite evidence quotes" in user_message
    assert "Use span IDs only." not in user_message


def test_grounded_span_prompt_includes_selected_paper_field_contract():
    evidence = _evidence()[:1]
    spans = _build_evidence_spans(evidence)
    messages = build_structured_messages(
        LLMRequest(
            question="Return paper title, authors, abstract clue, and source link.",
            evidence=evidence,
            metadata={
                "required_sources": ["paper"],
                "evidence_spans": [span.model_dump(mode="json") for span in spans],
                "paper_contract": {
                    "selected_source_ids": ["S1"],
                    "required_fields": ["title", "authors", "abstract_clue", "link"],
                    "span_semantics": {"S1:text:1": "paper_abstract"},
                    "target_count": 1,
                },
            },
        ),
        GroundedSpanAnswer.__name__,
        GroundedSpanAnswer.model_json_schema(),
    )

    user_message = messages[-1]["content"]
    assert "Selected paper field contract" in user_message
    assert '"selected_source_ids": ["S1"]' in user_message
    assert '"required_fields": ["title", "authors", "abstract_clue", "link"]' in user_message


def test_selected_paper_field_gate_checks_author_abstract_and_actual_https_url():
    evidence = [
        _external(
            "S1",
            text=(
                "Author information: Alice Example, Bob Example.\n"
                "Abstract: 5G core network attack detection is evaluated on a defensive dataset."
            ),
            title="5G Core Network Attack Detection",
            external_id="2610.01001",
            url="https://arxiv.org/abs/2610.01001",
            resource_type="paper",
            tool_name="paper_search_real",
        )
    ]
    spans = _build_evidence_spans(evidence)
    semantics = _paper_span_semantics(evidence, spans)
    author_span = next(key for key, value in semantics.items() if value == "paper_author_summary")
    abstract_span = next(key for key, value in semantics.items() if value == "paper_abstract")
    answer = _span_answer(
        [
            {"claim_text": "The paper is titled 5G Core Network Attack Detection", "source_id": "S1", "evidence_span_id": "S1:title:1"},
            {"claim_text": "Author information: Alice Example, Bob Example.", "source_id": "S1", "evidence_span_id": author_span},
            {"claim_text": "Abstract: 5G core network attack detection is evaluated on a defensive dataset.", "source_id": "S1", "evidence_span_id": abstract_span},
        ]
    )
    contract = {
        "selected_source_ids": ["S1"],
        "required_fields": ["title", "authors", "abstract_clue", "link"],
        "span_semantics": semantics,
        "citation_urls": {"S1": "https://arxiv.org/abs/2610.01001"},
    }

    claims = _validate_grounded_span_answer(
        answer,
        evidence,
        spans,
        required_sources=["paper"],
        paper_contract=contract,
    )
    assert len(claims) == 3

    missing_author = _span_answer([answer.claims[0].model_dump(), answer.claims[2].model_dump()])
    with pytest.raises(GroundedAnswerValidationError) as exc_info:
        _validate_grounded_span_answer(
            missing_author,
            evidence,
            spans,
            required_sources=["paper"],
            paper_contract=contract,
        )
    assert exc_info.value.error_code == "PAPER_FIELD_COVERAGE"
    assert exc_info.value.failure_detail == {"source_id": "S1", "missing_fields": ["authors"]}

    wrong_url = {**contract, "citation_urls": {"S1": "https://example.test/wrong"}}
    with pytest.raises(GroundedAnswerValidationError) as exc_info:
        _validate_grounded_span_answer(
            answer,
            evidence,
            spans,
            required_sources=["paper"],
            paper_contract=wrong_url,
        )
    assert exc_info.value.error_code == "PAPER_FIELD_COVERAGE"
    assert exc_info.value.failure_detail == {"source_id": "S1", "missing_fields": ["link"]}


def test_grounded_span_prompt_does_not_inherit_freeform_citation_instructions():
    spans = _build_evidence_spans(_evidence())
    messages = build_structured_messages(
        LLMRequest(
            question="Compare paper, CVE, and GitHub evidence.",
            evidence=_evidence(),
            instructions=(
                "Cite factual claims with the exact evidence IDs such as [S1]. "
                "Put one factual claim in each sentence or list item and place its citation nearby. "
                "For a mixed-source answer, write exactly one concise factual list item per required source."
            ),
            metadata={
                "required_sources": ["paper", "vulnerability", "project"],
                "evidence_spans": [span.model_dump(mode="json") for span in spans],
            },
        ),
        GroundedSpanAnswer.__name__,
        GroundedSpanAnswer.model_json_schema(),
    )

    user_message = messages[-1]["content"]
    assert "GroundedSpanAnswer contract" in user_message
    assert "Do not include citation markers such as [S1] inside claim_text." in user_message
    assert "Cite factual claims with the exact evidence IDs" not in user_message
    assert "place its citation nearby" not in user_message
    assert "exactly one concise factual list item per required source" not in user_message
    assert "evidence_quote" not in user_message


def test_grounded_span_answer_resolves_server_quotes_and_allows_same_source_multiple_claims():
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    span_answer = _span_answer(
        [
            {
                "claim_text": "The paper title is 5G NAS Replay Detection Study",
                "source_id": "S1",
                "evidence_span_id": _span_id(evidence, "S1", "title"),
            },
            {
                "claim_text": "The paper studies 5G NAS replay detection with state-machine analysis",
                "source_id": "S1",
                "evidence_span_id": _span_id(evidence, "S1", "text"),
            },
            {
                "claim_text": "CVE-2024-12345 is an NVD record for improper input validation",
                "source_id": "S2",
                "evidence_span_id": _span_id(evidence, "S2", "text"),
            },
            {
                "claim_text": "The GitHub repository open5gs/open5gs provides an open-source 5G core",
                "source_id": "S3",
                "evidence_span_id": _span_id(evidence, "S3", "text"),
            },
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["paper", "vulnerability", "project"],
    )

    rendered = _render_grounded_answer(claims)
    citation = CitationGuard().enforce(
        rendered,
        evidence,
        question="Compare paper, CVE, and GitHub evidence for 5G NAS.",
        required_source_types={"paper", "vulnerability", "project"},
        verified_spans=_span_map(spans),
        verified_bindings=_bindings(span_answer, claims),
    )

    assert claims[0].evidence_quote == "5G NAS Replay Detection Study"
    assert claims[1].evidence_quote.startswith("The paper studies 5G NAS replay detection")
    assert [claim.source_id for claim in claims] == ["S1", "S1", "S2", "S3"]
    assert citation.ok is True


def test_verified_renderer_bindings_prevent_evidence_quote_metadata_from_polluting_claim():
    evidence = [
        EvidenceItem(
            source_id="S2",
            text="NVD record",
            source="NVD",
            title="CVE-2024-12345",
            source_url="https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
            external_id="CVE-2024-12345",
            resource_type="vulnerability",
            origin_kind="remote",
            delivery_mode="live",
            content_hash="hash-s2",
            trusted=True,
        )
    ]
    spans = _build_evidence_spans(evidence)
    span_answer = _span_answer(
        [
            {
                "claim_text": "CVE-2024-12345 是 NVD 记录",
                "source_id": "S2",
                "evidence_span_id": "S2:title:1",
            }
        ]
    )
    span_answer = _span_answer(
        [
            {
                "claim_text": "CVE-2024-12345 is an NVD record",
                "source_id": "S2",
                "evidence_span_id": "S2:title:1",
            }
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["vulnerability"],
    )
    rendered = _render_grounded_answer(claims)

    legacy = CitationGuard().enforce(
        rendered,
        evidence,
        question="请给出 NVD CVE 漏洞样例",
        required_source_types={"vulnerability"},
    )
    fixed = CitationGuard().enforce(
        rendered,
        evidence,
        question="请给出 NVD CVE 漏洞样例",
        required_source_types={"vulnerability"},
        verified_spans=_span_map(spans),
        verified_bindings=_bindings(span_answer, claims),
    )

    assert legacy.ok is False
    assert legacy.error_code == "UNSUPPORTED_CITATION"
    assert fixed.ok is True


def test_grounded_span_support_rejects_model_overreach_beyond_selected_span():
    evidence = _nvd_span_support_evidence()
    spans = _build_evidence_spans(evidence)
    metadata_span = _span_id(evidence, "S4", "text", 2)
    span_answer = _span_answer(
        [
            {
                "claim_text": (
                    "CVE-2021-44228 has CVSS 10.0 CRITICAL and CWE-20, "
                    "allows JNDI LDAP remote code execution, and is useful for a 5G core experiment"
                ),
                "source_id": "S4",
                "evidence_span_id": metadata_span,
            }
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["vulnerability"],
    )

    citation = CitationGuard().enforce(
        _render_grounded_answer(_build_grounded_render_units(span_answer, claims)),
        evidence,
        question="Use NVD evidence for a 5G NAS lab.",
        required_source_types={"vulnerability"},
        verified_spans=_span_map(spans),
        verified_bindings=_bindings(span_answer, claims),
    )

    metadata_text = next(span.exact_text for span in spans if span.span_id == metadata_span)
    assert "CVSS 10.0 CRITICAL" in metadata_text
    assert "JNDI" not in metadata_text
    assert citation.ok is False
    assert citation.error_code == "UNSUPPORTED_CITATION"


def test_grounded_span_support_accepts_atomic_claims_split_by_selected_span():
    evidence = _nvd_span_support_evidence()
    spans = _build_evidence_spans(evidence)
    span_answer = _span_answer(
        [
            {
                "claim_text": "CVE-2021-44228 has CVSS 10.0 CRITICAL and CWE-20",
                "source_id": "S4",
                "evidence_span_id": _span_id(evidence, "S4", "text", 2),
            },
            {
                "claim_text": "CVE-2021-44228 allows JNDI LDAP remote code execution",
                "source_id": "S4",
                "evidence_span_id": _span_id(evidence, "S4", "text", 1),
            },
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["vulnerability"],
    )

    citation = CitationGuard().enforce(
        _render_grounded_answer(_build_grounded_render_units(span_answer, claims)),
        evidence,
        question="Use NVD evidence for a 5G NAS lab.",
        required_source_types={"vulnerability"},
        verified_spans=_span_map(spans),
        verified_bindings=_bindings(span_answer, claims),
    )

    assert citation.ok is True


@pytest.mark.parametrize(
    ("claim_text", "source_id", "field", "required"),
    [
        (
            "The paper is titled 5G NAS Replay Detection Study",
            "S1",
            "title",
            ["paper"],
        ),
        (
            "The paper is identified by arXiv ID 2610.01001",
            "S1",
            "external_id",
            ["paper"],
        ),
        (
            "The NVD record is identified as CVE-2024-12345",
            "S2",
            "external_id",
            ["vulnerability"],
        ),
        (
            "The GitHub project is named open5gs/open5gs",
            "S3",
            "title",
            ["project"],
        ),
    ],
)
def test_grounded_span_identity_templates_accept_title_and_external_id_claims(
    claim_text: str,
    source_id: str,
    field: str,
    required: list[str],
):
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    span_answer = _span_answer(
        [
            {
                "claim_text": claim_text,
                "source_id": source_id,
                "evidence_span_id": _span_id(evidence, source_id, field),
            }
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=required,
    )

    citation = CitationGuard().enforce(
        _render_grounded_answer(_build_grounded_render_units(span_answer, claims)),
        evidence,
        question="",
        required_source_types=set(required),
        verified_spans=_span_map(spans),
        verified_bindings=_bindings(span_answer, claims),
    )

    assert citation.ok is True


@pytest.mark.parametrize(
    ("claim_text", "source_id", "field", "required"),
    [
        (
            "The paper title is 5G NAS Replay Detection Study and its arXiv ID is 2610.01001",
            "S1",
            "title",
            ["paper"],
        ),
        (
            "The NVD record CVE-2024-12345 causes a 99 percent 5G core improvement",
            "S2",
            "external_id",
            ["vulnerability"],
        ),
        (
            "The GitHub project open5gs/open5gs proves secure 5G core performance",
            "S3",
            "title",
            ["project"],
        ),
    ],
)
def test_grounded_span_identity_templates_reject_extra_facts(
    claim_text: str,
    source_id: str,
    field: str,
    required: list[str],
):
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    span_answer = _span_answer(
        [
            {
                "claim_text": claim_text,
                "source_id": source_id,
                "evidence_span_id": _span_id(evidence, source_id, field),
            }
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=required,
    )

    citation = CitationGuard().enforce(
        _render_grounded_answer(_build_grounded_render_units(span_answer, claims)),
        evidence,
        question="",
        required_source_types=set(required),
        verified_spans=_span_map(spans),
        verified_bindings=_bindings(span_answer, claims),
    )

    assert citation.ok is False
    assert citation.error_code == "UNSUPPORTED_CITATION"


def test_trusted_paper_exact_external_id_span_accepts_bounded_doi_identity():
    evidence = [
        _external(
            "S1",
            text="Trusted bibliographic metadata.",
            title="NAS DOI Study",
            external_id="10.1234/NAS.2026.7",
            url="https://doi.org/10.1234/NAS.2026.7",
            resource_type="paper",
        )
    ]
    spans = _build_evidence_spans(evidence)
    answer = _span_answer(
        [
            {
                "claim_text": "The paper DOI is 10.1234/NAS.2026.7",
                "source_id": "S1",
                "evidence_span_id": "S1:external_id:1",
            }
        ]
    )
    claims = _validate_grounded_span_answer(answer, evidence, spans, required_sources=["paper"])

    citation = CitationGuard().enforce(
        _render_grounded_answer(_build_grounded_render_units(answer, claims)),
        evidence,
        required_source_types={"paper"},
        verified_spans=_span_map(spans),
        verified_bindings=_bindings(answer, claims),
    )

    assert citation.ok is True


def test_doi_compatibility_never_infers_from_url_or_accepts_wrong_resource_type():
    paper = _external(
        "S1",
        text="Trusted bibliographic metadata.",
        title="NAS Study",
        external_id="2610.01001",
        url="https://doi.org/10.1234/nas.7",
        resource_type="paper",
    )
    vulnerability = paper.model_copy(update={"resource_type": "vulnerability"})

    assert _external_id_span_is_compatible(paper, "10.1234/nas.7", "paper") is False
    assert _external_id_span_is_compatible(vulnerability, "10.1234/nas.7", "vulnerability") is False


def test_grounded_failure_detail_is_hashed_and_contains_no_candidate_text():
    evidence = _evidence()[:1]
    spans = _build_evidence_spans(evidence)
    answer = _span_answer(
        [
            {
                "claim_text": "The paper proves an unsupported security conclusion",
                "source_id": "S1",
                "evidence_span_id": "S1:text:1",
            }
        ]
    )
    claims = _validate_grounded_span_answer(answer, evidence, spans, required_sources=["paper"])

    citation = CitationGuard().enforce(
        _render_grounded_answer(_build_grounded_render_units(answer, claims)),
        evidence,
        required_source_types={"paper"},
        verified_spans=_span_map(spans),
        verified_bindings=_bindings(answer, claims),
    )

    assert citation.ok is False
    assert citation.failure_detail
    assert citation.failure_detail["ordinal"] == 1
    assert citation.failure_detail["source_id"] == "S1"
    assert citation.failure_detail["span_id"] == "S1:text:1"
    assert "unsupported security conclusion" not in repr(citation.failure_detail)


@pytest.mark.parametrize(
    ("span_text", "claim_text", "expected_ok"),
    [
        ("The protocol does not protect NAS replay.", "The protocol protects NAS replay", False),
        ("The protocol protects NAS replay.", "The protocol does not protect NAS replay", False),
        ("The protocol uses i.e.", "The protocol uses i.e", True),
    ],
)
def test_grounded_span_support_preserves_negation_and_boundary_punctuation(
    span_text: str,
    claim_text: str,
    expected_ok: bool,
):
    evidence = [
        _external(
            "S1",
            text=span_text,
            title="NAS replay note",
            external_id="2610.01002",
            url="https://arxiv.org/abs/2610.01002",
            resource_type="paper",
        )
    ]
    spans = _build_evidence_spans(evidence)
    span_answer = _span_answer(
        [
            {
                "claim_text": claim_text,
                "source_id": "S1",
                "evidence_span_id": _span_id(evidence, "S1", "text"),
            }
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["paper"],
    )

    citation = CitationGuard().enforce(
        _render_grounded_answer(_build_grounded_render_units(span_answer, claims)),
        evidence,
        question="",
        required_source_types={"paper"},
        verified_spans=_span_map(spans),
        verified_bindings=_bindings(span_answer, claims),
    )

    assert citation.ok is expected_ok


def test_non_grounded_single_source_support_still_uses_existing_item_text():
    evidence = _nvd_span_support_evidence()

    citation = CitationGuard().enforce(
        "CVE-2021-44228 allows JNDI LDAP remote code execution [S4]",
        evidence,
        question="Use NVD evidence for a 5G NAS lab.",
        required_source_types={"vulnerability"},
    )

    assert citation.ok is True


def test_grounded_renderer_rejects_missing_extra_and_swapped_bindings():
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    span_answer = _span_answer(
        [
            {
                "claim_text": "The paper title is 5G NAS Replay Detection Study",
                "source_id": "S1",
                "evidence_span_id": "S1:title:1",
            },
            {
                "claim_text": "CVE-2024-12345 is an NVD record",
                "source_id": "S2",
                "evidence_span_id": "S2:title:1",
            },
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["paper", "vulnerability"],
    )
    rendered = _render_grounded_answer(claims)
    bindings = _bindings(span_answer, claims)

    common = {
        "question": "Compare paper and NVD evidence for 5G NAS.",
        "required_source_types": {"paper", "vulnerability"},
        "verified_spans": _span_map(spans),
    }

    missing = CitationGuard().enforce(rendered, evidence, verified_bindings=bindings[:1], **common)
    extra = CitationGuard().enforce(rendered, evidence, verified_bindings=[*bindings, bindings[0]], **common)
    swapped = CitationGuard().enforce(rendered, evidence, verified_bindings=list(reversed(bindings)), **common)
    unreferenced = CitationGuard().enforce(
        rendered + "\n- This extra unreferenced fact has no citation.",
        evidence,
        verified_bindings=bindings,
        **common,
    )

    assert missing.ok is False
    assert missing.error_code == "GROUNDING_BINDING_COUNT_MISMATCH"
    assert extra.ok is False
    assert extra.error_code == "GROUNDING_BINDING_COUNT_MISMATCH"
    assert swapped.ok is False
    assert swapped.error_code == "GROUNDING_BINDING_MISMATCH"
    assert unreferenced.ok is False
    assert unreferenced.error_code == "GROUNDING_RENDER_FORMAT"


@pytest.mark.parametrize(
    "author_line",
    [
        "Author information: Ashish Vaswani, Aidan N. Gomez, and Illia Polosukhin.",
        "Author information: Alice Example, Bob J. Smith, and Carol Example.",
        "Author information: A. Researcher, B. Researcher, and C. Researcher.",
    ],
)
def test_grounded_renderer_keeps_verified_physical_line_with_english_abbreviations(author_line):
    evidence = [
        _external(
            "S1",
            text=author_line,
            title="Attention Is All You Need",
            external_id="1706.03762v7",
            url="https://arxiv.org/abs/1706.03762v7",
            resource_type="paper",
        )
    ]
    spans = _build_evidence_spans(evidence)
    author_span = next(span for span in spans if author_line in span.exact_text)
    span_answer = _span_answer(
        [
            {
                "claim_text": author_line,
                "source_id": "S1",
                "evidence_span_id": author_span.span_id,
            }
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["paper"],
    )
    units = _build_grounded_render_units(span_answer, claims)

    check = CitationGuard().enforce(
        _render_grounded_answer(units),
        evidence,
        question=f"Who is listed in this author information: {author_line}",
        required_source_types={"paper"},
        verified_spans=_span_map(spans),
        verified_bindings=_grounded_span_citation_bindings(units),
    )

    assert check.ok is True, check.error_code


def test_grounded_renderer_physical_line_still_rejects_two_units_and_binding_tampering():
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    span_answer = _span_answer(
        [
            {
                "claim_text": "The paper title is 5G NAS Replay Detection Study",
                "source_id": "S1",
                "evidence_span_id": "S1:title:1",
            }
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["paper"],
    )
    units = _build_grounded_render_units(span_answer, claims)
    rendered = _render_grounded_answer(units)
    bindings = _grounded_span_citation_bindings(units)
    common = {
        "question": "What is the paper title?",
        "required_source_types": {"paper"},
        "verified_spans": _span_map(spans),
    }

    joined = CitationGuard().enforce(
        rendered + " " + rendered,
        evidence,
        verified_bindings=bindings,
        **common,
    )
    missing_line = CitationGuard().enforce(
        rendered + "\n- unbound fact",
        evidence,
        verified_bindings=bindings,
        **common,
    )
    swapped = CitationGuard().enforce(
        rendered,
        evidence,
        verified_bindings=[{**bindings[0], "evidence_span_id": "S1:text:1"}],
        **common,
    )

    assert joined.ok is False
    assert joined.error_code in {"GROUNDING_RENDER_FORMAT", "GROUNDING_BINDING_MISMATCH"}
    assert missing_line.error_code in {"GROUNDING_RENDER_FORMAT", "GROUNDING_BINDING_COUNT_MISMATCH"}
    assert swapped.error_code in {"GROUNDING_BINDING_MISMATCH", "GROUNDING_QUOTE_MISMATCH"}


def test_grounded_renderer_rejects_unknown_span_wrong_span_and_quote_tampering():
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    span_answer = _span_answer(
        [
            {
                "claim_text": "CVE-2024-12345 is an NVD record",
                "source_id": "S2",
                "evidence_span_id": "S2:title:1",
            }
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["vulnerability"],
    )
    rendered = _render_grounded_answer(claims)
    bindings = _bindings(span_answer, claims)
    span_map = _span_map(spans)

    common = {
        "question": "Which NVD CVE record is available?",
        "required_source_types": {"vulnerability"},
        "verified_spans": span_map,
    }

    unknown_span = [{**bindings[0], "evidence_span_id": "S2:text:99"}]
    wrong_span = [{**bindings[0], "evidence_span_id": "S1:title:1"}]
    truncated_quote = rendered.replace("evidence quote: CVE-2024-12345", "evidence quote: CVE-2024")
    empty_quote = rendered.replace("evidence quote: CVE-2024-12345", "evidence quote: ")
    spans_only = CitationGuard().enforce(rendered, evidence, **common)

    assert CitationGuard().enforce(rendered, evidence, verified_bindings=unknown_span, **common).error_code == "GROUNDING_UNKNOWN_SPAN"
    assert CitationGuard().enforce(rendered, evidence, verified_bindings=wrong_span, **common).error_code == "GROUNDING_SPAN_SOURCE_MISMATCH"
    assert CitationGuard().enforce(truncated_quote, evidence, verified_bindings=bindings, **common).error_code == "GROUNDING_QUOTE_MISMATCH"
    assert CitationGuard().enforce(empty_quote, evidence, verified_bindings=bindings, **common).error_code == "GROUNDING_RENDER_FORMAT"
    assert spans_only.ok is False
    assert spans_only.error_code == "GROUNDING_BINDING_REQUIRED"


@pytest.mark.parametrize(
    ("claims", "code"),
    [
        (
            [
                {
                    "claim_text": "Unknown source claim",
                    "source_id": "S9",
                    "evidence_span_id": "S1:text:1",
                }
            ],
            "GROUNDING_UNKNOWN_SOURCE",
        ),
        (
            [
                {
                    "claim_text": "Unknown span claim",
                    "source_id": "S1",
                    "evidence_span_id": "S1:text:99",
                }
            ],
            "GROUNDING_UNKNOWN_SPAN",
        ),
        (
            [
                {
                    "claim_text": "The paper title is 5G NAS Replay Detection Study",
                    "source_id": "S1",
                    "evidence_span_id": "S2:text:1",
                }
            ],
            "GROUNDING_SPAN_SOURCE_MISMATCH",
        ),
    ],
)
def test_grounded_span_answer_rejects_unknown_and_mismatched_spans(claims, code):
    evidence = _evidence()
    with pytest.raises(GroundedAnswerValidationError) as exc_info:
        _validate_grounded_span_answer(
            _span_answer(claims),
            evidence,
            _build_evidence_spans(evidence),
            required_sources=["paper"],
        )
    assert exc_info.value.error_code == code


def test_grounded_span_answer_rejects_duplicate_claim_and_binding():
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    duplicate_claims = [
        {
            "claim_text": "The paper studies 5G NAS replay detection",
            "source_id": "S1",
            "evidence_span_id": _span_id(evidence, "S1", "text"),
        },
        {
            "claim_text": "The paper studies 5G NAS replay detection.",
            "source_id": "S1",
            "evidence_span_id": _span_id(evidence, "S1", "title"),
        },
    ]
    with pytest.raises(GroundedAnswerValidationError) as exc_info:
        _validate_grounded_span_answer(
            _span_answer(duplicate_claims),
            evidence,
            spans,
            required_sources=["paper"],
        )
    assert exc_info.value.error_code == "GROUNDING_DUPLICATE_CLAIM"

    duplicate_binding = [
        {
            "claim_text": "The paper studies 5G NAS replay detection",
            "source_id": "S1",
            "evidence_span_id": _span_id(evidence, "S1", "text"),
        },
        {
            "claim_text": "The paper studies 5G NAS replay detection",
            "source_id": "S1",
            "evidence_span_id": _span_id(evidence, "S1", "text"),
        },
    ]
    with pytest.raises(GroundedAnswerValidationError) as exc_info:
        _validate_grounded_span_answer(
            _span_answer(duplicate_binding),
            evidence,
            spans,
            required_sources=["paper"],
        )
    assert exc_info.value.error_code == "GROUNDING_DUPLICATE_BINDING"


def test_grounded_span_answer_keeps_identifier_and_coverage_guards():
    evidence = _evidence()
    spans = _build_evidence_spans(evidence)
    with pytest.raises(GroundedAnswerValidationError) as exc_info:
        _validate_grounded_span_answer(
            _span_answer(
                [
                    {
                        "claim_text": "The paper explains CVE-2024-12345",
                        "source_id": "S1",
                        "evidence_span_id": _span_id(evidence, "S1", "text"),
                    },
                    {
                        "claim_text": "CVE-2024-12345 is an NVD record",
                        "source_id": "S2",
                        "evidence_span_id": _span_id(evidence, "S2", "text"),
                    },
                    {
                        "claim_text": "The GitHub repository open5gs/open5gs provides an open-source 5G core",
                        "source_id": "S3",
                        "evidence_span_id": _span_id(evidence, "S3", "text"),
                    },
                ]
            ),
            evidence,
            spans,
            required_sources=["paper", "vulnerability", "project"],
        )
    assert exc_info.value.error_code == "GROUNDING_IDENTIFIER_MISMATCH"

    with pytest.raises(GroundedAnswerValidationError) as exc_info:
        _validate_grounded_span_answer(
            _span_answer(
                [
                    {
                        "claim_text": "The paper studies 5G NAS replay detection",
                        "source_id": "S1",
                        "evidence_span_id": _span_id(evidence, "S1", "text"),
                    },
                    {
                        "claim_text": "CVE-2024-12345 is an NVD record",
                        "source_id": "S2",
                        "evidence_span_id": _span_id(evidence, "S2", "text"),
                    },
                ]
            ),
            evidence,
            spans,
            required_sources=["paper", "vulnerability", "project"],
        )
    assert exc_info.value.error_code == "GROUNDING_SOURCE_COVERAGE"


def test_identifier_guard_accepts_current_span_identifiers_before_other_evidence():
    evidence = _identifier_edge_evidence()
    spans = _build_evidence_spans(evidence)
    claims = _validate_grounded_span_answer(
        _span_answer(
            [
                {
                    "claim_text": "The paper discusses 5G/NAS security experiments",
                    "source_id": "S1",
                    "evidence_span_id": _span_id(evidence, "S1", "text", 1),
                },
                {
                    "claim_text": "The paper span explicitly mentions CVE-2024-12345 for comparison",
                    "source_id": "S1",
                    "evidence_span_id": _span_id(evidence, "S1", "text", 2),
                },
                {
                    "claim_text": "The project notes LTE/NR lab deployment terminology",
                    "source_id": "S3",
                    "evidence_span_id": _span_id(evidence, "S3", "text", 2),
                },
                {
                    "claim_text": "The project span also mentions CVE-2024-12345 as a reproduced classroom sample",
                    "source_id": "S3",
                    "evidence_span_id": _span_id(evidence, "S3", "text", 3),
                },
            ]
        ),
        evidence,
        spans,
        required_sources=["paper", "project"],
    )

    assert [claim.source_id for claim in claims] == ["S1", "S1", "S3", "S3"]


def test_identifier_guard_accepts_urls_without_repository_path_rematching():
    evidence = _identifier_edge_evidence()
    spans = _build_evidence_spans(evidence)
    claims = _validate_grounded_span_answer(
        _span_answer(
            [
                {
                    "claim_text": "The arXiv landing page is https://arxiv.org/abs/2610.01001",
                    "source_id": "S1",
                    "evidence_span_id": _span_id(evidence, "S1", "text", 3),
                },
                {
                    "claim_text": "The NVD detail page is https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
                    "source_id": "S2",
                    "evidence_span_id": _span_id(evidence, "S2", "text", 2),
                },
                {
                    "claim_text": "The GitHub project page is https://github.com/open5gs/open5gs",
                    "source_id": "S3",
                    "evidence_span_id": _span_id(evidence, "S3", "text", 4),
                },
            ]
        ),
        evidence,
        spans,
        required_sources=["paper", "vulnerability", "project"],
    )

    assert [claim.source_id for claim in claims] == ["S1", "S2", "S3"]


def test_identifier_guard_rejects_only_other_source_identifiers():
    evidence = _identifier_edge_evidence()
    spans = _build_evidence_spans(evidence)

    invalid_cases = [
        (
            "paper claim borrows vulnerability CVE outside its span",
            [
                {
                    "claim_text": "The paper explains CVE-2021-44228",
                    "source_id": "S1",
                    "evidence_span_id": _span_id(evidence, "S1", "text", 1),
                }
            ],
            ["paper"],
        ),
        (
            "paper claim adds vulnerability CVE not in selected span",
            [
                {
                    "claim_text": "The paper discusses 5G/NAS and CVE-2024-12345",
                    "source_id": "S1",
                    "evidence_span_id": _span_id(evidence, "S1", "text", 1),
                }
            ],
            ["paper"],
        ),
        (
            "vulnerability claim uses wrong CVE",
            [
                {
                    "claim_text": "CVE-2021-44228 is an NVD record",
                    "source_id": "S2",
                    "evidence_span_id": _span_id(evidence, "S2", "text"),
                }
            ],
            ["vulnerability"],
        ),
        (
            "project claim uses invented GitHub repository in explicit context",
            [
                {
                    "claim_text": "The GitHub repository attacker/fakecore provides a lab project",
                    "source_id": "S3",
                    "evidence_span_id": _span_id(evidence, "S3", "text"),
                }
            ],
            ["project"],
        ),
        (
            "paper claim borrows project repository",
            [
                {
                    "claim_text": "The paper recommends open5gs/open5gs",
                    "source_id": "S1",
                    "evidence_span_id": _span_id(evidence, "S1", "text", 1),
                }
            ],
            ["paper"],
        ),
        (
            "paper claim borrows NVD URL",
            [
                {
                    "claim_text": "The paper links https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
                    "source_id": "S1",
                    "evidence_span_id": _span_id(evidence, "S1", "text", 1),
                }
            ],
            ["paper"],
        ),
    ]

    for label, raw_claims, required in invalid_cases:
        with pytest.raises(GroundedAnswerValidationError) as exc_info:
            _validate_grounded_span_answer(
                _span_answer(raw_claims),
                evidence,
                spans,
                required_sources=required,
            )
        assert exc_info.value.error_code == "GROUNDING_IDENTIFIER_MISMATCH", label


def test_identifier_occurrences_preserve_claim_quote_and_both_origins():
    evidence = _identifier_edge_evidence()
    occurrences = _identifier_occurrences(
        "The paper compares CVE-2024-12345 with https://arxiv.org/abs/2610.01001",
        "The quote includes CVE-2024-12345 and the GitHub repository open5gs/open5gs.",
        evidence,
    )

    by_value = {item["normalized"]: item for item in occurrences}
    assert by_value["cve-2024-12345"]["origin"] == "both"
    assert by_value["https://arxiv.org/abs/2610.01001"]["origin"] == "claim_text"
    assert by_value["open5gs/open5gs"]["origin"] == "evidence_quote"


def test_identifier_guard_accepts_same_source_canonical_identity_outside_bound_span():
    evidence = _canonical_identifier_evidence()
    spans = _build_evidence_spans(evidence)

    claims = _validate_grounded_span_answer(
        _span_answer(
            [
                {
                    "claim_text": "The paper arXiv ID is 1904.08394v2",
                    "source_id": "S1",
                    "evidence_span_id": _span_id(evidence, "S1", "title"),
                },
                {
                    "claim_text": "Paper 1904.08394v2 analyzes NAS authentication traces without a performance number",
                    "source_id": "S1",
                    "evidence_span_id": _span_id(evidence, "S1", "text", 2),
                },
                {
                    "claim_text": "CVE-2024-12345 is the NVD identifier for the lab parser record",
                    "source_id": "S2",
                    "evidence_span_id": _span_id(evidence, "S2", "text"),
                },
                {
                    "claim_text": "The GitHub repository open5gs/open5gs is the project source",
                    "source_id": "S3",
                    "evidence_span_id": _span_id(evidence, "S3", "text"),
                },
                {
                    "claim_text": "The repository free5gc/free5gc is the second project source",
                    "source_id": "S6",
                    "evidence_span_id": _span_id(evidence, "S6", "text"),
                },
                {
                    "claim_text": "The paper source URL is https://arxiv.org/abs/1904.08394v2/",
                    "source_id": "S1",
                    "evidence_span_id": _span_id(evidence, "S1", "title"),
                },
            ]
        ),
        evidence,
        spans,
        required_sources=["paper", "vulnerability", "project"],
    )

    assert [claim.source_id for claim in claims] == ["S1", "S1", "S2", "S3", "S6", "S1"]


def test_identifier_guard_rejects_noncanonical_or_foreign_identifiers_outside_span():
    evidence = _canonical_identifier_evidence()
    spans = _build_evidence_spans(evidence)
    invalid_cases = [
        (
            "paper uses another paper arxiv id",
            "The paper arXiv ID is 2103.08436v1",
            "S1",
            _span_id(evidence, "S1", "title"),
            ["paper"],
        ),
        (
            "paper uses nvd cve outside span",
            "The paper analyzes CVE-2024-12345",
            "S1",
            _span_id(evidence, "S1", "title"),
            ["paper"],
        ),
        (
            "vulnerability uses wrong cve",
            "CVE-2021-44228 is the NVD identifier for the lab parser record",
            "S2",
            _span_id(evidence, "S2", "text"),
            ["vulnerability"],
        ),
        (
            "project uses another repository",
            "The GitHub repository free5gc/free5gc is the project source",
            "S3",
            _span_id(evidence, "S3", "text"),
            ["project"],
        ),
        (
            "paper uses invented arxiv id",
            "The paper arXiv ID is 9999.99999v1",
            "S1",
            _span_id(evidence, "S1", "title"),
            ["paper"],
        ),
        (
            "current item other paragraph identifier is not selected span support",
            "The selected paragraph cites 9999.00001v1",
            "S9",
            _span_id(evidence, "S9", "text", 1),
            ["paper"],
        ),
        (
            "title repository is not canonical fallback",
            "The GitHub repository titleonly/repo is available",
            "S7",
            _span_id(evidence, "S7", "text"),
            ["project"],
        ),
    ]

    for label, claim_text, source_id, span_id, required in invalid_cases:
        with pytest.raises(GroundedAnswerValidationError) as exc_info:
            _validate_grounded_span_answer(
                _span_answer(
                    [
                        {
                            "claim_text": claim_text,
                            "source_id": source_id,
                            "evidence_span_id": span_id,
                        }
                    ]
                ),
                evidence,
                spans,
                required_sources=required,
            )
        assert exc_info.value.error_code == "GROUNDING_IDENTIFIER_MISMATCH", label


def test_canonical_identity_does_not_support_extra_content_claims():
    evidence = _canonical_identifier_evidence()
    spans = _build_evidence_spans(evidence)
    span_answer = _span_answer(
        [
            {
                "claim_text": "The paper 1904.08394v2 proves a 99 percent detection improvement",
                "source_id": "S1",
                "evidence_span_id": _span_id(evidence, "S1", "title"),
            }
        ]
    )
    claims = _validate_grounded_span_answer(
        span_answer,
        evidence,
        spans,
        required_sources=["paper"],
    )
    citation = CitationGuard().enforce(
        _render_grounded_answer(claims),
        evidence,
        question="What paper evidence is available for 5G NAS?",
        required_source_types={"paper"},
        verified_spans=_span_map(spans),
        verified_bindings=_bindings(span_answer, claims),
    )

    assert citation.ok is False
    assert citation.error_code == "UNSUPPORTED_CITATION"


def test_grounded_span_answer_untrusted_source_and_extra_claim_are_rejected():
    evidence = _evidence()
    untrusted = evidence.copy()
    untrusted[0] = untrusted[0].model_copy(update={"trusted": False})
    with pytest.raises(GroundedAnswerValidationError) as exc_info:
        _validate_grounded_span_answer(
            _span_answer(
                [
                    {
                        "claim_text": "The paper studies 5G NAS replay detection",
                        "source_id": "S1",
                        "evidence_span_id": _span_id(untrusted, "S1", "text"),
                    }
                ]
            ),
            untrusted,
            _build_evidence_spans(untrusted),
            required_sources=["paper"],
        )
    assert exc_info.value.error_code == "GROUNDING_UNTRUSTED_SOURCE"

    spans = _build_evidence_spans(evidence)
    claims = _validate_grounded_span_answer(
        _span_answer(
            [
                {
                    "claim_text": "The paper proves NAS replay attacks are prevented by 95%",
                    "source_id": "S1",
                    "evidence_span_id": _span_id(evidence, "S1", "text"),
                }
            ]
        ),
        evidence,
        spans,
        required_sources=["paper"],
    )
    citation = CitationGuard().enforce(
        _render_grounded_answer(claims),
        evidence,
        question="What exact paper result is available?",
        required_source_types={"paper"},
    )
    assert citation.ok is False
    assert citation.error_code == "UNSUPPORTED_CITATION"


def test_duplicate_claim_text_is_rejected_even_when_quotes_differ():
    _assert_grounding_error(
        "GROUNDING_DUPLICATE_CLAIM",
        [
            {
                "claim_text": "The paper studies 5G NAS replay detection",
                "source_id": "S1",
                "evidence_quote": "5G NAS replay detection",
            },
            {
                "claim_text": "The paper studies 5G NAS replay detection.",
                "source_id": "S1",
                "evidence_quote": "state-machine analysis",
            },
        ],
    )


def test_duplicate_complete_binding_is_rejected():
    _assert_grounding_error(
        "GROUNDING_DUPLICATE_BINDING",
        [
            {
                "claim_text": "The paper studies 5G NAS replay detection",
                "source_id": "S1",
                "evidence_quote": "5G NAS replay detection",
            },
            {
                "claim_text": "The paper studies 5G NAS replay detection",
                "source_id": "S1",
                "evidence_quote": "5G NAS replay detection",
            },
        ],
    )


def test_same_source_second_claim_wrong_quote_is_rejected():
    _assert_grounding_error(
        "GROUNDING_QUOTE_MISMATCH",
        [
            {
                "claim_text": "The paper studies 5G NAS replay detection",
                "source_id": "S1",
                "evidence_quote": "5G NAS replay detection",
            },
            {
                "claim_text": "The paper mentions a blockchain covert channel result",
                "source_id": "S1",
                "evidence_quote": "blockchain covert channel result",
            },
        ],
    )


def test_required_paper_cve_and_github_coverage_accepts_extra_valid_paper_claim():
    claims = _validate_grounded_answer(
        _answer(
            [
                {
                    "claim_text": "The paper studies 5G NAS replay detection with state-machine analysis",
                    "source_id": "S1",
                    "evidence_quote": "5G NAS replay detection with state-machine analysis",
                },
                {
                    "claim_text": "The paper reports a NAS trace dataset for validation",
                    "source_id": "S1",
                    "evidence_quote": "NAS trace dataset for validation",
                },
                {
                    "claim_text": "CVE-2024-12345 is an NVD record",
                    "source_id": "S2",
                    "evidence_quote": "CVE-2024-12345 is an NVD record",
                },
                {
                    "claim_text": "The GitHub repository open5gs/open5gs provides an open-source 5G core",
                    "source_id": "S3",
                    "evidence_quote": "open5gs/open5gs provides an open-source 5G core",
                },
            ]
        ),
        _evidence(),
        required_sources=["paper", "vulnerability", "project"],
    )

    assert [claim.source_id for claim in claims] == ["S1", "S1", "S2", "S3"]


def test_missing_required_source_type_is_rejected():
    _assert_grounding_error(
        "GROUNDING_SOURCE_COVERAGE",
        [
            {
                "claim_text": "The paper studies 5G NAS replay detection",
                "source_id": "S1",
                "evidence_quote": "5G NAS replay detection",
            },
            {
                "claim_text": "CVE-2024-12345 is an NVD record",
                "source_id": "S2",
                "evidence_quote": "CVE-2024-12345 is an NVD record",
            },
        ],
        required=["paper", "vulnerability", "project"],
    )


def test_forged_untrusted_and_wrong_identifier_bindings_are_rejected():
    _assert_grounding_error(
        "GROUNDING_UNKNOWN_SOURCE",
        [
            {
                "claim_text": "A forged source is cited",
                "source_id": "S9",
                "evidence_quote": "forged",
            }
        ],
    )

    evidence = _evidence()
    evidence[0] = evidence[0].model_copy(update={"trusted": False})
    with pytest.raises(GroundedAnswerValidationError) as exc_info:
        _validate_grounded_answer(
            _answer(
                [
                    {
                        "claim_text": "The paper studies 5G NAS replay detection",
                        "source_id": "S1",
                        "evidence_quote": "5G NAS replay detection",
                    }
                ]
            ),
            evidence,
            required_sources=["paper"],
        )
    assert exc_info.value.error_code == "GROUNDING_UNTRUSTED_SOURCE"

    _assert_grounding_error(
        "GROUNDING_IDENTIFIER_MISMATCH",
        [
            {
                "claim_text": "The paper explains CVE-2024-12345",
                "source_id": "S1",
                "evidence_quote": "5G NAS replay detection",
            },
            {
                "claim_text": "CVE-2024-12345 is an NVD record",
                "source_id": "S2",
                "evidence_quote": "CVE-2024-12345 is an NVD record",
            },
            {
                "claim_text": "The GitHub repository open5gs/open5gs provides an open-source 5G core",
                "source_id": "S3",
                "evidence_quote": "open5gs/open5gs provides an open-source 5G core",
            },
        ],
        required=["paper", "vulnerability", "project"],
    )
