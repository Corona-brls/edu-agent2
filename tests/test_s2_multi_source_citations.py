from backend.agent.orchestrator import _answer_evidence
from backend.agent.state import AgentState
from backend.guards.citation_guard import CitationGuard
from backend.llm.schemas import EvidenceItem


QUESTION = (
    "Compare a research paper about 5G NAS replay detection, CVE-2024-12345, "
    "and the GitHub repository open5gs/open5gs."
)


def _external(
    source_id: str,
    *,
    text: str,
    source: str,
    title: str,
    url: str,
    external_id: str,
    resource_type: str,
    trusted: bool = True,
) -> EvidenceItem:
    return EvidenceItem(
        source_id=source_id,
        text=text,
        source=source,
        title=title,
        source_url=url,
        external_id=external_id,
        resource_type=resource_type,
        origin_kind="remote",
        delivery_mode="live",
        content_hash=f"hash-{source_id.lower()}",
        trusted=trusted,
    )


def _three_sources() -> list[EvidenceItem]:
    return [
        _external(
            "S1",
            text="The study evaluates replay detection for 5G NAS with a state-machine method.",
            source="arxiv",
            title="NAS Sentinel Replay Detection",
            url="https://arxiv.org/abs/2610.01001",
            external_id="2610.01001",
            resource_type="paper",
        ),
        _external(
            "S2",
            text="CVE-2024-12345 is an NVD record for improper input validation in a lab parser.",
            source="nvd",
            title="CVE-2024-12345",
            url="https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
            external_id="CVE-2024-12345",
            resource_type="vulnerability",
        ),
        _external(
            "S3",
            text="The GitHub repository open5gs/open5gs provides an open-source 5G core for controlled labs.",
            source="github",
            title="open5gs/open5gs",
            url="https://github.com/open5gs/open5gs",
            external_id="open5gs/open5gs",
            resource_type="project",
        ),
    ]


def _valid_answer() -> str:
    return (
        "The study evaluates replay detection for 5G NAS with a state-machine method [S1].\n"
        "CVE-2024-12345 is an NVD record for improper input validation [S2].\n"
        "The GitHub repository open5gs/open5gs provides an open-source 5G core for controlled labs [S3]."
    )


def test_two_source_question_validates_each_claim_without_requiring_every_identifier():
    evidence = _three_sources()[:2]
    answer = (
        "The study evaluates replay detection for 5G NAS with a state-machine method [S1].\n"
        "CVE-2024-12345 is an NVD record for improper input validation [S2]."
    )

    result = CitationGuard().enforce(answer, evidence, question=QUESTION)

    assert result.ok is True


def test_three_source_answer_accepts_one_supported_source_per_fact():
    result = CitationGuard().enforce(
        _valid_answer(),
        _three_sources(),
        question=QUESTION,
        required_source_types={"paper", "vulnerability", "project"},
    )

    assert result.ok is True


def test_trusted_external_identity_facts_accept_matching_ids_and_urls():
    answer = (
        "The paper title is NAS Sentinel Replay Detection at "
        "https://arxiv.org/abs/2610.01001 [S1].\n"
        "The NVD record CVE-2024-12345 exists at "
        "https://nvd.nist.gov/vuln/detail/CVE-2024-12345 [S2].\n"
        "The GitHub repository open5gs/open5gs exists at "
        "https://github.com/open5gs/open5gs [S3]."
    )

    result = CitationGuard().enforce(
        answer,
        _three_sources(),
        question=QUESTION,
        required_source_types={"paper", "vulnerability", "project"},
    )

    assert result.ok is True


def test_external_identity_metadata_cannot_support_an_extra_conclusion():
    answer = (
        "The NVD record CVE-2024-12345 exists and proves that the parser prevents attacks "
        "https://nvd.nist.gov/vuln/detail/CVE-2024-12345 [S2].\n"
        "The GitHub repository open5gs/open5gs exists at "
        "https://github.com/open5gs/open5gs [S3]."
    )

    result = CitationGuard().enforce(
        answer,
        _three_sources(),
        question=QUESTION,
        required_source_types={"vulnerability", "project"},
    )

    assert result.ok is False
    assert result.error_code == "UNSUPPORTED_CITATION"


def test_multi_source_answer_selects_the_explicitly_requested_evidence_per_type():
    evidence = [
        _external(
            "S0",
            text="An unrelated paper about radio scheduling.",
            source="arxiv",
            title="Radio Scheduling Survey",
            url="https://arxiv.org/abs/2601.00001",
            external_id="2601.00001",
            resource_type="paper",
        ),
        *_three_sources(),
        _external(
            "S4",
            text="An unrelated GitHub repository provides cooking examples.",
            source="github",
            title="example/cooking",
            url="https://github.com/example/cooking",
            external_id="example/cooking",
            resource_type="project",
        ),
    ]
    state = AgentState(
        user_goal=QUESTION,
        required_sources=["paper", "vulnerability", "project"],
        evidence=evidence,
    )

    selected = _answer_evidence(state)

    assert [item.source_id for item in selected] == ["S1", "S2", "S3"]


def test_single_external_source_keeps_only_trusted_explicit_type_candidates():
    papers = _three_sources()
    evidence = [
        papers[0],
        _external(
            "S4",
            text="A second trusted paper record.",
            source="crossref",
            title="Second NAS Paper",
            url="https://example.test/second-paper",
            external_id="10.1234/nas.2",
            resource_type="paper",
        ),
        papers[1],
        papers[2],
        _external(
            "S5",
            text="Untrusted paper record.",
            source="caller",
            title="Caller Paper",
            url="https://example.test/caller-paper",
            external_id="10.1234/caller",
            resource_type="paper",
            trusted=False,
        ),
    ]
    state = AgentState(user_goal="Find papers", required_sources=["paper"], evidence=evidence)

    selected = _answer_evidence(state)

    assert [item.source_id for item in selected] == ["S1", "S4"]


def test_citation_immediately_after_sentence_punctuation_remains_adjacent():
    answer = (
        "The study evaluates replay detection for 5G NAS with a state-machine method. [S1]\n"
        "CVE-2024-12345 is an NVD record for improper input validation. [S2]\n"
        "The GitHub repository open5gs/open5gs provides an open-source 5G core for controlled labs. [S3]"
    )

    result = CitationGuard().enforce(
        answer,
        _three_sources(),
        question=QUESTION,
        required_source_types={"paper", "vulnerability", "project"},
    )

    assert result.ok is True


def test_paper_fact_cannot_be_bound_to_nvd():
    answer = "The study evaluates replay detection for 5G NAS with a state-machine method [S2]."

    result = CitationGuard().enforce(answer, _three_sources(), question=QUESTION)

    assert result.ok is False
    assert result.error_code == "UNSUPPORTED_CITATION"


def test_cve_fact_cannot_be_bound_to_arxiv():
    answer = "CVE-2024-12345 is an NVD record for improper input validation [S1]."

    result = CitationGuard().enforce(answer, _three_sources(), question=QUESTION)

    assert result.ok is False
    assert result.error_code in {"IRRELEVANT_CITATION", "UNSUPPORTED_CITATION"}


def test_github_repository_cannot_be_bound_to_paper():
    answer = "The GitHub repository open5gs/open5gs provides an open-source 5G core [S1]."

    result = CitationGuard().enforce(answer, _three_sources(), question=QUESTION)

    assert result.ok is False
    assert result.error_code in {"IRRELEVANT_CITATION", "UNSUPPORTED_CITATION"}


def test_required_source_coverage_rejects_missing_github_source():
    answer = (
        "The study evaluates replay detection for 5G NAS with a state-machine method [S1].\n"
        "CVE-2024-12345 is an NVD record for improper input validation [S2]."
    )

    result = CitationGuard().enforce(
        answer,
        _three_sources(),
        question=QUESTION,
        required_source_types={"paper", "vulnerability", "project"},
    )

    assert result.ok is False
    assert result.error_code == "INCOMPLETE_CITATION_COVERAGE"


def test_uncited_factual_list_item_is_rejected():
    answer = (
        "The study evaluates replay detection for 5G NAS with a state-machine method [S1].\n"
        "CVE-2024-12345 is an NVD record for improper input validation [S2].\n"
        "The GitHub repository open5gs/open5gs provides an open-source 5G core."
    )

    result = CitationGuard().enforce(answer, _three_sources(), question=QUESTION)

    assert result.ok is False
    assert result.error_code == "MISSING_CLAIM_CITATION"


def test_compound_fact_with_multiple_trailing_citations_is_rejected_as_ambiguous():
    answer = (
        "The paper evaluates replay detection, CVE-2024-12345 is an NVD record, "
        "and open5gs/open5gs provides a 5G core [S1] [S2] [S3]."
    )

    result = CitationGuard().enforce(answer, _three_sources(), question=QUESTION)

    assert result.ok is False
    assert result.error_code == "AMBIGUOUS_CITATION"


def test_multi_source_answer_still_rejects_forged_and_unknown_sources():
    forged = CitationGuard().enforce(
        _valid_answer().replace("[S3]", "[S9]"),
        _three_sources(),
        question=QUESTION,
    )
    unknown = _three_sources()
    unknown[2] = unknown[2].model_copy(update={"trusted": False})
    untrusted = CitationGuard().enforce(_valid_answer(), unknown, question=QUESTION)

    assert forged.ok is False
    assert forged.error_code == "FORGED_CITATION"
    assert untrusted.ok is False
    assert untrusted.error_code == "UNLOCATABLE_CITATION"


def test_cross_course_and_off_topic_questions_remain_rejected():
    course_a = EvidenceItem(
        source_id="S1",
        text="COURSE_A_UNIQUE_MARKER is available only in course A.",
        source="course-a.md",
        title="Course A",
        document_id=1,
        chunk_id=1,
        section="scope",
        resource_type="course",
        trusted=True,
    )
    cross_course = CitationGuard().enforce(
        "COURSE_A_UNIQUE_MARKER is available [S1].",
        [course_a],
        question="Explain COURSE_B_UNIQUE_MARKER from course B.",
    )
    off_topic = CitationGuard().enforce(
        "The study evaluates replay detection for 5G NAS [S1].",
        [_three_sources()[0]],
        question="Give me a dessert recipe for a birthday party.",
    )

    assert cross_course.ok is False
    assert cross_course.error_code == "IRRELEVANT_CITATION"
    assert off_topic.ok is False
    assert off_topic.error_code == "IRRELEVANT_CITATION"
