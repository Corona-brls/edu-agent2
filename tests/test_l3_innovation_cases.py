from __future__ import annotations

import json

from backend.innovation.recommender import select_cases
from backend.innovation.service import InnovationService
from backend.storage.logger import LearningLogger


class _EmptyRetriever:
    def retrieve(self, query, top_k=None, course_id=None):
        class Response:
            results = []

        return Response()


class _ExplodingGateway:
    def __init__(self) -> None:
        self.calls = 0

    def call_tool(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("innovation must not use unscoped local MCP clues")


class _FixedPortal:
    def __init__(self, items):
        self.items = items

    def search(self, query):
        return {"items": list(self.items), "total": len(self.items)}


def _nas_resources() -> list[dict]:
    return [
        {
            "resource_id": "external:10",
            "type": "paper",
            "title": "5G NAS authentication replay analysis",
            "summary": "Evaluates NAS authentication replay defenses.",
            "source": "arxiv",
            "source_name": "arxiv",
            "external_id": "2609.12345",
            "url": "https://arxiv.org/abs/2609.12345",
            "tags": ["5G NAS", "authentication"],
            "origin_kind": "remote",
            "delivery_mode": "fresh_cache",
            "cache_status": "fresh_cache",
            "fetched_at": "2026-09-30T00:00:00Z",
            "provenance_verified": True,
        },
        {
            "resource_id": "competition:nas-fuzzing",
            "type": "competition",
            "title": "5G NAS protocol fuzzing challenge",
            "summary": "Mutate NAS messages and inspect authentication state transitions.",
            "source": "teacher_dataset",
            "source_name": "teacher_dataset",
            "external_id": "nas-fuzzing",
            "url": "",
            "tags": ["5G NAS", "fuzzing"],
            "origin_kind": "local_catalog",
            "delivery_mode": "local_dataset",
            "cache_status": "teacher_maintained",
            "fetched_at": "2026-09-24T00:00:00Z",
            "provenance_verified": True,
        },
        {
            "resource_id": "competition:cooking",
            "type": "competition",
            "title": "Regional cooking contest",
            "summary": "Prepare desserts and compare flavor scores.",
            "source": "teacher_dataset",
            "source_name": "teacher_dataset",
            "external_id": "cooking",
            "url": "",
            "tags": ["cooking", "food"],
            "origin_kind": "local_catalog",
            "delivery_mode": "local_dataset",
            "cache_status": "teacher_maintained",
            "fetched_at": "2026-09-24T00:00:00Z",
            "provenance_verified": True,
        },
        {
            "resource_id": "rag:2:20",
            "type": "course_document",
            "title": "5G NAS defensive experiment notes",
            "summary": "Course-scoped NAS authentication experiment evidence.",
            "source": "nas-course.md",
            "source_name": "rag",
            "external_id": "20",
            "url": "",
            "tags": ["markdown", "5G NAS"],
            "origin_kind": "course_document",
            "delivery_mode": "local_rag",
            "cache_status": "local",
            "fetched_at": "2026-10-01T00:00:00Z",
            "provenance_verified": True,
            "metadata": {"course_id": 2, "document_id": 2, "chunk_id": 20},
        },
        {
            "resource_id": "document:2",
            "type": "course_document",
            "title": "5G NAS defensive experiment notes",
            "summary": "Course-scoped NAS authentication experiment evidence.",
            "source": "course_document",
            "source_name": "local",
            "external_id": "2",
            "url": "",
            "tags": ["markdown", "5G NAS"],
            "origin_kind": "course_document",
            "delivery_mode": "local_document",
            "cache_status": "local",
            "fetched_at": "2026-10-01T00:00:00Z",
            "provenance_verified": True,
            "metadata": {"course_id": 2, "document_id": 2},
        },
        {
            "resource_id": "external:fake",
            "type": "paper",
            "title": "5G NAS fake paper",
            "summary": "Matching words from an untrusted historical row.",
            "source": "fake",
            "source_name": "fake",
            "external_id": "fake-nas",
            "url": "https://example.test/fake",
            "tags": ["5G NAS"],
            "origin_kind": "unknown",
            "delivery_mode": "unknown",
            "fetched_at": "",
            "provenance_verified": False,
        },
    ]


def _trusted_resource(
    resource_id: str,
    *,
    resource_type: str,
    title: str,
    summary: str,
    source_name: str = "arxiv",
    external_id: str | None = None,
    url: str | None = None,
    tags: list[str] | None = None,
    module: str = "",
    rank: int | None = None,
) -> dict:
    origin_kind = "local_catalog" if resource_type == "competition" else "remote"
    delivery_mode = "local_dataset" if resource_type == "competition" else "fresh_cache"
    cache_status = "teacher_maintained" if resource_type == "competition" else "fresh_cache"
    return {
        "resource_id": resource_id,
        "type": resource_type,
        "title": title,
        "summary": summary,
        "source": source_name,
        "source_name": source_name,
        "external_id": external_id or resource_id,
        "url": url or f"https://example.test/{resource_id}",
        "tags": tags or [],
        "origin_kind": origin_kind,
        "delivery_mode": delivery_mode,
        "cache_status": cache_status,
        "fetched_at": "2026-09-30T00:00:00Z",
        "provenance_verified": True,
        "metadata": {"module": module, "connector_rank": rank} if module or rank is not None else {},
    }


def test_select_cases_rejects_generic_5g_only_noma_or_handover_candidates():
    cases = select_cases(
        [
            _trusted_resource(
                "paper:noma",
                resource_type="paper",
                title="5G NOMA handover optimization",
                summary="Studies 5G wireless handover throughput and NOMA scheduling.",
                tags=["5G", "handover"],
            ),
            _trusted_resource(
                "paper:generic",
                resource_type="paper",
                title="5G wireless security survey",
                summary="A broad mobile wireless security survey for radio access systems.",
                tags=["5G", "security"],
            ),
        ],
        topic="5G NAS security / open5gs reproduction experiment",
        current_plan="Reproduce a NAS authentication lab with open5gs and compare replay behavior.",
        limit=6,
    )

    assert cases == []


def test_select_cases_scores_four_course_topics_with_explainable_relevance():
    scenarios = [
        (
            "5G NAS security / open5gs reproduction experiment",
            "Reproduce NAS authentication traces with open5gs and compare replay defenses.",
            [
                _trusted_resource(
                    "paper:nas",
                    resource_type="paper",
                    title="5G NAS authentication replay analysis",
                    summary="Evaluates NAS authentication replay defenses using protocol traces.",
                    tags=["5G NAS", "authentication", "replay"],
                    rank=2,
                ),
                _trusted_resource(
                    "project:open5gs",
                    resource_type="project",
                    title="open5gs NAS lab reproduction",
                    summary="Uses open5gs to reproduce 5G core NAS authentication behavior.",
                    source_name="github",
                    url="https://github.com/open5gs/open5gs",
                    tags=["open5gs", "5G core", "NAS"],
                    rank=1,
                ),
            ],
            {"paper", "project"},
        ),
        (
            "blockchain covert communication",
            "Design a covert channel over transaction metadata and evaluate detectability.",
            [
                _trusted_resource(
                    "paper:blockchain-covert",
                    resource_type="paper",
                    title="Blockchain covert channel detection",
                    summary="Studies covert communication through blockchain transaction metadata.",
                    tags=["blockchain", "covert channel"],
                ),
                _trusted_resource(
                    "project:blockchain-covert",
                    resource_type="project",
                    title="covert-ledger-lab",
                    summary="Open-source lab for blockchain covert communication experiments.",
                    source_name="github",
                    url="https://github.com/example/covert-ledger-lab",
                    tags=["blockchain", "covert"],
                ),
            ],
            {"paper", "project"},
        ),
        (
            "LLM vulnerability detection",
            "Detect vulnerable code with an LLM and measure false positives.",
            [
                _trusted_resource(
                    "paper:llm-vuln",
                    resource_type="paper",
                    title="LLM vulnerability detection benchmark",
                    summary="Evaluates large language models for vulnerability detection.",
                    tags=["LLM", "vulnerability detection"],
                ),
                _trusted_resource(
                    "vuln:cve",
                    resource_type="vulnerability",
                    title="CVE-2024-12345",
                    summary="NVD vulnerability record used as a defensive detection sample.",
                    source_name="nvd",
                    url="https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
                    tags=["CVE", "vulnerability"],
                ),
            ],
            {"paper", "vulnerability"},
        ),
        (
            "general network security experiment",
            "Build an IDS lab, compare a baseline, and report reproducible metrics.",
            [
                _trusted_resource(
                    "paper:ids-lab",
                    resource_type="paper",
                    title="Network security IDS lab baseline",
                    summary="A reproducible network security experiment with IDS baseline metrics.",
                    tags=["network security", "lab", "baseline"],
                ),
                _trusted_resource(
                    "competition:ids",
                    resource_type="competition",
                    title="Network security lab challenge",
                    summary="Teacher-maintained IDS experiment challenge.",
                    source_name="teacher_dataset",
                    url="",
                    tags=["network security", "experiment"],
                ),
            ],
            {"paper", "competition"},
        ),
    ]

    for topic, plan, resources, expected_types in scenarios:
        cases = select_cases(resources, topic=topic, current_plan=plan, limit=6)
        assert expected_types.issubset({case["type"] for case in cases}), topic
        assert all(case["relevance_score"] > 0 for case in cases)
        assert all(case["relevance_reasons"] for case in cases)
        assert all("connector_rank" in case for case in cases)


def test_select_cases_returns_empty_for_unrelated_trusted_resources():
    cases = select_cases(
        [
            _trusted_resource(
                "paper:vision",
                resource_type="paper",
                title="Computer vision image segmentation",
                summary="Studies image segmentation datasets and visual feature extraction.",
                tags=["vision"],
            )
        ],
        topic="5G NAS security / open5gs reproduction experiment",
        current_plan="Reproduce NAS authentication traces.",
        limit=6,
    )

    assert cases == []


def test_select_cases_filters_unrelated_and_untrusted_candidates_without_padding():
    cases = select_cases(
        _nas_resources(),
        topic="5G NAS authentication",
        current_plan="Compare replay defenses with a baseline experiment.",
        limit=6,
    )

    titles = {case["title"] for case in cases}
    assert titles == {
        "5G NAS authentication replay analysis",
        "5G NAS protocol fuzzing challenge",
        "5G NAS defensive experiment notes",
    }
    assert len(cases) == len(titles)
    assert next(case for case in cases if case["type"] == "course_document")["source_id"] == "rag:2:20"
    required = {
        "source_name",
        "origin_kind",
        "delivery_mode",
        "fetched_at",
        "url",
        "external_id",
        "provenance_verified",
        "topic_category",
    }
    assert all(required.issubset(case) for case in cases)
    assert all(case["origin_kind"] != "unknown" for case in cases)
    assert all(case["relevance_score"] > 0 for case in cases)
    assert all(case["relevance_reasons"] for case in cases)
    assert all("connector_rank" in case for case in cases)


def test_new_session_persists_complete_case_provenance_and_skips_unscoped_mcp(tmp_path):
    logger = LearningLogger(tmp_path / "l3-new-session.db")
    identity = logger.seed_demo_identity()
    user = logger.user_by_username("student_demo")
    gateway = _ExplodingGateway()
    service = InnovationService(
        logger,
        resource_portal=_FixedPortal(_nas_resources()),
        retriever=_EmptyRetriever(),
        mcp_gateway=gateway,
    )

    session = service.start_session(
        user=user,
        topic="5G NAS authentication",
        current_plan="Compare replay defenses with a baseline experiment.",
        course_id=int(identity["course_id"]),
    )

    assert gateway.calls == 0
    assert session["cases"]
    assert "Regional cooking contest" not in {case["title"] for case in session["cases"]}
    with logger._connect() as conn:
        stored_json = conn.execute(
            "SELECT cases_json FROM innovation_sessions WHERE id = ?",
            (session["id"],),
        ).fetchone()["cases_json"]
    stored_cases = json.loads(stored_json)
    assert stored_cases == session["cases"]
    assert all(case["source_name"] != "unknown" for case in stored_cases)
    assert all("relevance_score" in case for case in stored_cases)
    assert all("relevance_reasons" in case for case in stored_cases)
    assert all("connector_rank" in case for case in stored_cases)


def test_legacy_session_adds_unknown_display_defaults_without_rewriting_history(tmp_path):
    logger = LearningLogger(tmp_path / "l3-legacy-session.db")
    identity = logger.seed_demo_identity()
    user = logger.user_by_username("student_demo")
    legacy_cases = [
        {
            "source_id": "paper:legacy",
            "type": "paper",
            "title": "Legacy recommendation",
            "summary": "Historical entry without delivery proof.",
            "source": "arxiv",
            "url": "https://arxiv.org/abs/legacy",
            "tags": ["legacy"],
            "cache_status": "",
        }
    ]
    created = logger.create_innovation_session(
        user_id=int(user["id"]),
        course_id=int(identity["course_id"]),
        class_id=int(identity["class_id"]),
        topic="legacy topic",
        current_plan="Review an old saved session.",
        cases=legacy_cases,
        limitations=[],
        questions=["What evidence is available?"],
    )
    service = InnovationService(logger, retriever=_EmptyRetriever(), mcp_gateway=_ExplodingGateway())

    payload = service.get_session(user=user, session_id=int(created["id"]))
    legacy = payload["cases"][0]

    assert legacy["source_name"] == "unknown"
    assert legacy["origin_kind"] == "unknown"
    assert legacy["delivery_mode"] == "unverified_history"
    assert legacy["fetched_at"] == ""
    assert legacy["provenance_verified"] is False
    assert legacy["topic_category"] == "unknown"
    assert legacy["relevance_score"] is None
    assert legacy["relevance_reasons"] == []
    assert legacy["connector_rank"] is None
    with logger._connect() as conn:
        stored_json = conn.execute(
            "SELECT cases_json FROM innovation_sessions WHERE id = ?",
            (created["id"],),
        ).fetchone()["cases_json"]
    assert json.loads(stored_json) == legacy_cases
