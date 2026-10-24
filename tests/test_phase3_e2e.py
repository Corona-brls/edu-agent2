from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend import main
from backend.connectors.registry import ConnectorRegistry, FakeConnector
from backend.connectors.schemas import ExternalResource
from backend.rag.answerer import CourseAnswerer
from backend.rag.document_parser import Document, Page
from backend.resources.normalizer import content_hash
from backend.storage.logger import LearningLogger, sha256_bytes


class EmptyMCPGateway:
    is_available = False

    def call_tool(self, *args, **kwargs):
        class Result:
            success = False
            data = {}

        class Call:
            result = Result()
            fallback_used = False
            latency_ms = 0.0

        return Call()


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _login(client: TestClient, username: str) -> tuple[str, dict]:
    response = client.post("/api/auth/login", json={"username": username, "password": "demo123456"})
    assert response.status_code == 200
    payload = response.json()
    if payload.get("must_change_password"):
        changed = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {payload['access_token']}"},
            json={"current_password": "demo123456", "new_password": f"{username}-changed-123"},
        )
        assert changed.status_code == 200
        payload = changed.json()
    return payload["access_token"], payload["user"]


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-phase3-e2e-long-enough")
    original_logger = main.learning_logger
    original_registry = main.connector_registry
    original_answerer = main.answerer
    original_kb_dir = main.KNOWLEDGE_BASE_DIR
    original_mcp_gateway = main.mcp_gateway

    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    _seed_knowledge(kb_dir)

    main.KNOWLEDGE_BASE_DIR = kb_dir
    main.answerer = CourseAnswerer(kb_dir)
    main.learning_logger = LearningLogger(tmp_path / "phase3_e2e.db")
    main.learning_logger.seed_demo_identity()
    main.learning_logger.ensure_demo_assignments()
    _seed_document_rows(main.learning_logger, kb_dir)
    _seed_external_cache(main.learning_logger)
    main.connector_registry = ConnectorRegistry([FakeConnector(resources=_external_resources())])
    main.mcp_gateway = EmptyMCPGateway()

    client = TestClient(main.app)
    client._original_logger = original_logger  # type: ignore[attr-defined]
    client._original_registry = original_registry  # type: ignore[attr-defined]
    client._original_answerer = original_answerer  # type: ignore[attr-defined]
    client._original_kb_dir = original_kb_dir  # type: ignore[attr-defined]
    client._original_mcp_gateway = original_mcp_gateway  # type: ignore[attr-defined]
    return client


def _restore(client: TestClient) -> None:
    main.learning_logger = client._original_logger  # type: ignore[attr-defined]
    main.connector_registry = client._original_registry  # type: ignore[attr-defined]
    main.answerer = client._original_answerer  # type: ignore[attr-defined]
    main.KNOWLEDGE_BASE_DIR = client._original_kb_dir  # type: ignore[attr-defined]
    main.mcp_gateway = client._original_mcp_gateway  # type: ignore[attr-defined]


def _seed_knowledge(kb_dir: Path) -> None:
    (kb_dir / "5g_nas.md").write_text(
        "\n".join(
            [
                "# 5G NAS Security Lab",
                "",
                "Research goal: compare normal 5G NAS authentication with replay detection using only isolated course traces.",
                "Method design: label the same local defensive trace fields for normal authentication and replay samples, then compare event sequences and alerts.",
                "Tool selection: use an offline NAS log parser, a local metrics table, and a local visualization tool.",
                "Experiment step 1: prepare the isolated course traces without collecting real subscriber data.",
                "Experiment step 2: label normal authentication events in the local dataset.",
                "Experiment step 3: label replay detection events in the local dataset.",
                "Experiment step 4: run the offline parser against both labeled groups.",
                "Experiment step 5: compare defensive alerts, false positives, and processing latency.",
                "Experiment step 6: save a reproducible local record for course review.",
                "Result analysis: compare event counts, replay detection hits, false positives, identity exposure indicators, security context setup, integrity protection, and abnormal state transitions without inferring facts outside the traces.",
                "Innovation point: evaluate normal authentication and replay detection in one controlled field schema for reproducible comparison.",
                "Risks and cautions: use only local isolated defensive traces, never target public systems, and never collect real user data.",
            ]
        ),
        encoding="utf-8",
    )


def _seed_document_rows(logger: LearningLogger, kb_dir: Path) -> None:
    course_id = logger.default_course_id()
    text = (kb_dir / "5g_nas.md").read_text(encoding="utf-8")
    document = Document(
        source_path=kb_dir / "5g_nas.md",
        filename="5g_nas.md",
        file_type=".md",
        title="5G NAS Security Lab",
        text=text,
        pages=[
            Page(
                text=text,
                page=1,
                section="5G NAS Security Lab",
                source_type="markdown",
            )
        ],
    )
    saved = logger.save_document(
        document=document,
        content_hash=sha256_bytes(text.encode("utf-8")),
        chunks=[text],
        byte_size=len(text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
    )
    logger.set_document_publication(int(saved["id"]), published=True, actor_id=1)


def _external_resources() -> list[ExternalResource]:
    return [
        ExternalResource(
            external_id="paper-phase3-001",
            resource_type="paper",
            source_name="semantic_scholar",
            title="5G NAS Authentication Security Evaluation",
            authors=["Demo Researcher"],
            abstract="Mock paper metadata about 5G NAS replay-risk evaluation and defensive experiments.",
            url="https://example.test/paper-phase3-001",
            published_at="2026",
            tags=["5G NAS", "paper"],
            module="5G NAS",
            rights_note="Mock metadata for offline test.",
        ),
        ExternalResource(
            external_id="CVE-2021-44228",
            resource_type="vulnerability",
            source_name="nvd",
            title="CVE-2021-44228",
            abstract="Mock NVD vulnerability metadata for teaching lookup and cache behavior.",
            url="https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
            published_at="2021",
            tags=["CVSS 10.0", "CWE-20"],
            module="vulnerability sample",
            rights_note="Mock NVD metadata for offline test.",
        ),
        ExternalResource(
            external_id="3gpp-ts-24-501",
            resource_type="standard",
            source_name="standard_source",
            title="3GPP TS 24.501 metadata",
            abstract="Mock standard metadata only; no copyrighted full text is fetched.",
            url="https://www.3gpp.org/dynareport/24501.htm",
            published_at="2026",
            tags=["3GPP", "NAS", "standard"],
            module="5G NAS",
            rights_note="Official metadata link only.",
        ),
        ExternalResource(
            external_id="project-phase3-001",
            resource_type="project",
            source_name="github",
            title="nas-security-lab",
            abstract="Mock open-source teaching project metadata for a defensive NAS lab.",
            url="https://github.com/example/nas-security-lab",
            published_at="2026",
            tags=["project", "lab"],
            module="5G NAS",
            raw_metadata={
                "full_name": "example/nas-security-lab",
                "private": False,
                "visibility": "public",
            },
            rights_note="Mock project metadata for offline test.",
        ),
    ]


def _seed_external_cache(logger: LearningLogger) -> None:
    for resource in _external_resources():
        query = {"query": resource.title, "page": 1, "page_size": 10, "allow_remote": True}
        run_id = logger.create_connector_sync_run(
            connector_name=resource.source_name,
            query=query,
            status="RUNNING",
            cache_hit=False,
        )
        record = logger.commit_verified_sync(
            run_id=run_id,
            connector_name=resource.source_name,
            origin_kind="local_catalog" if resource.source_name == "standard_source" else "remote",
            query=query,
            resources=[{"resource": resource.model_dump(mode="json"), "content_hash": content_hash(resource)}],
            ttl_hours=24,
            latency_ms=0,
        )
        logger.mark_external_resource_indexed(
            int(record[0]["id"]),
            f"mock:{resource.source_name}:{resource.external_id}",
        )


def test_phase3_full_demo_flow_offline(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        student_token, student = _login(client, "student_demo")
        teacher_token, _teacher = _login(client, "teacher_demo")
        student_headers = _headers(student_token)
        teacher_headers = _headers(teacher_token)
        sid = "phase3-e2e-session"

        assert client.post("/api/learning/sessions", headers=student_headers, json={"client_session_id": sid}).status_code == 200
        assert client.get("/api/courses", headers=student_headers).json()
        assert client.get("/api/classes", headers=student_headers).json()

        connector_status = client.get("/api/connectors/status")
        assert connector_status.status_code == 200
        assert connector_status.json()["connectors"][0]["name"] == "fake"

        resources = client.get(
            "/api/resources/portal",
            headers=student_headers,
            params={"keyword": "5G NAS", "page_size": 50, "client_session_id": sid},
        )
        assert resources.status_code == 200
        items = resources.json()["items"]
        types = {item["type"] for item in items}
        assert {"course_document", "paper", "standard", "project"}.issubset(types)
        assert any(item["source_name"] == "semantic_scholar" for item in items)
        assert any(item["source_name"] == "standard_source" for item in items)
        resource_id = next(item["resource_id"] for item in items if item["type"] == "paper")
        opened = client.post(
            "/api/resources/open",
            headers=student_headers,
            json={"resource_id": resource_id, "client_session_id": sid, "metadata": {"source": "e2e"}},
        )
        assert opened.status_code == 200

        vulnerability = client.get(
            "/api/resources/portal",
            headers=student_headers,
            params={"resource_type": "vulnerability", "keyword": "CVE-2021-44228", "page_size": 10},
        )
        assert vulnerability.status_code == 200
        assert vulnerability.json()["items"][0]["source_name"] == "nvd"

        assignments = client.get("/api/assignments", headers=student_headers).json()["assignments"]
        assignment = assignments[0]
        step = assignment["steps"][0]
        progress = client.post(
            f"/api/assignments/{assignment['id']}/progress",
            headers=student_headers,
            json={
                "step_id": step["id"],
                "status": "in_progress",
                "progress_percent": 35,
                "note": "Selected local 5G NAS defensive replay-risk lab.",
                "client_session_id": sid,
            },
        )
        assert progress.status_code == 200

        practice = client.post(
            "/api/practice/guide",
            headers=student_headers,
            json={
                "topic": "5G NAS replay-risk experiment",
                "client_session_id": sid,
                "assignment_id": assignment["id"],
                "step_id": step["id"],
            },
        )
        assert practice.status_code == 200
        practice_payload = practice.json()
        assert practice_payload["assignment_id"] == assignment["id"]
        assert practice_payload["citations"]
        assert practice_payload["citations"][0].get("source") == "5g_nas.md"

        question_result = {
            "answer": "Use local traces, replay-risk metrics, and defensive logging evidence.",
            "module": "5G NAS",
            "intent": "course_qa",
            "confidence": 0.88,
            "abstained": False,
            "citations": practice_payload["citations"],
            "course_id": main.learning_logger.default_course_id(),
            "class_id": main.learning_logger.default_class_id(),
        }
        qid_1 = main.learning_logger.log_question(
            str(student["id"]),
            "How should I evaluate 5G NAS replay risk in a safe local lab?",
            question_result,
        )
        qid_2 = main.learning_logger.log_question(
            str(student["id"]),
            "What metrics should I use for a 5G NAS replay-risk lab?",
            question_result,
        )
        for question_id in [qid_1, qid_2]:
            asked = client.post(
                "/api/learning/events",
                headers=student_headers,
                json={
                    "client_session_id": sid,
                    "event_type": "QUESTION_ASKED",
                    "page_name": "student_chat",
                    "module": "5G NAS",
                    "assignment_id": str(assignment["id"]),
                    "metadata": {"question_id": question_id, "step_id": step["id"]},
                },
            )
            assert asked.status_code == 200

        stuck = client.post(
            "/api/stuck-reports",
            headers=student_headers,
            json={
                "assignment_id": assignment["id"],
                "step_id": step["id"],
                "module": "5G NAS",
                "reason": "Need help choosing a safe replay-risk metric.",
                "client_session_id": sid,
            },
        )
        assert stuck.status_code == 200
        stuck_id = stuck.json()["report"]["id"]

        innovation = client.post(
            "/api/innovation/sessions",
            headers=student_headers,
            json={
                "topic": "5G NAS",
                "current_plan": "Compare a baseline detector with an improved local replay-risk detector.",
                "client_session_id": sid,
            },
        )
        assert innovation.status_code == 200
        innovation_payload = innovation.json()
        assert {case["type"] for case in innovation_payload["cases"]} & {"paper", "project"}
        question = innovation_payload["questions"][0]
        answered = client.post(
            f"/api/innovation/sessions/{innovation_payload['id']}/answers",
            headers=student_headers,
            json={"question_id": question["id"], "answer": "I will compare false positives and detection latency.", "client_session_id": sid},
        )
        assert answered.status_code == 200
        finalized = client.post(
            f"/api/innovation/sessions/{innovation_payload['id']}/finalize",
            headers=student_headers,
            json={"client_session_id": sid},
        )
        assert finalized.status_code == 200
        assert finalized.json()["final_ideas"]

        rebuild = client.post(
            "/api/teacher/question-clusters/rebuild",
            headers=teacher_headers,
            json={"min_cluster_size": 2, "distance_threshold": 1.0},
        )
        assert rebuild.status_code == 200
        clusters = client.get("/api/teacher/question-clusters", headers=teacher_headers).json()["clusters"]
        cluster = next(item for item in clusters if item["count"] >= 2)
        renamed = client.patch(
            f"/api/teacher/question-clusters/{cluster['id']}",
            headers=teacher_headers,
            json={"label": "5G NAS replay-risk lab"},
        )
        approved = client.post(f"/api/teacher/question-clusters/{cluster['id']}/approve", headers=teacher_headers)
        assert renamed.status_code == 200
        assert approved.status_code == 200

        faq_csv = client.get("/api/teacher/phase3/faq-export.csv", headers=teacher_headers)
        assert faq_csv.status_code == 200
        assert b"5G NAS" in faq_csv.content

        teacher_summary = client.get("/api/teacher/phase3/summary", headers=teacher_headers)
        assert teacher_summary.status_code == 200
        summary = teacher_summary.json()
        assert summary["access"]["visit_count"] >= 0
        assert summary["resources"]["search_count"] >= 1
        assert summary["resources"]["open_count"] >= 1
        assert summary["qa"]["question_count"] >= 2
        assert summary["innovation"]["session_count"] >= 1
        assert any(item.get("id") == stuck_id for item in summary["stuck"]["items"])

        timeline = client.get(f"/api/teacher/phase3/students/{student['id']}/timeline", headers=teacher_headers)
        assert timeline.status_code == 200
        event_types = [item["event_type"] for item in timeline.json()["events"]]
        for expected in [
            "LOGIN",
            "RESOURCE_SEARCH",
            "RESOURCE_OPEN",
            "QUESTION_ASKED",
            "PRACTICE_GUIDE_GENERATED",
            "ASSIGNMENT_PROGRESS_UPDATED",
            "STUCK_REPORTED",
            "INNOVATION_SESSION_STARTED",
            "INNOVATION_QUESTION_ANSWERED",
        ]:
            assert expected in event_types

        forbidden_sync = client.post("/api/connectors/fake/sync", headers=student_headers, json={"query": "5G NAS"})
        forbidden_teacher = client.get("/api/teacher/phase3/summary", headers=student_headers)
        forbidden_delete = client.delete("/api/knowledge/documents/1", headers=student_headers)
        assert forbidden_sync.status_code == 403
        assert forbidden_teacher.status_code == 403
        assert forbidden_delete.status_code == 403
    finally:
        _restore(client)
