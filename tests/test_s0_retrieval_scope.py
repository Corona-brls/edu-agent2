from __future__ import annotations

from dataclasses import replace

from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.schemas import ActionDecision, AgentAction
from backend.agent.state import AgentRunStatus
from backend.llm.base import LLMProvider
from backend.llm.schemas import LLMRequest, LLMResponse
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import RetrievalResponse, RetrievalResult
from backend.storage.logger import LearningLogger, sha256_bytes
from backend.tools.mcp_client import MCPClientGateway


class _RetrieveProvider(LLMProvider):
    provider_name = "s0-test"
    model_name = "s0-test-model"

    def decide_next_action(self, state_summary, available_tools):
        return ActionDecision(
            action=AgentAction.RETRIEVE,
            arguments={"query": "S0 scoped marker"},
            reason="Retrieve course-scoped evidence.",
        )

    def generate_answer(self, request: LLMRequest) -> LLMResponse:
        assert request.evidence
        return LLMResponse(
            text=f"{request.evidence[0].text} [S1]",
            provider=self.provider_name,
            model=self.model_name,
        )

    def generate_structured(self, request, schema):
        raise AssertionError("Structured generation is not used in S0 tests")


class _ScopeBreakingRetriever:
    def __init__(self, unscoped_result: RetrievalResult):
        self.unscoped_result = unscoped_result
        self.calls: list[int | None] = []

    def retrieve(self, query: str, top_k=None, course_id: int | None = None):
        self.calls.append(course_id)
        if course_id is not None:
            raise TypeError("simulated scoped retriever signature failure")
        return RetrievalResponse(
            results=[self.unscoped_result],
            abstained=False,
            threshold=0.18,
            fallback_used=False,
        )


class _FixedRetriever:
    def __init__(self, *results: RetrievalResult):
        self.results = list(results)
        self.calls: list[int | None] = []

    def retrieve(self, query: str, top_k=None, course_id: int | None = None):
        self.calls.append(course_id)
        return RetrievalResponse(
            results=self.results,
            abstained=not self.results,
            threshold=0.18,
            fallback_used=False,
        )


def _course(storage: LearningLogger, slug: str) -> int:
    with storage._connect() as conn:
        cursor = conn.execute(
            "INSERT INTO courses (slug, name, description, assessment_method) VALUES (?, ?, '', '')",
            (slug, f"S0 {slug}"),
        )
        return int(cursor.lastrowid)


def _table_count(storage: LearningLogger, table: str) -> int:
    with storage._connect() as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _indexed_result(storage: LearningLogger, course_id: int, marker: str) -> RetrievalResult:
    text = f"{marker} is course-scoped evidence."
    document = Document(
        source_path=f"knowledge_base/course-{course_id}/{marker}.md",
        filename=f"{marker}.md",
        file_type=".md",
        title=marker,
        text=text,
        pages=[Page(text=text, page=1, section="S0", source_type="markdown")],
    )
    record = storage.save_document(
        document=document,
        content_hash=sha256_bytes(text.encode("utf-8")),
        chunks=[{"chunk_text": text, "page": 1, "section": "S0", "source_type": "md"}],
        byte_size=len(text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
    )
    storage.set_document_publication(int(record["id"]), published=True, actor_id=1)
    chunk = storage.chunks_for_document(int(record["id"]))[0]
    return RetrievalResult(
        document_id=int(record["id"]),
        chunk_id=int(chunk["id"]),
        source=document.filename,
        title=document.title,
        snippet=text,
        page=1,
        section="S0",
        source_type="md",
        keyword_score=1.0,
        vector_score=0.0,
        final_score=1.0,
    )


def _run(storage: LearningLogger, retriever, course_id: int) -> dict:
    provider = _RetrieveProvider()
    return AgentOrchestrator(
        storage=storage,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=provider,
        retriever=retriever,
        mcp_gateway=MCPClientGateway(),
    ).run("Explain the S0 scoped marker.", user_id="s0-student", course_id=course_id)


def test_scoped_type_error_fails_closed_without_unscoped_retry(tmp_path):
    storage = LearningLogger(tmp_path / "s0-type-error.db")
    course_a = _course(storage, "s0-a")
    course_b = _course(storage, "s0-b")
    course_a_result = _indexed_result(storage, course_a, "S0_A_ONLY_MARKER")
    retriever = _ScopeBreakingRetriever(course_a_result)
    before = {
        "documents": len(storage.documents()),
        "chunks": len(storage.document_chunks()),
        "questions": _table_count(storage, "questions"),
    }

    result = _run(storage, retriever, course_b)

    assert retriever.calls == [course_b]
    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["stop_reason"] == "RETRIEVER_FAILED"
    assert result["citations"] == []
    assert "S0_A_ONLY_MARKER" not in result["answer"]
    assert {
        "documents": len(storage.documents()),
        "chunks": len(storage.document_chunks()),
        "questions": _table_count(storage, "questions"),
    } == before


def test_scoped_retrieval_rejects_document_owned_by_another_course(tmp_path):
    storage = LearningLogger(tmp_path / "s0-cross-course.db")
    course_a = _course(storage, "s0-cross-a")
    course_b = _course(storage, "s0-cross-b")
    course_a_result = _indexed_result(storage, course_a, "S0_FOREIGN_MARKER")

    result = _run(storage, _FixedRetriever(course_a_result), course_b)

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["stop_reason"] == "RETRIEVER_SCOPE_MISMATCH"
    assert result["citations"] == []
    assert "S0_FOREIGN_MARKER" not in result["answer"]


def test_scoped_retrieval_rejects_forged_local_fallback_result(tmp_path):
    storage = LearningLogger(tmp_path / "s0-forged-local.db")
    course_b = _course(storage, "s0-local-b")
    forged = RetrievalResult(
        document_id=0,
        chunk_id=1,
        source="foreign.md",
        title="Forged local result",
        snippet="S0_FORGED_LOCAL_MARKER",
        page=None,
        section="S0",
        source_type="md",
        keyword_score=1.0,
        vector_score=0.0,
        final_score=1.0,
    )

    result = _run(storage, _FixedRetriever(forged), course_b)

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["stop_reason"] == "RETRIEVER_SCOPE_MISMATCH"
    assert result["citations"] == []
    assert "S0_FORGED_LOCAL_MARKER" not in result["answer"]


def test_scoped_retrieval_accepts_selected_course_document_and_chunk(tmp_path):
    storage = LearningLogger(tmp_path / "s0-positive.db")
    course_b = _course(storage, "s0-positive-b")
    course_b_result = _indexed_result(storage, course_b, "S0_B_ONLY_MARKER")
    retriever = _FixedRetriever(course_b_result)

    result = _run(storage, retriever, course_b)

    assert retriever.calls == [course_b]
    assert result["run_status"] == AgentRunStatus.COMPLETED
    assert result["stop_reason"] == "ANSWERED"
    assert result["citations"][0]["document_id"] == course_b_result.document_id
    assert result["citations"][0]["chunk_id"] == course_b_result.chunk_id
    assert "S0_B_ONLY_MARKER" in result["answer"]


def test_scoped_retrieval_rejects_chunk_owned_by_another_document(tmp_path):
    storage = LearningLogger(tmp_path / "s0-chunk-mismatch.db")
    course_b = _course(storage, "s0-chunk-b")
    first = _indexed_result(storage, course_b, "S0_FIRST_DOCUMENT")
    second = _indexed_result(storage, course_b, "S0_SECOND_DOCUMENT")
    mismatched = replace(first, chunk_id=second.chunk_id)

    result = _run(storage, _FixedRetriever(mismatched), course_b)

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["stop_reason"] == "RETRIEVER_SCOPE_MISMATCH"
    assert result["citations"] == []


def test_scoped_retrieval_abstains_when_scope_validation_storage_fails(tmp_path, monkeypatch):
    storage = LearningLogger(tmp_path / "s0-scope-storage-failure.db")
    course_b = _course(storage, "s0-storage-b")
    course_b_result = _indexed_result(storage, course_b, "S0_STORAGE_FAILURE_MARKER")

    def fail_document_lookup(document_id: int):
        raise RuntimeError("controlled scope validation failure")

    monkeypatch.setattr(storage, "document", fail_document_lookup)
    result = _run(storage, _FixedRetriever(course_b_result), course_b)

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["stop_reason"] == "RETRIEVER_SCOPE_MISMATCH"
    assert result["citations"] == []
    assert "S0_STORAGE_FAILURE_MARKER" not in result["answer"]


def test_empty_scoped_retrieval_never_scans_unregistered_local_markdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage = LearningLogger(tmp_path / "s0-local-fallback.db")
    course_b = _course(storage, "s0-fallback-b")
    knowledge_dir = tmp_path / "knowledge_base" / f"course-{course_b}"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "s0-local.md").write_text(
        "# S0\n\nS0 scoped marker is local evidence for the selected course.",
        encoding="utf-8",
    )

    result = _run(storage, _FixedRetriever(), course_b)

    assert result["run_status"] == AgentRunStatus.ABSTAINED
    assert result["citations"] == []
    assert "S0 scoped marker" not in result["answer"]
