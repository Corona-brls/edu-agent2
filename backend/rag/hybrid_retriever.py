"""Hybrid keyword + vector retrieval over indexed document chunks."""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from backend.core.config import get_settings
from backend.rag.course_document_constraint import (
    CourseDocumentConstraintError,
    normalize_course_document_title,
    parse_explicit_course_document_title,
)
from backend.rag.retriever import tokenize
from backend.rag.visibility import is_student_visible_chunk
from backend.rag.vector_store import ChromaVectorStore
from backend.storage.logger import LearningLogger


_VectorInitConfig = tuple[str, str, str, bool]
_VectorInitFailure = tuple[str, str]
_VECTOR_INIT_TYPE_LABEL_MAX_CHARS = 120
_VECTOR_INIT_MESSAGE_MAX_CHARS = 500
_VECTOR_INIT_FALLBACK_TYPE_LABEL = "Exception"
_VECTOR_INIT_FALLBACK_MESSAGE = "Vector store initialization failed"
_VECTOR_INIT_LOCK = threading.Lock()
_VECTOR_INIT_EVENT: threading.Event | None = None
_VECTOR_INIT_CONFIG: _VectorInitConfig | None = None
_VECTOR_INIT_WORKER: threading.Thread | None = None
_VECTOR_INIT_RESULT: Any | None = None
_VECTOR_INIT_FAILURE: _VectorInitFailure | None = None


@dataclass(frozen=True)
class RetrievalResult:
    document_id: int
    chunk_id: int
    source: str
    title: str
    snippet: str
    page: int | None
    section: str
    source_type: str
    keyword_score: float
    vector_score: float
    final_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResponse:
    results: list[RetrievalResult]
    abstained: bool
    threshold: float
    fallback_used: bool = False


@dataclass(frozen=True)
class CourseQueryUpdate:
    """Server-owned classification for one explicit-textbook continuation."""

    update_type: str
    effective_query: str
    previous_fact_fingerprint: tuple[str, ...]
    effective_fact_fingerprint: tuple[str, ...]
    required_fact_fingerprint: tuple[str, ...]


def resolve_course_document_constraint(
    storage: LearningLogger,
    *,
    course_id: int | None,
    user_goal: str,
) -> tuple[str | None, frozenset[int] | None]:
    """Resolve one explicit title to current student-visible SQLite documents."""

    title = parse_explicit_course_document_title(user_goal)
    if title is None:
        return None, None
    if course_id is None:
        raise CourseDocumentConstraintError("COURSE_DOCUMENT_NOT_FOUND")
    try:
        chunks = storage.document_chunks(course_id=int(course_id))
    except Exception as exc:
        raise CourseDocumentConstraintError("COURSE_DOCUMENT_CONSTRAINT_UNAVAILABLE") from exc
    document_ids = {
        int(chunk["document_id"])
        for chunk in chunks
        if is_student_visible_chunk(chunk)
        and normalize_course_document_title(str(chunk.get("title") or "")) == title
    }
    if not document_ids:
        raise CourseDocumentConstraintError("COURSE_DOCUMENT_NOT_FOUND")
    if len(document_ids) != 1:
        raise CourseDocumentConstraintError("COURSE_DOCUMENT_AMBIGUOUS")
    return title, frozenset(document_ids)


class HybridRetriever:
    def __init__(
        self,
        storage: LearningLogger | None = None,
        vector_store: Any | None = None,
        keyword_weight: float | None = None,
        vector_weight: float | None = None,
        min_score: float | None = None,
    ) -> None:
        settings = get_settings()
        self.storage = storage or LearningLogger()
        self.vector_store = vector_store
        self.keyword_weight = settings.retrieval_keyword_weight if keyword_weight is None else keyword_weight
        self.vector_weight = settings.retrieval_vector_weight if vector_weight is None else vector_weight
        self.min_score = settings.retrieval_min_score if min_score is None else min_score

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        course_id: int | None = None,
        allowed_document_ids: set[int] | frozenset[int] | None = None,
        snippet_query: str | None = None,
    ) -> RetrievalResponse:
        results, fallback_used = self.search_with_metadata(
            query,
            top_k=top_k,
            course_id=course_id,
            allowed_document_ids=allowed_document_ids,
            snippet_query=snippet_query,
        )
        return RetrievalResponse(
            results=results,
            abstained=not results or results[0].final_score < self.min_score,
            threshold=self.min_score,
            fallback_used=fallback_used,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        course_id: int | None = None,
        allowed_document_ids: set[int] | frozenset[int] | None = None,
        snippet_query: str | None = None,
    ) -> list[RetrievalResult]:
        return self.retrieve(
            query,
            top_k=top_k,
            course_id=course_id,
            allowed_document_ids=allowed_document_ids,
            snippet_query=snippet_query,
        ).results

    def search_with_metadata(
        self,
        query: str,
        top_k: int | None = None,
        course_id: int | None = None,
        allowed_document_ids: set[int] | frozenset[int] | None = None,
        snippet_query: str | None = None,
    ) -> tuple[list[RetrievalResult], bool]:
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k
        if top_k <= 0 or not query.strip():
            return [], False

        indexed_chunks = _indexed_chunks(self.storage.document_chunks(course_id=course_id))
        allowed: set[int] | None = None
        if allowed_document_ids is not None:
            allowed = {int(value) for value in allowed_document_ids}
            indexed_chunks = [
                chunk for chunk in indexed_chunks if int(chunk["document_id"]) in allowed
            ]
        if not indexed_chunks:
            return [], False

        constrained_document_support = allowed is not None and len(allowed) == 1
        keyword_scores = _keyword_scores(
            query,
            indexed_chunks,
            top_k=len(indexed_chunks) if constrained_document_support else top_k,
        )
        vector_scores, fallback_used = self._vector_scores(query, indexed_chunks, top_k=top_k)
        chunk_ids = set(keyword_scores) | set(vector_scores)
        if not chunk_ids:
            return [], fallback_used

        initial_chunk_by_id = {int(chunk["id"]): chunk for chunk in indexed_chunks}
        try:
            current_chunks = _indexed_chunks(self.storage.document_chunks(course_id=course_id))
        except Exception:
            return [], fallback_used
        current_chunk_by_id: dict[int, dict[str, Any]] = {}
        for current in current_chunks:
            if (
                allowed_document_ids is not None
                and int(current["document_id"]) not in allowed_document_ids
            ):
                continue
            chunk_id = int(current["id"])
            initial = initial_chunk_by_id.get(chunk_id)
            if initial is None:
                continue
            if int(current["document_id"]) != int(initial["document_id"]):
                continue
            if str(current.get("content_hash") or "") != str(initial.get("content_hash") or ""):
                continue
            if str(current.get("chunk_hash") or "") != str(initial.get("chunk_hash") or ""):
                continue
            current_chunk_by_id[chunk_id] = current

        scored_candidates: list[tuple[dict[str, Any], int, float, float, float, float]] = []
        for chunk_id in chunk_ids:
            chunk = current_chunk_by_id.get(chunk_id)
            if not chunk:
                continue
            keyword_score = keyword_scores.get(chunk_id, 0.0)
            vector_score = vector_scores.get(chunk_id, 0.0)
            final_score = _weighted_score(
                keyword_score=keyword_score,
                vector_score=vector_score,
                keyword_weight=self.keyword_weight,
                vector_weight=self.vector_weight if not fallback_used else 0.0,
            )
            coverage = _query_coverage(query, chunk.get("chunk_text", ""))
            if coverage < 0.2 and len(tokenize(query)) >= 4:
                final_score *= coverage
            scored_candidates.append(
                (chunk, chunk_id, keyword_score, vector_score, final_score, coverage)
            )

        anchor_exists = any(
            final_score >= self.min_score
            for _chunk, _chunk_id, _keyword, _vector, final_score, _coverage in scored_candidates
        )
        results: list[RetrievalResult] = []
        for chunk, chunk_id, keyword_score, vector_score, final_score, coverage in scored_candidates:
            is_anchor = final_score >= self.min_score
            is_support = (
                constrained_document_support
                and anchor_exists
                and final_score > 0.0
                and coverage >= 0.2
            )
            if not is_anchor and not is_support:
                continue
            snippet = retrieval_snippet(
                chunk.get("chunk_text", ""),
                query=(snippet_query or query) if constrained_document_support else None,
            )
            if constrained_document_support and not snippet:
                continue
            results.append(
                RetrievalResult(
                    document_id=int(chunk["document_id"]),
                    chunk_id=chunk_id,
                    source=chunk.get("filename", ""),
                    title=chunk.get("title", ""),
                    snippet=snippet,
                    page=chunk.get("page"),
                    section=chunk.get("section", ""),
                    source_type=chunk.get("source_type") or chunk.get("file_type", ""),
                    keyword_score=round(keyword_score, 4),
                    vector_score=round(vector_score, 4),
                    final_score=round(final_score, 4),
                    metadata={
                        "fallback": fallback_used,
                        "content_hash": chunk.get("content_hash", ""),
                        "query_coverage": round(coverage, 4),
                        "constraint_support": bool(is_support and not is_anchor),
                    },
                )
            )

        results.sort(key=lambda item: (-item.final_score, item.document_id, item.chunk_id))
        return results[:top_k], fallback_used

    def _vector_scores(
        self,
        query: str,
        indexed_chunks: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[dict[int, float], bool]:
        vector_store = self.vector_store
        if vector_store is None:
            try:
                vector_store = _bounded_vector_store_init()
            except Exception:
                return {}, True
        try:
            matches: list[dict[str, Any]] = []
            requested = max(1, top_k)
            previous_count = -1
            # The vector collection can still contain withdrawn, deleted, or
            # cross-course rows.  Keep expanding until enough current SQLite
            # rows are found or the finite collection stops growing; a fixed
            # cap can permanently starve a valid row ranked after that cap.
            while True:
                matches = vector_store.query(query, top_k=requested)
                valid_ids = {int(chunk["id"]) for chunk in indexed_chunks}
                eligible_count = 0
                for match in matches:
                    metadata = match.get("metadata") or {}
                    raw_chunk_id = metadata.get("chunk_id") or match.get("id")
                    try:
                        eligible_count += int(raw_chunk_id) in valid_ids
                    except (TypeError, ValueError):
                        continue
                returned_count = len(matches)
                if (
                    eligible_count >= top_k
                    or returned_count < requested
                    or returned_count <= previous_count
                ):
                    break
                previous_count = returned_count
                requested *= 2
        except Exception:
            return {}, True

        valid_chunk_ids = {int(chunk["id"]) for chunk in indexed_chunks}
        raw_scores: dict[int, float] = {}
        for match in matches:
            metadata = match.get("metadata") or {}
            raw_chunk_id = metadata.get("chunk_id") or match.get("id")
            try:
                chunk_id = int(raw_chunk_id)
            except (TypeError, ValueError):
                continue
            if chunk_id not in valid_chunk_ids:
                continue
            raw_score = max(0.0, min(1.0, float(match.get("score", 0.0))))
            raw_scores[chunk_id] = max(raw_scores.get(chunk_id, 0.0), raw_score)
        return raw_scores, not raw_scores


def _bounded_vector_store_init() -> Any:
    global _VECTOR_INIT_FAILURE

    timeout_seconds = max(0.01, float(os.getenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "2")))
    deadline = time.monotonic() + timeout_seconds
    config = _vector_init_config()

    while True:
        start_worker: threading.Thread | None = None
        wait_event: threading.Event | None = None
        join_worker: threading.Thread | None = None
        cached_result: Any | None = None
        cached_failure: _VectorInitFailure | None = None

        with _VECTOR_INIT_LOCK:
            if _VECTOR_INIT_CONFIG == config and _VECTOR_INIT_EVENT is not None:
                if _VECTOR_INIT_EVENT.is_set():
                    cached_result = _VECTOR_INIT_RESULT
                    cached_failure = _VECTOR_INIT_FAILURE
                else:
                    wait_event = _VECTOR_INIT_EVENT
            elif _VECTOR_INIT_EVENT is not None and not _VECTOR_INIT_EVENT.is_set():
                wait_event = _VECTOR_INIT_EVENT
            elif _VECTOR_INIT_WORKER is not None and _VECTOR_INIT_WORKER.is_alive():
                join_worker = _VECTOR_INIT_WORKER
            else:
                event = threading.Event()
                worker = threading.Thread(
                    target=_construct_vector_store,
                    args=(config, event),
                    daemon=True,
                    name="chroma-vector-store-init",
                )
                _replace_vector_init_slot(config=config, event=event, worker=worker)
                start_worker = worker
                wait_event = event

        if cached_failure is not None:
            _raise_vector_init_failure(cached_failure)
        if cached_result is not None:
            return cached_result

        if start_worker is not None:
            start_failure: _VectorInitFailure | None = None
            try:
                start_worker.start()
            except BaseException as exc:
                start_failure = _describe_vector_init_failure(exc)
                try:
                    with _VECTOR_INIT_LOCK:
                        if _VECTOR_INIT_CONFIG == config and _VECTOR_INIT_WORKER is start_worker:
                            _VECTOR_INIT_FAILURE = start_failure
                finally:
                    wait_event.set()
            if start_failure is not None:
                _raise_vector_init_failure(start_failure)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Chroma initialization exceeded {timeout_seconds:.2f}s")
        if join_worker is not None:
            join_worker.join(remaining)
            if join_worker.is_alive():
                raise TimeoutError(f"Chroma initialization exceeded {timeout_seconds:.2f}s")
            continue
        if wait_event is None or not wait_event.wait(remaining):
            raise TimeoutError(f"Chroma initialization exceeded {timeout_seconds:.2f}s")


def _vector_init_config() -> _VectorInitConfig:
    settings = get_settings()
    offline = os.getenv("EMBEDDING_OFFLINE", "true").strip().lower() in {"1", "true", "yes", "on"}
    return (
        str(settings.vector_store_dir.expanduser().resolve()),
        settings.vector_collection_name,
        settings.embedding_model,
        offline,
    )


def _describe_vector_init_failure(error: BaseException) -> _VectorInitFailure:
    try:
        type_label = type(error).__name__
        if type(type_label) is not str:
            type_label = _VECTOR_INIT_FALLBACK_TYPE_LABEL
        else:
            type_label = type_label[:_VECTOR_INIT_TYPE_LABEL_MAX_CHARS]
            if not type_label:
                type_label = _VECTOR_INIT_FALLBACK_TYPE_LABEL
    except BaseException:
        type_label = _VECTOR_INIT_FALLBACK_TYPE_LABEL

    try:
        message = str(error)
        if type(message) is not str:
            message = _VECTOR_INIT_FALLBACK_MESSAGE
        else:
            message = " ".join(message.split())[:_VECTOR_INIT_MESSAGE_MAX_CHARS]
            if not message:
                message = _VECTOR_INIT_FALLBACK_MESSAGE
    except BaseException:
        message = _VECTOR_INIT_FALLBACK_MESSAGE
    return type_label, message


def _raise_vector_init_failure(failure: _VectorInitFailure) -> None:
    type_label, message = failure
    raise RuntimeError(f"{type_label}: {message}") from None


def _replace_vector_init_slot(
    *,
    config: _VectorInitConfig,
    event: threading.Event,
    worker: threading.Thread,
) -> None:
    global _VECTOR_INIT_CONFIG
    global _VECTOR_INIT_EVENT
    global _VECTOR_INIT_WORKER
    global _VECTOR_INIT_RESULT
    global _VECTOR_INIT_FAILURE

    _VECTOR_INIT_CONFIG = config
    _VECTOR_INIT_EVENT = event
    _VECTOR_INIT_WORKER = worker
    _VECTOR_INIT_RESULT = None
    _VECTOR_INIT_FAILURE = None


def _construct_vector_store(config: _VectorInitConfig, event: threading.Event) -> None:
    global _VECTOR_INIT_RESULT
    global _VECTOR_INIT_FAILURE

    result: Any | None = None
    failure: _VectorInitFailure | None = None
    try:
        result = ChromaVectorStore(
            persist_dir=config[0],
            collection_name=config[1],
            embedding_model=config[2],
        )
    except BaseException as exc:
        failure = _describe_vector_init_failure(exc)

    try:
        with _VECTOR_INIT_LOCK:
            if (
                _VECTOR_INIT_CONFIG == config
                and _VECTOR_INIT_EVENT is event
                and _VECTOR_INIT_WORKER is threading.current_thread()
            ):
                _VECTOR_INIT_RESULT = result
                _VECTOR_INIT_FAILURE = failure
    finally:
        event.set()


def _indexed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [chunk for chunk in chunks if is_student_visible_chunk(chunk)]


def _keyword_scores(query: str, chunks: list[dict[str, Any]], top_k: int) -> dict[int, float]:
    corpus = [chunk.get("chunk_text", "") for chunk in chunks]
    if not corpus:
        return {}
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    ranked_indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
    raw_scores = {
        int(chunks[index]["id"]): float(scores[index])
        for index in ranked_indexes
        if float(scores[index]) > 0
    }
    return raw_scores


def _weighted_score(
    keyword_score: float,
    vector_score: float,
    keyword_weight: float,
    vector_weight: float,
) -> float:
    total_weight = keyword_weight + vector_weight
    if total_weight <= 0:
        return 0.0
    return (keyword_score * keyword_weight + vector_score * vector_weight) / total_weight


def _query_coverage(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def retrieval_snippet(
    text: str,
    max_chars: int = 240,
    *,
    query: str | None = None,
) -> str:
    structured = _normalize_source_structure(text)
    if len(structured) <= max_chars:
        return " ".join(structured.split())
    if query and query.strip():
        selected = _query_aware_complete_sentences(structured, query, max_chars=max_chars)
        if selected:
            return selected
        return ""
    cleaned = " ".join(structured.split())
    return f"{cleaned[:max_chars].rstrip()}..."


def _query_aware_complete_sentences(text: str, query: str, *, max_chars: int) -> str:
    """Select source-locatable logical units without treating the query as evidence."""

    concepts = _query_fact_concepts(query)
    if not concepts or max_chars <= 0:
        return ""
    units = _source_logical_units(text)
    candidates: list[tuple[int, str, set[int], int]] = []
    for index, unit in enumerate(units):
        covered = {
            concept_index
            for concept_index, aliases in enumerate(concepts)
            if any(alias in unit.casefold() for alias in aliases)
        }
        if covered and not _is_source_front_matter(unit):
            match_strength = sum(
                1
                for aliases in concepts
                for alias in aliases
                if alias in unit.casefold()
            )
            candidates.append((index, unit, covered, match_strength))
    if not candidates:
        return ""

    selected: list[tuple[int, str]] = []
    covered: set[int] = set()
    remaining = list(candidates)
    while remaining:
        remaining.sort(
            key=lambda item: (
                -item[3],
                -len(item[2] - covered),
                -_source_unit_specificity(item[1]),
                -len(item[2]),
                len(item[1]),
                item[0],
            )
        )
        index, unit, unit_coverage, _match_strength = remaining.pop(0)
        if not (unit_coverage - covered):
            continue
        proposed = selected + [(index, unit)]
        rendered = _render_selected_sentences(proposed)
        if len(rendered) > max_chars:
            continue
        selected = proposed
        covered.update(unit_coverage)

    if selected:
        return _render_selected_sentences(selected)

    best = min(
        candidates,
        key=lambda item: (-item[3], -len(item[2]), item[0]),
    )
    return _bounded_query_window(best[1], concepts, max_chars=max_chars)


def _normalize_source_structure(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _source_logical_units(text: str) -> list[str]:
    units: list[str] = []
    for raw_line in _normalize_source_structure(text).split("\n"):
        line = " ".join(raw_line.split())
        if not line:
            continue
        if re.match(r"^(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)", line):
            prefix_match = re.match(r"^(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)", line)
            prefix = prefix_match.group(0) if prefix_match else ""
            body = line[len(prefix) :]
        else:
            prefix = ""
            body = line
        pieces = _split_source_sentences(body)
        if len(pieces) == 1:
            units.append(f"{prefix}{pieces[0]}".strip())
        else:
            units.extend(piece for piece in pieces if piece)
    return units


def _split_source_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    pieces: list[str] = []
    start = 0
    for match in re.finditer(r"[。！？!?；;]|(?<!\d)\.(?=\s|$)", text):
        end = match.end()
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces or [text.strip()]


_QUERY_CONTROL_TERMS = frozenset(
    {
        "请", "只", "根据", "依据", "当前", "课程", "发布", "教材", "资料", "回答",
        "解释", "说明", "关系", "引用", "给出", "本次", "continue", "继续", "补充",
        "信息", "新的", "事实", "主题",
        "增加", "详细", "一点", "展开", "一种", "另一种", "说法", "简要", "总结",
        "具体", "一些", "不要", "不再", "再", "改为", "改成", "转为", "而是",
        "请用",
        "please", "elaborate", "briefly", "summarize", "summary", "rephrase",
        "wording", "explain", "describe", "discuss", "answer", "only", "instead",
        "rather", "on", "the", "a", "an", "more", "detail", "detailed", "specifically",
    }
)

_QUERY_CONCEPT_ALIASES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("基线", "baseline"), ("基线", "基线组", "baseline")),
    (("对照", "control"), ("对照", "对照组", "处理组", "比较组", "control")),
    (
        ("评价指标", "评估指标", "evaluation metric", "metrics"),
        ("评价指标", "评估指标", "指标", "混淆矩阵", "精确率", "召回率", "样本量", "metrics"),
    ),
    (
        ("复现实验记录", "复现", "reproducible", "reproduction"),
        (
            "版本", "数据划分", "随机种子", "运行资源", "固定环境", "version", "dataset", "random seed",
        ),
    ),
    (
        ("复现实验记录", "复现", "reproducible", "reproduction"),
        (
            "复现实验记录", "复现", "重复实验", "重复运行", "至少重复", "每轮",
            "reproducible", "reproduction", "record",
        ),
    ),
)


def _query_fact_concepts(query: str) -> list[tuple[str, ...]]:
    normalized = re.sub(r"《[^《》]*》", " ", _normalize_source_structure(query)).casefold()
    concepts: list[tuple[str, ...]] = []
    consumed: set[str] = set()
    for triggers, aliases in _QUERY_CONCEPT_ALIASES:
        matched_triggers = [
            trigger for trigger in triggers if trigger.casefold() in normalized
        ]
        if matched_triggers:
            concepts.append(tuple(alias.casefold() for alias in aliases))
            consumed.update(
                token.casefold()
                for trigger in matched_triggers
                for token in tokenize(trigger)
            )
    for token in tokenize(normalized):
        folded = token.casefold()
        if (
            folded in _QUERY_CONTROL_TERMS
            or folded in consumed
            or any(folded in aliases for aliases in concepts)
            or len(folded) < 2
        ):
            continue
        concepts.append((folded,))
    return concepts


def query_fact_fingerprint(query: str) -> tuple[str, ...]:
    """Return a stable, source-independent fingerprint for requested facts."""

    return tuple(
        sorted(
            {
                "\x1f".join(aliases)
                for aliases in _query_fact_concepts(query)
                if aliases
            }
        )
    )


_COURSE_QUERY_REPLACEMENT_PATTERNS = (
    re.compile(
        r"^(?:不要(?:再)?(?:回答|说明|解释|讨论)?|不再(?:回答|说明|解释|讨论)?)"
        r".{1,300}?[，,；;]\s*(?:改为|转为|改成|而是|只)\s*"
        r"(?:问|回答|说明|解释|讨论)?\s*(?P<replacement>.+)$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^(?:改为|转为|改成)\s*(?:只)?\s*(?:问|回答|说明|解释|讨论)?\s*"
        r"(?P<replacement>.+)$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^instead\s+(?:(?:explain|describe|discuss|answer)\s+)?(?:only\s+)?"
        r"(?P<replacement>.+)$",
        flags=re.IGNORECASE,
    ),
)


def _explicit_replacement_clause(supplemental_query: str) -> str | None:
    normalized = " ".join(_normalize_source_structure(supplemental_query).split())
    for pattern in _COURSE_QUERY_REPLACEMENT_PATTERNS:
        matched = pattern.fullmatch(normalized)
        if matched:
            replacement = str(matched.group("replacement") or "").strip(" ，,；;。.")
            return replacement or None
    return None


def classify_course_query_update(
    previous_query: str,
    supplemental_query: str,
) -> CourseQueryUpdate:
    """Separate content facts from presentation-only and explicit replacement updates."""

    previous_facts = query_fact_fingerprint(previous_query)
    replacement = _explicit_replacement_clause(supplemental_query)
    if replacement is not None:
        replacement_facts = query_fact_fingerprint(replacement)
        if replacement_facts:
            return CourseQueryUpdate(
                update_type="replacement",
                effective_query=replacement,
                previous_fact_fingerprint=previous_facts,
                effective_fact_fingerprint=replacement_facts,
                required_fact_fingerprint=replacement_facts,
            )

    supplemental_facts = query_fact_fingerprint(supplemental_query)
    previous_set = set(previous_facts)
    new_facts = tuple(sorted(set(supplemental_facts) - previous_set))
    if not supplemental_facts or not new_facts:
        return CourseQueryUpdate(
            update_type="control_only",
            effective_query=previous_query,
            previous_fact_fingerprint=previous_facts,
            effective_fact_fingerprint=previous_facts,
            required_fact_fingerprint=(),
        )

    effective_query = f"{previous_query}\n补充信息: {supplemental_query}"
    return CourseQueryUpdate(
        update_type="additive",
        effective_query=effective_query,
        previous_fact_fingerprint=previous_facts,
        effective_fact_fingerprint=query_fact_fingerprint(effective_query),
        required_fact_fingerprint=new_facts,
    )


def supported_query_fact_fingerprint(
    query: str,
    source_texts: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Return requested fact concepts that are present in approved source text."""

    folded_sources = tuple(str(text or "").casefold() for text in source_texts)
    supported: set[str] = set()
    for aliases in _query_fact_concepts(query):
        if aliases and any(
            alias and alias in source
            for source in folded_sources
            for alias in aliases
        ):
            supported.add("\x1f".join(aliases))
    return tuple(sorted(supported))


def _is_source_front_matter(unit: str) -> bool:
    stripped = unit.strip()
    return bool(
        re.match(
            r"^>\s*(?:source_id|revision|review_state|reviewer|reviewed_at)\s*:",
            stripped,
            flags=re.IGNORECASE,
        )
    )


def _source_unit_specificity(unit: str) -> int:
    folded = unit.casefold()
    markers = (
        "建立", "固定", "采用", "应当", "应该", "应保存", "至少", "报告", "记录",
        "计算", "比较", "保持", "use ", "uses ", "record", "report", "compare",
    )
    return sum(1 for marker in markers if marker in folded)


def _bounded_query_window(
    unit: str,
    concepts: list[tuple[str, ...]],
    *,
    max_chars: int,
) -> str:
    if max_chars <= 0:
        return ""
    folded = unit.casefold()
    positions = [
        folded.find(alias)
        for aliases in concepts
        for alias in aliases
        if alias and folded.find(alias) >= 0
    ]
    if not positions:
        return ""
    center = min(positions)
    prefix = "…" if center > 0 else ""
    available = max_chars - len(prefix)
    start = max(0, center - available // 3)
    if start > 0:
        boundary = max(unit.rfind(mark, 0, start) for mark in (" ", "，", "、", "；", ";", "。"))
        if boundary >= 0:
            start = boundary + 1
    suffix_needed = start + available < len(unit)
    if suffix_needed:
        available -= 1
    end = min(len(unit), start + max(0, available))
    if end < len(unit):
        boundary = max(unit.rfind(mark, start, end) for mark in (" ", "，", "、", "；", ";", "。"))
        if boundary > start:
            end = boundary + 1
    rendered = f"{'…' if start > 0 else ''}{unit[start:end].strip()}{'…' if end < len(unit) else ''}"
    return rendered[:max_chars]


def _render_selected_sentences(selected: list[tuple[int, str]]) -> str:
    if not selected:
        return ""
    ordered = sorted(selected)
    rendered: list[str] = []
    previous_index: int | None = None
    for index, sentence in ordered:
        if previous_index is not None and index != previous_index + 1:
            rendered.append("…")
        rendered.append(sentence)
        previous_index = index
    return "".join(rendered)
