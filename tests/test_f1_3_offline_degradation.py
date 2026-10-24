from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.core.errors import LLMProviderError
from backend.llm.deepseek_provider import DeepSeekProvider
from backend.llm.schemas import LLMRequest
from backend.rag import hybrid_retriever as hybrid_module
from backend.rag.document_parser import Document, Page
from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.retriever import tokenize
from backend.rag.vector_store import _sentence_transformer_embedding
from backend.storage.logger import LearningLogger, sha256_text


class _EmbeddingProbe:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _SlowFailVectorStore:
    workers: list[threading.Thread] = []
    completed = threading.Event()

    def __init__(self, **kwargs):
        self.workers.append(threading.current_thread())
        try:
            time.sleep(0.2)
            raise RuntimeError("embedding model unavailable")
        finally:
            self.completed.set()


class _SlowFailCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        time.sleep(0.2)
        raise TimeoutError("upstream timeout")


class _FakeClient:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()


def _indexed_storage(tmp_path):
    storage = LearningLogger(tmp_path / "offline.db")
    storage.seed_demo_identity()
    course_id = storage.default_course_id()
    document = Document(
        source_path=tmp_path / "nas.md",
        filename="nas.md",
        file_type=".md",
        title="NAS offline notes",
        text="NAS authentication fallback evidence",
        pages=[Page(text="NAS authentication fallback evidence", page=1, section="NAS", source_type="markdown")],
    )
    saved = storage.save_document(
        document=document,
        content_hash=sha256_text(document.text),
        chunks=["NAS authentication fallback evidence"],
        byte_size=40,
        index_status="INDEXED",
        course_id=course_id,
    )
    storage.set_document_publication(int(saved["id"]), published=True, actor_id=1)
    return storage


def test_default_sentence_transformer_loader_only_uses_local_cache(monkeypatch):
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setattr(
        "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
        _EmbeddingProbe,
    )

    embedding = _sentence_transformer_embedding("offline-test-model")

    assert embedding.kwargs["local_files_only"] is True


def test_hybrid_retriever_bounds_vector_store_initialization(tmp_path, monkeypatch):
    _SlowFailVectorStore.workers.clear()
    _SlowFailVectorStore.completed.clear()
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "bounded-vector"))
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.02")
    monkeypatch.setattr(hybrid_module, "ChromaVectorStore", _SlowFailVectorStore)
    retriever = HybridRetriever(storage=_indexed_storage(tmp_path), min_score=0.01)
    tokenize("warm up tokenizer")

    try:
        started = time.perf_counter()
        response = retriever.retrieve("NAS authentication", top_k=1)
        elapsed = time.perf_counter() - started
    finally:
        assert _SlowFailVectorStore.completed.wait(1.0)
        for worker in _SlowFailVectorStore.workers:
            worker.join(1.0)

    assert elapsed < 0.15
    assert len(_SlowFailVectorStore.workers) == 1
    assert all(not worker.is_alive() for worker in _SlowFailVectorStore.workers)
    assert response.fallback_used is True
    assert response.results
    assert response.results[0].metadata["fallback"] is True


def test_vector_store_initialization_does_not_overlap_for_same_config(tmp_path, monkeypatch):
    release = threading.Event()
    overlap = threading.Event()
    state_lock = threading.Lock()
    workers: list[threading.Thread] = []
    active = 0

    class _BlockingVectorStore:
        def __init__(self, **kwargs):
            nonlocal active
            with state_lock:
                workers.append(threading.current_thread())
                active += 1
                if active > 1:
                    overlap.set()
            try:
                release.wait(1.0)
            finally:
                with state_lock:
                    active -= 1

    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "overlap-vector"))
    monkeypatch.setenv("VECTOR_COLLECTION_NAME", "overlap_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "overlap-model")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.02")
    monkeypatch.setattr(hybrid_module, "ChromaVectorStore", _BlockingVectorStore)

    try:
        with pytest.raises(TimeoutError):
            hybrid_module._bounded_vector_store_init()
        with pytest.raises(TimeoutError):
            hybrid_module._bounded_vector_store_init()
        observed_overlap = overlap.is_set() or len(workers) > 1
    finally:
        release.set()
        for worker in workers:
            worker.join(1.0)

    assert all(not worker.is_alive() for worker in workers)
    assert len(workers) == 1
    assert observed_overlap is False


def test_vector_store_late_success_is_reused(tmp_path, monkeypatch):
    release = threading.Event()
    started = threading.Event()
    workers: list[threading.Thread] = []
    instances: list[object] = []

    class _LateSuccessVectorStore:
        def __init__(self, **kwargs):
            workers.append(threading.current_thread())
            instances.append(self)
            started.set()
            release.wait(1.0)

    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "late-success-vector"))
    monkeypatch.setenv("VECTOR_COLLECTION_NAME", "late_success_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "late-success-model")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.02")
    monkeypatch.setattr(hybrid_module, "ChromaVectorStore", _LateSuccessVectorStore)

    try:
        with pytest.raises(TimeoutError):
            hybrid_module._bounded_vector_store_init()
        assert started.wait(0.5)
        release.set()
        workers[0].join(1.0)
        assert not workers[0].is_alive()

        reused = hybrid_module._bounded_vector_store_init()
    finally:
        release.set()
        for worker in workers:
            worker.join(1.0)

    assert len(workers) == 1
    assert reused is instances[0]


def test_vector_store_failure_is_cached_for_same_config(tmp_path, monkeypatch):
    calls = 0

    class _FailingVectorStore:
        def __init__(self, **kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("deterministic vector init failure")

    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "failed-vector"))
    monkeypatch.setenv("VECTOR_COLLECTION_NAME", "failed_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "failed-model")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setattr(hybrid_module, "ChromaVectorStore", _FailingVectorStore)

    with pytest.raises(RuntimeError, match="deterministic vector init failure"):
        hybrid_module._bounded_vector_store_init()
    with pytest.raises(RuntimeError, match="deterministic vector init failure"):
        hybrid_module._bounded_vector_store_init()

    assert calls == 1


def test_vector_store_cached_failure_uses_fresh_bounded_exceptions(tmp_path, monkeypatch):
    calls = 0
    failures: list[RuntimeError] = []

    class _FailingVectorStore:
        def __init__(self, **kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("cached failure traceback probe")

    def traceback_depth(error: BaseException) -> int:
        depth = 0
        traceback = error.__traceback__
        while traceback is not None:
            depth += 1
            traceback = traceback.tb_next
        return depth

    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "cached-failure-vector"))
    monkeypatch.setenv("VECTOR_COLLECTION_NAME", "cached_failure_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "cached-failure-model")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setattr(hybrid_module, "ChromaVectorStore", _FailingVectorStore)

    try:
        for _ in range(101):
            try:
                hybrid_module._bounded_vector_store_init()
            except RuntimeError as exc:
                failures.append(exc)
            else:
                pytest.fail("vector initialization unexpectedly succeeded")
    finally:
        worker = hybrid_module._VECTOR_INIT_WORKER
        if worker is not None:
            worker.join(1.0)

    depths = [traceback_depth(error) for error in failures]
    unique_instances = len({id(error) for error in failures})

    assert calls == 1
    assert len(failures) == 101
    assert unique_instances == len(failures) and max(depths) <= 4 and len(set(depths)) == 1, {
        "unique_instances": unique_instances,
        "failure_count": len(failures),
        "min_depth": min(depths),
        "max_depth": max(depths),
    }
    assert isinstance(hybrid_module._VECTOR_INIT_FAILURE, tuple)
    assert all(isinstance(part, str) for part in hybrid_module._VECTOR_INIT_FAILURE)
    assert all(error.__context__ is None and error.__cause__ is None for error in failures)


def test_vector_store_cached_failure_keeps_real_retriever_fallback_available(tmp_path, monkeypatch):
    calls = 0

    class _FailingVectorStore:
        def __init__(self, **kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("retriever fallback vector failure")

    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "retriever-failure-vector"))
    monkeypatch.setenv("VECTOR_COLLECTION_NAME", "retriever_failure_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "retriever-failure-model")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setattr(hybrid_module, "ChromaVectorStore", _FailingVectorStore)
    retriever = HybridRetriever(storage=_indexed_storage(tmp_path), min_score=0.01)
    tokenize("warm up tokenizer")

    try:
        responses = [retriever.retrieve("NAS authentication", top_k=1) for _ in range(101)]
    finally:
        worker = hybrid_module._VECTOR_INIT_WORKER
        if worker is not None:
            worker.join(1.0)

    assert calls == 1
    assert all(response.fallback_used for response in responses)
    assert all(response.results for response in responses)
    assert all(response.results[0].metadata["fallback"] is True for response in responses)
    assert isinstance(hybrid_module._VECTOR_INIT_FAILURE, tuple)
    assert not any(isinstance(part, BaseException) for part in hybrid_module._VECTOR_INIT_FAILURE)


def test_vector_store_cached_failure_is_independent_for_concurrent_readers(tmp_path, monkeypatch):
    calls = 0
    reader_count = 8
    start = threading.Barrier(reader_count + 1)
    result_lock = threading.Lock()
    failures: list[BaseException] = []
    unexpected: list[object] = []
    readers: list[threading.Thread] = []

    class _FailingVectorStore:
        def __init__(self, **kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("concurrent cached vector failure")

    def read_failure() -> None:
        try:
            start.wait(1.0)
            hybrid_module._bounded_vector_store_init()
        except RuntimeError as exc:
            with result_lock:
                failures.append(exc)
        except BaseException as exc:
            with result_lock:
                unexpected.append(exc)
        else:
            with result_lock:
                unexpected.append("vector initialization unexpectedly succeeded")

    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "concurrent-failure-vector"))
    monkeypatch.setenv("VECTOR_COLLECTION_NAME", "concurrent_failure_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "concurrent-failure-model")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setattr(hybrid_module, "ChromaVectorStore", _FailingVectorStore)

    try:
        with pytest.raises(RuntimeError, match="concurrent cached vector failure"):
            hybrid_module._bounded_vector_store_init()
        for index in range(reader_count):
            reader = threading.Thread(target=read_failure, name=f"cached-failure-reader-{index}")
            readers.append(reader)
            reader.start()
        start.wait(1.0)
        for reader in readers:
            reader.join(1.0)
    finally:
        try:
            start.abort()
        except threading.BrokenBarrierError:
            pass
        for reader in readers:
            reader.join(1.0)
        worker = hybrid_module._VECTOR_INIT_WORKER
        if worker is not None:
            worker.join(1.0)

    assert all(not reader.is_alive() for reader in readers)
    assert unexpected == []
    assert calls == 1
    assert len(failures) == reader_count
    assert len({id(error) for error in failures}) == reader_count
    assert all(error.__context__ is None and error.__cause__ is None for error in failures)


def test_vector_store_thread_start_failure_uses_lightweight_cached_description(tmp_path, monkeypatch):
    original_error = RuntimeError("controlled thread start failure")
    start_calls = 0

    class _StartFailThread:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get("name")

        def start(self):
            nonlocal start_calls
            start_calls += 1
            raise original_error

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    class _ThreadingFacade:
        Event = threading.Event
        Thread = _StartFailThread
        current_thread = staticmethod(threading.current_thread)

    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "thread-start-failure-vector"))
    monkeypatch.setenv("VECTOR_COLLECTION_NAME", "thread_start_failure_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "thread-start-failure-model")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setattr(hybrid_module, "threading", _ThreadingFacade)

    failures: list[RuntimeError] = []
    for _ in range(2):
        with pytest.raises(RuntimeError, match="controlled thread start failure") as captured:
            hybrid_module._bounded_vector_store_init()
        failures.append(captured.value)

    assert start_calls == 1
    assert failures[0] is not failures[1]
    assert all(error is not original_error for error in failures)
    assert all(error.__context__ is None and error.__cause__ is None for error in failures)
    assert isinstance(hybrid_module._VECTOR_INIT_FAILURE, tuple)
    assert all(isinstance(part, str) for part in hybrid_module._VECTOR_INIT_FAILURE)
    assert not any(isinstance(part, BaseException) for part in hybrid_module._VECTOR_INIT_FAILURE)


def _reset_vector_init_slot_for_test() -> None:
    with hybrid_module._VECTOR_INIT_LOCK:
        event = hybrid_module._VECTOR_INIT_EVENT
        worker = hybrid_module._VECTOR_INIT_WORKER
    if event is not None:
        event.set()
    if worker is not None:
        worker.join(1.0)
    with hybrid_module._VECTOR_INIT_LOCK:
        hybrid_module._VECTOR_INIT_EVENT = None
        hybrid_module._VECTOR_INIT_CONFIG = None
        hybrid_module._VECTOR_INIT_WORKER = None
        hybrid_module._VECTOR_INIT_RESULT = None
        hybrid_module._VECTOR_INIT_FAILURE = None


@pytest.fixture
def isolated_vector_init_slot():
    _reset_vector_init_slot_for_test()
    with hybrid_module._VECTOR_INIT_LOCK:
        assert hybrid_module._VECTOR_INIT_EVENT is None
        assert hybrid_module._VECTOR_INIT_CONFIG is None
        assert hybrid_module._VECTOR_INIT_WORKER is None
        assert hybrid_module._VECTOR_INIT_RESULT is None
        assert hybrid_module._VECTOR_INIT_FAILURE is None
    yield
    _reset_vector_init_slot_for_test()


class _SecondaryStringificationFailure(Exception):
    pass


class _BadStringVectorFailure(BaseException):
    def __str__(self):
        raise _SecondaryStringificationFailure("secondary stringification failure")


def _observed_failure(error: BaseException) -> dict[str, str]:
    try:
        message = str(error)
    except BaseException as string_error:
        message = f"<str raised {type(string_error).__name__}>"
    return {"type": type(error).__name__, "message": message}


def test_vector_store_bad_exception_string_worker_reaches_terminal_failure(
    tmp_path,
    monkeypatch,
    isolated_vector_init_slot,
):
    calls = 0
    original_error = _BadStringVectorFailure("unreadable worker failure")
    failures: list[BaseException] = []

    class _BadStringFailingVectorStore:
        def __init__(self, **kwargs):
            nonlocal calls
            calls += 1
            raise original_error

    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "bad-string-worker-vector"))
    monkeypatch.setenv("VECTOR_COLLECTION_NAME", "bad_string_worker_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "bad-string-worker-model")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.03")
    monkeypatch.setattr(hybrid_module, "ChromaVectorStore", _BadStringFailingVectorStore)

    event = None
    worker = None
    try:
        for _ in range(3):
            try:
                hybrid_module._bounded_vector_store_init()
            except BaseException as exc:
                failures.append(exc)
            else:
                failures.append(AssertionError("vector initialization unexpectedly succeeded"))
        with hybrid_module._VECTOR_INIT_LOCK:
            event = hybrid_module._VECTOR_INIT_EVENT
            failure_state = hybrid_module._VECTOR_INIT_FAILURE
            result_state = hybrid_module._VECTOR_INIT_RESULT
            worker = hybrid_module._VECTOR_INIT_WORKER
        if worker is not None:
            worker.join(1.0)
        event_is_set = event is not None and event.is_set()
        worker_is_alive = worker is not None and worker.is_alive()
    finally:
        if event is not None:
            event.set()
        if worker is not None:
            worker.join(1.0)

    observed = [_observed_failure(error) for error in failures]
    messages = [item["message"] for item in observed]
    valid_description = (
        isinstance(failure_state, tuple)
        and len(failure_state) == 2
        and all(type(part) is str and len(part) <= 500 for part in failure_state)
    )
    target_terminal_state = (
        calls == 1
        and len(failures) == 3
        and all(type(error) is RuntimeError for error in failures)
        and len({id(error) for error in failures}) == 3
        and not any(isinstance(error, TimeoutError) for error in failures)
        and not any(isinstance(error, _SecondaryStringificationFailure) for error in failures)
        and all("Vector store initialization failed" in message for message in messages)
        and all("secondary stringification failure" not in message for message in messages)
        and event_is_set
        and valid_description
        and result_state is None
        and worker is not None
        and not worker_is_alive
        and all(error is not original_error for error in failures)
        and all(error.__context__ is None and error.__cause__ is None for error in failures)
    )
    facts = {
        "observed": observed,
        "event_is_set": event_is_set,
        "failure_state": failure_state,
        "result_state": result_state,
        "worker_present": worker is not None,
        "worker_is_alive": worker_is_alive,
        "constructor_calls": calls,
    }
    print(f"D1D_WORKER_FACTS={facts}")
    assert target_terminal_state, facts


def test_vector_store_bad_exception_string_start_failure_reaches_terminal_failure(
    tmp_path,
    monkeypatch,
    isolated_vector_init_slot,
):
    original_error = _BadStringVectorFailure("unreadable start failure")
    start_calls = 0
    failures: list[BaseException] = []

    class _BadStartThread:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get("name")

        def start(self):
            nonlocal start_calls
            start_calls += 1
            raise original_error

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    class _ThreadingFacade:
        Event = threading.Event
        Thread = _BadStartThread
        current_thread = staticmethod(threading.current_thread)

    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "bad-string-start-vector"))
    monkeypatch.setenv("VECTOR_COLLECTION_NAME", "bad_string_start_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "bad-string-start-model")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.03")
    monkeypatch.setattr(hybrid_module, "threading", _ThreadingFacade)

    event = None
    worker = None
    try:
        for _ in range(3):
            try:
                hybrid_module._bounded_vector_store_init()
            except BaseException as exc:
                failures.append(exc)
            else:
                failures.append(AssertionError("vector initialization unexpectedly succeeded"))
        with hybrid_module._VECTOR_INIT_LOCK:
            event = hybrid_module._VECTOR_INIT_EVENT
            failure_state = hybrid_module._VECTOR_INIT_FAILURE
            result_state = hybrid_module._VECTOR_INIT_RESULT
            worker = hybrid_module._VECTOR_INIT_WORKER
        if worker is not None:
            worker.join(1.0)
        event_is_set = event is not None and event.is_set()
        worker_is_alive = worker is not None and worker.is_alive()
    finally:
        if event is not None:
            event.set()
        if worker is not None:
            worker.join(1.0)

    observed = [_observed_failure(error) for error in failures]
    messages = [item["message"] for item in observed]
    valid_description = (
        isinstance(failure_state, tuple)
        and len(failure_state) == 2
        and all(type(part) is str and len(part) <= 500 for part in failure_state)
    )
    target_terminal_state = (
        start_calls == 1
        and len(failures) == 3
        and all(type(error) is RuntimeError for error in failures)
        and len({id(error) for error in failures}) == 3
        and not any(isinstance(error, TimeoutError) for error in failures)
        and not any(isinstance(error, _SecondaryStringificationFailure) for error in failures)
        and all("Vector store initialization failed" in message for message in messages)
        and all("secondary stringification failure" not in message for message in messages)
        and event_is_set
        and valid_description
        and result_state is None
        and worker is not None
        and not worker_is_alive
        and all(error is not original_error for error in failures)
        and all(error.__context__ is None and error.__cause__ is None for error in failures)
    )
    facts = {
        "observed": observed,
        "event_is_set": event_is_set,
        "failure_state": failure_state,
        "result_state": result_state,
        "worker_present": worker is not None,
        "worker_is_alive": worker_is_alive,
        "start_calls": start_calls,
    }
    print(f"D1D_START_FACTS={facts}")
    assert target_terminal_state, facts


def test_vector_store_config_switch_waits_for_old_worker(tmp_path, monkeypatch):
    first_release = threading.Event()
    first_started = threading.Event()
    workers: list[threading.Thread] = []
    started_dirs: list[str] = []
    first_dir = str((tmp_path / "first-vector").resolve())
    second_dir = str((tmp_path / "second-vector").resolve())

    class _SwitchingVectorStore:
        def __init__(self, **kwargs):
            persist_dir = str(kwargs["persist_dir"])
            started_dirs.append(persist_dir)
            workers.append(threading.current_thread())
            if persist_dir == first_dir:
                first_started.set()
                first_release.wait(1.0)

    monkeypatch.setenv("VECTOR_STORE_DIR", first_dir)
    monkeypatch.setenv("VECTOR_COLLECTION_NAME", "switch_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "switch-model")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.02")
    monkeypatch.setattr(hybrid_module, "ChromaVectorStore", _SwitchingVectorStore)

    try:
        with pytest.raises(TimeoutError):
            hybrid_module._bounded_vector_store_init()
        assert first_started.wait(0.5)

        monkeypatch.setenv("VECTOR_STORE_DIR", second_dir)
        with pytest.raises(TimeoutError):
            hybrid_module._bounded_vector_store_init()
        assert started_dirs == [first_dir]

        first_release.set()
        workers[0].join(1.0)
        assert not workers[0].is_alive()
        monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.2")
        switched = hybrid_module._bounded_vector_store_init()
    finally:
        first_release.set()
        for worker in workers:
            worker.join(1.0)

    assert switched is not None
    assert started_dirs == [first_dir, second_dir]
    assert all(not worker.is_alive() for worker in workers)


def test_deepseek_retries_stay_within_agent_deadline(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "0.1")
    completions = _SlowFailCompletions()
    provider = DeepSeekProvider(client=_FakeClient(completions))

    started = time.perf_counter()
    with pytest.raises(LLMProviderError):
        provider.generate_answer(LLMRequest(question="offline test"))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.3
    assert completions.calls <= 2


def test_http_chat_cold_start_returns_bounded_offline_result(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setenv("EMBEDDING_OFFLINE", "true")
    monkeypatch.setenv("EMBEDDING_MODEL", "missing-offline-model")
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "http-cold-start-vector"))
    monkeypatch.setenv("RAG_VECTOR_INIT_TIMEOUT_SECONDS", "0.02")
    monkeypatch.setenv("JWT_SECRET_KEY", "f1-3-http-test-jwt-secret-32-bytes")

    original_logger = main.learning_logger
    main.learning_logger = main.LearningLogger(tmp_path / "http-cold-start.db")
    main.learning_logger.seed_demo_identity()
    course_id = main.learning_logger.default_course_id()
    class_id = main.learning_logger.default_class_id()
    course_document = Document(
        source_path=tmp_path / "nas-offline.md",
        filename="nas-offline.md",
        file_type=".md",
        title="NAS offline evidence",
        text="5G NAS authentication offline cold start evidence",
        pages=[Page(text="5G NAS authentication offline cold start evidence", page=1, section="NAS", source_type="markdown")],
    )
    saved = main.learning_logger.save_document(
        document=course_document,
        content_hash=sha256_text(course_document.text),
        chunks=[course_document.text],
        byte_size=len(course_document.text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
        class_id=class_id,
    )
    main.learning_logger.set_document_publication(int(saved["id"]), published=True, actor_id=1)
    monkeypatch.setenv("EXTERNAL_MODEL_COURSE_ALLOWLIST", str(course_id))
    try:
        with TestClient(main.app) as client:
            login = client.post(
                "/api/auth/login",
                json={"username": "student_demo", "password": "demo123456"},
            )
            assert login.status_code == 200
            payload = login.json()
            token = payload["access_token"]
            if payload.get("must_change_password"):
                changed = client.post(
                    "/api/auth/change-password",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "current_password": "demo123456",
                        "new_password": "f1-3-http-student-2026",
                    },
                )
                assert changed.status_code == 200
                token = changed.json()["access_token"]

            started = time.perf_counter()
            response = client.post(
                "/api/chat",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "question": "5G NAS authentication offline cold start",
                    "course_id": course_id,
                    "class_id": class_id,
                    "use_course_references": True,
                },
            )
            elapsed = time.perf_counter() - started

        assert response.status_code == 200
        body = response.json()
        assert body["run_status"] in {"COMPLETED", "ABSTAINED", "LIMIT_REACHED"}
        assert isinstance(body["fallback_used"], bool)
        # Includes the first-process jieba/Chroma initialization while retaining
        # the original cold-start budget expected by this offline test.
        assert elapsed < 3.0
    finally:
        main.learning_logger = original_logger
