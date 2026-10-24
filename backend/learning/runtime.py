"""Small process-local runtime for model-backed learning requests."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time
from typing import Callable, TypeVar

from backend.learning.errors import GeneralLearningError


T = TypeVar("T")


class LearningRuntime:
    """Bound requests while retaining slots until real workers finish."""

    def __init__(
        self,
        *,
        max_process_requests: int = 2,
        max_user_requests: int = 1,
        rate_limit: int = 6,
        rate_window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.max_process_requests = int(max_process_requests)
        self.max_user_requests = int(max_user_requests)
        self.rate_limit = int(rate_limit)
        self.rate_window_seconds = float(rate_window_seconds)
        self.clock = clock
        self.executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="learning-model")
        self._owns_executor = executor is None
        self._lock = threading.Lock()
        self._active_total = 0
        self._active_users: dict[int, int] = defaultdict(int)
        self._rate: dict[int, deque[float]] = defaultdict(deque)
        self._closed = False

    def begin(self, user_id: int) -> "RuntimeLease":
        now = self.clock()
        uid = int(user_id)
        with self._lock:
            if self._closed:
                raise GeneralLearningError("MODEL_UNAVAILABLE", "模型服务正在关闭。", status_code=503, retryable=True)
            bucket = self._rate[uid]
            cutoff = now - self.rate_window_seconds
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.rate_limit:
                retry_after = max(1, int(self.rate_window_seconds - (now - bucket[0]) + 0.999))
                raise GeneralLearningError(
                    "RATE_LIMITED",
                    "请求过于频繁，请稍后手动重试。",
                    status_code=429,
                    retryable=True,
                    retry_after=retry_after,
                )
            if self._active_users[uid] >= self.max_user_requests or self._active_total >= self.max_process_requests:
                raise GeneralLearningError(
                    "MODEL_BUSY",
                    "模型服务繁忙，请稍后手动重试。",
                    status_code=429,
                    retryable=True,
                    retry_after=1,
                )
            bucket.append(now)
            self._active_users[uid] += 1
            self._active_total += 1
        return RuntimeLease(runtime=self, user_id=uid)

    def _release(self, user_id: int) -> None:
        with self._lock:
            if self._active_users.get(user_id, 0) > 0:
                self._active_users[user_id] -= 1
                if self._active_users[user_id] == 0:
                    self._active_users.pop(user_id, None)
                self._active_total -= 1

    def shutdown(self, *, wait: bool = False) -> None:
        with self._lock:
            self._closed = True
        if self._owns_executor:
            self.executor.shutdown(wait=wait, cancel_futures=True)


@dataclass
class RuntimeLease:
    runtime: LearningRuntime
    user_id: int
    response_closed: bool = False
    active_work_count: int = 0
    released: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    async def run(
        self,
        function: Callable[[], T],
        *,
        deadline: float,
        disconnected: threading.Event | None = None,
    ) -> T:
        if deadline <= self.runtime.clock():
            raise GeneralLearningError("MODEL_TIMEOUT", "请求处理超时。", status_code=504, retryable=True)
        with self._lock:
            if self.response_closed:
                raise GeneralLearningError("CLIENT_DISCONNECTED", "客户端已断开连接。", status_code=409)
            self.active_work_count += 1
        future: Future[T] = self.runtime.executor.submit(function)
        future.add_done_callback(lambda _future: self._work_done())
        wrapped = asyncio.wrap_future(future)
        while True:
            if disconnected is not None and disconnected.is_set():
                wrapped.add_done_callback(self._consume_late_result)
                raise GeneralLearningError(
                    "CLIENT_DISCONNECTED",
                    "客户端已断开，本次结果未保存。",
                    status_code=409,
                )
            remaining = deadline - self.runtime.clock()
            if remaining <= 0:
                wrapped.add_done_callback(self._consume_late_result)
                raise GeneralLearningError(
                    "MODEL_TIMEOUT",
                    "模型响应超时，请稍后再试。",
                    status_code=504,
                    retryable=True,
                )
            done, _pending = await asyncio.wait({wrapped}, timeout=min(0.05, remaining))
            if done:
                return wrapped.result()

    def close(self) -> None:
        release = False
        with self._lock:
            self.response_closed = True
            if self.active_work_count == 0 and not self.released:
                self.released = True
                release = True
        if release:
            self.runtime._release(self.user_id)

    def _work_done(self) -> None:
        release = False
        with self._lock:
            self.active_work_count = max(0, self.active_work_count - 1)
            if self.response_closed and self.active_work_count == 0 and not self.released:
                self.released = True
                release = True
        if release:
            self.runtime._release(self.user_id)

    @staticmethod
    def _consume_late_result(future: asyncio.Future) -> None:
        """Observe a late worker error without cancelling the still-running worker."""

        try:
            future.exception()
        except (asyncio.CancelledError, Exception):
            pass

