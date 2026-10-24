"""Safe domain errors shared by the new model-backed learning features."""

from __future__ import annotations


class GeneralLearningError(RuntimeError):
    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        status_code: int,
        retryable: bool = False,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = int(status_code)
        self.retryable = bool(retryable)
        self.retry_after = int(retry_after) if retry_after is not None else None

