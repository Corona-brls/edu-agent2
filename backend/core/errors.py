"""Domain errors that are safe to surface without leaking secrets."""

from __future__ import annotations

from typing import Any


class CourseAgentError(RuntimeError):
    error_code = "COURSE_AGENT_ERROR"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        if error_code:
            self.error_code = error_code


class LLMProviderError(CourseAgentError):
    error_code = "LLM_PROVIDER_ERROR"


class LLMConfigurationError(LLMProviderError):
    error_code = "LLM_CONFIGURATION_ERROR"


class LLMResponseError(LLMProviderError):
    error_code = "LLM_RESPONSE_ERROR"


class LLMEmptyResponseError(LLMResponseError):
    """The provider returned no final answer content; candidate correction cannot fix it."""


LLM_FAILURE_CATEGORIES = {
    "transport",
    "timeout",
    "authentication",
    "rate_limit",
    "api_status",
    "empty_response",
    "response_parse",
    "schema_validation",
    "unknown",
}


def classify_llm_failure(exc: BaseException, *, phase: str) -> dict[str, Any]:
    """Return sanitized audit metadata for an LLM failure.

    The exception chain is inspected only in memory. The returned payload is a
    fixed whitelist and intentionally excludes messages, prompts, response
    bodies, headers, stack traces, and other potentially sensitive material.
    """

    chain = _exception_chain(exc)
    root = chain[-1] if chain else exc
    http_status = _http_status_from_chain(chain)
    category = _failure_category(chain, http_status)
    return {
        "failure_category": category,
        "root_error_type": _safe_error_type(root),
        "http_status": http_status,
        "retryable": _is_retryable(category, http_status),
        "phase": str(phase)[:80],
    }


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 12:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _safe_error_type(exc: BaseException) -> str:
    name = exc.__class__.__name__ or "UnknownError"
    safe = "".join(ch for ch in name if ch.isalnum() or ch == "_").strip("_")
    return safe[:80] or "UnknownError"


def _http_status_from_chain(chain: list[BaseException]) -> int | None:
    for item in chain:
        status = _http_status(item)
        if status is not None:
            return status
    return None


def _http_status(exc: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    for attr in ("status_code", "status"):
        value = getattr(response, attr, None)
        if isinstance(value, int):
            return value
    return None


def _failure_category(chain: list[BaseException], http_status: int | None) -> str:
    if http_status in {401, 403}:
        return "authentication"
    if http_status == 429:
        return "rate_limit"
    if http_status is not None and http_status >= 400:
        return "api_status"

    names = " ".join(_safe_error_type(item).lower() for item in chain)
    messages = " ".join(str(item).lower() for item in chain)

    if "timeout" in names or "timeout" in messages or "timed out" in messages:
        return "timeout"
    if any(token in names for token in ("connect", "connection", "ssl", "transport", "network")):
        return "transport"
    if "jsondecodeerror" in names or (
        any(isinstance(item, ValueError) for item in chain)
        and any(token in messages for token in ("json", "parse", "actiondecision"))
    ):
        return "response_parse"
    if any(token in names for token in ("validationerror", "schema")):
        return "schema_validation"
    if any(token in messages for token in ("empty response", "empty content", "content empty", "为空", "空")):
        return "empty_response"
    return "unknown"


def _is_retryable(category: str, http_status: int | None) -> bool:
    if category in {"transport", "timeout", "rate_limit"}:
        return True
    return category == "api_status" and http_status is not None and http_status >= 500
