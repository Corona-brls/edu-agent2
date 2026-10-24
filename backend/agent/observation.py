"""Unified observations passed back to the agent planner."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mcp_server.schemas import ToolResult


SECRET_KEY_PARTS = ("api_key", "apikey", "token", "secret", "password", "authorization")
SECRET_VALUE_RE = re.compile(
    r"(?i)(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,}|deepseek[_\-]?[A-Za-z0-9_\-]{8,})"
)
MAX_CONTENT_CHARS = 4000


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_number: int = Field(..., ge=1)
    action: str
    tool_name: str | None = None
    success: bool
    content: str = ""
    error_code: str | None = None
    summary: str
    latency_ms: float = Field(default=0, ge=0)
    fallback_used: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


def sanitize_payload(value: Any) -> Any:
    """Mask secrets before observations or tool-call records are stored."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(part in key_text for part in SECRET_KEY_PARTS):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = sanitize_payload(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, str):
        return SECRET_VALUE_RE.sub("***REDACTED***", value)
    return value


def tool_result_to_observation(
    result: ToolResult | dict[str, Any],
    *,
    step_number: int,
    tool_name: str,
    action: str = "call_tool",
    fallback_used: bool = False,
    run_id: str | None = None,
    step_id: str | None = None,
    arguments: dict[str, Any] | None = None,
    latency_ms: float | None = None,
) -> Observation:
    if isinstance(result, dict):
        result = ToolResult.model_validate(result)
    safe_data = sanitize_payload(result.data)
    safe_metadata = sanitize_payload(result.metadata)
    content = _to_text(safe_data if result.success else {"error": result.error_message})
    summary = (
        f"{tool_name} returned data as untrusted observation"
        if result.success
        else f"{tool_name} failed with {result.error_code or 'UNKNOWN_ERROR'}"
    )
    observed_latency = latency_ms
    if observed_latency is None:
        observed_latency = float(safe_metadata.get("latency_ms") or 0)
    return Observation(
        step_number=step_number,
        action=action,
        tool_name=tool_name,
        success=result.success,
        content=content,
        error_code=result.error_code,
        summary=summary,
        latency_ms=round(float(observed_latency), 2),
        fallback_used=fallback_used,
        metadata={
            "run_id": run_id,
            "step_id": step_id,
            "arguments": sanitize_payload(arguments or {}),
            "tool_metadata": safe_metadata,
            "content_trust": "untrusted",
        },
    )


def retrieval_results_to_observation(
    results: Any,
    *,
    step_number: int,
    fallback_used: bool = False,
    run_id: str | None = None,
    step_id: str | None = None,
    latency_ms: float = 0,
) -> Observation:
    safe_results = sanitize_payload(_model_dump(results))
    count = _result_count(safe_results)
    return Observation(
        step_number=step_number,
        action="retrieve",
        tool_name="hybrid_retriever",
        success=True,
        content=_to_text(safe_results),
        error_code=None,
        summary=f"hybrid_retriever returned {count} untrusted retrieval results",
        latency_ms=round(float(latency_ms), 2),
        fallback_used=fallback_used,
        metadata={
            "run_id": run_id,
            "step_id": step_id,
            "content_trust": "untrusted",
            "result_count": count,
        },
    )


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_model_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _model_dump(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _model_dump(getattr(value, key))
            for key in value.__dataclass_fields__
        }
    return value


def _result_count(value: Any) -> int:
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        return len(value["results"])
    if isinstance(value, list):
        return len(value)
    return 1 if value else 0


def _to_text(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= MAX_CONTENT_CHARS:
        return text
    return text[:MAX_CONTENT_CHARS] + "...[truncated]"
