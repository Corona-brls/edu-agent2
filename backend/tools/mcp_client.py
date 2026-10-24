"""Reusable MCP client gateway for FastAPI and the agent loop."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, ConfigDict

from backend.agent.observation import Observation, sanitize_payload, tool_result_to_observation
from backend.core.config import get_settings
from backend.tools.tool_policy import (
    call_policy_tool,
    can_agent_execute_tool,
    get_agent_tool_catalog as policy_agent_tool_catalog,
    get_tool_definition,
    list_policy_tools,
)
from mcp_server.schemas import ToolResult


class _MCPInvocationError(RuntimeError):
    def __init__(self, message: str, *, dispatched: bool) -> None:
        super().__init__(message)
        self.dispatched = dispatched


class GatewayToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: ToolResult
    observation: Observation
    fallback_used: bool = False
    run_id: str
    step_id: str


class MCPClientGateway:
    """Owns one reusable MCP session and enforces agent tool policy."""

    def __init__(
        self,
        *,
        max_concurrent_calls: int = 4,
        max_retries: int | None = None,
        acquire_timeout_seconds: float = 1,
        mcp_command: str | None = None,
        mcp_args: list[str] | None = None,
        mcp_cwd: str | Path | None = None,
        mcp_timeout_seconds: float | None = None,
    ) -> None:
        self._server_params: StdioServerParameters | None = None
        self._last_error: str | None = None
        self._started = False
        self._lock = threading.RLock()
        self._semaphore = threading.BoundedSemaphore(max_concurrent_calls)
        self._max_retries = max_retries
        self._acquire_timeout_seconds = acquire_timeout_seconds
        self._mcp_command = mcp_command or sys.executable
        self._mcp_args = list(mcp_args) if mcp_args is not None else ["-m", "mcp_server.server"]
        self._mcp_cwd = str(mcp_cwd or Path(__file__).resolve().parents[2])
        configured_timeout = (
            get_settings().mcp_tool_timeout_seconds
            if mcp_timeout_seconds is None
            else float(mcp_timeout_seconds)
        )
        if configured_timeout <= 0:
            raise ValueError("mcp_timeout_seconds must be positive")
        self._mcp_timeout_seconds = configured_timeout
        self._tool_calls: list[dict[str, Any]] = []
        self.start_count = 0

    @property
    def is_available(self) -> bool:
        return self._started and self._server_params is not None

    def start(self) -> None:
        with self._lock:
            if self._started and self._server_params is not None:
                return
            self._started = True
            self.start_count += 1
        try:
            params = self._make_server_params()
            self._run_async(self._probe_server(params), timeout_seconds=self._mcp_timeout_seconds)
        except Exception as exc:
            with self._lock:
                self._server_params = None
                self._last_error = str(exc) or exc.__class__.__name__
            return
        with self._lock:
            self._server_params = params
            self._last_error = None

    def stop(self) -> None:
        with self._lock:
            self._server_params = None
            self._started = False

    def list_tools(self) -> list[dict[str, Any]]:
        tools = list_policy_tools()
        for tool in tools:
            tool["mcp_available"] = self.is_available
            tool["status"] = "mcp" if self.is_available else "local_fallback"
            if self._last_error and not self.is_available:
                tool["mcp_error"] = self._last_error
        return tools

    def get_agent_tool_catalog(self, role: str = "student") -> list[dict[str, Any]]:
        catalog = policy_agent_tool_catalog(role=role)
        for tool in catalog:
            tool["mcp_available"] = self.is_available
        return catalog

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        run_id: str,
        step_id: str,
        step_number: int = 1,
        role: str = "student",
        remaining_timeout_seconds: float | None = None,
    ) -> GatewayToolCall:
        started = time.perf_counter()
        fallback_used = False
        definition = get_tool_definition(name)
        if definition is None:
            result = ToolResult(
                success=False,
                error_code="TOOL_NOT_FOUND",
                error_message=f"工具不存在：{name}",
                metadata={
                    "tool_name": name,
                    "attempt_count": 0,
                    "outcome": "not_dispatched",
                    "outcome_unknown": False,
                    "retry_suppressed": True,
                },
            )
        elif not can_agent_execute_tool(name, role=role):
            result = ToolResult(
                success=False,
                error_code="TOOL_POLICY_DENIED",
                error_message=f"当前策略不允许自动执行工具：{name}",
                metadata={
                    "tool_name": name,
                    "risk": definition.risk,
                    "network": definition.network,
                    "read_only": definition.read_only,
                    "enabled": definition.enabled,
                    "attempt_count": 0,
                    "outcome": "not_dispatched",
                    "outcome_unknown": False,
                    "retry_suppressed": True,
                },
            )
        else:
            result, fallback_used = self._execute_allowed_tool(
                name,
                arguments,
                remaining_timeout_seconds=remaining_timeout_seconds,
            )

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        observation = tool_result_to_observation(
            result,
            step_number=step_number,
            tool_name=name,
            fallback_used=fallback_used,
            run_id=run_id,
            step_id=step_id,
            arguments=arguments,
            latency_ms=latency_ms,
        )
        record = {
            "run_id": run_id,
            "step_id": step_id,
            "tool_name": name,
            "arguments": sanitize_payload(arguments),
            "result": sanitize_payload(result.model_dump()),
            "fallback_used": fallback_used,
            "latency_ms": latency_ms,
        }
        with self._lock:
            self._tool_calls.append(record)
        return GatewayToolCall(
            result=result,
            observation=observation,
            fallback_used=fallback_used,
            run_id=run_id,
            step_id=step_id,
        )

    def tool_call_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._tool_calls]

    def _execute_allowed_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        remaining_timeout_seconds: float | None = None,
    ) -> tuple[ToolResult, bool]:
        definition = get_tool_definition(name)
        if definition is None:
            return (
                ToolResult(
                    success=False,
                    error_code="TOOL_NOT_FOUND",
                    error_message=f"Tool not found: {name}",
                    metadata={"tool_name": name, "attempt_count": 0, "outcome": "not_dispatched"},
                ),
                False,
            )
        transport_grace = min(2.0, max(0.25, float(definition.timeout_seconds) * 0.2))
        required_budget = float(definition.timeout_seconds) + transport_grace
        if remaining_timeout_seconds is not None and remaining_timeout_seconds <= required_budget:
            return (
                ToolResult(
                    success=False,
                    error_code="AGENT_TOOL_BUDGET_EXHAUSTED",
                    error_message="The remaining agent budget is too short to dispatch this tool safely.",
                    metadata={
                        "tool_name": name,
                        "attempt_count": 0,
                        "outcome": "not_dispatched",
                        "outcome_unknown": False,
                        "retry_suppressed": True,
                        "remaining_timeout_seconds": max(0.0, float(remaining_timeout_seconds)),
                        "required_timeout_seconds": required_budget,
                    },
                ),
                False,
            )
        effective_timeout = max(self._mcp_timeout_seconds, required_budget)
        if remaining_timeout_seconds is not None:
            effective_timeout = min(effective_timeout, float(remaining_timeout_seconds))
        acquired = self._semaphore.acquire(timeout=self._acquire_timeout_seconds)
        if not acquired:
            return (
                ToolResult(
                    success=False,
                    error_code="TOOL_CONCURRENCY_LIMIT",
                    error_message="工具调用并发保护已触发。",
                    metadata={
                        "tool_name": name,
                        "attempt_count": 0,
                        "outcome": "not_dispatched",
                        "outcome_unknown": False,
                        "retry_suppressed": True,
                    },
                ),
                False,
            )
        try:
            if not self.is_available:
                result = call_policy_tool(name, arguments)
                return self._with_attempt_metadata(
                    result,
                    attempt_count=1,
                    timeout_seconds=effective_timeout,
                    outcome="completed",
                    outcome_unknown=False,
                    retry_suppressed=False,
                ), True

            attempts = max(
                1,
                self._max_retries
                if self._max_retries is not None
                else get_settings().llm_max_retries + 1,
            )
            last_error: Exception | None = None
            last_result: ToolResult | None = None
            for attempt_count in range(1, attempts + 1):
                try:
                    if effective_timeout == self._mcp_timeout_seconds:
                        last_result = self._call_mcp_tool_once(name, arguments)
                    else:
                        last_result = self._call_mcp_tool_once(
                            name,
                            arguments,
                            timeout_seconds=effective_timeout,
                        )
                    uncertain = last_result.error_code in {"MCP_CALL_TIMEOUT", "MCP_CALL_FAILED"}
                    dispatch_started = last_result.metadata.get("dispatch_started")
                    if uncertain and definition.network and dispatch_started is False:
                        if attempt_count < attempts:
                            continue
                        return self._with_attempt_metadata(
                            last_result,
                            attempt_count=attempt_count,
                            timeout_seconds=effective_timeout,
                            outcome="not_dispatched",
                            outcome_unknown=False,
                            retry_suppressed=False,
                        ), False
                    if uncertain and definition.network:
                        return self._with_attempt_metadata(
                            last_result,
                            attempt_count=attempt_count,
                            timeout_seconds=effective_timeout,
                            outcome="unknown",
                            outcome_unknown=True,
                            retry_suppressed=True,
                        ), False
                    if uncertain and attempt_count < attempts:
                        continue
                    return self._with_attempt_metadata(
                        last_result,
                        attempt_count=attempt_count,
                        timeout_seconds=effective_timeout,
                        outcome="unknown" if uncertain else "completed",
                        outcome_unknown=uncertain,
                        retry_suppressed=False,
                    ), bool(last_result.data.get("fallback_used"))
                except _MCPInvocationError as exc:
                    last_error = exc
                    if definition.network and exc.dispatched:
                        result = ToolResult(
                            success=False,
                            error_code="MCP_CALL_FAILED",
                            error_message="MCP connection failed after dispatch; the outcome is unknown.",
                            metadata={
                                "tool_name": name,
                                "error_type": exc.__cause__.__class__.__name__ if exc.__cause__ else exc.__class__.__name__,
                            },
                        )
                        return self._with_attempt_metadata(
                            result,
                            attempt_count=attempt_count,
                            timeout_seconds=effective_timeout,
                            outcome="unknown",
                            outcome_unknown=True,
                            retry_suppressed=True,
                        ), False
                except Exception as exc:
                    last_error = exc
                    if definition.network:
                        result = ToolResult(
                            success=False,
                            error_code="MCP_CALL_FAILED",
                            error_message="MCP connection failed; the remote outcome cannot be confirmed.",
                            metadata={"tool_name": name, "error_type": exc.__class__.__name__},
                        )
                        return self._with_attempt_metadata(
                            result,
                            attempt_count=attempt_count,
                            timeout_seconds=effective_timeout,
                            outcome="unknown",
                            outcome_unknown=True,
                            retry_suppressed=True,
                        ), False
            if last_result is not None:
                return self._with_attempt_metadata(
                    last_result,
                    attempt_count=attempts,
                    timeout_seconds=effective_timeout,
                    outcome="unknown",
                    outcome_unknown=True,
                    retry_suppressed=False,
                ), False
            result = ToolResult(
                    success=False,
                    error_code="MCP_CALL_FAILED",
                    error_message="MCP 工具调用失败。",
                    metadata={"tool_name": name, "error": str(last_error) if last_error else None},
            )
            return self._with_attempt_metadata(
                result,
                attempt_count=attempts,
                timeout_seconds=effective_timeout,
                outcome="not_dispatched",
                outcome_unknown=False,
                retry_suppressed=False,
            ), False
        finally:
            self._semaphore.release()

    def _call_mcp_tool_once(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        if not self.is_available:
            raise RuntimeError("MCP session is not available")
        timeout = timeout_seconds or self._mcp_timeout_seconds
        dispatch_state = {"dispatched": False}
        try:
            return self._run_async(
                self._call_mcp_tool_async(
                    name,
                    arguments,
                    timeout_seconds=timeout,
                    dispatch_state=dispatch_state,
                ),
                timeout_seconds=timeout,
            )
        except TimeoutError:
            return ToolResult(
                success=False,
                error_code="MCP_CALL_TIMEOUT",
                error_message="MCP 工具调用超时。",
                metadata={
                    "tool_name": name,
                    "timeout_seconds": timeout,
                    "dispatch_started": bool(dispatch_state["dispatched"]),
                },
            )

    def _make_server_params(self) -> StdioServerParameters:
        env = dict(os.environ)
        configured_db = env.get("SQLITE_DB_PATH", "").strip()
        if configured_db:
            env["SQLITE_DB_PATH"] = str(Path(configured_db).expanduser().resolve())
        return StdioServerParameters(
            command=self._mcp_command,
            args=self._mcp_args,
            env=env,
            cwd=self._mcp_cwd,
            encoding="utf-8",
            encoding_error_handler="replace",
        )

    async def _probe_server(self, params: StdioServerParameters) -> None:
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=self._mcp_timeout_seconds,
            ) as session:
                await session.initialize()
                await session.list_tools()

    async def _call_mcp_tool_async(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float,
        dispatch_state: dict[str, bool],
    ) -> ToolResult:
        params = self._server_params
        if params is None:
            raise RuntimeError("MCP session is not available")
        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timeout_seconds,
                ) as session:
                    await session.initialize()
                    dispatch_state["dispatched"] = True
                    result = await session.call_tool(
                        name,
                        arguments,
                        read_timeout_seconds=timeout_seconds,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _MCPInvocationError(
                "MCP stdio invocation failed",
                dispatched=bool(dispatch_state["dispatched"]),
            ) from exc
        is_error = bool(getattr(result, "is_error", False))
        content = getattr(result, "structured_content", None)
        if isinstance(content, dict):
            parsed = ToolResult.model_validate(content)
            metadata = {**parsed.metadata, "protocol_is_error": is_error}
            if is_error and parsed.success:
                return parsed.model_copy(
                    update={
                        "success": False,
                        "error_code": "MCP_PROTOCOL_MISMATCH",
                        "error_message": "MCP protocol marked a successful payload as an error.",
                        "metadata": metadata,
                    }
                )
            return parsed.model_copy(update={"metadata": metadata})
        return ToolResult(
            success=not is_error,
            data={"content": [item.model_dump(mode="json") for item in getattr(result, "content", [])]},
            error_code="MCP_TOOL_ERROR" if is_error else None,
            error_message="MCP 工具返回错误。" if is_error else None,
            metadata={"tool_name": name},
        )

    @staticmethod
    def _with_attempt_metadata(
        result: ToolResult,
        *,
        attempt_count: int,
        timeout_seconds: float,
        outcome: str,
        outcome_unknown: bool,
        retry_suppressed: bool,
    ) -> ToolResult:
        return result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "attempt_count": attempt_count,
                    "timeout_seconds": timeout_seconds,
                    "outcome": outcome,
                    "outcome_unknown": outcome_unknown,
                    "retry_suppressed": retry_suppressed,
                }
            }
        )

    def _run_async(self, coro, *, timeout_seconds: float):
        async def runner():
            return await asyncio.wait_for(coro, timeout=timeout_seconds)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(runner())

        result: dict[str, Any] = {}

        def run_in_thread() -> None:
            try:
                result["value"] = asyncio.run(runner())
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        thread.join(timeout_seconds + 1)
        if thread.is_alive():
            raise TimeoutError("MCP operation timed out")
        if "error" in result:
            raise result["error"]
        return result.get("value")
