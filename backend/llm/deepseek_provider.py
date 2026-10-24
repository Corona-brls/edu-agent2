"""DeepSeek provider using the OpenAI-compatible chat API."""

from __future__ import annotations

import threading
import time
import json
from typing import Any, Callable

from pydantic import ValidationError

from backend.agent.schemas import ActionDecision, ToolSpec
from backend.core.config import Settings, get_settings
from backend.core.errors import LLMConfigurationError, LLMEmptyResponseError, LLMProviderError, LLMResponseError, classify_llm_failure
from backend.learning.context import CourseReference
from backend.learning.errors import GeneralLearningError
from backend.learning.schemas import GeneralModelAnswer
from backend.llm.general_prompts import build_general_messages
from backend.llm.practice_draft_prompts import build_practice_draft_messages
from backend.llm.innovation_prompts import build_innovation_suggestion_messages
from backend.llm.base import LLMProvider, StructuredModel
from backend.llm.prompts import (
    build_action_decision_messages,
    build_answer_messages,
    build_structured_messages,
    messages_sha256,
)
from backend.llm.schemas import GroundedPracticeGuide, LLMRequest, LLMResponse
from backend.llm.data_policy import ExternalModelDataPolicy, ModelCallPreflightError, PolicyStatus
from backend.practice.schemas import PracticeDraftCandidate, PracticeDraftRequest
from backend.innovation.schemas import InnovationSuggestionCandidate


class DeepSeekProvider(LLMProvider):
    provider_name = "deepseek"

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.deepseek_model
        self.last_request_sha256: str | None = None
        self.data_policy = ExternalModelDataPolicy.from_ids(self.settings.external_model_course_allowlist)
        self.course_reference_status: PolicyStatus = "blocked_by_policy"
        self._outbound_guard: Callable[[str], None] | None = None
        if not self.settings.deepseek_api_key and client is None:
            raise LLMConfigurationError("DeepSeek API Key 未配置")
        self.client = client or self._create_client()

    def generate_answer(self, request: LLMRequest) -> LLMResponse:
        projected = self.data_policy.project_request(request, status=self.course_reference_status)
        content = self._chat(build_answer_messages(projected))
        return LLMResponse(
            text=content,
            provider=self.provider_name,
            model=self.model_name,
            fallback_used=False,
        )

    def generate_structured(
        self,
        request: LLMRequest,
        schema: type[StructuredModel],
    ) -> StructuredModel:
        projected = self.data_policy.project_request(request, status=self.course_reference_status)
        messages = build_structured_messages(
            projected,
            schema.__name__,
            schema.model_json_schema(),
        )
        practice_shape = schema is GroundedPracticeGuide
        content = self._chat(
            messages,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}} if practice_shape else None,
            sdk_max_retries=0 if practice_shape else None,
        )
        try:
            return schema.model_validate_json(_json_payload(content))
        except (ValidationError, ValueError) as exc:
            raise LLMResponseError("DeepSeek 返回的结构化 JSON 无法通过校验") from exc

    def generate_general_answer(
        self,
        question: str,
        references: list[CourseReference],
        *,
        deadline: float,
        outbound_guard: Callable[[str], None] | None = None,
    ) -> GeneralModelAnswer:
        """Generate one general-learning result without the legacy detached timeout thread."""

        messages = self.data_policy.project_messages(build_general_messages(question, references))
        self.last_request_sha256 = messages_sha256(messages)
        last_error: BaseException | None = None
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("DeepSeek general request exceeded its deadline")
            try:
                self._call_outbound_guard(outbound_guard, "before_create")
                request: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.settings.llm_temperature,
                    "max_tokens": self.settings.llm_max_output_tokens,
                    "response_format": {"type": "json_object"},
                    "timeout": max(0.05, min(float(self.settings.llm_timeout_seconds), remaining)),
                }
                client = self.client
                with_options = getattr(client, "with_options", None)
                if callable(with_options):
                    client = with_options(
                        timeout=request["timeout"],
                        max_retries=0,
                    )
                response = client.chat.completions.create(**request)
                self._call_outbound_guard(outbound_guard, "after")
                content = _extract_message_content(response)
                if not content:
                    raise LLMEmptyResponseError("DeepSeek 返回内容为空")
                return GeneralModelAnswer.model_validate_json(_json_payload(content))
            except (ModelCallPreflightError, GeneralLearningError):
                raise
            except (ValidationError, ValueError) as exc:
                raise LLMResponseError("DeepSeek 通用问答响应无法通过校验") from exc
            except BaseException as exc:
                last_error = exc
                failure = classify_llm_failure(exc, phase="general_chat")
                if attempt or not failure["retryable"]:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.25:
                    break
                time.sleep(min(0.25, remaining))
        if isinstance(last_error, LLMProviderError):
            raise last_error
        raise LLMProviderError("DeepSeek 通用问答调用失败") from last_error

    def generate_practice_draft(
        self,
        payload: PracticeDraftRequest,
        span_catalog: list[dict[str, str]],
        *,
        deadline: float,
        feedback: dict[str, Any] | None = None,
        outbound_guard: Callable[[str], None] | None = None,
        transport_state: dict[str, bool] | None = None,
    ) -> PracticeDraftCandidate:
        """Generate one draft candidate inside the caller-owned runtime worker."""

        messages = self.data_policy.project_messages(
            build_practice_draft_messages(payload, span_catalog, feedback=feedback)
        )
        self.last_request_sha256 = messages_sha256(messages)
        last_error: BaseException | None = None
        shared_transport_state = transport_state if transport_state is not None else {"recovery_used": False}
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("DeepSeek practice draft request exceeded its deadline")
            try:
                self._call_outbound_guard(outbound_guard, "before_create")
                request: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.settings.llm_temperature,
                    "max_tokens": self.settings.llm_max_output_tokens,
                    "response_format": {"type": "json_object"},
                    "extra_body": {"thinking": {"type": "disabled"}},
                    "timeout": max(0.05, min(float(self.settings.llm_timeout_seconds), remaining)),
                }
                client = self.client
                with_options = getattr(client, "with_options", None)
                if callable(with_options):
                    client = with_options(timeout=request["timeout"], max_retries=0)
                response = client.chat.completions.create(**request)
                self._call_outbound_guard(outbound_guard, "after")
                content = _extract_message_content(response)
                if not content:
                    raise LLMEmptyResponseError("DeepSeek 返回内容为空")
                return PracticeDraftCandidate.model_validate_json(_json_payload(content))
            except (ModelCallPreflightError, GeneralLearningError):
                raise
            except (ValidationError, ValueError) as exc:
                raise LLMResponseError("DeepSeek 探索草案响应无法通过校验") from exc
            except BaseException as exc:
                last_error = exc
                failure = classify_llm_failure(exc, phase="practice_draft")
                if attempt or not failure["retryable"] or shared_transport_state.get("recovery_used", False):
                    break
                shared_transport_state["recovery_used"] = True
                remaining = deadline - time.monotonic()
                if remaining <= 0.25:
                    break
                time.sleep(min(0.25, remaining))
        if isinstance(last_error, LLMProviderError):
            raise last_error
        raise LLMProviderError("DeepSeek 探索草案调用失败") from last_error

    def generate_innovation_suggestions(
        self,
        context: dict[str, Any],
        *,
        deadline: float,
        outbound_guard: Callable[[str], None] | None = None,
    ) -> InnovationSuggestionCandidate:
        """Generate a single strictly validated innovation candidate."""

        messages = self.data_policy.project_messages(build_innovation_suggestion_messages(context))
        self.last_request_sha256 = messages_sha256(messages)
        last_error: BaseException | None = None
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("DeepSeek innovation request exceeded its deadline")
            try:
                self._call_outbound_guard(outbound_guard, "before_create")
                request: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.settings.llm_temperature,
                    "max_tokens": self.settings.llm_max_output_tokens,
                    "response_format": {"type": "json_object"},
                    "extra_body": {"thinking": {"type": "disabled"}},
                    "timeout": max(0.05, min(float(self.settings.llm_timeout_seconds), remaining)),
                }
                client = self.client
                with_options = getattr(client, "with_options", None)
                if callable(with_options):
                    client = with_options(timeout=request["timeout"], max_retries=0)
                response = client.chat.completions.create(**request)
                self._call_outbound_guard(outbound_guard, "after")
                content = _extract_message_content(response)
                if not content:
                    raise LLMEmptyResponseError("DeepSeek 返回内容为空")
                return InnovationSuggestionCandidate.model_validate_json(_json_payload(content))
            except (ModelCallPreflightError, GeneralLearningError):
                raise
            except (ValidationError, ValueError) as exc:
                raise LLMResponseError("DeepSeek 创新建议响应无法通过校验") from exc
            except BaseException as exc:
                last_error = exc
                failure = classify_llm_failure(exc, phase="innovation_suggestion")
                if attempt or not failure["retryable"]:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.25:
                    break
                time.sleep(min(0.25, remaining))
        if isinstance(last_error, LLMProviderError):
            raise last_error
        raise LLMProviderError("DeepSeek 创新建议调用失败") from last_error

    def decide_next_action(
        self,
        state_summary: str,
        available_tools: list[ToolSpec],
    ) -> ActionDecision:
        tool_payload = [tool.model_dump() for tool in available_tools]
        content = self._chat(build_action_decision_messages(state_summary, tool_payload))
        try:
            return ActionDecision.model_validate_json(_json_payload(content))
        except (ValidationError, ValueError) as exc:
            raise LLMResponseError("DeepSeek 返回的 ActionDecision 无法通过校验") from exc

    def configure_course_policy(
        self,
        *,
        course_id: int | None,
        use_course_references: bool,
        currently_authorized: bool,
    ) -> PolicyStatus:
        self.course_reference_status = self.data_policy.status(
            course_id=course_id,
            use_course_references=use_course_references,
            currently_authorized=currently_authorized,
        )
        return self.course_reference_status

    def decide_from_state(self, state: Any, available_tools: list[ToolSpec]) -> ActionDecision:
        return self.decide_next_action(
            self.data_policy.planner_summary(state, status=self.course_reference_status),
            available_tools,
        )

    def set_outbound_guard(self, guard: Callable[[str], None] | None) -> None:
        """Install a request-scoped check around every physical HTTP attempt."""

        self._outbound_guard = guard

    def _create_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            timeout=self.settings.llm_timeout_seconds,
        )

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        sdk_max_retries: int | None = None,
    ) -> str:
        messages = self.data_policy.project_messages(messages)
        self.last_request_sha256 = messages_sha256(messages)
        try:
            return self._chat_with_retry(
                messages,
                response_format=response_format,
                extra_body=extra_body,
                sdk_max_retries=sdk_max_retries,
            )
        except ModelCallPreflightError:
            raise
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError("DeepSeek 调用失败") from exc

    def _chat_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        sdk_max_retries: int | None = None,
    ) -> str:
        attempts = max(1, self.settings.llm_max_retries + 1)
        deadline = time.monotonic() + max(
            0.05,
            float(self.settings.agent_timeout_seconds),
        )
        last_error: Exception | None = None
        for attempt in range(attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            attempt_guard = self._outbound_guard
            try:
                self._call_outbound_guard(attempt_guard, "before")
                once_kwargs: dict[str, Any] = {
                    "response_format": response_format,
                    "outbound_guard": attempt_guard,
                }
                if extra_body is not None:
                    once_kwargs["extra_body"] = extra_body
                if sdk_max_retries is not None:
                    once_kwargs["sdk_max_retries"] = sdk_max_retries
                content = self._run_with_timeout(
                    lambda call_kwargs=dict(once_kwargs): self._chat_once(
                        messages,
                        **call_kwargs,
                    ),
                    timeout_seconds=min(float(self.settings.llm_timeout_seconds), remaining),
                )
                self._call_outbound_guard(attempt_guard, "after")
                return content
            except ModelCallPreflightError:
                raise
            except Exception as exc:
                last_error = exc
            if attempt >= attempts - 1:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.25 * (2**attempt), remaining))
        raise last_error or TimeoutError("DeepSeek request exceeded the agent deadline")

    def _check_outbound_guard(self, stage: str) -> None:
        self._call_outbound_guard(self._outbound_guard, stage)

    @staticmethod
    def _call_outbound_guard(
        guard: Callable[[str], None] | None,
        stage: str,
    ) -> None:
        if guard is not None:
            guard(stage)

    @staticmethod
    def _run_with_timeout(fn, *, timeout_seconds: float) -> str:
        result: dict[str, Any] = {}

        def run() -> None:
            try:
                result["value"] = fn()
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(max(0.01, timeout_seconds))
        if thread.is_alive():
            raise TimeoutError("DeepSeek request timed out")
        if "error" in result:
            raise result["error"]
        return str(result.get("value") or "")

    def _chat_once(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, str] | None = None,
        outbound_guard: Callable[[str], None] | None = None,
        extra_body: dict[str, Any] | None = None,
        sdk_max_retries: int | None = None,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_output_tokens,
        }
        if response_format is not None:
            request["response_format"] = response_format
        if extra_body is not None:
            request["extra_body"] = extra_body
        self._call_outbound_guard(outbound_guard, "before_create")
        client = self.client
        with_options = getattr(client, "with_options", None)
        if sdk_max_retries is not None and callable(with_options):
            client = with_options(max_retries=sdk_max_retries)
        response = client.chat.completions.create(
            **request,
        )
        content = _extract_message_content(response)
        if not content:
            raise LLMEmptyResponseError("DeepSeek 返回内容为空")
        return content


def _extract_message_content(response: Any) -> str:
    if isinstance(response, dict):
        content = response["choices"][0]["message"]["content"]
    else:
        choice = response.choices[0]
        message = choice["message"] if isinstance(choice, dict) else choice.message
        content = message["content"] if isinstance(message, dict) else message.content
    return "" if content is None else str(content)


def _json_payload(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid json") from exc
    return stripped
