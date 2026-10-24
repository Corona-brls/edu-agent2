"""Application service for independent, model-generated learning answers."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from backend.core.errors import LLMConfigurationError, LLMProviderError, LLMResponseError, classify_llm_failure
from backend.learning.context import (
    CourseReference,
    CourseReferenceError,
    load_course_references,
    revalidate_course_references,
)
from backend.learning.errors import GeneralLearningError
from backend.learning.runtime import LearningRuntime
from backend.learning.schemas import GeneralChatRequest
from backend.llm.data_policy import ExternalModelDataPolicy, ModelCallPreflightError


GENERAL_REQUEST_SECONDS = 90.0
GENERAL_REFERENCE_SECONDS = 25.0


class GeneralLearningService:
    def __init__(
        self,
        *,
        storage: Any,
        retriever: Any,
        provider_factory: Callable[[], Any],
        data_policy: ExternalModelDataPolicy,
        runtime: LearningRuntime,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.storage = storage
        self.retriever = retriever
        self.provider_factory = provider_factory
        self.data_policy = data_policy
        self.runtime = runtime
        self.clock = clock

    async def answer(
        self,
        *,
        user: dict[str, Any],
        scope: dict[str, int | None],
        payload: GeneralChatRequest,
        request_id: str,
        context_validator: Callable[[], None],
        session_validator: Callable[[], None],
        event_recorder: Callable[[dict[str, Any]], None] | None = None,
        disconnected: threading.Event | None = None,
    ) -> dict[str, Any]:
        course_id = int(scope["course_id"] or 0)
        if course_id <= 0:
            raise GeneralLearningError("COURSE_REQUIRED", "请选择课程后再提问。", status_code=400)
        deadline = self.clock() + GENERAL_REQUEST_SECONDS
        lease = self.runtime.begin(int(user["id"]))
        references: list[CourseReference] = []
        warnings: list[str] = []
        reference_status = "disabled"
        try:
            self._check_open(disconnected)
            context_validator()
            session_validator()
            policy_status = self.data_policy.status(
                course_id=course_id,
                use_course_references=payload.use_course_references,
                currently_authorized=True,
            )
            if payload.use_course_references and policy_status == "blocked_by_policy":
                reference_status = "blocked_by_policy"
            elif payload.use_course_references and policy_status == "allowed":
                reference_deadline = min(deadline, self.clock() + GENERAL_REFERENCE_SECONDS)
                try:
                    references = await lease.run(
                        lambda: load_course_references(
                            self.storage,
                            self.retriever,
                            course_id=course_id,
                            query=payload.question,
                            top_k=4,
                            max_chars=4000,
                        ),
                        deadline=reference_deadline,
                        disconnected=disconnected,
                    )
                except CourseReferenceError as exc:
                    if exc.error_code == "REFERENCE_SCOPE_VIOLATION":
                        raise GeneralLearningError(
                            "REFERENCE_SCOPE_VIOLATION",
                            "课程参考作用域校验失败。",
                            status_code=409,
                        ) from exc
                    reference_status = "unavailable"
                    references = []
                except GeneralLearningError:
                    raise
                except Exception:
                    reference_status = "unavailable"
                    references = []
                else:
                    reference_status = "used" if references else "none"
            elif payload.use_course_references:
                reference_status = "unavailable"

            self._check_open(disconnected)
            context_validator()
            session_validator()
            if references:
                references = self._revalidate_references(references, course_id=course_id)

            try:
                provider = self.provider_factory()
            except LLMConfigurationError as exc:
                raise GeneralLearningError(
                    "MODEL_UNAVAILABLE",
                    "通用模型尚未配置。",
                    status_code=503,
                    retryable=False,
                ) from exc
            if str(getattr(provider, "provider_name", "")) != "deepseek":
                raise GeneralLearningError("MODEL_UNAVAILABLE", "通用模型尚未配置。", status_code=503)

            def outbound_guard(_stage: str) -> None:
                self._check_open(disconnected)
                context_validator()
                session_validator()
                current_status = self.data_policy.status(
                    course_id=course_id,
                    use_course_references=payload.use_course_references,
                    currently_authorized=True,
                )
                if references and current_status != "allowed":
                    raise ModelCallPreflightError(
                        "REFERENCE_CONTEXT_CHANGED",
                        "课程参考授权已变化。",
                        status_code=409,
                    )
                if references:
                    try:
                        revalidate_course_references(self.storage, references, course_id=course_id)
                    except CourseReferenceError as exc:
                        raise ModelCallPreflightError(
                            "REFERENCE_CONTEXT_CHANGED",
                            "课程参考状态已变化。",
                            status_code=409,
                        ) from exc

            try:
                model_answer = await lease.run(
                    lambda: provider.generate_general_answer(
                        payload.question,
                        references,
                        deadline=deadline,
                        outbound_guard=outbound_guard,
                    ),
                    deadline=deadline,
                    disconnected=disconnected,
                )
            except GeneralLearningError:
                raise
            except ModelCallPreflightError as exc:
                raise GeneralLearningError(
                    exc.error_code,
                    "课程或学习会话状态已变化，本次结果未保存。",
                    status_code=exc.status_code,
                ) from exc
            except CourseReferenceError as exc:
                raise GeneralLearningError(
                    "REFERENCE_CONTEXT_CHANGED",
                    "课程参考状态已变化，本次结果未保存。",
                    status_code=409,
                ) from exc
            except LLMResponseError as exc:
                raise GeneralLearningError("MODEL_FAILED", "模型响应格式无效。", status_code=502) from exc
            except LLMProviderError as exc:
                failure = classify_llm_failure(exc, phase="general_chat")
                if failure["failure_category"] == "timeout":
                    raise GeneralLearningError("MODEL_TIMEOUT", "模型响应超时，请稍后再试。", status_code=504, retryable=True) from exc
                raise GeneralLearningError(
                    "MODEL_FAILED",
                    "模型服务暂不可用。",
                    status_code=503,
                    retryable=bool(failure["retryable"]),
                ) from exc

            self._check_open(disconnected)
            context_validator()
            session_validator()
            if references:
                references = self._revalidate_references(references, course_id=course_id)

            if event_recorder is not None:
                try:
                    event_recorder(
                        {
                            "request_id": request_id,
                            "mode": "general_chat",
                            "status": model_answer.status,
                            "reference_status": reference_status,
                            "reference_count": len(references),
                        }
                    )
                except GeneralLearningError:
                    raise
                except Exception:
                    warnings.append("学习事件暂未记录，不影响本次回答。")

            return {
                "request_id": request_id,
                "course_id": course_id,
                "class_id": scope.get("class_id"),
                "status": model_answer.status,
                "answer_kind": "model_generated",
                "answer": model_answer.answer,
                "provider": "deepseek",
                "model": str(getattr(provider, "model_name", "")),
                "fallback_used": False,
                "reference_status": reference_status,
                "reference_materials": [
                    {
                        "document_id": item.document_id,
                        "chunk_id": item.chunk_id,
                        "title": item.title,
                        "excerpt": item.text[:300],
                        "usage": "context_only",
                    }
                    for item in references
                ],
                "limitations": [
                    "回答由外部模型生成，可能存在错误。",
                    "课程参考仅作上下文，未作为逐句事实引用。",
                ],
                "warnings": warnings,
            }
        finally:
            lease.close()

    def _revalidate_references(
        self,
        references: list[CourseReference],
        *,
        course_id: int,
    ) -> list[CourseReference]:
        try:
            return revalidate_course_references(self.storage, references, course_id=course_id)
        except CourseReferenceError as exc:
            raise GeneralLearningError(
                "REFERENCE_CONTEXT_CHANGED",
                "课程参考状态已变化，本次结果未保存。",
                status_code=409,
            ) from exc

    @staticmethod
    def _check_open(disconnected: threading.Event | None) -> None:
        if disconnected is not None and disconnected.is_set():
            raise GeneralLearningError(
                "CLIENT_DISCONNECTED",
                "客户端已断开，本次结果未保存。",
                status_code=409,
            )
