"""Application service for evidence-constrained exploratory practice drafts."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any, Callable

from pydantic import ValidationError

from backend.connectors.standard_source import normalize_standard_resource_view
from backend.connectors.schemas import ResourceQuery
from backend.core.errors import LLMConfigurationError, LLMEmptyResponseError, LLMProviderError, LLMResponseError, classify_llm_failure
from backend.learning.context import (
    CourseReference,
    CourseReferenceError,
    load_course_references,
)
from backend.learning.errors import GeneralLearningError
from backend.learning.runtime import LearningRuntime
from backend.llm.data_policy import ExternalModelDataPolicy, ModelCallPreflightError
from backend.practice.evidence import (
    DraftEvidenceCatalog,
    DraftEvidenceError,
    build_draft_evidence,
    render_background_quotes,
    validate_draft_candidate,
)
from backend.practice.schemas import DraftReferenceStatus, PracticeDraftCandidate, PracticeDraftRequest
from backend.rag.visibility import is_student_visible_document


PRACTICE_DRAFT_REQUEST_SECONDS = 90.0
PRACTICE_DRAFT_REFERENCE_SECONDS = 25.0


class ExploratoryPracticeService:
    def __init__(
        self,
        *,
        storage: Any,
        retriever: Any,
        resource_service: Any,
        provider_factory: Callable[[], Any],
        data_policy: ExternalModelDataPolicy,
        runtime: LearningRuntime,
        remote_rate_limiter: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.storage = storage
        self.retriever = retriever
        self.resource_service = resource_service
        self.provider_factory = provider_factory
        self.data_policy = data_policy
        self.runtime = runtime
        self.remote_rate_limiter = remote_rate_limiter
        self.clock = clock

    async def generate(
        self,
        *,
        user: dict[str, Any],
        scope: dict[str, int | None],
        payload: PracticeDraftRequest,
        request_id: str,
        context_validator: Callable[[], None],
        session_validator: Callable[[], None],
        event_recorder: Callable[[dict[str, Any]], None] | None = None,
        disconnected: threading.Event | None = None,
    ) -> dict[str, Any]:
        course_id = int(scope.get("course_id") or 0)
        if course_id <= 0:
            raise GeneralLearningError("COURSE_REQUIRED", "请选择课程后再生成草案。", status_code=400)
        deadline = self.clock() + PRACTICE_DRAFT_REQUEST_SECONDS
        lease = self.runtime.begin(int(user["id"]))
        course_references: list[CourseReference] = []
        external_resources: list[dict[str, Any]] = []
        reference_statuses: list[DraftReferenceStatus] = []
        warnings: list[str] = []
        reference_deadline = min(deadline, self.clock() + PRACTICE_DRAFT_REFERENCE_SECONDS)
        try:
            self._check_open(disconnected)
            context_validator()
            session_validator()

            policy_status = self.data_policy.status(
                course_id=course_id,
                use_course_references=payload.use_course_references,
                currently_authorized=True,
            )
            if payload.use_course_references and policy_status == "allowed":
                try:
                    course_references = await lease.run(
                        lambda: load_course_references(
                            self.storage,
                            self.retriever,
                            course_id=course_id,
                            query=" ".join((payload.topic, payload.objective)),
                            top_k=4,
                            max_chars=4000,
                        ),
                        deadline=reference_deadline,
                        disconnected=disconnected,
                    )
                    course_references = self._with_original_course_text(course_references)
                except CourseReferenceError as exc:
                    if exc.error_code == "REFERENCE_SCOPE_VIOLATION":
                        raise GeneralLearningError(
                            "REFERENCE_SCOPE_VIOLATION",
                            "课程参考作用域校验失败。",
                            status_code=409,
                        ) from exc
                    course_references = []
                    reference_statuses.append(
                        DraftReferenceStatus(
                            source="course",
                            lookup_state="UNAVAILABLE",
                            delivery_mode="course_reference",
                            error_code=exc.error_code,
                        )
                    )
                else:
                    reference_statuses.append(
                        DraftReferenceStatus(
                            source="course",
                            lookup_state="FOUND" if course_references else "NOT_FOUND",
                            delivery_mode="course_reference",
                        )
                    )
            else:
                reference_statuses.append(
                    DraftReferenceStatus(
                        source="course",
                        lookup_state=("NOT_CHECKED" if not payload.use_course_references else "UNAVAILABLE"),
                        delivery_mode=("disabled_by_user" if not payload.use_course_references else policy_status),
                        error_code=(None if not payload.use_course_references else "MODEL_CONTEXT_NOT_AUTHORIZED"),
                    )
                )

            for external_query in payload.external_queries:
                self._check_open(disconnected)
                context_validator()
                session_validator()

                def before_remote(source_name: str) -> None:
                    self._check_open(disconnected)
                    context_validator()
                    session_validator()
                    if self.remote_rate_limiter is not None:
                        self.remote_rate_limiter(source_name)

                response = await self.resource_service.search(
                    ResourceQuery(
                        query=external_query.query,
                        resource_type=external_query.resource_type,
                        page=1,
                        page_size=3,
                        allow_remote=True,
                    ),
                    deadline=reference_deadline,
                    before_remote=before_remote,
                )
                reference_statuses.append(
                    DraftReferenceStatus(
                        source=str(response.get("connector") or external_query.resource_type),
                        query=external_query.query,
                        lookup_state=str(response.get("lookup_state") or "UNAVAILABLE"),
                        delivery_mode=str(response.get("delivery_mode") or "unavailable"),
                        error_code=str(response.get("error") or "") or None,
                    )
                )
                if str(response.get("lookup_state") or "") != "FOUND":
                    continue
                for resource in list(response.get("resources") or [])[:3]:
                    item = dict(resource)
                    item.setdefault("delivery_mode", response.get("delivery_mode"))
                    item.setdefault("origin_kind", response.get("origin_kind"))
                    external_resources.append(item)

            catalog = build_draft_evidence(
                course_references=course_references,
                external_resources=external_resources,
            )
            self._check_open(disconnected)
            context_validator()
            session_validator()
            self._revalidate_catalog(catalog, course_id=course_id)

            try:
                provider = self.provider_factory()
            except LLMConfigurationError as exc:
                raise GeneralLearningError("MODEL_UNAVAILABLE", "草案模型尚未配置。", status_code=503) from exc
            if str(getattr(provider, "provider_name", "")) != "deepseek":
                raise GeneralLearningError("MODEL_UNAVAILABLE", "草案必须使用已配置的 DeepSeek 模型。", status_code=503)

            def outbound_guard(_stage: str) -> None:
                self._check_open(disconnected)
                context_validator()
                session_validator()
                current_policy = self.data_policy.status(
                    course_id=course_id,
                    use_course_references=payload.use_course_references,
                    currently_authorized=True,
                )
                if course_references and current_policy != "allowed":
                    raise ModelCallPreflightError(
                        "REFERENCE_CONTEXT_CHANGED",
                        "课程参考授权已变化。",
                        status_code=409,
                    )
                try:
                    self._revalidate_catalog(catalog, course_id=course_id)
                except GeneralLearningError as exc:
                    raise ModelCallPreflightError(exc.error_code, str(exc), status_code=exc.status_code) from exc

            candidate: PracticeDraftCandidate | None = None
            feedback: dict[str, Any] | None = None
            transport_state = {"recovery_used": False}
            for candidate_index in range(2):
                try:
                    candidate = await lease.run(
                        lambda current_feedback=feedback: provider.generate_practice_draft(
                            payload,
                            catalog.model_payload(),
                            deadline=deadline,
                            feedback=current_feedback,
                            outbound_guard=outbound_guard,
                            transport_state=transport_state,
                        ),
                        deadline=deadline,
                        disconnected=disconnected,
                    )
                    validate_draft_candidate(candidate, catalog)
                    break
                except DraftEvidenceError as exc:
                    feedback = exc.feedback()
                    candidate = None
                except LLMEmptyResponseError:
                    raise
                except (LLMResponseError, ValidationError, ValueError):
                    feedback = {"error_code": "DRAFT_CANDIDATE_SCHEMA_INVALID", "field": "candidate"}
                    candidate = None
                if candidate_index == 1:
                    break
            if candidate is None:
                raise GeneralLearningError(
                    "DRAFT_SELECTION_INVALID",
                    "模型两次返回的草案选择均未通过服务器校验。",
                    status_code=502,
                )

            try:
                selections = validate_draft_candidate(candidate, catalog)
            except DraftEvidenceError as exc:
                raise GeneralLearningError("DRAFT_SELECTION_INVALID", "草案选择未通过服务器校验。", status_code=502) from exc

            self._check_open(disconnected)
            context_validator()
            session_validator()
            self._revalidate_catalog(catalog, course_id=course_id)
            quotes = render_background_quotes(selections, catalog)
            status = "refused" if candidate.status == "refused" else ("grounded_draft" if quotes else "proposal_only")

            if event_recorder is not None:
                try:
                    event_recorder(
                        {
                            "request_id": request_id,
                            "mode": "practice_draft",
                            "status": status,
                            "reference_status": "used" if quotes else "none",
                            "reference_count": len(quotes),
                        }
                    )
                except GeneralLearningError:
                    raise
                except Exception:
                    warnings.append("学习事件暂未记录，不影响本次草案。")

            return {
                "request_id": request_id,
                "course_id": course_id,
                "class_id": scope.get("class_id"),
                "status": status,
                "background_quotes": [item.model_dump(mode="json") for item in quotes],
                "proposed_plan": candidate.proposed_plan.model_dump(mode="json") if candidate.proposed_plan else None,
                "refusal_reason": candidate.refusal_reason,
                "proposal_label": "模型提出的待验证教学草案",
                "reference_statuses": [item.model_dump(mode="json") for item in reference_statuses],
                "provider": "deepseek",
                "model": str(getattr(provider, "model_name", "")),
                "fallback_used": False,
                "warnings": warnings,
            }
        except GeneralLearningError:
            raise
        except ModelCallPreflightError as exc:
            raise GeneralLearningError(
                exc.error_code,
                "课程或学习会话状态已变化，本次草案未保存。",
                status_code=exc.status_code,
            ) from exc
        except LLMProviderError as exc:
            failure = classify_llm_failure(exc, phase="practice_draft")
            if failure["failure_category"] == "timeout":
                raise GeneralLearningError("MODEL_TIMEOUT", "草案模型响应超时。", status_code=504, retryable=True) from exc
            raise GeneralLearningError(
                "MODEL_FAILED",
                "草案模型暂不可用。",
                status_code=503,
                retryable=bool(failure["retryable"]),
            ) from exc
        finally:
            lease.close()

    def _revalidate_catalog(self, catalog: DraftEvidenceCatalog, *, course_id: int) -> None:
        for source in catalog.sources:
            if source.source_kind == "course":
                document_id, chunk_id = source.identity
                span = next(item for item in catalog.spans if item.source_id == source.source_id)
                document = self.storage.document(int(document_id))
                chunks = self.storage.chunks_for_document(int(document_id)) if document is not None else []
                chunk = next(
                    (item for item in chunks if int(item.get("id") or 0) == int(chunk_id)),
                    None,
                )
                document_hash, _, chunk_hash = source.content_hash.partition(":")
                if (
                    document is None
                    or int(document.get("course_id") or 0) != int(course_id)
                    or not is_student_visible_document(document)
                    or chunk is None
                    or str(chunk.get("index_status") or "").upper() != "INDEXED"
                    or str(document.get("content_hash") or "") != document_hash
                    or str(chunk.get("chunk_hash") or "") != chunk_hash
                    or span.exact_text not in str(chunk.get("chunk_text") or "")
                ):
                    raise GeneralLearningError(
                        "REFERENCE_CONTEXT_CHANGED",
                        "课程参考状态已变化，本次草案未保存。",
                        status_code=409,
                    )
            else:
                source_name, external_id = source.identity
                verified_reader = getattr(self.storage, "verified_external_resource_by_key", None)
                current = (
                    verified_reader(str(source_name), str(external_id))
                    if callable(verified_reader)
                    else None
                )
                if current is None or str(current.get("content_hash") or "") != source.content_hash:
                    raise GeneralLearningError(
                        "REFERENCE_CONTEXT_CHANGED",
                        "公开资料状态已变化，本次草案未保存。",
                        status_code=409,
                    )
                span = next(item for item in catalog.spans if item.source_id == source.source_id)
                current_view = (
                    normalize_standard_resource_view(current)
                    if str(current.get("resource_type") or "").casefold() == "standard"
                    else current
                )
                if span.exact_text not in str(current_view.get("abstract") or ""):
                    raise GeneralLearningError(
                        "REFERENCE_CONTEXT_CHANGED",
                        "公开资料正文已变化，本次草案未保存。",
                        status_code=409,
                    )
                if source.url and str(current.get("url") or "") != source.url:
                    raise GeneralLearningError(
                        "REFERENCE_CONTEXT_CHANGED",
                        "公开资料来源已变化，本次草案未保存。",
                        status_code=409,
                    )

    def _with_original_course_text(
        self,
        references: list[CourseReference],
    ) -> list[CourseReference]:
        originals: list[CourseReference] = []
        for reference in references:
            chunk = next(
                (
                    item
                    for item in self.storage.chunks_for_document(reference.document_id)
                    if int(item.get("id") or 0) == int(reference.chunk_id)
                ),
                None,
            )
            if chunk is None or str(chunk.get("chunk_hash") or "") != reference.chunk_hash:
                raise GeneralLearningError(
                    "REFERENCE_CONTEXT_CHANGED",
                    "课程参考切片已变化，本次草案未保存。",
                    status_code=409,
                )
            original = str(chunk.get("chunk_text") or "")
            if not original.strip():
                continue
            originals.append(replace(reference, text=original))
        return originals

    @staticmethod
    def _check_open(disconnected: threading.Event | None) -> None:
        if disconnected is not None and disconnected.is_set():
            raise GeneralLearningError("CLIENT_DISCONNECTED", "客户端已断开，本次草案未保存。", status_code=409)
