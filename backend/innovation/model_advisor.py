"""Request-scoped, fail-closed innovation suggestion advisor."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
import time
from typing import Any, Callable

from pydantic import ValidationError

from backend.core.errors import LLMConfigurationError, LLMProviderError, LLMResponseError, classify_llm_failure
from backend.innovation.schemas import InnovationSuggestionCandidate
from backend.learning.errors import GeneralLearningError
from backend.learning.runtime import LearningRuntime
from backend.llm.data_policy import ExternalModelDataPolicy, ModelCallPreflightError
from backend.rag.visibility import is_student_visible_document
from backend.resources.portal import ResourcePortalQuery, ResourcePortalService, is_public_github_record


INNOVATION_SUGGESTION_REQUEST_SECONDS = 90.0


@dataclass(frozen=True)
class AdvisorBackground:
    context_id: str
    source_id: str
    source_type: str
    title: str
    text: str
    delivery_mode: str
    identity: tuple[Any, ...]
    content_hash: str

    def model_payload(self) -> dict[str, str]:
        return {
            "context_id": self.context_id,
            "type": self.source_type,
            "title": self.title,
            "text": self.text,
        }


@dataclass(frozen=True)
class AdvisorCatalog:
    sources: tuple[AdvisorBackground, ...]
    statuses: tuple[dict[str, Any], ...]
    course_reference_status: str
    warnings: tuple[str, ...] = ()


class InnovationAdvisorService:
    def __init__(
        self,
        *,
        storage: Any,
        provider_factory: Callable[[], Any],
        data_policy: ExternalModelDataPolicy,
        runtime: LearningRuntime,
        competitions_dir: str = "datasets/competitions",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.storage = storage
        self.provider_factory = provider_factory
        self.data_policy = data_policy
        self.runtime = runtime
        self.competitions_dir = competitions_dir
        self.clock = clock

    async def suggest(
        self,
        *,
        user: dict[str, Any],
        advisor_context: dict[str, Any],
        use_course_references: bool,
        request_id: str,
        context_validator: Callable[[], dict[str, Any]],
        session_validator: Callable[[], None],
        event_recorder: Callable[[dict[str, Any]], None] | None = None,
        disconnected: threading.Event | None = None,
    ) -> dict[str, Any]:
        session = dict(advisor_context["session"])
        course_id = int(session.get("course_id") or 0)
        if course_id <= 0:
            raise GeneralLearningError("REFERENCE_CONTEXT_CHANGED", "创新会话缺少可信课程上下文。", status_code=409)
        deadline = self.clock() + INNOVATION_SUGGESTION_REQUEST_SECONDS
        lease = self.runtime.begin(int(user["id"]))
        try:
            self._check_open(disconnected)
            context_validator()
            session_validator()
            catalog = self._build_catalog(
                session,
                use_course_references=use_course_references,
                course_id=course_id,
            )
            self._revalidate_catalog(catalog, course_id=course_id)
            try:
                provider = self.provider_factory()
            except LLMConfigurationError as exc:
                raise GeneralLearningError("MODEL_UNAVAILABLE", "创新建议模型尚未配置。", status_code=503) from exc
            if str(getattr(provider, "provider_name", "")) != "deepseek":
                raise GeneralLearningError("MODEL_UNAVAILABLE", "创新建议必须使用已配置的 DeepSeek 模型。", status_code=503)

            outbound_context = {
                "topic": session.get("topic") or "",
                "current_plan": session.get("current_plan") or "",
                "answers": [
                    {"question": item.get("question") or "", "answer": item.get("answer") or ""}
                    for item in list(advisor_context.get("answers") or [])
                    if str(item.get("answer") or "").strip()
                ],
                "background_catalog": [item.model_payload() for item in catalog.sources],
            }

            def outbound_guard(_stage: str) -> None:
                self._check_open(disconnected)
                context_validator()
                session_validator()
                self._revalidate_catalog(catalog, course_id=course_id)

            try:
                def call_provider() -> tuple[bool, Any]:
                    try:
                        return (
                            True,
                            provider.generate_innovation_suggestions(
                                outbound_context,
                                deadline=deadline,
                                outbound_guard=outbound_guard,
                            ),
                        )
                    except (GeneralLearningError, ModelCallPreflightError) as exc:
                        return False, {
                            "kind": "domain",
                            "error_code": exc.error_code,
                            "message": str(exc),
                            "status_code": exc.status_code,
                            "retryable": getattr(exc, "retryable", False),
                            "retry_after": getattr(exc, "retry_after", None),
                        }
                    except LLMProviderError as exc:
                        return False, {
                            "kind": "provider",
                            **classify_llm_failure(exc, phase="innovation_suggestion"),
                        }
                    except BaseException:
                        return False, {"kind": "unknown", "retryable": False}

                succeeded, provider_result = await lease.run(
                    call_provider,
                    deadline=deadline,
                    disconnected=disconnected,
                )
                if not succeeded:
                    failure = dict(provider_result)
                    if failure.get("kind") == "domain":
                        raise GeneralLearningError(
                            str(failure["error_code"]),
                            str(failure["message"]),
                            status_code=int(failure["status_code"]),
                            retryable=bool(failure.get("retryable")),
                            retry_after=failure.get("retry_after"),
                        )
                    if failure.get("kind") == "provider":
                        if failure.get("failure_category") == "timeout":
                            raise GeneralLearningError(
                                "MODEL_TIMEOUT",
                                "创新建议模型响应超时。",
                                status_code=504,
                                retryable=True,
                            )
                        raise GeneralLearningError(
                            "MODEL_FAILED",
                            "创新建议模型暂不可用。",
                            status_code=503,
                            retryable=bool(failure.get("retryable")),
                        )
                    raise GeneralLearningError(
                        "MODEL_FAILED",
                        "创新建议模型暂不可用。",
                        status_code=503,
                    )
                raw_candidate = provider_result
                candidate = (
                    raw_candidate
                    if isinstance(raw_candidate, InnovationSuggestionCandidate)
                    else InnovationSuggestionCandidate.model_validate(raw_candidate)
                )
            except (ValidationError, LLMResponseError, ValueError) as exc:
                raise GeneralLearningError(
                    "INNOVATION_SOURCE_INVALID",
                    "模型创新建议未通过服务器结构校验。",
                    status_code=502,
                ) from exc

            mapped = self._validate_and_map(candidate, catalog)
            self._check_open(disconnected)
            context_validator()
            session_validator()
            self._revalidate_catalog(catalog, course_id=course_id)
            used_sources = _dedupe(
                source_id
                for item in mapped
                for source_id in item["background_source_ids"]
            )
            if event_recorder is not None:
                event_recorder(
                    {
                        "request_id": request_id,
                        "mode": "innovation_suggestion",
                        "status": candidate.status,
                        "reference_status": "used" if used_sources else "none",
                        "reference_count": len(used_sources),
                    }
                )
            return {
                "request_id": request_id,
                "session_id": int(session["id"]),
                "course_id": course_id,
                "status": candidate.status,
                "suggestions": mapped,
                "refusal_reason": candidate.refusal_reason,
                "generation_mode": "deepseek",
                "hypotheses_verified": False,
                "proposal_label": "模型提出的待验证创新建议",
                "source_statuses": list(catalog.statuses),
                "course_reference_status": catalog.course_reference_status,
                "warnings": list(catalog.warnings),
                "collection_status": "not_recorded",
                "background_source_ids": used_sources,
                "provider": "deepseek",
                "model": str(getattr(provider, "model_name", "")),
                "fallback_used": False,
            }
        except GeneralLearningError:
            raise
        except ModelCallPreflightError as exc:
            raise GeneralLearningError(exc.error_code, str(exc), status_code=exc.status_code) from exc
        except LLMProviderError as exc:
            failure = classify_llm_failure(exc, phase="innovation_suggestion")
            if failure["failure_category"] == "timeout":
                raise GeneralLearningError("MODEL_TIMEOUT", "创新建议模型响应超时。", status_code=504, retryable=True) from exc
            raise GeneralLearningError(
                "MODEL_FAILED",
                "创新建议模型暂不可用。",
                status_code=503,
                retryable=bool(failure["retryable"]),
            ) from exc
        finally:
            lease.close()

    def _build_catalog(
        self,
        session: dict[str, Any],
        *,
        use_course_references: bool,
        course_id: int,
    ) -> AdvisorCatalog:
        policy_status = self.data_policy.status(
            course_id=course_id,
            use_course_references=use_course_references,
            currently_authorized=True,
        )
        sources: list[AdvisorBackground] = []
        statuses: list[dict[str, Any]] = []
        warnings: list[str] = []
        seen: set[str] = set()
        for case in list(session.get("cases") or [])[:12]:
            source_id = str(case.get("source_id") or "")
            source_type, identity_conflict = _trusted_source_type(case)
            if not source_id or source_id in seen:
                continue
            seen.add(source_id)
            if source_type == "course_document" and policy_status != "allowed":
                statuses.append(_status(source_id, source_type, policy_status, policy_status))
                continue
            if identity_conflict:
                statuses.append(_status(source_id, source_type, "unavailable", "identity_conflict"))
                warnings.append(f"案例 {source_id} 的保存类型与服务器来源身份冲突，未发送给模型。")
                continue
            current = self._resolve_current_case(case, course_id=course_id)
            if current is None:
                statuses.append(_status(source_id, source_type, "unavailable", "current_state_rejected"))
                warnings.append(f"案例 {source_id} 当前不可用，未发送给模型。")
                continue
            context_id = f"B{len(sources) + 1}"
            sources.append(
                AdvisorBackground(
                    context_id=context_id,
                    source_id=source_id,
                    source_type=source_type,
                    title=str(current["title"])[:300],
                    text=str(current["text"])[:1200],
                    delivery_mode=str(current["delivery_mode"])[:60],
                    identity=tuple(current["identity"]),
                    content_hash=str(current["content_hash"]),
                )
            )
            statuses.append(_status(source_id, source_type, "available", str(current["delivery_mode"])))
        return AdvisorCatalog(
            tuple(sources[:8]),
            tuple(statuses),
            policy_status,
            tuple(warnings),
        )

    def _resolve_current_case(self, case: dict[str, Any], *, course_id: int) -> dict[str, Any] | None:
        source_id = str(case.get("source_id") or "")
        metadata = dict(case.get("metadata") or {})
        if source_id.startswith("rag:"):
            parts = source_id.split(":")
            if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
                return None
            document_id, chunk_id = int(parts[1]), int(parts[2])
            if (
                metadata.get("document_id") not in {None, document_id}
                or metadata.get("chunk_id") not in {None, chunk_id}
                or str(case.get("external_id") or "") not in {"", str(chunk_id)}
            ):
                return None
            return self._course_case(
                document_id,
                chunk_id,
                course_id=course_id,
                expected_document_hash=_optional_hash(metadata, "document_content_hash", "content_hash"),
                expected_chunk_hash=_optional_hash(metadata, "chunk_hash"),
            )
        if source_id.startswith("document:"):
            raw_id = source_id.partition(":")[2]
            if not raw_id.isdigit():
                return None
            document_id = int(raw_id)
            if (
                metadata.get("document_id") not in {None, document_id}
                or str(case.get("external_id") or "") not in {"", str(document_id)}
            ):
                return None
            raw_chunk_id = metadata.get("chunk_id")
            if raw_chunk_id is not None:
                try:
                    chunk_id = int(raw_chunk_id)
                except (TypeError, ValueError):
                    return None
                return self._course_case(
                    document_id,
                    chunk_id,
                    course_id=course_id,
                    expected_document_hash=_optional_hash(metadata, "document_content_hash", "content_hash"),
                    expected_chunk_hash=_optional_hash(metadata, "chunk_hash"),
                )
            chunks = sorted(
                (
                    item
                    for item in self.storage.chunks_for_document(document_id)
                    if str(item.get("index_status") or "").upper() == "INDEXED"
                ),
                key=lambda item: int(item.get("id") or 0),
            )
            if not chunks:
                return None
            return self._course_case(
                document_id,
                int(chunks[0]["id"]),
                course_id=course_id,
                expected_document_hash=_optional_hash(metadata, "document_content_hash", "content_hash"),
            )
        if source_id.startswith("competition:"):
            portal = ResourcePortalService(
                self.storage,
                competitions_dir=self.competitions_dir,
                course_id=course_id,
            )
            items = portal.search(ResourcePortalQuery(resource_type="competition", page_size=50))["items"]
            current = next((item for item in items if str(item.get("resource_id")) == source_id), None)
            if current is None or str(current.get("source_name") or "") != "teacher_dataset":
                return None
            text = str(current.get("raw_summary") or current.get("summary") or "")
            if not text:
                return None
            return {
                "title": current.get("title") or source_id,
                "text": text,
                "delivery_mode": "local_dataset",
                "identity": ("competition", source_id),
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        source_name = str(case.get("source_name") or "")
        external_id = str(case.get("external_id") or "")
        if not source_name or not external_id:
            return None
        current = self.storage.verified_external_resource_by_key(source_name, external_id)
        if current is None:
            return None
        if source_name == "github" and not is_public_github_record(current):
            return None
        text = str(current.get("abstract") or "")
        if not text:
            return None
        return {
            "title": current.get("title") or external_id,
            "text": text,
            "delivery_mode": current.get("delivery_mode") or current.get("cache_status") or "verified_cache",
            "identity": ("external", source_name, external_id),
            "content_hash": str(current.get("content_hash") or ""),
        }

    def _course_case(
        self,
        document_id: int,
        chunk_id: int,
        *,
        course_id: int,
        expected_document_hash: str | None = None,
        expected_chunk_hash: str | None = None,
    ) -> dict[str, Any] | None:
        document = self.storage.document(document_id)
        chunk = next(
            (item for item in self.storage.chunks_for_document(document_id) if int(item.get("id") or 0) == chunk_id),
            None,
        )
        if (
            document is None
            or int(document.get("course_id") or 0) != course_id
            or not is_student_visible_document(document)
            or chunk is None
            or str(chunk.get("index_status") or "").upper() != "INDEXED"
            or (expected_document_hash is not None and str(document.get("content_hash") or "") != expected_document_hash)
            or (expected_chunk_hash is not None and str(chunk.get("chunk_hash") or "") != expected_chunk_hash)
        ):
            return None
        text = str(chunk.get("chunk_text") or "")
        if not text:
            return None
        return {
            "title": chunk.get("title") or chunk.get("filename") or document.get("filename") or str(document_id),
            "text": text,
            "delivery_mode": "local_rag",
            "identity": ("course", document_id, chunk_id),
            "content_hash": f"{document.get('content_hash') or ''}:{chunk.get('chunk_hash') or ''}",
        }

    def _revalidate_catalog(self, catalog: AdvisorCatalog, *, course_id: int) -> None:
        for item in catalog.sources:
            if item.identity[0] == "course":
                current = self._course_case(int(item.identity[1]), int(item.identity[2]), course_id=course_id)
            elif item.identity[0] == "external":
                current_case = {
                    "source_id": item.source_id,
                    "source_name": item.identity[1],
                    "external_id": item.identity[2],
                }
                current = self._resolve_current_case(current_case, course_id=course_id)
            else:
                current_case = {"source_id": item.source_id, "type": "competition"}
                current = self._resolve_current_case(current_case, course_id=course_id)
            if current is None or str(current["content_hash"]) != item.content_hash or item.text not in str(current["text"]):
                raise GeneralLearningError(
                    "REFERENCE_CONTEXT_CHANGED",
                    "创新建议背景来源已变化，本次结果未保存。",
                    status_code=409,
                )

    @staticmethod
    def _validate_and_map(
        candidate: InnovationSuggestionCandidate,
        catalog: AdvisorCatalog,
    ) -> list[dict[str, Any]]:
        context_map = {item.context_id: item.source_id for item in catalog.sources}
        mapped: list[dict[str, Any]] = []
        used_context_ids: set[str] = set()
        for suggestion in candidate.suggestions:
            ids = list(suggestion.background_context_ids)
            if (
                len(ids) != len(set(ids))
                or any(value not in context_map for value in ids)
                or any(value in used_context_ids for value in ids)
            ):
                raise GeneralLearningError(
                    "INNOVATION_SOURCE_INVALID",
                    "模型引用了未知、重复或跨请求的背景标识。",
                    status_code=502,
                )
            used_context_ids.update(ids)
            payload = suggestion.model_dump(mode="json")
            payload["background_source_ids"] = [context_map[value] for value in ids]
            mapped.append(payload)
        return mapped

    @staticmethod
    def _check_open(disconnected: threading.Event | None) -> None:
        if disconnected is not None and disconnected.is_set():
            raise GeneralLearningError("CLIENT_DISCONNECTED", "客户端已断开，本次创新建议未保存。", status_code=409)


def _status(source_id: str, source_type: str, status: str, delivery_mode: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "status": status,
        "source_type": source_type,
        "delivery_mode": delivery_mode,
        "reason": None if status == "available" else delivery_mode,
    }


def _trusted_source_type(case: dict[str, Any]) -> tuple[str, bool]:
    """Derive the security category from server-issued identity, never history labels."""

    source_id = str(case.get("source_id") or "")
    saved_type = str(case.get("type") or "unknown")
    if source_id.startswith(("rag:", "document:")):
        return "course_document", saved_type not in {"", "unknown", "course_document"}
    if source_id.startswith("competition:"):
        return "competition", saved_type not in {"", "unknown", "competition"}
    prefix = source_id.partition(":")[0]
    supported_external = {"paper", "project", "vulnerability", "standard"}
    if saved_type not in supported_external:
        return saved_type or "unknown", True
    return saved_type, prefix in supported_external and prefix != saved_type


def _optional_hash(metadata: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in metadata and metadata.get(key) is not None:
            value = str(metadata.get(key) or "")
            return value or None
    return None


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result
