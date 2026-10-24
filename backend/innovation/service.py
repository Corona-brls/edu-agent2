"""Innovation coaching workflow service."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.analytics.events import LearningEventService
from backend.connectors.schemas import ResourceQuery
from backend.innovation import recommender
from backend.innovation.prompts import SAFETY_BOUNDARY
from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.visibility import is_student_visible_document
from backend.resources.portal import ResourcePortalQuery, ResourcePortalService, is_public_github_record
from backend.storage.logger import LearningLogger
from backend.tools.mcp_client import MCPClientGateway


LOGGER = logging.getLogger(__name__)


class InnovationError(ValueError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class CaseCollection:
    cases: list[dict[str, Any]]
    source_attempts: list[dict[str, Any]]


class InnovationService:
    def __init__(
        self,
        storage: LearningLogger,
        *,
        retriever: HybridRetriever | None = None,
        mcp_gateway: MCPClientGateway | None = None,
        resource_portal: ResourcePortalService | None = None,
        resource_service: Any | None = None,
        remote_rate_limiter: Callable[[int, str], None] | None = None,
    ) -> None:
        self.storage = storage
        self.retriever = retriever or HybridRetriever(storage=storage, vector_store=_KeywordOnlyVectorStore())
        self.mcp_gateway = mcp_gateway or MCPClientGateway()
        self.resource_portal = resource_portal or ResourcePortalService(storage)
        self.resource_service = resource_service
        self.remote_rate_limiter = remote_rate_limiter

    def start_session(
        self,
        *,
        user: dict[str, Any],
        topic: str,
        current_plan: str,
        client_session_id: str | None = None,
        course_id: int | None = None,
        class_id: int | None = None,
        disconnected: threading.Event | None = None,
    ) -> dict[str, Any]:
        if str(user.get("role") or "").lower() != "student":
            raise InnovationError("INNOVATION_FORBIDDEN", "只有当前课程学生可以创建创新会话。")
        topic = topic.strip()
        current_plan = current_plan.strip()
        if not topic or not current_plan:
            raise InnovationError("INVALID_INNOVATION_REQUEST", "专题和当前方案不能为空。")
        context = self._default_context(user, course_id=course_id, class_id=class_id)
        self._validate_learning_session(user, client_session_id, context)

        def validate_fresh_start() -> dict[str, Any]:
            return self._validate_start_state(
                user,
                context=context,
                client_session_id=client_session_id,
                disconnected=disconnected,
            )

        collection = self._collect_cases(
            topic=topic,
            current_plan=current_plan,
            course_id=context["course_id"],
            actor_id=int(user["id"]),
            before_remote=validate_fresh_start,
        )
        limitations = recommender.infer_limitations(topic, current_plan, collection.cases)
        questions = recommender.generate_questions(topic, current_plan, collection.cases)
        fresh_user = validate_fresh_start()
        session = self.storage.create_innovation_session(
            user_id=int(fresh_user["id"]),
            course_id=context["course_id"],
            class_id=context["class_id"],
            topic=topic,
            current_plan=current_plan,
            cases=collection.cases,
            limitations=limitations,
            questions=questions,
        )
        self._record_event(fresh_user, client_session_id, "INNOVATION_SESSION_STARTED", topic, {"session_id": session["id"]})
        return self._session_payload(session, source_attempts=collection.source_attempts)

    def answer_question(
        self,
        *,
        user: dict[str, Any],
        session_id: int,
        question_id: int,
        answer: str,
        client_session_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._require_write_session(user, session_id)
        self._validate_learning_session(user, client_session_id, session)
        answer = answer.strip()
        if not answer:
            raise InnovationError("EMPTY_INNOVATION_ANSWER", "学生回答不能为空。")
        follow_up = recommender.generate_follow_up(answer, session["topic"])
        try:
            answered = self.storage.answer_innovation_question(
                session_id=session_id,
                question_id=question_id,
                answer=answer,
                follow_up_question=follow_up,
            )
        except ValueError as exc:
            raise InnovationError("INNOVATION_QUESTION_NOT_FOUND", "创新问题不存在或不属于该会话。") from exc
        session = self.storage.innovation_session(session_id)
        self._record_event(
            user,
            client_session_id,
            "INNOVATION_QUESTION_ANSWERED",
            session["topic"],
            {"session_id": session_id, "question_id": question_id},
        )
        payload = self._session_payload(session)
        payload["answered_question"] = answered
        payload["follow_up_question"] = follow_up
        return payload

    def finalize_session(
        self,
        *,
        user: dict[str, Any],
        session_id: int,
        client_session_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._require_write_session(user, session_id)
        self._validate_learning_session(user, client_session_id, session)
        answers = self.storage.innovation_questions(session_id)
        ideas = recommender.build_final_ideas(
            topic=session["topic"],
            current_plan=session["current_plan"],
            cases=session["cases"],
            answers=answers,
        )
        finalized = self.storage.finalize_innovation_session(session_id, ideas)
        self._record_event(
            user,
            client_session_id,
            "INNOVATION_QUESTION_ANSWERED",
            finalized["topic"],
            {"session_id": session_id, "finalized": True},
        )
        return self._session_payload(finalized)

    def get_session(self, *, user: dict[str, Any], session_id: int) -> dict[str, Any]:
        return self._session_payload(self._require_read_session(user, session_id))

    def list_sessions(
        self, *, user: dict[str, Any], course_id: int, class_id: int | None
    ) -> dict[str, list[dict[str, Any]]]:
        fresh = self.storage.user(int(user["id"]))
        if not fresh or not fresh.get("enabled"):
            raise InnovationError("USER_DISABLED", "当前账号不可用。")
        if int(fresh.get("token_version") or 0) != int(user.get("token_version") or 0):
            raise InnovationError("TOKEN_REVOKED", "登录状态已失效。")
        if str(fresh.get("role") or "").lower() != "student":
            raise InnovationError("INNOVATION_FORBIDDEN", "只有学生可以查看自己的创新会话。")
        context = self._default_context(fresh, course_id=course_id, class_id=class_id)
        return {"items": self.storage.innovation_session_summaries(
            user_id=int(fresh["id"]), course_id=int(context["course_id"]),
            class_id=int(context["class_id"]) if context["class_id"] is not None else None,
        )}

    def get_advisor_context(
        self,
        *,
        user: dict[str, Any],
        session_id: int,
        client_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Return the currently authorized server-side context for one suggestion call."""

        session = self._require_write_session(user, session_id)
        self._validate_learning_session(user, client_session_id, session)
        return {
            "session": session,
            "answers": self.storage.innovation_questions(session_id),
        }

    def _collect_cases(
        self,
        *,
        topic: str,
        current_plan: str,
        course_id: int | None,
        actor_id: int | None = None,
        before_remote: Callable[[], Any] | None = None,
    ) -> CaseCollection:
        resource_portal = (
            self.resource_portal
            if getattr(self.resource_portal, "course_id", course_id) == course_id
            else ResourcePortalService(self.storage, course_id=course_id)
        )
        resources = resource_portal.search(
            ResourcePortalQuery(keyword=topic, page_size=50)
        )["items"]
        if len({item.get("type") for item in resources}) < 2:
            resources.extend(resource_portal.search(ResourcePortalQuery(topic=topic, page_size=50))["items"])
        source_attempts: list[dict[str, Any]] = []
        if self.resource_service is not None:
            remote_resources, source_attempts = self._on_demand_resources(
                topic,
                actor_id=actor_id,
                before_remote=before_remote,
            )
            resources.extend(remote_resources)
        resources.extend(self._rag_resources(f"{topic} {current_plan}", course_id=course_id))
        return CaseCollection(
            cases=recommender.select_cases(
                resources,
                topic=topic,
                current_plan=current_plan,
                limit=6,
            ),
            source_attempts=source_attempts,
        )

    def _on_demand_resources(
        self,
        topic: str,
        *,
        actor_id: int | None,
        before_remote: Callable[[], Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        resources: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        for resource_type in ("paper", "project", "vulnerability"):
            query = ResourceQuery(
                query=topic,
                resource_type=resource_type,
                page_size=3,
                allow_remote=True,
            )
            try:
                kwargs: dict[str, Any] = {}
                if before_remote is not None or (self.remote_rate_limiter is not None and actor_id is not None):
                    def remote_hook(source_name: str) -> None:
                        if before_remote is not None:
                            before_remote()
                        if self.remote_rate_limiter is not None and actor_id is not None:
                            self.remote_rate_limiter(actor_id, source_name)

                    kwargs["before_remote"] = remote_hook
                response = self.resource_service.search(query, **kwargs)
                if inspect.isawaitable(response):
                    response = _run_awaitable(response)
            except Exception as exc:
                if getattr(exc, "status_code", None) == 429:
                    raise
                attempts.append(
                    {
                        "resource_type": resource_type,
                        "source_name": resource_type,
                        "status": "UNAVAILABLE",
                        "result_count": 0,
                        "delivery_mode": "unavailable",
                        "error_code": str(getattr(exc, "error_code", type(exc).__name__)),
                        "fetched_at": None,
                        "incomplete_results": None,
                    }
                )
                continue
            mapped_count = 0
            for item in (response or {}).get("resources", []):
                mapped = _external_resource_to_case(item, response)
                if mapped:
                    resources.append(mapped)
                    mapped_count += 1
            detailed_attempts = list((response or {}).get("source_attempts") or [])
            if detailed_attempts:
                for item in detailed_attempts:
                    attempts.append(
                        {
                            "resource_type": resource_type,
                            "source_name": str(item.get("source_name") or (response or {}).get("connector") or resource_type),
                            "status": str(item.get("status") or "UNAVAILABLE"),
                            "result_count": int(item.get("result_count") or 0),
                            "delivery_mode": str(item.get("delivery_mode") or (response or {}).get("delivery_mode") or "unknown"),
                            "error_code": str(item.get("error_code") or "") or None,
                            "fetched_at": item.get("fetched_at"),
                            "incomplete_results": item.get("incomplete_results"),
                        }
                    )
            else:
                attempts.append(
                    {
                        "resource_type": resource_type,
                        "source_name": str((response or {}).get("connector") or resource_type),
                        "status": str((response or {}).get("lookup_state") or "UNAVAILABLE"),
                        "result_count": mapped_count,
                        "delivery_mode": str((response or {}).get("delivery_mode") or "unknown"),
                        "error_code": str((response or {}).get("error") or "") or None,
                        "fetched_at": None,
                        "incomplete_results": (response or {}).get("incomplete_results"),
                    }
                )
        return resources, attempts

    def _rag_resources(self, query: str, course_id: int | None) -> list[dict[str, Any]]:
        if course_id is None:
            LOGGER.warning("Innovation scoped RAG retrieval rejected: missing course scope")
            return []
        try:
            response = self.retriever.retrieve(query, top_k=3, course_id=course_id)
        except Exception as exc:
            LOGGER.warning(
                "Innovation scoped RAG retrieval failed: %s",
                type(exc).__name__,
            )
            return []

        indexed_documents = {
            int(document["id"]): document
            for document in self.storage.documents(course_id=course_id)
            if is_student_visible_document(document)
        }
        indexed_chunks = {
            (int(chunk["document_id"]), int(chunk["id"])): chunk
            for chunk in self.storage.document_chunks(course_id=course_id)
            if int(chunk["document_id"]) in indexed_documents
            and str(chunk.get("index_status", "")).upper() == "INDEXED"
        }
        resources = []
        for result in response.results:
            chunk = indexed_chunks.get((int(result.document_id), int(result.chunk_id)))
            if chunk is None:
                continue
            document = indexed_documents[int(result.document_id)]
            resources.append(
                {
                    "resource_id": f"rag:{result.document_id}:{result.chunk_id}",
                    "type": "course_document",
                    "title": chunk.get("title") or chunk.get("filename") or result.title or result.source,
                    "summary": chunk.get("chunk_text") or result.snippet,
                    "source": chunk.get("filename") or result.source,
                    "source_name": "rag",
                    "external_id": str(result.chunk_id),
                    "url": "",
                    "tags": [chunk.get("source_type") or result.source_type],
                    "cache_status": "local",
                    "origin_kind": "course_document",
                    "delivery_mode": "local_rag",
                    "fetched_at": document.get("updated_at") or document.get("created_at") or "",
                    "provenance_verified": True,
                    "metadata": {
                        "course_id": course_id,
                        "document_id": int(result.document_id),
                        "chunk_id": int(result.chunk_id),
                        "page": chunk.get("page"),
                        "section": chunk.get("section"),
                    },
                }
            )
        return resources

    def _mcp_resources(self, topic: str) -> list[dict[str, Any]]:
        resources = []
        for tool_name, arguments, resource_type in [
            ("paper_search", {"query": topic, "top_k": 2}, "paper"),
            ("standard_lookup", {"query": topic, "top_k": 2}, "standard"),
        ]:
            call = self.mcp_gateway.call_tool(
                tool_name,
                arguments,
                run_id=f"innovation-{topic[:24]}",
                step_id=tool_name,
                role="student",
            )
            if not call.result.success:
                continue
            data = call.result.data or {}
            tool_items = [*(data.get("resources") or []), *(data.get("results") or [])]
            for item in tool_items:
                actual_type = _resource_type_from_tool_item(item, default=resource_type)
                resources.append(
                    {
                        "resource_id": f"mcp:{tool_name}:{item.get('source', '')}:{item.get('section', '')}",
                        "type": actual_type,
                        "title": item.get("title") or item.get("source") or tool_name,
                        "summary": item.get("abstract") or item.get("summary") or item.get("snippet") or "",
                        "source": tool_name,
                        "source_name": "mcp",
                        "external_id": item.get("source") or tool_name,
                        "url": item.get("url") or item.get("source_url") or item.get("link") or "",
                        "tags": [tool_name],
                        "cache_status": "local_fallback" if call.fallback_used else "mcp",
                        "metadata": {
                            "verification_status": item.get("verification_status", ""),
                            "doi": item.get("doi", ""),
                            "arxiv_id": item.get("arxiv_id", ""),
                        },
                    }
                )
        return resources

    def _require_read_session(self, user: dict[str, Any], session_id: int) -> dict[str, Any]:
        session = self.storage.innovation_session(session_id)
        if not session:
            raise InnovationError("INNOVATION_SESSION_NOT_FOUND", "创新会话不存在。")
        role = str(user["role"]).lower()
        context_status = self._session_context_status(session)
        session["context_status"] = context_status
        if role == "admin":
            return session
        if context_status != "valid":
            raise InnovationError("SESSION_CONTEXT_REQUIRED", "创新会话缺少可信课程上下文。")
        if role != "student" or int(session["user_id"]) != int(user["id"]):
            raise InnovationError("INNOVATION_FORBIDDEN", "无权访问该创新会话。")
        self._assert_current_student_scope(user, session)
        return session

    def _require_write_session(self, user: dict[str, Any], session_id: int) -> dict[str, Any]:
        session = self._require_read_session(user, session_id)
        if str(user.get("role") or "").lower() != "student" or int(session["user_id"]) != int(user["id"]):
            raise InnovationError("INNOVATION_FORBIDDEN", "只有当前学生会话所有者可以修改创新会话。")
        return session

    def _session_context_status(self, session: dict[str, Any]) -> str:
        course_id = session.get("course_id")
        if course_id is None:
            return "missing"
        class_id = session.get("class_id")
        if class_id is not None and not self.storage.class_belongs_to_course(int(class_id), int(course_id)):
            return "invalid"
        return "valid"

    def _assert_current_student_scope(self, user: dict[str, Any], session: dict[str, Any]) -> None:
        course_id = int(session["course_id"])
        class_id = int(session["class_id"]) if session.get("class_id") is not None else None
        if class_id is not None:
            allowed = self.storage.user_in_class(int(user["id"]), class_id, course_id)
        else:
            allowed = bool(self.storage.classes_for_user(int(user["id"]), course_id=course_id))
        if not allowed:
            raise InnovationError("COURSE_FORBIDDEN", "当前学生已无权访问该课程或班级。")

    def _validate_learning_session(
        self,
        user: dict[str, Any],
        client_session_id: str | None,
        context: dict[str, Any],
    ) -> None:
        if not client_session_id:
            return
        try:
            LearningEventService(self.storage).validate_session(
                user=user,
                client_session_id=client_session_id,
                course_id=int(context["course_id"]),
                class_id=int(context["class_id"]) if context.get("class_id") is not None else None,
            )
        except Exception as exc:
            error_code = getattr(exc, "error_code", "SESSION_CONTEXT_CONFLICT")
            raise InnovationError(str(error_code), str(exc)) from exc

    def _validate_start_state(
        self,
        authenticated_user: dict[str, Any],
        *,
        context: dict[str, Any],
        client_session_id: str | None,
        disconnected: threading.Event | None,
    ) -> dict[str, Any]:
        if disconnected is not None and disconnected.is_set():
            raise InnovationError("CLIENT_DISCONNECTED", "客户端已断开，本次创新会话未创建。")
        fresh_user = self.storage.user(int(authenticated_user["id"]))
        if not fresh_user or not bool(fresh_user.get("enabled")):
            raise InnovationError("USER_DISABLED", "用户不存在或已禁用。")
        if int(fresh_user.get("token_version") or 0) != int(authenticated_user.get("token_version") or 0):
            raise InnovationError("TOKEN_REVOKED", "访问令牌已失效，请重新登录。")
        if str(fresh_user.get("role") or "").lower() != "student":
            raise InnovationError("INNOVATION_FORBIDDEN", "只有当前课程学生可以创建创新会话。")
        fresh_context = self._default_context(
            fresh_user,
            course_id=int(context["course_id"]),
            class_id=int(context["class_id"]) if context.get("class_id") is not None else None,
        )
        if fresh_context != context:
            raise InnovationError("COURSE_FORBIDDEN", "课程或班级权限已变化。")
        self._validate_learning_session(fresh_user, client_session_id, context)
        return fresh_user

    def _session_payload(
        self,
        session: dict[str, Any],
        *,
        source_attempts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        questions = self.storage.innovation_questions(int(session["id"]))
        payload_session = dict(session)
        payload_session["final_ideas"] = [
            recommender.present_template_idea(item)
            for item in list(session.get("final_ideas") or [])
        ]
        return {
            **payload_session,
            "cases": [recommender.present_saved_case(case) for case in session.get("cases", [])],
            "case_availability": {
                str(case.get("source_id") or ""): self._saved_case_availability(case, session=session)
                for case in session.get("cases", [])
                if str(case.get("source_id") or "")
            },
            "questions": questions,
            "safety_boundary": SAFETY_BOUNDARY,
            "generation_mode": "template",
            "hypotheses_verified": False,
            "source_attempts": list(source_attempts or []),
            "collection_status": "recorded" if source_attempts is not None else "not_recorded",
        }

    def _saved_case_availability(
        self,
        case: dict[str, Any],
        *,
        session: dict[str, Any],
    ) -> str:
        source_id = str(case.get("source_id") or "")
        course_id = int(session.get("course_id") or 0)
        available = False
        if source_id.startswith(("rag:", "document:")) and course_id > 0:
            parts = source_id.split(":")
            document_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            document = self.storage.document(document_id) if document_id else None
            if document is not None and int(document.get("course_id") or 0) == course_id and is_student_visible_document(document):
                chunks = [
                    item
                    for item in self.storage.chunks_for_document(document_id)
                    if str(item.get("index_status") or "").upper() == "INDEXED"
                ]
                if source_id.startswith("rag:"):
                    chunk_id = int(parts[2]) if len(parts) == 3 and parts[2].isdigit() else 0
                    available = any(int(item.get("id") or 0) == chunk_id for item in chunks)
                else:
                    available = bool(chunks)
        elif source_id.startswith("competition:"):
            available = ResourcePortalService(
                self.storage,
                course_id=course_id or None,
            ).resource_exists(source_id)
        else:
            source_name = str(case.get("source_name") or "")
            external_id = str(case.get("external_id") or "")
            current = (
                self.storage.verified_external_resource_by_key(source_name, external_id)
                if source_name and external_id
                else None
            )
            available = bool(
                current
                and (source_name != "github" or is_public_github_record(current))
            )
        return "available" if available else "unavailable"

    def _default_context(
        self,
        user: dict[str, Any],
        *,
        course_id: int | None = None,
        class_id: int | None = None,
    ) -> dict[str, int | None]:
        classes = self.storage.classes_for_user(int(user["id"]))
        if course_id is not None:
            classes = [item for item in classes if int(item["course_id"]) == int(course_id)]
        if class_id is not None:
            classes = [item for item in classes if int(item["id"]) == int(class_id)]
        if not classes:
            raise InnovationError("COURSE_FORBIDDEN", "当前账号没有可用于创新会话的课程。")
        if course_id is None and len({int(item["course_id"]) for item in classes}) > 1:
            raise InnovationError("COURSE_REQUIRED", "账号属于多个课程，请显式选择 course_id。")
        if class_id is None and len(classes) > 1:
            raise InnovationError("CLASS_REQUIRED", "账号在当前课程属于多个班级，请显式选择 class_id。")
        selected = classes[0]
        return {"course_id": int(selected["course_id"]), "class_id": int(selected["id"]) if selected.get("id") else None}

    def _record_event(
        self,
        user: dict[str, Any],
        client_session_id: str | None,
        event_type: str,
        topic: str,
        metadata: dict[str, Any],
    ) -> None:
        if not client_session_id:
            return
        try:
            LearningEventService(self.storage).record_event(
                user=user,
                client_session_id=client_session_id,
                event_type=event_type,
                page_name="innovation",
                module=topic,
                metadata=metadata,
            )
        except Exception:
            return


class _KeywordOnlyVectorStore:
    def query(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        raise RuntimeError("innovation uses keyword-only RAG fallback by default")


def _resource_type_from_tool_item(item: dict[str, Any], *, default: str) -> str:
    resource_type = str(item.get("resource_type") or item.get("type") or "").strip()
    if resource_type:
        return resource_type
    if default == "paper" and not _is_verifiable_paper(item):
        return "course_document"
    if default == "standard" and not _is_verifiable_standard(item):
        return "course_document"
    return default


def _is_verifiable_paper(item: dict[str, Any]) -> bool:
    title = str(item.get("title") or "").strip()
    locator = (
        item.get("doi")
        or item.get("arxiv_id")
        or item.get("arxiv")
        or item.get("url")
        or item.get("source_url")
        or item.get("link")
    )
    return bool(title and str(locator or "").strip())


def _is_verifiable_standard(item: dict[str, Any]) -> bool:
    title = str(item.get("title") or "").strip()
    source_name = str(item.get("source_name") or item.get("connector") or "").strip()
    locator = item.get("url") or item.get("source_url") or item.get("link")
    return bool(title and source_name == "standard_source" and str(locator or "").startswith("https://"))


def _external_resource_to_case(
    item: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any] | None:
    resource_type = str(item.get("resource_type") or "").strip()
    if resource_type not in {"paper", "project", "vulnerability", "standard"}:
        return None
    title = str(item.get("title") or "").strip()
    if not title:
        return None
    source_name = str(item.get("source_name") or "").strip()
    if not item.get("provenance_verified"):
        return None
    if resource_type == "paper" and not _is_verifiable_paper(item):
        return None
    return {
        "resource_id": f"external:{item.get('id') or source_name}:{item.get('external_id')}",
        "type": resource_type,
        "title": title,
        "summary": item.get("abstract") or "",
        "source": source_name,
        "source_name": source_name,
        "external_id": item.get("external_id") or "",
        "url": item.get("url") or "",
        "tags": item.get("tags") or [],
        "cache_status": item.get("delivery_mode") or response.get("delivery_mode") or "unknown",
        "origin_kind": item.get("origin_kind") or response.get("origin_kind") or "unknown",
        "delivery_mode": item.get("delivery_mode") or response.get("delivery_mode") or "unknown",
        "fetched_at": item.get("fetched_at") or "",
        "provenance_verified": True,
        "metadata": {
            "delivery_mode": item.get("delivery_mode") or response.get("delivery_mode") or "",
            "origin_kind": item.get("origin_kind") or response.get("origin_kind") or "unknown",
            "provenance_verified": True,
            "connector": response.get("connector") or source_name,
            "fetched_at": item.get("fetched_at") or "",
        },
    }


def _run_awaitable(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:
            result["error"] = exc

    import threading

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")
