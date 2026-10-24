"""Learning event recording and session lifecycle."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from backend.core.config import get_settings
from backend.storage.logger import LearningLogger


class LearningEventType(StrEnum):
    LOGIN = "LOGIN"
    PAGE_VIEW = "PAGE_VIEW"
    RESOURCE_SEARCH = "RESOURCE_SEARCH"
    RESOURCE_OPEN = "RESOURCE_OPEN"
    QUESTION_ASKED = "QUESTION_ASKED"
    PRACTICE_GUIDE_GENERATED = "PRACTICE_GUIDE_GENERATED"
    INNOVATION_SESSION_STARTED = "INNOVATION_SESSION_STARTED"
    INNOVATION_QUESTION_ANSWERED = "INNOVATION_QUESTION_ANSWERED"
    ASSIGNMENT_PROGRESS_UPDATED = "ASSIGNMENT_PROGRESS_UPDATED"
    STUCK_REPORTED = "STUCK_REPORTED"
    FEEDBACK_SUBMITTED = "FEEDBACK_SUBMITTED"


class ServerLearningEventType(StrEnum):
    GENERAL_QUESTION_ASKED = "GENERAL_QUESTION_ASKED"
    PRACTICE_DRAFT_GENERATED = "PRACTICE_DRAFT_GENERATED"
    INNOVATION_SUGGESTION_GENERATED = "INNOVATION_SUGGESTION_GENERATED"


def allowed_event_values() -> set[str]:
    return {item.value for item in LearningEventType}


class LearningEventError(ValueError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class LearningEventService:
    def __init__(self, storage: LearningLogger) -> None:
        self.storage = storage

    def create_session(
        self,
        *,
        user: dict[str, Any],
        client_session_id: str,
        course_id: int | None = None,
        class_id: int | None = None,
    ) -> dict[str, Any]:
        context = self._resolve_context(user=user, course_id=course_id, class_id=class_id)
        existing = self.storage.learning_session_by_client_id(client_session_id)
        if existing and int(existing["user_id"]) != int(user["id"]):
            raise LearningEventError("SESSION_CONFLICT", "client_session_id 已被其他用户使用。")
        if existing:
            if existing.get("ended_at"):
                raise LearningEventError("SESSION_ENDED", "学习会话已经结束，请创建新的会话。")
            self._ensure_payload_matches_session(
                session=existing,
                course_id=context["course_id"],
                class_id=context["class_id"],
                require_exact=True,
            )
            self._authorize_session_context(user=user, session=existing)
            return existing
        return self.storage.create_learning_session_with_login(
            user_id=int(user["id"]),
            course_id=context["course_id"],
            class_id=context["class_id"],
            client_session_id=client_session_id,
            metadata={"source": "session_create"},
            dedupe_key=f"user:{int(user['id'])}:session:{client_session_id}:key:LOGIN:{client_session_id}",
        )

    def heartbeat(self, *, user: dict[str, Any], client_session_id: str) -> dict[str, Any]:
        session = self._require_session(user=user, client_session_id=client_session_id)
        self._authorize_session_context(user=user, session=session)
        return self.storage.heartbeat_learning_session(
            session_id=int(session["id"]),
            idle_timeout_seconds=get_settings().session_idle_timeout_minutes * 60,
        )

    def end_session(self, *, user: dict[str, Any], client_session_id: str) -> dict[str, Any]:
        session = self._require_session(user=user, client_session_id=client_session_id, allow_ended=True)
        self._authorize_session_context(user=user, session=session)
        if session.get("ended_at"):
            return session
        return self.storage.end_learning_session(int(session["id"]))

    def validate_session(
        self,
        *,
        user: dict[str, Any],
        client_session_id: str,
        course_id: int | None,
        class_id: int | None,
    ) -> dict[str, Any]:
        session = self._require_session(user=user, client_session_id=client_session_id)
        self._ensure_payload_matches_session(
            session=session,
            course_id=course_id,
            class_id=class_id,
            require_exact=True,
        )
        self._authorize_session_context(user=user, session=session)
        return session

    def record_event(
        self,
        *,
        user: dict[str, Any],
        client_session_id: str,
        event_type: str,
        page_name: str = "",
        module: str = "",
        resource_id: str | None = None,
        assignment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        course_id: int | None = None,
        class_id: int | None = None,
    ) -> dict[str, Any]:
        if event_type not in allowed_event_values():
            raise LearningEventError("INVALID_EVENT_TYPE", f"不支持的学习事件类型：{event_type}")
        session = self._require_session(user=user, client_session_id=client_session_id)
        self._ensure_payload_matches_session(session=session, course_id=course_id, class_id=class_id)
        context = self._authorize_session_context(user=user, session=session)
        return self.storage.log_learning_event(
            session_id=int(session["id"]),
            user_id=int(user["id"]),
            course_id=context["course_id"],
            class_id=context["class_id"],
            event_type=event_type,
            page_name=page_name,
            module=module,
            resource_id=resource_id,
            assignment_id=assignment_id,
            metadata=metadata or {},
            dedupe_key=self._scoped_dedupe_key(
                user_id=int(user["id"]),
                client_session_id=client_session_id,
                dedupe_key=dedupe_key,
            ),
        )

    def record_server_event(
        self,
        *,
        user: dict[str, Any],
        client_session_id: str,
        event_type: ServerLearningEventType,
        metadata: dict[str, Any],
        course_id: int,
        class_id: int | None,
        dedupe_key: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(event_type, ServerLearningEventType):
            raise LearningEventError("INVALID_EVENT_TYPE", "该事件只能由服务器生成。")
        session = self._require_session(user=user, client_session_id=client_session_id)
        self._ensure_payload_matches_session(
            session=session,
            course_id=course_id,
            class_id=class_id,
            require_exact=True,
        )
        context = self._authorize_session_context(user=user, session=session)
        safe_metadata = {
            key: value
            for key, value in metadata.items()
            if key in {"request_id", "mode", "status", "reference_status", "reference_count", "failure_category"}
        }
        is_practice_draft = event_type == ServerLearningEventType.PRACTICE_DRAFT_GENERATED
        is_innovation = event_type == ServerLearningEventType.INNOVATION_SUGGESTION_GENERATED
        page_name = "innovation" if is_innovation else ("practice_draft" if is_practice_draft else "general_learning")
        module = "innovation_suggestion" if is_innovation else ("practice_draft" if is_practice_draft else "general_chat")
        return self.storage.log_learning_event(
            session_id=int(session["id"]),
            user_id=int(user["id"]),
            course_id=int(context["course_id"]),
            class_id=context["class_id"],
            event_type=event_type.value,
            page_name=page_name,
            module=module,
            metadata=safe_metadata,
            dedupe_key=self._scoped_dedupe_key(
                user_id=int(user["id"]),
                client_session_id=client_session_id,
                dedupe_key=dedupe_key,
            ),
        )

    @staticmethod
    def _scoped_dedupe_key(
        *,
        user_id: int,
        client_session_id: str,
        dedupe_key: str | None,
    ) -> str | None:
        if not dedupe_key:
            return None
        return f"user:{user_id}:session:{client_session_id}:key:{dedupe_key}"

    def _require_session(
        self,
        *,
        user: dict[str, Any],
        client_session_id: str,
        allow_ended: bool = False,
    ) -> dict[str, Any]:
        session = self.storage.learning_session_by_client_id(client_session_id)
        if not session or int(session["user_id"]) != int(user["id"]):
            raise LearningEventError("SESSION_NOT_FOUND", "学习会话不存在或不属于当前用户。")
        if session.get("ended_at") and not allow_ended:
            raise LearningEventError("SESSION_ENDED", "学习会话已经结束，请创建新的会话。")
        return session

    def _authorize_session_context(self, *, user: dict[str, Any], session: dict[str, Any]) -> dict[str, int | None]:
        return self._resolve_context(
            user=user,
            course_id=int(session["course_id"]) if session.get("course_id") is not None else None,
            class_id=int(session["class_id"]) if session.get("class_id") is not None else None,
        )

    @staticmethod
    def _ensure_payload_matches_session(
        *,
        session: dict[str, Any],
        course_id: int | None,
        class_id: int | None,
        require_exact: bool = False,
    ) -> None:
        session_course_id = int(session["course_id"]) if session.get("course_id") is not None else None
        session_class_id = int(session["class_id"]) if session.get("class_id") is not None else None
        if course_id is not None and int(course_id) != session_course_id:
            raise LearningEventError("SESSION_CONTEXT_CONFLICT", "事件课程必须与已有学习会话一致。")
        if class_id is not None and int(class_id) != session_class_id:
            raise LearningEventError("SESSION_CONTEXT_CONFLICT", "事件班级必须与已有学习会话一致。")
        if require_exact and (course_id != session_course_id or class_id != session_class_id):
            raise LearningEventError("SESSION_CONTEXT_CONFLICT", "请求作用域必须与已有学习会话完全一致。")

    def _resolve_context(
        self,
        *,
        user: dict[str, Any],
        course_id: int | None,
        class_id: int | None,
    ) -> dict[str, int | None]:
        role = str(user["role"]).lower()
        if role == "admin":
            classes = self.storage.classes(course_id=course_id)
        elif role == "teacher":
            classes = self.storage.classes_for_teacher(int(user["id"]), course_id=course_id)
        else:
            classes = self.storage.classes_for_user(int(user["id"]), course_id=course_id)

        visible_course_ids = {int(item["course_id"]) for item in classes}
        if course_id is None:
            if not classes:
                raise LearningEventError("NO_CLASS_CONTEXT", "当前账号没有可写入的课程班级。")
            if len(visible_course_ids) > 1:
                raise LearningEventError("COURSE_REQUIRED", "账号属于多个课程，请显式选择 course_id。")
            course_id = next(iter(visible_course_ids))
        classes = [item for item in classes if int(item["course_id"]) == int(course_id)]

        if class_id is not None:
            allowed = [item for item in classes if int(item["id"]) == int(class_id)]
            if not allowed:
                raise LearningEventError("CLASS_FORBIDDEN", "无权向该班级写入学习事件。")
            selected = allowed[0]
            if course_id is not None and int(selected["course_id"]) != int(course_id):
                raise LearningEventError("CONTEXT_CONFLICT", "课程与班级不匹配。")
        elif classes:
            if len(classes) > 1:
                raise LearningEventError("CLASS_REQUIRED", "账号在当前课程属于多个班级，请显式选择 class_id。")
            selected = classes[0]
        else:
            raise LearningEventError("COURSE_FORBIDDEN", "无权向该课程写入学习事件。")
        return {
            "course_id": int(selected["course_id"]),
            "class_id": int(selected["id"]) if selected.get("id") is not None else None,
        }
