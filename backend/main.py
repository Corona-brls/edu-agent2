from pathlib import Path
from contextlib import asynccontextmanager
import asyncio
from dataclasses import dataclass
from collections import deque
import math
import os
import re
import threading
import time
import uuid
from typing import Annotated, Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.router import route_intent
from backend.agent.schemas import AgentIntent
from backend.agent.state import state_from_record
from backend.analytics.activity import ActivityService
from backend.analytics.clustering import ClusterBuildOptions, QuestionClusteringService
from backend.analytics.events import LearningEventError, LearningEventService, ServerLearningEventType
from backend.analytics.stuck_analysis import StuckAnalysisService
from backend.api.teacher_phase3 import router as teacher_phase3_router
from backend.auth.dependencies import get_current_user, get_optional_current_user, require_roles
from backend.auth.schemas import ChangePasswordRequest, ChangePasswordResponse, LoginRequest, LoginResponse, LogoutResponse, MeResponse, UserPublic
from backend.auth.service import AuthError, authenticate_user, change_password, create_access_token, decode_access_token, public_user
from backend.courses.schemas import ClassPublic, CoursePublic
from backend.courses.service import visible_classes, visible_courses
from backend.connectors.registry import get_default_registry
from backend.connectors.schemas import ResourceQuery
from backend.innovation.service import InnovationError, InnovationService
from backend.innovation.model_advisor import InnovationAdvisorService
from backend.innovation.schemas import InnovationSuggestionRequest, InnovationSuggestionResponse
from backend.resources.portal import RESOURCE_TYPES, ResourcePortalQuery, ResourcePortalService
from backend.resources.service import ExternalResourceService
from backend.resources.query_intent import LookupInputError
from backend.rag.answerer import CourseAnswerer
from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.document_locks import DocumentBusyError, document_locks
from backend.rag.visibility import publication_projection
from backend.rag.document_parser import (
    DEFAULT_MAX_FILE_BYTES,
    SUPPORTED_SUFFIXES,
    DocumentParseError,
    parse_document,
    validate_upload_filename,
)
from backend.schemas import (
    AgentContinueRequest,
    AgentRunResponse,
    AssignmentProgressRequest,
    ChatRequest,
    ChatResponse,
    ClusterMergeRequest,
    ClusterRebuildRequest,
    ClusterRenameRequest,
    ClusterSplitRequest,
    FAQResponse,
    FeedbackRequest,
    FeedbackResponse,
    ActivitySummaryResponse,
    KnowledgeActionResponse,
    KnowledgeDocumentsResponse,
    InnovationAnswerRequest,
    InnovationFinalizeRequest,
    InnovationStartRequest,
    LearningEventRequest,
    LearningEventResponse,
    LearningSessionRequest,
    LearningSessionResponse,
    LearningTimelineResponse,
    PracticeGuideRequest,
    PracticeGuideResponse,
    ResourceOpenRequest,
    StuckActionRequest,
    StuckReportRequest,
    SystemStatusResponse,
    TeacherAnalyticsResponse,
    ToolInfo,
    UploadResponse,
)
from backend.core.config import get_settings
from backend.rag.vector_store import ChromaVectorStore
from backend.storage.logger import LearningLogger, sha256_bytes
from backend.tools.mcp_client import MCPClientGateway
from backend.llm.factory import create_llm_provider
from backend.llm.template_provider import TemplateProvider
from backend.llm.deepseek_provider import DeepSeekProvider
from backend.learning.errors import GeneralLearningError
from backend.learning.general_service import GeneralLearningService
from backend.learning.runtime import LearningRuntime
from backend.learning.schemas import GeneralChatRequest, GeneralChatResponse
from backend.practice.exploratory_service import ExploratoryPracticeService
from backend.practice.schemas import PracticeDraftRequest, PracticeDraftResponse
from backend.llm.data_policy import (
    ExternalModelDataPolicy,
    ModelCallPreflightError,
    ModelContextNotAuthorized,
    validate_current_course_evidence,
)

KNOWLEDGE_BASE_DIR = get_settings().knowledge_base_dir
ALLOWED_UPLOAD_SUFFIXES = SUPPORTED_SUFFIXES
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
_COURSE_REFERENCE_STATUSES = frozenset(
    {"allowed", "disabled_by_user", "blocked_by_policy", "revoked"}
)

answerer = CourseAnswerer(KNOWLEDGE_BASE_DIR)
learning_logger = LearningLogger()
mcp_gateway = MCPClientGateway(mcp_timeout_seconds=get_settings().mcp_tool_timeout_seconds)
connector_registry = get_default_registry()
question_clustering_service_factory = QuestionClusteringService
_auto_cluster_new_question_counts: dict[tuple[int | None, int | None], int] = {}
_RESOURCE_RATE_LIMIT_MAX_CALLS = 20
_RESOURCE_RATE_LIMIT_WINDOW_SECONDS = 60.0
_RESOURCE_RATE_BUCKETS: dict[tuple[int, str], deque[float]] = {}
_RESOURCE_RATE_LOCK = threading.Lock()
general_learning_runtime = LearningRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _recover_auto_question_cluster_jobs()
    mcp_gateway.start()
    try:
        yield
    finally:
        mcp_gateway.stop()


app = FastAPI(title="Course Agent API", lifespan=lifespan)
app.include_router(teacher_phase3_router)


def _require_teacher_api_access(current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))]) -> dict:
    return current_user


def _require_admin_api_access(current_user: Annotated[dict, Depends(require_roles("admin"))]) -> dict:
    return current_user


def _resolve_teacher_scope(
    current_user: dict,
    *,
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict[str, object]:
    role = str(current_user["role"]).lower()
    if role == "teacher":
        classes = learning_logger.classes_for_teacher(int(current_user["id"]), course_id=course_id)
        if not classes:
            raise HTTPException(
                status_code=403,
                detail={"error_code": "COURSE_FORBIDDEN", "message": "教师没有可访问的任课课程或班级。"},
            )
        if class_id is not None:
            classes = [item for item in classes if int(item["id"]) == int(class_id)]
            if not classes:
                raise HTTPException(
                    status_code=403,
                    detail={"error_code": "CLASS_FORBIDDEN", "message": "教师只能访问自己任课班级。"},
                )
        selected_course_id = course_id
        if selected_course_id is None and classes:
            selected_course_id = int(classes[0]["course_id"])
        return {
            "course_id": selected_course_id,
            "class_ids": [int(item["id"]) for item in classes],
            "classes": classes,
        }

    classes = learning_logger.classes(course_id=course_id)
    if class_id is not None:
        classes = [item for item in classes if int(item["id"]) == int(class_id)]
    return {
        "course_id": course_id,
        "class_ids": [int(item["id"]) for item in classes],
        "classes": classes,
    }


def _resolve_knowledge_scope(
    current_user: dict | None,
    course_id: int | None = None,
    *,
    require_course: bool = True,
) -> dict[str, object]:
    if current_user is None:
        if course_id is None and require_course:
            return {
                "course_id": learning_logger.default_course_id(),
                "class_id": learning_logger.default_class_id(),
            }
        return {"course_id": course_id, "class_id": None}
    role = str(current_user["role"]).lower()
    if role == "admin":
        if course_id is None:
            if require_course:
                raise HTTPException(
                    status_code=400,
                    detail={"error_code": "COURSE_REQUIRED", "message": "请显式选择要管理的课程。"},
                )
            return {"course_id": None, "class_id": None}
        if int(course_id) not in {int(item["id"]) for item in learning_logger.courses()}:
            raise HTTPException(
                status_code=404,
                detail={"error_code": "COURSE_NOT_FOUND", "message": "课程不存在。"},
            )
        return {"course_id": int(course_id), "class_id": None}
    classes = learning_logger.classes_for_teacher(int(current_user["id"]), course_id=course_id)
    if not classes:
        raise HTTPException(
            status_code=403,
            detail={"error_code": "COURSE_FORBIDDEN", "message": "教师没有可管理的任课课程。"},
        )
    course_ids = {int(item["course_id"]) for item in classes}
    if course_id is None and len(course_ids) > 1:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "COURSE_REQUIRED", "message": "教师管理多个课程，请显式选择 course_id。"},
        )
    selected_course_id = int(course_id) if course_id is not None else next(iter(course_ids))
    return {"course_id": selected_course_id, "class_id": None}


def _require_document_management_access(document: dict, current_user: dict | None) -> None:
    if current_user is None:
        return
    role = str(current_user["role"]).lower()
    if role == "admin":
        return
    if role == "teacher" and document.get("course_id") is not None:
        if learning_logger.classes_for_teacher(int(current_user["id"]), course_id=int(document["course_id"])):
            return
    raise HTTPException(
        status_code=403,
        detail={"error_code": "COURSE_FORBIDDEN", "message": "无权管理该课程的知识库文档。"},
    )


def _user_can_access_course(current_user: dict, course_id: int | None) -> bool:
    if course_id is None:
        return True
    role = str(current_user["role"]).lower()
    if role == "admin":
        return True
    if role == "teacher":
        return bool(learning_logger.classes_for_teacher(int(current_user["id"]), course_id=course_id))
    return bool(learning_logger.classes_for_user(int(current_user["id"]), course_id=course_id))


def _resolve_user_course_context(
    current_user: dict,
    *,
    course_id: int | None = None,
    class_id: int | None = None,
    require_course: bool = True,
    require_class: bool = False,
) -> dict[str, int | None]:
    role = str(current_user["role"]).lower()
    if role == "admin":
        if course_id is None:
            if require_course:
                raise HTTPException(
                    status_code=400,
                    detail={"error_code": "COURSE_REQUIRED", "message": "请显式选择 course_id。"},
                )
            return {"course_id": None, "class_id": None}
        classes = learning_logger.classes(course_id=course_id)
    elif role == "teacher":
        classes = learning_logger.classes_for_teacher(int(current_user["id"]), course_id=course_id)
    else:
        classes = learning_logger.classes_for_user(int(current_user["id"]), course_id=course_id)

    if not classes:
        raise HTTPException(
            status_code=403,
            detail={"error_code": "COURSE_FORBIDDEN", "message": "当前账号没有可访问的课程。"},
        )
    visible_course_ids = {int(item["course_id"]) for item in classes}
    if course_id is None and len(visible_course_ids) > 1:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "COURSE_REQUIRED", "message": "账号属于多个课程，请显式选择 course_id。"},
        )
    selected_course_id = int(course_id) if course_id is not None else next(iter(visible_course_ids))
    classes = [item for item in classes if int(item["course_id"]) == selected_course_id]
    if class_id is not None:
        classes = [item for item in classes if int(item["id"]) == int(class_id)]
        if not classes:
            raise HTTPException(
                status_code=403,
                detail={"error_code": "CLASS_FORBIDDEN", "message": "无权访问该班级或班级不属于当前课程。"},
            )
        return {"course_id": selected_course_id, "class_id": int(class_id)}
    if require_class and len(classes) > 1:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "CLASS_REQUIRED", "message": "账号在当前课程属于多个班级，请显式选择 class_id。"},
        )
    return {
        "course_id": selected_course_id,
        "class_id": int(classes[0]["id"]) if require_class and classes else None,
    }


def _require_assignment_scope(assignment: dict | None, current_user: dict) -> dict:
    if assignment is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "ASSIGNMENT_NOT_FOUND", "message": "实验任务不存在。"},
        )
    if not _user_can_access_course(current_user, assignment.get("course_id")):
        raise HTTPException(
            status_code=403,
            detail={"error_code": "COURSE_FORBIDDEN", "message": "无权访问该课程的实验任务。"},
        )
    assignment_class_id = assignment.get("class_id")
    if assignment_class_id is not None:
        role = str(current_user.get("role") or "").lower()
        if role == "teacher":
            allowed = learning_logger.teacher_in_class(
                int(current_user["id"]),
                int(assignment_class_id),
                int(assignment["course_id"]),
            )
        elif role == "student":
            allowed = learning_logger.user_in_class(
                int(current_user["id"]),
                int(assignment_class_id),
                int(assignment["course_id"]),
            )
        else:
            allowed = role == "admin"
        if not allowed:
            raise HTTPException(
                status_code=403,
                detail={"error_code": "CLASS_FORBIDDEN", "message": "无权访问该班级的实验任务。"},
            )
    return assignment


def _learning_error_status(error_code: str) -> int:
    if error_code in {"INVALID_EVENT_TYPE", "COURSE_REQUIRED", "CLASS_REQUIRED"}:
        return 400
    if error_code in {"SESSION_CONFLICT", "SESSION_CONTEXT_CONFLICT", "SESSION_ENDED", "CONTEXT_CONFLICT"}:
        return 409
    if error_code == "SESSION_NOT_FOUND":
        return 404
    return 403


def _innovation_error_status(error_code: str, *, missing_status: int = 400) -> int:
    if error_code in {"SESSION_CONTEXT_CONFLICT", "SESSION_ENDED", "INNOVATION_SESSION_FINALIZED", "CLIENT_DISCONNECTED"}:
        return 409
    if error_code in {"USER_DISABLED", "TOKEN_REVOKED"}:
        return 401
    if error_code in {"COURSE_REQUIRED", "CLASS_REQUIRED", "INVALID_INNOVATION_REQUEST", "EMPTY_INNOVATION_ANSWER"}:
        return 400
    if error_code in {"INNOVATION_SESSION_NOT_FOUND", "INNOVATION_QUESTION_NOT_FOUND"}:
        return 404
    if error_code.endswith("FORBIDDEN") or error_code in {"COURSE_FORBIDDEN", "SESSION_CONTEXT_REQUIRED"}:
        return 403
    return missing_status


def _validate_request_scope(
    *,
    current_user: dict,
    course_id: int,
    class_id: int | None,
    assignment_id: int | None = None,
    step_id: int | None = None,
    client_session_id: str | None = None,
) -> None:
    assignment = None
    if assignment_id is not None:
        assignment = _require_assignment_scope(learning_logger.assignment(int(assignment_id)), current_user)
    if step_id is not None:
        step = learning_logger.assignment_step(int(step_id))
        if step is None:
            raise HTTPException(
                status_code=404,
                detail={"error_code": "ASSIGNMENT_STEP_NOT_FOUND", "message": "实验步骤不存在。"},
            )
        if assignment is None:
            assignment = _require_assignment_scope(
                learning_logger.assignment(int(step["assignment_id"])), current_user
            )
        elif int(step["assignment_id"]) != int(assignment["id"]):
            raise HTTPException(
                status_code=409,
                detail={"error_code": "STEP_CONTEXT_CONFLICT", "message": "实验步骤不属于所选任务。"},
            )
    if assignment is not None:
        if int(assignment.get("course_id") or 0) != int(course_id):
            raise HTTPException(
                status_code=409,
                detail={"error_code": "ASSIGNMENT_CONTEXT_CONFLICT", "message": "实验任务不属于当前课程。"},
            )
        assignment_class_id = assignment.get("class_id")
        if assignment_class_id is not None and (
            class_id is None or int(assignment_class_id) != int(class_id)
        ):
            raise HTTPException(
                status_code=409,
                detail={"error_code": "ASSIGNMENT_CONTEXT_CONFLICT", "message": "实验任务不属于当前班级。"},
            )
    if client_session_id:
        try:
            LearningEventService(learning_logger).validate_session(
                user=current_user,
                client_session_id=client_session_id,
                course_id=int(course_id),
                class_id=class_id,
            )
        except LearningEventError as exc:
            raise HTTPException(
                status_code=_learning_error_status(exc.error_code),
                detail={"error_code": exc.error_code, "message": str(exc)},
            ) from exc


def _scope_from_learning_session(
    *,
    current_user: dict,
    client_session_id: str | None,
    course_id: int | None,
    class_id: int | None,
) -> tuple[int | None, int | None]:
    if not client_session_id:
        return course_id, class_id
    session = learning_logger.learning_session_by_client_id(client_session_id)
    if not session or int(session.get("user_id") or 0) != int(current_user["id"]):
        return course_id, class_id
    if course_id is None and session.get("course_id") is not None:
        course_id = int(session["course_id"])
    if class_id is None and session.get("class_id") is not None:
        class_id = int(session["class_id"])
    return course_id, class_id


def _configure_external_model_policy(
    provider: object,
    *,
    course_id: int,
    use_course_references: bool,
    require_course_context: bool,
    currently_authorized: bool,
) -> str:
    configure = getattr(provider, "configure_course_policy", None)
    if not callable(configure):
        return "local"
    status = configure(
        course_id=course_id,
        use_course_references=use_course_references,
        currently_authorized=currently_authorized,
    )
    if require_course_context and status != "allowed":
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": ModelContextNotAuthorized.error_code,
                "message": "课程资料未获本次外部模型处理授权，请确认课程许可并在本次请求中明确选择。",
                "course_reference_status": status,
            },
        )
    return str(status)


@dataclass
class _RequestModelCallGuard:
    provider: object
    user_id: int
    course_id: int
    class_id: int | None
    client_session_id: str | None
    use_course_references: bool
    require_course_context: bool
    last_policy_status: str = "local"

    def check_request_start(self) -> str:
        return self(
            provider=self.provider,
            state=None,
            request=None,
            phase="request",
            stage="before",
        )

    def __call__(
        self,
        *,
        provider: object,
        state: object | None,
        request: object | None,
        phase: str,
        stage: str,
    ) -> str:
        del phase, stage
        fresh_user = learning_logger.user(int(self.user_id))
        state_matches = state is None or (
            str(getattr(state, "user_id", "")) == str(self.user_id)
            and getattr(state, "course_id", None) is not None
            and int(getattr(state, "course_id")) == int(self.course_id)
            and _optional_int(getattr(state, "class_id", None)) == self.class_id
        )
        currently_authorized = bool(
            fresh_user
            and bool(fresh_user.get("enabled", 0))
            and state_matches
            and _user_has_run_scope(
                fresh_user,
                course_id=int(self.course_id),
                class_id=self.class_id,
            )
        )
        self.last_policy_status = _configure_external_model_policy(
            provider,
            course_id=int(self.course_id),
            use_course_references=self.use_course_references,
            require_course_context=False,
            currently_authorized=currently_authorized,
        )
        if not currently_authorized:
            raise ModelCallPreflightError(
                "MODEL_CONTEXT_REVOKED",
                "当前账号已无权继续使用该课程模型上下文。",
                policy_status="revoked",
            )
        if self.client_session_id:
            try:
                LearningEventService(learning_logger).validate_session(
                    user=fresh_user,
                    client_session_id=self.client_session_id,
                    course_id=int(self.course_id),
                    class_id=self.class_id,
                )
            except LearningEventError as exc:
                self._mark_revoked(provider)
                raise ModelCallPreflightError(
                    exc.error_code,
                    str(exc),
                    status_code=_learning_error_status(exc.error_code),
                    policy_status="revoked",
                ) from exc
        evidence = (
            list(getattr(request, "evidence", []) or [])
            if request is not None
            else list(getattr(state, "evidence", []) or [])
        )
        try:
            validate_current_course_evidence(
                learning_logger,
                evidence,
                course_id=int(self.course_id),
            )
        except ModelCallPreflightError:
            self._mark_revoked(provider)
            raise
        if self.require_course_context and self.last_policy_status not in {"allowed", "local"}:
            raise ModelCallPreflightError(
                ModelContextNotAuthorized.error_code,
                "课程资料未获本次外部模型处理授权，请确认课程许可并在本次请求中明确选择。",
                policy_status=self.last_policy_status,
            )
        return self.last_policy_status

    def _mark_revoked(self, provider: object) -> None:
        self.last_policy_status = _configure_external_model_policy(
            provider,
            course_id=int(self.course_id),
            use_course_references=self.use_course_references,
            require_course_context=False,
            currently_authorized=False,
        )


def _optional_int(value: object | None) -> int | None:
    return int(value) if value is not None else None


def _raise_model_preflight_http(exc: ModelCallPreflightError) -> None:
    raise HTTPException(
        status_code=exc.status_code,
        detail={
            "error_code": exc.error_code,
            "message": str(exc),
            "course_reference_status": exc.policy_status,
            "run_id": getattr(exc, "run_id", None),
        },
    ) from exc


def _close_rejected_orchestrator_attempt(
    orchestrator: object,
    exc: ModelCallPreflightError,
) -> None:
    close_attempt = getattr(orchestrator, "close_rejected_attempt", None)
    if callable(close_attempt):
        close_attempt(exc)


def _check_agent_result_before_business_write(
    *,
    model_guard: _RequestModelCallGuard,
    provider: object,
    result: dict[str, Any],
    orchestrator: AgentOrchestrator | None = None,
) -> None:
    """Re-read the just-persisted run so the route-level check includes its evidence."""

    run_id = str(result.get("run_id") or "")
    record = learning_logger.agent_run(run_id) if run_id else None
    if record is None or str(record.get("context_status") or "invalid") != "valid":
        raise ModelCallPreflightError(
            "RUN_CONTEXT_REQUIRED",
            "Agent Run 缺少可重新核验的可信上下文。",
        )
    state = state_from_record(record)
    if orchestrator is not None:
        orchestrator.validate_learning_context(state)
    model_guard(
        provider=provider,
        state=state,
        request=None,
        phase="response",
        stage="before_business_write",
    )


def _intent_requires_course_context(intent: str) -> bool:
    return intent in {AgentIntent.COURSE_QA.value, AgentIntent.PRACTICE_GUIDE.value}


def _cluster_by_id(cluster_id: int) -> dict | None:
    for cluster in learning_logger.question_clusters():
        if int(cluster["id"]) == int(cluster_id):
            return cluster
    return None


def _require_cluster_scope(cluster_id: int, current_user: dict) -> dict:
    cluster = _cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "CLUSTER_NOT_FOUND", "message": f"问题簇不存在：{cluster_id}"},
        )
    role = str(current_user["role"]).lower()
    if role == "admin":
        return cluster
    course_id = cluster.get("course_id")
    class_id = cluster.get("class_id")
    if course_id is None or class_id is None:
        raise HTTPException(
            status_code=403,
            detail={"error_code": "COURSE_FORBIDDEN", "message": "无权访问该课程的问题簇。"},
        )
    allowed_classes = learning_logger.classes_for_teacher(int(current_user["id"]), course_id=int(course_id))
    allowed_class_ids = {int(item["id"]) for item in allowed_classes}
    if int(class_id) not in allowed_class_ids:
        raise HTTPException(
            status_code=403,
            detail={"error_code": "CLASS_FORBIDDEN", "message": "无权访问该班级的问题簇。"},
        )
    member_class_ids = {int(member["class_id"]) for member in cluster.get("members", []) if member.get("class_id") is not None}
    if not member_class_ids or member_class_ids - allowed_class_ids or member_class_ids != {int(class_id)}:
        raise HTTPException(
            status_code=403,
            detail={"error_code": "CLASS_FORBIDDEN", "message": "问题簇成员不完全属于当前授权班级。"},
        )
    return cluster


async def _read_upload_file_limited(file: UploadFile, max_bytes: int | None = None) -> bytes:
    max_bytes = max_bytes or DEFAULT_MAX_FILE_BYTES
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "error_code": "UPLOAD_TOO_LARGE",
                    "message": f"上传文件超过大小限制：{max_bytes} 字节。",
                },
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict):
        payload = {
            "error_code": exc.detail.get("error_code", "HTTP_ERROR"),
            "message": exc.detail.get("message", str(exc.detail)),
            "run_id": exc.detail.get("run_id"),
            "trace_id": exc.detail.get("trace_id"),
        }
        course_reference_status = exc.detail.get("course_reference_status")
        if course_reference_status in _COURSE_REFERENCE_STATUSES:
            payload["course_reference_status"] = course_reference_status
        request_id = exc.detail.get("request_id")
        if isinstance(request_id, str) and request_id:
            payload["request_id"] = request_id
        retryable = exc.detail.get("retryable")
        if isinstance(retryable, bool):
            payload["retryable"] = retryable
    else:
        payload = {
            "error_code": "HTTP_ERROR",
            "message": str(exc.detail),
            "run_id": None,
            "trace_id": None,
        }
    if _is_learning_model_path(request.url.path):
        payload.setdefault("request_id", str(uuid.uuid4()))
        payload.setdefault("retryable", False)
    trusted_headers = {
        key: value
        for key, value in (exc.headers or {}).items()
        if key.lower() in {"retry-after", "www-authenticate"}
    }
    return JSONResponse(status_code=exc.status_code, content=payload, headers=trusted_headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    if not _is_learning_model_path(request.url.path):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
    locations = []
    for error in exc.errors():
        location = [str(part) for part in error.get("loc", ()) if part != "body"]
        if location:
            locations.append(".".join(location)[:120])
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "VALIDATION_ERROR",
            "message": "请求字段不符合要求。",
            "request_id": str(uuid.uuid4()),
            "retryable": False,
            "fields": locations[:8],
        },
    )


def _is_learning_model_path(path: str) -> bool:
    return path in {"/api/learning/general-chat", "/api/practice/drafts"} or (
        path.startswith("/api/innovation/sessions/") and path.endswith("/suggestions")
    )


@app.middleware("http")
async def must_change_password_gate(request: Request, call_next):
    allowed_paths = {
        "/api/auth/login",
        "/api/auth/me",
        "/api/auth/change-password",
        "/api/auth/logout",
        "/api/health",
    }
    path = request.url.path
    authorization = request.headers.get("Authorization", "")
    if path.startswith("/api/") and path not in allowed_paths and authorization.lower().startswith("bearer "):
        token = authorization.partition(" ")[2]
        try:
            payload = decode_access_token(token)
            user = learning_logger.user(int(payload["sub"]))
        except (AuthError, ValueError):
            user = None
        if user and bool(user.get("must_change_password", 0)):
            return JSONResponse(
                status_code=403,
                content={
                    "error_code": "PASSWORD_CHANGE_REQUIRED",
                    "message": "首次登录必须先修改默认密码。",
                    "run_id": None,
                    "trace_id": None,
                },
            )
    return await call_next(request)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "course-agent"}


@app.post("/api/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> dict:
    if get_settings().enable_demo_accounts:
        learning_logger.seed_demo_identity()
    try:
        user = authenticate_user(learning_logger, payload.username, payload.password)
        token, expires_in = create_access_token(user)
    except AuthError as exc:
        raise HTTPException(
            status_code=401,
            detail={"error_code": exc.error_code, "message": str(exc)},
        ) from exc
    must_change = bool(user.get("must_change_password", 0))
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": expires_in,
        "user": public_user(user),
        "must_change_password": must_change,
        "message": "本地演示账号使用默认密码，请首次登录后修改。" if must_change else "登录成功。",
    }


@app.post("/api/auth/change-password", response_model=ChangePasswordResponse)
def change_own_password(
    payload: ChangePasswordRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    try:
        user = change_password(
            learning_logger,
            int(current_user["id"]),
            payload.current_password,
            payload.new_password,
        )
    except AuthError as exc:
        status = 400 if exc.error_code == "PASSWORD_UNCHANGED" else 401
        raise HTTPException(
            status_code=status,
            detail={"error_code": exc.error_code, "message": str(exc)},
        ) from exc
    token, expires_in = create_access_token(user)
    return {
        "changed": True,
        "must_change_password": bool(user.get("must_change_password", 0)),
        "access_token": token,
        "expires_in": expires_in,
        "user": public_user(user),
        "message": "密码已更新，请使用新密码登录。",
    }


@app.get("/api/auth/me", response_model=MeResponse)
def me(current_user: Annotated[dict, Depends(get_current_user)]) -> dict:
    return {"user": public_user(current_user)}


@app.post("/api/auth/logout", response_model=LogoutResponse)
def logout(current_user: Annotated[dict, Depends(get_current_user)]) -> dict:
    learning_logger.revoke_user_tokens(int(current_user["id"]))
    return {"logged_out": True, "message": "已退出登录，当前访问令牌已失效。"}


@app.get("/api/courses", response_model=list[CoursePublic])
def courses(current_user: Annotated[dict, Depends(get_current_user)]) -> list[dict]:
    return visible_courses(learning_logger, current_user)


@app.get("/api/classes", response_model=list[ClassPublic])
def classes(
    current_user: Annotated[dict, Depends(get_current_user)],
    course_id: int | None = None,
) -> list[dict]:
    return visible_classes(learning_logger, current_user, course_id=course_id)


@app.post("/api/learning/sessions", response_model=LearningSessionResponse)
def create_learning_session(
    payload: LearningSessionRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    try:
        return LearningEventService(learning_logger).create_session(
            user=current_user,
            client_session_id=payload.client_session_id,
            course_id=payload.course_id,
            class_id=payload.class_id,
        )
    except LearningEventError as exc:
        raise HTTPException(status_code=_learning_error_status(exc.error_code), detail={"error_code": exc.error_code, "message": str(exc)}) from exc


@app.post("/api/learning/sessions/{client_session_id}/heartbeat", response_model=LearningSessionResponse)
def heartbeat_learning_session(
    client_session_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    try:
        return LearningEventService(learning_logger).heartbeat(
            user=current_user,
            client_session_id=client_session_id,
        )
    except LearningEventError as exc:
        raise HTTPException(status_code=_learning_error_status(exc.error_code), detail={"error_code": exc.error_code, "message": str(exc)}) from exc


@app.post("/api/learning/sessions/{client_session_id}/end", response_model=LearningSessionResponse)
def end_learning_session(
    client_session_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    try:
        return LearningEventService(learning_logger).end_session(
            user=current_user,
            client_session_id=client_session_id,
        )
    except LearningEventError as exc:
        raise HTTPException(status_code=_learning_error_status(exc.error_code), detail={"error_code": exc.error_code, "message": str(exc)}) from exc


@app.post("/api/learning/events", response_model=LearningEventResponse)
def record_learning_event(
    payload: LearningEventRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    try:
        assignment_id = None
        if payload.assignment_id is not None:
            try:
                assignment_id = int(payload.assignment_id)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail={"error_code": "ASSIGNMENT_CONTEXT_INVALID", "message": "assignment_id 必须是整数。"},
                ) from exc
        raw_step_id = payload.metadata.get("step_id")
        step_id = None
        if raw_step_id is not None:
            try:
                step_id = int(raw_step_id)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail={"error_code": "ASSIGNMENT_STEP_INVALID", "message": "step_id 必须是整数。"},
                ) from exc
        if assignment_id is not None or step_id is not None:
            session = learning_logger.learning_session_by_client_id(payload.client_session_id)
            if session and int(session.get("user_id") or 0) == int(current_user["id"]):
                _validate_request_scope(
                    current_user=current_user,
                    course_id=int(session["course_id"]),
                    class_id=int(session["class_id"]) if session.get("class_id") is not None else None,
                    assignment_id=assignment_id,
                    step_id=step_id,
                    client_session_id=payload.client_session_id,
                )
        return LearningEventService(learning_logger).record_event(
            user=current_user,
            client_session_id=payload.client_session_id,
            event_type=payload.event_type,
            page_name=payload.page_name,
            module=payload.module,
            resource_id=payload.resource_id,
            assignment_id=payload.assignment_id,
            metadata=payload.metadata,
            dedupe_key=payload.dedupe_key,
            course_id=payload.course_id,
            class_id=payload.class_id,
        )
    except LearningEventError as exc:
        status = _learning_error_status(exc.error_code)
        raise HTTPException(status_code=status, detail={"error_code": exc.error_code, "message": str(exc)}) from exc


def _create_general_provider() -> DeepSeekProvider:
    return DeepSeekProvider(settings=get_settings())


def _create_practice_draft_provider() -> DeepSeekProvider:
    return DeepSeekProvider(settings=get_settings())


def _create_innovation_suggestion_provider() -> DeepSeekProvider:
    return DeepSeekProvider(settings=get_settings())


async def _watch_general_disconnect(request: Request, disconnected: threading.Event) -> None:
    try:
        while not disconnected.is_set():
            if await request.is_disconnected():
                disconnected.set()
                return
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        raise


@app.post("/api/learning/general-chat", response_model=GeneralChatResponse)
async def general_chat(
    payload: GeneralChatRequest,
    request: Request,
    current_user: Annotated[dict, Depends(require_roles("student"))],
) -> dict[str, Any]:
    request_id = str(uuid.uuid4())
    requested_course_id, requested_class_id = _scope_from_learning_session(
        current_user=current_user,
        client_session_id=payload.client_session_id,
        course_id=payload.course_id,
        class_id=payload.class_id,
    )
    course_context = _resolve_user_course_context(
        current_user,
        course_id=requested_course_id,
        class_id=requested_class_id,
    )
    course_id = int(course_context["course_id"])
    class_id = course_context["class_id"]
    authenticated_user_id = int(current_user["id"])
    authenticated_token_version = int(current_user.get("token_version") or 0)

    def current_authorized_user() -> dict[str, Any]:
        fresh_user = learning_logger.user(authenticated_user_id)
        if not fresh_user or not bool(fresh_user.get("enabled")):
            raise GeneralLearningError(
                "USER_DISABLED",
                "用户不存在或已禁用。",
                status_code=401,
            )
        if int(fresh_user.get("token_version") or 0) != authenticated_token_version:
            raise GeneralLearningError(
                "TOKEN_REVOKED",
                "访问令牌已失效，请重新登录。",
                status_code=401,
            )
        if str(fresh_user.get("role") or "").lower() != "student":
            raise GeneralLearningError(
                "REFERENCE_CONTEXT_CHANGED",
                "课程访问权限已变化。",
                status_code=403,
            )
        if not _user_has_run_scope(fresh_user, course_id=course_id, class_id=class_id):
            raise GeneralLearningError(
                "REFERENCE_CONTEXT_CHANGED",
                "课程访问权限已变化。",
                status_code=403,
            )
        try:
            current = _resolve_user_course_context(
                fresh_user,
                course_id=course_id,
                class_id=class_id,
            )
        except HTTPException as exc:
            raise GeneralLearningError(
                "REFERENCE_CONTEXT_CHANGED",
                "课程访问权限已变化。",
                status_code=403,
            ) from exc
        if int(current["course_id"]) != course_id or current.get("class_id") != class_id:
            raise GeneralLearningError(
                "REFERENCE_CONTEXT_CHANGED",
                "课程或班级作用域已变化。",
                status_code=409,
            )
        return fresh_user

    def validate_context() -> None:
        current_authorized_user()

    def validate_session() -> None:
        fresh_user = current_authorized_user()
        if not payload.client_session_id:
            return
        try:
            LearningEventService(learning_logger).validate_session(
                user=fresh_user,
                client_session_id=payload.client_session_id,
                course_id=course_id,
                class_id=class_id,
            )
        except LearningEventError as exc:
            raise GeneralLearningError(
                exc.error_code,
                str(exc),
                status_code=_learning_error_status(exc.error_code),
            ) from exc

    def record_generated_event(metadata: dict[str, Any]) -> None:
        if not payload.client_session_id:
            return
        fresh_user = current_authorized_user()
        try:
            LearningEventService(learning_logger).record_server_event(
                user=fresh_user,
                client_session_id=payload.client_session_id,
                event_type=ServerLearningEventType.GENERAL_QUESTION_ASKED,
                metadata=metadata,
                course_id=course_id,
                class_id=class_id,
                dedupe_key=f"GENERAL_QUESTION_ASKED:{request_id}",
            )
        except LearningEventError as exc:
            raise GeneralLearningError(
                exc.error_code,
                str(exc),
                status_code=_learning_error_status(exc.error_code),
            ) from exc

    settings = get_settings()
    service = GeneralLearningService(
        storage=learning_logger,
        retriever=HybridRetriever(storage=learning_logger),
        provider_factory=_create_general_provider,
        data_policy=ExternalModelDataPolicy.from_ids(settings.external_model_course_allowlist),
        runtime=general_learning_runtime,
    )
    disconnected = threading.Event()
    watcher = asyncio.create_task(_watch_general_disconnect(request, disconnected))
    try:
        return await service.answer(
            user=current_user,
            scope={"course_id": course_id, "class_id": class_id},
            payload=payload,
            request_id=request_id,
            context_validator=validate_context,
            session_validator=validate_session,
            event_recorder=record_generated_event,
            disconnected=disconnected,
        )
    except GeneralLearningError as exc:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
        raise HTTPException(
            status_code=exc.status_code,
            headers=headers,
            detail={
                "error_code": exc.error_code,
                "message": str(exc),
                "request_id": request_id,
                "retryable": exc.retryable,
            },
        ) from exc
    finally:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass


@app.post("/api/practice/drafts", response_model=PracticeDraftResponse)
async def create_practice_draft(
    payload: PracticeDraftRequest,
    request: Request,
    current_user: Annotated[dict, Depends(require_roles("student"))],
) -> dict[str, Any]:
    request_id = str(uuid.uuid4())
    requested_course_id, requested_class_id = _scope_from_learning_session(
        current_user=current_user,
        client_session_id=payload.client_session_id,
        course_id=payload.course_id,
        class_id=payload.class_id,
    )
    course_context = _resolve_user_course_context(
        current_user,
        course_id=requested_course_id,
        class_id=requested_class_id,
        require_course=True,
        require_class=True,
    )
    course_id = int(course_context["course_id"])
    class_id = int(course_context["class_id"]) if course_context.get("class_id") is not None else None
    authenticated_user_id = int(current_user["id"])
    authenticated_token_version = int(current_user.get("token_version") or 0)

    def current_authorized_user() -> dict[str, Any]:
        fresh_user = learning_logger.user(authenticated_user_id)
        if not fresh_user or not bool(fresh_user.get("enabled")):
            raise GeneralLearningError("USER_DISABLED", "用户不存在或已禁用。", status_code=401)
        if int(fresh_user.get("token_version") or 0) != authenticated_token_version:
            raise GeneralLearningError("TOKEN_REVOKED", "访问令牌已失效，请重新登录。", status_code=401)
        if str(fresh_user.get("role") or "").lower() != "student":
            raise GeneralLearningError("REFERENCE_CONTEXT_CHANGED", "课程访问权限已变化。", status_code=403)
        if not _user_has_run_scope(fresh_user, course_id=course_id, class_id=class_id):
            raise GeneralLearningError("REFERENCE_CONTEXT_CHANGED", "课程访问权限已变化。", status_code=403)
        try:
            current = _resolve_user_course_context(
                fresh_user,
                course_id=course_id,
                class_id=class_id,
                require_course=True,
                require_class=True,
            )
        except HTTPException as exc:
            raise GeneralLearningError(
                "REFERENCE_CONTEXT_CHANGED", "课程访问权限已变化。", status_code=403
            ) from exc
        current_class_id = int(current["class_id"]) if current.get("class_id") is not None else None
        if int(current["course_id"]) != course_id or current_class_id != class_id:
            raise GeneralLearningError("REFERENCE_CONTEXT_CHANGED", "课程或班级作用域已变化。", status_code=409)
        return fresh_user

    def validate_context() -> None:
        current_authorized_user()

    def validate_session() -> None:
        fresh_user = current_authorized_user()
        if not payload.client_session_id:
            return
        try:
            LearningEventService(learning_logger).validate_session(
                user=fresh_user,
                client_session_id=payload.client_session_id,
                course_id=course_id,
                class_id=class_id,
            )
        except LearningEventError as exc:
            raise GeneralLearningError(
                exc.error_code,
                str(exc),
                status_code=_learning_error_status(exc.error_code),
            ) from exc

    def record_generated_event(metadata: dict[str, Any]) -> None:
        if not payload.client_session_id:
            return
        fresh_user = current_authorized_user()
        try:
            LearningEventService(learning_logger).record_server_event(
                user=fresh_user,
                client_session_id=payload.client_session_id,
                event_type=ServerLearningEventType.PRACTICE_DRAFT_GENERATED,
                metadata=metadata,
                course_id=course_id,
                class_id=class_id,
                dedupe_key=f"PRACTICE_DRAFT_GENERATED:{request_id}",
            )
        except LearningEventError as exc:
            raise GeneralLearningError(
                exc.error_code,
                str(exc),
                status_code=_learning_error_status(exc.error_code),
            ) from exc

    settings = get_settings()
    service = ExploratoryPracticeService(
        storage=learning_logger,
        retriever=HybridRetriever(storage=learning_logger),
        resource_service=ExternalResourceService(storage=learning_logger, registry=connector_registry),
        provider_factory=_create_practice_draft_provider,
        data_policy=ExternalModelDataPolicy.from_ids(settings.external_model_course_allowlist),
        runtime=general_learning_runtime,
        remote_rate_limiter=lambda source_name: _consume_resource_rate_limit(authenticated_user_id, source_name),
    )
    disconnected = threading.Event()
    watcher = asyncio.create_task(_watch_general_disconnect(request, disconnected))
    try:
        return await service.generate(
            user=current_user,
            scope={"course_id": course_id, "class_id": class_id},
            payload=payload,
            request_id=request_id,
            context_validator=validate_context,
            session_validator=validate_session,
            event_recorder=record_generated_event,
            disconnected=disconnected,
        )
    except LookupInputError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": exc.error_code, "message": str(exc), "request_id": request_id},
        ) from None
    except GeneralLearningError as exc:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
        raise HTTPException(
            status_code=exc.status_code,
            headers=headers,
            detail={
                "error_code": exc.error_code,
                "message": str(exc),
                "request_id": request_id,
                "retryable": exc.retryable,
            },
        ) from exc
    finally:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass


@app.get("/api/teacher/activity", response_model=ActivitySummaryResponse)
def teacher_activity(
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict:
    try:
        return ActivityService(learning_logger).teacher_activity(
            teacher=current_user,
            course_id=course_id,
            class_id=class_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail={"error_code": "FORBIDDEN", "message": str(exc)}) from exc


@app.get("/api/students/{student_id}/timeline", response_model=LearningTimelineResponse)
def student_timeline(
    student_id: int,
    current_user: Annotated[dict, Depends(get_current_user)],
    limit: int = 100,
) -> dict:
    try:
        events = ActivityService(learning_logger).student_timeline(
            requester=current_user,
            student_id=student_id,
            limit=limit,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail={"error_code": "FORBIDDEN", "message": str(exc)}) from exc
    return {"events": events}


@app.get("/api/system/status", response_model=SystemStatusResponse)
def system_status() -> dict:
    settings = get_settings()
    provider = create_llm_provider()
    sqlite_status = _sqlite_health()
    chroma_status = _chroma_health()
    mcp_status = {
        "status": "ok" if mcp_gateway.is_available else "fallback",
        "detail": "MCP 会话可用" if mcp_gateway.is_available else "MCP 未启动，低风险只读工具将使用本地 registry 降级",
    }
    return {
        "service": "course-agent",
        "provider": provider.provider_name,
        "model": provider.model_name,
        "deepseek_configured": bool(settings.deepseek_api_key),
        "fallback_provider": settings.llm_fallback_provider,
        "fallback_active": provider.provider_name != "deepseek",
        "sqlite": sqlite_status,
        "chroma": chroma_status,
        "mcp": mcp_status,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    background_tasks: BackgroundTasks = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    user_id = _effective_user_id(payload.user_id, current_user)
    requested_course_id, requested_class_id = _scope_from_learning_session(
        current_user=current_user,
        client_session_id=payload.client_session_id,
        course_id=payload.course_id,
        class_id=payload.class_id,
    )
    course_context = _resolve_user_course_context(
        current_user,
        course_id=requested_course_id,
        class_id=requested_class_id,
    )
    _validate_request_scope(
        current_user=current_user,
        course_id=int(course_context["course_id"]),
        class_id=course_context["class_id"],
        assignment_id=payload.assignment_id,
        step_id=payload.step_id,
        client_session_id=payload.client_session_id,
    )
    provider = create_llm_provider()
    intent = route_intent(payload.question)
    model_guard = _RequestModelCallGuard(
        provider=provider,
        user_id=int(current_user["id"]),
        course_id=int(course_context["course_id"]),
        class_id=course_context["class_id"],
        client_session_id=payload.client_session_id,
        use_course_references=payload.use_course_references,
        require_course_context=_intent_requires_course_context(intent),
    )
    try:
        model_guard.check_request_start()
    except ModelCallPreflightError as exc:
        _raise_model_preflight_http(exc)
    orchestrator = AgentOrchestrator(
        storage=learning_logger,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=TemplateProvider(),
        mcp_gateway=mcp_gateway,
        model_call_guard=model_guard,
        remote_rate_limiter=lambda tool_name, arguments: _reserve_agent_tool_rate_limit(
            int(current_user["id"]), tool_name, arguments
        ),
    )
    try:
        result = orchestrator.run(
            payload.question,
            user_id=user_id,
            course_id=course_context["course_id"],
            class_id=course_context["class_id"],
            assignment_id=payload.assignment_id,
            step_id=payload.step_id,
        )
        _check_agent_result_before_business_write(
            model_guard=model_guard,
            provider=provider,
            result=result,
            orchestrator=orchestrator,
        )
    except ModelCallPreflightError as exc:
        _close_rejected_orchestrator_attempt(orchestrator, exc)
        _raise_model_preflight_http(exc)
    result["tool_calls"] = _project_tool_calls(result.get("tool_calls") or [])
    result["conversation_id"] = payload.conversation_id
    result["assignment_id"] = payload.assignment_id
    result["step_id"] = payload.step_id
    result["course_id"] = course_context["course_id"]
    result["class_id"] = course_context["class_id"]
    result["course_reference_status"] = model_guard.last_policy_status
    result["question_id"] = _record_question(user_id, payload.question, result, background_tasks=background_tasks)
    _record_optional_learning_event(
        current_user=current_user,
        client_session_id=payload.client_session_id,
        event_type="QUESTION_ASKED",
        page_name="student_chat",
        module=result.get("module", ""),
        assignment_id=str(payload.assignment_id) if payload.assignment_id else None,
        metadata={"question_id": result["question_id"], "run_id": result.get("run_id"), "step_id": payload.step_id},
    )
    return result


@app.get("/api/agent/runs/{run_id}", response_model=AgentRunResponse)
def get_agent_run(
    run_id: str,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    record = learning_logger.agent_run(run_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "RUN_NOT_FOUND",
                "message": f"Agent run 不存在：{run_id}",
                "run_id": run_id,
            },
        )
    context_status = _require_run_access(record, current_user, for_continue=False)
    state = record["state"]
    return {
        "run_id": run_id,
        "status": record["status"],
        "current_step": int(record["current_step"]),
        "max_steps": int(record["max_steps"]),
        "tool_call_count": int(record["tool_call_count"]),
        "max_tool_calls": int(record["max_tool_calls"]),
        "pending_question": record.get("pending_question"),
        "final_answer": record.get("final_answer"),
        "stop_reason": record.get("stop_reason"),
        "trace_id": state.get("trace_id") if context_status == "valid" else None,
        "steps": _project_run_steps(learning_logger.agent_steps(run_id)),
        "traces": _project_run_traces(learning_logger.agent_trace_rows(run_id)),
        "context_status": context_status,
    }


@app.post("/api/agent/runs/{run_id}/continue", response_model=ChatResponse)
def continue_agent_run(
    run_id: str,
    payload: AgentContinueRequest,
    background_tasks: BackgroundTasks = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    user_id = _effective_user_id(payload.user_id, current_user)
    record = learning_logger.agent_run(run_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "RUN_NOT_FOUND", "message": f"Agent run 不存在：{run_id}", "run_id": run_id},
        )
    _require_run_access(record, current_user, for_continue=True)
    run_course_id = int(record["course_id"])
    run_class_id = int(record["class_id"]) if record.get("class_id") is not None else None
    _validate_request_scope(
        current_user=current_user,
        course_id=run_course_id,
        class_id=run_class_id,
        assignment_id=payload.assignment_id,
        step_id=payload.step_id,
        client_session_id=payload.client_session_id,
    )
    provider = create_llm_provider()
    model_guard = _RequestModelCallGuard(
        provider=provider,
        user_id=int(current_user["id"]),
        course_id=run_course_id,
        class_id=run_class_id,
        client_session_id=payload.client_session_id,
        use_course_references=payload.use_course_references,
        require_course_context=_intent_requires_course_context(str(record.get("intent") or "")),
    )
    try:
        model_guard.check_request_start()
    except ModelCallPreflightError as exc:
        _raise_model_preflight_http(exc)
    orchestrator = AgentOrchestrator(
        storage=learning_logger,
        planner_provider=provider,
        answer_provider=provider,
        fallback_provider=TemplateProvider(),
        mcp_gateway=mcp_gateway,
        model_call_guard=model_guard,
        remote_rate_limiter=lambda tool_name, arguments: _reserve_agent_tool_rate_limit(
            int(current_user["id"]), tool_name, arguments
        ),
    )
    try:
        result = orchestrator.continue_run(
            run_id,
            payload.user_input,
            user_id=user_id,
            course_id=run_course_id,
            class_id=run_class_id,
            assignment_id=payload.assignment_id,
            step_id=payload.step_id,
        )
        _check_agent_result_before_business_write(
            model_guard=model_guard,
            provider=provider,
            result=result,
            orchestrator=orchestrator,
        )
        result["tool_calls"] = _project_tool_calls(result.get("tool_calls") or [])
    except ModelCallPreflightError as exc:
        _close_rejected_orchestrator_attempt(orchestrator, exc)
        _raise_model_preflight_http(exc)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "RUN_CANNOT_CONTINUE",
                "message": str(exc),
                "run_id": run_id,
            },
        ) from exc
    result["conversation_id"] = payload.conversation_id
    result["assignment_id"] = payload.assignment_id
    result["step_id"] = payload.step_id
    result["course_id"] = run_course_id
    result["class_id"] = run_class_id
    result["course_reference_status"] = model_guard.last_policy_status
    result["question_id"] = _record_question(user_id, payload.user_input, result, background_tasks=background_tasks)
    _record_optional_learning_event(
        current_user=current_user,
        client_session_id=payload.client_session_id,
        event_type="QUESTION_ASKED",
        page_name="student_chat",
        module=result.get("module", ""),
        assignment_id=str(payload.assignment_id) if payload.assignment_id else None,
        metadata={"question_id": result["question_id"], "run_id": result.get("run_id"), "continued": True, "step_id": payload.step_id},
    )
    return result


@app.post("/api/practice/guide", response_model=PracticeGuideResponse)
def practice_guide(
    payload: PracticeGuideRequest,
    background_tasks: BackgroundTasks = None,
    current_user: Annotated[dict | None, Depends(get_optional_current_user)] = None,
) -> dict:
    if current_user is not None:
        user_id = str(current_user["id"])
        requested_course_id, requested_class_id = _scope_from_learning_session(
            current_user=current_user,
            client_session_id=payload.client_session_id,
            course_id=payload.course_id,
            class_id=payload.class_id,
        )
        course_context = _resolve_user_course_context(
            current_user,
            course_id=requested_course_id,
            class_id=requested_class_id,
        )
        _validate_request_scope(
            current_user=current_user,
            course_id=int(course_context["course_id"]),
            class_id=course_context["class_id"],
            assignment_id=payload.assignment_id,
            step_id=payload.step_id,
            client_session_id=payload.client_session_id,
        )
        provider = create_llm_provider()
        model_guard = _RequestModelCallGuard(
            provider=provider,
            user_id=int(current_user["id"]),
            course_id=int(course_context["course_id"]),
            class_id=course_context["class_id"],
            client_session_id=payload.client_session_id,
            use_course_references=payload.use_course_references,
            require_course_context=True,
        )
        try:
            model_guard.check_request_start()
        except ModelCallPreflightError as exc:
            _raise_model_preflight_http(exc)
        orchestrator = AgentOrchestrator(
            storage=learning_logger,
            planner_provider=provider,
            answer_provider=provider,
            fallback_provider=TemplateProvider(),
            mcp_gateway=mcp_gateway,
            model_call_guard=model_guard,
        )
        try:
            result = orchestrator.run(
                payload.topic,
                user_id=user_id,
                course_id=course_context["course_id"],
                class_id=course_context["class_id"],
                intent_override="practice_guide",
                assignment_id=payload.assignment_id,
                step_id=payload.step_id,
            )
            _check_agent_result_before_business_write(
                model_guard=model_guard,
                provider=provider,
                result=result,
                orchestrator=orchestrator,
            )
        except ModelCallPreflightError as exc:
            _close_rejected_orchestrator_attempt(orchestrator, exc)
            _raise_model_preflight_http(exc)
        result["tool_calls"] = _project_tool_calls(result.get("tool_calls") or [])
        result["course_id"] = course_context["course_id"]
        result["class_id"] = course_context["class_id"]
        result["course_reference_status"] = model_guard.last_policy_status
    else:
        if (
            payload.assignment_id is not None
            or payload.step_id is not None
            or payload.client_session_id is not None
        ):
            raise HTTPException(
                status_code=401,
                detail={
                    "error_code": "AUTH_REQUIRED",
                    "message": "匿名 Practice 不接受任务、步骤或学习会话作用域，请先登录。",
                },
            )
        user_id = "anonymous_demo"
        result = answerer.practice_guide(payload.topic)
        result.update(
            {
                "provider": "local-template",
                "model": "public-demo-template",
                "fallback_used": True,
                "delivery_mode": "offline_fallback",
                "run_status": "COMPLETED",
                "current_step": 0,
                "requires_user_input": False,
                "tool_calls": [],
                "course_id": None,
                "class_id": None,
            }
        )
    result["assignment_id"] = payload.assignment_id
    result["step_id"] = payload.step_id
    result["question_id"] = _record_question(
        user_id,
        payload.topic,
        result,
        background_tasks=background_tasks,
    )
    _record_optional_learning_event(
        current_user=current_user,
        client_session_id=payload.client_session_id,
        event_type="PRACTICE_GUIDE_GENERATED",
        page_name="practice_guide",
        module=result.get("module", ""),
        assignment_id=str(payload.assignment_id) if payload.assignment_id else None,
        metadata={"topic": payload.topic, "assignment_id": payload.assignment_id, "step_id": payload.step_id},
    )
    return result


@app.get("/api/assignments")
def list_assignments(
    current_user: Annotated[dict, Depends(get_current_user)],
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict:
    context = _resolve_user_course_context(
        current_user,
        course_id=course_id,
        class_id=class_id,
        require_course=True,
        require_class=class_id is not None,
    )
    progress_by_assignment = {
        int(item["assignment_id"]): item
        for item in learning_logger.assignment_progress(user_id=int(current_user["id"]))
    }
    assignments = []
    for assignment in learning_logger.assignments():
        if int(assignment.get("course_id") or 0) != int(context["course_id"]):
            continue
        if not _user_can_access_course(current_user, assignment.get("course_id")):
            continue
        assignment_class_id = assignment.get("class_id")
        if assignment_class_id is not None and (
            context["class_id"] is None or int(assignment_class_id) != int(context["class_id"])
        ):
            continue
        item = dict(assignment)
        item["progress"] = progress_by_assignment.get(int(item["id"]))
        assignments.append(item)
    return {"assignments": assignments}


@app.get("/api/assignments/{assignment_id}/progress")
def get_assignment_progress(
    assignment_id: int,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    assignment = _require_assignment_scope(learning_logger.assignment(assignment_id), current_user)
    progress = learning_logger.assignment_progress(user_id=int(current_user["id"]), assignment_id=assignment_id)
    return {"assignment_id": assignment_id, "progress": progress[0] if progress else None}


@app.post("/api/assignments/{assignment_id}/progress")
def update_assignment_progress(
    assignment_id: int,
    payload: AssignmentProgressRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    assignment = learning_logger.assignment(assignment_id)
    step = learning_logger.assignment_step(payload.step_id)
    assignment = _require_assignment_scope(assignment, current_user)
    if step is None or int(step["assignment_id"]) != int(assignment_id):
        raise HTTPException(
            status_code=400,
            detail={"error_code": "STEP_NOT_IN_ASSIGNMENT", "message": "步骤不属于该实验任务。"},
        )
    if payload.client_session_id:
        session = learning_logger.learning_session_by_client_id(payload.client_session_id)
        _validate_request_scope(
            current_user=current_user,
            course_id=int(assignment["course_id"]),
            class_id=int(session["class_id"]) if session and session.get("class_id") is not None else None,
            assignment_id=assignment_id,
            step_id=payload.step_id,
            client_session_id=payload.client_session_id,
        )
    progress = learning_logger.upsert_assignment_progress(
        user_id=int(current_user["id"]),
        assignment_id=assignment_id,
        step_id=payload.step_id,
        status=payload.status,
        progress_percent=payload.progress_percent,
        note=payload.note,
    )
    _record_optional_learning_event(
        current_user=current_user,
        client_session_id=payload.client_session_id,
        event_type="ASSIGNMENT_PROGRESS_UPDATED",
        page_name="assignment",
        module=str(step.get("module") or assignment.get("module") or ""),
        assignment_id=str(assignment_id),
        metadata={
            "step_id": payload.step_id,
            "status": payload.status,
            "progress_percent": payload.progress_percent,
        },
    )
    return {"assignment_id": assignment_id, "progress": progress}


@app.post("/api/stuck-reports")
def create_stuck_report(
    payload: StuckReportRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    assignment = learning_logger.assignment(payload.assignment_id)
    step = learning_logger.assignment_step(payload.step_id)
    assignment = _require_assignment_scope(assignment, current_user)
    if step is None or int(step["assignment_id"]) != int(payload.assignment_id):
        raise HTTPException(
            status_code=400,
            detail={"error_code": "STEP_NOT_IN_ASSIGNMENT", "message": "步骤不属于该实验任务。"},
        )
    if payload.client_session_id:
        session = learning_logger.learning_session_by_client_id(payload.client_session_id)
        _validate_request_scope(
            current_user=current_user,
            course_id=int(assignment["course_id"]),
            class_id=int(session["class_id"]) if session and session.get("class_id") is not None else None,
            assignment_id=payload.assignment_id,
            step_id=payload.step_id,
            client_session_id=payload.client_session_id,
        )
    module = payload.module or str(step.get("module") or assignment.get("module") or "")
    report = learning_logger.create_stuck_report(
        user_id=int(current_user["id"]),
        assignment_id=payload.assignment_id,
        step_id=payload.step_id,
        module=module,
        reason=payload.reason,
        source="active",
        evidence={"reported_by": current_user["id"]},
    )
    _record_optional_learning_event(
        current_user=current_user,
        client_session_id=payload.client_session_id,
        event_type="STUCK_REPORTED",
        page_name="assignment",
        module=module,
        assignment_id=str(payload.assignment_id),
        metadata={"step_id": payload.step_id, "report_id": report["id"], "reason": payload.reason},
    )
    return {"report": report}


@app.get("/api/teacher/stuck-analysis")
def teacher_stuck_analysis(
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
    assignment_id: int | None = None,
    step_id: int | None = None,
    module: str = "",
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict:
    if assignment_id is not None:
        _require_assignment_scope(learning_logger.assignment(assignment_id), current_user)
    if step_id is not None:
        step = learning_logger.assignment_step(step_id)
        if step is None:
            raise HTTPException(
                status_code=404,
                detail={"error_code": "ASSIGNMENT_STEP_NOT_FOUND", "message": f"实验步骤不存在：{step_id}"},
            )
        if assignment_id is not None and int(step["assignment_id"]) != int(assignment_id):
            raise HTTPException(
                status_code=400,
                detail={"error_code": "STEP_NOT_IN_ASSIGNMENT", "message": "步骤不属于该实验任务。"},
            )
        _require_assignment_scope(learning_logger.assignment(int(step["assignment_id"])), current_user)
    scope = _resolve_teacher_scope(current_user, course_id=course_id, class_id=class_id)
    try:
        return StuckAnalysisService(learning_logger).analyze(
            assignment_id=assignment_id,
            step_id=step_id,
            module=module,
            class_id=class_id,
            class_ids=scope["class_ids"] if class_id is None else None,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail={"error_code": "FORBIDDEN", "message": str(exc)}) from exc


@app.post("/api/teacher/stuck-reports/{report_id}/action")
def update_stuck_report(
    report_id: int,
    payload: StuckActionRequest,
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
) -> dict:
    report = next((item for item in learning_logger.stuck_reports() if int(item["id"]) == int(report_id)), None)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "STUCK_REPORT_NOT_FOUND", "message": f"卡点记录不存在：{report_id}"},
        )
    assignment = _require_assignment_scope(learning_logger.assignment(int(report["assignment_id"])), current_user)
    if str(current_user["role"]).lower() == "teacher":
        _resolve_teacher_scope(current_user, course_id=int(assignment["course_id"]))
        if not learning_logger.teacher_can_access_user(int(current_user["id"]), int(report["user_id"])):
            raise HTTPException(
                status_code=403,
                detail={"error_code": "CLASS_FORBIDDEN", "message": "教师只能修改自己班级学生的卡点记录。"},
            )
    try:
        report = StuckAnalysisService(learning_logger).update_status(report_id, payload.action)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "STUCK_REPORT_NOT_FOUND", "message": str(exc)},
        ) from exc
    return {"report": report}


@app.post("/api/knowledge/upload", response_model=UploadResponse)
async def upload_knowledge(
    file: Annotated[UploadFile, File(...)],
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))] = None,
    course_id: int | None = None,
) -> UploadResponse:
    try:
        filename = validate_upload_filename(file.filename)
    except DocumentParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="仅支持上传 Markdown、TXT、PDF 或 DOCX 文件。",
        )

    scope = _resolve_knowledge_scope(current_user, course_id=course_id)
    document_course_id = scope["course_id"]
    document_class_id = scope["class_id"]
    content = await _read_upload_file_limited(file)
    if not content:
        raise HTTPException(status_code=400, detail="上传文件不能为空。")

    content_hash = sha256_bytes(content)
    existing_document = learning_logger.document_by_hash(
        content_hash,
        course_id=int(document_course_id) if document_course_id is not None else None,
    )
    if existing_document:
        return UploadResponse(
            filename=filename,
            saved=False,
            message="检测到重复内容，已复用已有文档记录。",
            document_id=int(existing_document["id"]),
            content_hash=existing_document["content_hash"],
            chunk_count=int(existing_document["chunk_count"]),
            duplicate=True,
            index_status=existing_document.get("index_status", "PENDING"),
            job_id=None,
        )

    target_dir = _knowledge_course_dir(int(document_course_id) if document_course_id is not None else None)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    existing_filename_document = next(
        (
            document
            for document in learning_logger.documents(
                course_id=int(document_course_id) if document_course_id is not None else None
            )
            if document.get("filename") == filename
        ),
        None,
    )
    if existing_filename_document and existing_filename_document.get("content_hash") != content_hash:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "DOCUMENT_FILENAME_CONFLICT",
                "message": "已存在同名但内容不同的知识库文件，请更名后再上传。",
            },
        )
    if target.exists() and sha256_bytes(target.read_bytes()) != content_hash:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "KNOWLEDGE_FILE_CONFLICT",
                "message": "知识库目录中已存在同名但内容不同的文件，已拒绝覆盖。",
            },
        )

    wrote_target = False
    if not target.exists():
        target.write_bytes(content)
        wrote_target = True
    job_id = learning_logger.create_index_job(
        filename=filename,
        content_hash=content_hash,
        status="PENDING",
    )

    try:
        learning_logger.update_index_job(job_id, status="PARSING")
        document = parse_document(target)
    except DocumentParseError as exc:
        if wrote_target:
            target.unlink(missing_ok=True)
        learning_logger.fail_index_job(job_id, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document_record = learning_logger.save_document(
        document=document,
        content_hash=content_hash,
        chunks=None,
        byte_size=len(content),
        job_id=job_id,
        course_id=int(document_course_id) if document_course_id is not None else None,
        class_id=int(document_class_id) if document_class_id is not None else None,
        uploaded_by_user_id=int(current_user["id"]) if current_user else None,
    )
    if suffix in {".md", ".txt"}:
        answerer.retriever.reload()

    duplicate = bool(document_record["duplicate"])
    return UploadResponse(
        filename=filename,
        saved=not duplicate,
        message=(
            "检测到重复内容，已复用已有文档记录。"
            if duplicate
            else "知识库文件已上传并切片，当前状态为 PENDING；教师点击“建立索引”后才会进入检索。"
        ),
        document_id=int(document_record["id"]),
        content_hash=document_record["content_hash"],
        chunk_count=int(document_record["chunk_count"]),
        duplicate=duplicate,
        index_status=document_record["index_status"],
        job_id=int(document_record["job_id"]) if document_record.get("job_id") is not None else None,
    )


@app.get("/api/knowledge/documents", response_model=KnowledgeDocumentsResponse)
def list_knowledge_documents(
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))] = None,
    course_id: int | None = None,
) -> dict:
    if current_user is None:
        visible_documents = learning_logger.documents(course_id=course_id)
    elif str(current_user["role"]).lower() == "admin" and course_id is None:
        visible_documents = learning_logger.documents()
    else:
        scope = _resolve_knowledge_scope(current_user, course_id=course_id, require_course=False)
        visible_documents = learning_logger.documents(course_id=int(scope["course_id"]))
    visible_document_ids = {int(document["id"]) for document in visible_documents}
    jobs = [
        job
        for job in learning_logger.index_jobs()
        if job.get("document_id") is not None and int(job["document_id"]) in visible_document_ids
    ]
    jobs_by_document: dict[int, list[dict]] = {}
    for job in jobs:
        document_id = job.get("document_id")
        if document_id is not None:
            jobs_by_document.setdefault(int(document_id), []).append(job)
    documents = []
    for document in visible_documents:
        document_jobs = jobs_by_document.get(int(document["id"]), [])
        latest_job = document_jobs[0] if document_jobs else None
        documents.append(
            {
                **document,
                "jobs": document_jobs,
                "latest_job_status": latest_job.get("status") if latest_job else None,
                "latest_job_updated_at": latest_job.get("updated_at") if latest_job else None,
                "latest_job_error": latest_job.get("error_message") if latest_job else None,
            }
        )
    return {"documents": documents, "index_jobs": jobs}


@app.post("/api/knowledge/documents/{document_id}/reindex", response_model=KnowledgeActionResponse)
def reindex_knowledge_document(
    document_id: int,
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))] = None,
) -> dict:
    try:
        lock_context = document_locks.acquire(document_id)
        lock_context.__enter__()
    except DocumentBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error_code": "DOCUMENT_BUSY", "message": "文档正在执行其他索引或发布操作。"},
        ) from exc
    try:
        return _reindex_knowledge_document_locked(document_id, current_user)
    finally:
        lock_context.__exit__(None, None, None)


def _reindex_knowledge_document_locked(document_id: int, current_user: dict | None) -> dict:
    document = learning_logger.document(document_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "DOCUMENT_NOT_FOUND", "message": f"文档不存在：{document_id}"},
        )
    _require_document_management_access(document, current_user)

    try:
        job_id = learning_logger.begin_document_reindex(document_id)
    except ValueError as exc:
        error_code = str(exc)
        if error_code == "DOCUMENT_NOT_FOUND":
            raise HTTPException(
                status_code=404,
                detail={"error_code": error_code, "message": f"文档不存在：{document_id}"},
            ) from exc
        raise HTTPException(
            status_code=409,
            detail={"error_code": "DOCUMENT_BUSY", "message": "文档已有索引任务正在执行。"},
        ) from exc
    try:
        chunks = _chunks_for_vector_index(document_id, document)
        if not chunks:
            raise ValueError("文档没有可索引的切片。")
        store = ChromaVectorStore()
        store.delete_by_document(document_id)
        inserted = store.add_chunks(chunks)
        learning_logger.mark_document_chunks_indexed([int(chunk["id"]) for chunk in chunks])
        learning_logger.update_index_job(job_id, status="INDEXED", document_id=document_id)
        return {
            "document_id": document_id,
            "status": "INDEXED",
            "message": "文档向量索引已重建。",
            "job_id": job_id,
            "chunk_count": inserted,
        }
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        learning_logger.update_document_index_status(document_id, "FAILED")
        learning_logger.fail_index_job(job_id, message, document_id=document_id)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "REINDEX_FAILED",
                "message": f"重建索引失败：{message}",
            },
        ) from exc


@app.post("/api/knowledge/documents/{document_id}/publish", response_model=KnowledgeActionResponse)
def publish_knowledge_document(
    document_id: int,
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))] = None,
) -> dict:
    return _set_document_publication(document_id, current_user, published=True)


@app.post("/api/knowledge/documents/{document_id}/unpublish", response_model=KnowledgeActionResponse)
def unpublish_knowledge_document(
    document_id: int,
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))] = None,
) -> dict:
    return _set_document_publication(document_id, current_user, published=False)


def _set_document_publication(document_id: int, current_user: dict, *, published: bool) -> dict:
    try:
        with document_locks.acquire(document_id):
            document = learning_logger.document(document_id)
            if document is None:
                raise HTTPException(
                    status_code=404,
                    detail={"error_code": "DOCUMENT_NOT_FOUND", "message": f"文档不存在：{document_id}"},
                )
            _require_document_management_access(document, current_user)
            try:
                updated = learning_logger.set_document_publication(
                    document_id,
                    published=published,
                    actor_id=int(current_user["id"]),
                )
            except ValueError as exc:
                code = str(exc)
                status = 409 if code in {"DOCUMENT_BUSY", "DOCUMENT_NOT_INDEXED", "DOCUMENT_HAS_NO_INDEXED_CHUNKS"} else 404
                raise HTTPException(
                    status_code=status,
                    detail={"error_code": code, "message": "文档当前状态不允许该发布操作。"},
                ) from exc
    except DocumentBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error_code": "DOCUMENT_BUSY", "message": "文档正在执行其他索引或发布操作。"},
        ) from exc
    projection = publication_projection(updated)
    return {
        "document_id": document_id,
        "status": "PUBLISHED" if published else "UNPUBLISHED",
        "message": "文档已发布给学生。" if published else "文档已撤回学生可见性。",
        "job_id": None,
        "chunk_count": int(updated.get("chunk_count") or 0),
        **projection,
    }


@app.delete("/api/knowledge/documents/{document_id}", response_model=KnowledgeActionResponse)
def delete_knowledge_document(
    document_id: int,
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))] = None,
) -> dict:
    try:
        with document_locks.acquire(document_id):
            document = learning_logger.document(document_id)
            if document is None:
                raise HTTPException(
                    status_code=404,
                    detail={"error_code": "DOCUMENT_NOT_FOUND", "message": f"文档不存在：{document_id}"},
                )
            _require_document_management_access(document, current_user)
            try:
                ChromaVectorStore().delete_by_document(document_id)
            except Exception as exc:
                message = str(exc) or exc.__class__.__name__
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error_code": "DELETE_VECTOR_FAILED",
                        "message": f"删除文档前清理 Chroma 向量失败：{message}",
                    },
                ) from exc
            learning_logger.delete_document(document_id)
            source_path = Path(document["source_path"])
            if _is_safe_knowledge_file(source_path):
                source_path.resolve().unlink(missing_ok=True)
    except DocumentBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error_code": "DOCUMENT_BUSY", "message": "文档正在执行其他索引或发布操作。"},
        ) from exc
    message = "文档已删除，SQLite 记录和 Chroma 向量已同步清理。"
    return {
        "document_id": document_id,
        "status": "DELETED",
        "message": message,
        "job_id": None,
        "chunk_count": int(document["chunk_count"]),
    }


@app.get("/api/teacher/analytics", response_model=TeacherAnalyticsResponse)
def teacher_analytics(
    current_user: Annotated[dict | None, Depends(_require_admin_api_access)] = None,
) -> dict:
    return learning_logger.analytics()


@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(
    payload: FeedbackRequest,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    user_id = _effective_user_id(payload.user_id, current_user)
    _require_question_feedback_access(payload.question_id, current_user)
    try:
        feedback_id = learning_logger.log_feedback(
            question_id=payload.question_id,
            user_id=user_id,
            rating=payload.rating,
            helpful=payload.helpful,
            comment=payload.comment,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "QUESTION_NOT_FOUND", "message": str(exc)},
        ) from exc
    _record_optional_learning_event(
        current_user=current_user,
        client_session_id=payload.client_session_id,
        event_type="FEEDBACK_SUBMITTED",
        page_name="student_feedback",
        metadata={"question_id": payload.question_id, "rating": payload.rating},
    )
    return {
        "feedback_id": feedback_id,
        "question_id": payload.question_id,
        "saved": True,
        "message": "反馈已记录。",
    }


@app.get("/api/teacher/faq", response_model=FAQResponse)
def teacher_faq(
    current_user: Annotated[dict | None, Depends(_require_admin_api_access)] = None,
) -> dict:
    return learning_logger.faq()


@app.post("/api/teacher/question-clusters/rebuild")
def rebuild_question_clusters(
    payload: ClusterRebuildRequest,
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
) -> dict:
    scope = _resolve_teacher_scope(current_user, course_id=payload.course_id, class_id=payload.class_id)
    course_id = scope["course_id"] if str(current_user["role"]).lower() == "teacher" else payload.course_id
    class_ids = tuple(int(item) for item in scope["class_ids"]) if str(current_user["role"]).lower() == "teacher" else None
    service = question_clustering_service_factory(learning_logger)
    return service.rebuild(
        ClusterBuildOptions(
            course_id=course_id,
            class_id=payload.class_id,
            class_ids=class_ids,
            start_at=payload.start_at,
            end_at=payload.end_at,
            distance_threshold=payload.distance_threshold,
            min_cluster_size=payload.min_cluster_size,
        )
    )


@app.get("/api/teacher/question-clusters")
def list_question_clusters(
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict:
    scope = _resolve_teacher_scope(current_user, course_id=course_id, class_id=class_id)
    role = str(current_user["role"]).lower()
    if role != "teacher":
        payload = question_clustering_service_factory(learning_logger).list_clusters(course_id=course_id, class_id=class_id)
        payload["jobs"] = learning_logger.question_cluster_jobs(course_id=course_id, class_id=class_id)
        return payload
    class_ids = tuple(int(item) for item in scope["class_ids"])
    payload = question_clustering_service_factory(learning_logger).list_clusters(
        course_id=scope["course_id"],
        class_id=class_id,
        class_ids=class_ids,
    )
    payload["jobs"] = learning_logger.question_cluster_jobs(
        course_id=scope["course_id"],
        class_id=class_id,
        class_ids=class_ids,
    )
    return payload


@app.patch("/api/teacher/question-clusters/{cluster_id}")
def rename_question_cluster(
    cluster_id: int,
    payload: ClusterRenameRequest,
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
) -> dict:
    _require_cluster_scope(cluster_id, current_user)
    return question_clustering_service_factory(learning_logger).rename(cluster_id, payload.label)


@app.post("/api/teacher/question-clusters/{cluster_id}/approve")
def approve_question_cluster(
    cluster_id: int,
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
) -> dict:
    _require_cluster_scope(cluster_id, current_user)
    return question_clustering_service_factory(learning_logger).approve(cluster_id)


@app.post("/api/teacher/question-clusters/merge")
def merge_question_clusters(
    payload: ClusterMergeRequest,
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
) -> dict:
    _require_cluster_scope(payload.target_cluster_id, current_user)
    target = _cluster_by_id(payload.target_cluster_id)
    for source_cluster_id in payload.source_cluster_ids:
        source = _require_cluster_scope(source_cluster_id, current_user)
        if target is not None and int(source["class_id"]) != int(target["class_id"]):
            raise HTTPException(
                status_code=403,
                detail={"error_code": "CLASS_FORBIDDEN", "message": "不能合并不同班级的问题簇。"},
            )
    return question_clustering_service_factory(learning_logger).merge(
        payload.target_cluster_id,
        payload.source_cluster_ids,
        payload.label,
    )


@app.post("/api/teacher/question-clusters/{cluster_id}/split")
def split_question_cluster(
    cluster_id: int,
    payload: ClusterSplitRequest,
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
) -> dict:
    _require_cluster_scope(cluster_id, current_user)
    return question_clustering_service_factory(learning_logger).split(cluster_id, payload.question_ids, payload.label)


@app.get("/api/teacher/question-clusters/faq-export")
def export_question_cluster_faq(
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict:
    scope = _resolve_teacher_scope(current_user, course_id=course_id, class_id=class_id)
    visible_course_id = scope["course_id"] if str(current_user["role"]).lower() == "teacher" else course_id
    class_ids = tuple(int(item) for item in scope["class_ids"]) if str(current_user["role"]).lower() == "teacher" else None
    return question_clustering_service_factory(learning_logger).export_faq(
        course_id=visible_course_id,
        class_id=class_id,
        class_ids=class_ids,
    )


@app.get("/api/tools", response_model=list[ToolInfo])
def tools() -> list[dict]:
    return mcp_gateway.list_tools()


@app.get("/api/resources/portal")
def resource_portal(
    current_user: Annotated[dict, Depends(get_current_user)],
    keyword: str = "",
    topic: str = "",
    resource_type: str = "",
    year_from: int | None = None,
    year_to: int | None = None,
    source: str = "",
    page: int = 1,
    page_size: int = 10,
    client_session_id: str | None = None,
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict:
    if resource_type and resource_type not in RESOURCE_TYPES:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "INVALID_RESOURCE_TYPE", "message": f"不支持的资料类型：{resource_type}"},
        )
    query = ResourcePortalQuery(
        keyword=keyword.strip(),
        topic=topic.strip(),
        resource_type=resource_type.strip(),
        year_from=year_from,
        year_to=year_to,
        source=source.strip(),
        page=page,
        page_size=page_size,
    )
    course_id, class_id = _scope_from_learning_session(
        current_user=current_user,
        client_session_id=client_session_id,
        course_id=course_id,
        class_id=class_id,
    )
    if str(current_user.get("role") or "").lower() == "admin" and course_id is None:
        course_context = {"course_id": None, "class_id": None}
    else:
        course_context = _resolve_user_course_context(
            current_user,
            course_id=course_id,
            class_id=class_id,
            require_course=True,
        )
    if client_session_id:
        if course_context["course_id"] is None:
            raise HTTPException(
                status_code=400,
                detail={"error_code": "COURSE_REQUIRED", "message": "记录课程门户事件前须明确课程。"},
            )
        _validate_request_scope(
            current_user=current_user,
            course_id=int(course_context["course_id"]),
            class_id=course_context["class_id"],
            client_session_id=client_session_id,
        )
    payload = ResourcePortalService(
        learning_logger,
        course_id=course_context["course_id"],
    ).search(query)
    role = str(current_user.get("role", "")).lower()
    payload["can_manage"] = role in {"teacher", "admin"}
    _record_optional_learning_event(
        current_user=current_user,
        client_session_id=client_session_id,
        event_type="RESOURCE_SEARCH",
        page_name="resources",
        module=topic,
        metadata={
            "keyword": keyword,
            "topic": topic,
            "resource_type": resource_type,
            "year_from": year_from,
            "year_to": year_to,
            "source": source,
            "result_count": payload["total"],
            "course_id": course_context["course_id"],
        },
        course_id=course_context["course_id"],
        class_id=course_context["class_id"],
    )
    return payload


@app.post("/api/resources/open")
def open_resource(
    payload: ResourceOpenRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    requested_course_id, requested_class_id = _scope_from_learning_session(
        current_user=current_user,
        client_session_id=payload.client_session_id,
        course_id=payload.course_id,
        class_id=payload.class_id,
    )
    is_document = payload.resource_id.startswith("document:")
    if is_document:
        course_context = _resolve_user_course_context(
            current_user,
            course_id=requested_course_id,
            class_id=requested_class_id,
            require_course=True,
        )
    elif requested_course_id is not None:
        course_context = _resolve_user_course_context(
            current_user,
            course_id=requested_course_id,
            class_id=requested_class_id,
            require_course=True,
        )
    else:
        course_context = {"course_id": None, "class_id": None}
    if payload.client_session_id:
        if course_context["course_id"] is None:
            raise HTTPException(
                status_code=400,
                detail={"error_code": "COURSE_REQUIRED", "message": "记录资料事件前须明确课程。"},
            )
        _validate_request_scope(
            current_user=current_user,
            course_id=int(course_context["course_id"]),
            class_id=course_context["class_id"],
            client_session_id=payload.client_session_id,
        )
    service = ResourcePortalService(learning_logger, course_id=course_context["course_id"])
    if not service.resource_exists(payload.resource_id):
        raise HTTPException(
            status_code=404,
            detail={"error_code": "RESOURCE_NOT_FOUND", "message": f"资料不存在：{payload.resource_id}"},
        )
    if payload.client_session_id:
        try:
            LearningEventService(learning_logger).record_event(
                user=current_user,
                client_session_id=payload.client_session_id,
                event_type="RESOURCE_OPEN",
                page_name="resources",
                resource_id=payload.resource_id,
                metadata=payload.metadata,
                course_id=course_context["course_id"],
                class_id=course_context["class_id"],
            )
        except LearningEventError as exc:
            raise HTTPException(status_code=_learning_error_status(exc.error_code), detail={"error_code": exc.error_code, "message": str(exc)}) from exc
    return {"opened": True, "resource_id": payload.resource_id}


@app.post("/api/innovation/sessions")
async def start_innovation_session(
    payload: InnovationStartRequest,
    request: Request,
    current_user: Annotated[dict, Depends(require_roles("student"))],
) -> dict:
    disconnected = threading.Event()
    watcher = asyncio.create_task(_watch_general_disconnect(request, disconnected))
    try:
        course_context = _resolve_user_course_context(
            current_user,
            course_id=payload.course_id,
            class_id=payload.class_id,
            require_class=True,
        )
        resource_service = ExternalResourceService(storage=learning_logger, registry=connector_registry)
        service = InnovationService(
            learning_logger,
            mcp_gateway=mcp_gateway,
            resource_service=resource_service,
            remote_rate_limiter=_consume_resource_rate_limit,
        )
        return await run_in_threadpool(
            service.start_session,
            user=current_user,
            topic=payload.topic,
            current_plan=payload.current_plan,
            client_session_id=payload.client_session_id,
            course_id=course_context["course_id"],
            class_id=course_context["class_id"],
            disconnected=disconnected,
        )
    except InnovationError as exc:
        status = _innovation_error_status(exc.error_code)
        raise HTTPException(status_code=status, detail={"error_code": exc.error_code, "message": str(exc)}) from exc
    finally:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass


@app.get("/api/innovation/sessions")
def list_innovation_sessions(
    course_id: int,
    current_user: Annotated[dict, Depends(require_roles("student"))],
    class_id: int | None = None,
) -> dict:
    try:
        return InnovationService(learning_logger, mcp_gateway=mcp_gateway).list_sessions(
            user=current_user, course_id=course_id, class_id=class_id,
        )
    except InnovationError as exc:
        status = _innovation_error_status(exc.error_code)
        raise HTTPException(status_code=status, detail={"error_code": exc.error_code, "message": str(exc)}) from exc


@app.get("/api/innovation/sessions/{session_id}")
def get_innovation_session(
    session_id: int,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    try:
        return InnovationService(learning_logger, mcp_gateway=mcp_gateway).get_session(
            user=current_user,
            session_id=session_id,
        )
    except InnovationError as exc:
        status = _innovation_error_status(exc.error_code, missing_status=404)
        raise HTTPException(status_code=status, detail={"error_code": exc.error_code, "message": str(exc)}) from exc


@app.post(
    "/api/innovation/sessions/{session_id}/suggestions",
    response_model=InnovationSuggestionResponse,
)
async def suggest_innovation_session(
    session_id: int,
    payload: InnovationSuggestionRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict[str, Any]:
    request_id = str(uuid.uuid4())
    innovation_service = InnovationService(learning_logger, mcp_gateway=mcp_gateway)
    try:
        initial_context = innovation_service.get_advisor_context(
            user=current_user,
            session_id=session_id,
            client_session_id=payload.client_session_id,
        )
    except InnovationError as exc:
        status = _innovation_error_status(exc.error_code, missing_status=404)
        raise HTTPException(
            status_code=status,
            detail={"error_code": exc.error_code, "message": str(exc), "request_id": request_id},
        ) from exc

    session = initial_context["session"]
    course_id = int(session.get("course_id") or 0)
    class_id = int(session["class_id"]) if session.get("class_id") is not None else None
    authenticated_user_id = int(current_user["id"])
    authenticated_token_version = int(current_user.get("token_version") or 0)

    def current_authorized_user() -> dict[str, Any]:
        fresh_user = learning_logger.user(authenticated_user_id)
        if not fresh_user or not bool(fresh_user.get("enabled")):
            raise GeneralLearningError("USER_DISABLED", "用户不存在或已禁用。", status_code=401)
        if int(fresh_user.get("token_version") or 0) != authenticated_token_version:
            raise GeneralLearningError("TOKEN_REVOKED", "访问令牌已失效，请重新登录。", status_code=401)
        if str(fresh_user.get("role") or "").lower() != "student":
            raise GeneralLearningError("REFERENCE_CONTEXT_CHANGED", "创新会话写权限已变化。", status_code=403)
        try:
            innovation_service.get_advisor_context(
                user=fresh_user,
                session_id=session_id,
                client_session_id=payload.client_session_id,
            )
        except InnovationError as exc:
            status = _innovation_error_status(exc.error_code, missing_status=404)
            if status == 404:
                status = 409
            raise GeneralLearningError(
                "REFERENCE_CONTEXT_CHANGED" if status in {403, 404} else exc.error_code,
                str(exc),
                status_code=status,
            ) from exc
        return fresh_user

    def validate_context() -> dict[str, Any]:
        return current_authorized_user()

    def validate_session() -> None:
        fresh_user = current_authorized_user()
        if not payload.client_session_id:
            return
        try:
            LearningEventService(learning_logger).validate_session(
                user=fresh_user,
                client_session_id=payload.client_session_id,
                course_id=course_id,
                class_id=class_id,
            )
        except LearningEventError as exc:
            raise GeneralLearningError(
                exc.error_code,
                str(exc),
                status_code=_learning_error_status(exc.error_code),
            ) from exc

    def record_generated_event(metadata: dict[str, Any]) -> None:
        if not payload.client_session_id:
            return
        fresh_user = current_authorized_user()
        try:
            LearningEventService(learning_logger).record_server_event(
                user=fresh_user,
                client_session_id=payload.client_session_id,
                event_type=ServerLearningEventType.INNOVATION_SUGGESTION_GENERATED,
                metadata=metadata,
                course_id=course_id,
                class_id=class_id,
                dedupe_key=f"INNOVATION_SUGGESTION_GENERATED:{request_id}",
            )
        except LearningEventError as exc:
            raise GeneralLearningError(
                exc.error_code,
                str(exc),
                status_code=_learning_error_status(exc.error_code),
            ) from exc

    settings = get_settings()
    advisor = InnovationAdvisorService(
        storage=learning_logger,
        provider_factory=_create_innovation_suggestion_provider,
        data_policy=ExternalModelDataPolicy.from_ids(settings.external_model_course_allowlist),
        runtime=general_learning_runtime,
    )
    disconnected = threading.Event()
    watcher = asyncio.create_task(_watch_general_disconnect(request, disconnected))
    try:
        return await advisor.suggest(
            user=current_user,
            advisor_context=initial_context,
            use_course_references=payload.use_course_references,
            request_id=request_id,
            context_validator=validate_context,
            session_validator=validate_session,
            event_recorder=record_generated_event,
            disconnected=disconnected,
        )
    except GeneralLearningError as exc:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
        raise HTTPException(
            status_code=exc.status_code,
            headers=headers,
            detail={
                "error_code": exc.error_code,
                "message": str(exc),
                "request_id": request_id,
                "retryable": exc.retryable,
            },
        ) from exc
    finally:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass


@app.post("/api/innovation/sessions/{session_id}/answers")
def answer_innovation_question(
    session_id: int,
    payload: InnovationAnswerRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    try:
        return InnovationService(learning_logger, mcp_gateway=mcp_gateway).answer_question(
            user=current_user,
            session_id=session_id,
            question_id=payload.question_id,
            answer=payload.answer,
            client_session_id=payload.client_session_id,
        )
    except InnovationError as exc:
        status = _innovation_error_status(exc.error_code)
        raise HTTPException(status_code=status, detail={"error_code": exc.error_code, "message": str(exc)}) from exc


@app.post("/api/innovation/sessions/{session_id}/finalize")
def finalize_innovation_session(
    session_id: int,
    payload: InnovationFinalizeRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    try:
        return InnovationService(learning_logger, mcp_gateway=mcp_gateway).finalize_session(
            user=current_user,
            session_id=session_id,
            client_session_id=payload.client_session_id,
        )
    except InnovationError as exc:
        status = _innovation_error_status(exc.error_code)
        raise HTTPException(status_code=status, detail={"error_code": exc.error_code, "message": str(exc)}) from exc


@app.get("/api/connectors/status")
async def connectors_status() -> dict:
    service = ExternalResourceService(storage=learning_logger, registry=connector_registry)
    return {"connectors": await service.connector_status()}


@app.post("/api/connectors/{name}/sync")
async def sync_connector(
    name: str,
    query: ResourceQuery,
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
) -> dict:
    if name.strip().lower() == "fake":
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "FAKE_CONNECTOR_FORBIDDEN",
                "message": "fake 连接器仅允许在测试或本地框架验证中显式注入。",
            },
        )
    service = ExternalResourceService(storage=learning_logger, registry=connector_registry)
    try:
        return (
            await service.sync(
                name,
                query,
                deadline=time.monotonic() + 25.0,
                before_remote=lambda source_name: _consume_resource_rate_limit(
                    int(current_user["id"]), source_name
                ),
            )
        ).model_dump(mode="json")
    except LookupInputError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": exc.error_code, "message": str(exc)},
        ) from None
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "CONNECTOR_NOT_FOUND", "message": f"Connector 不存在：{name}"},
        ) from exc


@app.get("/api/connectors/sync-runs")
def connector_sync_runs(
    current_user: Annotated[dict, Depends(require_roles("teacher", "admin"))],
    limit: int = 50,
) -> dict:
    return {"items": learning_logger.connector_sync_runs(limit=limit)}


@app.post("/api/resources/search")
async def search_external_resources(
    query: ResourceQuery,
    connector_name: str | None = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    if connector_name and connector_name.strip().lower() == "fake":
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "FAKE_CONNECTOR_FORBIDDEN",
                "message": "fake 连接器仅允许在测试或本地框架验证中显式注入。",
            },
        )
    service = ExternalResourceService(storage=learning_logger, registry=connector_registry)
    try:
        return await service.search(
            query,
            connector_name=connector_name,
            deadline=time.monotonic() + 25.0,
            before_remote=lambda source_name: _consume_resource_rate_limit(
                int(current_user["id"]), source_name
            ),
        )
    except LookupInputError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": exc.error_code, "message": str(exc)},
        ) from None


def _consume_resource_rate_limit(user_id: int, source_name: str) -> None:
    _reserve_resource_rate_limits(user_id, [source_name])


def _reserve_resource_rate_limits(user_id: int, source_names: list[str]) -> None:
    now = time.monotonic()
    window = max(1.0, float(_RESOURCE_RATE_LIMIT_WINDOW_SECONDS))
    limit = max(1, int(_RESOURCE_RATE_LIMIT_MAX_CALLS))
    normalized = list(dict.fromkeys(str(name).strip().lower() for name in source_names if str(name).strip()))
    if not normalized:
        return
    with _RESOURCE_RATE_LOCK:
        cutoff = now - window
        buckets: list[deque[float]] = []
        for source_name in normalized:
            bucket = _RESOURCE_RATE_BUCKETS.setdefault((int(user_id), source_name), deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            buckets.append(bucket)
        exhausted = [bucket for bucket in buckets if len(bucket) >= limit]
        if exhausted:
            retry_after = max(
                1,
                max(math.ceil(window - (now - bucket[0])) for bucket in exhausted),
            )
            raise HTTPException(
                status_code=429,
                detail={
                    "error_code": "RESOURCE_RATE_LIMIT",
                    "message": "公共资料查询过于频繁，请稍后重试。",
                },
                headers={"Retry-After": str(retry_after)},
            )
        for bucket in buckets:
            bucket.append(now)


def _agent_remote_sources(tool_name: str, arguments: dict[str, Any]) -> list[str]:
    requested: list[str]
    if tool_name == "paper_search_real":
        source = str(arguments.get("source") or "auto").strip().lower()
        requested = [source] if source != "auto" else ["semantic_scholar", "arxiv"]
    else:
        requested = {
            "vuln_lookup_real": ["nvd"],
            "github_project_search": ["github"],
            "standard_lookup_real": ["rfc_editor"],
        }.get(tool_name, [])
    enabled_remote: list[str] = []
    for source_name in requested:
        try:
            connector = connector_registry.get(source_name)
        except KeyError:
            continue
        if connector.enabled and getattr(connector, "supports_remote", True):
            enabled_remote.append(source_name)
    return enabled_remote


def _reserve_agent_tool_rate_limit(user_id: int, tool_name: str, arguments: dict[str, Any]) -> None:
    _reserve_resource_rate_limits(user_id, _agent_remote_sources(tool_name, arguments))


def _record_question(
    user_id: str,
    question: str,
    result: dict,
    *,
    background_tasks: BackgroundTasks | None = None,
) -> int:
    allow_demo_defaults = result.get("course_id") is None and str(user_id) in {"anonymous_demo", "demo_student"}
    question_id = learning_logger.log_question(
        user_id,
        question,
        result,
        allow_demo_defaults=allow_demo_defaults,
    )
    run_id = result.get("run_id")
    if run_id:
        learning_logger.attach_question_to_agent_run(run_id, question_id)
    _schedule_auto_question_cluster_rebuild(
        result.get("course_id"),
        class_id=result.get("class_id"),
        question_id=question_id,
        background_tasks=background_tasks,
    )
    return question_id


def _schedule_auto_question_cluster_rebuild(
    course_id: int | None,
    *,
    class_id: int | None = None,
    question_id: int | None = None,
    background_tasks: BackgroundTasks | None = None,
) -> None:
    settings = get_settings()
    threshold = max(1, int(settings.question_cluster_auto_threshold))
    scoped_course_id = int(course_id) if course_id is not None else learning_logger.default_course_id()
    scoped_class_id = int(class_id) if class_id is not None else None
    if question_id is None:
        return
    ready = learning_logger.register_question_cluster_for_processing(
        question_id=int(question_id),
        course_id=scoped_course_id,
        class_id=scoped_class_id,
        threshold=threshold,
    )
    if not ready:
        return
    if background_tasks is not None:
        background_tasks.add_task(_run_auto_question_cluster_rebuild, scoped_course_id, scoped_class_id)
    else:
        _run_auto_question_cluster_rebuild(scoped_course_id, scoped_class_id)


def _run_auto_question_cluster_rebuild(course_id: int | None, class_id: int | None = None) -> None:
    settings = get_settings()
    scoped_course_id = int(course_id) if course_id is not None else learning_logger.default_course_id()
    worker_id = f"pid-{os.getpid()}-{uuid.uuid4().hex}"
    claimed = learning_logger.claim_question_cluster_job(
        scoped_course_id,
        class_id,
        worker_id=worker_id,
    )
    if claimed is None:
        return
    try:
        question_clustering_service_factory(learning_logger).rebuild(
            ClusterBuildOptions(
                course_id=scoped_course_id,
                class_id=class_id,
                distance_threshold=settings.question_cluster_distance_threshold,
                min_cluster_size=settings.question_cluster_min_size,
            )
        )
    except Exception as exc:
        error_code, error_message = _safe_cluster_error(exc)
        learning_logger.fail_question_cluster_job(
            scoped_course_id,
            class_id,
            worker_id=worker_id,
            error_code=error_code,
            error_message=error_message,
        )
        return
    learning_logger.complete_question_cluster_job(
        scoped_course_id,
        class_id,
        worker_id=worker_id,
    )


def _recover_auto_question_cluster_jobs() -> None:
    settings = get_settings()
    scopes = learning_logger.recover_question_cluster_jobs(
        threshold=max(1, int(settings.question_cluster_auto_threshold)),
    )
    for course_id, class_id in scopes:
        _run_auto_question_cluster_rebuild(course_id, class_id)


def _safe_cluster_error(exc: Exception) -> tuple[str, str]:
    error_code = f"{type(exc).__name__.upper()}_CLUSTER_REBUILD"
    message = " ".join(str(exc).split())
    message = re.sub(
        r"(?i)(api[_ -]?key|token|secret|password)\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        message,
    )
    return error_code[:80], message[:240] or "cluster rebuild failed"


def _record_optional_learning_event(
    *,
    current_user: dict | None,
    client_session_id: str | None,
    event_type: str,
    page_name: str = "",
    module: str = "",
    assignment_id: str | None = None,
    metadata: dict | None = None,
    course_id: int | None = None,
    class_id: int | None = None,
) -> None:
    if current_user is None or not client_session_id:
        return
    try:
        LearningEventService(learning_logger).record_event(
            user=current_user,
            client_session_id=client_session_id,
            event_type=event_type,
            page_name=page_name,
            module=module,
            assignment_id=assignment_id,
            metadata=metadata or {},
            dedupe_key=None,
            course_id=course_id,
            class_id=class_id,
        )
    except LearningEventError:
        return


def _sqlite_health() -> dict[str, str]:
    try:
        count = len(learning_logger.documents())
        return {"status": "ok", "detail": f"SQLite 可用，documents={count}"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def _chroma_health() -> dict[str, str]:
    settings = get_settings()
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(settings.vector_store_dir))
        collections = [item.name for item in client.list_collections()]
        if settings.vector_collection_name in collections:
            detail = f"Chroma 可用，collection={settings.vector_collection_name}"
        else:
            detail = "Chroma 可用，尚未创建课程 collection"
        return {"status": "ok", "detail": detail}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def _chunks_for_vector_index(document_id: int, document: dict) -> list[dict]:
    chunks = learning_logger.chunks_for_document(document_id)
    enriched = []
    for chunk in chunks:
        enriched.append(
            {
                **chunk,
                "filename": document.get("filename", ""),
                "title": document.get("title", ""),
                "source_path": document.get("source_path", ""),
                "file_type": document.get("file_type", ""),
                "content_hash": document.get("content_hash", ""),
            }
        )
    return enriched


def _knowledge_course_dir(course_id: int | None) -> Path:
    if course_id is None:
        return KNOWLEDGE_BASE_DIR
    return KNOWLEDGE_BASE_DIR / f"course-{course_id}"


def _is_safe_knowledge_file(path: Path) -> bool:
    try:
        resolved_file = path.resolve()
        resolved_base = KNOWLEDGE_BASE_DIR.resolve()
    except OSError:
        return False
    return resolved_file.is_file() and resolved_file.is_relative_to(resolved_base)


def _effective_user_id(request_user_id: str, current_user: dict | None) -> str:
    if current_user is None:
        return request_user_id or "demo_student"
    return str(current_user["id"])


def _require_run_access(
    record: dict,
    current_user: dict | None,
    *,
    for_continue: bool,
) -> str:
    if current_user is None:
        raise HTTPException(
            status_code=401,
            detail={"error_code": "AUTH_REQUIRED", "message": "请先登录。"},
        )
    role = str(current_user["role"]).lower()
    context_status = str(record.get("context_status") or "invalid")
    if context_status != "valid":
        if role == "admin" and not for_continue:
            return context_status
        raise HTTPException(
            status_code=403,
            detail={"error_code": "RUN_CONTEXT_REQUIRED", "message": "Agent Run 缺少可信的课程上下文。"},
        )
    course_id = int(record["course_id"])
    class_id = int(record["class_id"]) if record.get("class_id") is not None else None
    if not _user_has_run_scope(current_user, course_id=course_id, class_id=class_id):
        raise HTTPException(
            status_code=403,
            detail={"error_code": "COURSE_FORBIDDEN", "message": "当前账号已无权访问该 Agent Run 的课程范围。"},
        )
    owner = str(record.get("user_id")) == str(current_user["id"])
    if for_continue:
        if owner:
            return context_status
        raise HTTPException(
            status_code=403,
            detail={"error_code": "RUN_CONTINUE_FORBIDDEN", "message": "只有 Agent Run 的创建者可以继续该任务。"},
        )
    if owner or role == "admin":
        return context_status
    if role == "teacher" and _teacher_can_review_run_user(
        teacher_id=int(current_user["id"]),
        target_user_id=record.get("user_id", ""),
        course_id=course_id,
        class_id=class_id,
    ):
        return context_status
    raise HTTPException(
        status_code=403,
        detail={"error_code": "FORBIDDEN", "message": "无权访问该 Agent Run。"},
    )


def _user_has_run_scope(user: dict, *, course_id: int, class_id: int | None) -> bool:
    role = str(user.get("role") or "").lower()
    if role == "admin":
        return int(course_id) in {int(item["id"]) for item in learning_logger.courses()}
    if role == "teacher":
        if class_id is not None:
            return learning_logger.teacher_in_class(int(user["id"]), class_id, course_id)
        return bool(learning_logger.classes_for_teacher(int(user["id"]), course_id=course_id))
    if role == "student":
        if class_id is not None:
            return learning_logger.user_in_class(int(user["id"]), class_id, course_id)
        return bool(learning_logger.classes_for_user(int(user["id"]), course_id=course_id))
    return False


def _teacher_can_review_run_user(
    *,
    teacher_id: int,
    target_user_id: str | int,
    course_id: int,
    class_id: int | None,
) -> bool:
    try:
        target_id = int(target_user_id)
    except (TypeError, ValueError):
        target = learning_logger.user_by_username(str(target_user_id))
        if target is None:
            return False
        target_id = int(target["id"])
    if class_id is not None:
        return (
            learning_logger.teacher_in_class(teacher_id, class_id, course_id)
            and learning_logger.user_in_class(target_id, class_id, course_id)
        )
    teacher_classes = {
        int(item["id"])
        for item in learning_logger.classes_for_teacher(teacher_id, course_id=course_id)
    }
    student_classes = {
        int(item["id"])
        for item in learning_logger.classes_for_user(target_id, course_id=course_id)
    }
    return bool(teacher_classes & student_classes)


_REDACTED_LOCAL_TOOLS = {"paper_search", "vuln_lookup", "standard_lookup"}


def _project_tool_calls(rows: list[dict]) -> list[dict]:
    projected: list[dict] = []
    for row in rows:
        item = dict(row)
        if str(item.get("tool_name") or "") in _REDACTED_LOCAL_TOOLS:
            item["tool_name"] = "redacted_legacy_tool"
            item["arguments_json"] = "{}"
            item["result_json"] = '{"redacted":true,"reason":"legacy_local_file_tool"}'
        projected.append(item)
    return projected


def _project_run_steps(rows: list[dict]) -> list[dict]:
    projected: list[dict] = []
    for row in rows:
        item = dict(row)
        decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
        observation = item.get("observation") if isinstance(item.get("observation"), dict) else {}
        tool_name = str(decision.get("tool_name") or observation.get("tool_name") or "")
        if tool_name in _REDACTED_LOCAL_TOOLS:
            item["decision"] = {"tool_name": "redacted_legacy_tool", "redacted": True}
            item["observation"] = {
                "tool_name": "redacted_legacy_tool",
                "redacted": True,
                "reason": "legacy_local_file_tool",
            }
        projected.append(item)
    return projected


def _project_run_traces(rows: list[dict]) -> list[dict]:
    projected: list[dict] = []
    for row in rows:
        item = dict(row)
        metadata = item.get("metadata")
        searchable = " ".join(
            [
                str(item.get("event_type") or ""),
                str(item.get("message") or ""),
                str(metadata or ""),
            ]
        )
        if any(name in searchable for name in _REDACTED_LOCAL_TOOLS):
            item["message"] = "Legacy local sample tool trace redacted."
            item["metadata"] = {"redacted": True, "reason": "legacy_local_file_tool"}
        projected.append(item)
    return projected


def _require_question_feedback_access(question_id: int, current_user: dict | None) -> None:
    if current_user is None:
        raise HTTPException(
            status_code=401,
            detail={"error_code": "AUTH_REQUIRED", "message": "请先登录。"},
        )
    with learning_logger._connect() as conn:
        row = conn.execute(
            "SELECT id, user_id FROM questions WHERE id = ?",
            (question_id,),
        ).fetchone()
    if row is None:
        return
    role = str(current_user["role"]).lower()
    owner_id = str(row["user_id"])
    if role == "admin":
        return
    if role == "student" and owner_id == str(current_user["id"]):
        return
    if role == "teacher" and learning_logger.teacher_can_access_user(int(current_user["id"]), owner_id):
        return
    raise HTTPException(
        status_code=403,
        detail={"error_code": "FORBIDDEN", "message": "无权反馈他人的问题。"},
    )


def _require_management_role(role: str | None, current_user: dict | None = None) -> None:
    if current_user is not None:
        if current_user["role"] in {"teacher", "admin"}:
            return
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "FORBIDDEN",
                "message": "当前角色无权执行教师管理操作。",
            },
        )
    if (role or "").strip().lower() not in {"teacher", "admin"}:
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "FORBIDDEN",
                "message": "当前角色无权执行教师管理操作。",
            },
        )

