from typing import Any, Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str | None = Field(default=None, pattern=r"^S\d+$", max_length=32)
    source: str
    title: str
    snippet: str
    score: float | None = None
    document_id: int | None = None
    chunk_id: int | None = None
    page: int | None = None
    section: str = ""
    source_type: str = ""
    source_url: str | None = None
    external_id: str | None = None
    resource_type: str = ""
    origin_kind: str = ""
    delivery_mode: str = ""
    fetched_at: str | None = None
    content_hash: str = ""


class SelectedTool(BaseModel):
    name: str
    description: str
    risk: str = "LOW"
    enabled: bool = True


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    user_id: str = "demo_student"
    conversation_id: str | None = None
    client_session_id: str | None = None
    course_id: int | None = None
    class_id: int | None = None
    assignment_id: int | None = None
    step_id: int | None = None
    use_course_references: bool = False


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    abstained: bool
    module: str
    intent: str
    tool: SelectedTool | None = None
    run_id: str | None = None
    run_status: str = "COMPLETED"
    current_step: int = 0
    requires_user_input: bool = False
    pending_question: str | None = None
    conversation_id: str | None = None
    question_id: int | None = None
    trace_id: str | None = None
    provider: str = ""
    model: str = ""
    fallback_used: bool = False
    stop_reason: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    course_id: int | None = None
    class_id: int | None = None
    assignment_id: int | None = None
    step_id: int | None = None
    delivery_mode: str = "none"
    course_reference_status: str | None = None


class AgentContinueRequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    user_id: str = "demo_student"
    conversation_id: str | None = None
    client_session_id: str | None = None
    assignment_id: int | None = None
    step_id: int | None = None
    use_course_references: bool = False


class AgentRunResponse(BaseModel):
    run_id: str
    status: str
    current_step: int
    max_steps: int
    tool_call_count: int
    max_tool_calls: int
    pending_question: str | None = None
    final_answer: str | None = None
    stop_reason: str | None = None
    trace_id: str | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    traces: list[dict[str, Any]] = Field(default_factory=list)
    context_status: str | None = None


class ComponentHealth(BaseModel):
    status: str
    detail: str = ""


class SystemStatusResponse(BaseModel):
    service: str
    provider: str
    model: str
    deepseek_configured: bool
    fallback_provider: str
    fallback_active: bool
    sqlite: ComponentHealth
    chroma: ComponentHealth
    mcp: ComponentHealth


class APIError(BaseModel):
    error_code: str
    message: str
    run_id: str | None = None
    trace_id: str | None = None
    course_reference_status: Literal[
        "allowed",
        "disabled_by_user",
        "blocked_by_policy",
        "revoked",
    ] | None = None


class PracticeGuideRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    user_id: str = "demo_student"
    client_session_id: str | None = None
    assignment_id: int | None = None
    step_id: int | None = None
    course_id: int | None = None
    class_id: int | None = None
    use_course_references: bool = False


class PracticeGuideResponse(ChatResponse):
    intent: str = "practice_guide"


class FeedbackRequest(BaseModel):
    question_id: int
    rating: int = Field(..., ge=1, le=5)
    helpful: bool | None = None
    comment: str = Field(default="", max_length=500)
    user_id: str = "demo_student"
    client_session_id: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: int
    question_id: int
    saved: bool
    message: str


class ToolInfo(BaseModel):
    name: str
    description: str
    risk: str = "LOW"
    enabled: bool = True
    status: str = "local_fallback"
    mcp_available: bool = False
    network: bool = False
    read_only: bool = True
    input_schema: dict[str, Any] | None = None


class UploadResponse(BaseModel):
    filename: str
    saved: bool
    message: str
    document_id: int | None = None
    content_hash: str | None = None
    chunk_count: int = 0
    duplicate: bool = False
    index_status: str = "PENDING"
    job_id: int | None = None


class KnowledgeDocument(BaseModel):
    id: int
    filename: str
    source_path: str
    file_type: str
    title: str
    content_hash: str
    byte_size: int
    text_length: int
    chunk_count: int
    index_status: str
    course_id: int | None = None
    class_id: int | None = None
    uploaded_by_user_id: int | None = None
    visibility: str = "course"
    created_at: str
    updated_at: str
    jobs: list[dict[str, Any]] = Field(default_factory=list)
    latest_job_status: str | None = None
    latest_job_updated_at: str | None = None
    latest_job_error: str | None = None
    published_to_students: bool = False
    published_by: int | None = None
    published_at: str | None = None
    publication_last_action: dict[str, Any] | None = None


class KnowledgeDocumentsResponse(BaseModel):
    documents: list[KnowledgeDocument]
    index_jobs: list[dict[str, Any]]


class KnowledgeActionResponse(BaseModel):
    document_id: int
    status: str
    message: str
    job_id: int | None = None
    chunk_count: int = 0
    published_to_students: bool | None = None
    published_by: int | None = None
    published_at: str | None = None
    publication_last_action: dict[str, Any] | None = None


class RecentQuestion(BaseModel):
    user_id: str
    question: str
    module: str
    intent: str
    confidence: float
    abstained: bool


class TeacherAnalyticsResponse(BaseModel):
    total_questions: int
    abstained_count: int
    average_confidence: float
    abstain_rate: float = 0.0
    module_distribution: dict[str, int]
    recent_questions: list[RecentQuestion]
    model_metrics: dict[str, Any] = Field(default_factory=dict)
    agent_metrics: dict[str, Any] = Field(default_factory=dict)
    tool_metrics: dict[str, Any] = Field(default_factory=dict)
    retrieval_metrics: dict[str, Any] = Field(default_factory=dict)
    module_stuck_points: list[dict[str, Any]] = Field(default_factory=list)
    feedback_metrics: dict[str, Any] = Field(default_factory=dict)


class FAQItem(BaseModel):
    representative_question: str
    count: int
    module: str
    intent: str


class FAQResponse(BaseModel):
    items: list[FAQItem]


class LearningSessionRequest(BaseModel):
    client_session_id: str = Field(..., min_length=1, max_length=128)
    course_id: int | None = None
    class_id: int | None = None


class LearningSessionResponse(BaseModel):
    id: int
    client_session_id: str
    user_id: int
    course_id: int
    class_id: int | None = None
    started_at: str
    last_active_at: str
    ended_at: str | None = None
    active_seconds: int = 0


class LearningEventRequest(BaseModel):
    client_session_id: str = Field(..., min_length=1, max_length=128)
    event_type: str = Field(..., min_length=1)
    page_name: str = Field(default="", max_length=120)
    module: str = Field(default="", max_length=120)
    resource_id: str | None = Field(default=None, max_length=120)
    assignment_id: str | None = Field(default=None, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict)
    dedupe_key: str | None = Field(default=None, max_length=200)
    course_id: int | None = None
    class_id: int | None = None


class LearningEventResponse(BaseModel):
    id: int
    session_id: int
    user_id: int
    course_id: int
    class_id: int | None = None
    event_type: str
    page_name: str = ""
    module: str = ""
    created_at: str
    duplicate: bool = False


class ActivitySummaryResponse(BaseModel):
    unique_active_students: int
    visit_count: int
    active_days: int
    total_active_seconds: int
    recent_active_at: str | None = None
    page_distribution: dict[str, int] = Field(default_factory=dict)
    event_distribution: dict[str, int] = Field(default_factory=dict)


class LearningTimelineResponse(BaseModel):
    events: list[dict[str, Any]]


class ResourceOpenRequest(BaseModel):
    resource_id: str = Field(..., min_length=1, max_length=160)
    client_session_id: str | None = Field(default=None, max_length=128)
    course_id: int | None = None
    class_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InnovationStartRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)
    current_plan: str = Field(..., min_length=1, max_length=4000)
    client_session_id: str | None = Field(default=None, max_length=128)
    course_id: int | None = None
    class_id: int | None = None


class InnovationAnswerRequest(BaseModel):
    question_id: int
    answer: str = Field(..., min_length=1, max_length=4000)
    client_session_id: str | None = Field(default=None, max_length=128)


class InnovationFinalizeRequest(BaseModel):
    client_session_id: str | None = Field(default=None, max_length=128)


class ClusterRebuildRequest(BaseModel):
    course_id: int | None = None
    class_id: int | None = None
    start_at: str | None = None
    end_at: str | None = None
    distance_threshold: float = Field(default=0.35, ge=0.0, le=2.0)
    min_cluster_size: int = Field(default=2, ge=1, le=20)


class ClusterRenameRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=120)


class ClusterMergeRequest(BaseModel):
    target_cluster_id: int
    source_cluster_ids: list[int] = Field(default_factory=list)
    label: str | None = Field(default=None, max_length=120)


class ClusterSplitRequest(BaseModel):
    question_ids: list[int] = Field(..., min_length=1)
    label: str = Field(..., min_length=1, max_length=120)


class AssignmentProgressRequest(BaseModel):
    step_id: int
    status: str = Field(default="in_progress", max_length=40)
    progress_percent: int = Field(default=0, ge=0, le=100)
    note: str = Field(default="", max_length=500)
    client_session_id: str | None = Field(default=None, max_length=128)


class StuckReportRequest(BaseModel):
    assignment_id: int
    step_id: int
    reason: str = Field(..., min_length=1, max_length=800)
    module: str = Field(default="", max_length=120)
    client_session_id: str | None = Field(default=None, max_length=128)


class StuckActionRequest(BaseModel):
    action: str = Field(..., pattern="^(confirm|ignore|resolve)$")
