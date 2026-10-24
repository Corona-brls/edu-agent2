"""Finite, evidence-grounded agent loop orchestration."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar
from urllib.parse import urlparse
from uuid import uuid4

from backend.agent.observation import Observation, retrieval_results_to_observation, sanitize_payload
from backend.agent.planner import AgentPlanner
from backend.agent.router import requested_source_slots, route_intent
from backend.agent.schemas import AgentAction, AgentIntent, ActionDecision, ToolSpec
from backend.agent.state import AgentRunStatus, AgentState, state_from_record
from backend.core.config import Settings, get_settings
from backend.core.errors import classify_llm_failure
from backend.guards.action_guard import ActionGuard, ActionGuardError
from backend.guards.citation_guard import (
    CITATION_RE,
    MULTI_SOURCE_TEACHING_BACKGROUND_PROFILE,
    CitationGuard,
    render_grounded_presentation,
)
from backend.guards.safety_guard import SafetyGuard
from backend.llm.base import LLMProvider
from backend.llm.data_policy import (
    COURSE_SPAN_PROFILE_GROUNDED,
    COURSE_SPAN_PROFILE_KEY,
    COURSE_SPAN_PROFILE_PLAIN,
    COURSE_SPAN_PROFILE_PRACTICE,
    ModelCallPreflightError,
    build_course_text_spans,
    is_course_evidence,
    select_relevant_course_text_spans,
)
from backend.llm.factory import create_llm_provider
from backend.llm.prompts import build_answer_messages, build_structured_messages, messages_sha256
from backend.llm.schemas import (
    EvidenceItem,
    EvidenceSpan,
    GroundedAnswer,
    GroundedClaim,
    GroundedPracticeGuide,
    GroundedSpanClaim,
    GroundedSpanAnswer,
    LLMRequest,
)
from backend.llm.template_provider import TemplateProvider
from backend.rag.course_document_constraint import (
    CourseDocumentConstraintError,
    has_explicit_course_document_constraint_syntax,
    normalize_course_document_title,
)
from backend.rag.hybrid_retriever import (
    classify_course_query_update,
    HybridRetriever,
    RetrievalResponse,
    RetrievalResult,
    query_fact_fingerprint,
    retrieval_snippet,
    resolve_course_document_constraint,
    supported_query_fact_fingerprint,
)
from backend.rag.retriever import LocalRetriever
from backend.rag.visibility import is_student_visible_document
from backend.resources.normalizer import canonical_authors
from backend.resources.query_intent import (
    LookupInputError,
    LookupTarget,
    matches_exact_identity,
    parse_lookup_target,
)
from backend.storage.logger import LearningLogger
from backend.tools.mcp_client import MCPClientGateway
from backend.tools.tool_policy import get_tool_definition


_AUTO_TOOL_ROUTE_REASON = "AUTO_TOOL_ROUTE"
_TRUSTED_ORIGINS = {"remote", "local_catalog"}
_TRUSTED_DELIVERY = {"live", "fresh_cache", "stale_cache", "local_metadata"}
_SOURCE_TO_TOOLS = {
    "paper": ("paper_search_real", "paper_search"),
    "vulnerability": ("vuln_lookup_real", "vuln_lookup"),
    "standard": ("standard_lookup_real", "standard_metadata_lookup", "standard_lookup"),
    "project": ("github_project_search",),
}
_TOOL_TO_SOURCE = {
    tool: source for source, tools in _SOURCE_TO_TOOLS.items() for tool in tools
}
_PRACTICE_EXTERNAL_SOURCE_TYPES = frozenset(_SOURCE_TO_TOOLS)
_PRACTICE_EXTERNAL_TEXT_MAX_CHARS = 4000
_PRACTICE_COURSE_ORIGINS = frozenset({"local_document", "course_document"})
_PRACTICE_MAX_COURSE_EVIDENCE = 5
_PRACTICE_SUPPORT_MAX_DOCUMENTS = 1
_PRACTICE_SUPPORT_MAX_CHUNKS_PER_DOCUMENT = 2
_PRACTICE_SUPPORT_MAX_CHARS = 6000
_PRACTICE_COURSE_ALREADY_COLLECTED = "PRACTICE_COURSE_ALREADY_COLLECTED"


def _is_local_course_evidence_shape(item: EvidenceItem) -> bool:
    return (
        str(item.resource_type or "").casefold() == "course"
        and str(item.origin_kind or "").casefold() == "local_document"
        and str(item.delivery_mode or "").casefold() == "local_rag"
        and not str(item.tool_name or "").strip()
        and item.document_id is not None
        and item.chunk_id is not None
    )


def _is_constrained_course_evidence(state: AgentState, item: EvidenceItem) -> bool:
    return (
        _is_local_course_evidence_shape(item)
        and int(item.document_id or 0) in set(state.allowed_course_document_ids)
        and int(item.chunk_id or 0) in state.allowed_course_chunk_hashes
    )
_PRACTICE_COURSE_NO_PROGRESS = "PRACTICE_COURSE_NO_PROGRESS"
_PRACTICE_READY_ABSTAIN_NORMALIZED = "PRACTICE_READY_ABSTAIN_NORMALIZED"
_PLANNER_NORMALIZATION_REASON_CODES = frozenset(
    {
        _PRACTICE_COURSE_ALREADY_COLLECTED,
        _PRACTICE_COURSE_NO_PROGRESS,
        _PRACTICE_READY_ABSTAIN_NORMALIZED,
    }
)

_PAPER_AUTHOR_SUMMARY_MAX_CHARS = 300
_PAPER_CONCEPT_VERSION = "network-security-bilingual-v1"
_PAPER_DOMAIN_CONCEPTS = {
    "5g": ("5g", "fifth generation", "第五代移动通信"),
    "core_network": ("core network", "5gc", "核心网"),
    "nas": ("non-access stratum", "nas", "非接入层"),
    "network_slicing": ("network slicing", "网络切片"),
    "ran": ("radio access network", "ran", "无线接入网"),
    "sdn_nfv": ("sdn", "nfv", "软件定义网络", "网络功能虚拟化"),
}
_PAPER_SECURITY_CONCEPTS = {
    "security": ("security", "secure", "安全"),
    "attack": ("attack", "attacks", "攻击"),
    "detection": ("detection", "detect", "检测", "识别"),
    "authentication": ("authentication", "authenticate", "认证", "鉴权"),
    "replay": ("replay", "重放"),
    "vulnerability": ("vulnerability", "vulnerabilities", "漏洞"),
    "intrusion_anomaly": ("intrusion", "anomaly", "入侵", "异常"),
    "privacy": ("privacy", "隐私"),
    "ddos": ("ddos", "distributed denial of service", "分布式拒绝服务"),
}
_PAPER_CONCEPT_QUERY_LABELS = {
    "5g": "5G",
    "core_network": "core network",
    "nas": "NAS",
    "network_slicing": "network slicing",
    "ran": "radio access network",
    "sdn_nfv": "SDN NFV",
    "attack": "attack",
    "security": "security",
    "detection": "detection",
    "authentication": "authentication",
    "replay": "replay",
    "vulnerability": "vulnerability",
    "intrusion_anomaly": "intrusion anomaly",
    "privacy": "privacy",
    "ddos": "DDoS",
}
_PAPER_STOPWORDS = frozenset(
    {
        "a", "an", "and", "about", "for", "from", "in", "of", "on", "or", "paper",
        "papers", "real", "return", "search", "source", "the", "to", "use", "with",
        "使用", "真实", "论文", "搜索", "检索", "查找", "返回", "篇", "给出", "提供", "来源",
        "标题", "作者", "摘要", "线索", "链接", "可核验", "不要", "本地", "样例", "代替", "结果",
    }
)
_PAPER_AUTHOR_PREFIX = "Author information: "
_PAPER_AUTHOR_EXCERPT_PREFIX = "Author information (excerpt, showing "
_PAPER_ABSTRACT_PREFIX = "Abstract: "

_ModelResult = TypeVar("_ModelResult")


def _optional_int(value: object | None) -> int | None:
    return int(value) if value is not None else None


def _assignment_context_changed() -> ModelCallPreflightError:
    return ModelCallPreflightError(
        "ASSIGNMENT_CONTEXT_CHANGED",
        "当前实验任务、步骤或本人学习状态已变化，请刷新后重试。",
        status_code=409,
        policy_status="revoked",
    )


@dataclass(frozen=True)
class _PaperRequestContract:
    required_fields: tuple[str, ...]
    target_count: int
    explicit_count: bool


@dataclass(frozen=True)
class _PaperSelection:
    contract: _PaperRequestContract
    selected: tuple[EvidenceItem, ...]
    requested_count: int
    related_count: int
    field_complete_count: int
    terminal_error: str | None = None
    completion_reason: str = "ANSWERED"
    shortfall_reason: str = ""


class GroundedAnswerValidationError(ValueError):
    def __init__(
        self,
        error_code: str,
        message: str,
        failure_detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.failure_detail = failure_detail


class AgentOrchestrator:
    def __init__(
        self,
        *,
        storage: LearningLogger | None = None,
        planner_provider: LLMProvider | None = None,
        answer_provider: LLMProvider | None = None,
        fallback_provider: LLMProvider | None = None,
        retriever: Any | None = None,
        mcp_gateway: MCPClientGateway | None = None,
        settings: Settings | None = None,
        model_call_guard: Callable[..., str | None] | None = None,
        remote_rate_limiter: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.storage = storage or LearningLogger()
        self.planner_provider = planner_provider or create_llm_provider()
        self.answer_provider = answer_provider or self.planner_provider
        self.fallback_provider = fallback_provider or TemplateProvider()
        self.retriever = retriever or HybridRetriever(storage=self.storage)
        self.mcp_gateway = mcp_gateway or MCPClientGateway(
            mcp_timeout_seconds=self.settings.mcp_tool_timeout_seconds
        )
        self.model_call_guard = model_call_guard
        self.remote_rate_limiter = remote_rate_limiter
        self.action_guard = ActionGuard()
        self.citation_guard = CitationGuard()
        self.safety_guard = SafetyGuard()

    def run(
        self,
        user_goal: str,
        *,
        user_id: str = "demo_student",
        course_id: int | None = None,
        class_id: int | None = None,
        run_id: str | None = None,
        supplemental_input: str | None = None,
        question_id: int | None = None,
        intent_override: str | None = None,
        assignment_id: int | None = None,
        step_id: int | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        restore_run = run_id is not None
        checkpoint = self.storage.agent_attempt_checkpoint(run_id) if run_id else None
        state = self._load_or_create_state(
            user_goal=user_goal,
            user_id=user_id,
            course_id=course_id,
            class_id=class_id,
            run_id=run_id,
            supplemental_input=supplemental_input,
            intent_override=intent_override,
            assignment_id=assignment_id,
            step_id=step_id,
        )
        self._refresh_course_document_constraint(state, preserve_identity=restore_run)
        if checkpoint is None:
            checkpoint = self.storage.agent_attempt_checkpoint(state.run_id)
        self._attempt_run_id = state.run_id
        self._attempt_checkpoint = checkpoint
        self._attempt_restore_run = restore_run
        self._attempt_trace_id = state.trace_id
        try:
            return self._run_state_machine(
                state,
                started=started,
                question_id=question_id,
            )
        except ModelCallPreflightError as exc:
            self.close_rejected_attempt(exc)
            raise

    def _refresh_course_document_constraint(
        self,
        state: AgentState,
        *,
        preserve_identity: bool,
    ) -> None:
        if state.intent != AgentIntent.COURSE_QA.value:
            state.course_document_constraint_title = ""
            state.course_document_snippet_query = ""
            state.allowed_course_document_ids = []
            state.allowed_course_document_hashes = {}
            state.allowed_course_chunk_hashes = {}
            return
        constraint_query = (
            self._course_document_identity_query(state)
            if preserve_identity and state.course_document_constraint_title
            else state.user_goal
        )
        try:
            title, allowed = resolve_course_document_constraint(
                self.storage,
                course_id=state.course_id,
                user_goal=constraint_query,
            )
        except CourseDocumentConstraintError as exc:
            self._reject_course_document_constraint(state, exc.error_code)
            return
        if title is None or allowed is None:
            state.course_document_constraint_title = ""
            state.course_document_snippet_query = ""
            state.allowed_course_document_ids = []
            state.allowed_course_document_hashes = {}
            state.allowed_course_chunk_hashes = {}
            return
        try:
            document_hashes, chunk_hashes = self._course_document_constraint_snapshot(allowed)
        except CourseDocumentConstraintError as exc:
            self._reject_course_document_constraint(state, exc.error_code)
            return
        except Exception:
            self._reject_course_document_constraint(
                state,
                "COURSE_DOCUMENT_CONSTRAINT_UNAVAILABLE",
            )
            return
        if preserve_identity and state.course_document_constraint_title:
            if (
                normalize_course_document_title(state.course_document_constraint_title) != title
                or set(state.allowed_course_document_ids) != set(allowed)
                or state.allowed_course_document_hashes != document_hashes
                or state.allowed_course_chunk_hashes != chunk_hashes
            ):
                self._reject_course_document_constraint(
                    state,
                    "COURSE_DOCUMENT_CONTEXT_CHANGED",
                )
                return
        state.course_document_constraint_title = title
        if not state.course_document_snippet_query:
            state.course_document_snippet_query = state.user_goal
        state.allowed_course_document_ids = sorted(allowed)
        state.allowed_course_document_hashes = document_hashes
        state.allowed_course_chunk_hashes = chunk_hashes

    def _course_document_constraint_snapshot(
        self,
        allowed_document_ids: set[int] | frozenset[int],
    ) -> tuple[dict[int, str], dict[int, str]]:
        document_hashes: dict[int, str] = {}
        chunk_hashes: dict[int, str] = {}
        for document_id in sorted(int(value) for value in allowed_document_ids):
            document = self.storage.document(document_id)
            if document is None or not is_student_visible_document(document):
                raise CourseDocumentConstraintError("COURSE_DOCUMENT_CONTEXT_CHANGED")
            document_hashes[document_id] = str(document.get("content_hash") or "")
            for chunk in self.storage.chunks_for_document(document_id):
                if str(chunk.get("index_status") or "").upper() == "INDEXED":
                    chunk_hashes[int(chunk["id"])] = str(chunk.get("chunk_hash") or "")
        if not chunk_hashes:
            raise CourseDocumentConstraintError("COURSE_DOCUMENT_CONTEXT_CHANGED")
        return document_hashes, chunk_hashes

    @staticmethod
    def _course_document_identity_query(state: AgentState) -> str:
        original_goal = state.user_goal.splitlines()[0]
        if has_explicit_course_document_constraint_syntax(original_goal):
            return original_goal
        if has_explicit_course_document_constraint_syntax(
            state.course_document_snippet_query
        ):
            return state.course_document_snippet_query
        return state.user_goal

    def _validate_course_document_constraint_before_answer(self, state: AgentState) -> bool:
        if not state.course_document_constraint_title:
            return True
        try:
            title, allowed = resolve_course_document_constraint(
                self.storage,
                course_id=state.course_id,
                user_goal=self._course_document_identity_query(state),
            )
            if title is None or allowed is None:
                raise CourseDocumentConstraintError("COURSE_DOCUMENT_CONTEXT_CHANGED")
            document_hashes, chunk_hashes = self._course_document_constraint_snapshot(allowed)
        except CourseDocumentConstraintError as exc:
            self._reject_course_document_constraint(state, exc.error_code)
            return False
        except Exception:
            self._reject_course_document_constraint(
                state,
                "COURSE_DOCUMENT_CONSTRAINT_UNAVAILABLE",
            )
            return False
        if (
            title != normalize_course_document_title(state.course_document_constraint_title)
            or set(allowed) != set(state.allowed_course_document_ids)
            or document_hashes != state.allowed_course_document_hashes
            or chunk_hashes != state.allowed_course_chunk_hashes
        ):
            self._reject_course_document_constraint(state, "COURSE_DOCUMENT_CONTEXT_CHANGED")
            return False

        allowed_ids = set(allowed)
        course_evidence = [
            item
            for item in state.evidence
            if _is_constrained_course_evidence(state, item)
        ]
        try:
            current_chunks = {
                int(chunk["id"]): chunk
                for chunk in self.storage.document_chunks(course_id=state.course_id)
                if int(chunk["document_id"]) in allowed_ids
                and str(chunk.get("index_status") or "").upper() == "INDEXED"
            }
        except Exception:
            self._reject_course_document_constraint(
                state,
                "COURSE_DOCUMENT_CONSTRAINT_UNAVAILABLE",
            )
            return False
        invalid: list[EvidenceItem] = [
            item
            for item in state.evidence
            if not _is_constrained_course_evidence(state, item)
        ]
        for item in course_evidence:
            current = current_chunks.get(int(item.chunk_id or 0))
            expected_text = (
                self.safety_guard.sanitize_untrusted_text(
                    retrieval_snippet(
                        str(current.get("chunk_text") or ""),
                        query=state.course_document_snippet_query or state.user_goal,
                    )
                )
                if current is not None
                else ""
            )
            if (
                item.document_id not in allowed_ids
                or current is None
                or int(current["document_id"]) != int(item.document_id or 0)
                or normalize_course_document_title(item.title) != title
                or str(item.source or "") != str(current.get("filename") or "")
                or str(item.content_hash or "") != str(current.get("content_hash") or "")
                or item.text != expected_text
            ):
                invalid.append(item)
        if invalid:
            state.evidence = [item for item in state.evidence if item not in invalid]
            self._reject_course_document_constraint(
                state,
                "COURSE_DOCUMENT_CONSTRAINT_VIOLATION",
            )
            return False
        if not course_evidence:
            self._reject_course_document_constraint(state, "INSUFFICIENT_EVIDENCE")
            return False
        return True

    @staticmethod
    def _reject_course_document_constraint(state: AgentState, error_code: str) -> None:
        state.status = AgentRunStatus.ABSTAINED
        state.stop_reason = error_code
        state.final_answer = {
            "COURSE_DOCUMENT_NOT_FOUND": "当前课程中没有可用的指定教材，已停止回答。",
            "COURSE_DOCUMENT_AMBIGUOUS": "当前课程中指定教材名称不唯一，已停止回答。",
            "COURSE_DOCUMENT_CONSTRAINT_INVALID": "指定教材的表达不受支持，已停止回答。",
            "COURSE_DOCUMENT_CONTEXT_CHANGED": "指定教材状态已变化，请刷新后重试。",
        }.get(error_code, "指定教材暂时无法核验，已停止回答。")

    def _run_state_machine(
        self,
        state: AgentState,
        *,
        started: float,
        question_id: int | None,
    ) -> dict[str, Any]:
        self._persist_state(state, question_id=question_id)
        self._trace(state, None, question_id, "run_started", "Agent run started")

        while state.status == AgentRunStatus.RUNNING:
            if self._timed_out(started):
                self._finish_with_limit(state, "AGENT_TIMEOUT", "Agent timed out.")
                break
            if state.current_step >= state.max_steps:
                self._finish_with_limit(state, "MAX_STEPS", "Agent reached the step limit.")
                break

            step_number = state.current_step + 1
            step_id = str(uuid4())
            self.storage.create_agent_step(
                run_id=state.run_id,
                step_id=step_id,
                step_number=step_number,
                action="planning",
                status=state.status.value,
                trace_id=state.trace_id,
            )
            planner_decision = self._decide(state, step_id, question_id)
            decision = planner_decision
            decision = self._prefer_explicit_tool_request(state, decision)
            decision = self._enforce_initial_retrieval(state, decision)
            decision = self._normalize_decision(state, decision)
            decision = self._bind_scoped_tool_arguments(state, decision)
            guard_observation = self._validate_decision(
                state=state,
                decision=decision,
                step_id=step_id,
                step_number=step_number,
                started_at=started,
            )
            state.current_step = step_number
            decision_metadata: dict[str, Any] = {
                "decision": decision.model_dump(mode="json")
            }
            normalization_audit = _planner_normalization_audit(planner_decision, decision)
            if normalization_audit is not None:
                decision_metadata["normalization"] = normalization_audit
            self._trace(
                state,
                step_id,
                question_id,
                "decision",
                f"Planner selected {decision.action.value}",
                decision_metadata,
            )
            observation = guard_observation
            if observation is None:
                observation = self._execute_decision(
                    state=state,
                    decision=decision,
                    step_id=step_id,
                    step_number=step_number,
                    question_id=question_id,
                    started_at=started,
                )
            if state.status != AgentRunStatus.RUNNING:
                self._check_model_access(
                    provider=self.answer_provider,
                    state=state,
                    request=None,
                    phase="result",
                    stage="before_result_write",
                )
            self.storage.update_agent_step(
                step_id=step_id,
                action=decision.action.value,
                status=state.status.value,
                decision=decision.model_dump(mode="json"),
                observation=observation.model_dump(mode="json") if observation else {},
                trace_id=state.trace_id,
                error_code=observation.error_code if observation else None,
            )
            self._persist_state(state, question_id=question_id)

        self._persist_state(state, question_id=question_id)
        return self._to_chat_response(state)

    def close_rejected_attempt(self, exc: ModelCallPreflightError) -> None:
        run_id = getattr(self, "_attempt_run_id", None)
        checkpoint = getattr(self, "_attempt_checkpoint", None)
        trace_id = getattr(self, "_attempt_trace_id", None)
        if not run_id or not isinstance(checkpoint, dict) or not trace_id:
            return
        self.storage.close_rejected_agent_attempt(
            run_id=str(run_id),
            checkpoint=checkpoint,
            error_code=exc.error_code,
            trace_id=str(trace_id),
            restore_run=bool(getattr(self, "_attempt_restore_run", False)),
        )
        exc.run_id = str(run_id)

    def continue_run(
        self,
        run_id: str,
        user_input: str,
        *,
        user_id: str = "demo_student",
        course_id: int | None = None,
        class_id: int | None = None,
        question_id: int | None = None,
        assignment_id: int | None = None,
        step_id: int | None = None,
    ) -> dict[str, Any]:
        return self.run(
            user_input,
            user_id=user_id,
            course_id=course_id,
            class_id=class_id,
            run_id=run_id,
            supplemental_input=user_input,
            question_id=question_id,
            assignment_id=assignment_id,
            step_id=step_id,
        )

    def _load_or_create_state(
        self,
        *,
        user_goal: str,
        user_id: str,
        course_id: int | None,
        class_id: int | None,
        run_id: str | None,
        supplemental_input: str | None,
        intent_override: str | None,
        assignment_id: int | None = None,
        step_id: int | None = None,
    ) -> AgentState:
        if run_id:
            record = self.storage.agent_run(run_id)
            if record is None:
                raise ValueError(f"Agent run does not exist: {run_id}")
            state = state_from_record(record)
            self._bind_or_validate_learning_context(
                state,
                assignment_id=assignment_id,
                step_id=step_id,
                allow_new_binding=False,
            )
            if not state.demo_compatibility:
                state.evidence = [
                    item
                    for item in state.evidence
                    if _evidence_is_currently_available(item, state.course_id, self.storage)
                ]
            available_sources = {
                str(item.resource_type or item.source_type or "course") for item in state.evidence
            }
            state.completed_sources = [
                source
                for source in state.completed_sources
                if source != "course" or "course" in available_sources
            ]
            if state.status != AgentRunStatus.REQUIRES_INPUT:
                raise ValueError(f"Agent run cannot continue from status: {state.status}")
            state.status = AgentRunStatus.RUNNING
            state.pending_question = None
            state.add_user_message(supplemental_input or user_goal)
            original_goal = state.user_goal.splitlines()[0]
            state.user_goal = f"{original_goal}\n\u8865\u5145\u4fe1\u606f: {supplemental_input or user_goal}"
            supplemental_text = supplemental_input or user_goal
            existing_constraint = bool(state.course_document_constraint_title) or (
                has_explicit_course_document_constraint_syntax(original_goal)
            )
            supplemental_constraint = has_explicit_course_document_constraint_syntax(
                supplemental_text
            )
            if existing_constraint and supplemental_constraint and state.course_document_constraint_title:
                try:
                    supplemental_title, supplemental_allowed = resolve_course_document_constraint(
                        self.storage,
                        course_id=state.course_id,
                        user_goal=supplemental_text,
                    )
                except CourseDocumentConstraintError as exc:
                    self._reject_course_document_constraint(state, exc.error_code)
                    return state
                if (
                    supplemental_title
                    != normalize_course_document_title(state.course_document_constraint_title)
                    or supplemental_allowed is None
                    or set(supplemental_allowed) != set(state.allowed_course_document_ids)
                ):
                    self._reject_course_document_constraint(
                        state,
                        "COURSE_DOCUMENT_CONTEXT_CHANGED",
                    )
                    return state
            if existing_constraint or supplemental_constraint:
                dropped = [
                    item
                    for item in state.evidence
                    if not _is_local_course_evidence_shape(item)
                ]
                if supplemental_constraint and not existing_constraint:
                    dropped = list(state.evidence)
                    state.evidence = []
                    state.completed_sources = []
                    state.course_document_constraint_title = ""
                    state.course_document_snippet_query = supplemental_text
                    state.allowed_course_document_ids = []
                    state.allowed_course_document_hashes = {}
                    state.allowed_course_chunk_hashes = {}
                elif dropped:
                    state.evidence = [item for item in state.evidence if item not in dropped]
                    state.completed_sources = [
                        source for source in state.completed_sources if source == "course"
                    ]
                if existing_constraint and state.course_document_constraint_title:
                    previous_query = state.course_document_snippet_query or original_goal
                    query_update = classify_course_query_update(
                        previous_query,
                        supplemental_text,
                    )
                    if query_update.update_type != "control_only":
                        removed_course = [
                            item for item in state.evidence if _is_local_course_evidence_shape(item)
                        ]
                        state.evidence = [
                            item for item in state.evidence if item not in removed_course
                        ]
                        state.completed_sources = [
                            source for source in state.completed_sources if source != "course"
                        ]
                        state.course_document_snippet_query = query_update.effective_query
                        state.add_observation(
                            Observation(
                                step_number=max(1, state.current_step or 1),
                                action="constraint_query_refresh",
                                success=True,
                                content="{}",
                                summary="The explicit textbook facts changed; bounded local evidence will be refreshed.",
                                metadata={
                                    "content_trust": "trusted_system",
                                    "dropped_evidence_count": len(removed_course),
                                    "update_type": query_update.update_type,
                                    "previous_fact_count": len(
                                        query_update.previous_fact_fingerprint
                                    ),
                                    "effective_fact_count": len(
                                        query_update.effective_fact_fingerprint
                                    ),
                                    "required_fact_fingerprint": list(
                                        query_update.required_fact_fingerprint
                                    ),
                                },
                            )
                        )
                    else:
                        state.course_document_snippet_query = previous_query
                if dropped:
                    state.add_observation(
                        Observation(
                            step_number=max(1, state.current_step or 1),
                            action="constraint_scope_reset",
                            success=True,
                            content="{}",
                            summary="Prior non-course evidence was removed for an explicit course-document constraint.",
                            metadata={
                                "content_trust": "trusted_system",
                                "dropped_evidence_count": len(dropped),
                            },
                        )
                    )
                state.intent = AgentIntent.COURSE_QA.value
                state.required_sources = ["course"]
            else:
                scope_changed = _apply_external_continue_scope(
                    state,
                    supplemental_text,
                )
                supplemental_intent = route_intent(supplemental_text)
                if not scope_changed and state.intent == AgentIntent.COURSE_QA.value and supplemental_intent in {
                    AgentIntent.PAPER_SEARCH.value,
                    AgentIntent.VULN_LOOKUP.value,
                    AgentIntent.STANDARD_LOOKUP.value,
                    AgentIntent.PROJECT_SEARCH.value,
                    AgentIntent.MULTI_SOURCE.value,
                }:
                    state.intent = supplemental_intent
                    state.required_sources = requested_source_slots(
                        supplemental_text,
                        supplemental_intent,
                    )
            if not state.required_sources:
                state.required_sources = requested_source_slots(original_goal, state.intent)
            return state

        intent = intent_override or route_intent(user_goal)
        demo_compatibility = course_id is None and not str(user_id).isdigit()
        if demo_compatibility:
            course_id = self.storage.default_course_id()
            class_id = self.storage.default_class_id() if class_id is None else class_id
        catalog_loader = getattr(self.mcp_gateway, "get_agent_tool_catalog", None)
        catalog_role = "demo" if demo_compatibility else "student"
        catalog = catalog_loader(role=catalog_role) if callable(catalog_loader) else []
        state = AgentState(
            user_id=user_id,
            course_id=course_id,
            class_id=class_id,
            demo_compatibility=demo_compatibility,
            user_goal=user_goal,
            intent=intent,
            max_steps=max(1, self.settings.agent_max_steps),
            max_tool_calls=max(0, self.settings.agent_max_tool_calls),
            available_tools=[ToolSpec.model_validate(tool) for tool in catalog],
            required_sources=requested_source_slots(user_goal, intent),
        )
        self._bind_or_validate_learning_context(
            state,
            assignment_id=assignment_id,
            step_id=step_id,
            allow_new_binding=True,
        )
        state.add_user_message(user_goal)
        self.storage.create_agent_run(
            run_id=state.run_id,
            user_id=user_id,
            goal=user_goal,
            intent=state.intent,
            status=state.status.value,
            max_steps=state.max_steps,
            max_tool_calls=state.max_tool_calls,
            state=state.model_dump(mode="json"),
            course_id=course_id,
            class_id=class_id,
        )
        return state

    def _course_constraint_projection_changed(
        self,
        state: AgentState,
        *,
        previous_query: str,
        candidate_query: str,
    ) -> bool:
        if not previous_query.strip():
            return True
        return query_fact_fingerprint(previous_query) != query_fact_fingerprint(
            candidate_query
        )

    def _bind_or_validate_learning_context(
        self,
        state: AgentState,
        *,
        assignment_id: int | None,
        step_id: int | None,
        allow_new_binding: bool,
    ) -> None:
        bound_assignment = state.assignment_context_assignment_id
        bound_step = state.assignment_context_step_id
        if bound_assignment is not None:
            if assignment_id is not None and int(assignment_id) != int(bound_assignment):
                raise _assignment_context_changed()
            if step_id is not None and _optional_int(step_id) != _optional_int(bound_step):
                raise _assignment_context_changed()
            assignment_id = int(bound_assignment)
            step_id = _optional_int(bound_step)
        elif not allow_new_binding and (assignment_id is not None or step_id is not None):
            raise _assignment_context_changed()
        if assignment_id is None and step_id is None:
            state.learning_context = {}
            state.learning_context_hash = ""
            return
        context, resolved_assignment_id, resolved_step_id, snapshot_hash = self._current_learning_context(
            state,
            assignment_id=assignment_id,
            step_id=step_id,
        )
        if state.learning_context_hash and state.learning_context_hash != snapshot_hash:
            raise _assignment_context_changed()
        state.assignment_context_assignment_id = resolved_assignment_id
        state.assignment_context_step_id = resolved_step_id
        state.learning_context = context
        state.learning_context_hash = snapshot_hash

    def _current_learning_context(
        self,
        state: AgentState,
        *,
        assignment_id: int | None,
        step_id: int | None,
    ) -> tuple[dict[str, Any], int, int | None, str]:
        step = self.storage.assignment_step(int(step_id)) if step_id is not None else None
        if step_id is not None and step is None:
            raise _assignment_context_changed()
        resolved_assignment_id = int(
            assignment_id if assignment_id is not None else step.get("assignment_id")
        )
        assignment = self.storage.assignment(resolved_assignment_id)
        if assignment is None or int(assignment.get("course_id") or 0) != int(state.course_id or 0):
            raise _assignment_context_changed()
        if step is not None and int(step.get("assignment_id") or 0) != resolved_assignment_id:
            raise _assignment_context_changed()
        resolved_step_id = int(step["id"]) if step is not None else None
        progress = next(
            (
                item for item in self.storage.assignment_progress(
                    user_id=int(state.user_id), assignment_id=resolved_assignment_id
                )
                if resolved_step_id is None or int(item.get("step_id") or 0) == resolved_step_id
            ),
            None,
        )
        blockers = [
            item for item in self.storage.stuck_reports()
            if int(item.get("user_id") or 0) == int(state.user_id)
            and int(item.get("assignment_id") or 0) == resolved_assignment_id
            and (resolved_step_id is None or int(item.get("step_id") or 0) == resolved_step_id)
            and str(item.get("status") or "").casefold() in {"active", "confirmed"}
        ]
        blocker = blockers[0] if blockers else None
        safe = self.safety_guard.sanitize_untrusted_text
        context: dict[str, Any] = {
            "assignment_title": safe(str(assignment.get("title") or ""))[:160],
            "assignment_description": safe(str(assignment.get("description") or ""))[:500],
            "module": safe(str((step or assignment).get("module") or ""))[:120],
            "step_index": int(step.get("step_index") or 0) if step is not None else None,
            "step_title": safe(str(step.get("title") or ""))[:160] if step is not None else "",
            "step_description": safe(str(step.get("description") or ""))[:500] if step is not None else "",
            "progress_status": safe(str(progress.get("status") or ""))[:40] if progress else "",
            "progress_percent": int(progress.get("progress_percent") or 0) if progress else None,
            "progress_note": safe(str(progress.get("note") or ""))[:500] if progress else "",
            "blocker_status": safe(str(blocker.get("status") or ""))[:40] if blocker else "",
            "blocker_reason": safe(str(blocker.get("reason") or ""))[:500] if blocker else "",
        }
        snapshot = {
            "assignment_id": resolved_assignment_id,
            "step_id": resolved_step_id,
            "context": context,
        }
        snapshot_hash = hashlib.sha256(
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return context, resolved_assignment_id, resolved_step_id, snapshot_hash

    def validate_learning_context(self, state: AgentState) -> None:
        if state.assignment_context_assignment_id is None:
            return
        _context, _assignment_id, _step_id, snapshot_hash = self._current_learning_context(
            state,
            assignment_id=state.assignment_context_assignment_id,
            step_id=state.assignment_context_step_id,
        )
        if snapshot_hash != state.learning_context_hash:
            raise _assignment_context_changed()

    def _decide(self, state: AgentState, step_id: str, question_id: int | None) -> ActionDecision:
        started = time.perf_counter()
        try:
            decision = self._guarded_model_call(
                provider=self.planner_provider,
                state=state,
                request=None,
                phase="planner",
                call=lambda: AgentPlanner(self.planner_provider).decide(state),
            )
            self.storage.log_model_call(
                run_id=state.run_id,
                step_id=step_id,
                question_id=question_id,
                provider=self.planner_provider.provider_name,
                model=self.planner_provider.model_name,
                purpose="decide_next_action",
                success=True,
                latency_ms=(time.perf_counter() - started) * 1000,
                metadata={"state_sha256": _sha256_text(state.to_summary())},
            )
            return decision
        except ModelCallPreflightError:
            raise
        except Exception as exc:
            failure_metadata = classify_llm_failure(exc, phase="decide_next_action")
            self.storage.log_model_call(
                run_id=state.run_id,
                step_id=step_id,
                question_id=question_id,
                provider=self.planner_provider.provider_name,
                model=self.planner_provider.model_name,
                purpose="decide_next_action",
                success=False,
                error_code="MODEL_DECISION_FAILED",
                error_message="Model decision failed",
                latency_ms=(time.perf_counter() - started) * 1000,
                fallback_used=True,
                metadata={
                    "state_sha256": _sha256_text(state.to_summary()),
                    "fallback_path": "deterministic_evidence_policy",
                    "fallback_trigger": "MODEL_DECISION_FAILED",
                    "error_type": failure_metadata["root_error_type"],
                    **failure_metadata,
                },
            )
            self._trace(
                state,
                step_id,
                question_id,
                "model_error",
                "Planner failed; applying deterministic evidence policy.",
                {
                    "error_type": failure_metadata["root_error_type"],
                    "failure_category": failure_metadata["failure_category"],
                    "phase": failure_metadata["phase"],
                },
            )
            state.fallback_used = True
            if state.evidence and not self._missing_sources(state):
                return ActionDecision(
                    action=AgentAction.ANSWER,
                    arguments={"fallback": True},
                    reason="Planner failed but required evidence is available.",
                    task_complete=True,
                )
            return self._decision_for_missing_source(state, reason="Planner unavailable")

    def _prefer_explicit_tool_request(self, state: AgentState, decision: ActionDecision) -> ActionDecision:
        if state.current_step != 0 or state.evidence or state.observations:
            return decision
        if decision.action == AgentAction.ASK_USER:
            return decision
        if state.intent in {
            AgentIntent.PRACTICE_GUIDE.value,
            AgentIntent.INNOVATION.value,
            AgentIntent.MULTI_SOURCE.value,
            AgentIntent.OFF_TOPIC.value,
        }:
            return decision
        if decision.action == AgentAction.CALL_TOOL and decision.tool_name == "course_retriever":
            return decision
        tool_names = {tool.name for tool in state.available_tools}
        query = state.user_goal.strip()
        lowered = query.lower()
        if any(token in lowered for token in ("mermaid", "diagram", "flowchart", "mindmap")):
            diagram_type = "sequence" if "sequence" in lowered else "mindmap" if "mindmap" in lowered else "flowchart"
            return self._tool_decision(
                "mermaid_generator",
                {"topic": query[:200], "diagram_type": diagram_type},
                tool_names,
                "explicit diagram request",
                decision,
            )
        slot = state.required_sources[0] if state.required_sources else None
        if slot in _SOURCE_TO_TOOLS:
            tool_name = self._available_tool_for_source(state, slot)
            if tool_name:
                return self._tool_decision(
                    tool_name,
                    {"query": query[:500], "top_k": 3},
                    tool_names,
                    f"explicit {slot} request",
                    decision,
                )
        return decision

    def _enforce_initial_retrieval(self, state: AgentState, decision: ActionDecision) -> ActionDecision:
        if state.current_step != 0 or state.evidence:
            return decision
        if state.intent == AgentIntent.PRACTICE_GUIDE.value and decision.action != AgentAction.ASK_USER:
            return ActionDecision(
                action=AgentAction.RETRIEVE,
                arguments={"query": state.user_goal},
                reason="Practice guidance requires course evidence before synthesis.",
            )
        if state.intent != AgentIntent.COURSE_QA.value:
            return decision
        if decision.action == AgentAction.RETRIEVE and isinstance(self.planner_provider, TemplateProvider):
            return decision.model_copy(update={"arguments": {"query": state.user_goal}})
        if decision.action not in {AgentAction.ANSWER, AgentAction.ABSTAIN}:
            return decision
        return ActionDecision(
            action=AgentAction.RETRIEVE,
            arguments={"query": state.user_goal},
            reason="Course questions require retrieval before answering.",
        )

    def _normalize_decision(self, state: AgentState, decision: ActionDecision) -> ActionDecision:
        if state.intent == AgentIntent.OFF_TOPIC.value:
            return ActionDecision(
                action=AgentAction.ABSTAIN,
                arguments={"message": "\u8be5\u95ee\u9898\u4e0d\u5728\u5f53\u524d\u8bfe\u7a0b\u77e5\u8bc6\u8303\u56f4\u5185\uff0c\u65e0\u6cd5\u57fa\u4e8e\u8bfe\u7a0b\u8bc1\u636e\u56de\u7b54\u3002"},
                reason="The request is outside the course scope.",
                task_complete=True,
            )
        if (
            state.course_document_constraint_title
            and decision.action == AgentAction.CALL_TOOL
        ):
            return ActionDecision(
                action=AgentAction.RETRIEVE,
                arguments={
                    "query": state.course_document_snippet_query or state.user_goal
                },
                reason="Explicit course-document constraints use only the bounded local retriever.",
            )
        if (
            state.course_document_constraint_title
            and decision.action == AgentAction.RETRIEVE
            and any(
                observation.action == "constraint_query_refresh"
                for observation in reversed(state.observations)
            )
        ):
            return decision.model_copy(
                update={
                    "arguments": {
                        "query": state.course_document_snippet_query or state.user_goal
                    }
                }
            )
        missing = self._missing_sources(state)
        if decision.action == AgentAction.ASK_USER:
            if not missing and state.evidence:
                return ActionDecision(
                    action=AgentAction.ANSWER,
                    reason="All required evidence slots are satisfied; synthesize the final answer.",
                    task_complete=True,
                )
            return decision
        if decision.action == AgentAction.CALL_TOOL:
            requested_source = _TOOL_TO_SOURCE.get(decision.tool_name or "")
            if (
                requested_source in _EXTERNAL_QUERY_SOURCES
                and requested_source not in state.required_sources
                and any(
                    observation.action == "external_source_scope_update"
                    for observation in state.observations
                )
            ):
                return self._decision_for_missing_source(
                    state,
                    reason="Planner requested a source excluded by the current caller scope.",
                )
            normalized_arguments = _normalized_tool_arguments(
                decision.tool_name or "",
                decision.arguments,
                _caller_query_messages(state),
            )
            decision = decision.model_copy(update={"arguments": normalized_arguments})
            signature = _tool_signature(decision.tool_name or "", decision.arguments)
            if (
                state.intent == AgentIntent.PRACTICE_GUIDE.value
                and decision.tool_name == "course_retriever"
            ):
                if self._practice_course_collected(state):
                    return self._answer_or_missing(state, _PRACTICE_COURSE_ALREADY_COLLECTED)
                if self._practice_course_no_progress(state):
                    return ActionDecision(
                        action=AgentAction.RETRIEVE,
                        arguments={"query": state.user_goal[:500]},
                        reason=_PRACTICE_COURSE_NO_PROGRESS,
                    )
            source_slot = _TOOL_TO_SOURCE.get(decision.tool_name or "")
            if source_slot and source_slot in state.completed_sources:
                return self._answer_or_missing(state, "Requested source is already satisfied.")
            if (
                source_slot
                and source_slot in state.required_sources
                and _source_was_attempted(state, source_slot)
            ):
                return self._answer_or_missing(state, "Requested source has already been attempted.")
            if signature in state.tool_signatures:
                return self._answer_or_missing(state, "Duplicate tool call was blocked.")
            return decision
        if decision.action == AgentAction.RETRIEVE and "course" in state.completed_sources:
            if (
                state.intent != AgentIntent.PRACTICE_GUIDE.value
                or self._practice_course_collected(state)
            ):
                return self._answer_or_missing(state, "Course evidence is already available.")
            return ActionDecision(
                action=AgentAction.ABSTAIN,
                arguments={"message": "课程证据未通过当前课程的授权与索引校验，本次不生成实践指导。"},
                reason=_PRACTICE_COURSE_NO_PROGRESS,
                task_complete=True,
            )
        if decision.action == AgentAction.ANSWER:
            if not state.evidence:
                return self._decision_for_missing_source(state, reason="Answer requires evidence.")
            if missing:
                return self._decision_for_missing_source(state, reason="Required evidence is incomplete.")
        if decision.action == AgentAction.ABSTAIN:
            if self._practice_answer_convergence_ready(state):
                return ActionDecision(
                    action=AgentAction.ANSWER,
                    reason=_PRACTICE_READY_ABSTAIN_NORMALIZED,
                    task_complete=True,
                )
            if missing and self._can_collect_more(state):
                return self._decision_for_missing_source(
                    state,
                    reason="Required evidence has not been attempted.",
                )
        return decision

    def _practice_answer_convergence_ready(self, state: AgentState) -> bool:
        return self._practice_course_collected(state) and not self._missing_sources(state)

    def _practice_course_collected(self, state: AgentState) -> bool:
        if (
            state.intent != AgentIntent.PRACTICE_GUIDE.value
            or state.course_id is None
            or "course" not in state.completed_sources
        ):
            return False
        try:
            for item in state.evidence:
                if (
                    not item.trusted
                    or item.resource_type != "course"
                    or item.origin_kind not in _PRACTICE_COURSE_ORIGINS
                    or item.document_id is None
                    or item.chunk_id is None
                ):
                    continue
                document = self.storage.document(int(item.document_id))
                if (
                    document is None
                    or int(document.get("course_id") or 0) != int(state.course_id)
                    or document.get("index_status") != "INDEXED"
                ):
                    continue
                chunks = self.storage.chunks_for_document(int(item.document_id))
                if any(
                    int(chunk.get("id") or 0) == int(item.chunk_id)
                    and int(chunk.get("document_id") or 0) == int(item.document_id)
                    and chunk.get("index_status") == "INDEXED"
                    for chunk in chunks
                ):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def _practice_course_no_progress(state: AgentState) -> bool:
        if state.intent != AgentIntent.PRACTICE_GUIDE.value:
            return False
        for observation in reversed(state.observations):
            metadata = observation.metadata
            orchestration = metadata.get("orchestration")
            if (
                metadata.get("run_id") != state.run_id
                or metadata.get("action") != "call_tool"
                or metadata.get("tool_name") != "course_retriever"
                or metadata.get("success") is not True
                or not isinstance(orchestration, dict)
                or orchestration.get("source_slot") != "course"
                or type(orchestration.get("new_evidence_count")) is not int
                or orchestration.get("new_evidence_count") != 0
                or orchestration.get("no_progress") is not True
            ):
                continue
            return True
        return False

    def _answer_or_missing(self, state: AgentState, reason: str) -> ActionDecision:
        if self._missing_sources(state):
            return self._decision_for_missing_source(state, reason=reason)
        if state.evidence:
            return ActionDecision(action=AgentAction.ANSWER, reason=reason, task_complete=True)
        return ActionDecision(
            action=AgentAction.ABSTAIN,
            arguments={"message": "\u6ca1\u6709\u53ef\u9a8c\u8bc1\u8bc1\u636e\uff0c\u65e0\u6cd5\u56de\u7b54\u3002"},
            reason=reason,
            task_complete=True,
        )

    @staticmethod
    def _bind_scoped_tool_arguments(state: AgentState, decision: ActionDecision) -> ActionDecision:
        if decision.action != AgentAction.CALL_TOOL or decision.tool_name != "course_retriever":
            return decision
        if state.course_id is None:
            return ActionDecision(
                action=AgentAction.ABSTAIN,
                arguments={"message": "课程上下文未确定，不能执行课程知识库检索。"},
                reason="COURSE_REQUIRED: course_retriever requires an authenticated course scope.",
                task_complete=True,
            )
        arguments = dict(decision.arguments)
        arguments["course_id"] = int(state.course_id)
        return decision.model_copy(update={"arguments": arguments})

    def _decision_for_missing_source(self, state: AgentState, *, reason: str) -> ActionDecision:
        for source in self._missing_sources(state):
            if source == "course":
                if "course" not in state.completed_sources:
                    return ActionDecision(
                        action=AgentAction.RETRIEVE,
                        arguments={
                            "query": (
                                state.course_document_snippet_query
                                if state.course_document_constraint_title
                                and state.course_document_snippet_query
                                else state.user_goal
                            )[:500]
                        },
                        reason=reason,
                    )
                continue
            if _source_was_attempted(state, source):
                continue
            tool_name = self._available_tool_for_source(state, source)
            if not tool_name:
                continue
            arguments = _normalized_tool_arguments(
                tool_name,
                {"query": state.user_goal[:500], "top_k": 3},
                _caller_query_messages(state),
            )
            signature = _tool_signature(tool_name, arguments)
            if signature in state.tool_signatures:
                continue
            if state.tool_call_count < state.max_tool_calls:
                return ActionDecision(
                    action=AgentAction.CALL_TOOL,
                    tool_name=tool_name,
                    arguments=arguments,
                    reason=reason,
                )
        if state.evidence and not self._missing_sources(state):
            return ActionDecision(action=AgentAction.ANSWER, reason=reason, task_complete=True)
        return ActionDecision(
            action=AgentAction.ABSTAIN,
            arguments={"message": "\u7f3a\u5c11\u5fc5\u9700\u7684\u53ef\u9a8c\u8bc1\u8bc1\u636e\uff0c\u672c\u6b21\u4e0d\u751f\u6210\u7ed3\u8bba\u3002"},
            reason=f"{reason} No unused trusted source remains.",
            task_complete=True,
        )

    def _available_tool_for_source(self, state: AgentState, source: str) -> str | None:
        names = {tool.name for tool in state.available_tools if tool.enabled}
        if source == "standard":
            caller_query = _explicit_external_query_origin(
                _caller_query_messages(state),
                fallback="public standard metadata",
            )
            if not re.search(r"\bRFC[\s-]*\d+\b", caller_query, flags=re.IGNORECASE):
                names.discard("standard_lookup_real")
        return next((name for name in _SOURCE_TO_TOOLS.get(source, ()) if name in names), None)

    def _missing_sources(self, state: AgentState) -> list[str]:
        return [item for item in state.required_sources if item not in state.completed_sources]

    def _can_collect_more(self, state: AgentState) -> bool:
        if state.current_step >= state.max_steps - 1:
            return False
        if state.tool_call_count >= state.max_tool_calls and any(
            source != "course" for source in self._missing_sources(state)
        ):
            return False
        return bool(self._missing_sources(state))

    @staticmethod
    def _tool_decision(
        tool_name: str,
        arguments: dict[str, Any],
        available_tool_names: set[str],
        reason: str,
        fallback_decision: ActionDecision,
    ) -> ActionDecision:
        if tool_name not in available_tool_names:
            return fallback_decision
        return ActionDecision(
            action=AgentAction.CALL_TOOL,
            tool_name=tool_name,
            arguments=arguments,
            reason=f"{_AUTO_TOOL_ROUTE_REASON}: {reason}",
        )

    def _validate_decision(
        self,
        *,
        state: AgentState,
        decision: ActionDecision,
        step_id: str,
        step_number: int,
        started_at: float,
    ) -> Observation | None:
        try:
            self.action_guard.validate(
                state=state,
                decision=decision,
                started_at=started_at,
                timeout_seconds=self.settings.agent_timeout_seconds,
                role="demo" if state.demo_compatibility else "student",
            )
            return None
        except ActionGuardError as exc:
            state.status = (
                AgentRunStatus.LIMIT_REACHED
                if exc.error_code in {"MAX_STEPS", "MAX_TOOL_CALLS", "AGENT_TIMEOUT"}
                else AgentRunStatus.ABSTAINED
            )
            state.stop_reason = exc.error_code
            state.final_answer = f"Action blocked by policy: {exc}"
            observation = Observation(
                step_number=step_number,
                action=decision.action.value,
                tool_name=decision.tool_name,
                success=False,
                content="{}",
                error_code=exc.error_code,
                summary=str(exc),
                latency_ms=0,
                fallback_used=False,
                metadata={"run_id": state.run_id, "step_id": step_id, "content_trust": "trusted_system"},
            )
            state.add_observation(observation)
            return observation

    def _execute_decision(
        self,
        *,
        state: AgentState,
        decision: ActionDecision,
        step_id: str,
        step_number: int,
        question_id: int | None,
        started_at: float,
    ) -> Observation | None:
        if decision.action == AgentAction.RETRIEVE:
            return self._retrieve(state, str(decision.arguments.get("query") or state.user_goal), step_id, step_number, question_id)
        if decision.action == AgentAction.CALL_TOOL:
            return self._call_tool(
                state,
                decision,
                step_id,
                step_number,
                question_id,
                started_at=started_at,
            )
        if decision.action == AgentAction.ASK_USER:
            return self._ask_user(state, decision, step_id, step_number)
        if decision.action == AgentAction.ANSWER:
            self._answer(state, step_id, question_id)
            return None
        if decision.action == AgentAction.ABSTAIN:
            state.status = AgentRunStatus.ABSTAINED
            default_stop_reason = {
                AgentIntent.PAPER_SEARCH.value: "PAPER_EVIDENCE_REQUIRED",
                AgentIntent.VULN_LOOKUP.value: "VULNERABILITY_EVIDENCE_REQUIRED",
                AgentIntent.STANDARD_LOOKUP.value: "STANDARD_EVIDENCE_REQUIRED",
                AgentIntent.PROJECT_SEARCH.value: "PROJECT_EVIDENCE_REQUIRED",
            }.get(state.intent, "ABSTAINED")
            state.stop_reason = (
                "PRACTICE_EVIDENCE_INCOMPLETE"
                if state.intent == AgentIntent.PRACTICE_GUIDE.value
                else default_stop_reason
            )
            state.final_answer = str(
                decision.arguments.get("message")
                or "\u5f53\u524d\u8bc1\u636e\u4e0d\u8db3\uff0c\u65e0\u6cd5\u7ed9\u51fa\u53ef\u9760\u56de\u7b54\u3002"
            )
            return None
        state.status = AgentRunStatus.FAILED
        state.stop_reason = "INVALID_ACTION"
        state.final_answer = "Agent returned an invalid action."
        return None

    def _ask_user(self, state: AgentState, decision: ActionDecision, step_id: str, step_number: int) -> Observation:
        state.status = AgentRunStatus.REQUIRES_INPUT
        state.pending_question = str(
            decision.arguments.get("question") or "\u8bf7\u8865\u5145\u5b9e\u9a8c\u76ee\u6807\u3001\u6570\u636e\u6765\u6e90\u6216\u7ea6\u675f\u6761\u4ef6\u3002"
        )
        state.stop_reason = "REQUIRES_INPUT"
        observation = Observation(
            step_number=step_number,
            action="ask_user",
            success=True,
            content=state.pending_question,
            summary="Agent paused and requires user input",
            metadata={"run_id": state.run_id, "step_id": step_id, "content_trust": "trusted_system"},
        )
        state.add_observation(observation)
        return observation

    def _retrieve(
        self,
        state: AgentState,
        query: str,
        step_id: str,
        step_number: int,
        question_id: int | None,
    ) -> Observation:
        started = time.perf_counter()
        try:
            retrieve_kwargs: dict[str, Any] = {"course_id": state.course_id}
            if state.course_document_constraint_title:
                retrieve_kwargs["allowed_document_ids"] = frozenset(
                    state.allowed_course_document_ids
                )
                if isinstance(self.retriever, HybridRetriever):
                    snippet_query = state.course_document_snippet_query or state.user_goal
                    state.course_document_snippet_query = snippet_query
                    retrieve_kwargs["snippet_query"] = snippet_query
            response = self.retriever.retrieve(query, **retrieve_kwargs)
        except Exception as exc:
            observation = Observation(
                step_number=step_number,
                action="retrieve",
                tool_name="hybrid_retriever",
                success=False,
                content="[]",
                error_code="RETRIEVER_FAILED",
                summary="Hybrid retriever failed; no evidence was added",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                fallback_used=False,
                metadata={"run_id": state.run_id, "step_id": step_id, "error_type": exc.__class__.__name__, "content_trust": "untrusted"},
            )
            observation = self._sanitize_observation(observation)
            state.add_observation(observation)
            state.status = AgentRunStatus.ABSTAINED
            state.stop_reason = "RETRIEVER_FAILED"
            state.final_answer = "\u68c0\u7d22\u670d\u52a1\u4e0d\u53ef\u7528\uff0c\u5f53\u524d\u6ca1\u6709\u8db3\u591f\u8bc1\u636e\u652f\u6301\u56de\u7b54\u3002"
            return observation

        if state.course_document_constraint_title:
            snippet_query = state.course_document_snippet_query or state.user_goal
            required_facts: set[str] = set()
            for prior in reversed(state.observations):
                if prior.action != "constraint_query_refresh":
                    continue
                raw_required = prior.metadata.get("required_fact_fingerprint") or []
                required_facts = {
                    str(value) for value in raw_required if str(value).strip()
                }
                break
            supported_facts = set(
                supported_query_fact_fingerprint(
                    snippet_query,
                    tuple(item.snippet for item in response.results),
                )
            )
            if required_facts and not required_facts.issubset(supported_facts):
                response = RetrievalResponse(
                    results=[],
                    abstained=True,
                    threshold=response.threshold,
                    fallback_used=response.fallback_used,
                )

        if (
            not response.results
            and state.demo_compatibility
            and not state.course_document_constraint_title
        ):
            local_results = LocalRetriever(self.settings.knowledge_base_dir).search(query, top_k=3)
            response = RetrievalResponse(
                results=[
                    _retrieval_from_demo_local(item, index)
                    for index, item in enumerate(local_results, 1)
                ],
                abstained=not local_results,
                threshold=response.threshold,
                fallback_used=bool(local_results),
            )
        try:
            scope_error = (
                None
                if state.demo_compatibility
                else _retrieval_scope_error(response.results, state.course_id, self.storage)
            )
        except Exception:
            scope_error = "SCOPE_VALIDATION_FAILED"
        if scope_error is not None:
            observation = Observation(
                step_number=step_number,
                action="retrieve",
                tool_name="hybrid_retriever",
                success=False,
                content="[]",
                error_code="RETRIEVER_SCOPE_MISMATCH",
                summary="Retriever returned evidence outside the authenticated course scope",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                fallback_used=False,
                metadata={
                    "run_id": state.run_id,
                    "step_id": step_id,
                    "error_type": scope_error,
                    "content_trust": "untrusted",
                },
            )
            observation = self._sanitize_observation(observation)
            state.add_observation(observation)
            state.status = AgentRunStatus.ABSTAINED
            state.stop_reason = "RETRIEVER_SCOPE_MISMATCH"
            state.final_answer = "检索结果无法通过当前课程范围校验，已停止回答。"
            return observation

        new_evidence = [
            _evidence_from_result(item, len(state.evidence) + index + 1, self.safety_guard)
            for index, item in enumerate(response.results)
        ]
        state.evidence.extend(new_evidence)
        observation = retrieval_results_to_observation(
            response.results,
            step_number=step_number,
            fallback_used=response.fallback_used,
            run_id=state.run_id,
            step_id=step_id,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        observation = self._sanitize_observation(observation)
        state.add_observation(observation)
        state.fallback_used = state.fallback_used or response.fallback_used
        if response.abstained or not new_evidence:
            state.status = AgentRunStatus.ABSTAINED
            state.stop_reason = {
                AgentIntent.PRACTICE_GUIDE.value: "PRACTICE_EVIDENCE_INCOMPLETE",
                AgentIntent.PAPER_SEARCH.value: "PAPER_EVIDENCE_REQUIRED",
                AgentIntent.VULN_LOOKUP.value: "VULNERABILITY_EVIDENCE_REQUIRED",
                AgentIntent.STANDARD_LOOKUP.value: "STANDARD_EVIDENCE_REQUIRED",
                AgentIntent.PROJECT_SEARCH.value: "PROJECT_EVIDENCE_REQUIRED",
            }.get(state.intent, "INSUFFICIENT_EVIDENCE")
            state.final_answer = (
                "\u7f3a\u5c11\u5fc5\u9700\u7684\u53ef\u9a8c\u8bc1\u8bc1\u636e\uff0c\u672c\u6b21\u4e0d\u751f\u6210\u7ed3\u8bba\u3002"
                if state.stop_reason.endswith("_EVIDENCE_REQUIRED")
                else "\u5f53\u524d\u8bc1\u636e\u4e0d\u8db3\uff0c\u65e0\u6cd5\u7ed9\u51fa\u53ef\u9760\u56de\u7b54\u3002"
            )
        else:
            if "course" not in state.completed_sources:
                state.completed_sources.append("course")
            if state.intent == AgentIntent.COURSE_QA.value:
                self._answer(state, step_id, question_id)
        return observation

    def _call_tool(
        self,
        state: AgentState,
        decision: ActionDecision,
        step_id: str,
        step_number: int,
        question_id: int | None,
        *,
        started_at: float,
    ) -> Observation:
        signature = _tool_signature(decision.tool_name or "", decision.arguments)
        if signature in state.tool_signatures:
            observation = Observation(
                step_number=step_number,
                action="call_tool",
                tool_name=decision.tool_name,
                success=False,
                content="{}",
                error_code="DUPLICATE_TOOL_CALL",
                summary="Duplicate tool call was blocked before execution",
                metadata={"run_id": state.run_id, "step_id": step_id, "content_trust": "trusted_system"},
            )
            state.add_observation(observation)
            return observation
        remaining_timeout_seconds = max(
            0.0,
            self.settings.agent_timeout_seconds - (time.perf_counter() - started_at),
        )
        definition = get_tool_definition(decision.tool_name or "")
        if definition is not None and definition.network and self.remote_rate_limiter is not None:
            transport_grace = min(2.0, max(0.25, float(definition.timeout_seconds) * 0.2))
            if remaining_timeout_seconds > float(definition.timeout_seconds) + transport_grace:
                self.remote_rate_limiter(decision.tool_name or "", dict(decision.arguments))
        state.tool_signatures.append(signature)
        state.tool_call_count += 1
        try:
            call_kwargs: dict[str, Any] = {}
            if isinstance(self.mcp_gateway, MCPClientGateway):
                call_kwargs["role"] = "demo" if state.demo_compatibility else "student"
                call_kwargs["remaining_timeout_seconds"] = remaining_timeout_seconds
            call = self.mcp_gateway.call_tool(
                decision.tool_name or "",
                decision.arguments,
                run_id=state.run_id,
                step_id=step_id,
                step_number=step_number,
                **call_kwargs,
            )
        except Exception as exc:
            observation = Observation(
                step_number=step_number,
                action="call_tool",
                tool_name=decision.tool_name,
                success=False,
                content="{}",
                error_code="MCP_CALL_FAILED",
                summary="MCP gateway call failed",
                fallback_used=False,
                metadata={"run_id": state.run_id, "step_id": step_id, "error_type": exc.__class__.__name__, "content_trust": "untrusted"},
            )
            state.add_observation(observation)
            return observation

        result_payload = call.result.model_dump(mode="json")
        new_evidence = self._trusted_tool_evidence(decision.tool_name or "", result_payload, state)
        state.evidence.extend(new_evidence)
        source_slot = _planner_source_slot(decision.tool_name or "")
        if new_evidence and source_slot and source_slot not in state.completed_sources:
            state.completed_sources.append(source_slot)
        metadata = dict(call.observation.metadata)
        metadata.update(
            {
                "run_id": state.run_id,
                "step_id": step_id,
                "action": "call_tool",
                "tool_name": decision.tool_name,
                "success": bool(call.result.success),
                "evidence_ids": [item.source_id for item in new_evidence],
                "resource_ids": [item.external_id for item in new_evidence],
                "content_hashes": [item.content_hash for item in new_evidence],
                "provenance_verified": bool(new_evidence),
                "orchestration": {
                    "source_slot": source_slot,
                    "new_evidence_count": len(new_evidence),
                    "no_progress": bool(call.result.success and not new_evidence),
                },
            }
        )
        observation = self._sanitize_observation(call.observation.model_copy(update={"metadata": metadata}))
        state.add_observation(observation)
        state.fallback_used = state.fallback_used or call.fallback_used
        self.storage.log_tool_call(
            run_id=state.run_id,
            step_id=step_id,
            question_id=question_id,
            tool_name=decision.tool_name or "",
            arguments=sanitize_payload(decision.arguments),
            result=sanitize_payload(result_payload),
            success=call.result.success,
            latency_ms=observation.latency_ms,
            fallback_used=call.fallback_used,
        )

        if decision.tool_name == "mermaid_generator" and call.result.success:
            state.final_answer = _format_tool_answer(decision.tool_name, result_payload, call.fallback_used)
            state.status = AgentRunStatus.COMPLETED
            state.stop_reason = "TOOL_ANSWERED"
            state.response_provider = "mcp" if not call.fallback_used else "local_fallback"
            state.response_model = decision.tool_name
        elif not new_evidence and not call.result.success and not self._can_collect_more(state):
            state.status = AgentRunStatus.ABSTAINED
            state.stop_reason = call.result.error_code or "TOOL_FAILED"
            state.final_answer = "\u5de5\u5177\u8c03\u7528\u5931\u8d25\uff0c\u4e14\u6ca1\u6709\u53ef\u9a8c\u8bc1\u7f13\u5b58\u8bc1\u636e\uff0c\u672c\u6b21\u62d2\u7b54\u3002"
        return observation

    def _trusted_tool_evidence(
        self,
        tool_name: str,
        result: dict[str, Any],
        state: AgentState,
    ) -> list[EvidenceItem]:
        if not result.get("success"):
            return []
        data = result.get("data") or {}
        if tool_name == "course_retriever":
            return self._trusted_course_tool_evidence(data, state)
        candidates: list[dict[str, Any]] = []
        for key in ("resources", "verified_papers"):
            value = data.get(key)
            if isinstance(value, list):
                candidates.extend(item for item in value if isinstance(item, dict))
        evidence: list[EvidenceItem] = []
        seen = {item.content_hash or item.external_id for item in state.evidence}
        expected_type = _TOOL_TO_SOURCE.get(tool_name, "")
        for item in candidates:
            if not _is_trusted_resource(item, expected_type):
                continue
            content_hash = str(item.get("content_hash") or "").strip()
            external_id = str(item.get("external_id") or item.get("id") or "").strip()
            dedupe_key = content_hash or external_id
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            title = str(item.get("title") or external_id).strip()
            text = _resource_evidence_text(item, title)
            evidence.append(
                EvidenceItem(
                    source_id=f"S{len(state.evidence) + len(evidence) + 1}",
                    text=self.safety_guard.sanitize_untrusted_text(text),
                    source=str(item.get("source_name") or item.get("source") or item.get("url") or "external"),
                    title=title,
                    source_type="external",
                    score=1.0,
                    source_url=str(item.get("url") or item.get("source_url") or ""),
                    external_id=external_id,
                    resource_type=str(item.get("resource_type") or expected_type),
                    origin_kind=str(item.get("origin_kind") or ""),
                    delivery_mode=str(item.get("delivery_mode") or ""),
                    fetched_at=str(item.get("fetched_at") or item.get("updated_at") or ""),
                    content_hash=content_hash,
                    tool_name=tool_name,
                    trusted=True,
                )
            )
        return evidence

    def _trusted_course_tool_evidence(
        self,
        data: dict[str, Any],
        state: AgentState,
    ) -> list[EvidenceItem]:
        if state.course_id is None or int(data.get("course_id") or 0) != int(state.course_id):
            return []
        candidates = data.get("results")
        if not isinstance(candidates, list):
            return []
        allowed_chunks = {
            int(chunk["id"]): chunk
            for chunk in self.storage.document_chunks(course_id=int(state.course_id))
            if chunk.get("index_status") == "INDEXED"
        }
        seen = {
            (item.document_id, item.chunk_id)
            for item in state.evidence
            if item.document_id is not None and item.chunk_id is not None
        }
        evidence: list[EvidenceItem] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            try:
                chunk_id = int(item.get("chunk_id"))
                document_id = int(item.get("document_id"))
            except (TypeError, ValueError):
                continue
            chunk = allowed_chunks.get(chunk_id)
            if chunk is None or int(chunk["document_id"]) != document_id:
                continue
            if (document_id, chunk_id) in seen:
                continue
            seen.add((document_id, chunk_id))
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            evidence.append(
                EvidenceItem(
                    source_id=f"S{len(state.evidence) + len(evidence) + 1}",
                    text=self.safety_guard.sanitize_untrusted_text(
                        str(item.get("snippet") or chunk.get("chunk_text") or "")
                    ),
                    source=str(item.get("source") or chunk.get("filename") or "course_document"),
                    title=str(item.get("title") or chunk.get("title") or chunk.get("filename") or "Course document"),
                    document_id=document_id,
                    chunk_id=chunk_id,
                    page=item.get("page") if item.get("page") is not None else chunk.get("page"),
                    section=str(item.get("section") or chunk.get("section") or ""),
                    source_type=str(item.get("source_type") or chunk.get("source_type") or ""),
                    score=float(item.get("final_score") or item.get("score") or 0.0),
                    resource_type="course",
                    origin_kind="course_document",
                    delivery_mode="local_rag",
                    content_hash=str(metadata.get("content_hash") or chunk.get("content_hash") or ""),
                    tool_name="course_retriever",
                    trusted=True,
                )
            )
        return evidence

    def _answer(self, state: AgentState, step_id: str, question_id: int | None) -> None:
        if not self._validate_course_document_constraint_before_answer(state):
            return
        missing = self._missing_sources(state)
        if not state.evidence or missing:
            state.status = AgentRunStatus.ABSTAINED
            state.stop_reason = "INSUFFICIENT_EVIDENCE"
            state.final_answer = "\u5fc5\u9700\u8bc1\u636e\u4e0d\u5b8c\u6574\uff0c\u672c\u6b21\u4e0d\u751f\u6210\u7ed3\u8bba\u3002"
            return
        if state.intent == AgentIntent.PRACTICE_GUIDE.value:
            self._answer_practice(
                state=state,
                step_id=step_id,
                question_id=question_id,
                answer_evidence=_answer_evidence(state),
            )
            return
        paper_selection = _paper_evidence_selection(state)
        if paper_selection is not None:
            selection_summary = {
                "requested_count": paper_selection.requested_count,
                "related_count": paper_selection.related_count,
                "field_complete_count": paper_selection.field_complete_count,
                "selected_count": len(paper_selection.selected),
                "shortfall_reason": paper_selection.shortfall_reason,
                "terminal_error": paper_selection.terminal_error,
            }
            self._trace(
                state,
                step_id,
                question_id,
                "paper_selection",
                "Paper evidence was filtered and selected by the server contract.",
                selection_summary,
            )
            if paper_selection.terminal_error:
                state.status = AgentRunStatus.ABSTAINED
                state.stop_reason = paper_selection.terminal_error
                state.final_answer = (
                    "没有检索到与主题相关且可核验的论文，本次拒答。"
                    if paper_selection.terminal_error == "PAPER_NO_RELEVANT_RESULTS"
                    else "相关论文缺少请求所需的可信字段，本次拒答。"
                )
                return
        answer_evidence = _answer_evidence(state, paper_selection)
        grounded_answer_required = _requires_grounded_answer(state)
        outbound_course_spans_required = any(is_course_evidence(item) for item in answer_evidence)
        if grounded_answer_required:
            evidence_spans = _build_evidence_spans(answer_evidence)
        elif outbound_course_spans_required:
            evidence_spans = select_relevant_course_text_spans(
                answer_evidence,
                question=state.user_goal,
            )
        else:
            evidence_spans = []
        request_metadata: dict[str, Any] = {
            "intent": state.intent,
            "required_sources": state.required_sources,
        }
        if state.learning_context:
            request_metadata["learning_context"] = dict(state.learning_context)
        if evidence_spans:
            request_metadata["evidence_spans"] = [
                span.model_dump(mode="json") for span in evidence_spans
            ]
        if outbound_course_spans_required:
            request_metadata[COURSE_SPAN_PROFILE_KEY] = (
                COURSE_SPAN_PROFILE_GROUNDED
                if grounded_answer_required
                else COURSE_SPAN_PROFILE_PLAIN
            )
        if grounded_answer_required:
            if paper_selection is not None:
                request_metadata["paper_contract"] = _paper_selection_metadata(
                    paper_selection,
                    evidence_spans,
                )
        request = LLMRequest(
            question=state.user_goal,
            evidence=answer_evidence,
            instructions="" if grounded_answer_required else _answer_instructions(state),
            metadata=request_metadata,
        )
        provider = self.answer_provider
        started = time.perf_counter()
        grounded_generation_attempts = 0
        citation_verified_spans = {span.span_id: span for span in evidence_spans} if grounded_answer_required else None
        citation_verified_bindings: list[dict[str, str]] | None = None
        expected_messages = (
            build_structured_messages(
                request,
                GroundedSpanAnswer.__name__,
                GroundedSpanAnswer.model_json_schema(),
            )
            if grounded_answer_required
            else build_answer_messages(request)
        )
        metadata = _answer_call_metadata(request, messages_sha256(expected_messages))
        if grounded_answer_required:
            metadata["output_contract"] = "grounded_claims_v2"
            metadata["evidence_span_ids"] = [span.span_id for span in evidence_spans]
            metadata["grounding_retry_attempt"] = 0
        try:
            if grounded_answer_required:
                grounded_answer = self._guarded_model_call(
                    provider=provider,
                    state=state,
                    request=request,
                    phase="answer_structured",
                    call=lambda: provider.generate_structured(request, GroundedSpanAnswer),
                )
                metadata["candidate_sha256"] = _grounded_candidate_sha256(grounded_answer)
                grounded_claims = _validate_grounded_span_answer(
                    grounded_answer,
                    answer_evidence,
                    evidence_spans,
                    required_sources=state.required_sources,
                    paper_contract=request_metadata.get("paper_contract"),
                )
                render_units = _build_grounded_render_units(grounded_answer, grounded_claims)
                presentation_profile = _grounded_presentation_profile(state)
                answer_text = _render_grounded_answer(
                    render_units,
                    evidence=answer_evidence,
                    presentation_profile=presentation_profile,
                )
                metadata["claim_count"] = len(grounded_claims)
                metadata["resolved_evidence_span_ids"] = [
                    unit.evidence_span_id for unit in render_units
                ]
                metadata["resolved_span_hashes"] = [
                    _sha256_text(claim.evidence_quote) for claim in grounded_claims
                ]
                metadata["canonical_claim_hashes"] = [
                    _sha256_text(unit.claim_text) for unit in render_units
                ]
                metadata["canonical_quote_hashes"] = [
                    _sha256_text(unit.evidence_quote) for unit in render_units
                ]
                metadata["candidate_accepted"] = True
                citation_verified_bindings = _grounded_span_citation_bindings(render_units)
                grounded_generation_attempts = 1
                fallback_used = False
                purpose = "generate_answer"
            else:
                response = self._guarded_model_call(
                    provider=provider,
                    state=state,
                    request=request,
                    phase="answer",
                    call=lambda: provider.generate_answer(request),
                )
                answer_text = response.text
                fallback_used = response.fallback_used
                purpose = "generate_answer"
            metadata["raw_answer_sha256"] = _sha256_text(answer_text)
            provider_hash = getattr(provider, "last_request_sha256", None)
            metadata["provider_prompt_sha256"] = provider_hash
            metadata["prompt_hash_matches"] = provider_hash in {None, metadata["answer_prompt_sha256"]}
            state.response_provider = provider.provider_name
            state.response_model = provider.model_name
            self.storage.log_model_call(
                run_id=state.run_id,
                step_id=step_id,
                question_id=question_id,
                provider=provider.provider_name,
                model=provider.model_name,
                purpose=purpose,
                success=True,
                latency_ms=(time.perf_counter() - started) * 1000,
                fallback_used=fallback_used,
                metadata=metadata,
            )
        except GroundedAnswerValidationError as exc:
            metadata.update(classify_llm_failure(exc, phase="generate_answer"))
            metadata["grounding_error_code"] = exc.error_code
            metadata["grounding_error_type"] = exc.__class__.__name__
            metadata["candidate_accepted"] = False
            paper_failure = _safe_paper_failure_detail(exc.failure_detail)
            if paper_failure:
                metadata["paper_failure"] = paper_failure
            if grounded_answer_required and exc.error_code == "GROUNDING_EMBEDDED_CITATION":
                metadata["retry_reason"] = "embedded_citation"
            state.response_provider = provider.provider_name
            state.response_model = provider.model_name
            self.storage.log_model_call(
                run_id=state.run_id,
                step_id=step_id,
                question_id=question_id,
                provider=provider.provider_name,
                model=provider.model_name,
                purpose="generate_answer",
                success=False,
                error_code=exc.error_code,
                error_message=exc.__class__.__name__,
                latency_ms=(time.perf_counter() - started) * 1000,
                metadata=metadata,
            )
            if (
                grounded_answer_required
                and grounded_generation_attempts < 2
                and getattr(provider, "provider_name", "") != self.fallback_provider.provider_name
            ):
                retry = self._retry_grounded_answer_once(
                    state=state,
                    step_id=step_id,
                    question_id=question_id,
                    provider=provider,
                    request=request,
                    answer_evidence=answer_evidence,
                    evidence_spans=evidence_spans,
                    previous_answer=str(metadata.get("candidate_sha256") or exc.error_code),
                    previous_error_code=exc.error_code,
                    previous_failure_detail=exc.failure_detail,
                )
                grounded_generation_attempts = 2
                if retry is not None:
                    answer_text, citation_verified_bindings = retry
                    fallback_used = False
                elif state.status == AgentRunStatus.ABSTAINED:
                    return
                else:
                    state.final_answer = "\u591a\u6765\u6e90\u56de\u7b54\u7684\u8bc1\u636e\u7ed1\u5b9a\u672a\u901a\u8fc7\u6821\u9a8c\uff0c\u672c\u6b21\u62d2\u7b54\u3002"
                    state.status = AgentRunStatus.ABSTAINED
                    state.stop_reason = exc.error_code
                    return
            else:
                state.final_answer = "\u591a\u6765\u6e90\u56de\u7b54\u7684\u8bc1\u636e\u7ed1\u5b9a\u672a\u901a\u8fc7\u6821\u9a8c\uff0c\u672c\u6b21\u62d2\u7b54\u3002"
                state.status = AgentRunStatus.ABSTAINED
                state.stop_reason = exc.error_code
                return
        except ModelCallPreflightError:
            raise
        except Exception as exc:
            failure_metadata = classify_llm_failure(exc, phase="generate_answer")
            validation_errors = _safe_validation_error_details(exc)
            if validation_errors:
                failure_metadata["validation_errors"] = validation_errors
            if grounded_answer_required:
                metadata.update(failure_metadata)
                metadata["candidate_accepted"] = False
                state.response_provider = provider.provider_name
                state.response_model = provider.model_name
                self.storage.log_model_call(
                    run_id=state.run_id,
                    step_id=step_id,
                    question_id=question_id,
                    provider=provider.provider_name,
                    model=provider.model_name,
                    purpose="generate_answer",
                    success=False,
                    error_code="MODEL_ANSWER_FAILED",
                    error_message=exc.__class__.__name__,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    fallback_used=False,
                    metadata=metadata,
                )
                if (
                    grounded_generation_attempts < 2
                    and getattr(provider, "provider_name", "") != self.fallback_provider.provider_name
                    and str(failure_metadata.get("failure_category") or "")
                    in {"empty_response", "response_parse", "schema_validation"}
                ):
                    retry = self._retry_grounded_answer_once(
                        state=state,
                        step_id=step_id,
                        question_id=question_id,
                        provider=provider,
                        request=request,
                        answer_evidence=answer_evidence,
                        evidence_spans=evidence_spans,
                        previous_answer=str(metadata.get("candidate_sha256") or "MODEL_ANSWER_FAILED"),
                        previous_error_code="MODEL_ANSWER_FAILED",
                        previous_failure_detail=None,
                    )
                    grounded_generation_attempts = 2
                    if retry is not None:
                        answer_text, citation_verified_bindings = retry
                        fallback_used = False
                    else:
                        state.final_answer = "模型未能生成可核验的外部证据回答，本次拒答。"
                        state.status = AgentRunStatus.ABSTAINED
                        state.stop_reason = "MODEL_ANSWER_FAILED"
                        return
                else:
                    state.final_answer = "模型未能生成可核验的外部证据回答，本次拒答。"
                    state.status = AgentRunStatus.ABSTAINED
                    state.stop_reason = "MODEL_ANSWER_FAILED"
                    return
            else:
                fallback_started = time.perf_counter()
                answer_text = self.fallback_provider.generate_answer(request).text
                state.fallback_used = True
                state.response_provider = self.fallback_provider.provider_name
                state.response_model = self.fallback_provider.model_name
                self.storage.log_model_call(
                    run_id=state.run_id,
                    step_id=step_id,
                    question_id=question_id,
                    provider=provider.provider_name,
                    model=provider.model_name,
                    purpose="generate_answer",
                    success=False,
                    error_code="MODEL_ANSWER_FAILED",
                    error_message=exc.__class__.__name__,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    metadata={**metadata, **failure_metadata},
                )
                self.storage.log_model_call(
                    run_id=state.run_id,
                    step_id=step_id,
                    question_id=question_id,
                    provider=self.fallback_provider.provider_name,
                    model=self.fallback_provider.model_name,
                    purpose="generate_answer_fallback",
                    success=True,
                    latency_ms=(time.perf_counter() - fallback_started) * 1000,
                    fallback_used=True,
                    metadata={
                        **metadata,
                        "fallback": True,
                        "raw_answer_sha256": _sha256_text(answer_text),
                    },
                )
                fallback_used = True

        citation_check = self.citation_guard.enforce(
            answer_text,
            state.evidence,
            question=state.user_goal,
            required_source_types=set(state.required_sources),
            verified_spans=citation_verified_spans,
            verified_bindings=citation_verified_bindings,
            verified_presentation_profile=_grounded_presentation_profile(state),
        )
        course_missing_citation_retry = (
            not grounded_answer_required
            and citation_check.error_code == "MISSING_CITATION"
            and {
                _normalized_required_source(source)
                for source in state.required_sources
                if _normalized_required_source(source)
            }
            == {"course"}
            and outbound_course_spans_required
            and bool(evidence_spans)
            and not fallback_used
            and getattr(provider, "provider_name", "")
            not in {"template", self.fallback_provider.provider_name}
        )
        if (
            not citation_check.ok
            and (grounded_answer_required or course_missing_citation_retry)
            and not fallback_used
            and grounded_generation_attempts < 2
            and getattr(provider, "provider_name", "") != self.fallback_provider.provider_name
        ):
            self._trace(
                state,
                step_id,
                question_id,
                "grounded_candidate_rejected",
                "Grounded answer candidate failed citation validation",
                _grounded_rejection_trace_summary(
                    citation_check.error_code,
                    citation_check.failure_detail,
                ),
            )
            retry = self._retry_grounded_answer_once(
                state=state,
                step_id=step_id,
                question_id=question_id,
                provider=provider,
                request=request,
                answer_evidence=answer_evidence,
                evidence_spans=evidence_spans,
                previous_answer=answer_text,
                previous_error_code=citation_check.error_code or "UNSUPPORTED_CITATION",
                previous_failure_detail=citation_check.failure_detail,
            )
            grounded_generation_attempts = 2
            if retry is not None:
                answer_text, citation_verified_bindings = retry
                if course_missing_citation_retry:
                    citation_verified_spans = {
                        span.span_id: span for span in evidence_spans
                    }
                citation_check = self.citation_guard.enforce(
                    answer_text,
                    state.evidence,
                    question=state.user_goal,
                    required_source_types=set(state.required_sources),
                    verified_spans=citation_verified_spans,
                    verified_bindings=citation_verified_bindings,
                    verified_presentation_profile=_grounded_presentation_profile(state),
                )
            elif state.status == AgentRunStatus.ABSTAINED:
                return
        if not citation_check.ok:
            if grounded_answer_required:
                self._trace(
                    state,
                    step_id,
                    question_id,
                    "grounded_candidate_rejected",
                    "Grounded answer candidate failed final citation validation",
                    _grounded_rejection_trace_summary(
                        citation_check.error_code,
                        citation_check.failure_detail,
                    ),
                )
            state.final_answer = citation_check.answer
            state.status = AgentRunStatus.ABSTAINED
            state.stop_reason = citation_check.error_code
            return
        cited_ids = set(CITATION_RE.findall(citation_check.answer))
        cited_source_types = {
            item.resource_type for item in state.evidence if item.source_id in cited_ids
        }
        required_external_types = {
            source for source in state.required_sources if source != "course"
        }
        if required_external_types and not required_external_types.issubset(cited_source_types):
            state.final_answer = "\u6700\u7ec8\u56de\u7b54\u672a\u8986\u76d6\u6240\u6709\u5fc5\u9700\u8bc1\u636e\u6765\u6e90\uff0c\u5df2\u62d2\u7b54\u3002"
            state.status = AgentRunStatus.ABSTAINED
            state.stop_reason = "INCOMPLETE_CITATION_COVERAGE"
            return
        state.final_answer = citation_check.answer
        state.fallback_used = state.fallback_used or fallback_used
        state.status = AgentRunStatus.COMPLETED
        state.stop_reason = (
            paper_selection.completion_reason
            if paper_selection is not None
            else "ANSWERED"
        )

    def _answer_practice(
        self,
        *,
        state: AgentState,
        step_id: str,
        question_id: int | None,
        answer_evidence: list[EvidenceItem],
    ) -> None:
        """Generate a practice guide whose every body unit is bound to one server span."""

        try:
            hydrated_evidence = _hydrate_practice_evidence(
                answer_evidence,
                course_id=state.course_id,
                storage=self.storage,
                safety_guard=self.safety_guard,
                run_id=state.run_id,
                observations=state.observations,
                required_sources=state.required_sources,
            )
            required_step_count = _extract_requested_practice_step_count(state.user_goal)
            hydrated_evidence = _extend_practice_course_evidence(
                hydrated_evidence,
                course_id=state.course_id,
                storage=self.storage,
                safety_guard=self.safety_guard,
                required_step_count=required_step_count,
            )
            # Keep the original bounded snippets for already selected sources
            # (model-call diagnostics must not persist whole hydrated chunks),
            # while adding only the newly selected sibling support identity.
            # CitationGuard and the response DTO therefore see the same source
            # set as the model without rewriting old evidence bodies.
            state.evidence = _merge_practice_state_evidence(
                state.evidence,
                hydrated_evidence,
            )
            evidence_spans = _build_practice_evidence_spans(hydrated_evidence)
            _validate_practice_evidence_completeness(
                evidence_spans,
                required_step_count=required_step_count,
            )
        except GroundedAnswerValidationError as exc:
            state.final_answer = "当前课程资料不足以生成逐项可核验的实践指导，本次拒答。"
            state.status = AgentRunStatus.ABSTAINED
            state.stop_reason = exc.error_code
            return

        request_metadata: dict[str, Any] = {
            "intent": state.intent,
            "required_sources": state.required_sources,
            "evidence_spans": [span.model_dump(mode="json") for span in evidence_spans],
            "required_practice_step_count": required_step_count,
            "practice_contract": "grounded_practice_v1",
            COURSE_SPAN_PROFILE_KEY: COURSE_SPAN_PROFILE_PRACTICE,
        }
        if state.learning_context:
            request_metadata["learning_context"] = dict(state.learning_context)
        request = LLMRequest(
            question=state.user_goal,
            evidence=hydrated_evidence,
            instructions="",
            metadata=request_metadata,
        )
        verified_spans = {span.span_id: span for span in evidence_spans}
        provider = self.answer_provider
        provider_fallback_used = isinstance(provider, TemplateProvider)
        state.response_provider = provider.provider_name
        state.response_model = provider.model_name
        retry_feedback: dict[str, Any] | None = None

        for attempt in range(2):
            attempt_started = time.perf_counter()
            attempt_request = request
            if attempt:
                safe_retry_feedback = retry_feedback or _practice_retry_feedback(
                    error_code=state.stop_reason or "PRACTICE_GROUNDING_FAILED",
                    failure_detail=None,
                    evidence_spans=evidence_spans,
                )
                attempt_request = request.model_copy(
                    update={
                        "metadata": {
                            **request.metadata,
                            "grounding_retry_attempt": 1,
                            "previous_error_code": state.stop_reason or "PRACTICE_GROUNDING_FAILED",
                            "practice_retry_feedback": safe_retry_feedback,
                        }
                    }
                )
            expected_messages = build_structured_messages(
                attempt_request,
                GroundedPracticeGuide.__name__,
                GroundedPracticeGuide.model_json_schema(),
            )
            metadata = _answer_call_metadata(
                attempt_request,
                messages_sha256(expected_messages),
            )
            metadata.update(
                {
                    "output_contract": "grounded_practice_v1",
                    "grounding_retry_attempt": attempt,
                    "evidence_span_ids": [span.span_id for span in evidence_spans],
                    "required_practice_step_count": required_step_count,
                }
            )
            if attempt and retry_feedback:
                metadata["practice_retry_feedback"] = dict(retry_feedback)
            try:
                guide = self._guarded_model_call(
                    provider=provider,
                    state=state,
                    request=attempt_request,
                    phase="practice_structured",
                    call=lambda: provider.generate_structured(attempt_request, GroundedPracticeGuide),
                )
                metadata["candidate_sha256"] = _grounded_candidate_sha256(guide)
                validated = _validate_grounded_practice_guide(
                    guide,
                    hydrated_evidence,
                    evidence_spans,
                    required_step_count=required_step_count,
                    required_sources=state.required_sources,
                )
                render_units = _build_practice_render_units(guide, validated)
                practice_answer = _render_grounded_practice_guide(render_units)
                self.validate_learning_context(state)
                learning_context_block = _render_verified_learning_context(
                    state.learning_context,
                    self.safety_guard,
                )
                answer_text = (
                    f"{practice_answer}\n\n{learning_context_block}"
                    if learning_context_block
                    else practice_answer
                )
                bindings = [unit.as_binding() for unit in render_units]
                citation_check = self.citation_guard.enforce(
                    answer_text,
                    state.evidence,
                    question=state.user_goal,
                    required_source_types=set(state.required_sources),
                    verified_spans=verified_spans,
                    verified_practice_bindings=bindings,
                    verified_practice_context=learning_context_block or None,
                )
                metadata["claim_count"] = len(render_units)
                metadata["resolved_evidence_span_ids"] = [
                    unit.evidence_span_id for unit in render_units
                ]
                metadata["resolved_span_hashes"] = [
                    _sha256_text(unit.evidence_quote) for unit in render_units
                ]
                metadata["canonical_claim_hashes"] = [
                    _sha256_text(unit.claim_text) for unit in render_units
                ]
                metadata["raw_answer_sha256"] = _sha256_text(answer_text)
                metadata["candidate_accepted"] = citation_check.ok
                if not citation_check.ok:
                    metadata["candidate_rejection"] = _practice_candidate_rejection(
                        grounding_retry_attempt=attempt,
                        error_code=citation_check.error_code or "PRACTICE_GROUNDING_FAILED",
                        failure_detail=citation_check.failure_detail,
                    )
                provider_hash = getattr(provider, "last_request_sha256", None)
                metadata["provider_prompt_sha256"] = provider_hash
                metadata["prompt_hash_matches"] = provider_hash in {
                    None,
                    metadata["answer_prompt_sha256"],
                }
                self.storage.log_model_call(
                    run_id=state.run_id,
                    step_id=step_id,
                    question_id=question_id,
                    provider=provider.provider_name,
                    model=provider.model_name,
                    purpose="generate_practice_guide",
                    success=True,
                    latency_ms=(time.perf_counter() - attempt_started) * 1000,
                    fallback_used=provider_fallback_used,
                    metadata=metadata,
                )
                if citation_check.ok:
                    state.final_answer = citation_check.answer
                    state.fallback_used = provider_fallback_used
                    state.status = AgentRunStatus.COMPLETED
                    state.stop_reason = "ANSWERED"
                    return

                state.stop_reason = citation_check.error_code or "PRACTICE_GROUNDING_FAILED"
                retry_feedback = _practice_retry_feedback(
                    error_code=state.stop_reason,
                    failure_detail=citation_check.failure_detail,
                    evidence_spans=evidence_spans,
                )
                rejection = metadata["candidate_rejection"]
                self._trace(
                    state,
                    step_id,
                    question_id,
                    "citation_guard_failed",
                    "Practice citation guard rejected a candidate",
                    _practice_rejection_trace_summary(rejection),
                )
                if attempt == 0:
                    continue
                state.final_answer = citation_check.answer
                state.status = AgentRunStatus.ABSTAINED
                return
            except GroundedAnswerValidationError as exc:
                metadata.update(classify_llm_failure(exc, phase="generate_answer"))
                metadata["grounding_error_code"] = exc.error_code
                metadata["grounding_error_type"] = exc.__class__.__name__
                metadata["candidate_accepted"] = False
                metadata["candidate_rejection"] = _practice_candidate_rejection(
                    grounding_retry_attempt=attempt,
                    error_code=exc.error_code,
                    failure_detail=exc.failure_detail,
                )
                self.storage.log_model_call(
                    run_id=state.run_id,
                    step_id=step_id,
                    question_id=question_id,
                    provider=provider.provider_name,
                    model=provider.model_name,
                    purpose="generate_practice_guide",
                    success=False,
                    error_code=exc.error_code,
                    error_message=exc.__class__.__name__,
                    latency_ms=(time.perf_counter() - attempt_started) * 1000,
                    fallback_used=False,
                    metadata=metadata,
                )
                state.stop_reason = exc.error_code
                retry_feedback = _practice_retry_feedback(
                    error_code=exc.error_code,
                    failure_detail=exc.failure_detail,
                    evidence_spans=evidence_spans,
                )
                self._trace(
                    state,
                    step_id,
                    question_id,
                    "practice_candidate_rejected",
                    "Practice candidate failed structured grounding validation",
                    _practice_rejection_trace_summary(metadata["candidate_rejection"]),
                )
                if attempt == 0:
                    continue
                state.final_answer = "实践指导的逐项证据绑定未通过校验，本次拒答。"
                state.status = AgentRunStatus.ABSTAINED
                return
            except ModelCallPreflightError:
                raise
            except Exception as exc:
                failure_metadata = classify_llm_failure(exc, phase="generate_answer")
                metadata.update(failure_metadata)
                metadata["candidate_accepted"] = False
                self.storage.log_model_call(
                    run_id=state.run_id,
                    step_id=step_id,
                    question_id=question_id,
                    provider=provider.provider_name,
                    model=provider.model_name,
                    purpose="generate_practice_guide",
                    success=False,
                    error_code="MODEL_ANSWER_FAILED",
                    error_message=exc.__class__.__name__,
                    latency_ms=(time.perf_counter() - attempt_started) * 1000,
                    fallback_used=False,
                    metadata=metadata,
                )
                state.stop_reason = "MODEL_ANSWER_FAILED"
                retry_feedback = _practice_retry_feedback(
                    error_code=state.stop_reason,
                    failure_detail=None,
                    evidence_spans=evidence_spans,
                )
                structured_failure = str(failure_metadata.get("failure_category") or "") in {
                    "empty_response",
                    "response_parse",
                    "schema_validation",
                }
                if attempt == 0 and structured_failure:
                    continue
                state.final_answer = "模型未能生成可核验的实践指导，本次拒答。"
                state.status = AgentRunStatus.ABSTAINED
                return

    def _retry_grounded_answer_once(
        self,
        *,
        state: AgentState,
        step_id: str,
        question_id: int | None,
        provider: LLMProvider,
        request: LLMRequest,
        answer_evidence: list[EvidenceItem],
        evidence_spans: list[EvidenceSpan],
        previous_answer: str,
        previous_error_code: str,
        previous_failure_detail: dict[str, Any] | None,
    ) -> tuple[str, list[dict[str, str]]] | None:
        retry_reason = {
            "GROUNDING_EMBEDDED_CITATION": "embedded_citation",
            "MISSING_CITATION": "missing_citation",
            "UNSUPPORTED_CITATION": "unsupported_citation",
            "IRRELEVANT_CITATION": "irrelevant_citation",
            "INCOMPLETE_CITATION_COVERAGE": "required_source_coverage",
        }.get(previous_error_code, "grounding_validation")
        retry_guidance = (
            "Retry once with schema-only grounded claims. claim_text must not contain citation markers, "
            "Markdown citations, evidence quotes, renderer delimiters, preambles, conclusions, tool names, "
            "or internal steps; final citations and quotes are added by the server renderer."
            if retry_reason == "embedded_citation"
            else (
                "Retry once because the otherwise supported answer omitted its citation. Select only "
                "a current server-provided source/span pair and return schema-only grounded claims; "
                "the server renderer adds the exact quote and final citation."
                if retry_reason == "missing_citation"
                else (
                    "Retry once because selected-span support failed. For text spans, copy the selected "
                    "exact_text or one contiguous extractive clause from it; keep the same language, do not "
                    "translate, summarize, replace subjects such as we/authors, or add facts from another "
                    "span. For title spans, use only a full identity template such as The paper is titled "
                    "<exact title>, The NVD record is titled <exact title>, The standard is titled "
                    "<exact title>, or The GitHub project is named <exact title>. For external_id spans, "
                    "use only The paper arXiv ID is <exact id>, The paper DOI is <exact DOI>, The NVD "
                    "record is identified as <exact CVE>, The standard identifier is <exact id>, or The "
                    "GitHub repository is <exact owner/repo>. "
                    "Bind title, external_id, and text facts to separate spans; omit unsupported usage, "
                    "causal, comparison, performance, numeric, or security-impact claims."
                )
            )
        )
        if retry_reason in {"irrelevant_citation", "grounding_validation"}:
            retry_guidance += (
                " Replace any invalid, irrelevant, unknown, or mismatched source/span pair with a legal "
                "same-source pair from the server catalog; do not invent identifiers."
            )
        if retry_reason == "required_source_coverage":
            retry_guidance += " Include at least one valid claim for every explicitly required source type."
        safe_failure = _safe_grounded_failure_detail(previous_failure_detail)
        paper_failure = _safe_paper_failure_detail(previous_failure_detail)
        if previous_error_code == "PAPER_FIELD_COVERAGE":
            retry_guidance += (
                " Re-audit every selected paper and every required paper field. Use the server-provided "
                "title, paper_author_summary, and paper_abstract spans exactly as required; do not add or "
                "remove selected papers and do not invent missing metadata."
            )
            if paper_failure:
                retry_guidance += (
                    " First incomplete selected paper: "
                    f"source_id={paper_failure['source_id']}, "
                    f"missing_fields={','.join(paper_failure['missing_fields'])}."
                )
        if safe_failure:
            retry_guidance += (
                " Re-audit every claim. First rejected binding: "
                f"ordinal={safe_failure['ordinal']}, source_id={safe_failure['source_id']}, "
                f"evidence_span_id={safe_failure['span_id']}."
            )
        retry_metadata = {
            **request.metadata,
            "grounding_retry_attempt": 1,
            "grounded_retry_guidance": retry_guidance,
            "retry_reason": retry_reason,
            "previous_error_code": previous_error_code,
            "previous_guard_error_code": previous_error_code,
            "previous_candidate_sha256": _sha256_text(previous_answer),
        }
        retry_request = LLMRequest(
            question=request.question,
            evidence=request.evidence,
            instructions="",
            metadata=retry_metadata,
        )
        expected_messages = build_structured_messages(
            retry_request,
            GroundedSpanAnswer.__name__,
            GroundedSpanAnswer.model_json_schema(),
        )
        metadata = _answer_call_metadata(retry_request, messages_sha256(expected_messages))
        metadata["output_contract"] = "grounded_claims_v2"
        metadata["grounding_retry_attempt"] = 1
        metadata["retry_reason"] = retry_reason
        metadata["previous_error_code"] = previous_error_code
        metadata["previous_guard_error_code"] = previous_error_code
        metadata["previous_candidate_sha256"] = _sha256_text(previous_answer)
        metadata["evidence_span_ids"] = [span.span_id for span in evidence_spans]
        if safe_failure:
            metadata["previous_guard_failure"] = safe_failure
        if paper_failure:
            metadata["previous_paper_failure"] = paper_failure
        started = time.perf_counter()
        try:
            grounded_answer = self._guarded_model_call(
                provider=provider,
                state=state,
                request=retry_request,
                phase="answer_structured_retry",
                call=lambda: provider.generate_structured(retry_request, GroundedSpanAnswer),
            )
            metadata["candidate_sha256"] = _grounded_candidate_sha256(grounded_answer)
            grounded_claims = _validate_grounded_span_answer(
                grounded_answer,
                answer_evidence,
                evidence_spans,
                required_sources=state.required_sources,
                paper_contract=request.metadata.get("paper_contract"),
            )
            render_units = _build_grounded_render_units(grounded_answer, grounded_claims)
            answer_text = _render_grounded_answer(
                render_units,
                evidence=answer_evidence,
                presentation_profile=_grounded_presentation_profile(state),
            )
            metadata["claim_count"] = len(grounded_claims)
            metadata["resolved_evidence_span_ids"] = [
                unit.evidence_span_id for unit in render_units
            ]
            metadata["resolved_span_hashes"] = [
                _sha256_text(claim.evidence_quote) for claim in grounded_claims
            ]
            metadata["canonical_claim_hashes"] = [
                _sha256_text(unit.claim_text) for unit in render_units
            ]
            metadata["canonical_quote_hashes"] = [
                _sha256_text(unit.evidence_quote) for unit in render_units
            ]
            metadata["raw_answer_sha256"] = _sha256_text(answer_text)
            provider_hash = getattr(provider, "last_request_sha256", None)
            metadata["provider_prompt_sha256"] = provider_hash
            metadata["prompt_hash_matches"] = provider_hash in {None, metadata["answer_prompt_sha256"]}
            metadata["candidate_validated"] = True
            metadata["candidate_accepted"] = True
            self.storage.log_model_call(
                run_id=state.run_id,
                step_id=step_id,
                question_id=question_id,
                provider=provider.provider_name,
                model=provider.model_name,
                purpose="generate_answer",
                success=True,
                latency_ms=(time.perf_counter() - started) * 1000,
                fallback_used=False,
                metadata=metadata,
            )
            return answer_text, _grounded_span_citation_bindings(render_units)
        except GroundedAnswerValidationError as exc:
            metadata.update(classify_llm_failure(exc, phase="generate_answer"))
            metadata["grounding_error_code"] = exc.error_code
            metadata["grounding_error_type"] = exc.__class__.__name__
            metadata["candidate_accepted"] = False
            final_paper_failure = _safe_paper_failure_detail(exc.failure_detail)
            if final_paper_failure:
                metadata["paper_failure"] = final_paper_failure
            self.storage.log_model_call(
                run_id=state.run_id,
                step_id=step_id,
                question_id=question_id,
                provider=provider.provider_name,
                model=provider.model_name,
                purpose="generate_answer",
                success=False,
                error_code=exc.error_code,
                error_message=exc.__class__.__name__,
                latency_ms=(time.perf_counter() - started) * 1000,
                fallback_used=False,
                metadata=metadata,
            )
            state.final_answer = "\u591a\u6765\u6e90\u56de\u7b54\u7684\u8bc1\u636e\u7ed1\u5b9a\u672a\u901a\u8fc7\u6821\u9a8c\uff0c\u672c\u6b21\u62d2\u7b54\u3002"
            state.status = AgentRunStatus.ABSTAINED
            state.stop_reason = exc.error_code
            return None
        except ModelCallPreflightError:
            raise
        except Exception as exc:
            metadata.update(classify_llm_failure(exc, phase="generate_answer"))
            validation_errors = _safe_validation_error_details(exc)
            if validation_errors:
                metadata["validation_errors"] = validation_errors
            metadata["candidate_accepted"] = False
            self.storage.log_model_call(
                run_id=state.run_id,
                step_id=step_id,
                question_id=question_id,
                provider=provider.provider_name,
                model=provider.model_name,
                purpose="generate_answer",
                success=False,
                error_code="MODEL_ANSWER_FAILED",
                error_message=exc.__class__.__name__,
                latency_ms=(time.perf_counter() - started) * 1000,
                fallback_used=False,
                metadata=metadata,
            )
            return None

    def _sanitize_observation(self, observation: Observation) -> Observation:
        return observation.model_copy(
            update={
                "content": self.safety_guard.sanitize_untrusted_text(observation.content),
                "metadata": self.safety_guard.sanitize_payload(observation.metadata),
            }
        )

    def _finish_with_limit(self, state: AgentState, reason: str, message: str) -> None:
        state.status = AgentRunStatus.LIMIT_REACHED
        state.stop_reason = reason
        state.final_answer = message + (
            " Existing evidence was preserved for review."
            if state.evidence
            else " No sufficient evidence was collected."
        )

    def _persist_state(self, state: AgentState, question_id: int | None = None) -> None:
        if state.status != AgentRunStatus.RUNNING:
            self._check_model_access(
                provider=self.answer_provider,
                state=state,
                request=None,
                phase="result",
                stage="before_result_write",
            )
        self.storage.update_agent_run(
            run_id=state.run_id,
            status=state.status.value,
            state=state.model_dump(mode="json"),
            stop_reason=state.stop_reason,
            final_answer=state.final_answer,
            pending_question=state.pending_question,
            question_id=question_id,
        )

    def _guarded_model_call(
        self,
        *,
        provider: LLMProvider,
        state: AgentState,
        request: LLMRequest | None,
        phase: str,
        call: Callable[[], _ModelResult],
    ) -> _ModelResult:
        self._check_model_access(
            provider=provider,
            state=state,
            request=request,
            phase=phase,
            stage="before",
        )
        install_outbound_guard = getattr(provider, "set_outbound_guard", None)
        if callable(install_outbound_guard):
            install_outbound_guard(
                lambda outbound_stage: self._check_model_access(
                    provider=provider,
                    state=state,
                    request=request,
                    phase=phase,
                    stage=f"outbound_{outbound_stage}",
                )
            )
        try:
            result = call()
            self._check_model_access(
                provider=provider,
                state=state,
                request=request,
                phase=phase,
                stage="after",
            )
            return result
        finally:
            if callable(install_outbound_guard):
                install_outbound_guard(None)

    def _check_model_access(
        self,
        *,
        provider: LLMProvider,
        state: AgentState,
        request: LLMRequest | None,
        phase: str,
        stage: str,
    ) -> str | None:
        self.validate_learning_context(state)
        if self.model_call_guard is None:
            return None
        return self.model_call_guard(
            provider=provider,
            state=state,
            request=request,
            phase=phase,
            stage=stage,
        )

    def _trace(
        self,
        state: AgentState,
        step_id: str | None,
        question_id: int | None,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.storage.log_agent_trace(
            trace_id=state.trace_id,
            run_id=state.run_id,
            step_id=step_id,
            question_id=question_id,
            event_type=event_type,
            message=message,
            metadata=metadata,
        )

    def _timed_out(self, started: float) -> bool:
        return (time.perf_counter() - started) >= self.settings.agent_timeout_seconds

    def _to_chat_response(self, state: AgentState) -> dict[str, Any]:
        show_citations = state.status == AgentRunStatus.COMPLETED
        cited_ids = set(CITATION_RE.findall(state.final_answer or "")) if show_citations else set()
        response_evidence = (
            [
                item
                for item in state.evidence
                if _is_constrained_course_evidence(state, item)
            ]
            if state.course_document_constraint_title
            else state.evidence
        )
        citations = [
            _citation_from_evidence(item)
            for item in response_evidence
            if item.source_id in cited_ids
        ][:10]
        tool_calls = self.storage.tool_call_rows(state.run_id)
        return {
            "answer": state.final_answer or state.pending_question or "Agent did not produce a final answer.",
            "citations": citations,
            "confidence": _rule_confidence(state, citations),
            "abstained": state.status in {AgentRunStatus.ABSTAINED, AgentRunStatus.LIMIT_REACHED},
            "module": response_evidence[0].section if response_evidence and response_evidence[0].section else (citations[0]["title"] if citations else "unknown"),
            "intent": state.intent,
            "tool": _selected_tool_from_calls(tool_calls),
            "run_id": state.run_id,
            "run_status": state.status.value,
            "current_step": state.current_step,
            "requires_user_input": state.status == AgentRunStatus.REQUIRES_INPUT,
            "pending_question": state.pending_question,
            "trace_id": state.trace_id,
            "provider": state.response_provider or self.answer_provider.provider_name,
            "model": state.response_model or self.answer_provider.model_name,
            "fallback_used": state.fallback_used,
            "stop_reason": state.stop_reason,
            "tool_calls": tool_calls,
            "course_id": state.course_id,
            "class_id": state.class_id,
            "delivery_mode": _response_delivery_mode(state),
        }


def _evidence_from_result(result: RetrievalResult, index: int, safety_guard: SafetyGuard) -> EvidenceItem:
    text = safety_guard.sanitize_untrusted_text(result.snippet)
    return EvidenceItem(
        source_id=f"S{index}",
        text=text,
        source=result.source,
        title=result.title,
        document_id=result.document_id,
        chunk_id=result.chunk_id,
        page=result.page,
        section=result.section,
        source_type=result.source_type,
        resource_type="course",
        origin_kind="local_document",
        delivery_mode="local_rag",
        content_hash=str(result.metadata.get("content_hash") or _sha256_text(text)),
        score=result.final_score,
        trusted=True,
    )


def _retrieval_scope_error(
    results: list[RetrievalResult],
    course_id: int | None,
    storage: LearningLogger,
) -> str | None:
    for result in results:
        if int(result.document_id) <= 0:
            return "UNVERIFIED_LOCAL_RESULT"
        if course_id is None:
            continue

        document = storage.document(int(result.document_id))
        if document is None or int(document.get("course_id") or 0) != int(course_id):
            return "DOCUMENT_COURSE_MISMATCH"
        if not is_student_visible_document(document):
            return "DOCUMENT_NOT_VISIBLE"

        chunks = storage.chunks_for_document(int(result.document_id))
        chunk = next((item for item in chunks if int(item["id"]) == int(result.chunk_id)), None)
        if chunk is None:
            return "CHUNK_DOCUMENT_MISMATCH"
        if str(chunk.get("index_status") or "").upper() != "INDEXED":
            return "CHUNK_NOT_INDEXED"
    return None


def _evidence_is_currently_available(
    item: EvidenceItem,
    course_id: int | None,
    storage: LearningLogger,
) -> bool:
    is_course = (
        item.resource_type in {"course", "course_document"}
        or item.origin_kind == "local_document"
        or item.delivery_mode == "local_rag"
        or item.document_id is not None
        or item.chunk_id is not None
    )
    if not is_course:
        return True
    if item.document_id is None or item.chunk_id is None or course_id is None:
        return False
    document = storage.document(int(item.document_id))
    if document is None or int(document.get("course_id") or 0) != int(course_id):
        return False
    if not is_student_visible_document(document):
        return False
    return any(
        int(chunk["id"]) == int(item.chunk_id)
        and str(chunk.get("index_status") or "").upper() == "INDEXED"
        for chunk in storage.chunks_for_document(int(item.document_id))
    )


def _citation_from_evidence(item: EvidenceItem) -> dict[str, Any]:
    return {
        "source_id": item.source_id,
        "source": item.source,
        "title": item.title,
        "snippet": item.text,
        "score": item.score,
        "document_id": item.document_id,
        "chunk_id": item.chunk_id,
        "page": item.page,
        "section": item.section,
        "source_type": item.source_type,
        "source_url": item.source_url or None,
        "external_id": item.external_id or None,
        "resource_type": item.resource_type,
        "origin_kind": item.origin_kind,
        "delivery_mode": item.delivery_mode,
        "fetched_at": item.fetched_at or None,
        "content_hash": item.content_hash,
    }


def _selected_tool_from_calls(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tool_calls:
        return None
    tool_name = str(tool_calls[-1].get("tool_name") or "")
    if not tool_name:
        return None
    definition = get_tool_definition(tool_name)
    if definition is None:
        return {"name": tool_name, "description": "", "risk": "LOW", "enabled": True}
    return {"name": definition.name, "description": definition.description, "risk": definition.risk, "enabled": definition.enabled}


def _retrieval_from_demo_local(item: dict[str, Any], index: int) -> RetrievalResult:
    """Legacy local-file evidence is restricted to explicit non-authenticated Demo runs."""

    return RetrievalResult(
        document_id=0,
        chunk_id=index,
        source=str(item.get("source", "")),
        title=str(item.get("title", "")),
        snippet=str(item.get("snippet", "")),
        page=None,
        section=str(item.get("module", "")),
        source_type="md",
        keyword_score=float(item.get("tfidf_score", item.get("score", 0.0)) or 0.0),
        vector_score=0.0,
        final_score=float(item.get("score", 0.0) or 0.0),
        metadata={"fallback": True, "demo_only": True},
    )


def _rule_confidence(state: AgentState, citations: list[dict[str, Any]]) -> float:
    if state.status in {AgentRunStatus.ABSTAINED, AgentRunStatus.LIMIT_REACHED, AgentRunStatus.FAILED}:
        return 0.0
    if not citations:
        return 0.2
    scores = [float(item.get("score") or 0.0) for item in citations]
    return round(max(0.2, min(0.95, sum(scores) / max(1, len(scores)))), 4)


def _tool_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    normalized = json.dumps(arguments, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"{tool_name}:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def _planner_source_slot(tool_name: str) -> str | None:
    if tool_name == "course_retriever":
        return "course"
    return _TOOL_TO_SOURCE.get(tool_name)


def _planner_normalization_audit(
    planner_decision: ActionDecision,
    final_decision: ActionDecision,
) -> dict[str, str | None] | None:
    if (
        planner_decision.action == final_decision.action
        and planner_decision.tool_name == final_decision.tool_name
    ):
        return None
    reason_code = final_decision.reason
    if reason_code not in _PLANNER_NORMALIZATION_REASON_CODES:
        return None
    return {
        "planner_action": planner_decision.action.value,
        "planner_tool_name": planner_decision.tool_name,
        "final_action": final_decision.action.value,
        "final_tool_name": final_decision.tool_name,
        "normalization_reason_code": reason_code,
    }


def _source_was_attempted(state: AgentState, source: str) -> bool:
    source_tools = set(_SOURCE_TO_TOOLS.get(source, ()))
    attempted_tools = {signature.split(":", 1)[0] for signature in state.tool_signatures}
    return bool(source_tools & attempted_tools)


def _paper_request_contract(question: str) -> _PaperRequestContract:
    text = " ".join(str(question or "").split())
    lowered = text.casefold()
    fields: list[str] = []
    aliases = {
        "title": ("标题", "题名", "title", "titles"),
        "authors": ("作者", "author", "authors"),
        "abstract_clue": ("摘要线索", "摘要", "abstract clue", "abstract", "summary clue"),
        "link": ("来源链接", "可核验链接", "链接", "source link", "source url", "url", "link"),
    }
    for field, values in aliases.items():
        if any(value.casefold() in lowered for value in values):
            fields.append(field)
    count_match = re.search(r"(?<!\d)([1-3])\s*(?:篇|papers?|results?)", lowered)
    if count_match:
        count = int(count_match.group(1))
        explicit = True
    else:
        chinese_match = re.search(r"([一二三])\s*篇", text)
        count = {"一": 1, "二": 2, "三": 3}.get(
            chinese_match.group(1) if chinese_match else "",
            3,
        )
        explicit = chinese_match is not None
    return _PaperRequestContract(
        required_fields=tuple(fields),
        target_count=max(1, min(3, count)),
        explicit_count=explicit,
    )


def _paper_search_query(question: str) -> str:
    text = " ".join(str(question or "").split()).strip()
    topic = ""
    markers = [(text.rfind(marker), marker) for marker in ("检索", "搜索", "查找")]
    marker_pos, marker = max(markers, key=lambda item: item[0])
    if marker_pos >= 0:
        topic = text[marker_pos + len(marker) :]
        topic = re.split(r"[，。；;\n]", topic, maxsplit=1)[0].strip()
    else:
        match = re.search(r"(?:search(?:\s+for)?|find)\s+(.+?)(?:[,.;]|$)", text, flags=re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
    if not topic:
        topic = text
    topic = re.sub(r"^(?:真实(?:的)?|相关(?:的)?|论文(?:搜索)?\s*)+", "", topic).strip()
    topic = re.sub(r"(?:相关)?论文$", "", topic).strip()
    topic = re.split(
        r"(?:，|。|；|;|\breturn\b|\bprovide\b|\binclude\b|返回|给出|提供|包含|不要)",
        topic,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,，。；;:")
    if not topic:
        return "paper research"

    canonicalized = topic
    for concept, aliases in {**_PAPER_DOMAIN_CONCEPTS, **_PAPER_SECURITY_CONCEPTS}.items():
        label = _PAPER_CONCEPT_QUERY_LABELS[concept]
        for alias in sorted(aliases, key=len, reverse=True):
            if re.search(r"[\u4e00-\u9fff]", alias):
                canonicalized = canonicalized.replace(alias, f" {label} ")
    canonicalized = " ".join(canonicalized.split())
    return canonicalized[:300] or "paper research"


def _contains_paper_alias(text: str, alias: str) -> bool:
    lowered = text.casefold()
    normalized_alias = alias.casefold()
    if re.search(r"[\u4e00-\u9fff]", normalized_alias):
        return normalized_alias in lowered
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])",
            lowered,
        )
    )


def _paper_concept_profile(text: str) -> tuple[set[str], set[str]]:
    domain = {
        concept
        for concept, aliases in _PAPER_DOMAIN_CONCEPTS.items()
        if any(_contains_paper_alias(text, alias) for alias in aliases)
    }
    security = {
        concept
        for concept, aliases in _PAPER_SECURITY_CONCEPTS.items()
        if any(_contains_paper_alias(text, alias) for alias in aliases)
    }
    return domain, security


def _paper_topic_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_.:/-]*|[\u4e00-\u9fff]{2,}", text.casefold())
        if len(token) > 1 and token not in _PAPER_STOPWORDS
    }


def _caller_query_messages(state: AgentState) -> list[str]:
    """Return only persisted caller messages, excluding rendered state and model data."""
    return [
        str(message.get("content") or "")
        for message in state.messages
        if str(message.get("role") or "").casefold() == "user"
    ]


_GENERIC_QUERY_CONTROLS = {
    "继续",
    "请继续",
    "请继续查询",
    "继续查询",
    "请继续检索",
    "继续检索",
    "按前述继续",
    "请按前述继续",
    "按上述继续",
    "请按上述继续",
    "只使用上面的公开检索词",
    "使用上面的公开检索词",
    "请使用上面的公开检索词",
    "只使用上述公开检索词",
    "使用上述公开检索词",
    "请使用上述公开检索词",
    "continuequerying",
    "continuesearching",
    "usethepreviousquery",
    "usetheabovequery",
}

_INHERITABLE_SAFE_LOOKUP_CONTROLS = {
    "仅做授权防御性教学资料梳理分别列来源身份和可核对链接不做真实系统攻击",
    "仅做授权防御性教学资料梳理分别列来源身份和可核对链接不做真实系统攻击或三来源错误关联",
}

_LOOKUP_RESOURCE_KINDS = {
    "paper": {"arxiv", "doi"},
    "vulnerability": {"cve"},
    "project": {"github"},
}
_LOOKUP_RESOURCE_TERMS = {
    "paper": ("论文", "paper", "arxiv", "doi"),
    "vulnerability": ("漏洞", "vulnerability", "cve", "nvd"),
    "standard": ("标准", "standard", "rfc", "3gpp", "specification"),
    "project": ("开源项目", "项目", "仓库", "project", "repository", "github"),
}
_EXTERNAL_QUERY_SOURCES = ("paper", "vulnerability", "standard", "project")
_SOURCE_INTENTS = {
    "paper": AgentIntent.PAPER_SEARCH.value,
    "vulnerability": AgentIntent.VULN_LOOKUP.value,
    "standard": AgentIntent.STANDARD_LOOKUP.value,
    "project": AgentIntent.PROJECT_SEARCH.value,
}
_SOURCE_SCOPE_ACTION_RE = re.compile(
    r"(?:^|[,，;；.!。!?！？]\s*|\band\s+)"
    r"(?:(?:请帮我|请|麻烦你?|please)\s*)?"
    r"(?:"
    r"(?P<only_zh>(?:只|仅)(?:查询|检索|核对|查))|"
    r"(?P<stop_zh>(?:不要再|不再|停止)(?:查询|检索|核对|查))|"
    r"(?P<only_en>only\s+(?:check|search|query|verify|look\s+up))|"
    r"(?P<stop_en>(?:do\s+not|don't|stop)\s+"
    r"(?:check(?:ing)?|search(?:ing)?|query(?:ing)?|look(?:ing)?\s+up))"
    r")\s*",
    flags=re.IGNORECASE,
)
_SOURCE_SCOPE_RELATION_TAIL_RE = re.compile(
    r"(?:并在|用于|作为|以便|然后|并(?:说明|解释|比较)|来(?:说明|解释|比较))|"
    r"\b(?:and|to)\s+(?:explain|describe|use|compare)\b|"
    r"\b(?:for|in|as)\s+(?:(?:a|the)\s+)?"
    r"(?:paper|github|project|teaching|risk|background)\b",
    flags=re.IGNORECASE,
)
_SOURCE_SCOPE_LIST_SEPARATOR_RE = re.compile(
    r"\s*(?:、|，|,|和|与|或|\band\b|\bor\b)\s*",
    flags=re.IGNORECASE,
)


def _is_generic_query_control(value: str) -> bool:
    compact = re.sub(r"[\s,.;:!?，。；：！？、\"'“”‘’]+", "", value.casefold())
    return compact in _GENERIC_QUERY_CONTROLS


def _is_inheritable_lookup_control(value: str) -> bool:
    compact = re.sub(r"[\s,.;:!?，。；：！？、\"'“”‘’]+", "", value.casefold())
    return (
        compact in _GENERIC_QUERY_CONTROLS
        or compact in _INHERITABLE_SAFE_LOOKUP_CONTROLS
    )


def _source_mentions(value: str) -> set[str]:
    lowered = str(value or "").casefold()
    found: set[str] = set()
    for source, terms in _LOOKUP_RESOURCE_TERMS.items():
        for term in terms:
            normalized = term.casefold()
            if re.search(r"[\u4e00-\u9fff]", normalized):
                matched = normalized in lowered
            else:
                matched = bool(
                    re.search(
                        rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])",
                        lowered,
                    )
                )
            if matched:
                found.add(source)
                break
    return found


def _source_for_exact_target(target: Any) -> str | None:
    for source, kinds in _LOOKUP_RESOURCE_KINDS.items():
        if target.kind in kinds:
            return source
    if target.kind == "rfc":
        return "standard"
    return None


def _source_scope_atom(value: str) -> str:
    """Parse one bounded source-list atom without scanning identifier internals."""

    atom = str(value or "").strip(" \t\r\n\"'“”‘’()（）[]【】。.!?！？")
    if not atom:
        raise LookupInputError(
            "当前调用者消息的来源范围不明确。",
            error_code="SOURCE_SCOPE_INVALID",
        )
    github_match = re.fullmatch(r"github\s+(.+)", atom, flags=re.IGNORECASE)
    if github_match:
        github_value = github_match.group(1).strip()
        if github_value.casefold() in {"项目", "仓库", "project", "repository"}:
            return "project"
        github_target = parse_lookup_target(github_value, "project")
        if github_target.mode == "EXACT" and github_target.kind == "github":
            return "project"
        raise LookupInputError(
            "当前调用者消息的 GitHub 来源范围不明确。",
            error_code="SOURCE_SCOPE_INVALID",
        )

    target = parse_lookup_target(atom, None)
    if target.mode == "EXACT":
        source = _source_for_exact_target(target)
        if source is not None:
            return source

    mentioned = _source_mentions(atom)
    if len(mentioned) == 1:
        return next(iter(mentioned))
    raise LookupInputError(
        "当前调用者消息的来源范围不明确。",
        error_code="SOURCE_SCOPE_INVALID",
    )


def _bounded_source_action_atoms(segment: str) -> set[str]:
    bounded = str(segment or "").strip()
    relation = _SOURCE_SCOPE_RELATION_TAIL_RE.search(bounded)
    if relation is not None:
        relation_tail = bounded[relation.end() :].strip(" \t\r\n,，;；。.!?！？")
        if not relation_tail:
            raise LookupInputError(
                "当前调用者消息的来源用途说明不完整。",
                error_code="SOURCE_SCOPE_INVALID",
            )
        bounded = bounded[: relation.start()]
    bounded = bounded.strip(" \t\r\n,，;；。.!?！？")
    if not bounded:
        raise LookupInputError(
            "当前调用者消息的来源范围不明确。",
            error_code="SOURCE_SCOPE_INVALID",
        )
    atoms = _SOURCE_SCOPE_LIST_SEPARATOR_RE.split(bounded)
    if not atoms or any(not atom.strip() for atom in atoms):
        raise LookupInputError(
            "当前调用者消息的来源列表不完整。",
            error_code="SOURCE_SCOPE_INVALID",
        )
    return {_source_scope_atom(atom) for atom in atoms}


def _source_scope_directives(value: str) -> tuple[set[str] | None, set[str], bool]:
    """Return explicit-only sources, explicit exclusions, and whether an action exists."""

    text = " ".join(str(value or "").split()).strip()
    matches = list(_SOURCE_SCOPE_ACTION_RE.finditer(text))
    if not matches:
        return None, set(), False
    only_sources: set[str] | None = None
    exclusions: set[str] = set()
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sources = _bounded_source_action_atoms(text[match.end() : end])
        if match.group("only_zh") or match.group("only_en"):
            only_sources = (only_sources or set()) | sources
        else:
            exclusions.update(sources)
    return only_sources, exclusions, True


def _effective_external_source_scope(
    previous_sources: list[str],
    caller_message: str,
) -> list[str] | None:
    only_sources, exclusions, directive_seen = _source_scope_directives(caller_message)
    if not directive_seen:
        return None
    previous = {
        source for source in previous_sources if source in _EXTERNAL_QUERY_SOURCES
    }
    selected = set(only_sources) if only_sources is not None else set(previous)
    selected.difference_update(exclusions)
    if not selected:
        raise LookupInputError(
            "当前调用者消息未保留可用来源。",
            error_code="SOURCE_SCOPE_EMPTY",
        )
    if not selected.issubset(previous):
        raise LookupInputError(
            "当前会话已收窄来源范围；如需新增来源，请发起新的查询。",
            error_code="SOURCE_SCOPE_EXPANSION_REQUIRED",
        )
    for source in selected:
        # Reuse the authority parser so malformed or conflicting identifiers fail closed.
        parse_lookup_target(caller_message, source)
    return [source for source in _EXTERNAL_QUERY_SOURCES if source in selected]


def _apply_external_continue_scope(state: AgentState, caller_message: str) -> bool:
    if not any(source in _EXTERNAL_QUERY_SOURCES for source in state.required_sources):
        return False
    effective = _effective_external_source_scope(state.required_sources, caller_message)
    if effective is None:
        return False
    previous = list(state.required_sources)
    allowed = set(effective)
    state.required_sources = effective
    state.intent = (
        _SOURCE_INTENTS[effective[0]]
        if len(effective) == 1
        else AgentIntent.MULTI_SOURCE.value
    )
    state.evidence = [
        item
        for item in state.evidence
        if str(item.resource_type or item.source_type or "").casefold()
        not in _EXTERNAL_QUERY_SOURCES
        or str(item.resource_type or item.source_type or "").casefold() in allowed
    ]
    state.completed_sources = [
        source
        for source in state.completed_sources
        if source not in _EXTERNAL_QUERY_SOURCES or source in allowed
    ]
    state.observations = [
        item
        for item in state.observations
        if _TOOL_TO_SOURCE.get(str(item.tool_name or "")) not in _EXTERNAL_QUERY_SOURCES
        or _TOOL_TO_SOURCE.get(str(item.tool_name or "")) in allowed
    ]
    state.available_tools = [
        tool
        for tool in state.available_tools
        if _TOOL_TO_SOURCE.get(tool.name) not in _EXTERNAL_QUERY_SOURCES
        or _TOOL_TO_SOURCE.get(tool.name) in allowed
    ]
    state.add_observation(
        Observation(
            step_number=max(1, state.current_step or 1),
            action="external_source_scope_update",
            success=True,
            content="{}",
            summary="The caller explicitly updated the active external source scope.",
            metadata={
                "content_trust": "trusted_system",
                "previous_sources": previous,
                "effective_sources": list(effective),
            },
        )
    )
    return True


def _explicitly_excludes_lookup_source(value: str, resource_type: str) -> bool:
    if resource_type not in _LOOKUP_RESOURCE_TERMS:
        return False
    only_sources, exclusions, directive_seen = _source_scope_directives(value)
    if not directive_seen:
        return False
    return resource_type in exclusions or (
        only_sources is not None and resource_type not in only_sources
    )


def _bounded_caller_query(value: str, max_chars: int = 300) -> str:
    if len(value) <= max_chars:
        return value
    head_chars = (max_chars - 1) // 2
    tail_chars = max_chars - head_chars - 1
    return f"{value[:head_chars]} {value[-tail_chars:]}"


def _explicit_external_query_origin(
    caller_input: str | list[str],
    *,
    fallback: str,
) -> str:
    """Select a bounded query only from complete persisted caller messages."""
    raw_messages = [caller_input] if isinstance(caller_input, str) else list(caller_input)
    normalized_messages = [
        " ".join(str(message or "").split()).strip()
        for message in raw_messages
    ]
    normalized_messages = [message for message in normalized_messages if message]
    if not normalized_messages:
        return fallback
    selected = next(
        (
            message
            for message in reversed(normalized_messages)
            if not _is_inheritable_lookup_control(message)
        ),
        fallback,
    )
    return _bounded_caller_query(selected) or fallback


def _exact_caller_lookup_target(
    caller_input: str | list[str],
    resource_type: str,
) -> Any | None:
    """Resolve exact caller intent without crossing a newer substantive message."""

    raw_messages = [caller_input] if isinstance(caller_input, str) else list(caller_input)
    for raw_message in reversed(raw_messages):
        message = " ".join(str(raw_message or "").split()).strip()
        if not message:
            continue
        if _explicitly_excludes_lookup_source(message, resource_type):
            raise LookupInputError(
                "当前调用者消息已明确排除该来源。",
                error_code="SOURCE_EXPLICITLY_EXCLUDED",
            )
        target = parse_lookup_target(message, resource_type)
        if target.mode == "EXACT":
            return target
        if _is_inheritable_lookup_control(message):
            continue
        mentioned_sources = _source_mentions(message)
        try:
            cross_source_target = parse_lookup_target(message, None)
        except LookupInputError:
            cross_source_target = None
        if (
            resource_type not in mentioned_sources
            and (
                mentioned_sources
                or (
                    cross_source_target is not None
                    and cross_source_target.mode == "EXACT"
                    and cross_source_target.kind
                    not in _LOOKUP_RESOURCE_KINDS.get(resource_type, set())
                )
            )
        ):
            # A newer message about another still-active source updates only that source.
            continue
        return None
    return None


def _canonical_tool_query_from_target(target: Any) -> str:
    identifier = str(target.identifier or "").strip()
    if target.kind == "arxiv":
        return f"arXiv:{identifier}"
    if target.kind == "doi":
        return f"doi:{identifier}"
    return identifier


def _normalized_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    caller_input: str | list[str],
) -> dict[str, Any]:
    normalized = dict(arguments)
    if tool_name == "paper_search_real":
        contract_text = "\n".join(caller_input) if isinstance(caller_input, list) else caller_input
        contract = _paper_request_contract(contract_text)
        exact_target = _exact_caller_lookup_target(caller_input, "paper")
        if exact_target is not None:
            normalized["query"] = _canonical_tool_query_from_target(exact_target)
            normalized["source"] = "auto"
        else:
            query_origin = _explicit_external_query_origin(
                caller_input,
                fallback="public security research",
            )
            normalized["query"] = _paper_search_query(query_origin)
        normalized["top_k"] = min(10, max(6, contract.target_count * 3))
        if exact_target is None and "source" in normalized:
            normalized["source"] = str(normalized.get("source") or "auto")
    elif tool_name == "vuln_lookup_real":
        has_tool_query = bool(str(normalized.get("query") or "").strip())
        exact_target = _exact_caller_lookup_target(caller_input, "vulnerability")
        query_origin = _explicit_external_query_origin(
            caller_input,
            fallback="public vulnerability metadata",
        )
        cve = re.search(r"CVE-\d{4}-\d{4,8}", query_origin, flags=re.IGNORECASE)
        if exact_target is not None:
            normalized["query"] = _canonical_tool_query_from_target(exact_target)
        elif cve:
            normalized["query"] = cve.group(0).upper()
        elif "cve" in query_origin.lower() and any(
            token in query_origin.lower() for token in ("5g", "nas", "free5gc")
        ):
            normalized["query"] = "5G NAS security vulnerability CVE"
        elif any(
            token in query_origin.lower()
            for token in ("suitable", "sample", "course lab", "\u6837\u4f8b")
        ):
            normalized["query"] = "CVE-2021-44228"
        elif has_tool_query:
            normalized["query"] = query_origin
    elif tool_name == "github_project_search":
        has_tool_query = bool(str(normalized.get("query") or "").strip())
        exact_target = _exact_caller_lookup_target(caller_input, "project")
        query_origin = _explicit_external_query_origin(
            caller_input,
            fallback="public GitHub repository metadata",
        )
        lowered = query_origin.lower()
        if exact_target is not None:
            normalized["query"] = _canonical_tool_query_from_target(exact_target)
        elif "open5gs" in lowered or "5g" in lowered or "nas" in lowered:
            normalized["query"] = "open5gs"
        elif has_tool_query:
            normalized["query"] = query_origin
    elif tool_name in {"standard_metadata_lookup", "standard_lookup_real"} and str(normalized.get("query") or "").strip():
        normalized["query"] = _explicit_external_query_origin(
            caller_input,
            fallback="public standard metadata",
        )
    return normalized


def _resource_evidence_text(item: dict[str, Any], title: str) -> str:
    resource_type = str(item.get("resource_type") or item.get("type") or "").strip().casefold()
    abstract = str(item.get("abstract") or item.get("description") or item.get("summary") or "").strip()
    parts: list[str] = []
    if resource_type == "paper":
        author_summary = _paper_author_summary(item)
        if author_summary:
            parts.append(author_summary)
        if abstract:
            parts.append(_PAPER_ABSTRACT_PREFIX + abstract)
    else:
        parts.append(abstract or title)
    tags = item.get("tags")
    if isinstance(tags, list) and tags:
        parts.append("Metadata tags: " + ", ".join(str(value) for value in tags if value))
    authors = item.get("authors")
    if resource_type != "paper" and isinstance(authors, list):
        safe_authors = [
            _bounded_resource_metadata_value(value, max_chars=160)
            for value in authors[:20]
        ]
        safe_authors = [value for value in safe_authors if value]
        if safe_authors:
            parts.append("Authors: " + ", ".join(safe_authors))
    raw = item.get("raw_metadata")
    if isinstance(raw, dict):
        safe_fields = {
            key: raw.get(key)
            for key in (
                "cvss",
                "cwe",
                "published",
                "lastModified",
                "doi",
                "arxiv_id",
                "stars",
                "updated_at",
                "version",
                "official_metadata_only",
                "local_uploaded",
                "local_verification_status",
            )
            if raw.get(key) not in (None, "")
        }
        if safe_fields:
            bounded_fields = {
                key: _bounded_resource_metadata_value(value, max_chars=300)
                for key, value in safe_fields.items()
            }
            parts.append("Metadata: " + json.dumps(bounded_fields, ensure_ascii=False, sort_keys=True))
    return "\n".join(part for part in parts if part)


def _paper_author_summary(item: dict[str, Any]) -> str:
    if item.get("authors_hash_covered") is not True:
        return ""
    authors = canonical_authors(item.get("authors"))
    if not authors:
        return ""
    complete = _PAPER_AUTHOR_PREFIX + ", ".join(authors) + "."
    if len(complete) <= _PAPER_AUTHOR_SUMMARY_MAX_CHARS:
        return complete
    for count in range(len(authors) - 1, 0, -1):
        prefix = f"{_PAPER_AUTHOR_EXCERPT_PREFIX}{count} of {len(authors)}): "
        candidate = prefix + ", ".join(authors[:count]) + "."
        if len(candidate) <= _PAPER_AUTHOR_SUMMARY_MAX_CHARS:
            return candidate
    return ""


def _bounded_resource_metadata_value(value: Any, *, max_chars: int) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    cleaned = " ".join(str(value or "").split())
    return cleaned[:max_chars]


def _is_trusted_resource(item: dict[str, Any], expected_type: str) -> bool:
    resource_type = str(item.get("resource_type") or item.get("type") or "").strip().lower()
    if expected_type and resource_type != expected_type:
        return False
    if not bool(item.get("provenance_verified")):
        return False
    origin = str(item.get("origin_kind") or "").strip()
    delivery = str(item.get("delivery_mode") or "").strip()
    if origin not in _TRUSTED_ORIGINS or delivery not in _TRUSTED_DELIVERY:
        return False
    title = str(item.get("title") or "").strip()
    url = str(item.get("url") or item.get("source_url") or "").strip()
    parsed = urlparse(url)
    return bool(title and item.get("content_hash") and parsed.scheme == "https" and parsed.netloc)


def _answer_instructions(state: AgentState) -> str:
    base = (
        "Use only the supplied evidence. Cite factual claims with the exact evidence IDs "
        "such as [S1]. Put one factual claim in each sentence or list item and place only its "
        "supporting citation immediately beside it. Cover every required source and do not "
        "invent titles, IDs, URLs, or findings. Select only the best evidence needed for the "
        "question instead of summarizing every supplied record. For a mixed-source answer, "
        "write exactly one concise factual list item per required source, omit unused candidates, "
        "prefer directly verifiable title, ID, or repository identity facts, and do not translate "
        "dates or add numeric details unless the question requires them. Do not add an uncited "
        "factual preamble or conclusion."
    )
    if state.intent == AgentIntent.PRACTICE_GUIDE.value:
        return base + " Produce the six-part course practice framework and keep it within authorized teaching use."
    if state.intent == AgentIntent.INNOVATION.value:
        return base + " Separate evidence-backed cases from model suggestions and include three heuristic questions."
    return base


def _requires_grounded_answer(state: AgentState) -> bool:
    external_sources = {
        _normalized_required_source(source)
        for source in state.required_sources
        if _normalized_required_source(source) in _SOURCE_TO_TOOLS
    }
    return state.intent != AgentIntent.PRACTICE_GUIDE.value and bool(external_sources)


def _build_evidence_spans(evidence: list[EvidenceItem]) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for item in evidence:
        if is_course_evidence(item):
            spans.extend(
                build_course_text_spans(
                    [item],
                    profile=COURSE_SPAN_PROFILE_GROUNDED,
                )
            )
            continue
        counters = {"title": 0, "external_id": 0, "text": 0}

        def add_span(field: str, value: str) -> None:
            cleaned = _clean_evidence_span(value)
            if not cleaned:
                return
            counters[field] += 1
            spans.append(
                EvidenceSpan(
                    span_id=f"{item.source_id}:{field}:{counters[field]}",
                    source_id=item.source_id,
                    field=field,  # type: ignore[arg-type]
                    exact_text=cleaned,
                    resource_type=item.resource_type,
                )
            )

        add_span("title", item.title)
        add_span("external_id", item.external_id)
        for segment in _evidence_text_segments(item.text):
            add_span("text", segment)
    return spans


def _paper_span_semantics(
    evidence: list[EvidenceItem],
    spans: list[EvidenceSpan],
) -> dict[str, str]:
    semantics: dict[str, str] = {}
    evidence_by_id = {item.source_id: item for item in evidence}
    for span in spans:
        item = evidence_by_id.get(span.source_id)
        if item is None or _normalized_required_source(item.resource_type) != "paper":
            continue
        if span.field == "title":
            semantics[span.span_id] = "paper_title"
        elif span.field == "external_id":
            semantics[span.span_id] = "paper_external_id"
        elif span.field == "text":
            author_summary = _paper_author_summary_from_text(item.text)
            if (
                item.tool_name == "paper_search_real"
                and author_summary
                and span.exact_text == author_summary
            ):
                semantics[span.span_id] = "paper_author_summary"
            elif span.exact_text.startswith(_PAPER_ABSTRACT_PREFIX):
                semantics[span.span_id] = "paper_abstract"
    return semantics


_PRACTICE_FIELD_ORDER = (
    "research_goal",
    "method_design",
    "tools",
    "experiment_steps",
    "result_analysis",
    "innovation_points",
    "risks",
)
_PRACTICE_FIELD_LABELS = {
    "research_goal": "研究目标",
    "method_design": "方法设计",
    "tools": "工具选择",
    "experiment_steps": "实验步骤",
    "result_analysis": "结果分析",
    "innovation_points": "创新点",
    "risks": "风险与注意事项",
}
_PRACTICE_RETRY_ERROR_RE = re.compile(r"^[A-Z0-9_]{1,64}$")
_PRACTICE_RETRY_SOURCE_RE = re.compile(r"^S\d+$")
_PRACTICE_RETRY_SPAN_RE = re.compile(r"^S\d+:(?:text|title|external_id):\d+$")
_PRACTICE_FIELD_CUES = {
    "research_goal": ("研究目标", "研究目的", "目标是", "objective", "research goal"),
    "method_design": ("方法设计", "研究方法", "方法采用", "method design", "methodology"),
    "tools": ("工具选择", "实验工具", "使用工具", "工具只需", "工具限", "tool selection", "tools"),
    "result_analysis": ("结果分析", "分析结果", "result analysis", "analyze the result"),
    "innovation_points": ("创新点", "创新之处", "innovation point", "innovation"),
    "risks": ("风险与注意事项", "注意事项", "风险", "risk and caution", "risks"),
}
_PRACTICE_STEP_CUE_RE = re.compile(
    r"(?:实验)?步骤\s*(?:[1-9]|1[0-2]|[一二三四五六七八九十]{1,2})|"
    r"\bstep\s*(?:[1-9]|1[0-2])\b",
    flags=re.IGNORECASE,
)
_PRACTICE_PROCEDURAL_CUE_RE = re.compile(
    r"(?:实验(?:应当|应|需|需要)?记录|实验记录|记录实验)(?!\s*(?:无需|不需要|不是|并非))",
    flags=re.IGNORECASE,
)
_PRACTICE_NEGATION_PREFIX_RE = re.compile(r"(?:不|未|没有|无需|禁止)\s*$", flags=re.IGNORECASE)
_PRACTICE_REQUESTED_STEPS_RE = re.compile(
    r"(?<!\d)(?P<arabic>[1-9]|1[0-2])\s*(?:步|steps?\b)|"
    r"(?P<chinese>十二|十一|十|九|八|七|六|五|四|三|二|一)\s*步",
    flags=re.IGNORECASE,
)
_CHINESE_STEP_COUNTS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


def _practice_retry_feedback(
    *,
    error_code: str,
    failure_detail: dict[str, Any] | None,
    evidence_spans: list[EvidenceSpan],
) -> dict[str, Any]:
    """Return only bounded server-owned details for one Practice regeneration."""

    normalized_error = str(error_code or "").strip().upper()
    if not _PRACTICE_RETRY_ERROR_RE.fullmatch(normalized_error):
        normalized_error = "PRACTICE_GROUNDING_FAILED"
    feedback: dict[str, Any] = {"error_code": normalized_error}
    if not isinstance(failure_detail, dict):
        return feedback
    practice_field = str(failure_detail.get("practice_field") or "").strip()
    source_id = str(failure_detail.get("source_id") or "").strip()
    span_id = str(failure_detail.get("span_id") or "").strip()
    ordinal = failure_detail.get("ordinal")
    if not (
        practice_field in _PRACTICE_FIELD_ORDER
        and isinstance(ordinal, int)
        and not isinstance(ordinal, bool)
        and 1 <= ordinal <= 12
        and _PRACTICE_RETRY_SOURCE_RE.fullmatch(source_id)
        and _PRACTICE_RETRY_SPAN_RE.fullmatch(span_id)
        and span_id.startswith(f"{source_id}:")
    ):
        return feedback
    matching_span = next(
        (
            span
            for span in evidence_spans
            if span.span_id == span_id and span.source_id == source_id
        ),
        None,
    )
    if matching_span is None:
        return feedback
    feedback.update(
        {
            "practice_field": practice_field,
            "ordinal": ordinal,
            "source_id": source_id,
            "evidence_span_id": span_id,
        }
    )
    return feedback


def _practice_candidate_rejection(
    *,
    grounding_retry_attempt: int,
    error_code: str,
    failure_detail: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the canonical bounded diagnostic for one rejected Practice candidate."""

    normalized_error = str(error_code or "").strip().upper()
    if not _PRACTICE_RETRY_ERROR_RE.fullmatch(normalized_error):
        normalized_error = "PRACTICE_GROUNDING_FAILED"
    rejection: dict[str, Any] = {
        "grounding_retry_attempt": 1 if grounding_retry_attempt else 0,
        "error_code": normalized_error,
        "practice_field": "",
        "ordinal": 0,
        "source_id": "",
        "span_id": "",
        "claim_length": 0,
        "claim_sha256": "",
        "normalized_claim_sha256": "",
        "normalization_flags": {
            "space_normalized": False,
            "casefolded": False,
        },
        "failure_detail_available": False,
    }
    if not isinstance(failure_detail, dict):
        return rejection

    practice_field = str(failure_detail.get("practice_field") or "").strip()
    source_id = str(failure_detail.get("source_id") or "").strip()
    span_id = str(failure_detail.get("span_id") or "").strip()
    ordinal = failure_detail.get("ordinal")
    claim_length = failure_detail.get("claim_length")
    claim_sha256 = str(failure_detail.get("claim_sha256") or "").strip().lower()
    normalized_claim_sha256 = str(
        failure_detail.get("normalized_claim_sha256") or ""
    ).strip().lower()
    raw_flags = failure_detail.get("normalization_flags")
    flags = raw_flags if isinstance(raw_flags, dict) else {}
    valid_detail = bool(
        practice_field in _PRACTICE_FIELD_ORDER
        and isinstance(ordinal, int)
        and not isinstance(ordinal, bool)
        and 1 <= ordinal <= 12
        and _PRACTICE_RETRY_SOURCE_RE.fullmatch(source_id)
        and _PRACTICE_RETRY_SPAN_RE.fullmatch(span_id)
        and isinstance(claim_length, int)
        and not isinstance(claim_length, bool)
        and 0 <= claim_length <= 360
        and re.fullmatch(r"[0-9a-f]{64}", claim_sha256)
        and re.fullmatch(r"[0-9a-f]{64}", normalized_claim_sha256)
    )
    if not valid_detail:
        return rejection
    rejection.update(
        {
            "practice_field": practice_field,
            "ordinal": ordinal,
            "source_id": source_id,
            "span_id": span_id,
            "claim_length": claim_length,
            "claim_sha256": claim_sha256,
            "normalized_claim_sha256": normalized_claim_sha256,
            "normalization_flags": {
                "space_normalized": flags.get("space_normalized") is True,
                "casefolded": flags.get("casefolded") is True,
            },
            "failure_detail_available": True,
        }
    )
    return rejection


def _practice_rejection_trace_summary(rejection: dict[str, Any]) -> dict[str, Any]:
    """Return a locator-only trace summary; full hashes stay on model-call metadata."""

    return {
        "grounding_retry_attempt": rejection["grounding_retry_attempt"],
        "error_code": rejection["error_code"],
        "practice_field": rejection["practice_field"],
        "ordinal": rejection["ordinal"],
        "source_id": rejection["source_id"],
        "span_id": rejection["span_id"],
        "failure_detail_available": rejection["failure_detail_available"],
    }


def _extract_requested_practice_step_count(question: str) -> int | None:
    match = _PRACTICE_REQUESTED_STEPS_RE.search(str(question or ""))
    if not match:
        return None
    if match.group("arabic"):
        value = int(match.group("arabic"))
    else:
        value = _CHINESE_STEP_COUNTS.get(str(match.group("chinese") or ""), 0)
    return value if 1 <= value <= 12 else None


def _hydrate_practice_evidence(
    evidence: list[EvidenceItem],
    *,
    course_id: int | None,
    storage: LearningLogger,
    safety_guard: SafetyGuard,
    run_id: str = "",
    observations: list[Observation] | None = None,
    required_sources: list[str] | None = None,
) -> list[EvidenceItem]:
    """Prepare selected course chunks and trusted external summaries for one request."""

    requested_external = {
        _normalized_required_source(source)
        for source in (required_sources or ["course"])
        if _normalized_required_source(source) != "course"
    }
    if requested_external - _PRACTICE_EXTERNAL_SOURCE_TYPES:
        raise GroundedAnswerValidationError(
            "PRACTICE_EVIDENCE_SCOPE_MISMATCH",
            "Practice requested an unsupported external evidence type.",
        )
    trusted_tool_sources = _practice_trusted_tool_sources(observations or [], run_id=run_id)
    hydrated: list[EvidenceItem] = []
    for item in evidence:
        item_type = _normalized_required_source(item.resource_type)
        if item_type != "course":
            if item_type not in requested_external or item_type not in _PRACTICE_EXTERNAL_SOURCE_TYPES:
                raise GroundedAnswerValidationError(
                    "PRACTICE_EVIDENCE_SCOPE_MISMATCH",
                    "Practice evidence contains an unrequested external source type.",
                )
            if not _is_trusted_practice_external_evidence(
                item,
                item_type=item_type,
                trusted_tool_sources=trusted_tool_sources,
            ):
                raise GroundedAnswerValidationError(
                    "PRACTICE_EVIDENCE_SCOPE_MISMATCH",
                    "Practice external evidence failed trusted tool provenance validation.",
                )
            external_text = safety_guard.sanitize_untrusted_text(item.text)[
                :_PRACTICE_EXTERNAL_TEXT_MAX_CHARS
            ]
            if not external_text:
                raise GroundedAnswerValidationError(
                    "PRACTICE_EVIDENCE_INCOMPLETE",
                    "The trusted external summary contains no usable evidence.",
                )
            hydrated.append(item.model_copy(update={"text": external_text}))
            continue

        if item.origin_kind not in {"local_document", "course_document"}:
            raise GroundedAnswerValidationError(
                "PRACTICE_EVIDENCE_SCOPE_MISMATCH",
                "Practice evidence is not an authorized local course source.",
            )
        document_id = int(item.document_id or 0)
        chunk_id = int(item.chunk_id or 0)
        if document_id <= 0:
            local_text = safety_guard.sanitize_untrusted_text(item.text)
            if not local_text:
                raise GroundedAnswerValidationError(
                    "PRACTICE_EVIDENCE_INCOMPLETE",
                    "The controlled local fallback contains no usable evidence.",
                )
            hydrated.append(item.model_copy(update={"text": local_text}))
            continue
        if course_id is None or chunk_id <= 0:
            raise GroundedAnswerValidationError(
                "PRACTICE_EVIDENCE_SCOPE_MISMATCH",
                "Practice evidence is missing an authenticated course or chunk locator.",
            )
        document = storage.document(document_id)
        if (
            document is None
            or int(document.get("course_id") or 0) != int(course_id)
            or str(document.get("index_status") or "").upper() != "INDEXED"
        ):
            raise GroundedAnswerValidationError(
                "PRACTICE_EVIDENCE_SCOPE_MISMATCH",
                "Practice document failed course or index-state validation.",
            )
        chunk = next(
            (
                candidate
                for candidate in storage.chunks_for_document(document_id)
                if int(candidate.get("id") or 0) == chunk_id
                and int(candidate.get("document_id") or 0) == document_id
            ),
            None,
        )
        if chunk is None or str(chunk.get("index_status") or "").upper() != "INDEXED":
            raise GroundedAnswerValidationError(
                "PRACTICE_EVIDENCE_SCOPE_MISMATCH",
                "Practice chunk failed document or index-state validation.",
            )
        chunk_text = safety_guard.sanitize_untrusted_text(str(chunk.get("chunk_text") or ""))
        if not chunk_text:
            raise GroundedAnswerValidationError(
                "PRACTICE_EVIDENCE_INCOMPLETE",
                "The authorized practice chunk contains no usable evidence.",
            )
        hydrated.append(
            item.model_copy(
                update={
                    "text": chunk_text,
                    "content_hash": str(chunk.get("chunk_hash") or _sha256_text(chunk_text)),
                }
            )
        )

    covered_types = {
        _normalized_required_source(item.resource_type)
        for item in hydrated
        if _normalized_required_source(item.resource_type)
    }
    required_coverage = {"course", *requested_external}
    if not required_coverage.issubset(covered_types):
        raise GroundedAnswerValidationError(
            "PRACTICE_EVIDENCE_SCOPE_MISMATCH",
            "Practice evidence is missing a required course or external source.",
        )
    return hydrated


def _extend_practice_course_evidence(
    evidence: list[EvidenceItem],
    *,
    course_id: int | None,
    storage: LearningLogger,
    safety_guard: SafetyGuard,
    required_step_count: int | None,
) -> list[EvidenceItem]:
    """Add only bounded, current sibling chunks that close a Practice evidence gap.

    This is not another retrieval pass.  It starts from the highest-ranked
    current course evidence already selected for this run, reads at most one
    document through the trusted storage interface, and only retains sibling
    chunks that contribute to an explicit missing Practice field or step.
    """

    initial_spans = _build_practice_evidence_spans(evidence)
    missing_fields, missing_steps = _practice_evidence_gaps(
        initial_spans,
        required_step_count=required_step_count,
    )
    if not missing_fields and missing_steps <= 0:
        return evidence
    if course_id is None:
        return evidence

    course_items = [item for item in evidence if is_course_evidence(item)]
    anchors = [
        item
        for item in course_items
        if int(item.document_id or 0) > 0 and int(item.chunk_id or 0) > 0
    ][:_PRACTICE_SUPPORT_MAX_DOCUMENTS]
    if not anchors:
        return evidence

    selected = list(evidence)
    existing_keys = {
        (int(item.document_id or 0), int(item.chunk_id or 0))
        for item in selected
        if is_course_evidence(item)
    }
    max_source_number = max(
        (
            int(match.group(1))
            for item in selected
            if (match := re.fullmatch(r"S([1-9][0-9]*)", item.source_id))
        ),
        default=len(selected),
    )
    used_support_chars = 0

    for anchor in anchors:
        document_id = int(anchor.document_id or 0)
        document = storage.document(document_id)
        if (
            document is None
            or int(document.get("course_id") or 0) != int(course_id)
            or not is_student_visible_document(document)
        ):
            continue
        chunks = storage.chunks_for_document(document_id)
        anchor_chunk = next(
            (
                chunk
                for chunk in chunks
                if int(chunk.get("id") or 0) == int(anchor.chunk_id or 0)
                and int(chunk.get("document_id") or 0) == document_id
                and str(chunk.get("index_status") or "").upper() == "INDEXED"
            ),
            None,
        )
        if anchor_chunk is None:
            continue
        anchor_index = int(anchor_chunk.get("chunk_index") or 0)
        sibling_candidates = sorted(
            (
                chunk
                for chunk in chunks
                if int(chunk.get("document_id") or 0) == document_id
                and int(chunk.get("id") or 0) != int(anchor.chunk_id or 0)
                and str(chunk.get("index_status") or "").upper() == "INDEXED"
                and abs(int(chunk.get("chunk_index") or 0) - anchor_index) <= 2
            ),
            key=lambda chunk: (
                abs(int(chunk.get("chunk_index") or 0) - anchor_index),
                int(chunk.get("chunk_index") or 0),
                int(chunk.get("id") or 0),
            ),
        )
        accepted_for_document = 0
        for chunk in sibling_candidates:
            if accepted_for_document >= _PRACTICE_SUPPORT_MAX_CHUNKS_PER_DOCUMENT:
                break
            key = (document_id, int(chunk.get("id") or 0))
            if key in existing_keys:
                continue
            text = safety_guard.sanitize_untrusted_text(str(chunk.get("chunk_text") or ""))
            if not text or used_support_chars + len(text) > _PRACTICE_SUPPORT_MAX_CHARS:
                continue
            max_source_number += 1
            candidate = EvidenceItem(
                source_id=f"S{max_source_number}",
                text=text,
                source=str(document.get("filename") or anchor.source or "course_document"),
                title=str(document.get("title") or anchor.title or document.get("filename") or "Course document"),
                document_id=document_id,
                chunk_id=int(chunk.get("id") or 0),
                page=chunk.get("page"),
                section=str(chunk.get("section") or ""),
                source_type=str(chunk.get("source_type") or document.get("file_type") or ""),
                score=anchor.score,
                resource_type="course",
                origin_kind="local_document",
                delivery_mode="local_rag",
                content_hash=str(chunk.get("chunk_hash") or _sha256_text(text)),
                trusted=True,
            )
            candidate_spans = _build_practice_evidence_spans([candidate])
            candidate_fields, candidate_steps = _practice_evidence_coverage(
                candidate_spans,
                required_step_count=required_step_count,
            )
            if not (set(missing_fields) & candidate_fields) and not (
                missing_steps > 0 and candidate_steps > 0
            ):
                continue

            # Put support immediately after its trusted anchor, then enforce
            # the existing five-course-source outbound limit by dropping only
            # the lowest-ranked tail course item.  External evidence is never
            # displaced by this local sibling selection.
            anchor_position = selected.index(anchor)
            selected.insert(anchor_position + 1 + accepted_for_document, candidate)
            while sum(1 for item in selected if is_course_evidence(item)) > _PRACTICE_MAX_COURSE_EVIDENCE:
                removable = next(
                    (
                        item
                        for item in reversed(selected)
                        if is_course_evidence(item) and item is not anchor and item is not candidate
                    ),
                    None,
                )
                if removable is None:
                    break
                selected.remove(removable)
            existing_keys.add(key)
            used_support_chars += len(text)
            accepted_for_document += 1

            same_document_selection = [
                item
                for item in selected
                if not is_course_evidence(item)
                or int(item.document_id or 0) == document_id
            ]
            same_document_spans = _build_practice_evidence_spans(same_document_selection)
            same_document_missing, same_document_missing_steps = _practice_evidence_gaps(
                same_document_spans,
                required_step_count=required_step_count,
            )
            if not same_document_missing and same_document_missing_steps <= 0:
                return same_document_selection

            spans = _build_practice_evidence_spans(selected)
            missing_fields, missing_steps = _practice_evidence_gaps(
                spans,
                required_step_count=required_step_count,
            )
            if not missing_fields and missing_steps <= 0:
                return selected
        if not missing_fields and missing_steps <= 0:
            break
    return selected


def _merge_practice_state_evidence(
    original: list[EvidenceItem],
    selected: list[EvidenceItem],
) -> list[EvidenceItem]:
    original_by_identity = {
        (
            str(item.source_id),
            int(item.document_id or 0),
            int(item.chunk_id or 0),
            str(item.resource_type or ""),
        ): item
        for item in original
    }
    merged: list[EvidenceItem] = []
    for item in selected:
        identity = (
            str(item.source_id),
            int(item.document_id or 0),
            int(item.chunk_id or 0),
            str(item.resource_type or ""),
        )
        merged.append(original_by_identity.get(identity, item))
    return merged


def _practice_trusted_tool_sources(
    observations: list[Observation],
    *,
    run_id: str,
) -> dict[str, set[tuple[str, str]]]:
    """Return server-recorded source/tool pairs for successful current-run evidence."""

    trusted: dict[str, set[tuple[str, str]]] = {}
    for observation in observations:
        if observation.action != "call_tool" or not observation.success:
            continue
        tool_name = str(observation.tool_name or "")
        source_type = _TOOL_TO_SOURCE.get(tool_name, "")
        if not source_type:
            continue
        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        if run_id and str(metadata.get("run_id") or "") != run_id:
            continue
        if metadata.get("provenance_verified") is not True:
            continue
        evidence_ids = metadata.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            continue
        for source_id in evidence_ids:
            normalized_id = str(source_id or "").strip()
            if normalized_id:
                trusted.setdefault(normalized_id, set()).add((tool_name, source_type))
    return trusted


def _is_trusted_practice_external_evidence(
    item: EvidenceItem,
    *,
    item_type: str,
    trusted_tool_sources: dict[str, set[tuple[str, str]]],
) -> bool:
    tool_name = str(item.tool_name or "")
    if _TOOL_TO_SOURCE.get(tool_name) != item_type:
        return False
    if (tool_name, item_type) not in trusted_tool_sources.get(item.source_id, set()):
        return False
    parsed = urlparse(str(item.source_url or ""))
    return bool(
        item_type in _PRACTICE_EXTERNAL_SOURCE_TYPES
        and _is_groundable_external_evidence(item)
        and parsed.scheme.casefold() == "https"
        and parsed.netloc
    )


def _build_practice_evidence_spans(evidence: list[EvidenceItem]) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for item in evidence:
        if is_course_evidence(item):
            spans.extend(
                build_course_text_spans(
                    [item],
                    profile=COURSE_SPAN_PROFILE_PRACTICE,
                )
            )
            continue
        counter = 0
        for segment in _practice_evidence_text_segments(item.text):
            counter += 1
            spans.append(
                EvidenceSpan(
                    span_id=f"{item.source_id}:text:{counter}",
                    source_id=item.source_id,
                    field="text",
                    exact_text=segment,
                    resource_type=item.resource_type,
                )
            )
    return spans


def _practice_evidence_text_segments(text: str) -> list[str]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return []
    raw_segments = re.split(r"[.!?。！？；;]+\s*", normalized)
    segments = [_clean_evidence_span(raw) for raw in raw_segments if _clean_evidence_span(raw)]
    return segments[:60]


def _validate_practice_evidence_completeness(
    spans: list[EvidenceSpan],
    *,
    required_step_count: int | None,
) -> None:
    missing_fields, missing_steps = _practice_evidence_gaps(
        spans,
        required_step_count=required_step_count,
    )
    if missing_fields or missing_steps > 0:
        raise GroundedAnswerValidationError(
            "PRACTICE_EVIDENCE_INCOMPLETE",
            "Practice evidence does not explicitly cover every required field and step.",
        )


def _practice_evidence_gaps(
    spans: list[EvidenceSpan],
    *,
    required_step_count: int | None,
) -> tuple[list[str], int]:
    covered_fields, step_count = _practice_evidence_coverage(
        spans,
        required_step_count=required_step_count,
    )
    missing_fields = [
        field for field in _PRACTICE_FIELD_CUES if field not in covered_fields
    ]
    return missing_fields, max(0, (required_step_count or 1) - step_count)


def _practice_evidence_coverage(
    spans: list[EvidenceSpan],
    *,
    required_step_count: int | None,
) -> tuple[set[str], int]:
    texts = [span.exact_text for span in spans if span.field == "text"]
    covered_fields = {
        field
        for field, cues in _PRACTICE_FIELD_CUES.items()
        if any(_practice_has_positive_cue(text, cues) for text in texts)
    }
    if required_step_count is not None:
        step_count = sum(1 for text in texts if _PRACTICE_STEP_CUE_RE.search(text))
    else:
        step_count = sum(
            1
            for text in texts
            if _PRACTICE_STEP_CUE_RE.search(text)
            or _practice_has_positive_procedural_cue(text)
        )
    return covered_fields, step_count


def _practice_has_positive_cue(text: str, cues: tuple[str, ...]) -> bool:
    lowered = str(text or "").casefold()
    for cue in cues:
        normalized_cue = cue.casefold()
        start = 0
        while True:
            index = lowered.find(normalized_cue, start)
            if index < 0:
                break
            prefix = lowered[max(0, index - 4) : index]
            if not _PRACTICE_NEGATION_PREFIX_RE.search(prefix):
                return True
            start = index + len(normalized_cue)
    return False


def _practice_has_positive_procedural_cue(text: str) -> bool:
    value = str(text or "")
    for match in _PRACTICE_PROCEDURAL_CUE_RE.finditer(value):
        prefix = value[max(0, match.start() - 4) : match.start()]
        if not _PRACTICE_NEGATION_PREFIX_RE.search(prefix):
            return True
    return False


def _practice_model_evidence(
    evidence: list[EvidenceItem],
    spans: list[EvidenceSpan],
) -> list[EvidenceItem]:
    spans_by_source: dict[str, list[EvidenceSpan]] = {}
    for span in spans:
        spans_by_source.setdefault(span.source_id, []).append(span)
    return [
        item.model_copy(
            update={
                "text": "\n".join(
                    f"[{span.span_id}] {span.exact_text}"
                    for span in spans_by_source.get(item.source_id, [])
                )
            }
        )
        for item in evidence
    ]


def _evidence_text_segments(text: str) -> list[str]:
    segments: list[str] = []
    for line in str(text or "").splitlines() or [str(text or "")]:
        normalized = " ".join(line.split())
        if not normalized:
            continue
        if normalized.startswith((_PAPER_AUTHOR_PREFIX, _PAPER_AUTHOR_EXCERPT_PREFIX)):
            cleaned = _clean_evidence_span(normalized)
            if cleaned:
                segments.append(cleaned)
            continue
        for raw in _split_evidence_sentences(normalized):
            cleaned = _clean_evidence_span(raw)
            if cleaned:
                segments.append(cleaned)
    if not segments:
        fallback = _clean_evidence_span(text)
        if fallback:
            segments.append(fallback)
    return segments[:8]


_EVIDENCE_ABBREVIATION_RE = re.compile(
    r"(?i)(?:\b(?:i\.e|e\.g|etc|vs|mr|mrs|ms|dr|prof|fig|eq|no)\.|(?<![A-Za-z])[A-Z]\.)"
)


def _split_evidence_sentences(text: str) -> list[str]:
    protected_dots = {
        index
        for match in _EVIDENCE_ABBREVIATION_RE.finditer(text)
        for index in range(match.start(), match.end())
        if text[index] == "."
    }
    segments: list[str] = []
    start = 0
    for index, character in enumerate(text):
        if character not in ".!?。！？":
            continue
        if character == "." and index in protected_dots:
            continue
        if index + 1 < len(text) and not text[index + 1].isspace():
            continue
        segment = text[start : index + 1].strip()
        if segment:
            segments.append(segment)
        start = index + 1
        while start < len(text) and text[start].isspace():
            start += 1
    tail = text[start:].strip()
    if tail:
        segments.append(tail)
    return segments


def _paper_author_summary_from_text(text: str) -> str:
    for line in str(text or "").splitlines():
        normalized = " ".join(line.split())
        if normalized.startswith((_PAPER_AUTHOR_PREFIX, _PAPER_AUTHOR_EXCERPT_PREFIX)):
            return _clean_evidence_span(normalized)
    return ""


def _clean_evidence_span(value: str) -> str:
    return " ".join(str(value or "").split())[:1200]


def _validate_grounded_practice_guide(
    guide: GroundedPracticeGuide,
    evidence: list[EvidenceItem],
    spans: list[EvidenceSpan],
    *,
    required_step_count: int | None,
    required_sources: list[str] | None = None,
) -> dict[str, list[GroundedClaim]]:
    raw_claims: list[GroundedSpanClaim] = []
    field_lengths: dict[str, int] = {}
    evidence_by_id = {item.source_id: item for item in evidence}
    spans_by_id = {span.span_id: span for span in spans}
    seen_bindings: set[tuple[str, str, str]] = set()
    seen_claims: set[str] = set()
    for field in _PRACTICE_FIELD_ORDER:
        field_claims = list(getattr(guide, field))
        if not field_claims:
            raise GroundedAnswerValidationError(
                "PRACTICE_EVIDENCE_INCOMPLETE",
                "A required practice field has no evidence-bound claim.",
            )
        field_lengths[field] = len(field_claims)
        for ordinal, claim in enumerate(field_claims, 1):
            span = spans_by_id.get(claim.evidence_span_id)
            if (
                claim.source_id not in evidence_by_id
                or span is None
                or span.source_id != claim.source_id
            ):
                # Preserve the existing unknown/mismatched source and span
                # error precedence in the authoritative lower validator.
                raw_claims.append(claim)
                continue
            normalized_claim = _normalized_grounded_claim_key(claim.claim_text)
            binding_key = (
                claim.source_id,
                claim.evidence_span_id,
                normalized_claim,
            )
            failure_detail = _practice_duplicate_failure_detail(
                practice_field=field,
                ordinal=ordinal,
                claim=claim,
                normalized_claim=normalized_claim,
            )
            if binding_key in seen_bindings:
                raise GroundedAnswerValidationError(
                    "GROUNDING_DUPLICATE_BINDING",
                    "Practice guide repeated the same source, span, and normalized claim.",
                    failure_detail=failure_detail,
                )
            if normalized_claim in seen_claims:
                raise GroundedAnswerValidationError(
                    "GROUNDING_DUPLICATE_CLAIM",
                    "Practice guide repeated the same normalized claim.",
                    failure_detail=failure_detail,
                )
            seen_bindings.add(binding_key)
            seen_claims.add(normalized_claim)
            raw_claims.append(claim)
    if required_step_count is not None and field_lengths["experiment_steps"] != required_step_count:
        raise GroundedAnswerValidationError(
            "PRACTICE_EVIDENCE_INCOMPLETE",
            "The evidence-bound experiment step count does not match the explicit request.",
        )
    requested_external = {
        _normalized_required_source(source)
        for source in (required_sources or ["course"])
        if _normalized_required_source(source) != "course"
    }
    allowed_types = {"course", *requested_external}
    for claim in raw_claims:
        item = evidence_by_id.get(claim.source_id)
        if item is None:
            continue
        if _normalized_required_source(item.resource_type) not in allowed_types:
            raise GroundedAnswerValidationError(
                "GROUNDING_UNREQUESTED_SOURCE",
                "Practice guide used an unrequested external source type.",
            )
    synthetic = GroundedSpanAnswer.model_construct(claims=raw_claims)
    resolved = _validate_grounded_span_answer(
        synthetic,
        evidence,
        spans,
        required_sources=[],
    )
    covered_types = {
        _normalized_required_source(evidence_by_id[claim.source_id].resource_type)
        for claim in raw_claims
        if claim.source_id in evidence_by_id
    }
    required_coverage = {"course", *requested_external}
    if not required_coverage.issubset(covered_types):
        raise GroundedAnswerValidationError(
            "GROUNDING_SOURCE_COVERAGE",
            f"Practice guide missing source types: {sorted(required_coverage - covered_types)}",
        )
    result: dict[str, list[GroundedClaim]] = {}
    offset = 0
    for field in _PRACTICE_FIELD_ORDER:
        count = field_lengths[field]
        result[field] = resolved[offset : offset + count]
        offset += count
    return result


def _practice_duplicate_failure_detail(
    *,
    practice_field: str,
    ordinal: int,
    claim: GroundedSpanClaim,
    normalized_claim: str,
) -> dict[str, Any]:
    """Return a bounded locator for Practice duplicate correction.

    Candidate and evidence text deliberately stay out of this structure.  The
    hashes let diagnostics distinguish failures without making model output or
    course material recoverable from persisted metadata.
    """

    raw_claim = str(claim.claim_text or "")
    return {
        "practice_field": practice_field,
        "ordinal": ordinal,
        "source_id": str(claim.source_id or ""),
        "span_id": str(claim.evidence_span_id or ""),
        "claim_length": len(raw_claim),
        "claim_sha256": hashlib.sha256(raw_claim.encode("utf-8")).hexdigest(),
        "normalized_claim_sha256": hashlib.sha256(
            normalized_claim.encode("utf-8")
        ).hexdigest(),
        "normalization_flags": {
            "space_normalized": True,
            "casefolded": True,
        },
    }


@dataclass(frozen=True)
class _PracticeRenderUnit:
    practice_field: str
    ordinal: int
    claim_text: str
    source_id: str
    evidence_span_id: str
    evidence_quote: str

    def as_binding(self) -> dict[str, Any]:
        return {
            "practice_field": self.practice_field,
            "ordinal": self.ordinal,
            "claim_text": self.claim_text,
            "source_id": self.source_id,
            "evidence_span_id": self.evidence_span_id,
            "evidence_quote": self.evidence_quote,
        }


def _build_practice_render_units(
    guide: GroundedPracticeGuide,
    validated: dict[str, list[GroundedClaim]],
) -> list[_PracticeRenderUnit]:
    units: list[_PracticeRenderUnit] = []
    for field in _PRACTICE_FIELD_ORDER:
        raw_claims = list(getattr(guide, field))
        resolved_claims = validated.get(field, [])
        if len(raw_claims) != len(resolved_claims):
            raise GroundedAnswerValidationError(
                "PRACTICE_GROUNDING_RENDER_FORMAT",
                "Practice field claim count does not match validated claims.",
            )
        for ordinal, (raw, resolved) in enumerate(
            zip(raw_claims, resolved_claims, strict=True),
            1,
        ):
            raw_claim = str(raw.claim_text or "")
            if (
                "\n" in raw_claim
                or "\r" in raw_claim
                or _GROUNDED_RENDER_DELIMITER_RE.search(raw_claim)
                or CITATION_RE.search(raw_claim)
            ):
                raise GroundedAnswerValidationError(
                    "PRACTICE_GROUNDING_RENDER_FORMAT",
                    "Practice claim contains reserved renderer syntax.",
                )
            claim_text = _clean_grounded_text(raw_claim)
            quote = _clean_grounded_text(resolved.evidence_quote)
            if not claim_text or not quote:
                raise GroundedAnswerValidationError(
                    "PRACTICE_GROUNDING_RENDER_FORMAT",
                    "Practice claim or quote became empty after canonical rendering.",
                )
            units.append(
                _PracticeRenderUnit(
                    practice_field=field,
                    ordinal=ordinal,
                    claim_text=claim_text,
                    source_id=raw.source_id,
                    evidence_span_id=raw.evidence_span_id,
                    evidence_quote=quote,
                )
            )
    return units


def _render_grounded_practice_guide(units: list[_PracticeRenderUnit]) -> str:
    grouped: dict[str, list[_PracticeRenderUnit]] = {field: [] for field in _PRACTICE_FIELD_ORDER}
    for unit in units:
        if unit.practice_field not in grouped:
            raise GroundedAnswerValidationError(
                "PRACTICE_GROUNDING_RENDER_FORMAT",
                "Practice renderer received an unknown field.",
            )
        grouped[unit.practice_field].append(unit)
    sections: list[str] = []
    for field in _PRACTICE_FIELD_ORDER:
        lines = [f"## {_PRACTICE_FIELD_LABELS[field]}"]
        for unit in grouped[field]:
            prefix = f"{unit.ordinal}." if field == "experiment_steps" else "-"
            lines.append(
                f"{prefix} {unit.claim_text} | evidence quote: {unit.evidence_quote} [{unit.source_id}]"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _validate_grounded_span_answer(
    grounded_answer: GroundedSpanAnswer,
    evidence: list[EvidenceItem],
    spans: list[EvidenceSpan],
    *,
    required_sources: list[str],
    paper_contract: dict[str, Any] | None = None,
) -> list[GroundedClaim]:
    evidence_by_id = {item.source_id: item for item in evidence}
    span_by_id = {span.span_id: span for span in spans}
    required_external = {
        _normalized_required_source(source)
        for source in required_sources
        if _normalized_required_source(source) and _normalized_required_source(source) != "course"
    }
    claims: list[GroundedClaim] = []
    seen_claims: set[str] = set()
    seen_bindings: set[tuple[str, str, str]] = set()
    for claim in grounded_answer.claims:
        if CITATION_RE.search(claim.claim_text) or CITATION_RE.search(claim.evidence_span_id):
            raise GroundedAnswerValidationError(
                "GROUNDING_EMBEDDED_CITATION",
                "Grounded claims must not include citation markers inside model fields.",
            )
        normalized_claim = _normalized_grounded_claim_key(claim.claim_text)
        binding_key = (claim.source_id, claim.evidence_span_id, normalized_claim)
        if binding_key in seen_bindings:
            raise GroundedAnswerValidationError(
                "GROUNDING_DUPLICATE_BINDING",
                f"Grounded answer repeated the same source, span, and claim: {claim.source_id}",
            )
        if normalized_claim in seen_claims:
            raise GroundedAnswerValidationError(
                "GROUNDING_DUPLICATE_CLAIM",
                "Grounded answer repeated a claim.",
            )
        item = evidence_by_id.get(claim.source_id)
        if item is None:
            raise GroundedAnswerValidationError(
                "GROUNDING_UNKNOWN_SOURCE",
                f"Grounded claim references unknown source: {claim.source_id}",
            )
        span = span_by_id.get(claim.evidence_span_id)
        if span is None:
            raise GroundedAnswerValidationError(
                "GROUNDING_UNKNOWN_SPAN",
                f"Grounded claim references unknown evidence span: {claim.evidence_span_id}",
            )
        if span.source_id != claim.source_id:
            raise GroundedAnswerValidationError(
                "GROUNDING_SPAN_SOURCE_MISMATCH",
                f"Grounded claim binds source {claim.source_id} to span {claim.evidence_span_id}.",
            )
        item_type = _normalized_required_source(item.resource_type)
        if required_external and item_type not in required_external:
            raise GroundedAnswerValidationError(
                "GROUNDING_UNREQUESTED_SOURCE",
                f"Grounded claim used an unrequested source type: {item_type}",
            )
        if item_type != "course" and not _is_groundable_external_evidence(item):
            raise GroundedAnswerValidationError(
                "GROUNDING_UNTRUSTED_SOURCE",
                f"Grounded claim references untrusted evidence: {claim.source_id}",
            )
        resolved = GroundedClaim(
            claim_text=claim.claim_text,
            source_id=claim.source_id,
            evidence_quote=span.exact_text,
        )
        if not _normalized_quote_match(resolved.evidence_quote, item):
            raise GroundedAnswerValidationError(
                "GROUNDING_QUOTE_MISMATCH",
                f"Grounded span does not belong to source: {claim.source_id}",
            )
        if _contains_foreign_identifier(resolved, item, evidence):
            raise GroundedAnswerValidationError(
                "GROUNDING_IDENTIFIER_MISMATCH",
                f"Grounded claim binds an identifier to the wrong source: {claim.source_id}",
            )
        seen_bindings.add(binding_key)
        seen_claims.add(normalized_claim)
        claims.append(resolved)

    covered_types = {
        _normalized_required_source(evidence_by_id[claim.source_id].resource_type)
        for claim in claims
        if claim.source_id in evidence_by_id
    }
    if required_external and not required_external.issubset(covered_types):
        raise GroundedAnswerValidationError(
            "GROUNDING_SOURCE_COVERAGE",
            f"Grounded answer missing source types: {sorted(required_external - covered_types)}",
        )
    if paper_contract:
        _validate_selected_paper_fields(
            grounded_answer,
            evidence,
            spans,
            paper_contract,
        )
    return claims


def _validate_selected_paper_fields(
    grounded_answer: GroundedSpanAnswer,
    evidence: list[EvidenceItem],
    spans: list[EvidenceSpan],
    paper_contract: dict[str, Any],
) -> None:
    selected_ids = [
        str(value)
        for value in paper_contract.get("selected_source_ids", [])
        if isinstance(value, str) and re.fullmatch(r"S\d+", value)
    ]
    required_fields = {
        str(value)
        for value in paper_contract.get("required_fields", [])
        if str(value) in {"title", "authors", "abstract_clue", "link"}
    }
    raw_semantics = paper_contract.get("span_semantics")
    semantics = raw_semantics if isinstance(raw_semantics, dict) else {}
    raw_urls = paper_contract.get("citation_urls")
    citation_urls = raw_urls if isinstance(raw_urls, dict) else {}
    evidence_by_id = {item.source_id: item for item in evidence}
    spans_by_id = {span.span_id: span for span in spans}
    selected_set = set(selected_ids)

    for claim in grounded_answer.claims:
        item = evidence_by_id.get(claim.source_id)
        if (
            item is not None
            and _normalized_required_source(item.resource_type) == "paper"
            and claim.source_id not in selected_set
        ):
            raise GroundedAnswerValidationError(
                "PAPER_UNSELECTED_SOURCE",
                "A model claim referenced a paper outside the server-selected set.",
                {"source_id": claim.source_id, "missing_fields": []},
            )

    for source_id in selected_ids:
        item = evidence_by_id.get(source_id)
        source_claims = [claim for claim in grounded_answer.claims if claim.source_id == source_id]
        missing: list[str] = []
        if item is None or _normalized_required_source(item.resource_type) != "paper":
            missing = sorted(required_fields or {"paper"})
        else:
            if "title" in required_fields and not any(
                (span := spans_by_id.get(claim.evidence_span_id)) is not None
                and span.field == "title"
                for claim in source_claims
            ):
                missing.append("title")
            if "authors" in required_fields:
                author_claims = [
                    claim
                    for claim in source_claims
                    if semantics.get(claim.evidence_span_id) == "paper_author_summary"
                ]
                valid_author = any(
                    (span := spans_by_id.get(claim.evidence_span_id)) is not None
                    and " ".join(claim.claim_text.split()) == " ".join(span.exact_text.split())
                    for claim in author_claims
                )
                if not valid_author:
                    missing.append("authors")
            if "abstract_clue" in required_fields and not any(
                semantics.get(claim.evidence_span_id) == "paper_abstract"
                for claim in source_claims
            ):
                missing.append("abstract_clue")
            if "link" in required_fields:
                expected_url = _paper_citation_url(item)
                recorded_url = str(citation_urls.get(source_id) or "")
                if not expected_url or recorded_url != expected_url:
                    missing.append("link")
        if missing:
            raise GroundedAnswerValidationError(
                "PAPER_FIELD_COVERAGE",
                f"Selected paper {source_id} is missing required fields: {missing}",
                {"source_id": source_id, "missing_fields": missing},
            )


def _validate_grounded_answer(
    grounded_answer: GroundedAnswer,
    evidence: list[EvidenceItem],
    *,
    required_sources: list[str],
) -> list[GroundedClaim]:
    evidence_by_id = {item.source_id: item for item in evidence}
    required_external = {
        _normalized_required_source(source)
        for source in required_sources
        if _normalized_required_source(source) and _normalized_required_source(source) != "course"
    }
    claims: list[GroundedClaim] = []
    seen_claims: set[str] = set()
    seen_bindings: set[tuple[str, str, str]] = set()
    for claim in grounded_answer.claims:
        if CITATION_RE.search(claim.claim_text) or CITATION_RE.search(claim.evidence_quote):
            raise GroundedAnswerValidationError(
                "GROUNDING_EMBEDDED_CITATION",
                "Grounded claims must not include citation markers inside model fields.",
            )
        normalized_claim = _normalized_grounded_claim_key(claim.claim_text)
        normalized_quote = _normalized_grounded_quote_key(claim.evidence_quote)
        binding_key = (claim.source_id, normalized_claim, normalized_quote)
        if binding_key in seen_bindings:
            raise GroundedAnswerValidationError(
                "GROUNDING_DUPLICATE_BINDING",
                f"Grounded answer repeated the same source, claim, and quote: {claim.source_id}",
            )
        if normalized_claim in seen_claims:
            raise GroundedAnswerValidationError(
                "GROUNDING_DUPLICATE_CLAIM",
                "Grounded answer repeated a claim.",
            )
        item = evidence_by_id.get(claim.source_id)
        if item is None:
            raise GroundedAnswerValidationError(
                "GROUNDING_UNKNOWN_SOURCE",
                f"Grounded claim references unknown source: {claim.source_id}",
            )
        item_type = _normalized_required_source(item.resource_type)
        if required_external and item_type not in required_external:
            raise GroundedAnswerValidationError(
                "GROUNDING_UNREQUESTED_SOURCE",
                f"Grounded claim used an unrequested source type: {item_type}",
            )
        if item_type != "course" and not _is_groundable_external_evidence(item):
            raise GroundedAnswerValidationError(
                "GROUNDING_UNTRUSTED_SOURCE",
                f"Grounded claim references untrusted evidence: {claim.source_id}",
            )
        if not _normalized_quote_match(claim.evidence_quote, item):
            raise GroundedAnswerValidationError(
                "GROUNDING_QUOTE_MISMATCH",
                f"Grounded quote does not belong to source: {claim.source_id}",
            )
        if _contains_foreign_identifier(claim, item, evidence):
            raise GroundedAnswerValidationError(
                "GROUNDING_IDENTIFIER_MISMATCH",
                f"Grounded claim binds an identifier to the wrong source: {claim.source_id}",
            )
        seen_bindings.add(binding_key)
        seen_claims.add(normalized_claim)
        claims.append(claim)

    covered_types = {
        _normalized_required_source(evidence_by_id[claim.source_id].resource_type)
        for claim in claims
        if claim.source_id in evidence_by_id
    }
    if required_external and not required_external.issubset(covered_types):
        raise GroundedAnswerValidationError(
            "GROUNDING_SOURCE_COVERAGE",
            f"Grounded answer missing source types: {sorted(required_external - covered_types)}",
        )
    return claims


@dataclass(frozen=True)
class _GroundedRenderUnit:
    claim_text: str
    source_id: str
    evidence_span_id: str
    evidence_quote: str


_GROUNDED_RENDER_DELIMITER_RE = re.compile(r"\|\s*evidence\s+quote\s*:", flags=re.IGNORECASE)


def _build_grounded_render_units(
    grounded_answer: GroundedSpanAnswer,
    claims: list[GroundedClaim],
) -> list[_GroundedRenderUnit]:
    if len(grounded_answer.claims) != len(claims):
        raise GroundedAnswerValidationError(
            "GROUNDING_RENDER_FORMAT",
            "Grounded answer claim count does not match validated claims.",
        )
    units: list[_GroundedRenderUnit] = []
    for raw, resolved in zip(grounded_answer.claims, claims, strict=True):
        raw_claim = str(raw.claim_text or "")
        if (
            "\n" in raw_claim
            or "\r" in raw_claim
            or _GROUNDED_RENDER_DELIMITER_RE.search(raw_claim)
            or CITATION_RE.search(raw_claim)
        ):
            raise GroundedAnswerValidationError(
                "GROUNDING_RENDER_FORMAT",
                "Grounded claim contains reserved renderer syntax.",
            )
        claim_text = _clean_grounded_text(raw_claim)
        quote = _clean_grounded_text(resolved.evidence_quote)
        if not claim_text or not quote:
            raise GroundedAnswerValidationError(
                "GROUNDING_RENDER_FORMAT",
                "Grounded claim or evidence quote became empty after canonical rendering.",
            )
        units.append(
            _GroundedRenderUnit(
                claim_text=claim_text,
                source_id=raw.source_id,
                evidence_span_id=raw.evidence_span_id,
                evidence_quote=quote,
            )
        )
    return units


def _render_grounded_answer(
    claims: list[GroundedClaim] | list[_GroundedRenderUnit],
    *,
    evidence: list[EvidenceItem] | None = None,
    presentation_profile: str | None = None,
) -> str:
    units = [
        claim
        if isinstance(claim, _GroundedRenderUnit)
        else _GroundedRenderUnit(
            claim_text=_clean_grounded_text(claim.claim_text),
            source_id=claim.source_id,
            evidence_span_id="",
            evidence_quote=_clean_grounded_text(claim.evidence_quote),
        )
        for claim in claims
    ]
    return render_grounded_presentation(
        units,
        evidence or [],
        presentation_profile=presentation_profile,
    )


def _grounded_presentation_profile(state: AgentState) -> str | None:
    required = {
        _normalized_required_source(source)
        for source in state.required_sources
        if _normalized_required_source(source)
    }
    question = " ".join(str(state.user_goal or "").split()).casefold()
    asks_for_source_paragraphs = (
        ("三段" in question and "分别" in question and "教学背景" in question)
        or (
            "three paragraphs" in question
            and "teaching background" in question
            and any(token in question for token in ("separately", "each source", "by source"))
        )
    )
    if required == {"paper", "vulnerability", "project"} and asks_for_source_paragraphs:
        return MULTI_SOURCE_TEACHING_BACKGROUND_PROFILE
    return None


def _grounded_span_citation_bindings(
    units: list[_GroundedRenderUnit],
) -> list[dict[str, str]]:
    return [
        {
            "claim_text": unit.claim_text,
            "source_id": unit.source_id,
            "evidence_span_id": unit.evidence_span_id,
            "evidence_quote": unit.evidence_quote,
        }
        for unit in units
    ]


def _clean_grounded_text(value: str) -> str:
    text = " ".join(str(value or "").split())
    return text.strip(" \t\r\n.,;:!?。，；：！？")


def _normalized_grounded_claim_key(value: str) -> str:
    text = CITATION_RE.sub("", str(value or ""))
    return re.sub(r"\s+", " ", _clean_grounded_text(text)).casefold()


def _normalized_grounded_quote_key(value: str) -> str:
    return re.sub(r"\s+", " ", _clean_grounded_text(value)).casefold()


def _normalized_quote_match(quote: str, item: EvidenceItem) -> bool:
    needle = " ".join(str(quote or "").split())
    if not needle:
        return False
    haystacks = (
        item.text,
        item.title,
        item.external_id,
    )
    return any(needle in " ".join(str(value or "").split()) for value in haystacks if value)


def _is_groundable_external_evidence(item: EvidenceItem) -> bool:
    return bool(
        item.trusted
        and item.source_url.startswith("https://")
        and item.content_hash
        and (item.external_id or item.title)
        and item.origin_kind in _TRUSTED_ORIGINS
        and item.delivery_mode in _TRUSTED_DELIVERY
    )


def _contains_foreign_identifier(
    claim: GroundedClaim,
    item: EvidenceItem,
    evidence: list[EvidenceItem],
) -> bool:
    quote_haystack = _normalize_identifier_text(claim.evidence_quote)
    canonical_identifiers = _canonical_identifiers_for_item(item)
    for occurrence in _identifier_occurrences(claim.claim_text, claim.evidence_quote, evidence):
        normalized_value = occurrence["normalized"]
        if len(normalized_value) < 4:
            continue
        if normalized_value in quote_haystack:
            continue
        if normalized_value in canonical_identifiers.get(occurrence["type"], set()):
            continue
        return True
    return False


def _normalize_identifier_text(value: str) -> str:
    text = str(value or "").strip().strip(".,;:!?)]}\uff0c\u3002\uff1b\uff1a\uff01\uff1f\uff09\u3011")
    return re.sub(r"\s+", "", text).casefold()


def _identifier_occurrences(
    claim_text: str,
    evidence_quote: str,
    evidence: list[EvidenceItem],
) -> list[dict[str, str]]:
    known_repositories = _trusted_project_repository_identifiers(evidence)
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for origin, text in (("claim_text", claim_text), ("evidence_quote", evidence_quote)):
        for identifier_type, value in _extract_identifier_values(text, known_repositories):
            normalized = _normalize_identifier_value(identifier_type, value)
            if not normalized:
                continue
            key = (identifier_type, normalized)
            entry = merged.setdefault(
                key,
                {
                    "type": identifier_type,
                    "value": value,
                    "normalized": normalized,
                    "origins": set(),
                },
            )
            entry["origins"].add(origin)
    occurrences: list[dict[str, str]] = []
    for entry in merged.values():
        origins = entry["origins"]
        if origins == {"claim_text", "evidence_quote"}:
            origin = "both"
        elif "claim_text" in origins:
            origin = "claim_text"
        else:
            origin = "evidence_quote"
        occurrences.append(
            {
                "type": entry["type"],
                "value": entry["value"],
                "normalized": entry["normalized"],
                "origin": origin,
            }
        )
    return occurrences


def _normalize_identifier_value(identifier_type: str, value: str) -> str:
    if identifier_type == "url":
        return _normalize_source_url_identifier(value)
    return _normalize_identifier_text(value)


def _normalize_source_url_identifier(value: str) -> str:
    text = str(value or "").strip().strip(".,;:!?)]}\uff0c\u3002\uff1b\uff1a\uff01\uff1f\uff09\u3011")
    parsed = urlparse(text)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return _normalize_identifier_text(text)
    path = parsed.path or ""
    if path != "/":
        path = path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme.casefold()}://{parsed.netloc.casefold()}{path}{query}".casefold()


def _extract_identifier_values(text: str, known_repositories: set[str]) -> list[tuple[str, str]]:
    source = str(text or "")
    identifiers: list[tuple[str, str]] = []
    url_ranges: list[tuple[int, int]] = []
    for match in re.finditer(r"https://[^\s<>\]\[)]+", source, flags=re.IGNORECASE):
        value = match.group(0).rstrip(".,;:\uff0c\u3002\uff1b\uff1a")
        end = match.start() + len(value)
        url_ranges.append((match.start(), end))
        identifiers.append(("url", value))
    for match in re.finditer(r"\bCVE-\d{4}-\d{4,8}\b", source, flags=re.IGNORECASE):
        identifiers.append(("cve", match.group(0)))
    for match in re.finditer(r"(?<!\d)(\d{4}\.\d{4,5}(?:v\d+)?)(?!\d)", source, flags=re.IGNORECASE):
        identifiers.append(("arxiv", match.group(1)))
    repository_pattern = re.compile(
        r"(?<![A-Za-z0-9_.\-/])([A-Za-z0-9][A-Za-z0-9_.-]{0,100}/[A-Za-z0-9][A-Za-z0-9_.-]{0,100})(?![A-Za-z0-9_.\-/])"
    )
    for match in repository_pattern.finditer(source):
        if _range_overlaps(match.start(1), match.end(1), url_ranges):
            continue
        value = match.group(1)
        normalized = _normalize_identifier_text(value)
        if normalized in known_repositories or _has_repository_context(source, match.start(1), match.end(1)):
            identifiers.append(("repository", value))
    return identifiers


def _canonical_identifiers_for_item(item: EvidenceItem) -> dict[str, set[str]]:
    canonical: dict[str, set[str]] = {
        "arxiv": set(),
        "cve": set(),
        "repository": set(),
        "url": set(),
    }
    if not _is_groundable_external_evidence(item):
        return canonical

    item_type = _normalized_required_source(item.resource_type)
    source_url = _normalize_source_url_identifier(item.source_url)
    if source_url and str(item.source_url or "").strip().casefold().startswith("https://"):
        canonical["url"].add(source_url)

    if item_type == "paper":
        arxiv_id = _arxiv_identifier_from_external_id(item.external_id)
        if arxiv_id:
            canonical["arxiv"].add(arxiv_id)
        arxiv_id = _arxiv_identifier_from_url(item.source_url)
        if arxiv_id:
            canonical["arxiv"].add(arxiv_id)
    elif item_type == "vulnerability":
        cve = _cve_identifier_from_external_id(item.external_id)
        if cve:
            canonical["cve"].add(cve)
        cve = _cve_identifier_from_nvd_url(item.source_url)
        if cve:
            canonical["cve"].add(cve)
    elif item_type == "project":
        if _is_repository_literal(item.external_id):
            canonical["repository"].add(_normalize_identifier_text(item.external_id))
        repo_from_url = _github_repository_from_url(item.source_url)
        if repo_from_url:
            canonical["repository"].add(_normalize_identifier_text(repo_from_url))
    return canonical


def _arxiv_identifier_from_external_id(value: str) -> str:
    text = str(value or "").strip().strip(".,;:!?)]}")
    if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", text, flags=re.IGNORECASE):
        return _normalize_identifier_text(text)
    return ""


def _arxiv_identifier_from_url(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme != "https" or parsed.netloc.casefold() not in {"arxiv.org", "www.arxiv.org"}:
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0].casefold() != "abs":
        return ""
    return _arxiv_identifier_from_external_id(parts[1])


def _cve_identifier_from_external_id(value: str) -> str:
    text = str(value or "").strip().strip(".,;:!?)]}").upper()
    if re.fullmatch(r"CVE-\d{4}-\d{4,8}", text):
        return _normalize_identifier_text(text)
    return ""


def _cve_identifier_from_nvd_url(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme != "https" or parsed.netloc.casefold() != "nvd.nist.gov":
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 3 or [part.casefold() for part in parts[:2]] != ["vuln", "detail"]:
        return ""
    return _cve_identifier_from_external_id(parts[2])


def _trusted_project_repository_identifiers(evidence: list[EvidenceItem]) -> set[str]:
    repositories: set[str] = set()
    for item in evidence:
        if _normalized_required_source(item.resource_type) != "project":
            continue
        if not _is_groundable_external_evidence(item):
            continue
        for value in (item.title, item.external_id):
            if _is_repository_literal(value):
                repositories.add(_normalize_identifier_text(value))
        repo_from_url = _github_repository_from_url(item.source_url)
        if repo_from_url:
            repositories.add(_normalize_identifier_text(repo_from_url))
    return repositories


def _is_repository_literal(value: str) -> bool:
    text = str(value or "").strip()
    return bool(
        re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}/[A-Za-z0-9][A-Za-z0-9_.-]{0,100}",
            text,
        )
    )


def _github_repository_from_url(value: str) -> str:
    parsed = urlparse(str(value or ""))
    if parsed.scheme != "https" or parsed.netloc.casefold() != "github.com":
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return ""
    return f"{parts[0]}/{parts[1]}"


def _has_repository_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 48) : min(len(text), end + 48)].casefold()
    return any(marker in window for marker in ("github", "repository", "repo", "\u4ed3\u5e93"))


def _range_overlaps(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < range_end and end > range_start for range_start, range_end in ranges)


def _normalized_required_source(source: str) -> str:
    normalized = str(source or "").strip().lower()
    aliases = {
        "vuln": "vulnerability",
        "cve": "vulnerability",
        "github": "project",
        "repository": "project",
        "document": "course",
    }
    return aliases.get(normalized, normalized)


def _paper_evidence_selection(state: AgentState) -> _PaperSelection | None:
    required_types = {
        _normalized_required_source(source)
        for source in state.required_sources
        if _normalized_required_source(source)
    }
    if "paper" not in required_types:
        return None
    contract = _paper_request_contract(state.user_goal)
    exact_target, invalid_exact = _paper_exact_lookup_target(state.user_goal)
    external_types = {source for source in required_types if source in _SOURCE_TO_TOOLS}
    requested_count = (
        1
        if exact_target is not None or invalid_exact or len(external_types) > 1
        else contract.target_count
    )
    candidates = [
        item
        for item in state.evidence
        if _normalized_required_source(item.resource_type) == "paper"
        and item.trusted
        and _is_groundable_external_evidence(item)
    ]
    ranked: list[tuple[tuple[int, int, int, float], EvidenceItem]] = []
    for item in candidates:
        score = _paper_relevance_score(
            state.user_goal,
            item,
            exact_target=exact_target,
            invalid_exact=invalid_exact,
        )
        if score is not None:
            ranked.append((score, item))
    ranked.sort(
        key=lambda pair: (
            -pair[0][0],
            -pair[0][1],
            -pair[0][2],
            -pair[0][3],
            pair[1].external_id.casefold(),
            pair[1].source_id,
        )
    )
    related: list[EvidenceItem] = []
    identities: set[str] = set()
    for _score, item in ranked:
        identity = "|".join(
            part for part in (
                item.external_id.casefold(),
                _normalize_source_url_identifier(item.source_url),
                item.content_hash,
            ) if part
        )
        if not identity or identity in identities:
            continue
        identities.add(identity)
        related.append(item)
    if not related:
        return _PaperSelection(
            contract=contract,
            selected=(),
            requested_count=requested_count,
            related_count=0,
            field_complete_count=0,
            terminal_error="PAPER_NO_RELEVANT_RESULTS",
            shortfall_reason="NO_RELEVANT_RESULTS",
        )

    field_complete = [
        item
        for item in related
        if set(contract.required_fields).issubset(_paper_available_fields(item))
    ]
    if not field_complete:
        return _PaperSelection(
            contract=contract,
            selected=(),
            requested_count=requested_count,
            related_count=len(related),
            field_complete_count=0,
            terminal_error="PAPER_REQUIRED_FIELD_UNAVAILABLE",
            shortfall_reason="REQUIRED_FIELD_UNAVAILABLE",
        )
    selected = tuple(field_complete[:requested_count])
    completion_reason = "ANSWERED"
    shortfall = ""
    if contract.explicit_count and len(selected) < requested_count:
        if len(related) >= requested_count:
            completion_reason = "PAPER_PARTIAL_FIELD_RESULTS"
            shortfall = "REQUIRED_FIELD_SHORTFALL"
        else:
            completion_reason = "PAPER_PARTIAL_RELEVANCE_RESULTS"
            shortfall = "RELATED_COUNT_SHORTFALL"
    return _PaperSelection(
        contract=contract,
        selected=selected,
        requested_count=requested_count,
        related_count=len(related),
        field_complete_count=len(field_complete),
        completion_reason=completion_reason,
        shortfall_reason=shortfall,
    )


def _paper_relevance_score(
    question: str,
    item: EvidenceItem,
    *,
    exact_target: LookupTarget | None = None,
    invalid_exact: bool | None = None,
) -> tuple[int, int, int, float] | None:
    if exact_target is None and invalid_exact is None:
        exact_target, invalid_exact = _paper_exact_lookup_target(question)
    if invalid_exact:
        return None
    if exact_target is not None:
        if not _paper_matches_exact_target(exact_target, item):
            return None
        return (2, 2, len(str(exact_target.identifier or "")), float(item.score or 0.0))
    normalized_question = " ".join(question.casefold().split())
    for locator in (item.external_id, item.title, item.source_url):
        normalized_locator = " ".join(str(locator or "").casefold().split())
        if normalized_locator and normalized_locator in normalized_question:
            return (1, 2, len(normalized_locator), float(item.score or 0.0))
    query = _paper_search_query(question)
    query_domain, query_security = _paper_concept_profile(query)
    candidate_text = " ".join((item.title, item.text, item.external_id))
    candidate_domain, candidate_security = _paper_concept_profile(candidate_text)
    domain_overlap = query_domain & candidate_domain
    security_overlap = query_security & candidate_security
    if query_domain and query_security and (not domain_overlap or not security_overlap):
        return None
    if query_domain and not query_security and not domain_overlap:
        return None
    if query_security and not query_domain and not security_overlap:
        return None
    query_tokens = _paper_topic_tokens(query)
    candidate_tokens = _paper_topic_tokens(candidate_text)
    token_overlap = query_tokens & candidate_tokens
    if not query_domain and not query_security and query_tokens and not token_overlap:
        return None
    return (
        0,
        len(domain_overlap) + len(security_overlap),
        len(token_overlap),
        float(item.score or 0.0),
    )


def _paper_exact_lookup_target(question: str) -> tuple[LookupTarget | None, bool]:
    try:
        target = parse_lookup_target(question, "paper")
    except LookupInputError:
        return None, True
    if target.mode == "EXACT" and target.kind in {"arxiv", "doi"}:
        return target, False
    return None, False


def _paper_matches_exact_target(target: LookupTarget, item: EvidenceItem) -> bool:
    if item.origin_kind != "remote" or item.delivery_mode not in {"live", "fresh_cache"}:
        return False
    resource = {
        "external_id": item.external_id,
        "raw_metadata": {"arxiv_id": item.external_id},
    }
    if not matches_exact_identity(target, item.source, resource):
        return False
    if target.kind != "arxiv":
        return True
    try:
        url_target = parse_lookup_target(item.source_url, "paper")
    except LookupInputError:
        return False
    parsed = urlparse(item.source_url)
    if parsed.scheme.casefold() != "https" or parsed.netloc.casefold() != "arxiv.org":
        return False
    if not parsed.path.casefold().startswith("/abs/"):
        return False
    return (
        url_target.mode == "EXACT"
        and url_target.kind == "arxiv"
        and matches_exact_identity(target, "arxiv", {
            "external_id": url_target.identifier,
            "raw_metadata": {"arxiv_id": url_target.identifier},
        })
        and matches_exact_identity(url_target, item.source, resource)
    )


def _paper_available_fields(item: EvidenceItem) -> set[str]:
    fields: set[str] = set()
    if item.title.strip():
        fields.add("title")
    if item.tool_name == "paper_search_real" and _paper_author_summary_from_text(item.text):
        fields.add("authors")
    if any(
        segment.startswith(_PAPER_ABSTRACT_PREFIX) and segment != _PAPER_ABSTRACT_PREFIX
        for segment in _evidence_text_segments(item.text)
    ):
        fields.add("abstract_clue")
    if _paper_citation_url(item):
        fields.add("link")
    return fields


def _paper_citation_url(item: EvidenceItem) -> str:
    citation = _citation_from_evidence(item)
    source_url = str(citation.get("source_url") or "").strip()
    parsed = urlparse(source_url)
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.netloc
        or source_url != str(item.source_url or "").strip()
        or not item.content_hash
        or not item.external_id
    ):
        return ""
    return source_url


def _paper_selection_metadata(
    selection: _PaperSelection,
    evidence_spans: list[EvidenceSpan],
) -> dict[str, Any]:
    selected = list(selection.selected)
    return {
        "version": "paper-selection-v1",
        "concept_profile_version": _PAPER_CONCEPT_VERSION,
        "selected_source_ids": [item.source_id for item in selected],
        "required_fields": list(selection.contract.required_fields),
        "target_count": selection.requested_count,
        "requested_count": selection.requested_count,
        "related_count": selection.related_count,
        "field_complete_count": selection.field_complete_count,
        "selected_count": len(selected),
        "completion_reason": selection.completion_reason,
        "shortfall_reason": selection.shortfall_reason,
        "span_semantics": _paper_span_semantics(selected, evidence_spans),
        "citation_urls": {
            item.source_id: _paper_citation_url(item)
            for item in selected
            if _paper_citation_url(item)
        },
    }


def _answer_evidence(
    state: AgentState,
    paper_selection: _PaperSelection | None = None,
) -> list[EvidenceItem]:
    if state.course_document_constraint_title:
        return [
            item
            for item in state.evidence
            if _is_constrained_course_evidence(state, item)
        ]
    if state.intent == AgentIntent.PRACTICE_GUIDE.value:
        return state.evidence
    required_types = [
        normalized
        for source in state.required_sources
        if (normalized := _normalized_required_source(source))
    ]
    external_types = [source for source in required_types if source in _SOURCE_TO_TOOLS]
    if not external_types:
        return state.evidence

    selected: list[EvidenceItem] = []
    for resource_type in required_types:
        candidates = [
            item
            for item in state.evidence
            if _normalized_required_source(item.resource_type) == resource_type and item.trusted
        ]
        if not candidates:
            return state.evidence
        if resource_type == "paper" and paper_selection is not None:
            selected.extend(paper_selection.selected)
        elif len(external_types) == 1 and resource_type == external_types[0]:
            selected.extend(candidates)
        else:
            selected.append(
                max(candidates, key=lambda item: _evidence_goal_score(state.user_goal, item))
            )
    return selected


def _evidence_goal_score(question: str, item: EvidenceItem) -> tuple[int, int, float]:
    normalized_question = " ".join(question.casefold().split())
    locators = [item.external_id, item.title, item.source_url]
    exact_matches = [
        value
        for value in locators
        if value and " ".join(str(value).casefold().split()) in normalized_question
    ]
    evidence_tokens = {
        token
        for token in re.findall(
            r"[a-z0-9][a-z0-9_.-]*|[\u4e00-\u9fff]{2,}",
            " ".join(str(value or "") for value in locators).casefold(),
        )
        if len(token) > 1
    }
    question_tokens = set(
        re.findall(r"[a-z0-9][a-z0-9_.-]*|[\u4e00-\u9fff]{2,}", normalized_question)
    )
    return (
        sum(len(str(value)) for value in exact_matches),
        len(evidence_tokens & question_tokens),
        float(item.score or 0.0),
    )


def _answer_call_metadata(request: LLMRequest, prompt_hash: str) -> dict[str, Any]:
    metadata = {
        "evidence_ids": [item.source_id for item in request.evidence],
        "resource_ids": [item.external_id or str(item.document_id or "") for item in request.evidence],
        "source_urls": [item.source_url for item in request.evidence if item.source_url],
        "content_hashes": [item.content_hash for item in request.evidence],
        "answer_prompt_sha256": prompt_hash,
    }
    paper_contract = request.metadata.get("paper_contract")
    if isinstance(paper_contract, dict):
        metadata["paper_result"] = {
            key: paper_contract.get(key)
            for key in (
                "version",
                "concept_profile_version",
                "selected_source_ids",
                "required_fields",
                "target_count",
                "requested_count",
                "related_count",
                "field_complete_count",
                "selected_count",
                "completion_reason",
                "shortfall_reason",
            )
        }
    return metadata


def _safe_validation_error_details(exc: Exception) -> list[dict[str, Any]]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(chain) < 6 and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    root_type = chain[-1].__class__.__name__ if chain else exc.__class__.__name__
    for item in chain:
        errors_method = getattr(item, "errors", None)
        if not callable(errors_method):
            continue
        try:
            raw_errors = errors_method()
        except Exception:
            continue
        if not isinstance(raw_errors, list):
            continue
        safe: list[dict[str, Any]] = []
        for raw in raw_errors[:8]:
            if not isinstance(raw, dict):
                continue
            loc = raw.get("loc")
            safe_loc = [str(value)[:80] for value in loc[:8]] if isinstance(loc, (list, tuple)) else []
            safe.append(
                {
                    "loc": safe_loc,
                    "type": str(raw.get("type") or "")[:100],
                    "root_error_type": root_type[:100],
                }
            )
        if safe:
            return safe
    return []


def _safe_grounded_failure_detail(detail: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(detail, dict):
        return None
    source_id = str(detail.get("source_id") or "")
    span_id = str(detail.get("span_id") or "")
    if not re.fullmatch(r"S\d+", source_id) or not re.fullmatch(
        r"S\d+:(?:text|title|external_id):\d+", span_id
    ):
        return None
    try:
        ordinal = max(0, min(1000, int(detail.get("ordinal") or 0)))
    except (TypeError, ValueError):
        ordinal = 0
    return {
        "error_code": str(detail.get("error_code") or "GROUNDING_FAILED")[:64],
        "ordinal": ordinal,
        "source_id": source_id,
        "span_id": span_id,
        "claim_sha256": str(detail.get("claim_sha256") or "")[:64],
        "normalized_claim_sha256": str(detail.get("normalized_claim_sha256") or "")[:64],
        "claim_length": max(0, min(100000, int(detail.get("claim_length") or 0))),
    }


def _safe_paper_failure_detail(detail: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(detail, dict):
        return None
    source_id = str(detail.get("source_id") or "")
    if not re.fullmatch(r"S\d+", source_id):
        return None
    allowed_fields = {"title", "authors", "abstract_clue", "link", "paper"}
    raw_fields = detail.get("missing_fields")
    if not isinstance(raw_fields, list):
        return None
    missing_fields = sorted(
        {
            str(value)
            for value in raw_fields[:8]
            if isinstance(value, str) and str(value) in allowed_fields
        }
    )
    return {"source_id": source_id, "missing_fields": missing_fields}


def _grounded_rejection_trace_summary(
    error_code: str | None,
    failure_detail: dict[str, Any] | None,
) -> dict[str, Any]:
    safe = _safe_grounded_failure_detail(failure_detail)
    if safe:
        return safe
    return {"error_code": str(error_code or "GROUNDING_FAILED")[:64]}


def _response_delivery_mode(state: AgentState) -> str:
    modes = {item.delivery_mode for item in state.evidence if item.delivery_mode}
    if not modes:
        return "none"
    if len(modes) == 1:
        return next(iter(modes))
    return "mixed"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _grounded_candidate_sha256(
    candidate: GroundedPracticeGuide | GroundedSpanAnswer | GroundedAnswer,
) -> str:
    return _sha256_text(candidate.model_dump_json(exclude_none=True, by_alias=False))


_PROGRESS_STATUS_LABELS = {
    "not_started": "尚未开始",
    "in_progress": "进行中",
    "stuck": "需要帮助",
    "completed": "已完成",
}
_BLOCKER_STATUS_LABELS = {
    "active": "待处理",
    "confirmed": "已确认",
}


def _safe_learning_context_text(value: Any, safety_guard: SafetyGuard, max_chars: int) -> str:
    text = safety_guard.sanitize_untrusted_text(str(value or ""))
    text = " ".join(text.split()).strip()
    text = CITATION_RE.sub("", text)
    text = re.sub(r"(?i)\|\s*evidence\s+quote\s*:", "", text)
    text = text.replace("#", "").replace("|", "｜")
    return " ".join(text.split()).strip()[:max_chars]


def _safe_learning_status(value: Any, labels: dict[str, str]) -> str:
    if not isinstance(value, str):
        return "状态待核实"
    return labels.get(value.strip().casefold(), "状态待核实")


def _render_verified_learning_context(
    context: Any,
    safety_guard: SafetyGuard,
) -> str:
    if not isinstance(context, dict) or not context:
        return ""
    assignment_title = _safe_learning_context_text(
        context.get("assignment_title"), safety_guard, 160
    )
    step_title = _safe_learning_context_text(context.get("step_title"), safety_guard, 160)
    progress_note = _safe_learning_context_text(
        context.get("progress_note"), safety_guard, 500
    )
    blocker_reason = _safe_learning_context_text(
        context.get("blocker_reason"), safety_guard, 500
    )
    lines = ["## 当前学习状态"]
    if assignment_title:
        lines.append(f"- 当前任务：{assignment_title}")
    if step_title:
        try:
            step_index = int(context.get("step_index"))
        except (TypeError, ValueError, OverflowError):
            step_index = 0
        prefix = f"第{step_index}步 " if step_index > 0 else ""
        lines.append(f"- 当前步骤：{prefix}{step_title}")
    progress_percent = context.get("progress_percent")
    progress_parts: list[str] = []
    if type(progress_percent) is int and 0 <= progress_percent <= 100:
        progress_parts.append(f"{progress_percent}%")
    progress_parts.append(
        _safe_learning_status(context.get("progress_status"), _PROGRESS_STATUS_LABELS)
    )
    lines.append(f"- 本人进度：{' · '.join(progress_parts)}")
    if progress_note:
        lines.append(f"- 进度备注：{progress_note}")
    if blocker_reason or context.get("blocker_status") not in (None, ""):
        blocker_label = _safe_learning_status(
            context.get("blocker_status"), _BLOCKER_STATUS_LABELS
        )
        suffix = f" · {blocker_reason}" if blocker_reason else ""
        lines.append(f"- 当前卡点：{blocker_label}{suffix}")
    return "\n".join(lines)


def _format_tool_answer(tool_name: str, result: dict[str, Any], fallback_used: bool) -> str:
    source_label = "local_fallback" if fallback_used else "mcp"
    if not result.get("success"):
        return f"Tool call failed via {source_label}: {result.get('error_code') or 'TOOL_FAILED'}"
    data = result.get("data") or {}
    if tool_name == "mermaid_generator":
        diagram = data.get("mermaid") or data.get("diagram") or ""
        return f"Tool source: {source_label}\n\n```mermaid\n{diagram}\n```"
    return f"Tool source: {source_label}\n\n{data}"


