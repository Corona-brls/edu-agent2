"""Strict request, model-candidate, and response schemas for practice drafts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DraftSourceQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_type: Literal["paper", "vulnerability", "project", "standard"]
    query: str = Field(..., min_length=1, max_length=300)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class PracticeDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=1, max_length=200)
    objective: str = Field(..., min_length=1, max_length=2000)
    lab_environment: str = Field(..., min_length=1, max_length=1500)
    available_tools: list[str] = Field(default_factory=list, max_length=10)
    course_id: int | None = Field(default=None, gt=0)
    class_id: int | None = Field(default=None, gt=0)
    client_session_id: str | None = Field(default=None, min_length=1, max_length=128)
    use_course_references: bool = False
    external_queries: list[DraftSourceQuery] = Field(default_factory=list, max_length=2)

    @field_validator("topic", "objective", "lab_environment")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must not be blank")
        return normalized

    @field_validator("available_tools")
    @classmethod
    def _tools_are_bounded_and_unique(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value).strip()
            if not item or len(item) > 100:
                raise ValueError("available_tools items must contain 1-100 characters")
            key = item.casefold()
            if key in seen:
                raise ValueError("available_tools must not contain duplicates")
            seen.add(key)
            normalized.append(item)
        return normalized

    @field_validator("external_queries")
    @classmethod
    def _queries_are_unique(cls, values: list[DraftSourceQuery]) -> list[DraftSourceQuery]:
        seen: set[tuple[str, str]] = set()
        for item in values:
            key = (item.resource_type, item.query.casefold())
            if key in seen:
                raise ValueError("external_queries must not contain duplicates")
            seen.add(key)
        return values


class DraftSpanSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1, max_length=32, pattern=r"^S[1-9][0-9]*$")
    evidence_span_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^S[1-9][0-9]*:text:[1-9][0-9]*$",
    )


class ProposedStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(..., min_length=1, max_length=600)
    expected_observation: str = Field(..., min_length=1, max_length=600)
    check: str = Field(..., min_length=1, max_length=600)

    @field_validator("action", "expected_observation", "check")
    @classmethod
    def _step_text_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("step fields must not be blank")
        return normalized


class ProposedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(..., min_length=1, max_length=2000)
    lab_environment: str = Field(..., min_length=1, max_length=1500)
    baseline_and_control: str = Field(..., min_length=1, max_length=2000)
    variables: list[str] = Field(default_factory=list, max_length=12)
    steps: list[ProposedStep] = Field(..., min_length=3, max_length=8)
    evaluation_metrics: list[str] = Field(default_factory=list, max_length=12)
    expected_observations: list[str] = Field(default_factory=list, max_length=12)
    risks: list[str] = Field(default_factory=list, max_length=12)
    unknowns: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("objective", "lab_environment", "baseline_and_control")
    @classmethod
    def _plan_text_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("plan fields must not be blank")
        return normalized

    @field_validator(
        "variables",
        "evaluation_metrics",
        "expected_observations",
        "risks",
        "unknowns",
    )
    @classmethod
    def _bounded_list_items(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            item = str(value).strip()
            if not item or len(item) > 600:
                raise ValueError("plan list items must contain 1-600 characters")
            result.append(item)
        return result


class PracticeDraftCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["proposed", "refused"]
    selected_spans: list[DraftSpanSelection] = Field(default_factory=list, max_length=8)
    proposed_plan: ProposedPlan | None = None
    refusal_reason: str | None = Field(default=None, max_length=1200)

    @model_validator(mode="after")
    def _status_fields_are_consistent(self) -> "PracticeDraftCandidate":
        if self.status == "proposed":
            if self.proposed_plan is None or self.refusal_reason is not None:
                raise ValueError("proposed candidates require proposed_plan and no refusal_reason")
        else:
            if self.proposed_plan is not None or self.selected_spans:
                raise ValueError("refused candidates cannot include a plan or selected spans")
            if not str(self.refusal_reason or "").strip():
                raise ValueError("refused candidates require refusal_reason")
        return self


class DraftBackgroundQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1, max_length=32)
    evidence_span_id: str = Field(..., min_length=1, max_length=64)
    quote_text: str = Field(..., min_length=1, max_length=360)
    source_title: str = Field(..., min_length=1, max_length=300)
    source_url: str | None = Field(default=None, max_length=2000)
    delivery_mode: str = Field(..., min_length=1, max_length=60)
    fetched_at: str | None = Field(default=None, max_length=80)


class DraftReferenceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., min_length=1, max_length=100)
    query: str | None = Field(default=None, max_length=300)
    lookup_state: str = Field(..., min_length=1, max_length=40)
    delivery_mode: str = Field(..., min_length=1, max_length=60)
    error_code: str | None = Field(default=None, max_length=120)


class PracticeDraftResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=1, max_length=128)
    course_id: int
    class_id: int | None = None
    status: Literal["grounded_draft", "proposal_only", "refused"]
    background_quotes: list[DraftBackgroundQuote] = Field(default_factory=list, max_length=8)
    proposed_plan: ProposedPlan | None = None
    refusal_reason: str | None = Field(default=None, max_length=1200)
    proposal_label: Literal["模型提出的待验证教学草案"] = "模型提出的待验证教学草案"
    reference_statuses: list[DraftReferenceStatus] = Field(default_factory=list, max_length=3)
    provider: Literal["deepseek"] = "deepseek"
    model: str = Field(..., min_length=1, max_length=200)
    fallback_used: Literal[False] = False
    warnings: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("warnings")
    @classmethod
    def _warnings_are_bounded(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            item = str(value).strip()
            if not item or len(item) > 600:
                raise ValueError("warnings items must contain 1-600 characters")
            result.append(item)
        return result
