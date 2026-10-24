"""Strict request, candidate, and response schemas for innovation suggestions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class InnovationSuggestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_session_id: str | None = Field(default=None, min_length=1, max_length=128)
    use_course_references: bool = False


class InnovationSuggestionItemCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(..., min_length=1, max_length=1200)
    comparison: str = Field(..., min_length=1, max_length=1200)
    metrics: list[str] = Field(default_factory=list, max_length=8)
    next_question: str = Field(..., min_length=1, max_length=600)
    risks: list[str] = Field(default_factory=list, max_length=8)
    background_context_ids: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("hypothesis", "comparison", "next_question")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("suggestion text must not be blank")
        return value

    @field_validator("metrics", "risks")
    @classmethod
    def _list_items_are_bounded(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            item = str(value).strip()
            if not item or len(item) > 600:
                raise ValueError("list items must contain 1-600 characters")
            result.append(item)
        return result


class InnovationSuggestionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["proposed", "refused"]
    suggestions: list[InnovationSuggestionItemCandidate] = Field(default_factory=list, max_length=3)
    refusal_reason: str | None = Field(default=None, max_length=1200)

    @model_validator(mode="after")
    def _status_fields_are_consistent(self) -> "InnovationSuggestionCandidate":
        if self.status == "proposed":
            if not self.suggestions or self.refusal_reason is not None:
                raise ValueError("proposed candidates require suggestions and no refusal_reason")
        elif self.suggestions or not str(self.refusal_reason or "").strip():
            raise ValueError("refused candidates require only refusal_reason")
        return self


class InnovationSuggestionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str
    comparison: str
    metrics: list[str] = Field(default_factory=list)
    next_question: str
    risks: list[str] = Field(default_factory=list)
    background_context_ids: list[str] = Field(default_factory=list)
    background_source_ids: list[str] = Field(default_factory=list)


class InnovationSourceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    status: str
    source_type: str
    delivery_mode: str
    reason: str | None = None


class InnovationSuggestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    session_id: int
    course_id: int
    status: Literal["proposed", "refused"]
    suggestions: list[InnovationSuggestionItem] = Field(default_factory=list)
    refusal_reason: str | None = None
    generation_mode: Literal["deepseek"] = "deepseek"
    hypotheses_verified: Literal[False] = False
    proposal_label: Literal["模型提出的待验证创新建议"] = "模型提出的待验证创新建议"
    source_statuses: list[InnovationSourceStatus] = Field(default_factory=list)
    course_reference_status: Literal["allowed", "disabled_by_user", "blocked_by_policy", "revoked"]
    warnings: list[str] = Field(default_factory=list)
    collection_status: Literal["recorded", "not_recorded"] = "not_recorded"
    background_source_ids: list[str] = Field(default_factory=list)
    provider: Literal["deepseek"] = "deepseek"
    model: str
    fallback_used: Literal[False] = False
