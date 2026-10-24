"""Public and internal schemas for general learning questions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GeneralChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=4000)
    course_id: int | None = Field(default=None, gt=0)
    class_id: int | None = Field(default=None, gt=0)
    use_course_references: bool = False
    client_session_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class GeneralModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "refused"]
    answer: str = Field(..., min_length=1, max_length=12000)


class GeneralReferenceMaterial(BaseModel):
    document_id: int
    chunk_id: int
    title: str
    excerpt: str
    usage: Literal["context_only"] = "context_only"


class GeneralChatResponse(BaseModel):
    request_id: str
    course_id: int
    class_id: int | None = None
    status: Literal["answered", "refused"]
    answer_kind: Literal["model_generated"] = "model_generated"
    answer: str
    provider: Literal["deepseek"] = "deepseek"
    model: str
    fallback_used: Literal[False] = False
    reference_status: Literal["used", "none", "disabled", "blocked_by_policy", "unavailable"]
    reference_materials: list[GeneralReferenceMaterial] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

