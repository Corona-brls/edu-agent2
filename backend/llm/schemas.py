"""Pydantic schemas for LLM requests and structured outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    source: str = ""
    title: str = ""
    document_id: int | None = None
    chunk_id: int | None = None
    page: int | None = None
    section: str = ""
    source_type: str = ""
    score: float | None = None
    source_url: str = ""
    external_id: str = ""
    resource_type: str = ""
    origin_kind: str = ""
    delivery_mode: str = ""
    fetched_at: str = ""
    content_hash: str = ""
    tool_name: str = ""
    trusted: bool = True


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    instructions: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    provider: str
    model: str
    fallback_used: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)


class GroundedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_text: str = Field(..., min_length=1, max_length=360)
    source_id: str = Field(..., pattern=r"^S\d+$")
    evidence_quote: str = Field(..., min_length=1, max_length=600)


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[GroundedClaim] = Field(..., min_length=1, max_length=10)


class EvidenceSpan(BaseModel):
    """Internal request-scoped quote anchor derived from EvidenceItem."""

    model_config = ConfigDict(extra="forbid")

    span_id: str = Field(..., pattern=r"^S\d+:(?:text|title|external_id):\d+$")
    source_id: str = Field(..., pattern=r"^S\d+$")
    field: Literal["text", "title", "external_id"]
    exact_text: str = Field(..., min_length=1, max_length=1200)
    resource_type: str = ""


class GroundedSpanClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_text: str = Field(..., min_length=1, max_length=360)
    source_id: str = Field(..., pattern=r"^S\d+$")
    evidence_span_id: str = Field(..., pattern=r"^S\d+:(?:text|title|external_id):\d+$")


class GroundedSpanAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[GroundedSpanClaim] = Field(..., min_length=1, max_length=10)


class GroundedPracticeGuide(BaseModel):
    """Internal, fully evidence-bound practice guide contract."""

    model_config = ConfigDict(extra="forbid")

    research_goal: list[GroundedSpanClaim] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Evidence-backed research goals; every item must bind to one supplied span.",
    )
    method_design: list[GroundedSpanClaim] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Evidence-backed method design; every item must bind to one supplied span.",
    )
    tools: list[GroundedSpanClaim] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Tools explicitly named by the evidence; no proposed or inferred tools.",
    )
    experiment_steps: list[GroundedSpanClaim] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Ordered experiment steps explicitly stated by the evidence.",
    )
    result_analysis: list[GroundedSpanClaim] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Evidence-backed result analysis boundaries.",
    )
    innovation_points: list[GroundedSpanClaim] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Innovation points explicitly stated by the evidence.",
    )
    risks: list[GroundedSpanClaim] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Risks and cautions explicitly stated by the evidence.",
    )


class PracticeGuide(BaseModel):
    model_config = ConfigDict(extra="forbid")

    research_goal: str
    method_design: str
    tools: list[str]
    experiment_steps: list[str]
    result_analysis: str
    innovation_points: list[str]
    risks: list[str]
