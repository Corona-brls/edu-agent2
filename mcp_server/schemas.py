"""Schemas for MCP tool definitions, inputs and results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    agent_description: str
    input_schema: dict[str, Any]
    risk: RiskLevel = "LOW"
    timeout_seconds: float = 5
    network: bool = False
    read_only: bool = True
    enabled: bool = True
    roles: list[str] = Field(default_factory=lambda: ["student", "teacher", "admin"])


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class CourseRetrieverInput(SearchInput):
    course_id: int = Field(..., ge=1)


class PaperSearchInput(SearchInput):
    pass


class VulnLookupInput(SearchInput):
    pass


class StandardLookupInput(SearchInput):
    pass


class ConnectorSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=300)
    top_k: int = Field(default=3, ge=1, le=10)


class RealPaperSearchInput(ConnectorSearchInput):
    source: Literal["semantic_scholar", "arxiv", "auto"] = "auto"


class RealVulnLookupInput(ConnectorSearchInput):
    pass


class StandardMetadataLookupInput(ConnectorSearchInput):
    pass


class RealStandardLookupInput(ConnectorSearchInput):
    pass


class GitHubProjectSearchInput(ConnectorSearchInput):
    pass


class MermaidGeneratorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=1, max_length=200)
    diagram_type: Literal["flowchart", "sequence", "mindmap"] = "flowchart"
