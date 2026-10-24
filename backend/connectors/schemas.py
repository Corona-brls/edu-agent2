from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ResourceType = Literal["paper", "standard", "vulnerability", "project", "competition"]


class ResourceQuery(BaseModel):
    query: str = Field(default="", max_length=300)
    resource_type: ResourceType | None = None
    module: str = ""
    year_from: int | None = None
    year_to: int | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=50)
    allow_remote: bool = True


class ExternalResource(BaseModel):
    external_id: str
    resource_type: ResourceType
    source_name: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    url: str = ""
    published_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    module: str = ""
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    rights_note: str = ""


class ConnectorHealth(BaseModel):
    name: str
    enabled: bool = True
    status: str
    detail: str = ""
    latency_ms: float = 0.0


class ConnectorSearchBatch(BaseModel):
    resources: list[ExternalResource] = Field(default_factory=list)
    incomplete_results: bool | None = None


class ConnectorSyncResult(BaseModel):
    connector_name: str
    status: str
    result_count: int = 0
    cache_hit: bool = False
    error_code: str | None = None
    error_message: str | None = None
    resources: list[ExternalResource] = Field(default_factory=list)
    incomplete_results: bool | None = None


class SourceAttempt(BaseModel):
    source_name: str
    status: Literal["FOUND", "EMPTY", "PARTIAL", "ERROR", "DISABLED", "UNSUPPORTED", "CACHE_MISS"]
    result_count: int = 0
    delivery_mode: str
    error_code: str | None = None
    fetched_at: str | None = None
    incomplete_results: bool | None = None


class ResourceSearchResponse(BaseModel):
    resources: list[dict[str, Any]] = Field(default_factory=list)
    status: str
    lookup_mode: Literal["SEARCH", "EXACT"]
    canonical_identifier: str | None = None
    lookup_state: Literal["FOUND", "PARTIAL", "NOT_FOUND", "UNAVAILABLE", "UNSUPPORTED", "NOT_CHECKED"]
    source_attempts: list[SourceAttempt] = Field(default_factory=list)
    incomplete_results: bool | None = None

    model_config = {"extra": "allow"}

