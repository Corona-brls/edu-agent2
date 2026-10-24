"""Optional GitHub repository search connector."""

from __future__ import annotations

import httpx

from backend.connectors.base import ConnectorError, ConnectorServiceError, ExternalConnector
from backend.connectors.http_utils import map_http_error
from backend.connectors.schemas import ConnectorHealth, ConnectorSearchBatch, ExternalResource, ResourceQuery
from backend.core.config import get_settings
from backend.resources.query_intent import matches_exact_identity, parse_lookup_target

_GITHUB_SEARCH_MAX_LENGTH = 256


class GitHubConnector(ExternalConnector):
    name = "github"

    def __init__(self, base_url: str = "https://api.github.com/search/repositories") -> None:
        settings = get_settings()
        self.enabled = settings.enable_github_search
        self.token = settings.github_token
        self.base_url = base_url

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, enabled=self.enabled, status="ok" if self.enabled else "disabled")

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        return (await self.search_with_metadata(query)).resources

    async def search_with_metadata(self, query: ResourceQuery) -> ConnectorSearchBatch:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        target = parse_lookup_target(query.query, "project")
        if target.mode == "EXACT" and target.kind == "github" and target.identifier:
            api_root = self.base_url.removesuffix("/search/repositories").rstrip("/")
            url = f"{api_root}/repos/{target.identifier}"
            params = None
        else:
            normalized_query = " ".join(query.query.split())
            if len(normalized_query) > _GITHUB_SEARCH_MAX_LENGTH:
                raise ConnectorError(
                    "GitHub 仓库搜索词不能超过256个字符。",
                    error_code="GITHUB_QUERY_TOO_LONG",
                )
            url = self.base_url
            params = {"q": normalized_query, "page": query.page, "per_page": query.page_size}
        async with httpx.AsyncClient(timeout=get_settings().external_http_timeout_seconds) as client:
            response = await client.get(url, params=params, headers=headers)
        if response.status_code == 404 and target.mode == "EXACT":
            return ConnectorSearchBatch(resources=[])
        map_http_error(response)
        payload = response.json()
        if target.mode == "EXACT":
            items = [payload]
            incomplete_results = None
        else:
            incomplete_results = payload.get("incomplete_results")
            if not isinstance(incomplete_results, bool):
                raise ConnectorServiceError(
                    "GitHub 搜索响应缺少有效的 incomplete_results 字段。",
                    error_code="CONNECTOR_RESPONSE_INVALID",
                )
            items = payload.get("items", [])
        resources: list[ExternalResource] = []
        for item in items:
            if (
                item.get("id") is None
                or not str(item.get("full_name") or "").strip()
                or item.get("private") is not False
                or str(item.get("visibility") or "public").lower() in {"private", "internal"}
            ):
                continue
            resource = ExternalResource(
                    external_id=str(item.get("id")),
                    resource_type="project",
                    source_name=self.name,
                    title=item.get("full_name") or item.get("name") or "GitHub repository",
                    abstract=item.get("description") or "",
                    url=item.get("html_url") or "",
                    published_at=item.get("updated_at"),
                    tags=[item.get("language", "")] if item.get("language") else [],
                    module=query.module,
                    raw_metadata={
                        "stars": item.get("stargazers_count"),
                        "updated_at": item.get("updated_at"),
                        "full_name": item.get("full_name"),
                        "private": item.get("private"),
                        "visibility": item.get("visibility") or "public",
                    },
                    rights_note="GitHub public repository metadata; code is not executed.",
                )
            if target.mode == "EXACT" and not matches_exact_identity(
                target, self.name, resource.model_dump(mode="json")
            ):
                continue
            resources.append(resource)
        return ConnectorSearchBatch(resources=resources, incomplete_results=incomplete_results)
