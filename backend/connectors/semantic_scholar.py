"""Semantic Scholar Graph API connector."""

from __future__ import annotations

import time
from urllib.parse import quote

import httpx

from backend.connectors.base import ExternalConnector
from backend.connectors.http_utils import map_http_error
from backend.connectors.schemas import ConnectorHealth, ExternalResource, ResourceQuery
from backend.core.config import get_settings
from backend.resources.query_intent import matches_exact_identity, parse_lookup_target


class SemanticScholarConnector(ExternalConnector):
    name = "semantic_scholar"

    def __init__(self, base_url: str = "https://api.semanticscholar.org/graph/v1") -> None:
        settings = get_settings()
        self.enabled = settings.enable_semantic_scholar
        self.api_key = settings.semantic_scholar_api_key
        self.base_url = base_url.rstrip("/")

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, enabled=self.enabled, status="ok" if self.enabled else "disabled")

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        target = parse_lookup_target(query.query, "paper")
        fields = "title,authors,abstract,year,url,externalIds,venue"
        if target.mode == "EXACT" and target.kind in {"doi", "arxiv"}:
            prefix = "DOI" if target.kind == "doi" else "ARXIV"
            paper_id = quote(f"{prefix}:{target.identifier}", safe=":/")
            url = f"{self.base_url}/paper/{paper_id}"
            params = {"fields": fields}
        else:
            url = f"{self.base_url}/paper/search"
            params = {
                "query": query.query,
                "limit": query.page_size,
                "offset": (query.page - 1) * query.page_size,
                "fields": fields,
            }
        async with httpx.AsyncClient(timeout=get_settings().external_http_timeout_seconds) as client:
            response = await client.get(url, params=params, headers=headers)
        if response.status_code == 404 and target.mode == "EXACT":
            return []
        map_http_error(response)
        payload = response.json()
        data = [payload] if target.mode == "EXACT" else payload.get("data", [])
        seen: set[str] = set()
        resources: list[ExternalResource] = []
        for item in data:
            ids = item.get("externalIds") or {}
            external_id = ids.get("DOI") or ids.get("ArXiv") or item.get("paperId")
            if not external_id or external_id in seen:
                continue
            seen.add(external_id)
            resource = ExternalResource(
                    external_id=str(external_id),
                    resource_type="paper",
                    source_name=self.name,
                    title=item.get("title") or "Untitled paper",
                    authors=[author.get("name", "") for author in item.get("authors", []) if author.get("name")],
                    abstract=item.get("abstract") or "",
                    url=item.get("url") or "",
                    published_at=str(item.get("year")) if item.get("year") else None,
                    tags=[item.get("venue", "")] if item.get("venue") else [],
                    module=query.module,
                    raw_metadata={"externalIds": ids, "year": item.get("year")},
                    rights_note="Semantic Scholar metadata; abstract only when provided by API.",
                )
            if target.mode == "EXACT" and not matches_exact_identity(
                target, self.name, resource.model_dump(mode="json")
            ):
                continue
            resources.append(resource)
        return resources

