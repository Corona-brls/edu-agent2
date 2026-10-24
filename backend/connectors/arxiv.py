"""arXiv API connector."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from backend.connectors.base import ExternalConnector
from backend.connectors.http_utils import map_http_error
from backend.connectors.schemas import ConnectorHealth, ExternalResource, ResourceQuery
from backend.core.config import get_settings
from backend.resources.query_intent import matches_exact_identity, parse_lookup_target

ATOM = "{http://www.w3.org/2005/Atom}"


class ArxivConnector(ExternalConnector):
    name = "arxiv"

    def __init__(self, base_url: str = "https://export.arxiv.org/api/query") -> None:
        self.enabled = get_settings().enable_arxiv
        self.base_url = base_url

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, enabled=self.enabled, status="ok" if self.enabled else "disabled")

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        target = parse_lookup_target(query.query, "paper")
        if target.mode == "EXACT" and target.kind == "arxiv":
            params = {"id_list": target.identifier, "start": 0, "max_results": 1}
        else:
            params = {
                "search_query": f"all:{query.query}",
                "start": (query.page - 1) * query.page_size,
                "max_results": query.page_size,
            }
        async with httpx.AsyncClient(timeout=get_settings().external_http_timeout_seconds) as client:
            response = await client.get(self.base_url, params=params)
        map_http_error(response)
        root = ET.fromstring(response.text)
        resources: list[ExternalResource] = []
        seen: set[str] = set()
        for entry in root.findall(f"{ATOM}entry"):
            id_text = (entry.findtext(f"{ATOM}id") or "").strip()
            arxiv_id = _arxiv_id_from_entry_url(id_text)
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            title = " ".join((entry.findtext(f"{ATOM}title") or "Untitled arXiv paper").split())
            abstract = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
            authors = [
                (author.findtext(f"{ATOM}name") or "").strip()
                for author in entry.findall(f"{ATOM}author")
                if (author.findtext(f"{ATOM}name") or "").strip()
            ]
            published = entry.findtext(f"{ATOM}published")
            source_url = id_text.replace("http://arxiv.org/", "https://arxiv.org/", 1)
            resource = ExternalResource(
                    external_id=arxiv_id,
                    resource_type="paper",
                    source_name=self.name,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=source_url,
                    published_at=published[:4] if published else None,
                    tags=["arXiv"],
                    module=query.module,
                    raw_metadata={"arxiv_id": arxiv_id, "published": published},
                    rights_note="arXiv public metadata and abstract.",
                )
            if target.mode == "EXACT" and not matches_exact_identity(
                target, self.name, resource.model_dump(mode="json")
            ):
                continue
            resources.append(resource)
        return resources


def _arxiv_id_from_entry_url(value: str) -> str:
    marker = "/abs/"
    if marker not in value:
        return ""
    return value.split(marker, 1)[1].split("?", 1)[0].split("#", 1)[0].strip("/")
