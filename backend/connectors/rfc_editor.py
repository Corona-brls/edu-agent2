"""RFC Editor official metadata connector for exact RFC identifiers."""

from __future__ import annotations

import httpx

from backend.connectors.base import ExternalConnector
from backend.connectors.http_utils import map_http_error
from backend.connectors.schemas import ConnectorHealth, ExternalResource, ResourceQuery
from backend.core.config import get_settings
from backend.resources.query_intent import matches_exact_identity, parse_lookup_target


class RFCEditorConnector(ExternalConnector):
    name = "rfc_editor"

    def __init__(self, base_url: str = "https://www.rfc-editor.org/rfc") -> None:
        self.enabled = get_settings().enable_real_rfc_lookup
        self.base_url = base_url.rstrip("/")

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, enabled=self.enabled, status="ok" if self.enabled else "disabled")

    @staticmethod
    def _lookup_target(query: ResourceQuery):
        return parse_lookup_target(query.query, "standard")

    @staticmethod
    def _is_remote_target(target) -> bool:
        return target.mode == "EXACT" and target.kind == "rfc" and bool(target.identifier)

    def will_attempt_remote(self, query: ResourceQuery) -> bool:
        return self._is_remote_target(self._lookup_target(query))

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        target = self._lookup_target(query)
        if not self._is_remote_target(target):
            return []
        number = target.identifier.removeprefix("RFC-")
        async with httpx.AsyncClient(timeout=get_settings().external_http_timeout_seconds) as client:
            response = await client.get(f"{self.base_url}/rfc{number}.json")
        if response.status_code == 404:
            return []
        map_http_error(response)
        payload = response.json()
        doc_id = str(payload.get("doc_id") or "").upper().replace("-", "")
        if doc_id != f"RFC{number}":
            return []
        abstract = payload.get("abstract")
        abstract_text = abstract.strip() if isinstance(abstract, str) else ""
        resource = ExternalResource(
            external_id=f"RFC-{number}",
            resource_type="standard",
            source_name=self.name,
            title=str(payload.get("title") or f"RFC {number}"),
            abstract=abstract_text,
            url=f"https://www.rfc-editor.org/info/rfc{number}",
            published_at=str(payload.get("pub_date") or "") or None,
            tags=[str(payload.get("pub_status") or payload.get("status") or "")] if payload.get("pub_status") or payload.get("status") else [],
            module=query.module,
            raw_metadata={
                "doc_id": payload.get("doc_id"),
                "status": payload.get("status"),
                "pub_status": payload.get("pub_status"),
                "format": payload.get("format"),
                "doi": payload.get("doi"),
                "abstract_available": bool(abstract_text),
            },
            rights_note="RFC Editor official public metadata; full text is not cached by this connector.",
        )
        if not matches_exact_identity(target, self.name, resource.model_dump(mode="json")):
            return []
        return [resource]
