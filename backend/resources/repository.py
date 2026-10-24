"""Repository wrapper around LearningLogger for external resources."""

from __future__ import annotations

from typing import Any

from backend.connectors.schemas import ExternalResource, ResourceQuery
from backend.storage.logger import LearningLogger
from backend.resources.query_intent import LookupTarget


class ExternalResourceRepository:
    def __init__(self, storage: LearningLogger) -> None:
        self.storage = storage

    def upsert(self, resource: ExternalResource, *, ttl_hours: int, content_hash: str) -> dict[str, Any]:
        return self.storage.upsert_external_resource(
            resource=resource.model_dump(mode="json"),
            ttl_hours=ttl_hours,
            content_hash=content_hash,
        )

    def search_cache(
        self,
        query: ResourceQuery,
        *,
        include_expired: bool = False,
        source_name: str | None = None,
        lookup_target: LookupTarget | None = None,
    ) -> list[dict[str, Any]]:
        return self.storage.search_external_resources(
            _cache_query(query, lookup_target),
            include_expired=include_expired,
            source_name=source_name,
            exact_only=bool(lookup_target and lookup_target.mode == "EXACT"),
        )

    def fresh_cache(
        self,
        query: ResourceQuery,
        *,
        source_name: str | None = None,
        lookup_target: LookupTarget | None = None,
    ) -> list[dict[str, Any]]:
        return self.search_cache(
            query,
            include_expired=False,
            source_name=source_name,
            lookup_target=lookup_target,
        )

    def store_query_cache(
        self,
        query: ResourceQuery,
        *,
        source_name: str,
        resources: list[dict[str, Any]],
        ttl_hours: int,
        lookup_target: LookupTarget | None = None,
        incomplete_results: bool | None = None,
    ) -> None:
        self.storage.store_resource_query_cache(
            _cache_query(query, lookup_target, incomplete_results=incomplete_results),
            source_name=source_name,
            resource_ids=[int(item["id"]) for item in resources],
            ttl_hours=ttl_hours,
        )

    def sync_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.storage.connector_sync_runs(limit=limit)


def cache_query_payload(
    query: ResourceQuery,
    lookup_target: LookupTarget | None,
    *,
    incomplete_results: bool | None = None,
) -> dict[str, Any]:
    return _cache_query(query, lookup_target, incomplete_results=incomplete_results)


def _cache_query(
    query: ResourceQuery,
    lookup_target: LookupTarget | None,
    *,
    incomplete_results: bool | None = None,
) -> dict[str, Any]:
    payload = query.model_dump(mode="json")
    target = lookup_target or LookupTarget(mode="SEARCH")
    payload["lookup_mode"] = target.mode
    payload["canonical_identifier"] = target.identifier
    if target.mode == "EXACT":
        payload["query"] = target.identifier or payload["query"]
        payload["page"] = 1
        payload["page_size"] = 1
    if incomplete_results is not None:
        payload["incomplete_results"] = incomplete_results
    return payload
