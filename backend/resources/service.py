"""External resource search/sync service."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any, Callable

import httpx

from backend.connectors.base import (
    ConnectorError,
    ConnectorRateLimitError,
    ConnectorServiceError,
    ConnectorTimeoutError,
)
from backend.connectors.registry import ConnectorRegistry
from backend.connectors.schemas import ConnectorSearchBatch, ConnectorSyncResult, ExternalResource, ResourceQuery
from backend.connectors.standard_source import normalize_standard_resource_view
from backend.core.config import Settings, get_settings
from backend.rag.vector_store import ChromaVectorStore
from backend.resources.normalizer import content_hash, resource_content
from backend.resources.query_intent import (
    LookupTarget,
    is_deliverable_exact_arxiv_resource,
    matches_exact_identity,
    parse_lookup_target,
    validate_connector_query,
    validate_lookup_filters,
)
from backend.resources.repository import ExternalResourceRepository, cache_query_payload
from backend.resources.trust import normalized_trust_policy, origin_for_source
from backend.storage.logger import LearningLogger


class ExternalResourceService:
    def __init__(
        self,
        *,
        storage: LearningLogger,
        registry: ConnectorRegistry,
        settings: Settings | None = None,
        vector_store: Any | None = None,
        trusted_sources: Mapping[str, str] | None = None,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.settings = settings or get_settings()
        self.repository = ExternalResourceRepository(storage)
        self.vector_store = vector_store
        self.trusted_sources = normalized_trust_policy(trusted_sources)

    async def connector_status(self) -> list[dict]:
        return await self.registry.statuses()

    async def search(
        self,
        query: ResourceQuery,
        connector_name: str | None = None,
        *,
        deadline: float | None = None,
        before_remote: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        target = parse_lookup_target(query.query, query.resource_type)
        validate_lookup_filters(
            target,
            page=query.page,
            year_from=query.year_from,
            year_to=query.year_to,
            module=query.module,
        )
        validate_connector_query(target, connector_name=connector_name, query=query.query)
        if connector_name and target.mode == "EXACT" and connector_name not in target.source_candidates:
            return self._unsupported_response(query, connector_name, target)
        connector_names = [connector_name] if connector_name else self._connector_candidates(query, target)
        if not query.allow_remote:
            cache_names = connector_names or self.registry.names()
            cache_attempts: list[dict[str, Any]] = []
            for cached_source in cache_names:
                cached_response = self._fresh_cache_response(
                    query,
                    target,
                    cached_source,
                    tried=[],
                    attempts=cache_attempts,
                    observed_error=False,
                )
                if cached_response is not None:
                    if cached_response.get("_continue_source_chain") is True:
                        cache_attempts.extend(cached_response["source_attempts"])
                        continue
                    return cached_response
            return {
                "cache_hit": False,
                "resources": [],
                "error": None,
                "connectors_tried": [],
                "connector": connector_name,
                "status": "CACHE_MISS",
                "delivery_mode": "cache_miss",
                "origin_kind": origin_for_source(connector_name, self.trusted_sources),
                "lookup_mode": target.mode,
                "canonical_identifier": target.identifier,
                "lookup_state": "NOT_CHECKED",
                "source_attempts": cache_attempts,
                "incomplete_results": None,
            }
        if not connector_names:
            return self._unsupported_response(query, connector_name, target)

        result: ConnectorSyncResult | None = None
        selected_resources: list[dict[str, Any]] = []
        stale_candidate: tuple[ConnectorSyncResult, list[dict[str, Any]]] | None = None
        tried: list[str] = []
        attempts: list[dict[str, Any]] = []
        observed_error = False
        for name in connector_names:
            cached_response = self._fresh_cache_response(
                query,
                target,
                name,
                tried=tried,
                attempts=attempts,
                observed_error=observed_error,
            )
            if cached_response is not None:
                if cached_response.get("_continue_source_chain") is True:
                    tried.append(name)
                    attempts.extend(cached_response["source_attempts"])
                    continue
                return cached_response
            tried.append(name)
            try:
                connector = self.registry.get(name)
            except KeyError:
                connector = None
            enters_remote = (
                connector is not None
                and connector.enabled
                and getattr(connector, "supports_remote", True)
                and origin_for_source(name, self.trusted_sources) == "remote"
                and connector.will_attempt_remote(query)
            )
            if before_remote is not None and enters_remote:
                try:
                    _ensure_before_deadline(deadline)
                except ConnectorTimeoutError:
                    pass
                else:
                    before_remote(name)
            result = await self.sync(name, query, deadline=deadline, lookup_target=target)
            attempt_status = _attempt_status(result)
            observed_error = observed_error or attempt_status in {"ERROR", "DISABLED"}
            attempts.append(
                {
                    "source_name": name,
                    "status": attempt_status,
                    "result_count": result.result_count,
                    "delivery_mode": "stale_cache" if result.status == "FAILED_WITH_CACHE" else "live",
                    "error_code": result.error_code,
                    "fetched_at": None,
                    "incomplete_results": result.incomplete_results,
                }
            )
            result_resources = self.repository.fresh_cache(
                query,
                source_name=name,
                lookup_target=target,
            )
            result_resources = self._deliverable_cached_resources(result_resources, target)
            if not result_resources and result.result_count > 0:
                result_resources = self.repository.search_cache(
                    query,
                    include_expired=True,
                    source_name=name,
                    lookup_target=target,
                )
                result_resources = self._deliverable_cached_resources(result_resources, target)
            if result.status in {"SUCCESS", "PARTIAL"} and (result_resources or result.incomplete_results is True):
                selected_resources = result_resources
                break
            if result.status == "FAILED_WITH_CACHE" and result_resources and stale_candidate is None:
                stale_candidate = (result, result_resources)
                if connector_name is not None:
                    selected_resources = result_resources
                    break
        if not selected_resources and stale_candidate is not None:
            result, selected_resources = stale_candidate
        resources = selected_resources
        delivery_mode = self._delivery_mode(result, resources)
        origin_kind = self._cached_origin(
            resources,
            result.connector_name if result else connector_name,
        )
        resources = self._present_resources(resources, delivery_mode, origin_kind)
        incomplete_results = result.incomplete_results if result else None
        if resources:
            lookup_state = "PARTIAL" if observed_error or delivery_mode == "stale_cache" or incomplete_results else "FOUND"
        elif incomplete_results is True:
            lookup_state = "PARTIAL"
        elif observed_error:
            lookup_state = "UNAVAILABLE"
        else:
            lookup_state = "NOT_FOUND"
        return {
            "cache_hit": bool(result.cache_hit) if result else False,
            "resources": resources,
            "error": None if result and result.status == "SUCCESS" else (result.error_code if result else "CONNECTOR_NOT_CONFIGURED"),
            "error_message": result.error_message if result else None,
            "upstream_error_code": result.error_code if result and result.status != "SUCCESS" else None,
            "upstream_error_message": result.error_message if result and result.status != "SUCCESS" else None,
            "status": result.status if result else "FAILED",
            "connector": result.connector_name if result else None,
            "connectors_tried": tried,
            "delivery_mode": delivery_mode,
            "origin_kind": origin_kind,
            "fallback_used": delivery_mode == "stale_cache",
            "lookup_mode": target.mode,
            "canonical_identifier": target.identifier,
            "lookup_state": lookup_state,
            "source_attempts": attempts,
            "incomplete_results": incomplete_results,
        }

    def _fresh_cache_response(
        self,
        query: ResourceQuery,
        target: LookupTarget,
        source_name: str,
        *,
        tried: list[str],
        attempts: list[dict[str, Any]],
        observed_error: bool,
    ) -> dict[str, Any] | None:
        cached = self.repository.fresh_cache(
            query,
            source_name=source_name,
            lookup_target=target,
        )
        cached = self._validated_cached_resources(cached, target)
        if not cached:
            return None
        cached_origin = self._cached_origin(cached, source_name)
        deliverable = self._deliverable_cached_resources(cached, target)
        resources = self._present_resources(deliverable, "fresh_cache", cached_origin)
        incomplete_results = _cached_incomplete_results(cached)
        cached_attempt = {
            "source_name": source_name,
            "status": "PARTIAL" if incomplete_results else "FOUND",
            "result_count": len(cached),
            "delivery_mode": "fresh_cache",
            "error_code": None,
            "fetched_at": resources[0].get("fetched_at") if resources else None,
            "incomplete_results": incomplete_results,
        }
        if target.kind == "arxiv" and target.mode == "EXACT" and not resources:
            return {
                "_continue_source_chain": True,
                "source_attempts": [cached_attempt],
            }
        return {
            "cache_hit": True,
            "resources": resources,
            "error": None,
            "connectors_tried": list(tried),
            "connector": source_name,
            "status": "SUCCESS",
            "delivery_mode": "fresh_cache",
            "origin_kind": cached_origin,
            "lookup_mode": target.mode,
            "canonical_identifier": target.identifier,
            "lookup_state": "PARTIAL" if incomplete_results or observed_error else "FOUND",
            "source_attempts": [*attempts, cached_attempt],
            "incomplete_results": incomplete_results,
        }

    def _unsupported_response(
        self,
        query: ResourceQuery,
        connector_name: str | None,
        target: LookupTarget,
    ) -> dict[str, Any]:
        return {
            "cache_hit": False,
            "resources": [],
            "error": "CONNECTOR_NOT_CONFIGURED",
            "connectors_tried": [],
            "connector": connector_name,
            "status": "FAILED",
            "delivery_mode": "failed",
            "origin_kind": origin_for_source(connector_name, self.trusted_sources),
            "lookup_mode": target.mode,
            "canonical_identifier": target.identifier,
            "lookup_state": "UNSUPPORTED",
            "source_attempts": [],
            "incomplete_results": None,
        }

    def _cached_origin(self, resources: list[dict[str, Any]], connector_name: str | None) -> str:
        if resources:
            recorded = str(resources[0].get("origin_kind") or "")
            if recorded in {"remote", "local_catalog"}:
                return recorded
        return origin_for_source(connector_name, self.trusted_sources)

    @staticmethod
    def _present_resources(
        resources: list[dict[str, Any]],
        delivery_mode: str,
        origin_kind: str,
    ) -> list[dict[str, Any]]:
        return [
            normalize_standard_resource_view(
                {
                    **resource,
                    "origin_kind": resource.get("origin_kind") or origin_kind,
                    "delivery_mode": delivery_mode,
                    "provenance_verified": bool(resource.get("provenance_verified")),
                }
            )
            for resource in resources
        ]

    def _delivery_mode(self, result: ConnectorSyncResult | None, resources: list[dict[str, Any]]) -> str:
        if result is not None:
            try:
                connector = self.registry.get(result.connector_name)
            except KeyError:
                connector = None
            if result.status == "SUCCESS" and connector is not None and not getattr(connector, "supports_remote", True):
                return "local_metadata"
        return _delivery_mode(result, resources)

    def _connector_candidates(self, query: ResourceQuery, target: LookupTarget) -> list[str]:
        available = set(self.registry.names())
        if target.mode == "EXACT":
            return [name for name in target.source_candidates if name in available]
        by_type = {
            "paper": ["semantic_scholar", "arxiv"],
            "vulnerability": ["nvd"],
            "standard": ["standard_source", "rfc_editor"],
            "project": ["github"],
        }
        names = by_type.get(query.resource_type or "", ["semantic_scholar", "arxiv", "nvd", "standard_source", "github"])
        return [name for name in names if name in available]

    @staticmethod
    def _validated_cached_resources(resources: list[dict[str, Any]], target: LookupTarget) -> list[dict[str, Any]]:
        return [
            resource
            for resource in resources
            if (
                str(resource.get("source_name") or "") != "github"
                or _is_public_github_record(resource)
            )
            and (
                target.mode != "EXACT"
                or matches_exact_identity(target, str(resource.get("source_name") or ""), resource)
            )
        ]

    @classmethod
    def _deliverable_cached_resources(
        cls,
        resources: list[dict[str, Any]],
        target: LookupTarget,
    ) -> list[dict[str, Any]]:
        validated = cls._validated_cached_resources(resources, target)
        if target.mode == "EXACT" and target.kind == "arxiv":
            return [item for item in validated if is_deliverable_exact_arxiv_resource(target, item)]
        return validated

    async def sync(
        self,
        connector_name: str,
        query: ResourceQuery,
        *,
        deadline: float | None = None,
        lookup_target: LookupTarget | None = None,
        before_remote: Callable[[str], None] | None = None,
    ) -> ConnectorSyncResult:
        target = lookup_target or parse_lookup_target(query.query, query.resource_type)
        validate_lookup_filters(
            target,
            page=query.page,
            year_from=query.year_from,
            year_to=query.year_to,
            module=query.module,
        )
        validate_connector_query(target, connector_name=connector_name, query=query.query)
        audit_query_payload = query.model_dump(mode="json")
        audit_query_payload["lookup_mode"] = target.mode
        audit_query_payload["canonical_identifier"] = target.identifier
        started = time.perf_counter()
        try:
            connector = self.registry.get(connector_name)
        except KeyError:
            connector = None
        enters_remote = (
            connector is not None
            and connector.enabled
            and getattr(connector, "supports_remote", True)
            and origin_for_source(connector.name, self.trusted_sources) == "remote"
            and connector.will_attempt_remote(query)
        )
        if before_remote is not None and enters_remote:
            _ensure_before_deadline(deadline)
            before_remote(connector.name)
        run_id = self.storage.create_connector_sync_run(
            connector_name=connector_name,
            query=audit_query_payload,
            status="RUNNING",
            cache_hit=False,
        )
        if connector is None:
            error = ConnectorError(f"Connector not found: {connector_name}", error_code="CONNECTOR_NOT_FOUND")
            self.storage.finish_connector_sync_run(
                run_id,
                status="FAILED",
                result_count=0,
                cache_hit=False,
                error_code=error.error_code,
                error_message=str(error),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return ConnectorSyncResult(
                connector_name=connector_name,
                status="FAILED",
                error_code=error.error_code,
                error_message=str(error),
            )
        if not connector.enabled:
            error = ConnectorError(f"Connector disabled: {connector.name}", error_code="CONNECTOR_DISABLED")
            cached = self.repository.search_cache(
                query,
                include_expired=True,
                source_name=connector.name,
                lookup_target=target,
            )
            self.storage.finish_connector_sync_run(
                run_id,
                status="FAILED_WITH_CACHE" if cached else "FAILED",
                result_count=len(cached),
                cache_hit=bool(cached),
                error_code=error.error_code,
                error_message=str(error),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return ConnectorSyncResult(
                connector_name=connector.name,
                status="FAILED_WITH_CACHE" if cached else "FAILED",
                result_count=len(cached),
                cache_hit=bool(cached),
                error_code=error.error_code,
                error_message=str(error),
            )
        try:
            _ensure_before_deadline(deadline)
            origin_kind = origin_for_source(connector.name, self.trusted_sources)
            if origin_kind == "unknown":
                raise ConnectorError(
                    f"Connector is not trusted for student evidence: {connector.name}",
                    error_code="UNTRUSTED_CONNECTOR",
                )
            batch = await self._search_with_retry(connector, query, deadline=deadline)
            resources = [
                resource
                for resource in batch.resources
                if matches_exact_identity(target, connector.name, resource.model_dump(mode="json"))
            ]
            _ensure_before_deadline(deadline)
            prepared: list[dict[str, Any]] = []
            previous_by_key: dict[tuple[str, str], dict[str, Any] | None] = {}
            for resource in resources:
                if resource.source_name != connector.name:
                    raise ConnectorServiceError(
                        "Connector returned a resource with a mismatched source.",
                        error_code="CONNECTOR_SOURCE_MISMATCH",
                    )
                if connector.name == "github" and not _is_public_github_resource(resource):
                    raise ConnectorServiceError(
                        "GitHub Connector returned a repository that is not verified as public.",
                        error_code="GITHUB_REPOSITORY_NOT_PUBLIC",
                    )
                if not resource.url.strip().lower().startswith("https://"):
                    raise ConnectorServiceError(
                        "Connector returned a resource without a valid HTTPS source URL.",
                        error_code="RESOURCE_URL_REQUIRED",
                    )
                digest = content_hash(resource)
                previous_by_key[(resource.source_name, resource.external_id)] = self.storage.external_resource_by_key(
                    resource.source_name,
                    resource.external_id,
                )
                prepared.append({"resource": resource.model_dump(mode="json"), "content_hash": digest})
            _ensure_before_deadline(deadline)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            saved = self.storage.commit_verified_sync(
                run_id=run_id,
                connector_name=connector.name,
                origin_kind=origin_kind,
                query=cache_query_payload(
                    query,
                    target,
                    incomplete_results=batch.incomplete_results,
                ),
                resources=prepared,
                ttl_hours=self.settings.external_cache_ttl_hours,
                latency_ms=latency_ms,
            )
            for resource, record in zip(resources, saved, strict=True):
                try:
                    self._index_committed_resource(
                        resource,
                        record,
                        previous_by_key.get((resource.source_name, resource.external_id)),
                    )
                except Exception:
                    continue
            return ConnectorSyncResult(
                connector_name=connector.name,
                status="SUCCESS",
                result_count=len(saved),
                cache_hit=False,
                resources=resources,
                incomplete_results=batch.incomplete_results,
            )
        except ConnectorError as exc:
            cached = self.repository.search_cache(
                query,
                include_expired=True,
                source_name=connector.name,
                lookup_target=target,
            )
            self.storage.finish_connector_sync_run(
                run_id,
                status="FAILED_WITH_CACHE" if cached else "FAILED",
                result_count=len(cached),
                cache_hit=bool(cached),
                error_code=exc.error_code,
                error_message=str(exc),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return ConnectorSyncResult(
                connector_name=connector.name,
                status="FAILED_WITH_CACHE" if cached else "FAILED",
                result_count=len(cached),
                cache_hit=bool(cached),
                error_code=exc.error_code,
                error_message=str(exc),
            )
        except Exception as exc:
            error = ConnectorServiceError(
                "External connector call failed.",
                error_code="CONNECTOR_SERVICE_ERROR",
            )
            cached = self.repository.search_cache(
                query,
                include_expired=True,
                source_name=connector.name,
                lookup_target=target,
            )
            status = "FAILED_WITH_CACHE" if cached else "FAILED"
            error_message = f"{error}: {exc.__class__.__name__}"
            self.storage.finish_connector_sync_run(
                run_id,
                status=status,
                result_count=len(cached),
                cache_hit=bool(cached),
                error_code=error.error_code,
                error_message=error_message,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return ConnectorSyncResult(
                connector_name=connector.name,
                status=status,
                result_count=len(cached),
                cache_hit=bool(cached),
                error_code=error.error_code,
                error_message=error_message,
            )

    async def _search_with_retry(
        self,
        connector: Any,
        query: ResourceQuery,
        *,
        deadline: float | None = None,
    ) -> ConnectorSearchBatch:
        attempts = max(1, self.settings.external_http_max_retries + 1)
        last_exc: ConnectorError | None = None
        for attempt in range(attempts):
            try:
                timeout_seconds = _connector_timeout_seconds(
                    self.settings.external_http_timeout_seconds,
                    deadline,
                )
                search_with_metadata = getattr(connector, "search_with_metadata", None)
                if callable(search_with_metadata):
                    batch = await asyncio.wait_for(search_with_metadata(query), timeout=timeout_seconds)
                else:
                    resources = await asyncio.wait_for(connector.search(query), timeout=timeout_seconds)
                    batch = ConnectorSearchBatch(resources=resources)
                if isinstance(batch, ConnectorSearchBatch):
                    return batch
                return ConnectorSearchBatch.model_validate(batch)
            except asyncio.TimeoutError as exc:
                last_exc = ConnectorTimeoutError("外部 Connector 请求超时。")
            except httpx.TimeoutException as exc:
                last_exc = ConnectorTimeoutError("External connector request timed out.")
            except httpx.TransportError as exc:
                last_exc = ConnectorServiceError(
                    f"External connector transport error: {exc.__class__.__name__}",
                    error_code="CONNECTOR_SERVICE_ERROR",
                )
            except ConnectorRateLimitError as exc:
                last_exc = exc
            except ConnectorServiceError as exc:
                last_exc = exc
            except ConnectorError as exc:
                last_exc = exc
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            if attempt < attempts - 1:
                await asyncio.sleep(0)
        raise last_exc or ConnectorError("外部 Connector 调用失败。")

    def _index_committed_resource(
        self,
        resource: ExternalResource,
        record: dict[str, Any],
        previous: dict[str, Any] | None,
    ) -> None:
        if previous and previous.get("content_hash") != record.get("content_hash") and previous.get("vector_id"):
            self._delete_vector(previous["vector_id"])
        if self.settings.external_resource_index_enabled and _is_indexable(resource):
            vector_id = f"external:{record['id']}"
            self._upsert_vector(vector_id=vector_id, record=record, resource=resource)
            self.storage.mark_external_resource_indexed(int(record["id"]), vector_id)
            record["indexed"] = True
            record["vector_id"] = vector_id

    def _vector_store(self) -> Any:
        if self.vector_store is None:
            self.vector_store = ChromaVectorStore()
        return self.vector_store

    def _upsert_vector(self, *, vector_id: str, record: dict[str, Any], resource: ExternalResource) -> None:
        store = self._vector_store()
        if hasattr(store, "upsert_external_resource"):
            store.upsert_external_resource(vector_id, resource_content(resource), record)
        else:
            store.collection.upsert(
                ids=[vector_id],
                documents=[resource_content(resource)],
                metadatas=[
                    {
                        "external_resource_id": int(record["id"]),
                        "source_name": resource.source_name,
                        "external_id": resource.external_id,
                        "resource_type": resource.resource_type,
                        "title": resource.title,
                    }
                ],
            )

    def _delete_vector(self, vector_id: str) -> None:
        store = self._vector_store()
        if hasattr(store, "delete_external_resource"):
            store.delete_external_resource(vector_id)
        else:
            store.collection.delete(ids=[vector_id])


def _is_indexable(resource: ExternalResource) -> bool:
    return resource.resource_type in {"paper", "vulnerability", "standard", "project"}


def _is_public_github_resource(resource: ExternalResource) -> bool:
    metadata = resource.raw_metadata
    visibility = metadata.get("visibility")
    return (
        isinstance(metadata.get("full_name"), str)
        and bool(metadata["full_name"].strip())
        and metadata.get("private") is False
        and isinstance(visibility, str)
        and bool(visibility.strip())
        and visibility.lower() not in {"private", "internal"}
    )


def _is_public_github_record(resource: dict[str, Any]) -> bool:
    metadata = resource.get("raw_metadata")
    if not isinstance(metadata, dict):
        return False
    visibility = metadata.get("visibility")
    return (
        isinstance(metadata.get("full_name"), str)
        and bool(metadata["full_name"].strip())
        and metadata.get("private") is False
        and isinstance(visibility, str)
        and bool(visibility.strip())
        and visibility.lower() not in {"private", "internal"}
    )


def _delivery_mode(result: ConnectorSyncResult | None, resources: list[dict[str, Any]]) -> str:
    if result is None:
        return "cache_miss"
    if result.status in {"SUCCESS", "PARTIAL"}:
        return "live"
    if result.status == "FAILED_WITH_CACHE" and resources:
        return "stale_cache"
    if result.cache_hit and resources:
        return "stale_cache"
    return "failed"


def _ensure_before_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise ConnectorTimeoutError("External connector deadline expired before cache commit.")


def _connector_timeout_seconds(configured_timeout: float, deadline: float | None) -> float:
    if deadline is None:
        return configured_timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ConnectorTimeoutError("External connector deadline expired before request.")
    return min(configured_timeout, remaining)


def _attempt_status(result: ConnectorSyncResult) -> str:
    if result.incomplete_results is True:
        return "PARTIAL"
    if result.status == "FAILED_WITH_CACHE":
        return "ERROR"
    if result.result_count > 0:
        return "FOUND"
    if result.status in {"SUCCESS", "PARTIAL"}:
        return "EMPTY"
    if result.error_code == "CONNECTOR_DISABLED":
        return "DISABLED"
    return "ERROR"


def _cached_incomplete_results(resources: list[dict[str, Any]]) -> bool | None:
    values = {resource.get("query_incomplete_results") for resource in resources}
    if True in values or 1 in values:
        return True
    if False in values or 0 in values:
        return False
    return None
