"""Connector registry with a deterministic FakeConnector for framework tests."""

from __future__ import annotations

import asyncio
import time
from typing import Iterable

from backend.connectors.base import ExternalConnector
from backend.connectors.arxiv import ArxivConnector
from backend.connectors.github import GitHubConnector
from backend.connectors.nvd import NVDConnector
from backend.connectors.rfc_editor import RFCEditorConnector
from backend.connectors.schemas import ConnectorHealth, ExternalResource, ResourceQuery
from backend.connectors.semantic_scholar import SemanticScholarConnector
from backend.connectors.standard_source import StandardSourceConnector


class FakeConnector(ExternalConnector):
    name = "fake"
    enabled = True
    supports_remote = False

    def __init__(
        self,
        resources: list[ExternalResource] | None = None,
        *,
        delay_seconds: float = 0.0,
        fail: Exception | None = None,
    ) -> None:
        self._resources = resources or _default_fake_resources()
        self._delay_seconds = delay_seconds
        self._fail = fail

    async def health_check(self) -> ConnectorHealth:
        started = time.perf_counter()
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        status = "error" if self._fail else "ok"
        return ConnectorHealth(
            name=self.name,
            enabled=self.enabled,
            status=status,
            detail="" if not self._fail else str(self._fail),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        if self._fail:
            raise self._fail
        q = query.query.strip().lower()
        items = [
            item
            for item in self._resources
            if (not query.resource_type or item.resource_type == query.resource_type)
            and (not query.module or item.module == query.module)
            and (not q or q in item.title.lower() or q in item.abstract.lower() or q in " ".join(item.tags).lower())
        ]
        start = (query.page - 1) * query.page_size
        end = start + query.page_size
        return items[start:end]


class ConnectorRegistry:
    def __init__(self, connectors: Iterable[ExternalConnector] | None = None) -> None:
        self._connectors = {connector.name: connector for connector in (connectors or [])}

    def names(self) -> list[str]:
        return sorted(self._connectors)

    def get(self, name: str) -> ExternalConnector:
        if name not in self._connectors:
            raise KeyError(name)
        return self._connectors[name]

    def enabled(self) -> list[ExternalConnector]:
        return [connector for connector in self._connectors.values() if connector.enabled]

    async def statuses(self) -> list[dict]:
        statuses = []
        for connector in self._connectors.values():
            health = await connector.health_check()
            statuses.append(health.model_dump())
        return statuses


def get_default_registry() -> ConnectorRegistry:
    return ConnectorRegistry(
        [
            SemanticScholarConnector(),
            ArxivConnector(),
            NVDConnector(),
            StandardSourceConnector(),
            GitHubConnector(),
            RFCEditorConnector(),
        ]
    )


def _default_fake_resources() -> list[ExternalResource]:
    return [
        ExternalResource(
            external_id="paper-5g-nas-001",
            resource_type="paper",
            source_name="fake",
            title="5G NAS Security Survey",
            authors=["Demo Author"],
            abstract="A survey about 5G NAS authentication, replay risk, and security context management.",
            url="https://example.test/paper-5g-nas-001",
            published_at="2026-01-01",
            tags=["5G NAS", "security"],
            module="5G NAS",
            rights_note="fake local connector data",
        ),
        ExternalResource(
            external_id="cve-demo-001",
            resource_type="vulnerability",
            source_name="fake",
            title="Demo CVE for LLM vulnerability detection",
            abstract="A fake vulnerability record for teaching cache and connector workflows.",
            url="https://example.test/cve-demo-001",
            tags=["CVE", "LLM"],
            module="LLM 漏洞检测",
            rights_note="fake local connector data",
        ),
    ]
