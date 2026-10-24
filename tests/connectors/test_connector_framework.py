from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from backend import main
from backend.connectors.base import ConnectorRateLimitError, ConnectorServiceError, ExternalConnector
from backend.connectors.github import GitHubConnector
from backend.connectors.registry import ConnectorRegistry, FakeConnector
from backend.connectors.nvd import NVDConnector
from backend.connectors.rfc_editor import RFCEditorConnector
from backend.connectors.standard_source import StandardSourceConnector
from backend.connectors.schemas import ConnectorHealth, ConnectorSearchBatch, ExternalResource, ResourceQuery
from backend.resources.portal import ResourcePortalQuery, ResourcePortalService
from backend.resources.query_intent import is_deliverable_exact_arxiv_resource, parse_lookup_target
from backend.resources.repository import cache_query_payload
from backend.resources.service import ExternalResourceService
from backend.storage.logger import LearningLogger


def _login_token(client: TestClient, username: str) -> str:
    response = client.post("/api/auth/login", json={"username": username, "password": "demo123456"})
    assert response.status_code == 200
    payload = response.json()
    token = payload["access_token"]
    if payload.get("must_change_password"):
        changed = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": "demo123456", "new_password": f"{username}-changed-123"},
        )
        assert changed.status_code == 200
        token = changed.json()["access_token"]
    return token


class RecordingVectorStore:
    def __init__(self) -> None:
        self.upserts: dict[str, str] = {}
        self.deleted: list[str] = []

    def upsert_external_resource(self, vector_id: str, text: str, record: dict) -> None:
        self.upserts[vector_id] = text

    def delete_external_resource(self, vector_id: str) -> None:
        self.deleted.append(vector_id)
        self.upserts.pop(vector_id, None)


class HTTPFakeConnector(ExternalConnector):
    name = "http_fake"
    enabled = True

    def __init__(self, url: str) -> None:
        self.url = url

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, status="ok")

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        async with httpx.AsyncClient(timeout=1) as client:
            response = await client.get(self.url, params={"q": query.query})
        if response.status_code == 429:
            raise ConnectorRateLimitError("rate limited", error_code="CONNECTOR_RATE_LIMIT")
        if response.status_code >= 500:
            raise ConnectorServiceError("upstream failed", error_code="CONNECTOR_SERVICE_ERROR")
        response.raise_for_status()
        payload = response.json()
        return [
            ExternalResource(
                external_id=item["id"],
                resource_type="paper",
                source_name=self.name,
                title=item["title"],
                abstract=item.get("abstract", ""),
                url=item.get("url", ""),
                module="5G NAS",
            )
            for item in payload.get("items", [])
        ]


class DisabledConnector(ExternalConnector):
    name = "disabled_fake"
    enabled = False

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, enabled=False, status="disabled")

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        raise AssertionError("disabled connector must not be called")


class ConnectionFailConnector(ExternalConnector):
    name = "connect_fail"
    enabled = True

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, status="error")

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        request = httpx.Request("GET", "https://connector.test/fail")
        raise httpx.ConnectError("network unreachable", request=request)


class NVDFakeConnector(FakeConnector):
    name = "nvd"
    supports_remote = True


class ArxivFakeConnector(FakeConnector):
    name = "arxiv"
    supports_remote = True


class CountingArxivConnector(ArxivFakeConnector):
    def __init__(self, resources: list[ExternalResource]) -> None:
        super().__init__(resources=resources)
        self.calls = 0

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        self.calls += 1
        return await super().search(query)


class CancellationResistantConnector(FakeConnector):
    name = "arxiv"
    supports_remote = True

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        try:
            await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            await asyncio.sleep(0.08)
        return self._resources[: query.page_size]


class GitHubFakeConnector(FakeConnector):
    name = "github"
    supports_remote = True


class UnfilteredConnector(FakeConnector):
    name = "semantic_scholar"

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        return self._resources[: query.page_size]


class CountingSemanticConnector(UnfilteredConnector):
    supports_remote = True

    def __init__(self, resources: list[ExternalResource]) -> None:
        super().__init__(resources=resources)
        self.calls = 0

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        self.calls += 1
        return await super().search(query)


class UnfilteredArxivConnector(UnfilteredConnector):
    name = "arxiv"
    supports_remote = True


class CountingUnfilteredArxivConnector(UnfilteredArxivConnector):
    def __init__(self, resources: list[ExternalResource]) -> None:
        super().__init__(resources=resources)
        self.calls = 0

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        self.calls += 1
        return await super().search(query)


class UnfilteredNVDConnector(UnfilteredConnector):
    name = "nvd"


class EmptySemanticConnector(ExternalConnector):
    name = "semantic_scholar"
    enabled = True
    supports_remote = True

    def __init__(self) -> None:
        self.calls = 0

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, status="ok")

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        self.calls += 1
        return []


class EmptyArxivConnector(EmptySemanticConnector):
    name = "arxiv"


class CountingRFCConnector(ExternalConnector):
    name = "rfc_editor"
    enabled = True
    supports_remote = True

    def __init__(self) -> None:
        self.calls = 0

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, status="ok")

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        self.calls += 1
        return [
            ExternalResource(
                external_id="RFC-9000",
                resource_type="standard",
                source_name=self.name,
                title="RFC 9000 live metadata",
                url="https://www.rfc-editor.org/rfc/rfc9000",
            )
        ]


class CountingNVDConnector(UnfilteredNVDConnector):
    def __init__(self, resources: list[ExternalResource]) -> None:
        super().__init__(resources=resources)
        self.calls = 0

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        self.calls += 1
        return await super().search(query)


class BatchGitHubConnector(ExternalConnector):
    name = "github"
    enabled = True
    supports_remote = True

    def __init__(self, resources: list[ExternalResource], *, incomplete_results: bool) -> None:
        self.resources = resources
        self.incomplete_results = incomplete_results
        self.calls = 0

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, status="ok")

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        return (await self.search_with_metadata(query)).resources

    async def search_with_metadata(self, query: ResourceQuery) -> ConnectorSearchBatch:
        self.calls += 1
        return ConnectorSearchBatch(resources=self.resources, incomplete_results=self.incomplete_results)


def _service(tmp_path, connectors, monkeypatch, ttl: int = 24, vector_store=None) -> ExternalResourceService:
    monkeypatch.setenv("EXTERNAL_CACHE_TTL_HOURS", str(ttl))
    monkeypatch.setenv("EXTERNAL_HTTP_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("EXTERNAL_HTTP_MAX_RETRIES", "1")
    storage = LearningLogger(tmp_path / "connectors.db")
    trusted_sources = {
        connector.name: "remote" if getattr(connector, "supports_remote", True) else "local_catalog"
        for connector in connectors
    }
    return ExternalResourceService(
        storage=storage,
        registry=ConnectorRegistry(connectors),
        vector_store=vector_store or RecordingVectorStore(),
        trusted_sources=trusted_sources,
    )


def test_lookup_target_parser_distinguishes_exact_identifiers_and_search():
    from backend.resources.query_intent import LookupInputError, parse_lookup_target

    cases = [
        ("doi:10.1234/ABC.Def", "paper", "doi", "10.1234/abc.def", ("semantic_scholar",)),
        ("https://doi.org/10.1234/ABC.Def", "paper", "doi", "10.1234/abc.def", ("semantic_scholar",)),
        ("arXiv:2601.00001v2", "paper", "arxiv", "2601.00001v2", ("semantic_scholar", "arxiv")),
        ("arXiv:2601.00001v2、CVE-2026-12345", "paper", "arxiv", "2601.00001v2", ("semantic_scholar", "arxiv")),
        ("doi:10.1234/ABC.Def、GitHub open5gs/open5gs", "paper", "doi", "10.1234/abc.def", ("semantic_scholar",)),
        ("https://arxiv.org/abs/hep-th/9901001v2", "paper", "arxiv", "hep-th/9901001v2", ("semantic_scholar", "arxiv")),
        ("CVE-2026-12345", "vulnerability", "cve", "CVE-2026-12345", ("nvd",)),
        ("https://github.com/open5gs/open5gs", "project", "github", "open5gs/open5gs", ("github",)),
        (
            "请核对 arXiv:1706.03762、CVE-2021-44228 和 GitHub open5gs/open5gs 的公开身份。",
            "project",
            "github",
            "open5gs/open5gs",
            ("github",),
        ),
        ("RFC 9000", "standard", "rfc", "RFC-9000", ("rfc_editor", "standard_source")),
    ]
    for raw, resource_type, kind, identifier, sources in cases:
        target = parse_lookup_target(raw, resource_type)
        assert target.mode == "EXACT"
        assert target.kind == kind
        assert target.identifier == identifier
        assert target.source_candidates == sources

    search = parse_lookup_target("post-quantum network security", "paper")
    assert search.mode == "SEARCH"
    assert search.identifier is None

    with pytest.raises(LookupInputError) as ambiguous:
        parse_lookup_target("CVE-2026-1234 and CVE-2026-5678", "vulnerability")
    assert ambiguous.value.error_code == "AMBIGUOUS_IDENTIFIER"


def test_lookup_target_parser_rejects_unsafe_github_forms_without_false_ambiguity():
    from backend.resources.query_intent import LookupInputError

    for unsafe in (
        "https://example.test/owner/repo",
        "https://user:pass@github.com/owner/repo",
        "https://github.com:443/owner/repo",
        "https://github.com/owner/repo/issues",
        "https://github.com/owner/../repo",
    ):
        with pytest.raises(LookupInputError) as rejected:
            parse_lookup_target(unsafe, "project")
        assert rejected.value.error_code == "INVALID_ARGUMENTS"

    assert parse_lookup_target("CI/CD pipeline", "project").mode == "SEARCH"
    github = parse_lookup_target("https://github.com/Open5GS/open5gs.git", "project")
    assert (github.mode, github.identifier) == ("EXACT", "open5gs/open5gs")
    schemeless = parse_lookup_target("github.com/Open5GS/open5gs.git", "project")
    assert (schemeless.mode, schemeless.identifier) == ("EXACT", "open5gs/open5gs")
    doi = parse_lookup_target("doi:10.1234/demo", None)
    arxiv = parse_lookup_target("arXiv:hep-th/9901001v2", None)
    assert (doi.kind, doi.identifier) == ("doi", "10.1234/demo")
    assert (arxiv.kind, arxiv.identifier) == ("arxiv", "hep-th/9901001v2")

    with pytest.raises(LookupInputError) as ambiguous:
        parse_lookup_target("doi:10.1234/one and doi:10.1234/two", None)
    assert ambiguous.value.error_code == "AMBIGUOUS_IDENTIFIER"


@pytest.mark.parametrize("resource_type", [None, "project"])
@pytest.mark.parametrize(
    "value",
    [
        "GitHub open5gs/open5gs?tab=readme",
        "GitHub open5gs/open5gs#readme",
        "GitHub open5gs/open5gs:443",
        "GitHub open5gs/open5gs@evil",
        "GitHub open5gs/open5gs/issues",
        "GitHub open5gs/open5gs/",
        "GitHub open5gs/open5gs.git?tab=readme",
        "GitHub open5gs/open5gs.git#readme",
        "GitHub open5gs/open5gs.git:443",
        "GitHub open5gs/open5gs.git@evil",
        "GitHub open5gs/open5gs.git/issues",
        "GitHub open5gs/open5gs.git/",
    ],
)
def test_github_prefixed_repository_rejects_attached_token_suffixes(value, resource_type):
    from backend.resources.query_intent import LookupInputError

    with pytest.raises(LookupInputError) as rejected:
        parse_lookup_target(value, resource_type)

    assert rejected.value.error_code == "INVALID_ARGUMENTS"


@pytest.mark.parametrize("resource_type", [None, "project"])
@pytest.mark.parametrize(
    "value",
    [
        "GitHub open5gs/open5gs.",
        "GitHub open5gs/open5gs,",
        "GitHub open5gs/open5gs;",
        "GitHub open5gs/open5gs?",
        "GitHub open5gs/open5gs!",
        "GitHub open5gs/open5gs。",
        "GitHub open5gs/open5gs，",
        "GitHub open5gs/open5gs；",
        "GitHub open5gs/open5gs？",
        "GitHub open5gs/open5gs！",
        'GitHub "open5gs/open5gs"',
        "GitHub 'open5gs/open5gs'",
        "GitHub open5gs/open5gs.git.",
    ],
)
def test_github_prefixed_repository_accepts_only_bounded_natural_terminators(value, resource_type):
    target = parse_lookup_target(value, resource_type)

    assert (target.mode, target.kind, target.identifier) == (
        "EXACT",
        "github",
        "open5gs/open5gs",
    )


@pytest.mark.parametrize("resource_type", [None, "project"])
@pytest.mark.parametrize(
    "value",
    [
        "https://github.com/a/b#",
        "https://github.com/a/b#.",
        "https://github.com/a/b?#",
        "https://github.com:/a/b",
        "github.com/a/b#",
        "github.com/a/b#.",
        "github.com/a/b?#",
        "github.com:/a/b",
        "GitHub https://github.com/a/b#",
        "GitHub https://github.com/a/b#.",
        "GitHub https://github.com/a/b?#",
        "GitHub github.com/a/b#",
        "GitHub github.com/a/b#.",
        "GitHub github.com/a/b?#",
        "GitHub https://github.com:/a/b",
        "GitHub github.com:/a/b",
    ],
)
def test_github_urls_reject_empty_fragment_and_incomplete_authority(value, resource_type):
    from backend.resources.query_intent import LookupInputError

    with pytest.raises(LookupInputError) as rejected:
        parse_lookup_target(value, resource_type)

    assert rejected.value.error_code == "INVALID_ARGUMENTS"


@pytest.mark.parametrize("resource_type", [None, "project"])
@pytest.mark.parametrize(
    "value",
    [
        "https://github.com/a/b",
        "https://GITHUB.COM/a/b",
        "https://github.com/a/b.",
        "https://github.com/a/b,",
        "https://github.com/a/b;",
        "https://github.com/a/b?",
        "https://github.com/a/b!",
        'https://github.com/a/b"',
        "https://github.com/a/b'",
        "https://github.com/a/b.git",
        "https://github.com/a/b.git.",
    ],
)
def test_github_urls_keep_bounded_natural_terminators(value, resource_type):
    target = parse_lookup_target(value, resource_type)

    assert (target.mode, target.kind, target.identifier) == ("EXACT", "github", "a/b")


def test_resource_service_rejects_invalid_github_url_before_connector(tmp_path, monkeypatch):
    from backend.resources.query_intent import LookupInputError

    connector = BatchGitHubConnector([], incomplete_results=False)
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(
                    query="GitHub https://github.com:/a/b",
                    resource_type="project",
                    allow_remote=True,
                ),
                connector_name="github",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


def test_lookup_target_parser_disambiguates_bare_paper_ids_and_fails_closed_for_urls():
    from backend.resources.query_intent import LookupInputError

    doi = parse_lookup_target("10.1234/demo", None)
    arxiv = parse_lookup_target("hep-th/9901001v2", None)
    assert (doi.mode, doi.kind, doi.identifier) == ("EXACT", "doi", "10.1234/demo")
    assert (arxiv.mode, arxiv.kind, arxiv.identifier) == ("EXACT", "arxiv", "hep-th/9901001v2")

    assert parse_lookup_target("CI/CD", "project").mode == "SEARCH"
    assert parse_lookup_target("docs/API", "project").mode == "SEARCH"
    repository = parse_lookup_target("open5gs/open5gs", "project")
    assert (repository.mode, repository.identifier) == ("EXACT", "open5gs/open5gs")

    for unsafe in (
        "https://example.test/owner/repo",
        "https://user:pass@github.com/owner/repo",
        "https://github.com:443/owner/repo",
        "https://github.com:not-a-port/owner/repo",
        "https://github.com/owner/../repo",
        "https://github.com/owner/repo/issues",
    ):
        with pytest.raises(LookupInputError) as rejected:
            parse_lookup_target(unsafe, None)
        assert rejected.value.error_code == "INVALID_ARGUMENTS"

    assert parse_lookup_target("https://doi.org/10.1234/demo", None).kind == "doi"
    assert parse_lookup_target("https://arxiv.org/abs/hep-th/9901001v2", None).kind == "arxiv"
    assert parse_lookup_target("https://github.com/Open5GS/open5gs.git", None).kind == "github"


@respx.mock
def test_project_context_routes_valid_alphabetic_repositories_to_exact_endpoint(monkeypatch):
    monkeypatch.setenv("ENABLE_GITHUB_SEARCH", "true")
    base = "https://github-alpha.test"
    repositories = ("pallets/flask", "facebook/react")
    exact_routes = []
    for index, repository in enumerate(repositories, start=1):
        exact_routes.append(
            respx.get(f"{base}/repos/{repository}").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": index,
                        "full_name": repository,
                        "private": False,
                        "visibility": "public",
                        "html_url": f"https://github.com/{repository}",
                    },
                )
            )
        )
    search_route = respx.get(f"{base}/search/repositories").mock(
        return_value=httpx.Response(200, json={"items": [], "incomplete_results": False})
    )

    targets = [parse_lookup_target(repository, "project") for repository in repositories]
    connector = GitHubConnector(base_url=f"{base}/search/repositories")
    results = [
        asyncio.run(connector.search(ResourceQuery(query=repository, resource_type="project")))
        for repository in repositories
    ]

    assert [(target.mode, target.kind, target.identifier) for target in targets] == [
        ("EXACT", "github", "pallets/flask"),
        ("EXACT", "github", "facebook/react"),
    ]
    assert [items[0].title for items in results] == list(repositories)
    assert all(route.called for route in exact_routes)
    assert not search_route.called
    assert parse_lookup_target("CI/CD", "project").mode == "SEARCH"
    assert parse_lookup_target("docs/API", "project").mode == "SEARCH"
    assert parse_lookup_target("compare client/server architecture", "project").mode == "SEARCH"
    assert parse_lookup_target("pallets/flask", None).mode == "SEARCH"


def test_identifier_urls_and_arxiv_tokens_are_validated_as_complete_inputs():
    from backend.resources.query_intent import LookupInputError

    invalid_paper_urls = (
        "https://doi.org/10.bad/foo",
        "https://doi.org/10.1234/",
        "https://arxiv.org/abs/not-an-id",
        "https://arxiv.org/pdf/not-an-id",
    )
    outcomes = []
    for resource_type in (None, "paper"):
        for value in invalid_paper_urls:
            try:
                outcomes.append(parse_lookup_target(value, resource_type))
            except LookupInputError as exc:
                outcomes.append(exc)

    pseudo_github = (
        "ftp://github.com/owner/repo",
        "javascript://github.com/owner/repo",
        "ssh://git@github.com/owner/repo",
        "mailto:github.com/owner/repo",
    )
    pseudo_results = [parse_lookup_target(value, None) for value in pseudo_github]
    bare_trailing_outcomes = []
    for resource_type in (None, "paper"):
        try:
            bare_trailing_outcomes.append(
                parse_lookup_target("2601.00001garbage", resource_type)
            )
        except LookupInputError as exc:
            bare_trailing_outcomes.append(exc)
    trailing_url_outcomes = []
    for resource_type in (None, "paper"):
        try:
            trailing_url_outcomes.append(
                parse_lookup_target("https://arxiv.org/abs/2601.00001garbage", resource_type)
            )
        except LookupInputError as exc:
            trailing_url_outcomes.append(exc)

    assert all(
        isinstance(outcome, LookupInputError) and outcome.error_code == "INVALID_ARGUMENTS"
        for outcome in outcomes
    )
    assert all(result.mode != "EXACT" for result in pseudo_results)
    assert all(
        isinstance(outcome, LookupInputError) and outcome.error_code == "INVALID_ARGUMENTS"
        for outcome in bare_trailing_outcomes
    )
    assert all(
        isinstance(outcome, LookupInputError) and outcome.error_code == "INVALID_ARGUMENTS"
        for outcome in trailing_url_outcomes
    )
    assert parse_lookup_target("10.1234/demo", None).kind == "doi"
    assert parse_lookup_target("hep-th/9901001v2", None).kind == "arxiv"
    assert parse_lookup_target("2601.00001v2", "paper").kind == "arxiv"
    assert parse_lookup_target("https://arxiv.org/abs/2601.00001v2", "paper").kind == "arxiv"
    assert parse_lookup_target("https://github.com/pallets/flask", None).kind == "github"
    assert parse_lookup_target("github.com/pallets/flask.git", None).kind == "github"


@pytest.mark.parametrize("resource_type", [None, "paper"])
@pytest.mark.parametrize(
    "value",
    [
        "https://user:pass@doi.org/10.1234/demo",
        "https://doi.org@evil.test/10.1234/demo",
        "https://evil.test/path/10.1234/demo",
        "ftp://doi.org/10.1234/demo",
        "ssh://doi.org/10.1234/demo",
        "javascript:10.1234/demo",
        "mailto:10.1234/demo",
        "https://user:pass@arxiv.org/abs/2601.00001",
        "https://arxiv.org@evil.test/abs/2601.00001",
        "https://evil.test/abs/2601.00001",
        "ftp://arxiv.org/abs/2601.00001",
    ],
)
def test_paper_identifier_uri_source_binding_rejects_unapproved(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = EmptySemanticConnector()
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="semantic_scholar",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize("resource_type", [None, "paper"])
@pytest.mark.parametrize(
    "value",
    [
        "2601.00001_",
        "2601.00001-",
        "2601.00001/",
        "2601.00001//",
        "2601.00001.pdf",
        "hep-th/9901001v2_",
        "hep-th/9901001v2-",
        "hep-th/9901001v2/",
        "hep-th/9901001v2//",
        "hep-th/9901001v2.pdf",
    ],
)
def test_arxiv_identifier_requires_complete_token(tmp_path, monkeypatch, resource_type, value):
    from backend.resources.query_intent import LookupInputError

    connector = EmptySemanticConnector()
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="semantic_scholar",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize(
    "value",
    [
        "10.1234/2601.00001",
        "doi:10.1234/2601.00001",
        "https://doi.org/10.1234/2601.00001",
    ],
)
def test_doi_span_does_not_create_nested_arxiv_identifier(value):
    from backend.resources.query_intent import LookupInputError

    target = parse_lookup_target(value, None)

    assert (target.mode, target.kind, target.identifier) == (
        "EXACT",
        "doi",
        "10.1234/2601.00001",
    )
    with pytest.raises(LookupInputError) as ambiguous:
        parse_lookup_target(f"{value} arXiv:2602.00002", None)
    assert ambiguous.value.error_code == "AMBIGUOUS_IDENTIFIER"


@pytest.mark.parametrize(
    ("value", "identifier"),
    [
        ("2601.00001 ", "2601.00001"),
        ("论文 2601.00001。", "2601.00001"),
        ("paper 2601.00001.", "2601.00001"),
        ("hep-th/9901001v2!", "hep-th/9901001v2"),
        ("https://arxiv.org/abs/2601.00001v2", "2601.00001v2"),
        ("https://arxiv.org/pdf/2601.00001v2.pdf", "2601.00001v2"),
    ],
)
def test_arxiv_identifier_complete_token_positive_forms(value, identifier):
    target = parse_lookup_target(value, "paper")
    assert (target.mode, target.kind, target.identifier) == ("EXACT", "arxiv", identifier)


@pytest.mark.parametrize("resource_type", [None, "paper"])
@pytest.mark.parametrize(
    "value",
    [
        "doi://evil.test/10.1234/demo",
        "arxiv://evil.test/2601.00001",
        "doi:ftp://evil.test/10.1234/demo",
        "arxiv:https://evil.test/2601.00001",
        "doi:garbage/10.1234/demo",
        "arxiv:garbage/2601.00001",
        "foo_2601.00001",
        "_2601.00001",
        "foo-2601.00001",
        "foo/2601.00001",
        "x_hep-th/9901001v2",
        "foo_10.1234/demo",
        "_10.1234/demo",
        "foo/10.1234/demo",
        "//doi.org/10.1234/demo",
        "doi.org/10.1234/demo",
        "//arxiv.org/abs/2601.00001",
        "arxiv.org/abs/2601.00001",
        "evil.test/abs/2601.00001",
        "10.1234/?",
        "10.1234/.",
        "10.1234/)",
        "doi:10.1234/?",
        "doi:10.1234/.",
        "doi:10.1234/)",
    ],
)
def test_paper_identifier_prefix_token_and_normalization_rejects_invalid_matrix(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = EmptySemanticConnector()
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="semantic_scholar",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize("resource_type", [None, "paper"])
@pytest.mark.parametrize(
    ("value", "kind", "identifier"),
    [
        ("doi:10.1234/demo", "doi", "10.1234/demo"),
        ("doi: 10.1234/demo", "doi", "10.1234/demo"),
        ("DOI:10.1234/DEMO", "doi", "10.1234/demo"),
        ("arXiv:2601.00001", "arxiv", "2601.00001"),
        ("arXiv: 2601.00001", "arxiv", "2601.00001"),
        ("ARXIV:HEP-TH/9901001V2", "arxiv", "hep-th/9901001v2"),
        ('"10.1234/demo"', "doi", "10.1234/demo"),
        ("（2601.00001）", "arxiv", "2601.00001"),
        ("doi:10.1234/demo。", "doi", "10.1234/demo"),
        ("arXiv:2601.00001！", "arxiv", "2601.00001"),
        ("https://doi.org/10.1234/demo。", "doi", "10.1234/demo"),
        ("https://arxiv.org/abs/2601.00001！", "arxiv", "2601.00001"),
        ("https://arxiv.org/pdf/2601.00001.pdf。", "arxiv", "2601.00001"),
    ],
)
def test_paper_identifier_prefix_token_and_punctuation_positive_matrix(
    resource_type,
    value,
    kind,
    identifier,
):
    target = parse_lookup_target(value, resource_type)

    assert (target.mode, target.kind, target.identifier) == ("EXACT", kind, identifier)


@pytest.mark.parametrize("resource_type", [None, "paper"])
@pytest.mark.parametrize(
    "value",
    [
        "_doi:10.1234/demo",
        "foo_doi:10.1234/demo",
        "x/doi:10.1234/demo",
        "x\\doi:10.1234/demo",
        "key=doi:10.1234/demo",
        "_arxiv:2601.00001",
        "foo_arxiv:2601.00001",
        "x/arxiv:2601.00001",
        "x\\arxiv:2601.00001",
        "key=arxiv:2601.00001",
        "_https://doi.org/10.1234/demo",
        "x/https://arxiv.org/abs/2601.00001",
        "x\\https://doi.org/10.1234/demo",
        "doi:",
        "doi: garbage",
        "doi : garbage",
        "arxiv:",
        "arxiv: garbage",
        "arxiv : garbage",
        "doi: garbage 10.1234/demo",
        "doi: arxiv:2601.00001",
        "arxiv: garbage 2601.00001",
        "arxiv: doi:10.1234/demo",
        "evil.test?doi=10.1234/demo",
        "evil.test?arxiv=2601.00001",
        "//evil.test?doi=10.1234/demo",
        "github.com/doi:10.1234/demo",
        "evil.test/path/10.1234/demo",
        "evil.test#2601.00001",
        "key=10.1234/demo",
        "path\\2601.00001",
        "doi.org/not-an-id",
        "//doi.org/not-an-id",
        "arxiv.org/abs/not-an-id",
        "//arxiv.org/abs/not-an-id",
        "2601.00001?download=1",
        "2601.00001?x=y",
        "2601.00001#fragment",
        "hep-th/9901001v2?download=1",
        "hep-th/9901001v2#fragment",
    ],
)
def test_paper_identifier_attachment_and_malformed_declaration_rejects_matrix(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = EmptySemanticConnector()
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="semantic_scholar",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize("resource_type", [None, "paper"])
@pytest.mark.parametrize(
    ("value", "kind", "identifier"),
    [
        ("doi:10.1234/demo", "doi", "10.1234/demo"),
        ("doi: 10.1234/demo", "doi", "10.1234/demo"),
        ("doi : 10.1234/demo", "doi", "10.1234/demo"),
        ("DOI : 10.1234/DEMO", "doi", "10.1234/demo"),
        ("arxiv:2601.00001", "arxiv", "2601.00001"),
        ("arxiv: 2601.00001", "arxiv", "2601.00001"),
        ("arxiv : 2601.00001", "arxiv", "2601.00001"),
        ("ARXIV : HEP-TH/9901001V2", "arxiv", "hep-th/9901001v2"),
        ("请查询doi:10.1234/demo", "doi", "10.1234/demo"),
        ("请查询arXiv:2601.00001", "arxiv", "2601.00001"),
        ("https://dx.doi.org/10.1234/demo", "doi", "10.1234/demo"),
        ("10.1234/demo，请总结", "doi", "10.1234/demo"),
        ("doi:10.1234/demo，请总结", "doi", "10.1234/demo"),
        ("arXiv:2601.00001，请总结", "arxiv", "2601.00001"),
        ("https://doi.org/10.1234/demo，请总结", "doi", "10.1234/demo"),
        ("https://arxiv.org/abs/2601.00001，请总结", "arxiv", "2601.00001"),
        ("2601.00001?请总结", "arxiv", "2601.00001"),
        ("10.1234/foo(bar);baz", "doi", "10.1234/foo(bar);baz"),
        ('"2601.00001"', "arxiv", "2601.00001"),
        ("（hep-th/9901001v2）", "arxiv", "hep-th/9901001v2"),
    ],
)
def test_paper_identifier_independent_declaration_and_natural_sentence_positive_matrix(
    resource_type,
    value,
    kind,
    identifier,
):
    target = parse_lookup_target(value, resource_type)

    assert (target.mode, target.kind, target.identifier) == ("EXACT", kind, identifier)


@pytest.mark.parametrize("resource_type", [None, "paper"])
@pytest.mark.parametrize(
    "value",
    [
        "user@10.1234/demo",
        "user@2601.00001",
        "user@doi:10.1234/demo",
        "user@arxiv:2601.00001",
        "user@https://doi.org/10.1234/demo",
        "doi: garbage 10.1234/doi:demo",
    ],
)
def test_final_parser_rejects_at_attached_paper_tokens(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = EmptySemanticConnector()
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="semantic_scholar",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize("resource_type", [None, "paper"])
@pytest.mark.parametrize(
    ("value", "identifier"),
    [
        ("10.1234/foo(bar)", "10.1234/foo(bar)"),
        ("doi:10.1234/foo[bar]", "10.1234/foo[bar]"),
        ("https://doi.org/10.1234/foo{bar}", "10.1234/foo{bar}"),
        ("10.1234/foo(bar).", "10.1234/foo(bar)"),
        ("(10.1234/foo(bar))", "10.1234/foo(bar)"),
        ("10.1234/foo[bar]，请总结", "10.1234/foo[bar]"),
        ("10.1234/doi:demo", "10.1234/doi:demo"),
        ("doi:10.1234/arxiv:2601.00001", "10.1234/arxiv:2601.00001"),
        (
            "https://doi.org/10.1234/arxiv:2601.00001",
            "10.1234/arxiv:2601.00001",
        ),
    ],
)
def test_final_parser_preserves_balanced_and_nested_doi_suffixes(
    resource_type,
    value,
    identifier,
):
    target = parse_lookup_target(value, resource_type)

    assert (target.mode, target.kind, target.identifier) == ("EXACT", "doi", identifier)


def test_supplier_doi_identity_normalization_preserves_balanced_suffix():
    from backend.resources.query_intent import LookupTarget, matches_exact_identity

    target = LookupTarget(
        mode="EXACT",
        kind="doi",
        identifier="10.1234/foo(bar)",
        source_candidates=("semantic_scholar",),
    )
    resource = {
        "external_id": "10.1234/foo(bar)",
        "raw_metadata": {"externalIds": {"DOI": "10.1234/foo(bar)"}},
    }

    assert matches_exact_identity(target, "semantic_scholar", resource) is True


@pytest.mark.parametrize(
    ("value", "explicit_type", "kind", "identifier"),
    [
        ("10.1234/CVE-2026-12345", "paper", "doi", "10.1234/cve-2026-12345"),
        ("10.1234/RFC9000", "paper", "doi", "10.1234/rfc9000"),
        ("10.1234/repo:owner/repo", "paper", "doi", "10.1234/repo:owner/repo"),
        (
            "https://github.com/CVE-2026-12345/repo",
            "project",
            "github",
            "cve-2026-12345/repo",
        ),
        (
            "https://github.com/owner/RFC9000",
            "project",
            "github",
            "owner/rfc9000",
        ),
        (
            "https://github.com/hep-th/9901001v2",
            "project",
            "github",
            "hep-th/9901001v2",
        ),
        (
            "github.com/CVE-2026-12345/repo",
            "project",
            "github",
            "cve-2026-12345/repo",
        ),
        ("repo: owner/RFC9000", "project", "github", "owner/rfc9000"),
        ("repo: hep-th/9901001v2", "project", "github", "hep-th/9901001v2"),
    ],
)
def test_final_parser_outer_exact_span_shields_nested_identifier_types(
    value,
    explicit_type,
    kind,
    identifier,
):
    implicit = parse_lookup_target(value, None)
    explicit = parse_lookup_target(value, explicit_type)

    assert (implicit.mode, implicit.kind, implicit.identifier) == ("EXACT", kind, identifier)
    assert explicit == implicit


@pytest.mark.parametrize(
    "value",
    [
        "10.1234/demo CVE-2026-12345",
        "https://github.com/owner/repo RFC9000",
    ],
)
def test_final_parser_keeps_non_overlapping_cross_type_ambiguity(value):
    from backend.resources.query_intent import LookupInputError

    with pytest.raises(LookupInputError) as ambiguous:
        parse_lookup_target(value, None)

    assert ambiguous.value.error_code == "AMBIGUOUS_IDENTIFIER"


@pytest.mark.parametrize("resource_type", [None, "project"])
@pytest.mark.parametrize(
    "value",
    [
        "evil.test?repo=github.com/owner/repo",
        "foo@github.com/owner/repo",
        "x/github.com/owner/repo",
        "key=https://github.com/owner/repo",
        "example.com/repo",
        "foo_bar/repo",
        "foo-/repo",
        "foo--bar/repo",
        "https://github.com/foo_bar/repo",
        "https://github.com/foo-/repo",
        "https://github.com/foo--bar/repo",
    ],
)
def test_final_parser_rejects_github_source_and_owner_syntax_before_dispatch(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = BatchGitHubConnector([], incomplete_results=False)
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="github",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize(
    ("value", "resource_type", "identifier"),
    [
        ("github.com/owner/repo", None, "owner/repo"),
        ("https://github.com/owner/repo", None, "owner/repo"),
        ("pallets/flask", "project", "pallets/flask"),
        ("foo-bar/repo_name", "project", "foo-bar/repo_name"),
        ("Open5GS/open5gs.git", "project", "open5gs/open5gs"),
    ],
)
def test_final_parser_keeps_valid_github_source_and_owner_forms(
    value,
    resource_type,
    identifier,
):
    target = parse_lookup_target(value, resource_type)

    assert (target.mode, target.kind, target.identifier) == ("EXACT", "github", identifier)


@pytest.mark.parametrize("resource_type", [None, "paper"])
@pytest.mark.parametrize(
    ("value", "identifier"),
    [
        ("2601.00001?", "2601.00001"),
        ("hep-th/9901001v2?", "hep-th/9901001v2"),
        ("arxiv:2601.00001?", "2601.00001"),
        ("https://arxiv.org/abs/2601.00001?", "2601.00001"),
        ("2601.00001?请总结", "2601.00001"),
        ("2601.00001? 后续自然文本 x=y", "2601.00001"),
        ("arxiv:2601.00001?请总结", "2601.00001"),
        ("https://arxiv.org/abs/2601.00001?请总结", "2601.00001"),
    ],
)
def test_final_parser_accepts_natural_arxiv_question_tails(
    resource_type,
    value,
    identifier,
):
    target = parse_lookup_target(value, resource_type)

    assert (target.mode, target.kind, target.identifier) == ("EXACT", "arxiv", identifier)


@pytest.mark.parametrize("resource_type", [None, "paper"])
@pytest.mark.parametrize(
    "value",
    [
        "2601.00001?download",
        "2601.00001?download=1",
        "2601.00001?x=y",
        "2601.00001?%64ownload",
        "2601.00001?+download",
        "2601.00001?/download",
        "2601.00001#fragment",
        "hep-th/9901001v2?download",
    ],
)
def test_final_parser_rejects_arxiv_url_query_and_fragment_tails(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = EmptySemanticConnector()
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="semantic_scholar",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize("resource_type", [None, "project"])
@pytest.mark.parametrize(
    "value",
    [
        "foo@repo:owner/repo",
        "x/repository:owner/repo",
        "notrepository:owner/repo",
        "repo: owner/repo/extra",
        "repo: owner/repo?x=y",
        "repository: owner/repo#fragment",
        "repo: owner/-repo",
        "repository: owner/re..po",
        "https://github.com/owner//repo",
        "https://github.com//owner/repo",
        "github.com/owner//repo",
        "github.com//owner/repo",
    ],
)
def test_complete_github_claim_rejects_attached_or_trailing_invalid_syntax(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = BatchGitHubConnector([], incomplete_results=False)
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="github",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize("resource_type", [None, "paper"])
@pytest.mark.parametrize(
    "value",
    [
        "foo$10.1234/demo",
        "foo$doi:10.1234/demo",
        "foo$https://doi.org/10.1234/demo",
        "foo^2601.00001",
        "foo|arxiv:2601.00001",
        "foo~https://arxiv.org/abs/2601.00001",
    ],
)
def test_identifier_start_policy_rejects_unapproved_same_token_structure(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = EmptySemanticConnector()
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="semantic_scholar",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize("resource_type", [None, "project"])
@pytest.mark.parametrize(
    "value",
    [
        "foo$github.com/owner/repo",
        "foo$https://github.com/owner/repo",
        "foo^github.com/owner/repo",
        "foo|https://github.com/owner/repo",
    ],
)
def test_github_start_policy_rejects_unapproved_same_token_structure(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = BatchGitHubConnector([], incomplete_results=False)
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="github",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize("resource_type", [None, "paper"])
@pytest.mark.parametrize(
    "value",
    [
        "2601.00001?请总结#fragment",
        "2601.00001?请总结&download=1",
        "arxiv:2601.00001?请总结#fragment",
        "arxiv:2601.00001?请总结&download=1",
        "https://arxiv.org/abs/2601.00001?请总结#fragment",
        "https://arxiv.org/abs/2601.00001?请总结&download=1",
    ],
)
def test_arxiv_natural_question_rejects_url_structure_in_adjacent_tail(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = EmptySemanticConnector()
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="semantic_scholar",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize("resource_type", [None, "project"])
@pytest.mark.parametrize(
    "value",
    [
        "foo.github.com/owner/repo",
        "foo-github.com/owner/repo",
        "foo_github.com/owner/repo",
        "agithub.com/owner/repo",
    ],
)
def test_schemeless_github_candidates_reject_attached_host_prefixes(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = BatchGitHubConnector([], incomplete_results=False)
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="github",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize("resource_type", [None, "project"])
@pytest.mark.parametrize(
    "value",
    [
        "repo: 10.1234/demo",
        "repo: 2601.00001",
        "repository: https://doi.org/10.1234/demo",
        "repository: https://arxiv.org/abs/2601.00001",
    ],
)
def test_explicit_github_context_rejects_embedded_paper_identifiers(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = BatchGitHubConnector([], incomplete_results=False)
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="github",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize(
    "value",
    [
        "owner/repo/extra",
        "owner//repo",
        "/owner/repo",
        "owner/repo/",
    ],
)
def test_explicit_project_rejects_invalid_complete_bare_repository_paths(
    tmp_path,
    monkeypatch,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = BatchGitHubConnector([], incomplete_results=False)
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type="project"),
                connector_name="github",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize("resource_type", [None, "project"])
@pytest.mark.parametrize(
    "value",
    [
        'repo: "10.1234/demo"',
        "repo: '2601.00001'",
        'repository: "https://doi.org/10.1234/demo"',
        "repository: 'https://arxiv.org/abs/2601.00001'",
        'repo: "owner/repo',
        "repo: 'owner/repo\"",
        "repo: \"owner/repo'",
        'repo: ""',
        'repo: "owner/repo"tail',
    ],
)
def test_quoted_explicit_github_context_rejects_invalid_complete_claims(
    tmp_path,
    monkeypatch,
    resource_type,
    value,
):
    from backend.resources.query_intent import LookupInputError

    connector = BatchGitHubConnector([], incomplete_results=False)
    service = _service(tmp_path, [connector], monkeypatch)

    with pytest.raises(LookupInputError) as rejected:
        asyncio.run(
            service.search(
                ResourceQuery(query=value, resource_type=resource_type),
                connector_name="github",
            )
        )

    assert rejected.value.error_code == "INVALID_ARGUMENTS"
    assert connector.calls == 0


@pytest.mark.parametrize("resource_type", [None, "project"])
@pytest.mark.parametrize(
    ("value", "identifier"),
    [
        ('repo: "owner/repo"', "owner/repo"),
        ("repository: 'owner/RFC9000'", "owner/rfc9000"),
    ],
)
def test_quoted_explicit_github_context_accepts_valid_repository_claims(
    resource_type,
    value,
    identifier,
):
    target = parse_lookup_target(value, resource_type)

    assert target.mode == "EXACT"
    assert target.kind == "github"
    assert target.identifier == identifier
    assert target.source_candidates == ("github",)


def test_historical_github_cache_is_public_fail_closed_for_search_exact_and_portal(tmp_path, monkeypatch):
    service = _service(tmp_path, [BatchGitHubConnector([], incomplete_results=False)], monkeypatch)
    query = ResourceQuery(query="teaching repository", resource_type="project")
    resources = [
        ExternalResource(
            external_id="legacy-missing",
            resource_type="project",
            source_name="github",
            title="legacy/missing",
            url="https://github.com/legacy/missing",
            raw_metadata={"full_name": "legacy/missing", "private": False},
        ),
        ExternalResource(
            external_id="legacy-private",
            resource_type="project",
            source_name="github",
            title="legacy/private",
            url="https://github.com/legacy/private",
            raw_metadata={"full_name": "legacy/private", "private": True, "visibility": "private"},
        ),
        ExternalResource(
            external_id="verified-public",
            resource_type="project",
            source_name="github",
            title="verified/public",
            url="https://github.com/verified/public",
            raw_metadata={"full_name": "verified/public", "private": False, "visibility": "public"},
        ),
    ]
    run_id = service.storage.create_connector_sync_run(
        connector_name="github",
        query=cache_query_payload(query, parse_lookup_target(query.query, query.resource_type)),
        status="RUNNING",
        cache_hit=False,
    )
    saved = service.storage.commit_verified_sync(
        run_id=run_id,
        connector_name="github",
        origin_kind="remote",
        query=cache_query_payload(query, parse_lookup_target(query.query, query.resource_type)),
        resources=[{"resource": item.model_dump(mode="json"), "content_hash": f"hash-{item.external_id}"} for item in resources],
        ttl_hours=24,
        latency_ms=0,
    )
    exact_query = ResourceQuery(query="verified/public", resource_type="project", allow_remote=False)
    service.repository.store_query_cache(
        exact_query,
        source_name="github",
        resources=saved,
        ttl_hours=24,
        lookup_target=parse_lookup_target(exact_query.query, exact_query.resource_type),
    )

    cached_search = asyncio.run(
        service.search(query.model_copy(update={"allow_remote": False}), connector_name="github")
    )
    cached_exact = asyncio.run(service.search(exact_query, connector_name="github"))
    portal = ResourcePortalService(service.storage).search(ResourcePortalQuery(resource_type="project", page_size=50))

    assert [item["external_id"] for item in cached_search["resources"]] == ["verified-public"]
    assert [item["external_id"] for item in cached_exact["resources"]] == ["verified-public"]
    assert [item["external_id"] for item in portal["items"]] == ["verified-public"]


def test_paper_auto_prefers_later_live_result_over_primary_stale_cache(tmp_path, monkeypatch):
    stale = ExternalResource(
        external_id="semantic-stale",
        resource_type="paper",
        source_name="semantic_scholar",
        title="Stale primary paper",
        url="https://example.test/semantic-stale",
    )
    seed = _service(tmp_path, [UnfilteredConnector(resources=[stale])], monkeypatch, ttl=0)
    query = ResourceQuery(query="fallback ordering", resource_type="paper")
    asyncio.run(seed.sync("semantic_scholar", query))

    live = ExternalResource(
        external_id="2601.12345v1",
        resource_type="paper",
        source_name="arxiv",
        title="Live fallback ordering paper",
        url="https://arxiv.org/abs/2601.12345v1",
        raw_metadata={"arxiv_id": "2601.12345v1"},
    )
    semantic_failure = UnfilteredConnector(fail=ConnectorServiceError("semantic unavailable"))
    arxiv_live = CountingArxivConnector(resources=[live])
    service = _service(tmp_path, [semantic_failure, arxiv_live], monkeypatch, ttl=0)
    service.storage = seed.storage
    service.repository.storage = seed.storage

    result = asyncio.run(service.search(query))

    assert result["connector"] == "arxiv"
    assert result["delivery_mode"] == "live"
    assert [item["external_id"] for item in result["resources"]] == ["2601.12345v1"]
    assert [(item["source_name"], item["status"]) for item in result["source_attempts"]] == [
        ("semantic_scholar", "ERROR"),
        ("arxiv", "FOUND"),
    ]
    assert arxiv_live.calls == 1

    all_failed = _service(tmp_path / "all-failed", [UnfilteredConnector(resources=[stale])], monkeypatch, ttl=0)
    asyncio.run(all_failed.sync("semantic_scholar", query))
    failing = _service(
        tmp_path / "all-failed",
        [
            UnfilteredConnector(fail=ConnectorServiceError("semantic unavailable")),
            ArxivFakeConnector(fail=ConnectorServiceError("arxiv unavailable")),
        ],
        monkeypatch,
        ttl=0,
    )
    failing.storage = all_failed.storage
    failing.repository.storage = all_failed.storage
    stale_result = asyncio.run(failing.search(query))
    assert stale_result["connector"] == "semantic_scholar"
    assert stale_result["delivery_mode"] == "stale_cache"
    assert [item["external_id"] for item in stale_result["resources"]] == ["semantic-stale"]


def test_auto_search_orders_fresh_and_live_within_each_source(tmp_path, monkeypatch):
    query = ResourceQuery(query="ordered source paper", resource_type="paper")
    cached_arxiv = ExternalResource(
        external_id="2601.20001v1",
        resource_type="paper",
        source_name="arxiv",
        title="Ordered source paper cached fallback",
        url="https://arxiv.org/abs/2601.20001v1",
        raw_metadata={"arxiv_id": "2601.20001v1"},
    )
    seed = _service(tmp_path / "paper", [UnfilteredArxivConnector(resources=[cached_arxiv])], monkeypatch)
    asyncio.run(seed.sync("arxiv", query))

    semantic_resource = ExternalResource(
        external_id="semantic-live",
        resource_type="paper",
        source_name="semantic_scholar",
        title="Ordered source paper live primary",
        url="https://example.test/semantic-live",
    )
    semantic = CountingSemanticConnector([semantic_resource])
    paper_service = _service(
        tmp_path / "paper",
        [semantic, EmptyArxivConnector()],
        monkeypatch,
    )
    paper_service.storage = seed.storage
    paper_service.repository.storage = seed.storage

    paper_result = asyncio.run(paper_service.search(query))

    rfc_query = ResourceQuery(query="RFC 9000", resource_type="standard")
    local = StandardSourceConnector(knowledge_dir=tmp_path / "rfc")
    local_seed = _service(tmp_path / "rfc", [local], monkeypatch)
    asyncio.run(local_seed.sync("standard_source", rfc_query))
    remote = CountingRFCConnector()
    rfc_service = _service(tmp_path / "rfc", [remote, local], monkeypatch)
    rfc_service.storage = local_seed.storage
    rfc_service.repository.storage = local_seed.storage

    rfc_result = asyncio.run(rfc_service.search(rfc_query))

    assert semantic.calls == 1
    assert paper_result["connector"] == "semantic_scholar"
    assert paper_result["delivery_mode"] == "live"
    assert [item["external_id"] for item in paper_result["resources"]] == ["semantic-live"]
    assert remote.calls == 1
    assert rfc_result["connector"] == "rfc_editor"
    assert rfc_result["delivery_mode"] == "live"


def test_remote_rate_hook_counts_only_actual_remote_attempts(tmp_path, monkeypatch):
    query = ResourceQuery(query="remote count", resource_type="paper")
    calls: list[str] = []

    disabled = _service(tmp_path / "disabled", [DisabledConnector()], monkeypatch)
    disabled_result = asyncio.run(
        disabled.search(query, connector_name="disabled_fake", before_remote=calls.append)
    )
    disabled_calls = list(calls)
    calls.clear()

    missing = _service(tmp_path / "missing", [], monkeypatch)
    missing_result = asyncio.run(
        missing.search(query, connector_name="missing", before_remote=calls.append)
    )
    missing_calls = list(calls)
    calls.clear()

    local = _service(
        tmp_path / "local",
        [StandardSourceConnector(knowledge_dir=tmp_path / "local")],
        monkeypatch,
    )
    local_result = asyncio.run(
        local.search(
            ResourceQuery(query="RFC 9000", resource_type="standard"),
            connector_name="standard_source",
            before_remote=calls.append,
        )
    )
    local_calls = list(calls)
    calls.clear()

    semantic = EmptySemanticConnector()
    arxiv = CountingArxivConnector(
        [
            ExternalResource(
                external_id="2601.30001v1",
                resource_type="paper",
                source_name="arxiv",
                title="Remote count",
                url="https://arxiv.org/abs/2601.30001v1",
                raw_metadata={"arxiv_id": "2601.30001v1"},
            )
        ]
    )
    remote = _service(tmp_path / "remote", [semantic, arxiv], monkeypatch)
    remote_result = asyncio.run(remote.search(query, before_remote=calls.append))

    assert disabled_result["lookup_state"] == "UNAVAILABLE"
    assert disabled_calls == []
    assert missing_result["lookup_state"] == "UNAVAILABLE"
    assert missing_calls == []
    assert local_result["lookup_state"] == "FOUND"
    assert local_calls == []
    assert remote_result["lookup_state"] == "FOUND"
    assert calls == ["semantic_scholar", "arxiv"]


def test_expired_deadline_does_not_consume_remote_quota_or_call_connector(tmp_path, monkeypatch):
    resource = ExternalResource(
        external_id="deadline-paper",
        resource_type="paper",
        source_name="semantic_scholar",
        title="Deadline paper",
        url="https://example.test/deadline-paper",
    )
    expired_connector = CountingSemanticConnector([resource])
    expired_service = _service(tmp_path / "expired", [expired_connector], monkeypatch)
    expired_rate_calls: list[str] = []

    expired = asyncio.run(
        expired_service.search(
            ResourceQuery(query="deadline paper", resource_type="paper"),
            connector_name="semantic_scholar",
            deadline=time.monotonic() - 1.0,
            before_remote=expired_rate_calls.append,
        )
    )

    valid_connector = CountingSemanticConnector([resource])
    valid_service = _service(tmp_path / "valid", [valid_connector], monkeypatch)
    valid_rate_calls: list[str] = []
    valid = asyncio.run(
        valid_service.search(
            ResourceQuery(query="deadline paper", resource_type="paper"),
            connector_name="semantic_scholar",
            deadline=time.monotonic() + 5.0,
            before_remote=valid_rate_calls.append,
        )
    )

    assert expired_rate_calls == []
    assert expired_connector.calls == 0
    assert expired["lookup_state"] == "UNAVAILABLE"
    assert expired["upstream_error_code"] == "CONNECTOR_TIMEOUT"
    assert valid_rate_calls == ["semantic_scholar"]
    assert valid_connector.calls == 1
    assert valid["lookup_state"] == "FOUND"


def test_rfc_exact_falls_back_to_local_catalog_when_remote_is_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_REAL_RFC_LOOKUP", "false")
    local = StandardSourceConnector(knowledge_dir=tmp_path)
    remote = RFCEditorConnector(base_url="https://rfc.invalid")
    service = _service(tmp_path, [remote, local], monkeypatch)
    query = ResourceQuery(query="RFC 9000", resource_type="standard")

    automatic = asyncio.run(service.search(query))
    explicit_local = asyncio.run(service.search(query, connector_name="standard_source"))

    assert automatic["connector"] == "standard_source"
    assert automatic["delivery_mode"] == "local_metadata"
    assert automatic["origin_kind"] == "local_catalog"
    assert automatic["resources"][0]["external_id"] == "RFC-9000"
    assert [(item["source_name"], item["status"]) for item in automatic["source_attempts"]] == [
        ("rfc_editor", "DISABLED"),
        ("standard_source", "FOUND"),
    ]
    assert explicit_local["lookup_state"] == "FOUND"
    assert explicit_local["connector"] == "standard_source"


def test_paper_empty_primary_falls_back_and_complete_empty_is_not_cached(tmp_path, monkeypatch):
    empty_primary = EmptySemanticConnector()
    paper = ExternalResource(
        external_id="2601.00001v1",
        resource_type="paper",
        source_name="arxiv",
        title="Fallback topic paper",
        abstract="Fallback metadata",
        url="https://arxiv.org/abs/2601.00001v1",
        raw_metadata={"arxiv_id": "2601.00001v1"},
    )
    fallback = CountingArxivConnector(resources=[paper])
    service = _service(tmp_path, [empty_primary, fallback], monkeypatch)

    found = asyncio.run(service.search(ResourceQuery(query="fallback topic", resource_type="paper")))

    assert found["lookup_state"] == "FOUND"
    assert found["connector"] == "arxiv"
    assert [item["source_name"] for item in found["resources"]] == ["arxiv"]
    assert [(item["source_name"], item["status"]) for item in found["source_attempts"]] == [
        ("semantic_scholar", "EMPTY"),
        ("arxiv", "FOUND"),
    ]
    assert empty_primary.calls == 1
    assert fallback.calls == 1

    empty_fallback = EmptyArxivConnector()
    empty_service = _service(tmp_path / "empty", [EmptySemanticConnector(), empty_fallback], monkeypatch)
    first = asyncio.run(empty_service.search(ResourceQuery(query="no such paper", resource_type="paper")))
    second = asyncio.run(empty_service.search(ResourceQuery(query="no such paper", resource_type="paper")))
    with empty_service.storage._connect() as conn:
        cache_rows = conn.execute("SELECT COUNT(*) FROM resource_query_cache").fetchone()[0]

    assert first["lookup_state"] == "NOT_FOUND"
    assert second["lookup_state"] == "NOT_FOUND"
    assert cache_rows == 0


def test_exact_arxiv_continues_past_semantic_identity_without_official_url(tmp_path, monkeypatch):
    semantic = CountingSemanticConnector([
        ExternalResource(
            external_id="1706.03762",
            resource_type="paper",
            source_name="semantic_scholar",
            title="Attention Is All You Need",
            abstract="Metadata from the primary paper catalog.",
            url="https://www.semanticscholar.org/paper/demo",
            raw_metadata={"externalIds": {"ArXiv": "1706.03762"}},
        )
    ])
    arxiv = CountingUnfilteredArxivConnector([
        ExternalResource(
            external_id="1706.03762v7",
            resource_type="paper",
            source_name="arxiv",
            title="Attention Is All You Need",
            abstract="Trusted arXiv metadata.",
            url="https://arxiv.org/abs/1706.03762v7",
            raw_metadata={"arxiv_id": "1706.03762v7"},
        )
    ])
    service = _service(tmp_path, [semantic, arxiv], monkeypatch)

    result = asyncio.run(service.search(ResourceQuery(
        query="arXiv:1706.03762", resource_type="paper", allow_remote=True,
    )))

    assert semantic.calls == 1
    assert arxiv.calls == 1
    assert result["connector"] == "arxiv"
    assert [item["source_name"] for item in result["resources"]] == ["arxiv"]
    assert [item["external_id"] for item in result["resources"]] == ["1706.03762v7"]
    assert [item["url"] for item in result["resources"]] == ["https://arxiv.org/abs/1706.03762v7"]
    assert [(item["source_name"], item["status"]) for item in result["source_attempts"]] == [
        ("semantic_scholar", "FOUND"),
        ("arxiv", "FOUND"),
    ]


def test_exact_arxiv_fresh_semantic_without_official_url_continues_to_arxiv(tmp_path, monkeypatch):
    query = ResourceQuery(query="arXiv:1706.03762", resource_type="paper", allow_remote=True)
    semantic = CountingSemanticConnector([
        ExternalResource(
            external_id="1706.03762",
            resource_type="paper",
            source_name="semantic_scholar",
            title="Attention Is All You Need",
            abstract="Cached metadata.",
            url="https://www.semanticscholar.org/paper/demo",
            raw_metadata={"externalIds": {"ArXiv": "1706.03762"}},
        )
    ])
    seed = _service(tmp_path, [semantic], monkeypatch)
    asyncio.run(seed.sync("semantic_scholar", query))

    arxiv = CountingUnfilteredArxivConnector([
        ExternalResource(
            external_id="1706.03762v7",
            resource_type="paper",
            source_name="arxiv",
            title="Attention Is All You Need",
            abstract="Trusted arXiv metadata.",
            url="https://arxiv.org/abs/1706.03762v7",
            raw_metadata={"arxiv_id": "1706.03762v7"},
        )
    ])
    service = _service(tmp_path, [CountingSemanticConnector([]), arxiv], monkeypatch)
    service.storage = seed.storage
    service.repository.storage = seed.storage

    result = asyncio.run(service.search(query))

    assert arxiv.calls == 1
    assert result["connector"] == "arxiv"
    assert [item["url"] for item in result["resources"]] == ["https://arxiv.org/abs/1706.03762v7"]
    assert [(item["source_name"], item["delivery_mode"]) for item in result["source_attempts"]] == [
        ("semantic_scholar", "fresh_cache"),
        ("arxiv", "live"),
    ]


def test_exact_arxiv_semantic_official_url_can_finish_without_arxiv_call(tmp_path, monkeypatch):
    semantic = CountingSemanticConnector([
        ExternalResource(
            external_id="1706.03762",
            resource_type="paper",
            source_name="semantic_scholar",
            title="Attention Is All You Need",
            abstract="Semantic Scholar metadata with an official landing URL.",
            url="https://arxiv.org/abs/1706.03762",
            raw_metadata={"externalIds": {"ArXiv": "1706.03762"}},
        )
    ])
    arxiv = CountingUnfilteredArxivConnector([])
    service = _service(tmp_path, [semantic, arxiv], monkeypatch)

    result = asyncio.run(service.search(ResourceQuery(
        query="arXiv:1706.03762", resource_type="paper", allow_remote=True,
    )))

    assert semantic.calls == 1
    assert arxiv.calls == 0
    assert result["connector"] == "semantic_scholar"
    assert result["resources"][0]["source_name"] == "semantic_scholar"
    assert result["resources"][0]["url"] == "https://arxiv.org/abs/1706.03762"


def test_exact_arxiv_live_semantic_without_official_url_uses_fresh_arxiv(tmp_path, monkeypatch):
    query = ResourceQuery(query="arXiv:1706.03762", resource_type="paper", allow_remote=True)
    cached_arxiv = ExternalResource(
        external_id="1706.03762v7",
        resource_type="paper",
        source_name="arxiv",
        title="Attention Is All You Need",
        abstract="Trusted cached arXiv metadata.",
        url="https://arxiv.org/abs/1706.03762v7",
        raw_metadata={"arxiv_id": "1706.03762v7"},
    )
    seed = _service(tmp_path, [CountingUnfilteredArxivConnector([cached_arxiv])], monkeypatch)
    asyncio.run(seed.sync("arxiv", query))

    semantic = CountingSemanticConnector([
        ExternalResource(
            external_id="1706.03762",
            resource_type="paper",
            source_name="semantic_scholar",
            title="Attention Is All You Need",
            abstract="Semantic metadata without the official landing URL.",
            url="https://www.semanticscholar.org/paper/demo",
            raw_metadata={"externalIds": {"ArXiv": "1706.03762"}},
        )
    ])
    arxiv = CountingUnfilteredArxivConnector([])
    service = _service(tmp_path, [semantic, arxiv], monkeypatch)
    service.storage = seed.storage
    service.repository.storage = seed.storage

    result = asyncio.run(service.search(query))

    assert semantic.calls == 1
    assert arxiv.calls == 0
    assert result["connector"] == "arxiv"
    assert result["delivery_mode"] == "fresh_cache"
    assert [item["url"] for item in result["resources"]] == ["https://arxiv.org/abs/1706.03762v7"]
    assert [(item["source_name"], item["delivery_mode"]) for item in result["source_attempts"]] == [
        ("semantic_scholar", "live"),
        ("arxiv", "fresh_cache"),
    ]


def test_exact_arxiv_weak_semantic_is_not_delivered_when_arxiv_unavailable(tmp_path, monkeypatch):
    semantic = CountingSemanticConnector([
        ExternalResource(
            external_id="1706.03762",
            resource_type="paper",
            source_name="semantic_scholar",
            title="Attention Is All You Need",
            abstract="Metadata without an official landing URL.",
            url="https://www.semanticscholar.org/paper/demo",
            raw_metadata={"externalIds": {"ArXiv": "1706.03762"}},
        )
    ])
    arxiv = ArxivFakeConnector(fail=ConnectorServiceError("arxiv unavailable"))
    service = _service(tmp_path, [semantic, arxiv], monkeypatch)

    result = asyncio.run(service.search(ResourceQuery(
        query="arXiv:1706.03762", resource_type="paper", allow_remote=True,
    )))

    assert semantic.calls == 1
    assert result["resources"] == []
    assert result["lookup_state"] == "UNAVAILABLE"
    assert [item["source_name"] for item in result["source_attempts"]] == ["semantic_scholar", "arxiv"]


@pytest.mark.parametrize(
    ("requested", "returned", "url", "overrides", "expected"),
    [
        ("1706.03762", "1706.03762v7", "https://arxiv.org/abs/1706.03762v7", {}, True),
        ("1706.03762v7", "1706.03762v7", "https://arxiv.org/abs/1706.03762v7", {}, True),
        ("1706.03762v6", "1706.03762v7", "https://arxiv.org/abs/1706.03762v7", {}, False),
        ("1706.03762", "1706.99999", "https://arxiv.org/abs/1706.99999", {}, False),
        ("1706.03762", "1706.03762", "http://arxiv.org/abs/1706.03762", {}, False),
        ("1706.03762", "1706.03762", "https://evil.test/abs/1706.03762", {}, False),
        ("1706.03762", "1706.03762", "https://user@arxiv.org/abs/1706.03762", {}, False),
        ("1706.03762", "1706.03762", "https://arxiv.org:8443/abs/1706.03762", {}, False),
        ("1706.03762", "1706.03762", "https://arxiv.org/pdf/1706.03762", {}, False),
        ("1706.03762", "1706.03762", "https://arxiv.org/abs/1706.03762?download=1", {}, False),
        ("1706.03762", "1706.03762", "https://arxiv.org/abs/1706.03762#paper", {}, False),
        ("1706.03762", "1706.03762", "https://arxiv.org/abs/1706.03762", {"content_hash": ""}, False),
        ("1706.03762", "1706.03762", "https://arxiv.org/abs/1706.03762", {"provenance_verified": False}, False),
        ("1706.03762", "1706.03762", "https://arxiv.org/abs/1706.03762", {"origin_kind": "local_catalog"}, False),
    ],
)
def test_exact_arxiv_deliverable_identity_is_strict(requested, returned, url, overrides, expected):
    target = parse_lookup_target(f"arXiv:{requested}", "paper")
    resource = {
        "external_id": returned,
        "resource_type": "paper",
        "source_name": "arxiv",
        "url": url,
        "raw_metadata": {"arxiv_id": returned},
        "content_hash": "trusted-hash",
        "provenance_verified": True,
        "origin_kind": "remote",
        **overrides,
    }

    assert is_deliverable_exact_arxiv_resource(target, resource) is expected


def _internal_response_keys(value, prefix=""):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).startswith("_"):
                found.append(path)
            found.extend(_internal_response_keys(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_internal_response_keys(item, f"{prefix}[{index}]"))
    return found


def _seed_exact_arxiv_cache(service, query, *source_names):
    for source_name in source_names:
        result = asyncio.run(service.sync(source_name, query))
        assert result.status == "SUCCESS"


def _weak_semantic_arxiv_resource(*, official_url=False):
    return ExternalResource(
        external_id="1706.03762",
        resource_type="paper",
        source_name="semantic_scholar",
        title="Attention Is All You Need",
        abstract="Cached catalog metadata.",
        url=(
            "https://arxiv.org/abs/1706.03762"
            if official_url
            else "https://www.semanticscholar.org/paper/demo"
        ),
        raw_metadata={"externalIds": {"ArXiv": "1706.03762"}},
    )


def _official_arxiv_cache_resource(*, url="https://arxiv.org/abs/1706.03762v7"):
    return ExternalResource(
        external_id="1706.03762v7",
        resource_type="paper",
        source_name="arxiv",
        title="Attention Is All You Need",
        abstract="Trusted cached arXiv metadata.",
        url=url,
        raw_metadata={"arxiv_id": "1706.03762v7"},
    )


def test_exact_arxiv_cache_only_skips_weak_semantic_and_delivers_official_arxiv(tmp_path, monkeypatch):
    query = ResourceQuery(query="arXiv:1706.03762", resource_type="paper", allow_remote=False)
    semantic = CountingSemanticConnector([_weak_semantic_arxiv_resource()])
    arxiv = CountingUnfilteredArxivConnector([_official_arxiv_cache_resource()])
    service = _service(tmp_path, [semantic, arxiv], monkeypatch)
    _seed_exact_arxiv_cache(service, query, "semantic_scholar", "arxiv")
    semantic.calls = 0
    arxiv.calls = 0
    remote_hooks = []

    result = asyncio.run(service.search(query, before_remote=remote_hooks.append))

    assert semantic.calls == arxiv.calls == 0
    assert remote_hooks == []
    assert result["cache_hit"] is True
    assert result["status"] == "SUCCESS"
    assert result["delivery_mode"] == "fresh_cache"
    assert result["lookup_state"] == "FOUND"
    assert result["canonical_identifier"] == "1706.03762"
    assert result["connector"] == "arxiv"
    assert result["connectors_tried"] == []
    assert [item["source_name"] for item in result["resources"]] == ["arxiv"]
    assert [item["external_id"] for item in result["resources"]] == ["1706.03762v7"]
    assert [item["url"] for item in result["resources"]] == ["https://arxiv.org/abs/1706.03762v7"]
    assert [(item["source_name"], item["delivery_mode"]) for item in result["source_attempts"]] == [
        ("semantic_scholar", "fresh_cache"),
        ("arxiv", "fresh_cache"),
    ]
    assert _internal_response_keys(result) == []


def test_exact_arxiv_cache_only_weak_semantic_returns_public_miss(tmp_path, monkeypatch):
    query = ResourceQuery(query="arXiv:1706.03762", resource_type="paper", allow_remote=False)
    semantic = CountingSemanticConnector([_weak_semantic_arxiv_resource()])
    arxiv = CountingUnfilteredArxivConnector([])
    service = _service(tmp_path, [semantic, arxiv], monkeypatch)
    _seed_exact_arxiv_cache(service, query, "semantic_scholar")
    semantic.calls = 0
    arxiv.calls = 0
    remote_hooks = []

    result = asyncio.run(service.search(query, before_remote=remote_hooks.append))

    assert semantic.calls == arxiv.calls == 0
    assert remote_hooks == []
    assert result["cache_hit"] is False
    assert result["status"] == "CACHE_MISS"
    assert result["delivery_mode"] == "cache_miss"
    assert result["lookup_state"] == "NOT_CHECKED"
    assert result["resources"] == []
    assert result["connectors_tried"] == []
    assert [item["source_name"] for item in result["source_attempts"]] == ["semantic_scholar"]
    assert _internal_response_keys(result) == []


@pytest.mark.parametrize("invalid_kind", ["wrong_identifier", "non_remote", "missing_hash", "nonofficial_url"])
def test_exact_arxiv_cache_only_rejects_untrusted_followup_cache(tmp_path, monkeypatch, invalid_kind):
    query = ResourceQuery(query="arXiv:1706.03762", resource_type="paper", allow_remote=False)
    semantic = CountingSemanticConnector([_weak_semantic_arxiv_resource()])
    arxiv = CountingUnfilteredArxivConnector([
        _official_arxiv_cache_resource(
            url=(
                "https://www.semanticscholar.org/paper/arxiv-shaped"
                if invalid_kind == "nonofficial_url"
                else "https://arxiv.org/abs/1706.03762v7"
            )
        )
    ])
    service = _service(tmp_path, [semantic, arxiv], monkeypatch)
    _seed_exact_arxiv_cache(service, query, "semantic_scholar", "arxiv")
    with service.storage._connect() as conn:
        if invalid_kind == "wrong_identifier":
            conn.execute(
                "UPDATE external_resources SET external_id=?, raw_metadata_json=? WHERE source_name='arxiv'",
                ("1706.99999", json.dumps({"arxiv_id": "1706.99999"})),
            )
        elif invalid_kind == "non_remote":
            conn.execute("UPDATE resource_provenance SET origin_kind='local_catalog' WHERE connector_name='arxiv'")
        elif invalid_kind == "missing_hash":
            conn.execute("UPDATE external_resources SET content_hash='' WHERE source_name='arxiv'")
            conn.execute("UPDATE resource_provenance SET content_hash='' WHERE connector_name='arxiv'")
    semantic.calls = 0
    arxiv.calls = 0
    remote_hooks = []

    result = asyncio.run(service.search(query, before_remote=remote_hooks.append))

    assert semantic.calls == arxiv.calls == 0
    assert remote_hooks == []
    assert result["status"] == "CACHE_MISS"
    assert result["delivery_mode"] == "cache_miss"
    assert result["lookup_state"] == "NOT_CHECKED"
    assert result["resources"] == []
    assert result["connectors_tried"] == []
    assert _internal_response_keys(result) == []


def test_exact_arxiv_cache_only_semantic_official_url_keeps_source_identity(tmp_path, monkeypatch):
    query = ResourceQuery(query="arXiv:1706.03762", resource_type="paper", allow_remote=False)
    semantic = CountingSemanticConnector([_weak_semantic_arxiv_resource(official_url=True)])
    arxiv = CountingUnfilteredArxivConnector([])
    service = _service(tmp_path, [semantic, arxiv], monkeypatch)
    _seed_exact_arxiv_cache(service, query, "semantic_scholar")
    semantic.calls = 0
    arxiv.calls = 0

    result = asyncio.run(service.search(query))

    assert semantic.calls == arxiv.calls == 0
    assert result["cache_hit"] is True
    assert result["connector"] == "semantic_scholar"
    assert result["delivery_mode"] == "fresh_cache"
    assert result["lookup_state"] == "FOUND"
    assert result["resources"][0]["source_name"] == "semantic_scholar"
    assert result["resources"][0]["url"] == "https://arxiv.org/abs/1706.03762"
    assert result["connectors_tried"] == []
    assert _internal_response_keys(result) == []


def test_exact_arxiv_cache_only_explicit_source_does_not_cross_sources(tmp_path, monkeypatch):
    query = ResourceQuery(query="arXiv:1706.03762", resource_type="paper", allow_remote=False)
    semantic = CountingSemanticConnector([_weak_semantic_arxiv_resource()])
    arxiv = CountingUnfilteredArxivConnector([_official_arxiv_cache_resource()])
    service = _service(tmp_path, [semantic, arxiv], monkeypatch)
    _seed_exact_arxiv_cache(service, query, "semantic_scholar", "arxiv")
    semantic.calls = 0
    arxiv.calls = 0

    weak = asyncio.run(service.search(query, connector_name="semantic_scholar"))
    official = asyncio.run(service.search(query, connector_name="arxiv"))

    assert semantic.calls == arxiv.calls == 0
    assert weak["status"] == "CACHE_MISS"
    assert weak["connector"] == "semantic_scholar"
    assert weak["resources"] == []
    assert [item["source_name"] for item in weak["source_attempts"]] == ["semantic_scholar"]
    assert official["status"] == "SUCCESS"
    assert official["connector"] == "arxiv"
    assert [item["source_name"] for item in official["resources"]] == ["arxiv"]
    assert _internal_response_keys(weak) == _internal_response_keys(official) == []


@pytest.mark.parametrize(
    ("requested", "expect_found"),
    [("1706.03762v7", True), ("1706.03762v6", False)],
)
def test_exact_arxiv_cache_only_explicit_version_remains_strict(tmp_path, monkeypatch, requested, expect_found):
    query = ResourceQuery(query=f"arXiv:{requested}", resource_type="paper", allow_remote=False)
    arxiv = CountingUnfilteredArxivConnector([_official_arxiv_cache_resource()])
    service = _service(tmp_path, [arxiv], monkeypatch)
    _seed_exact_arxiv_cache(service, query, "arxiv")
    arxiv.calls = 0

    result = asyncio.run(service.search(query, connector_name="arxiv"))

    assert arxiv.calls == 0
    assert result["connectors_tried"] == []
    assert _internal_response_keys(result) == []
    if expect_found:
        assert result["status"] == "SUCCESS"
        assert result["canonical_identifier"] == "1706.03762v7"
        assert [item["external_id"] for item in result["resources"]] == ["1706.03762v7"]
    else:
        assert result["status"] == "CACHE_MISS"
        assert result["lookup_state"] == "NOT_CHECKED"
        assert result["resources"] == []


def test_resource_search_cache_read_requires_login_and_live_rate_limit_has_retry_after(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-1a-resource-rate-limit")
    original_logger = main.learning_logger
    original_registry = main.connector_registry
    resource = ExternalResource(
        external_id="rate-paper",
        resource_type="paper",
        source_name="arxiv",
        title="Rate limited paper fixture",
        url="https://arxiv.org/abs/rate-paper",
    )
    main.learning_logger = LearningLogger(tmp_path / "rate-api.db")
    main.learning_logger.seed_demo_identity()
    main.connector_registry = ConnectorRegistry([UnfilteredArxivConnector(resources=[resource])])
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_MAX_CALLS", 1)
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_WINDOW_SECONDS", 60.0)
    main._RESOURCE_RATE_BUCKETS.clear()
    client = TestClient(main.app)
    try:
        anonymous_cache = client.post(
            "/api/resources/search?connector_name=arxiv",
            json={"query": "cached only", "resource_type": "paper", "allow_remote": False},
        )
        token = _login_token(client, "student_demo")
        first = client.post(
            "/api/resources/search?connector_name=arxiv",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "first live query", "resource_type": "paper", "allow_remote": True},
        )
        cached_again = client.post(
            "/api/resources/search?connector_name=arxiv",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "first live query", "resource_type": "paper", "allow_remote": True},
        )
        limited = client.post(
            "/api/resources/search?connector_name=arxiv",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "second distinct live query", "resource_type": "paper", "allow_remote": True},
        )
        github_too_long = client.post(
            "/api/resources/search?connector_name=github",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "x" * 257, "resource_type": "project", "allow_remote": True},
        )
        outer_too_long = client.post(
            "/api/resources/search?connector_name=github",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "x" * 301, "resource_type": "project", "allow_remote": True},
        )

        assert anonymous_cache.status_code == 401
        assert anonymous_cache.json()["error_code"] == "AUTH_REQUIRED"
        assert first.status_code == 200
        assert cached_again.status_code == 200
        assert cached_again.json()["cache_hit"] is True
        assert limited.status_code == 429
        assert limited.json()["error_code"] == "RESOURCE_RATE_LIMIT"
        assert int(limited.headers["Retry-After"]) >= 1
        runs = main.learning_logger.connector_sync_runs(limit=100)
        assert len(runs) == 1
        assert all(run["status"] != "RUNNING" for run in runs)
        assert github_too_long.status_code == 400
        assert github_too_long.json()["error_code"] == "GITHUB_QUERY_TOO_LONG"
        assert outer_too_long.status_code == 422
    finally:
        main._RESOURCE_RATE_BUCKETS.clear()
        main.learning_logger = original_logger
        main.connector_registry = original_registry


def test_exact_cache_uses_canonical_identity_and_never_falls_back_to_similar_search_rows(tmp_path, monkeypatch):
    exact = ExternalResource(
        external_id="CVE-2026-12345",
        resource_type="vulnerability",
        source_name="nvd",
        title="CVE-2026-12345",
        url="https://nvd.nist.gov/vuln/detail/CVE-2026-12345",
    )
    connector = CountingNVDConnector([exact])
    service = _service(tmp_path / "exact", [connector], monkeypatch)

    live = asyncio.run(
        service.search(
            ResourceQuery(query="Explain CVE-2026-12345", resource_type="vulnerability"),
            connector_name="nvd",
        )
    )
    cached = asyncio.run(
        service.search(
            ResourceQuery(query="cve-2026-12345", resource_type="vulnerability", allow_remote=False),
            connector_name="nvd",
        )
    )

    assert live["lookup_mode"] == "EXACT"
    assert live["lookup_state"] == "FOUND"
    assert cached["cache_hit"] is True
    assert cached["canonical_identifier"] == "CVE-2026-12345"
    assert connector.calls == 1

    similar = exact.model_copy(
        update={
            "external_id": "CVE-2026-99999",
            "title": "CVE-2026-99999 mentions CVE-2026-77777",
            "url": "https://nvd.nist.gov/vuln/detail/CVE-2026-99999",
        }
    )
    old_service = _service(tmp_path / "old-search", [UnfilteredNVDConnector(resources=[similar])], monkeypatch)
    asyncio.run(
        old_service.sync(
            "nvd",
            ResourceQuery(query="CVE security catalogue", resource_type="vulnerability"),
        )
    )
    miss = asyncio.run(
        old_service.search(
            ResourceQuery(query="CVE-2026-77777", resource_type="vulnerability", allow_remote=False),
            connector_name="nvd",
        )
    )
    assert miss["cache_hit"] is False
    assert miss["resources"] == []
    assert miss["lookup_state"] == "NOT_CHECKED"


def test_github_partial_query_state_is_cached_only_with_verified_items(tmp_path, monkeypatch):
    project = ExternalResource(
        external_id="123",
        resource_type="project",
        source_name="github",
        title="demo/repo",
        url="https://github.com/demo/repo",
        raw_metadata={"full_name": "demo/repo", "private": False, "visibility": "public"},
    )
    partial_connector = BatchGitHubConnector([project], incomplete_results=True)
    service = _service(tmp_path / "partial-items", [partial_connector], monkeypatch)
    query = ResourceQuery(query="secure teaching repository", resource_type="project")

    first = asyncio.run(service.search(query, connector_name="github"))
    cached = asyncio.run(
        service.search(query.model_copy(update={"allow_remote": False}), connector_name="github")
    )

    assert first["lookup_state"] == "PARTIAL"
    assert first["incomplete_results"] is True
    assert cached["lookup_state"] == "PARTIAL"
    assert cached["incomplete_results"] is True
    assert partial_connector.calls == 1

    empty_connector = BatchGitHubConnector([], incomplete_results=True)
    empty_service = _service(tmp_path / "partial-empty", [empty_connector], monkeypatch)
    empty_query = ResourceQuery(query="no returned repositories", resource_type="project")
    empty_first = asyncio.run(empty_service.search(empty_query, connector_name="github"))
    empty_second = asyncio.run(empty_service.search(empty_query, connector_name="github"))
    with empty_service.storage._connect() as conn:
        cache_rows = conn.execute("SELECT COUNT(*) FROM resource_query_cache").fetchone()[0]

    assert empty_first["lookup_state"] == "PARTIAL"
    assert empty_second["lookup_state"] == "PARTIAL"
    assert empty_connector.calls == 2
    assert cache_rows == 0

    private = project.model_copy(
        update={
            "external_id": "private-123",
            "raw_metadata": {"full_name": "demo/private", "private": True, "visibility": "private"},
        }
    )
    private_service = _service(
        tmp_path / "private",
        [BatchGitHubConnector([private], incomplete_results=False)],
        monkeypatch,
    )
    rejected = asyncio.run(
        private_service.search(
            ResourceQuery(query="private teaching repository", resource_type="project"),
            connector_name="github",
        )
    )
    with private_service.storage._connect() as conn:
        private_cache_rows = conn.execute("SELECT COUNT(*) FROM resource_query_cache").fetchone()[0]
    assert rejected["lookup_state"] == "UNAVAILABLE"
    assert rejected["error"] == "GITHUB_REPOSITORY_NOT_PUBLIC"
    assert private_cache_rows == 0


def test_exact_filter_conflict_and_complete_empty_state_are_explicit(tmp_path, monkeypatch):
    from backend.resources.query_intent import LookupInputError

    service = _service(tmp_path, [EmptySemanticConnector()], monkeypatch)
    with pytest.raises(LookupInputError) as conflict:
        asyncio.run(
            service.search(
                ResourceQuery(query="doi:10.1234/demo", resource_type="paper", page=2),
                connector_name="semantic_scholar",
            )
        )
    assert conflict.value.error_code == "EXACT_QUERY_FILTER_CONFLICT"

    empty = asyncio.run(
        service.search(
            ResourceQuery(query="unpublished topic", resource_type="paper"),
            connector_name="semantic_scholar",
        )
    )
    assert empty["lookup_state"] == "NOT_FOUND"
    assert empty["source_attempts"][0]["status"] == "EMPTY"


def test_fake_connector_contract_syncs_and_searches_cache(tmp_path, monkeypatch):
    vector_store = RecordingVectorStore()
    service = _service(tmp_path, [FakeConnector()], monkeypatch, vector_store=vector_store)
    query = ResourceQuery(query="5G NAS", resource_type="paper")

    result = asyncio.run(service.sync("fake", query))
    search = asyncio.run(service.search(ResourceQuery(query="5G NAS", resource_type="paper", allow_remote=False)))

    assert result.status == "SUCCESS"
    assert result.result_count == 1
    assert search["cache_hit"] is True
    assert search["resources"][0]["source_name"] == "fake"
    assert search["resources"][0]["fetched_at"]
    assert search["resources"][0]["indexed"] is False
    assert search["resources"][0]["vector_id"] is None
    assert vector_store.upserts == {}


def test_cache_ttl_expiry_refreshes_remote(tmp_path, monkeypatch):
    service = _service(tmp_path, [FakeConnector()], monkeypatch, ttl=0)
    query = ResourceQuery(query="5G NAS", resource_type="paper")

    first = asyncio.run(service.sync("fake", query))
    cached = asyncio.run(service.search(ResourceQuery(query="5G NAS", resource_type="paper", allow_remote=False)))
    refreshed = asyncio.run(service.search(query, connector_name="fake"))

    assert first.status == "SUCCESS"
    assert cached["cache_hit"] is False
    assert refreshed["cache_hit"] is False
    assert refreshed["resources"]


def test_remote_search_does_not_use_fake_as_implicit_default(tmp_path, monkeypatch):
    service = _service(tmp_path, [FakeConnector()], monkeypatch)

    result = asyncio.run(service.search(ResourceQuery(query="5G NAS", resource_type="paper")))

    assert result["resources"] == []
    assert result["error"] == "CONNECTOR_NOT_CONFIGURED"
    assert result["connectors_tried"] == []


def test_cache_miss_uses_specified_real_source_and_writes_cache(tmp_path, monkeypatch):
    paper = ExternalResource(
        external_id="arxiv-5g-nas",
        resource_type="paper",
        source_name="arxiv",
        title="5G NAS Security arXiv Paper",
        abstract="Real-source style arXiv metadata.",
        url="https://arxiv.org/abs/2609.00001",
        tags=["arXiv"],
    )
    service = _service(tmp_path, [ArxivFakeConnector(resources=[paper])], monkeypatch)

    result = asyncio.run(
        service.search(
            ResourceQuery(query="5G NAS", resource_type="paper", page_size=1),
            connector_name="arxiv",
        )
    )
    cached = asyncio.run(service.search(ResourceQuery(query="5G NAS", resource_type="paper", allow_remote=False)))

    assert result["error"] is None
    assert result["connector"] == "arxiv"
    assert result["resources"][0]["source_name"] == "arxiv"
    assert result["resources"][0]["url"].startswith("https://arxiv.org/")
    assert cached["cache_hit"] is True
    assert cached["resources"][0]["external_id"] == "arxiv-5g-nas"


def test_specified_source_cache_miss_does_not_fall_back_to_other_source_rows(tmp_path, monkeypatch):
    service = _service(tmp_path, [ArxivFakeConnector(resources=[])], monkeypatch)
    fake = ExternalResource(
        external_id="fake-overlap",
        resource_type="paper",
        source_name="fake",
        title="Overlapping 5G NAS Paper",
        abstract="5G NAS keyword overlap from fake source.",
        url="https://example.test/fake-overlap",
    )
    nvd = ExternalResource(
        external_id="CVE-2099-9999",
        resource_type="vulnerability",
        source_name="nvd",
        title="5G NAS CVE overlap",
        abstract="5G NAS keyword overlap from NVD.",
        url="https://nvd.nist.gov/vuln/detail/CVE-2099-9999",
    )
    service.repository.upsert(fake, ttl_hours=24, content_hash="fake-hash")
    service.repository.upsert(nvd, ttl_hours=24, content_hash="nvd-hash")

    result = asyncio.run(
        service.search(
            ResourceQuery(query="5G NAS", resource_type="paper", allow_remote=False),
            connector_name="arxiv",
        )
    )

    assert result["cache_hit"] is False
    assert result["resources"] == []
    assert result["connector"] == "arxiv"
    assert result["delivery_mode"] == "cache_miss"


def test_first_remote_search_calls_connector_then_second_search_hits_cache(tmp_path, monkeypatch):
    paper = ExternalResource(
        external_id="arxiv-counted",
        resource_type="paper",
        source_name="arxiv",
        title="Counted 5G NAS Paper",
        abstract="5G NAS source provenance test.",
        url="https://arxiv.org/abs/2609.00002",
    )
    connector = CountingArxivConnector(resources=[paper])
    service = _service(tmp_path, [connector], monkeypatch)
    query = ResourceQuery(query="5G NAS", resource_type="paper", page_size=1)

    first = asyncio.run(service.search(query, connector_name="arxiv"))
    second = asyncio.run(service.search(query, connector_name="arxiv"))

    assert connector.calls == 1
    assert first["cache_hit"] is False
    assert first["delivery_mode"] == "live"
    assert first["origin_kind"] == "remote"
    assert second["cache_hit"] is True
    assert second["delivery_mode"] == "fresh_cache"
    assert second["origin_kind"] == "remote"
    assert second["resources"][0]["source_name"] == "arxiv"


@respx.mock
def test_nvd_sync_keeps_original_query_for_audit_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_NVD", "true")
    base = "https://nvd.test/rest/json/cves/2.0"
    route = respx.get(base).mock(
        return_value=httpx.Response(
            200,
            json={
                "vulnerabilities": [
                    {
                        "cve": {
                            "id": "CVE-2021-44228",
                            "descriptions": [{"lang": "en", "value": "Log4j remote code execution."}],
                            "metrics": {},
                            "weaknesses": [],
                        }
                    }
                ]
            },
        )
    )
    connector = NVDConnector(base_url=base)
    service = _service(tmp_path, [connector], monkeypatch)
    original = "Explain the impact of CVE-2021-44228 on Log4j"
    query = ResourceQuery(query=original, resource_type="vulnerability", page_size=1)

    first = asyncio.run(service.search(query, connector_name="nvd"))
    second = asyncio.run(service.search(query, connector_name="nvd"))
    run = service.storage.connector_sync_runs(limit=1)[0]
    proof = service.storage.resource_provenance(int(first["resources"][0]["id"]))

    assert route.call_count == 1
    assert dict(route.calls.last.request.url.params)["cveId"] == "CVE-2021-44228"
    assert json.loads(run["query"])["query"] == original
    assert run["status"] == "SUCCESS"
    assert first["cache_hit"] is False
    assert first["delivery_mode"] == "live"
    assert second["cache_hit"] is True
    assert second["delivery_mode"] == "fresh_cache"
    assert proof["sync_status"] == "SUCCESS"
    assert proof["verified_url"].endswith("CVE-2021-44228")


def test_sync_writes_query_mapping_without_full_text_like_match(tmp_path, monkeypatch):
    resource = ExternalResource(
        external_id="semantic-hidden-match",
        resource_type="paper",
        source_name="semantic_scholar",
        title="Authentication Study",
        abstract="Metadata uses related wording but not the original query phrase.",
        url="https://example.test/paper",
    )
    service = _service(tmp_path, [UnfilteredConnector(resources=[resource])], monkeypatch)
    query = ResourceQuery(query="5G NAS zero trust slicing", resource_type="paper", allow_remote=False)

    synced = asyncio.run(service.sync("semantic_scholar", query))
    cached = asyncio.run(service.search(query))

    assert synced.status == "SUCCESS"
    assert cached["cache_hit"] is True
    assert cached["resources"][0]["external_id"] == "semantic-hidden-match"


def test_query_mapping_keeps_explicit_sources_separate(tmp_path, monkeypatch):
    arxiv = ExternalResource(
        external_id="arxiv-only",
        resource_type="paper",
        source_name="arxiv",
        title="Shared Topic",
        abstract="Shared abstract",
        url="https://arxiv.org/abs/arxiv-only",
    )
    nvd = ExternalResource(
        external_id="nvd-only",
        resource_type="vulnerability",
        source_name="nvd",
        title="Shared Topic",
        abstract="Shared abstract",
        url="https://nvd.nist.gov/vuln/detail/CVE-2099-0002",
    )
    service = _service(tmp_path, [UnfilteredArxivConnector(resources=[arxiv]), UnfilteredNVDConnector(resources=[nvd])], monkeypatch)
    paper_query = ResourceQuery(query="shared exact query", resource_type="paper", allow_remote=False)
    vuln_query = ResourceQuery(query="shared exact query", resource_type="vulnerability", allow_remote=False)

    asyncio.run(service.sync("arxiv", paper_query))
    asyncio.run(service.sync("nvd", vuln_query))
    paper_cached = service.repository.search_cache(paper_query, source_name="arxiv")
    vuln_cached = service.repository.search_cache(vuln_query, source_name="nvd")

    assert [item["external_id"] for item in paper_cached] == ["arxiv-only"]
    assert [item["external_id"] for item in vuln_cached] == ["nvd-only"]


def test_duplicate_resource_uses_unique_source_external_id(tmp_path, monkeypatch):
    duplicate = ExternalResource(
        external_id="same",
        resource_type="paper",
        source_name="fake",
        title="Duplicate Resource",
        abstract="first",
        url="https://example.test/duplicate",
    )
    service = _service(tmp_path, [FakeConnector(resources=[duplicate, duplicate])], monkeypatch)

    asyncio.run(service.sync("fake", ResourceQuery(query="Duplicate")))
    rows = service.storage.search_external_resources(ResourceQuery(query="Duplicate").model_dump(mode="json"), include_expired=True)

    assert len(rows) == 1


def test_unknown_and_disabled_connectors_record_failed_sync_runs(tmp_path, monkeypatch):
    service = _service(tmp_path, [DisabledConnector()], monkeypatch)

    missing = asyncio.run(service.sync("missing", ResourceQuery(query="NAS")))
    disabled = asyncio.run(service.sync("disabled_fake", ResourceQuery(query="NAS")))
    runs = service.repository.sync_runs(limit=10)

    assert missing.status == "FAILED"
    assert missing.error_code == "CONNECTOR_NOT_FOUND"
    assert disabled.status == "FAILED"
    assert disabled.error_code == "CONNECTOR_DISABLED"
    assert [run["connector_name"] for run in runs[:2]] == ["disabled_fake", "missing"]
    assert {run["error_code"] for run in runs[:2]} == {"CONNECTOR_NOT_FOUND", "CONNECTOR_DISABLED"}


def test_sync_remote_hook_runs_only_before_actual_remote_and_before_audit(tmp_path, monkeypatch):
    remote_resource = ExternalResource(
        external_id="rate-hook-paper",
        resource_type="paper",
        source_name="arxiv",
        title="5G NAS rate hook",
        abstract="Remote hook contract.",
        url="https://arxiv.org/abs/2601.00001",
    )
    service = _service(
        tmp_path,
        [ArxivFakeConnector(resources=[remote_resource]), DisabledConnector(), StandardSourceConnector()],
        monkeypatch,
    )
    counted = []

    remote = asyncio.run(
        service.sync(
            "arxiv",
            ResourceQuery(query="5G NAS", resource_type="paper"),
            before_remote=lambda name: counted.append(name),
        )
    )
    asyncio.run(
        service.sync(
            "disabled_fake",
            ResourceQuery(query="5G NAS"),
            before_remote=lambda name: counted.append(name),
        )
    )
    asyncio.run(
        service.sync(
            "standard_source",
            ResourceQuery(query="RFC 9000", resource_type="standard"),
            before_remote=lambda name: counted.append(name),
        )
    )

    before = len(service.repository.sync_runs(limit=100))
    with pytest.raises(RuntimeError, match="quota denied"):
        asyncio.run(
            service.sync(
                "arxiv",
                ResourceQuery(query="different paper", resource_type="paper"),
                before_remote=lambda _name: (_ for _ in ()).throw(RuntimeError("quota denied")),
            )
        )
    after = service.repository.sync_runs(limit=100)

    assert remote.status == "SUCCESS"
    assert counted == ["arxiv"]
    assert len(after) == before
    assert all(run["status"] != "RUNNING" for run in after)


@respx.mock
def test_rfc_editor_hooks_only_when_query_will_issue_http(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_REAL_RFC_LOOKUP", "true")
    endpoint = "https://rfc-count.test/rfc9000.json"
    route = respx.get(endpoint).mock(
        return_value=httpx.Response(200, json={"doc_id": "RFC9000", "title": "QUIC"})
    )

    search_connector = RFCEditorConnector(base_url="https://rfc-count.test")
    search_service = _service(tmp_path / "search", [search_connector], monkeypatch)
    search_hooks: list[str] = []
    skipped_search = asyncio.run(
        search_service.search(
            ResourceQuery(query="3GPP TS 24.501", resource_type="standard"),
            connector_name="rfc_editor",
            before_remote=search_hooks.append,
        )
    )
    actual_search = asyncio.run(
        search_service.search(
            ResourceQuery(query="RFC 9000", resource_type="standard"),
            connector_name="rfc_editor",
            before_remote=search_hooks.append,
        )
    )

    sync_connector = RFCEditorConnector(base_url="https://rfc-count.test")
    sync_service = _service(tmp_path / "sync", [sync_connector], monkeypatch)
    sync_hooks: list[str] = []
    skipped_sync = asyncio.run(
        sync_service.sync(
            "rfc_editor",
            ResourceQuery(query="3GPP TS 24.501", resource_type="standard"),
            before_remote=sync_hooks.append,
        )
    )
    before_runs = len(sync_service.repository.sync_runs(limit=100))
    with pytest.raises(RuntimeError, match="quota denied"):
        asyncio.run(
            sync_service.sync(
                "rfc_editor",
                ResourceQuery(query="RFC 9000", resource_type="standard"),
                before_remote=lambda _name: (_ for _ in ()).throw(RuntimeError("quota denied")),
            )
        )
    after_runs = sync_service.repository.sync_runs(limit=100)

    assert skipped_search["lookup_state"] == "NOT_FOUND"
    assert skipped_sync.status == "SUCCESS"
    assert search_hooks == ["rfc_editor"]
    assert sync_hooks == []
    assert route.call_count == 1
    assert actual_search["lookup_state"] == "FOUND"
    assert len(after_runs) == before_runs
    assert all(run["status"] != "RUNNING" for run in after_runs)


def test_fake_connector_pagination_and_timeout_mapping(tmp_path, monkeypatch):
    resources = [
        ExternalResource(
            external_id=f"paper-{index}",
            resource_type="paper",
            source_name="fake",
            title=f"Paged NAS Paper {index}",
            abstract="5G NAS pagination",
            url=f"https://example.test/paper-{index}",
        )
        for index in range(3)
    ]
    service = _service(tmp_path, [FakeConnector(resources=resources)], monkeypatch)
    page = asyncio.run(service.sync("fake", ResourceQuery(query="Paged", page=2, page_size=1)))

    timeout_service = _service(tmp_path, [FakeConnector(delay_seconds=0.2)], monkeypatch)
    timeout = asyncio.run(timeout_service.sync("fake", ResourceQuery(query="NoCacheTimeout")))

    assert page.status == "SUCCESS"
    assert page.result_count == 1
    assert page.resources[0].external_id == "paper-1"
    assert timeout.status == "FAILED"
    assert timeout.error_code == "CONNECTOR_TIMEOUT"


@respx.mock
def test_http_connector_success_429_and_5xx_error_mapping(tmp_path, monkeypatch):
    url = "https://connector.test/search"
    respx.get(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "paper-1",
                        "title": "NAS paper",
                        "abstract": "5G NAS",
                        "url": "https://connector.test/paper-1",
                    }
                ]
            },
        )
    )
    service = _service(tmp_path, [HTTPFakeConnector(url)], monkeypatch)
    ok = asyncio.run(service.sync("http_fake", ResourceQuery(query="NAS")))
    assert ok.status == "SUCCESS"

    respx.get(url).mock(return_value=httpx.Response(429, json={"error": "limited"}))
    limited = asyncio.run(service.sync("http_fake", ResourceQuery(query="missing")))
    assert limited.status == "FAILED"
    assert limited.error_code == "CONNECTOR_RATE_LIMIT"

    respx.get(url).mock(return_value=httpx.Response(503, json={"error": "down"}))
    failed = asyncio.run(service.sync("http_fake", ResourceQuery(query="missing")))
    assert failed.status == "FAILED"
    assert failed.error_code == "CONNECTOR_SERVICE_ERROR"


def test_external_failure_can_fall_back_to_expired_cache(tmp_path, monkeypatch):
    resource = ExternalResource(
        external_id="cached",
        resource_type="paper",
        source_name="fake",
        title="Cached NAS Paper",
        abstract="5G NAS cached abstract",
        url="https://example.test/cached",
    )
    service = _service(tmp_path, [FakeConnector(resources=[resource])], monkeypatch, ttl=0)
    asyncio.run(service.sync("fake", ResourceQuery(query="Cached")))
    failing = _service(
        tmp_path,
        [FakeConnector(fail=ConnectorServiceError("service down"))],
        monkeypatch,
        ttl=0,
    )
    failing.storage = service.storage
    failing.repository.storage = service.storage

    result = asyncio.run(failing.sync("fake", ResourceQuery(query="Cached")))
    delivered = asyncio.run(
        failing.search(ResourceQuery(query="Cached"), connector_name="fake")
    )

    assert result.status == "FAILED_WITH_CACHE"
    assert result.cache_hit is True
    assert result.error_code == "CONNECTOR_SERVICE_ERROR"
    assert delivered["status"] == "FAILED_WITH_CACHE"
    assert delivered["delivery_mode"] == "stale_cache"
    assert delivered["fallback_used"] is True
    assert delivered["upstream_error_code"] == "CONNECTOR_SERVICE_ERROR"


def test_expired_deadline_prevents_late_resource_commit(tmp_path, monkeypatch):
    resource = ExternalResource(
        external_id="late-paper",
        resource_type="paper",
        source_name="arxiv",
        title="Late paper",
        abstract="Must not be committed after the deadline.",
        url="https://arxiv.org/abs/late-paper",
    )
    service = _service(
        tmp_path,
        [CancellationResistantConnector(resources=[resource])],
        monkeypatch,
    )

    result = asyncio.run(
        service.sync(
            "arxiv",
            ResourceQuery(query="late paper", resource_type="paper"),
            deadline=time.monotonic() + 0.02,
        )
    )
    cached = service.repository.search_cache(
        ResourceQuery(query="late paper", resource_type="paper"),
        include_expired=True,
        source_name="arxiv",
    )
    run = service.repository.sync_runs(limit=1)[0]

    assert result.status == "FAILED"
    assert result.error_code == "CONNECTOR_TIMEOUT"
    assert cached == []
    assert run["status"] == "FAILED"
    assert run["error_code"] == "CONNECTOR_TIMEOUT"


def test_http_connection_error_records_failed_run_without_running_leftover(tmp_path, monkeypatch):
    service = _service(tmp_path, [ConnectionFailConnector()], monkeypatch)

    result = asyncio.run(service.sync("connect_fail", ResourceQuery(query="NAS")))
    runs = service.repository.sync_runs(limit=5)

    assert result.status == "FAILED"
    assert result.error_code == "CONNECTOR_SERVICE_ERROR"
    assert runs[0]["status"] == "FAILED"
    assert runs[0]["error_code"] == "CONNECTOR_SERVICE_ERROR"
    assert all(run["status"] != "RUNNING" for run in runs)


def test_vector_sync_replaces_old_vector_when_content_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("EXTERNAL_RESOURCE_INDEX_ENABLED", "true")
    vector_store = RecordingVectorStore()
    first = ExternalResource(
        external_id="paper-change",
        resource_type="paper",
        source_name="fake",
        title="Changing Paper",
        abstract="first version",
        url="https://example.test/changing-paper",
    )
    second = first.model_copy(update={"abstract": "second version"})
    service = _service(tmp_path, [FakeConnector(resources=[first])], monkeypatch, vector_store=vector_store)

    asyncio.run(service.sync("fake", ResourceQuery(query="Changing")))
    service.registry = ConnectorRegistry([FakeConnector(resources=[second])])
    asyncio.run(service.sync("fake", ResourceQuery(query="Changing")))

    assert "external:1" in vector_store.deleted
    assert vector_store.upserts["external:1"].count("second version") == 1


def test_connector_api_requires_teacher_for_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-connectors-long-enough")
    monkeypatch.setenv("EXTERNAL_RESOURCE_INDEX_ENABLED", "false")
    original_logger = main.learning_logger
    original_registry = main.connector_registry
    main.learning_logger = LearningLogger(tmp_path / "api.db")
    main.learning_logger.seed_demo_identity()
    main.connector_registry = ConnectorRegistry([ArxivFakeConnector()])
    client = TestClient(main.app)
    try:
        student_token = _login_token(client, "student_demo")
        teacher_token = _login_token(client, "teacher_demo")

        status = client.get("/api/connectors/status")
        denied = client.post(
            "/api/connectors/arxiv/sync",
            headers={"Authorization": f"Bearer {student_token}"},
            json={"query": "5G NAS", "resource_type": "paper"},
        )
        ok = client.post(
            "/api/connectors/arxiv/sync",
            headers={"Authorization": f"Bearer {teacher_token}"},
            json={"query": "5G NAS", "resource_type": "paper"},
        )
        runs = client.get(
            "/api/connectors/sync-runs",
            headers={"Authorization": f"Bearer {teacher_token}"},
        )

        assert status.status_code == 200
        assert denied.status_code == 403
        assert ok.status_code == 200
        assert runs.json()["items"]
    finally:
        main.learning_logger = original_logger
        main.connector_registry = original_registry


def test_connector_sync_rate_limit_uses_authenticated_teacher_actor(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-connector-sync-rate-limit")
    original_logger = main.learning_logger
    original_registry = main.connector_registry
    resource = ExternalResource(
        external_id="sync-rate-paper",
        resource_type="paper",
        source_name="arxiv",
        title="Sync rate paper",
        url="https://arxiv.org/abs/sync-rate-paper",
    )
    main.learning_logger = LearningLogger(tmp_path / "sync-rate.db")
    main.learning_logger.seed_demo_identity()
    main.connector_registry = ConnectorRegistry([ArxivFakeConnector(resources=[resource])])
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_MAX_CALLS", 1)
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_WINDOW_SECONDS", 60.0)
    main._RESOURCE_RATE_BUCKETS.clear()
    client = TestClient(main.app)
    try:
        teacher_token = _login_token(client, "teacher_demo")
        first = client.post(
            "/api/connectors/arxiv/sync",
            headers={"Authorization": f"Bearer {teacher_token}"},
            json={"query": "sync rate paper", "resource_type": "paper"},
        )
        limited = client.post(
            "/api/connectors/arxiv/sync",
            headers={"Authorization": f"Bearer {teacher_token}"},
            json={"query": "another sync query", "resource_type": "paper"},
        )

        assert first.status_code == 200
        assert first.json()["status"] == "SUCCESS"
        assert limited.status_code == 429
        assert limited.json()["error_code"] == "RESOURCE_RATE_LIMIT"
        assert int(limited.headers["Retry-After"]) >= 1
    finally:
        main._RESOURCE_RATE_BUCKETS.clear()
        main.learning_logger = original_logger
        main.connector_registry = original_registry


@respx.mock
def test_teacher_rfc_sync_counts_only_query_that_will_issue_http(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-rfc-sync-rate-limit")
    monkeypatch.setenv("ENABLE_REAL_RFC_LOOKUP", "true")
    original_logger = main.learning_logger
    original_registry = main.connector_registry
    route = respx.get("https://rfc-teacher.test/rfc9000.json").mock(
        return_value=httpx.Response(200, json={"doc_id": "RFC9000", "title": "QUIC"})
    )
    main.learning_logger = LearningLogger(tmp_path / "rfc-sync-rate.db")
    main.learning_logger.seed_demo_identity()
    main.connector_registry = ConnectorRegistry(
        [RFCEditorConnector(base_url="https://rfc-teacher.test")]
    )
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_MAX_CALLS", 1)
    monkeypatch.setattr(main, "_RESOURCE_RATE_LIMIT_WINDOW_SECONDS", 60.0)
    main._RESOURCE_RATE_BUCKETS.clear()
    client = TestClient(main.app)
    try:
        teacher_token = _login_token(client, "teacher_demo")
        teacher = main.learning_logger.user_by_username("teacher_demo")
        assert teacher is not None
        headers = {"Authorization": f"Bearer {teacher_token}"}

        skipped = client.post(
            "/api/connectors/rfc_editor/sync",
            headers=headers,
            json={"query": "3GPP TS 24.501", "resource_type": "standard"},
        )
        bucket_key = (int(teacher["id"]), "rfc_editor")
        assert skipped.status_code == 200
        assert bucket_key not in main._RESOURCE_RATE_BUCKETS
        assert route.call_count == 0

        main._consume_resource_rate_limit(int(teacher["id"]), "rfc_editor")
        before = len(main.learning_logger.connector_sync_runs(limit=100))
        limited = client.post(
            "/api/connectors/rfc_editor/sync",
            headers=headers,
            json={"query": "RFC 9000", "resource_type": "standard"},
        )
        after = main.learning_logger.connector_sync_runs(limit=100)

        assert limited.status_code == 429
        assert limited.json()["error_code"] == "RESOURCE_RATE_LIMIT"
        assert int(limited.headers["Retry-After"]) >= 1
        assert route.call_count == 0
        assert len(after) == before
        assert all(run["status"] != "RUNNING" for run in after)
    finally:
        main._RESOURCE_RATE_BUCKETS.clear()
        main.learning_logger = original_logger
        main.connector_registry = original_registry


def test_teacher_fake_sync_is_forbidden_without_writing_sync_run(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-fake-sync-boundary")
    original_logger = main.learning_logger
    original_registry = main.connector_registry
    main.learning_logger = LearningLogger(tmp_path / "fake-sync-api.db")
    main.learning_logger.seed_demo_identity()
    main.connector_registry = ConnectorRegistry([FakeConnector()])
    client = TestClient(main.app)
    try:
        teacher_token = _login_token(client, "teacher_demo")
        before = len(main.learning_logger.connector_sync_runs(limit=100))

        forbidden = client.post(
            "/api/connectors/fake/sync",
            headers={"Authorization": f"Bearer {teacher_token}"},
            json={"query": "5G NAS", "resource_type": "paper"},
        )
        after = len(main.learning_logger.connector_sync_runs(limit=100))

        assert forbidden.status_code == 400
        assert forbidden.json()["error_code"] == "FAKE_CONNECTOR_FORBIDDEN"
        assert after == before
    finally:
        main.learning_logger = original_logger
        main.connector_registry = original_registry


def test_standard_source_reports_local_metadata_not_live_remote(tmp_path, monkeypatch):
    service = _service(tmp_path, [StandardSourceConnector(knowledge_dir=tmp_path / "kb")], monkeypatch)
    query = ResourceQuery(query="3GPP TS 24.501", resource_type="standard", page_size=1)

    first = asyncio.run(service.search(query, connector_name="standard_source"))
    second = asyncio.run(service.search(query, connector_name="standard_source"))

    assert first["status"] == "SUCCESS"
    assert first["cache_hit"] is False
    assert first["delivery_mode"] == "local_metadata"
    assert first["origin_kind"] == "local_catalog"
    assert first["resources"][0]["source_name"] == "standard_source"
    assert first["resources"][0]["raw_metadata"]["official_metadata_only"] is True
    assert second["cache_hit"] is True
    assert second["delivery_mode"] == "fresh_cache"
    assert second["origin_kind"] == "local_catalog"


def test_nvd_sync_caches_one_vulnerability_and_student_portal_can_read_it(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-nvd-portal-long-enough")
    monkeypatch.setenv("EXTERNAL_RESOURCE_INDEX_ENABLED", "false")
    resource = ExternalResource(
        external_id="CVE-2099-0001",
        resource_type="vulnerability",
        source_name="nvd",
        title="CVE-2099-0001 Demo NAS Vulnerability",
        abstract="Demo NVD vulnerability metadata for course portal visibility.",
        url="https://nvd.nist.gov/vuln/detail/CVE-2099-0001",
        tags=["CVE", "NVD"],
        module="5G NAS",
    )
    original_logger = main.learning_logger
    original_registry = main.connector_registry
    main.learning_logger = LearningLogger(tmp_path / "nvd-api.db")
    main.learning_logger.seed_demo_identity()
    main.connector_registry = ConnectorRegistry([NVDFakeConnector(resources=[resource])])
    client = TestClient(main.app)
    try:
        teacher_token = _login_token(client, "teacher_demo")
        student_token = _login_token(client, "student_demo")

        sync = client.post(
            "/api/connectors/nvd/sync",
            headers={"Authorization": f"Bearer {teacher_token}"},
            json={"query": "CVE-2099", "resource_type": "vulnerability", "page_size": 1},
        )
        portal = client.get(
            "/api/resources/portal",
            headers={"Authorization": f"Bearer {student_token}"},
            params={"resource_type": "vulnerability", "keyword": "CVE-2099"},
        )

        assert sync.status_code == 200
        assert sync.json()["status"] == "SUCCESS"
        assert sync.json()["result_count"] == 1
        items = portal.json()["items"]
        assert len(items) == 1
        assert items[0]["source_name"] == "nvd"
        assert items[0]["url"].endswith("CVE-2099-0001")
        assert items[0]["cache_status"] == "fresh"
        assert items[0]["metadata"]["indexed"] is False
        assert items[0]["metadata"]["vector_id"] is None
    finally:
        main.learning_logger = original_logger
        main.connector_registry = original_registry


def test_teacher_syncs_real_source_types_and_student_portal_reads_links(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-real-source-portal-long-enough")
    monkeypatch.setenv("EXTERNAL_RESOURCE_INDEX_ENABLED", "false")
    paper = ExternalResource(
        external_id="arxiv-course-paper",
        resource_type="paper",
        source_name="arxiv",
        title="5G NAS Security Paper",
        abstract="Paper metadata from a real-source connector.",
        url="https://arxiv.org/abs/2609.00002",
        tags=["arXiv"],
        module="5G NAS",
    )
    cve = ExternalResource(
        external_id="CVE-2099-4444",
        resource_type="vulnerability",
        source_name="nvd",
        title="CVE-2099-4444",
        abstract="NVD metadata for a teaching sample.",
        url="https://nvd.nist.gov/vuln/detail/CVE-2099-4444",
        tags=["CVSS 7.5 HIGH", "CWE-20"],
        module="5G NAS",
    )
    project = ExternalResource(
        external_id="github-1001",
        resource_type="project",
        source_name="github",
        title="demo/nas-security-lab",
        abstract="GitHub project metadata for a teaching sample.",
        url="https://github.com/demo/nas-security-lab",
        tags=["Python"],
        module="5G NAS",
        raw_metadata={
            "full_name": "demo/nas-security-lab",
            "private": False,
            "visibility": "public",
        },
    )
    original_logger = main.learning_logger
    original_registry = main.connector_registry
    main.learning_logger = LearningLogger(tmp_path / "real-source-api.db")
    main.learning_logger.seed_demo_identity()
    main.connector_registry = ConnectorRegistry(
        [
            ArxivFakeConnector(resources=[paper]),
            NVDFakeConnector(resources=[cve]),
            GitHubFakeConnector(resources=[project]),
        ]
    )
    client = TestClient(main.app)
    try:
        teacher_token = _login_token(client, "teacher_demo")
        student_token = _login_token(client, "student_demo")
        teacher_headers = {"Authorization": f"Bearer {teacher_token}"}
        student_headers = {"Authorization": f"Bearer {student_token}"}

        paper_sync = client.post(
            "/api/connectors/arxiv/sync",
            headers=teacher_headers,
            json={"query": "5G NAS", "resource_type": "paper", "page_size": 1},
        )
        cve_sync = client.post(
            "/api/connectors/nvd/sync",
            headers=teacher_headers,
            json={"query": "CVE-2099", "resource_type": "vulnerability", "page_size": 1},
        )
        project_sync = client.post(
            "/api/connectors/github/sync",
            headers=teacher_headers,
            json={"query": "nas-security", "resource_type": "project", "page_size": 1},
        )
        portal = client.get(
            "/api/resources/portal",
            headers=student_headers,
            params={"keyword": "5G NAS", "page_size": 50},
        )

        assert paper_sync.status_code == 200
        assert cve_sync.status_code == 200
        assert project_sync.status_code == 200
        items = portal.json()["items"]
        by_type = {item["type"]: item for item in items if item["type"] in {"paper", "vulnerability", "project"}}
        assert by_type["paper"]["source_name"] == "arxiv"
        assert by_type["paper"]["url"].startswith("https://arxiv.org/")
        assert by_type["vulnerability"]["source_name"] == "nvd"
        assert by_type["vulnerability"]["url"].startswith("https://nvd.nist.gov/")
        assert by_type["project"]["source_name"] == "github"
        assert by_type["project"]["url"].startswith("https://github.com/")
        assert all(item["fetched_at"] for item in by_type.values())
    finally:
        main.learning_logger = original_logger
        main.connector_registry = original_registry
