from __future__ import annotations

import asyncio
import os

import httpx
import pytest
import respx

from backend.connectors.arxiv import ArxivConnector
from backend.connectors.base import (
    ConnectorAuthenticationError,
    ConnectorBadRequestError,
    ConnectorError,
    ConnectorForbiddenError,
    ConnectorNotFoundError,
    ConnectorRateLimitError,
    ConnectorServiceError,
)
from backend.connectors.http_utils import map_http_error
from backend.connectors.github import GitHubConnector
from backend.connectors.nvd import NVDConnector
from backend.connectors.schemas import ResourceQuery
from backend.connectors.semantic_scholar import SemanticScholarConnector
from backend.connectors.standard_source import StandardSourceConnector


def test_http_error_mapping_distinguishes_sanitized_upstream_statuses():
    expected = {
        400: (ConnectorBadRequestError, "CONNECTOR_UPSTREAM_BAD_REQUEST"),
        401: (ConnectorAuthenticationError, "CONNECTOR_UPSTREAM_AUTHENTICATION_FAILED"),
        403: (ConnectorForbiddenError, "CONNECTOR_UPSTREAM_FORBIDDEN"),
        404: (ConnectorNotFoundError, "CONNECTOR_UPSTREAM_NOT_FOUND"),
        429: (ConnectorRateLimitError, "CONNECTOR_RATE_LIMIT"),
        503: (ConnectorServiceError, "CONNECTOR_SERVICE_ERROR"),
    }
    for status, (error_type, error_code) in expected.items():
        response = httpx.Response(
            status,
            request=httpx.Request("GET", "https://upstream.test/resource"),
            text="SECRET_TOKEN_AND_BODY_MUST_NOT_ESCAPE",
            headers={"Authorization": "Bearer secret"},
        )
        with pytest.raises(error_type) as caught:
            map_http_error(response)
        assert caught.value.error_code == error_code
        assert "SECRET" not in str(caught.value)
        assert "Bearer" not in str(caught.value)


@respx.mock
def test_semantic_scholar_success_empty_rate_limit_and_5xx(monkeypatch):
    monkeypatch.setenv("ENABLE_SEMANTIC_SCHOLAR", "true")
    base = "https://semantic.test/graph/v1"
    route = respx.get(f"{base}/paper/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "p1",
                        "title": "5G NAS Security",
                        "authors": [{"name": "Alice"}],
                        "abstract": "NAS security context",
                        "year": 2026,
                        "url": "https://semanticscholar.org/paper/p1",
                        "externalIds": {"DOI": "10.123/demo"},
                        "venue": "DemoConf",
                    },
                    {
                        "paperId": "p1-dup",
                        "title": "duplicate",
                        "externalIds": {"DOI": "10.123/demo"},
                    },
                ]
            },
        )
    )
    results = asyncio.run(SemanticScholarConnector(base_url=base).search(ResourceQuery(query="NAS")))
    assert len(results) == 1
    assert results[0].external_id == "10.123/demo"
    assert results[0].url.startswith("https://")

    route.mock(return_value=httpx.Response(200, json={"data": []}))
    assert asyncio.run(SemanticScholarConnector(base_url=base).search(ResourceQuery(query="empty"))) == []

    route.mock(return_value=httpx.Response(429, json={"error": "limited"}))
    with pytest.raises(Exception) as limited:
        asyncio.run(SemanticScholarConnector(base_url=base).search(ResourceQuery(query="limited")))
    assert getattr(limited.value, "error_code", "") == "CONNECTOR_RATE_LIMIT"

    route.mock(return_value=httpx.Response(503, json={"error": "down"}))
    with pytest.raises(Exception) as failed:
        asyncio.run(SemanticScholarConnector(base_url=base).search(ResourceQuery(query="down")))
    assert getattr(failed.value, "error_code", "") == "CONNECTOR_SERVICE_ERROR"


@respx.mock
def test_arxiv_success_and_empty(monkeypatch):
    monkeypatch.setenv("ENABLE_ARXIV", "true")
    url = "https://arxiv.test/api/query"
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2601.00001v1</id>
        <title> Covert Communication over Blockchain </title>
        <summary> A teaching paper abstract. </summary>
        <published>2026-01-01T00:00:00Z</published>
        <author><name>Alice</name></author>
      </entry>
    </feed>"""
    respx.get(url).mock(return_value=httpx.Response(200, text=xml))
    results = asyncio.run(ArxivConnector(base_url=url).search(ResourceQuery(query="blockchain")))
    assert results[0].external_id == "2601.00001v1"
    assert results[0].published_at == "2026"
    assert results[0].url.startswith("https://arxiv.org")

    respx.get(url).mock(return_value=httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"></feed>'))
    assert asyncio.run(ArxivConnector(base_url=url).search(ResourceQuery(query="none"))) == []


@respx.mock
def test_nvd_success_empty_rate_limit_and_5xx(monkeypatch):
    monkeypatch.setenv("ENABLE_NVD", "true")
    base = "https://nvd.test/rest/json/cves/2.0"
    route = respx.get(base).mock(
        return_value=httpx.Response(
            200,
            json={
                "totalResults": 1,
                "vulnerabilities": [
                    {
                        "cve": {
                            "id": "CVE-2026-0001",
                            "published": "2026-01-01T00:00:00.000",
                            "lastModified": "2026-01-02T00:00:00.000",
                            "descriptions": [{"lang": "en", "value": "Demo vulnerability"}],
                            "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 7.5}, "baseSeverity": "HIGH"}]},
                            "weaknesses": [{"description": [{"lang": "en", "value": "CWE-79"}]}],
                        }
                    }
                ]
            },
        )
    )
    results = asyncio.run(NVDConnector(base_url=base).search(ResourceQuery(query="CVE-2026-0001")))
    assert results[0].external_id == "CVE-2026-0001"
    assert "CVSS 7.5 HIGH" in results[0].tags
    assert results[0].url.endswith("CVE-2026-0001")
    assert results[0].raw_metadata["nvd_api_status"] == 200
    assert results[0].raw_metadata["nvd_total_results"] == 1
    assert results[0].raw_metadata["nvd_query_mode"] == "cve_id"
    assert "cveId=CVE-2026-0001" in results[0].raw_metadata["nvd_api_url"]

    route.mock(return_value=httpx.Response(200, json={"vulnerabilities": []}))
    assert asyncio.run(NVDConnector(base_url=base).search(ResourceQuery(query="none"))) == []

    route.mock(return_value=httpx.Response(429, json={}))
    with pytest.raises(Exception) as limited:
        asyncio.run(NVDConnector(base_url=base).search(ResourceQuery(query="limited")))
    assert getattr(limited.value, "error_code", "") == "CONNECTOR_RATE_LIMIT"

    route.mock(return_value=httpx.Response(500, json={}))
    with pytest.raises(Exception) as failed:
        asyncio.run(NVDConnector(base_url=base).search(ResourceQuery(query="down")))
    assert getattr(failed.value, "error_code", "") == "CONNECTOR_SERVICE_ERROR"


@pytest.mark.parametrize(
    ("raw_query", "expected_key", "expected_value"),
    [
        ("CVE-2021-44228", "cveId", "CVE-2021-44228"),
        ("CVE-2021-44228 impact on Log4j", "cveId", "CVE-2021-44228"),
        ("Explain the impact of CVE-2021-44228 on Log4j", "cveId", "CVE-2021-44228"),
        ("Open5GS authentication vulnerability", "keywordSearch", "Open5GS authentication vulnerability"),
        (
            "Compare CVE-2021-44228 with CVE-2023-44487",
            "keywordSearch",
            "Compare CVE-2021-44228 with CVE-2023-44487",
        ),
        ("CVE-2021-44228foo impact", "keywordSearch", "CVE-2021-44228foo impact"),
    ],
)
@respx.mock
def test_nvd_normalizes_natural_language_query_parameters(
    monkeypatch,
    raw_query,
    expected_key,
    expected_value,
):
    monkeypatch.setenv("ENABLE_NVD", "true")
    base = "https://nvd.test/rest/json/cves/2.0"
    route = respx.get(base).mock(return_value=httpx.Response(200, json={"vulnerabilities": []}))

    asyncio.run(NVDConnector(base_url=base).search(ResourceQuery(query=raw_query, page=2, page_size=3)))

    sent = dict(route.calls.last.request.url.params)
    assert sent[expected_key] == expected_value
    assert set(sent).isdisjoint({"keywordSearch" if expected_key == "cveId" else "cveId"})
    if expected_key == "keywordSearch":
        assert sent["startIndex"] == "3"
        assert sent["resultsPerPage"] == "3"


@respx.mock
def test_nvd_limits_long_keyword_search(monkeypatch):
    monkeypatch.setenv("ENABLE_NVD", "true")
    base = "https://nvd.test/rest/json/cves/2.0"
    route = respx.get(base).mock(return_value=httpx.Response(200, json={"vulnerabilities": []}))

    asyncio.run(NVDConnector(base_url=base).search(ResourceQuery(query="Open5GS " + "impact " * 35)))

    sent = dict(route.calls.last.request.url.params)
    assert "cveId" not in sent
    assert len(sent["keywordSearch"]) <= 120


def test_standard_source_does_not_infer_local_full_text_from_filenames(tmp_path):
    connector = StandardSourceConnector(knowledge_dir=tmp_path)
    first = asyncio.run(connector.search(ResourceQuery(query="3GPP")))
    (tmp_path / "generic_standard_notes.pdf").write_bytes(b"not a matching standard full text")
    generic = asyncio.run(connector.search(ResourceQuery(query="3GPP TS 24.501")))
    (tmp_path / "3GPP_TS_24.501_teaching_notes.md").write_text(
        "Teaching notes only; not the copyrighted standard full text.",
        encoding="utf-8",
    )
    (tmp_path / "3GPP_TS_24.501.pdf").write_bytes(b"unverified upload")
    second = asyncio.run(connector.search(ResourceQuery(query="3GPP")))

    assert first[0].resource_type == "standard"
    assert first[0].raw_metadata["local_uploaded"] is False
    assert generic[0].external_id == "3GPP-TS-24.501"
    assert generic[0].raw_metadata["local_uploaded"] is False
    assert second[0].raw_metadata["local_uploaded"] is False
    assert second[0].raw_metadata["local_match"] is None
    assert "Local full text uploaded: True" not in second[0].abstract
    assert "copyrighted full text is not fetched" in second[0].rights_note


def test_standard_source_contains_course_acceptance_metadata(tmp_path):
    connector = StandardSourceConnector(knowledge_dir=tmp_path)
    nas = asyncio.run(connector.search(ResourceQuery(query="TS 24.501", page_size=5)))
    security = asyncio.run(connector.search(ResourceQuery(query="33.501", page_size=5)))
    rfc = asyncio.run(connector.search(ResourceQuery(query="RFC 9000", page_size=5)))

    assert nas[0].external_id == "3GPP-TS-24.501"
    assert nas[0].url.startswith("https://portal.3gpp.org/")
    assert security[0].external_id == "3GPP-TS-33.501"
    assert rfc[0].external_id == "RFC-9000"
    assert all(item.raw_metadata["official_metadata_only"] is True for item in [nas[0], security[0], rfc[0]])


@respx.mock
def test_github_project_search_optional(monkeypatch):
    monkeypatch.setenv("ENABLE_GITHUB_SEARCH", "true")
    url = "https://github.test/search/repositories"
    respx.get(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "incomplete_results": False,
                "items": [
                    {
                        "id": 1,
                        "full_name": "demo/repo",
                        "private": False,
                        "description": "security teaching project",
                        "html_url": "https://github.com/demo/repo",
                        "language": "Python",
                        "stargazers_count": 10,
                        "updated_at": "2026-01-01T00:00:00Z",
                    }
                ]
            },
        )
    )
    results = asyncio.run(GitHubConnector(base_url=url).search(ResourceQuery(query="security")))
    assert results[0].external_id == "1"
    assert results[0].resource_type == "project"
    assert results[0].url.startswith("https://github.com")


@respx.mock
def test_semantic_scholar_exact_doi_uses_lookup_path_and_rejects_mismatch(monkeypatch):
    monkeypatch.setenv("ENABLE_SEMANTIC_SCHOLAR", "true")
    base = "https://semantic.test/graph/v1"
    route = respx.get(f"{base}/paper/DOI:10.1234/demo").mock(
        return_value=httpx.Response(
            200,
            json={
                "paperId": "paper-1",
                "title": "Exact DOI paper",
                "externalIds": {"DOI": "10.1234/demo"},
                "url": "https://example.test/paper-1",
            },
        )
    )

    exact = asyncio.run(SemanticScholarConnector(base_url=base).search(ResourceQuery(query="doi:10.1234/demo")))
    assert [item.external_id for item in exact] == ["10.1234/demo"]
    assert route.called

    route.mock(
        return_value=httpx.Response(
            200,
            json={
                "paperId": "paper-wrong",
                "title": "Wrong DOI",
                "externalIds": {"DOI": "10.1234/other"},
                "url": "https://example.test/wrong",
            },
        )
    )
    mismatch = asyncio.run(SemanticScholarConnector(base_url=base).search(ResourceQuery(query="10.1234/demo")))
    assert mismatch == []


@respx.mock
def test_arxiv_exact_preserves_old_style_identifier_and_version(monkeypatch):
    monkeypatch.setenv("ENABLE_ARXIV", "true")
    url = "https://arxiv.test/api/query"
    route = respx.get(url).mock(
        return_value=httpx.Response(
            200,
            text="""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
              <id>http://arxiv.org/abs/hep-th/9901001v2</id><title>Exact</title><summary>Exact abstract</summary>
            </entry></feed>""",
        )
    )
    results = asyncio.run(ArxivConnector(base_url=url).search(ResourceQuery(query="arXiv:hep-th/9901001v2")))
    assert [item.external_id for item in results] == ["hep-th/9901001v2"]
    assert dict(route.calls.last.request.url.params)["id_list"] == "hep-th/9901001v2"

    route.mock(
        return_value=httpx.Response(
            200,
            text="""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
              <id>http://arxiv.org/abs/hep-th/9901001v1</id><title>Wrong version</title><summary>Mismatch</summary>
            </entry></feed>""",
        )
    )
    assert asyncio.run(ArxivConnector(base_url=url).search(ResourceQuery(query="hep-th/9901001v2"))) == []


@respx.mock
def test_nvd_exact_lookup_rejects_mismatched_cve(monkeypatch):
    monkeypatch.setenv("ENABLE_NVD", "true")
    base = "https://nvd.test/rest/json/cves/2.0"
    respx.get(base).mock(
        return_value=httpx.Response(
            200,
            json={"vulnerabilities": [{"cve": {"id": "CVE-2026-9999", "descriptions": []}}]},
        )
    )
    results = asyncio.run(NVDConnector(base_url=base).search(ResourceQuery(query="CVE-2026-1234")))
    assert results == []


@respx.mock
def test_github_exact_and_search_metadata_contract(monkeypatch):
    from backend.connectors.schemas import ConnectorSearchBatch

    monkeypatch.setenv("ENABLE_GITHUB_SEARCH", "true")
    base = "https://github.test"
    exact_route = respx.get(f"{base}/repos/open5gs/open5gs").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": 1,
                "full_name": "open5gs/open5gs",
                "private": False,
                "html_url": "https://github.com/open5gs/open5gs",
            },
        )
    )
    exact = asyncio.run(GitHubConnector(base_url=f"{base}/search/repositories").search(ResourceQuery(query="open5gs/open5gs")))
    assert [item.title for item in exact] == ["open5gs/open5gs"]
    assert exact_route.called

    exact_route.mock(
        return_value=httpx.Response(
            200,
            json={
                "id": 2,
                "full_name": "open5gs/open5gs",
                "private": True,
                "html_url": "https://github.com/open5gs/open5gs",
            },
        )
    )
    assert asyncio.run(GitHubConnector(base_url=f"{base}/search/repositories").search(ResourceQuery(query="open5gs/open5gs"))) == []

    search_route = respx.get(f"{base}/search/repositories").mock(
        return_value=httpx.Response(200, json={"items": [], "incomplete_results": True})
    )
    batch = asyncio.run(
        GitHubConnector(base_url=f"{base}/search/repositories").search_with_metadata(
            ResourceQuery(query="x" * 256)
        )
    )
    assert isinstance(batch, ConnectorSearchBatch)
    assert batch.incomplete_results is True
    assert search_route.called

    search_route.mock(return_value=httpx.Response(200, json={"items": []}))
    with pytest.raises(ConnectorServiceError) as malformed:
        asyncio.run(
            GitHubConnector(base_url=f"{base}/search/repositories").search_with_metadata(
                ResourceQuery(query="metadata contract")
            )
        )
    assert malformed.value.error_code == "CONNECTOR_RESPONSE_INVALID"

    with pytest.raises(ConnectorError, match="256") as too_long:
        asyncio.run(GitHubConnector(base_url=f"{base}/search/repositories").search(ResourceQuery(query="x" * 257)))
    assert too_long.value.error_code == "GITHUB_QUERY_TOO_LONG"


@respx.mock
def test_rfc_editor_exact_lookup_requires_matching_document_id(monkeypatch):
    from backend.connectors.rfc_editor import RFCEditorConnector

    monkeypatch.setenv("ENABLE_REAL_RFC_LOOKUP", "true")
    base = "https://rfc.test"
    route = respx.get(f"{base}/rfc9000.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "doc_id": "RFC9000",
                "title": "QUIC: A UDP-Based Multiplexed and Secure Transport",
                "abstract": "QUIC transport metadata.",
                "pub_status": "PROPOSED STANDARD",
            },
        )
    )
    results = asyncio.run(RFCEditorConnector(base_url=base).search(ResourceQuery(query="RFC 9000")))
    assert [item.external_id for item in results] == ["RFC-9000"]
    assert results[0].abstract == "QUIC transport metadata."
    assert route.called

    route.mock(return_value=httpx.Response(200, json={"doc_id": "RFC8999", "title": "Wrong RFC"}))
    assert asyncio.run(RFCEditorConnector(base_url=base).search(ResourceQuery(query="RFC 9000"))) == []


@pytest.mark.live
def test_live_semantic_scholar_or_arxiv_smoke(monkeypatch):
    if os.getenv("RUN_LIVE_CONNECTOR_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_LIVE_CONNECTOR_TESTS=true to run live connector smoke.")
    query = ResourceQuery(query="5G NAS security", page_size=1)
    try:
        results = _live_or_skip(SemanticScholarConnector().search(query), "semantic_scholar")
    except pytest.skip.Exception:
        results = _live_or_skip(ArxivConnector().search(query), "arxiv")
    assert results
    assert results[0].url


@pytest.mark.live
def test_live_nvd_smoke(monkeypatch):
    if os.getenv("RUN_LIVE_CONNECTOR_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_LIVE_CONNECTOR_TESTS=true to run live connector smoke.")
    results = _live_or_skip(NVDConnector().search(ResourceQuery(query="CVE-2021-44228", page_size=1)), "nvd")
    assert results
    assert results[0].external_id.startswith("CVE-")


def _live_or_skip(coro, connector_name: str):
    try:
        return asyncio.run(coro)
    except Exception as exc:
        pytest.skip(f"{connector_name} live smoke unavailable: {type(exc).__name__}: {exc}")
