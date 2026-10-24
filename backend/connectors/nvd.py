"""NVD CVE API 2.0 connector."""

from __future__ import annotations

import re
from hashlib import sha256

import httpx

from backend.connectors.base import ExternalConnector
from backend.connectors.http_utils import map_http_error
from backend.connectors.schemas import ConnectorHealth, ExternalResource, ResourceQuery
from backend.core.config import get_settings


_CVE_ID_RE = re.compile(r"(?<![A-Z0-9])CVE-\d{4}-\d{4,}(?![A-Z0-9])", re.IGNORECASE)
_CVE_TOKEN_RE = re.compile(r"(?<![A-Z0-9])CVE(?:-[A-Z0-9]+){0,2}", re.IGNORECASE)
_NVD_KEYWORD_MAX_LENGTH = 120


class NVDConnector(ExternalConnector):
    name = "nvd"

    def __init__(self, base_url: str = "https://services.nvd.nist.gov/rest/json/cves/2.0") -> None:
        settings = get_settings()
        self.enabled = settings.enable_nvd
        self.api_key = settings.nvd_api_key
        self.base_url = base_url

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, enabled=self.enabled, status="ok" if self.enabled else "disabled")

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        params = _build_search_params(query)
        headers = {"apiKey": self.api_key} if self.api_key else {}
        async with httpx.AsyncClient(timeout=get_settings().external_http_timeout_seconds) as client:
            response = await client.get(self.base_url, params=params, headers=headers)
        map_http_error(response)
        payload = response.json()
        request_url = str(response.request.url)
        request_mode = "cve_id" if "cveId" in params else "keyword"
        resources: list[ExternalResource] = []
        for item in payload.get("vulnerabilities", []):
            cve = item.get("cve", {})
            cve_id = cve.get("id")
            if not cve_id:
                continue
            if "cveId" in params and cve_id.upper() != str(params["cveId"]).upper():
                continue
            description = _english_description(cve.get("descriptions", []))
            metrics = cve.get("metrics", {})
            cvss = _cvss(metrics)
            cwe = _cwe(cve.get("weaknesses", []))
            published = cve.get("published")
            updated = cve.get("lastModified")
            resources.append(
                ExternalResource(
                    external_id=cve_id,
                    resource_type="vulnerability",
                    source_name=self.name,
                    title=cve_id,
                    abstract=description,
                    url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                    published_at=published,
                    tags=[tag for tag in [cvss, cwe] if tag],
                    module=query.module,
                    raw_metadata={
                        "cvss": cvss,
                        "cwe": cwe,
                        "published": published,
                        "lastModified": updated,
                        "nvd_api_status": response.status_code,
                        "nvd_api_url": request_url,
                        "nvd_total_results": payload.get("totalResults"),
                        "nvd_query_mode": request_mode,
                    },
                    rights_note="NVD public CVE metadata.",
                )
            )
        return resources


def _extract_cve_id(text: str) -> str | None:
    """Return one unambiguous CVE ID without treating surrounding prose as an ID."""
    matches = list(_CVE_ID_RE.finditer(text))
    if len(matches) != 1:
        return None
    remainder = f"{text[:matches[0].start()]} {text[matches[0].end():]}"
    if _CVE_TOKEN_RE.search(remainder):
        return None
    return matches[0].group(0).upper()


def _build_search_params(query: ResourceQuery) -> dict[str, str | int]:
    cve_id = _extract_cve_id(query.query)
    if cve_id:
        return {"cveId": cve_id}
    keywords = " ".join(query.query.split())[:_NVD_KEYWORD_MAX_LENGTH].strip()
    return {
        "keywordSearch": keywords,
        "startIndex": (query.page - 1) * query.page_size,
        "resultsPerPage": query.page_size,
    }


def nvd_query_metadata(query: ResourceQuery) -> dict[str, str]:
    """Expose non-sensitive normalization details without changing the cache key."""
    params = _build_search_params(query)
    mode = "cve_id" if "cveId" in params else "keyword"
    digest = sha256(query.query.encode("utf-8")).hexdigest()[:16]
    metadata = {"query_mode": mode, "query_summary": f"sha256:{digest}"}
    if mode == "cve_id":
        metadata["normalized_cve_id"] = str(params["cveId"])
    return metadata


def _english_description(descriptions: list[dict]) -> str:
    for item in descriptions:
        if item.get("lang") == "en":
            return item.get("value", "")
    return descriptions[0].get("value", "") if descriptions else ""


def _cvss(metrics: dict) -> str:
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        values = metrics.get(key) or []
        if values:
            data = values[0].get("cvssData", {})
            score = data.get("baseScore")
            severity = values[0].get("baseSeverity") or data.get("baseSeverity")
            return f"CVSS {score} {severity}".strip()
    return ""


def _cwe(weaknesses: list[dict]) -> str:
    for weakness in weaknesses:
        for desc in weakness.get("description", []):
            if desc.get("lang") == "en":
                return desc.get("value", "")
    return ""
