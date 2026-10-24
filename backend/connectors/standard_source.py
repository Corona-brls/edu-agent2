"""Standards metadata connector; does not fetch copyrighted full text."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from backend.connectors.base import ExternalConnector
from backend.connectors.schemas import ConnectorHealth, ExternalResource, ResourceQuery
from backend.core.config import get_settings


STANDARD_CATALOG = [
    {
        "id": "3GPP-TS-24.501",
        "title": "3GPP TS 24.501",
        "version": "Release 18 specification metadata",
        "url": "https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3370",
        "tags": ["3GPP", "5G", "NAS", "5GS", "mobility management"],
        "module": "5G NAS",
        "aliases": ["24.501", "24501", "ts24501", "ts_24_501", "3gpp-ts-24-501"],
    },
    {
        "id": "3GPP-TS-33.501",
        "title": "3GPP TS 33.501",
        "version": "Release 18 specification metadata",
        "url": "https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3169",
        "tags": ["3GPP", "5G", "security architecture", "authentication", "key agreement"],
        "module": "5G security",
        "aliases": ["33.501", "33501", "ts33501", "ts_33_501", "3gpp-ts-33-501"],
    },
    {
        "id": "3GPP-TS-23.501",
        "title": "3GPP TS 23.501",
        "version": "Release 18 specification metadata",
        "url": "https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3144",
        "tags": ["3GPP", "5G", "system architecture", "SBA", "core network"],
        "module": "5G architecture",
        "aliases": ["23.501", "23501", "ts23501", "ts_23_501", "3gpp-ts-23-501"],
    },
    {
        "id": "RFC-9000",
        "title": "RFC 9000",
        "version": "IETF QUIC transport, 2021 metadata",
        "url": "https://www.rfc-editor.org/rfc/rfc9000",
        "tags": ["IETF", "QUIC", "transport", "network security"],
        "module": "protocol security",
        "aliases": ["rfc9000", "rfc-9000", "9000"],
    },
]

STANDARD_METADATA_RIGHTS_NOTE = (
    "Metadata and official link only; copyrighted full text is not fetched."
)
STANDARD_METADATA_SOURCE_NOTE = (
    "本地维护的官方元数据与官方标准来源链接；未发现经核验的本地标准全文，不抓取受版权保护全文。"
)


class StandardSourceConnector(ExternalConnector):
    name = "standard_source"
    enabled = True
    supports_remote = False

    def __init__(self, knowledge_dir: str | Path | None = None) -> None:
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir is not None else get_settings().knowledge_base_dir

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, enabled=True, status="ok", detail="metadata only")

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        q = query.query.lower()
        resources = []
        for item in STANDARD_CATALOG:
            if q and not _matches_standard_query(q, item):
                continue
            resources.append(
                ExternalResource(
                    external_id=item["id"],
                    resource_type="standard",
                    source_name=self.name,
                    title=item["title"],
                    abstract=(
                        f"Standard metadata only. Version: {item['version']}. "
                        "Verified local full text: unavailable."
                    ),
                    url=item["url"],
                    tags=item["tags"],
                    module=query.module or item["module"],
                    raw_metadata={
                        "version": item["version"],
                        "local_uploaded": False,
                        "local_match": None,
                        "official_metadata_only": True,
                        "local_verification_status": "unverified",
                    },
                    rights_note=STANDARD_METADATA_RIGHTS_NOTE,
                )
            )
        return resources[: query.page_size]


def normalize_standard_resource_view(resource: Mapping[str, Any]) -> dict[str, Any]:
    """Correct unverified full-text claims without rewriting the audit record."""
    presented = dict(resource)
    resource_type = str(presented.get("resource_type") or presented.get("type") or "").lower()
    if resource_type != "standard":
        return presented

    raw_metadata = dict(presented.get("raw_metadata") or {})
    version = str(raw_metadata.get("version") or "").strip()
    raw_metadata.update(
        {
            "local_uploaded": False,
            "local_match": None,
            "official_metadata_only": True,
            "local_verification_status": "unverified",
        }
    )
    canonical_summary = "Standard metadata only."
    if version:
        canonical_summary += f" Version: {version}."
    canonical_summary += " Verified local full text: unavailable."
    presented["raw_metadata"] = raw_metadata
    for field in ("abstract", "summary", "raw_summary"):
        if field in presented:
            presented[field] = canonical_summary

    if isinstance(presented.get("metadata"), dict):
        metadata = dict(presented["metadata"])
        metadata["raw_metadata"] = raw_metadata
        metadata["metadata_only"] = True
        metadata["local_full_text_available"] = False
        metadata["source_note"] = STANDARD_METADATA_SOURCE_NOTE
        presented["metadata"] = metadata
    return presented


def _matches_standard_query(query: str, item: dict) -> bool:
    haystack = " ".join([item["id"], item["title"], item["module"], " ".join(item["tags"]), " ".join(item["aliases"])]).lower()
    normalized_query = _normalize_identifier(query)
    if query in haystack:
        return True
    return any(_normalize_identifier(alias) in normalized_query for alias in item["aliases"])


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
