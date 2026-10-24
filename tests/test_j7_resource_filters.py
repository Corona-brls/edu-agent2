from __future__ import annotations

from fastapi.testclient import TestClient

from backend import main
from backend.storage.logger import LearningLogger
from frontend.pages.student import (
    _normalize_remote_resource_payload,
    _resource_query_fingerprint,
    _select_resource_payload,
)


def _fingerprint(**overrides):
    values = {
        "keyword": "nas security",
        "topic": "5G NAS",
        "resource_type": "paper",
        "source": "arxiv",
        "year_from": 2020,
        "year_to": 2026,
        "page": 1,
        "page_size": 10,
        "course_id": 1,
        "class_id": 1,
    }
    values.update(overrides)
    return _resource_query_fingerprint(**values)


def _remote_payload() -> dict:
    return {
        "items": [
            {
                "resource_id": "external:1",
                "type": "paper",
                "title": "arXiv NAS paper",
                "summary": "5G NAS security evidence",
                "source": "arxiv",
                "source_name": "arxiv",
                "origin_kind": "remote",
                "delivery_mode": "fresh_cache",
                "fetched_at": "2026-10-01T00:00:00Z",
                "published_at": "2024-01-01",
            },
            {
                "resource_id": "external:2",
                "type": "project",
                "title": "NAS project",
                "summary": "5G NAS open source project",
                "source": "github",
                "source_name": "github",
                "origin_kind": "remote",
                "delivery_mode": "live",
                "fetched_at": "2026-10-01T00:01:00Z",
                "published_at": "2025-01-01",
            },
        ],
        "total": 2,
        "page": 1,
        "page_size": 10,
        "cache_mode": "mixed",
    }


def test_resource_query_fingerprint_changes_for_every_visible_filter():
    baseline = _fingerprint()

    for field, value in {
        "keyword": "blockchain",
        "topic": "covert channel",
        "resource_type": "project",
        "source": "github",
        "year_from": 2024,
        "year_to": 2025,
        "page": 2,
        "page_size": 20,
        "course_id": 2,
    }.items():
        assert _fingerprint(**{field: value}) != baseline

    assert _fingerprint(keyword="  NAS SECURITY ") == baseline
    assert _fingerprint(source=" ARXIV ") == baseline


def test_old_on_demand_payload_never_overrides_changed_portal_filters():
    remote = _remote_payload()
    portal = {"items": [], "total": 0, "page": 1, "page_size": 10, "cache_mode": "cache_first"}
    stored = _fingerprint()

    assert _select_resource_payload(portal, remote, stored, _fingerprint(source="github")) == portal
    assert _select_resource_payload(portal, remote, stored, _fingerprint(resource_type="project")) == portal
    assert _select_resource_payload(portal, remote, stored, _fingerprint(year_from=2025)) == portal
    assert _select_resource_payload(portal, remote, stored, _fingerprint(page=2)) == portal

    selected = _select_resource_payload(portal, remote, stored, stored)
    assert [item["resource_id"] for item in selected["items"]] == ["external:1"]
    assert selected["items"][0]["source_name"] == "arxiv"


def test_remote_normalization_preserves_backend_delivery_fields_without_guessing_live():
    payload = _normalize_remote_resource_payload(
        {
            "cache_hit": True,
            "delivery_mode": "stale_cache",
            "origin_kind": "remote",
            "resources": [
                {
                    "id": 7,
                    "external_id": "cached-paper",
                    "resource_type": "paper",
                    "source_name": "arxiv",
                    "title": "Cached paper",
                    "abstract": "Cached abstract",
                    "url": "https://arxiv.org/abs/cached-paper",
                    "published_at": "2023-01-01",
                    "fetched_at": "2026-09-30T00:00:00Z",
                    "provenance_verified": True,
                    "origin_kind": "remote",
                    "delivery_mode": "stale_cache",
                }
            ],
        }
    )

    item = payload["items"][0]
    assert item["source_name"] == "arxiv"
    assert item["origin_kind"] == "remote"
    assert item["delivery_mode"] == "stale_cache"
    assert item["fetched_at"] == "2026-09-30T00:00:00Z"
    assert item["delivery_mode"] != "live"


def test_anonymous_resource_portal_and_remote_search_are_rejected(tmp_path, monkeypatch):
    original = main.learning_logger
    main.learning_logger = LearningLogger(tmp_path / "j7-auth.db")
    try:
        with TestClient(main.app) as client:
            portal = client.get("/api/resources/portal")
            remote = client.post(
                "/api/resources/search",
                json={"query": "5G NAS", "resource_type": "paper", "allow_remote": True},
            )

        assert portal.status_code == 401
        assert remote.status_code == 401
        assert portal.json()["error_code"] == "AUTH_REQUIRED"
        assert remote.json()["error_code"] == "AUTH_REQUIRED"
    finally:
        main.learning_logger = original
