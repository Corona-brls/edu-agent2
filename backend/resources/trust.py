"""Trust policy for externally sourced course resources."""

from __future__ import annotations

from collections.abc import Mapping


PRODUCTION_SOURCE_ORIGINS: dict[str, str] = {
    "arxiv": "remote",
    "semantic_scholar": "remote",
    "nvd": "remote",
    "github": "remote",
    "rfc_editor": "remote",
    "standard_source": "local_catalog",
}

TRUSTED_ORIGIN_KINDS = {"remote", "local_catalog"}


def normalized_trust_policy(policy: Mapping[str, str] | None = None) -> dict[str, str]:
    selected = dict(PRODUCTION_SOURCE_ORIGINS if policy is None else policy)
    invalid = {
        name: origin
        for name, origin in selected.items()
        if not name.strip() or origin not in TRUSTED_ORIGIN_KINDS
    }
    if invalid:
        raise ValueError(f"Invalid external resource trust policy: {sorted(invalid)}")
    return selected


def origin_for_source(source_name: str | None, policy: Mapping[str, str]) -> str:
    return policy.get(str(source_name or "").strip(), "unknown")

