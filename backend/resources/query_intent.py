"""Deterministic public-resource identifier parsing.

This module intentionally depends only on the standard library and small value
types.  It must remain safe to import from connectors without pulling in the
registry, storage, service, or Agent layers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit


LookupMode = Literal["SEARCH", "EXACT"]


class LookupInputError(ValueError):
    def __init__(self, message: str, *, error_code: str = "INVALID_ARGUMENTS") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class LookupTarget:
    mode: LookupMode
    kind: str | None = None
    identifier: str | None = None
    source_candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ExactMatch:
    kind: str
    identifier: str
    source_candidates: tuple[str, ...]
    span: tuple[int, int]


_DOI_RE = re.compile(r"(?i)(10\.\d{4,9}/[^\s<>\"']+)")
_ARXIV_IDENTIFIER_PATTERN = r"(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})(?:v\d+)?"
_ARXIV_RE = re.compile(rf"(?i)({_ARXIV_IDENTIFIER_PATTERN})")
_ARXIV_IDENTIFIER_RE = re.compile(rf"(?i)^{_ARXIV_IDENTIFIER_PATTERN}$")
_DOI_IDENTIFIER_RE = re.compile(r"(?i)^10\.\d{4,9}/[^\s/?#][^\s?#]*$")
_CVE_RE = re.compile(r"(?i)(?<![A-Z0-9])CVE-\d{4}-\d{4,}(?![A-Z0-9])")
_GITHUB_OWNER_PATTERN = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?"
_GITHUB_REPO_PATTERN = r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,99})"
_GITHUB_REPO_RE = re.compile(
    rf"(?i)^({_GITHUB_OWNER_PATTERN})/({_GITHUB_REPO_PATTERN})(?:\.git)?$"
)
_URL_TOKEN_RE = re.compile(r"(?i)https?://[^\s<>\"']+")
_URI_TOKEN_RE = re.compile(r"(?i)(?<![A-Za-z0-9+.-])([a-z][a-z0-9+.-]*):[^\s<>\"']+")
_PAPER_DECLARATION_RE = re.compile(r"(?i)(doi|arxiv)\s*:")
_SCHEMELESS_PAPER_HOST_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9.-])(?://)?(?:dx\.)?doi\.org(?:[/\\?#]|$)"
    r"|(?<![A-Za-z0-9.-])(?://)?arxiv\.org(?:[/\\?#]|$)"
)
_SCHEMELESS_GITHUB_RE = re.compile(
    r"(?i)github\.com(?::[^\s/<>\"']*)?/[^\s<>\"']+"
)
_GITHUB_CONTEXT_RE = re.compile(
    r"(?i)(?:github\s+(?:repository|repo|project)|repository|repo|仓库|项目)"
    r"(?:\s*(?:is\b|为|[:：])\s*|\s+)"
    r"(?P<claim>[\"'][^\s<>]*|[^\s<>\"']+)"
)
_GITHUB_PREFIXED_REPO_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_.-])github\s+(?P<claim>[^\s<>]+)"
)
_RFC_RE = re.compile(r"(?i)(?<![A-Z0-9])RFC[\s-]*(\d{1,6})(?![A-Z0-9])")
_TERMINAL_IDENTIFIER_PUNCTUATION = ".,;:!?)]}，。；：！？、）】’”"


def parse_lookup_target(raw_query: str, resource_type: str | None = None) -> LookupTarget:
    text = " ".join(str(raw_query or "").split())
    exact: list[_ExactMatch] = []

    preliminary_doi_spans = _prevalidated_doi_spans(text)
    github_outer_spans = _recognized_github_outer_spans(text, preliminary_doi_spans)

    if resource_type in {None, "paper"}:
        paper_text = _mask_spans(text, github_outer_spans) if resource_type is None else text
        exact.extend(_paper_exact_matches(paper_text))

    paper_spans = [match.span for match in exact if match.kind in {"doi", "arxiv"}]
    remaining_for_github = _mask_spans(text, paper_spans)
    if resource_type in {None, "project"}:
        exact.extend(
            _github_exact_matches(
                remaining_for_github,
                allow_alphabetic_bare=resource_type == "project",
                allow_public_identifier_urls=resource_type is None,
            )
        )

    protected_spans = [match.span for match in exact]
    remaining = _mask_spans(text, protected_spans)
    if resource_type in {None, "vulnerability"}:
        exact.extend(
            _ExactMatch("cve", match.group(0).upper(), ("nvd",), match.span())
            for match in _CVE_RE.finditer(remaining)
        )
    if resource_type in {None, "standard"}:
        exact.extend(
            _ExactMatch(
                "rfc",
                f"RFC-{int(match.group(1))}",
                ("rfc_editor", "standard_source"),
                match.span(),
            )
            for match in _RFC_RE.finditer(remaining)
        )

    unique = list(
        dict.fromkeys(
            (match.kind, match.identifier, match.source_candidates) for match in exact
        )
    )
    if len(unique) > 1:
        raise LookupInputError(
            "同一次查询包含多个不同的精确标识，请分开查询。",
            error_code="AMBIGUOUS_IDENTIFIER",
        )
    if unique:
        kind, identifier, candidates = unique[0]
        return LookupTarget(mode="EXACT", kind=kind, identifier=identifier, source_candidates=candidates)
    return LookupTarget(mode="SEARCH")


def validate_lookup_filters(
    target: LookupTarget,
    *,
    page: int,
    year_from: int | None,
    year_to: int | None,
    module: str = "",
) -> None:
    if target.mode == "EXACT" and (
        page != 1 or year_from is not None or year_to is not None or bool(module.strip())
    ):
        raise LookupInputError(
            "精确编号查询不能叠加分页或年份筛选。",
            error_code="EXACT_QUERY_FILTER_CONFLICT",
        )


def validate_connector_query(target: LookupTarget, *, connector_name: str | None, query: str) -> None:
    if connector_name == "github" and target.mode == "SEARCH" and len(" ".join(query.split())) > 256:
        raise LookupInputError(
            "GitHub 仓库搜索词不能超过256个字符。",
            error_code="GITHUB_QUERY_TOO_LONG",
        )


def matches_exact_identity(target: LookupTarget, source_name: str, resource: dict) -> bool:
    if target.mode != "EXACT" or not target.identifier:
        return True
    identifier = target.identifier
    external_id = str(resource.get("external_id") or "")
    metadata = resource.get("raw_metadata") or {}
    if not isinstance(metadata, dict):
        return False
    if target.kind == "doi":
        ids = metadata.get("externalIds") or {}
        returned = ids.get("DOI") if isinstance(ids, dict) else None
        return _normalize_doi(str(returned or external_id)) == identifier
    if target.kind == "arxiv":
        ids = metadata.get("externalIds") or {}
        returned = ids.get("ArXiv") if isinstance(ids, dict) else None
        returned = str(returned or metadata.get("arxiv_id") or external_id)
        return _arxiv_identity_matches(identifier, returned)
    if target.kind == "cve":
        return external_id.upper() == identifier
    if target.kind == "github":
        full_name = str(metadata.get("full_name") or "").lower()
        visibility = str(metadata.get("visibility") or "public").lower()
        return (
            full_name == identifier.lower()
            and metadata.get("private") is False
            and visibility not in {"private", "internal"}
        )
    if target.kind == "rfc":
        return external_id.upper() == identifier
    return False


def is_deliverable_exact_arxiv_resource(target: LookupTarget, resource: dict) -> bool:
    """Require one trusted exact arXiv record with its real official landing URL."""
    if target.mode != "EXACT" or target.kind != "arxiv" or not target.identifier:
        return False
    source_name = str(resource.get("source_name") or "")
    if source_name not in target.source_candidates or not matches_exact_identity(target, source_name, resource):
        return False
    if (
        resource.get("provenance_verified") is not True
        or str(resource.get("origin_kind") or "") != "remote"
        or not str(resource.get("content_hash") or "").strip()
        or str(resource.get("resource_type") or "") != "paper"
    ):
        return False
    try:
        parsed = urlsplit(str(resource.get("url") or ""))
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() != "arxiv.org"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        return False
    match = re.fullmatch(rf"/abs/({_ARXIV_IDENTIFIER_PATTERN})", parsed.path, re.IGNORECASE)
    return bool(match and _arxiv_identity_matches(target.identifier, match.group(1)))


def _normalize_doi(value: str) -> str:
    """Normalize an already extracted/provider DOI without query punctuation rules."""
    return value.strip().lower()


def _normalize_arxiv(value: str) -> str:
    normalized = _strip_arxiv_natural_question(_strip_chinese_sentence_suffix(value)).strip()
    return normalized.rstrip(_TERMINAL_IDENTIFIER_PUNCTUATION).lower()


def _normalize_query_doi(value: str) -> str | None:
    candidate = _strip_chinese_sentence_suffix(value).strip()
    candidate = _strip_unpaired_terminal_delimiters(candidate)
    while candidate and candidate[-1] in ".,!?，。！？；：、":
        candidate = candidate[:-1]
        candidate = _strip_unpaired_terminal_delimiters(candidate)
    candidate = candidate.lower()
    return candidate if _DOI_IDENTIFIER_RE.fullmatch(candidate) else None


def _strip_unpaired_terminal_delimiters(value: str) -> str:
    candidate = value
    pairs = {")": "(", "]": "[", "}": "{"}
    while candidate and candidate[-1] in pairs:
        closer = candidate[-1]
        opener = pairs[closer]
        if candidate.count(closer) <= candidate.count(opener):
            break
        candidate = candidate[:-1]
    return candidate


def _strip_arxiv_natural_question(value: str) -> str:
    marker = value.find("?")
    if marker < 0:
        return value
    suffix = value[marker + 1 :]
    adjacent = suffix.split(maxsplit=1)[0] if suffix else ""
    if not adjacent or (
        _is_cjk_character(adjacent[0])
        and not any(character in adjacent for character in "#&=%+/?")
    ):
        return value[:marker]
    return value


def _is_cjk_character(value: str) -> bool:
    return bool(value) and (
        "\u3400" <= value[0] <= "\u4dbf"
        or "\u4e00" <= value[0] <= "\u9fff"
        or "\uf900" <= value[0] <= "\ufaff"
    )


def _prevalidated_doi_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _DOI_RE.finditer(text):
        token_start = _prefixed_token_start(text, match.start(1), "doi")
        if _has_invalid_identifier_left(text, token_start):
            continue
        if _normalize_query_doi(match.group(1)) is not None:
            spans.append((token_start, match.end(1)))
    return spans


def _recognized_github_outer_spans(
    text: str, protected_spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for token in _URL_TOKEN_RE.finditer(text):
        if any(_spans_overlap(token.span(), span) for span in protected_spans):
            continue
        if not _has_invalid_identifier_left(text, token.start()) and (
            _github_repository_from_url(token.group(0)) is not None
        ):
            spans.append(token.span())
    for token in _SCHEMELESS_GITHUB_RE.finditer(text):
        if any(_spans_overlap(token.span(), span) for span in protected_spans + spans):
            continue
        if not _has_invalid_identifier_left(text, token.start()) and (
            _github_repository_from_url(f"https://{token.group(0)}") is not None
        ):
            spans.append(token.span())
    for context in _github_context_matches(text):
        span = context.span()
        if any(
            item[0] <= span[0] and span[1] <= item[1]
            for item in protected_spans
        ):
            continue
        if (
            _github_repository_from_claim(context.group("claim")) is not None
            or _github_context_requires_exact_repository(text, context)
        ):
            spans.append(span)
    return spans


def _arxiv_identity_matches(requested: str, returned: str) -> bool:
    requested_normalized = _normalize_arxiv(requested)
    returned_normalized = _normalize_arxiv(returned)
    if re.search(r"v\d+$", requested_normalized):
        return returned_normalized == requested_normalized
    return re.sub(r"v\d+$", "", returned_normalized) == requested_normalized


def _github_exact_matches(
    text: str,
    *,
    allow_alphabetic_bare: bool,
    allow_public_identifier_urls: bool,
) -> list[_ExactMatch]:
    matches: list[_ExactMatch] = []
    claimed_spans: list[tuple[int, int]] = []
    address_spans = [token.span() for token in _URL_TOKEN_RE.finditer(text)]
    address_spans.extend(token.span() for token in _SCHEMELESS_GITHUB_RE.finditer(text))

    for context in _github_context_matches(text):
        if any(start <= context.start() < end for start, end in address_spans):
            continue
        if _has_invalid_identifier_left(text, context.start()):
            raise LookupInputError(
                "GitHub 仓库声明必须从独立 token 起点开始。",
                error_code="INVALID_ARGUMENTS",
            )
        repository = _github_repository_from_claim(context.group("claim"))
        if repository is None:
            if _github_context_requires_exact_repository(text, context):
                raise LookupInputError(
                    "GitHub 仓库声明必须包含完整合法的 owner/repo。",
                    error_code="INVALID_ARGUMENTS",
                )
            continue
        claimed_spans.append(context.span())
        matches.append(
            _ExactMatch("github", repository, ("github",), context.span())
        )

    for token in _URL_TOKEN_RE.finditer(text):
        if any(_spans_overlap(token.span(), span) for span in claimed_spans):
            continue
        claimed_spans.append(token.span())
        value = token.group(0)
        if _has_invalid_identifier_left(text, token.start()):
            raise LookupInputError(
                "GitHub 仓库 URL 必须从独立 token 起点开始。",
                error_code="INVALID_ARGUMENTS",
            )
        repository = _github_repository_from_url(value)
        if repository is not None:
            matches.append(_ExactMatch("github", repository, ("github",), token.span()))
            continue
        if allow_public_identifier_urls and _is_public_identifier_url(value):
            continue
        raise LookupInputError(
            "GitHub 仓库 URL 必须使用无凭证、无端口且仅含 owner/repo 的 github.com 地址。",
            error_code="INVALID_ARGUMENTS",
        )

    for token in _URI_TOKEN_RE.finditer(text):
        if token.group(1).lower() in {"http", "https"}:
            continue
        if "github.com/" in token.group(0).lower():
            claimed_spans.append(token.span())

    for token in _SCHEMELESS_GITHUB_RE.finditer(text):
        if any(_spans_overlap(token.span(), span) for span in claimed_spans):
            continue
        claimed_spans.append(token.span())
        if _has_invalid_identifier_left(text, token.start()):
            raise LookupInputError(
                "GitHub 仓库地址必须从独立 token 起点开始。",
                error_code="INVALID_ARGUMENTS",
            )
        repository = _github_repository_from_url(f"https://{token.group(0)}")
        if repository is None:
            raise LookupInputError(
                "GitHub 仓库 URL 必须仅含 owner/repo 路径。",
                error_code="INVALID_ARGUMENTS",
            )
        matches.append(_ExactMatch("github", repository, ("github",), token.span()))

    without_urls = _mask_spans(text, claimed_spans)
    stripped = without_urls.strip()
    stripped_start = len(without_urls) - len(without_urls.lstrip())
    repository = _normalize_repository(stripped)
    if repository is not None and _is_unambiguous_bare_repository(
        stripped,
        allow_alphabetic=allow_alphabetic_bare,
    ):
        matches.append(
            _ExactMatch(
                "github",
                repository,
                ("github",),
                (stripped_start, stripped_start + len(stripped)),
            )
        )
    elif _looks_like_invalid_repository_claim(stripped) or (
        allow_alphabetic_bare
        and _looks_like_invalid_complete_repository_path(stripped)
    ):
        raise LookupInputError(
            "GitHub owner/repo 语法无效。",
            error_code="INVALID_ARGUMENTS",
        )
    return matches


def _is_public_identifier_url(value: str) -> bool:
    return _public_identifier_from_url(value) is not None


def _public_identifier_url_kind(value: str) -> str | None:
    parsed_identifier = _public_identifier_from_url(value)
    return parsed_identifier[0] if parsed_identifier is not None else None


def _public_identifier_from_url(value: str) -> tuple[str, str] | None:
    try:
        cleaned = _strip_chinese_sentence_suffix(value).strip()
        if "arxiv.org" in cleaned.lower():
            cleaned = _strip_arxiv_natural_question(cleaned)
        cleaned = _strip_unpaired_terminal_delimiters(cleaned).rstrip(".,;:!，。；：！")
        parsed = urlsplit(cleaned)
        port = parsed.port
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    path = parsed.path
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    if host in {"doi.org", "dx.doi.org"}:
        identifier = path.removeprefix("/")
        identifier = _normalize_query_doi(identifier)
        if identifier is not None:
            return "doi", identifier
        return None
    if host == "arxiv.org":
        match = re.fullmatch(
            rf"/(?:abs|pdf)/({_ARXIV_IDENTIFIER_PATTERN})(?:\.pdf)?",
            path,
            re.IGNORECASE,
        )
        if match is not None:
            return "arxiv", _normalize_arxiv(match.group(1))
        return None
    return None


def _paper_exact_matches(text: str) -> list[_ExactMatch]:
    matches: list[_ExactMatch] = []
    occupied: list[tuple[int, int]] = []
    prevalidated_doi_spans = _prevalidated_doi_spans(text)

    for token in _URL_TOKEN_RE.finditer(text):
        if any(_spans_overlap(token.span(), span) for span in prevalidated_doi_spans):
            continue
        if _has_invalid_identifier_left(text, token.start()):
            raise LookupInputError(
                "论文标识 URL 必须从独立 token 起点开始。",
                error_code="INVALID_ARGUMENTS",
            )
        occupied.append(token.span())
        parsed = _public_identifier_from_url(token.group(0))
        if parsed is not None:
            kind, identifier = parsed
            matches.append(_paper_match(kind, identifier, token.span()))
        elif _uri_claims_paper_identifier(token.group(0)):
            raise LookupInputError(
                "论文标识 URL/URI 必须使用官方 HTTP(S) 来源并包含完整合法标识。",
                error_code="INVALID_ARGUMENTS",
            )

    for token in _URI_TOKEN_RE.finditer(text):
        if any(_spans_overlap(token.span(), span) for span in occupied):
            continue
        if any(_spans_overlap(token.span(), span) for span in prevalidated_doi_spans):
            continue
        scheme = token.group(1).lower()
        if scheme in {"doi", "arxiv"}:
            continue
        occupied.append(token.span())
        if _uri_claims_paper_identifier(token.group(0)):
            raise LookupInputError(
                "论文标识 URL/URI 必须使用官方 HTTP(S) 来源并包含完整合法标识。",
                error_code="INVALID_ARGUMENTS",
            )

    work = _mask_spans(text, occupied)
    _reject_schemeless_paper_host_claims(work)
    dois, doi_spans = _bare_doi_identifiers(work)
    for identifier, span in zip(dois, doi_spans):
        matches.append(_paper_match("doi", identifier, span))

    work_without_dois = _mask_spans(work, doi_spans)
    arxiv_items = _bare_arxiv_identifiers(work_without_dois, [])
    arxiv_spans = [span for _, span in arxiv_items]
    for identifier, span in arxiv_items:
        matches.append(_paper_match("arxiv", identifier, span))

    remaining = _mask_spans(work_without_dois, arxiv_spans)
    declaration = _PAPER_DECLARATION_RE.search(remaining)
    if declaration is not None:
        raise LookupInputError(
            "doi:/arXiv: 声明后必须紧跟对应的完整合法标识。",
            error_code="INVALID_ARGUMENTS",
        )
    return matches


def _paper_match(kind: str, identifier: str, span: tuple[int, int]) -> _ExactMatch:
    sources = ("semantic_scholar",) if kind == "doi" else ("semantic_scholar", "arxiv")
    return _ExactMatch(kind, identifier, sources, span)


def _reject_schemeless_paper_host_claims(text: str) -> None:
    if _SCHEMELESS_PAPER_HOST_RE.search(text):
        raise LookupInputError(
            "论文标识 URL 必须使用完整的官方 HTTP(S) 地址。",
            error_code="INVALID_ARGUMENTS",
        )


def _retained_prefix_identifier(value: str, kind: str) -> str | None:
    normalized = _strip_chinese_sentence_suffix(value).strip()
    prefix, separator, candidate = normalized.partition(":")
    if not separator or prefix.lower() != kind or not candidate:
        return None
    if kind == "doi":
        return _normalize_query_doi(candidate)
    identifier = _normalize_arxiv(candidate)
    return identifier if _ARXIV_IDENTIFIER_RE.fullmatch(identifier) else None


def _uri_claims_paper_identifier(value: str) -> bool:
    lowered = value.lower()
    return bool(
        "doi.org" in lowered
        or "arxiv.org" in lowered
        or re.search(r"(?i)10\.\d{4,9}/", value)
        or re.search(_ARXIV_IDENTIFIER_PATTERN, value, re.IGNORECASE)
    )


def _strip_chinese_sentence_suffix(value: str) -> str:
    positions = [value.find(marker) for marker in "，。；：！？、" if marker in value]
    return value[: min(positions)] if positions else value


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    masked = list(text)
    for start, end in spans:
        masked[start:end] = " " * (end - start)
    return "".join(masked)


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _bare_doi_identifiers(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    identifiers: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in _DOI_RE.finditer(text):
        value_span = match.span(1)
        token_start = _prefixed_token_start(text, value_span[0], "doi")
        if _has_invalid_identifier_left(text, token_start):
            raise LookupInputError(
                "DOI 标识必须从完整 token 边界开始。",
                error_code="INVALID_ARGUMENTS",
            )
        identifier = _normalize_query_doi(match.group(1))
        if identifier is None:
            raise LookupInputError(
                "DOI 标识规范化后必须仍为完整合法标识。",
                error_code="INVALID_ARGUMENTS",
            )
        identifiers.append(identifier)
        spans.append((token_start, value_span[1]))
    return identifiers, spans


def _bare_arxiv_identifiers(
    text: str, doi_spans: list[tuple[int, int]]
) -> list[tuple[str, tuple[int, int]]]:
    identifiers: list[tuple[str, tuple[int, int]]] = []
    for match in _ARXIV_RE.finditer(text):
        value_span = match.span(1)
        if any(_spans_overlap(value_span, doi_span) for doi_span in doi_spans):
            continue
        token_start = _prefixed_token_start(text, value_span[0], "arxiv")
        if _has_invalid_identifier_left(text, token_start) or _has_invalid_arxiv_tail(text, value_span[1]):
            raise LookupInputError(
                "arXiv 标识必须位于完整 token 边界内。",
                error_code="INVALID_ARGUMENTS",
            )
        identifier = _normalize_arxiv(match.group(1))
        if not _ARXIV_IDENTIFIER_RE.fullmatch(identifier):
            raise LookupInputError(
                "arXiv 标识规范化后必须仍为完整合法标识。",
                error_code="INVALID_ARGUMENTS",
            )
        identifiers.append((identifier, (token_start, value_span[1])))
    return identifiers


def _prefixed_token_start(text: str, value_start: int, prefix: str) -> int:
    prefix_match = re.search(rf"(?i){prefix}\s*:\s*$", text[:value_start])
    return prefix_match.start() if prefix_match is not None else value_start


def _has_invalid_identifier_left(text: str, token_start: int) -> bool:
    return not _is_allowed_identifier_start(text, token_start)


def _is_allowed_identifier_start(text: str, token_start: int) -> bool:
    if token_start <= 0:
        return True
    previous = text[token_start - 1]
    return bool(
        previous.isspace()
        or previous in "([{\"'（【“‘，。；：！？"
        or _is_cjk_character(previous)
    )


def _has_invalid_arxiv_tail(text: str, identifier_end: int) -> bool:
    if identifier_end >= len(text):
        return False
    next_character = text[identifier_end]
    if next_character.isspace():
        return False
    if next_character == "?":
        suffix = text[identifier_end + 1 :]
        immediate = suffix.split(maxsplit=1)[0] if suffix else ""
        if not immediate:
            return False
        return not (
            _is_cjk_character(immediate[0])
            and not any(character in immediate for character in "#&=%+/?")
        )
    if next_character == "#":
        return True
    if next_character in ",;:!?)]}\"'，。；：！？、）】’”":
        return False
    if next_character == ".":
        following = text[identifier_end + 1 : identifier_end + 2]
        return bool(following and not following.isspace() and following not in ")]},;:!?\"'")
    return True


def _is_unambiguous_bare_repository(value: str, *, allow_alphabetic: bool = False) -> bool:
    match = _GITHUB_REPO_RE.fullmatch(value.strip())
    if match is None:
        return False
    owner, repo = match.group(1), match.group(2)
    if any(not character.isalpha() for character in f"{owner}{repo}"):
        return True
    return allow_alphabetic and owner.islower() and repo.islower()


def _normalize_repository(value: str) -> str | None:
    match = _GITHUB_REPO_RE.fullmatch(value.strip())
    if match is None:
        return None
    owner, repo = match.group(1), match.group(2)
    if repo.lower().endswith(".git"):
        repo = repo[:-4]
    if not _valid_repository_parts(owner, repo):
        return None
    return f"{owner.lower()}/{repo.lower()}"


def _github_repository_from_claim(value: str) -> str | None:
    candidate = _unwrap_github_context_claim(_strip_github_context_terminator(value))
    if candidate is None:
        return None
    lowered = candidate.lower()
    if lowered.startswith(("http://", "https://")):
        return _github_repository_from_url(candidate)
    if lowered.startswith("github.com/"):
        return _github_repository_from_url(f"https://{candidate}")
    return _normalize_repository(candidate)


def _strip_github_context_terminator(value: str) -> str:
    """Remove only a complete claim's natural sentence terminator.

    The prefixed GitHub form is intentionally captured as one whole non-space
    token first.  Attached query, fragment, port, user-info or path data
    therefore remains in the token and cannot be hidden by a valid-looking
    owner/repo prefix.
    """
    candidate = value.strip()
    candidate = _strip_unpaired_terminal_delimiters(candidate)
    while candidate and candidate[-1] in ".,;:!?，。；：！？、":
        candidate = candidate[:-1]
        candidate = _strip_unpaired_terminal_delimiters(candidate)
    return candidate


def _github_context_matches(text: str) -> list[re.Match[str]]:
    return sorted(
        [*_GITHUB_CONTEXT_RE.finditer(text), *_GITHUB_PREFIXED_REPO_RE.finditer(text)],
        key=lambda match: (match.start(), match.end()),
    )


def _github_context_requires_exact_repository(text: str, match: re.Match[str]) -> bool:
    prefix = text[match.start() : match.start("claim")].casefold()
    claim = match.group("claim").strip()
    return bool(
        "/" in claim
        or re.search(r"(?:\brepository\b|\brepo\b|仓库)", prefix)
        or re.search(r"(?:\bis\b|为|[:：])\s*$", prefix)
    )


def _unwrap_github_context_claim(value: str) -> str | None:
    candidate = value.strip()
    if not candidate:
        return None
    if candidate[0] in {"\"", "'"}:
        quote = candidate[0]
        if len(candidate) < 2 or candidate[-1] != quote:
            return None
        candidate = candidate[1:-1]
        if not candidate or "\"" in candidate or "'" in candidate:
            return None
        return candidate
    if "\"" in candidate or "'" in candidate:
        return None
    return candidate


def _looks_like_invalid_repository_claim(value: str) -> bool:
    candidate = value.strip()
    if not candidate or any(character.isspace() for character in candidate):
        return False
    if candidate.count("/") != 1:
        return False
    owner, repo = candidate.split("/", 1)
    if not owner or not repo:
        return False
    return _normalize_repository(candidate) is None and (
        "." in owner
        or "_" in owner
        or owner.startswith("-")
        or owner.endswith("-")
        or "--" in owner
        or repo.startswith("-")
        or ".." in repo
        or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:\.git)?", repo)
    )


def _looks_like_invalid_complete_repository_path(value: str) -> bool:
    candidate = value.strip()
    if not candidate or any(character.isspace() for character in candidate):
        return False
    if "/" not in candidate:
        return False
    return (
        candidate.startswith("/")
        or candidate.endswith("/")
        or "//" in candidate
        or candidate.count("/") != 1
    )


def _github_repository_from_url(value: str) -> str | None:
    candidate = _strip_github_url_terminator(value)
    # urlsplit intentionally drops the distinction between an absent fragment
    # and a present-but-empty fragment.  For a repository identity, a literal
    # fragment delimiter is never natural sentence punctuation and therefore
    # must remain a hard failure even when the parsed fragment is empty.
    if not candidate or "#" in candidate:
        return None
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.lower() not in {"http", "https"}
        # Checking hostname alone loses an empty port separator (github.com:)
        # and other authority spelling.  Repository URLs permit exactly this
        # public authority, case-insensitively, with no credentials or port.
        or parsed.netloc.casefold() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    parts = parsed.path.split("/")
    if len(parts) != 3 or parts[0] or not parts[1] or not parts[2]:
        return None
    return _normalize_repository(f"{parts[1]}/{parts[2]}")


def _strip_github_url_terminator(value: str) -> str:
    """Strip only bounded natural-language punctuation after a complete URL.

    A final question mark may be a sentence terminator, matching the existing
    public query contract.  A hash is never stripped: even an empty fragment
    delimiter changes the URL token and must be rejected by the caller.
    Colons are likewise preserved so an empty authority port cannot disappear
    before the complete authority check.
    """

    candidate = value.strip()
    candidate = _strip_unpaired_terminal_delimiters(candidate)
    while candidate and candidate[-1] in ".,;!?，。；！？、":
        candidate = candidate[:-1]
        candidate = _strip_unpaired_terminal_delimiters(candidate)
    return candidate


def _valid_repository_parts(owner: str, repo: str) -> bool:
    return (
        owner not in {".", ".."}
        and repo not in {".", ".."}
        and ".." not in owner
        and ".." not in repo
        and not owner.startswith("-")
        and not owner.endswith("-")
        and "--" not in owner
        and "." not in owner
        and "_" not in owner
        and not repo.startswith("-")
    )
