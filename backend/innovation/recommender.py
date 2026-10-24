"""Evidence-grounded innovation recommendation helpers."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from backend.innovation.prompts import QUESTION_TEMPLATES
from backend.rag.retriever import tokenize


_TEMPLATE_SUGGESTION_LABEL = "模板启发／待验证方向"
_CASE_BACKGROUND_LABEL = "案例背景／待验证方向"


_EXTERNAL_TYPES = {"paper", "project", "vulnerability", "standard"}
_PREFERRED_TYPES = ["paper", "project", "competition", "standard", "vulnerability", "course_document"]
_GENERIC_TERMS = {
    "analysis",
    "case",
    "challenge",
    "course",
    "design",
    "evaluation",
    "experiment",
    "innovation",
    "method",
    "project",
    "research",
    "security",
    "study",
    "专题",
    "分析",
    "创新",
    "安全",
    "实验",
    "方法",
    "研究",
    "课程",
}
_TOPIC_ALIASES = (
    ("5g_nas", ("5g nas", "nas", "非接入层", "non-access stratum")),
    ("authentication", ("authentication", "auth", "认证", "鉴权", "身份验证")),
    ("replay", ("replay", "重放")),
    ("blockchain", ("blockchain", "区块链")),
    ("covert_channel", ("covert channel", "covert", "隐蔽通信", "隐蔽信道")),
    ("vulnerability", ("vulnerability", "cve", "漏洞")),
    ("llm", ("llm", "大模型", "语言模型")),
    ("standard", ("standard", "3gpp", "标准")),
)
_GENERIC_TERMS.update(
    {
        "5g",
        "mobile",
        "network",
        "radio",
        "secure",
        "survey",
        "system",
        "systems",
        "throughput",
        "wireless",
    }
)
_DOMAIN_ALIASES = {
    "5g_nas": ("5g nas", " nas ", "non-access stratum"),
    "5g_core": ("5g core", "open5gs"),
    "blockchain": ("blockchain",),
    "covert_channel": ("covert channel", "covert communication", "steganography"),
    "llm": ("llm", "large language model", "language model"),
    "network_security": ("network security", "ids", "intrusion detection"),
    "standard": ("standard", "3gpp", "ts 24.501"),
    "vulnerability": ("vulnerability", "vulnerabilities", "cve", "cwe"),
}
_ENTITY_ALIASES = {
    "3gpp": ("3gpp",),
    "nas": (" nas ", "5g nas", "non-access stratum"),
    "nvd": ("nvd",),
    "open5gs": ("open5gs",),
}
_TASK_ALIASES = {
    "authentication": ("authentication", "auth"),
    "baseline": ("baseline",),
    "covert": ("covert", "hidden channel"),
    "detection": ("detect", "detection", "detector"),
    "experiment": ("experiment", "experimental", "lab", "reproduce", "reproduction", "replication"),
    "fuzzing": ("fuzz", "fuzzing"),
    "handover": ("handover",),
    "replay": ("replay",),
    "vulnerability_analysis": ("vulnerability detection", "vulnerability analysis", "vulnerable code"),
}
_FIELD_WEIGHTS = {"title": 3.0, "summary": 1.4, "tags": 2.2, "module": 2.0}
_ANCHOR_WEIGHTS = {"domain": 3.0, "entity": 3.5, "task": 1.8, "generic": 0.2}
_MIN_RELEVANCE_SCORE = 2.5


def select_cases(
    resources: list[dict[str, Any]],
    *,
    topic: str,
    current_plan: str,
    limit: int = 6,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_type: dict[str, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    for order, item in enumerate(resources, start=1):
        case = _case_from_resource(item)
        if not _is_trusted_case(case):
            continue
        score, reasons = _topic_relevance_details(case, topic=topic, current_plan=current_plan)
        if score < _MIN_RELEVANCE_SCORE:
            continue
        connector_rank = _connector_rank(item, case, default=order)
        case["relevance_score"] = round(score, 4)
        case["relevance_reasons"] = reasons
        case["connector_rank"] = connector_rank
        case["topic_category"] = _topic_category(topic, case)
        by_type[case["type"]].append((score, connector_rank, case))

    for values in by_type.values():
        values.sort(key=lambda value: (-value[0], value[1], value[2]["title"]))

    seen: set[str] = set()
    seen_identity: set[tuple[str, str]] = set()
    for resource_type in _PREFERRED_TYPES:
        if by_type.get(resource_type):
            case = by_type[resource_type][0][2]
            selected.append(case)
            seen.add(case["source_id"])
            seen_identity.add(_case_identity(case))
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        remaining = sorted(
            (value for values in by_type.values() for value in values),
            key=lambda value: (-value[0], value[1], value[2]["title"]),
        )
        for _, _, case in remaining:
            identity = _case_identity(case)
            if case["source_id"] not in seen and identity not in seen_identity:
                selected.append(case)
                seen.add(case["source_id"])
                seen_identity.add(identity)
            if len(selected) >= limit:
                break
    return selected


def infer_limitations(topic: str, current_plan: str, cases: list[dict[str, Any]]) -> list[str]:
    limitations = []
    plan = current_plan.lower()
    if "指标" not in current_plan and "metric" not in plan:
        limitations.append("当前方案尚未明确评价指标，难以证明创新点有效。")
    if "对照" not in current_plan and "baseline" not in plan:
        limitations.append("当前方案缺少基线或对照组，实验结论可能不可复现。")
    if not any(case["type"] in {"paper", "standard"} for case in cases):
        limitations.append("当前证据缺少论文或标准来源，需要补充可定位依据。")
    if not any(case["type"] in {"project", "competition"} for case in cases):
        limitations.append("当前方案缺少可复现实作案例或竞赛题目参照。")
    if "公网" in current_plan or "真实目标" in current_plan:
        limitations.append("方案必须限定在授权、隔离、可审计的教学实验环境，不能面向公网目标执行。")
    return limitations[:5] or [f"{topic} 方案需要进一步明确研究假设、实验变量和证据来源。"]


def generate_questions(topic: str, current_plan: str, cases: list[dict[str, Any]], count: int = 5) -> list[str]:
    focus = topic or "当前专题"
    if cases:
        focus = cases[0].get("title") or focus
    questions = [template.format(focus=focus) for template in QUESTION_TEMPLATES]
    if "隐蔽" in topic or "covert" in current_plan.lower():
        questions.insert(0, "你的隐蔽通信方案如何同时评估吞吐量、可检测性和不可区分性？")
    if "LLM" in topic.upper() or "漏洞" in topic:
        questions.insert(0, "你的漏洞检测方案如何区分真实风险、误报和模型幻觉？")
    return _dedupe_text(questions)[: max(3, min(5, count))]


def generate_follow_up(answer: str, topic: str) -> str | None:
    if not answer.strip():
        return None
    if len(answer.strip()) < 30:
        return "请进一步说明你的评价指标、数据来源和预期对照组。"
    if "不知道" in answer or "不确定" in answer:
        return f"围绕 {topic}，你最需要系统帮助验证的是可行性、创新性还是实验安全边界？"
    return "基于你的回答，下一步你会优先验证哪个假设，准备用什么实验方法？"


def build_final_ideas(
    *,
    topic: str,
    current_plan: str,
    cases: list[dict[str, Any]],
    answers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence = [
        {
            "source_id": case["source_id"],
            "title": case["title"],
            "type": case["type"],
            "source": case["source"],
            "source_name": case.get("source_name", "unknown"),
            "origin_kind": case.get("origin_kind", "unknown"),
            "delivery_mode": case.get("delivery_mode", "unverified_history"),
            "fetched_at": case.get("fetched_at", ""),
            "external_id": case.get("external_id", ""),
            "provenance_verified": bool(case.get("provenance_verified", False)),
            "url": case.get("url", ""),
        }
        for case in cases[:4]
    ]
    answer_focus = "；".join(item.get("answer", "") for item in answers if item.get("answer"))[:240]
    ideas = [
        {
            "title": f"{topic} 的可复现实验评测框架",
            "hypothesis": "如果将方案拆成明确变量、基线和评价指标，就能更稳定地比较创新点收益。",
            "experiment_method": "在隔离教学环境中设计基线组、改进组和消融组，记录准确率、误报率、吞吐量、延迟或可检测性等指标。",
            "risks": ["样本规模过小导致结论不稳", "指标选择偏向当前方案", "不得迁移到未授权公网目标"],
            "evidence": evidence,
            "model_suggestion": False if evidence else True,
            "label": _CASE_BACKGROUND_LABEL if evidence else _TEMPLATE_SUGGESTION_LABEL,
            "generation_mode": "template",
            "hypotheses_verified": False,
        },
        {
            "title": f"{topic} 的安全边界与失效案例分析",
            "hypothesis": "通过系统记录失败案例和边界条件，可以形成比单点实现更有课程价值的创新方向。",
            "experiment_method": f"整理当前方案回答中的关键选择：{answer_focus or current_plan[:120]}，构造受控失败案例并分析原因。",
            "risks": ["失败案例可能暴露攻击性细节，需要只保留防御和评测视角", "结论必须引用可定位来源"],
            "evidence": evidence[:2],
            "model_suggestion": len(evidence) < 2,
            "label": _CASE_BACKGROUND_LABEL if len(evidence) >= 2 else _TEMPLATE_SUGGESTION_LABEL,
            "generation_mode": "template",
            "hypotheses_verified": False,
        },
    ]
    return ideas


def present_template_idea(item: dict[str, Any]) -> dict[str, Any]:
    """Project known template ideas safely without rewriting historical JSON."""

    presented = dict(item)
    presented["generation_mode"] = "template"
    presented["hypotheses_verified"] = False
    label = str(presented.get("label") or "").strip()
    if label in {"证据支持", _CASE_BACKGROUND_LABEL} and presented.get("evidence"):
        presented["label"] = _CASE_BACKGROUND_LABEL
    else:
        presented["label"] = _TEMPLATE_SUGGESTION_LABEL
    return presented


def _case_from_resource(item: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(item.get("metadata") or {})
    resource_id = str(item.get("resource_id") or f"{item.get('type')}:{item.get('external_id')}")
    resource_type = str(item.get("type") or "unknown")
    source_name = str(item.get("source_name") or "unknown")
    origin_kind = str(item.get("origin_kind") or "unknown")
    delivery_mode = str(item.get("delivery_mode") or "unknown")
    provenance_verified = bool(item.get("provenance_verified") or metadata.get("provenance_verified"))

    if resource_type == "course_document" and resource_id.startswith(("rag:", "document:")):
        origin_kind = "course_document"
        delivery_mode = "local_rag" if resource_id.startswith("rag:") else "local_document"
        provenance_verified = bool(metadata.get("course_id") is not None)
    elif resource_type == "competition" and source_name == "teacher_dataset" and resource_id.startswith("competition:"):
        origin_kind = "local_catalog"
        delivery_mode = "local_dataset"
        provenance_verified = True

    return {
        "source_id": resource_id,
        "type": resource_type,
        "title": item.get("title") or "",
        "summary": item.get("summary") or "",
        "source": item.get("source") or source_name,
        "source_name": source_name,
        "origin_kind": origin_kind,
        "delivery_mode": delivery_mode,
        "fetched_at": item.get("fetched_at") or "",
        "external_id": str(item.get("external_id") or ""),
        "provenance_verified": provenance_verified,
        "topic_category": item.get("topic_category") or "unknown",
        "url": item.get("url") or "",
        "tags": item.get("tags") or [],
        "cache_status": item.get("cache_status") or delivery_mode,
        "metadata": metadata,
        "relevance_score": item.get("relevance_score"),
        "relevance_reasons": list(item.get("relevance_reasons") or []),
        "connector_rank": item.get("connector_rank") or metadata.get("connector_rank") or item.get("rank"),
    }


def present_saved_case(item: dict[str, Any]) -> dict[str, Any]:
    """Add safe display defaults to historical JSON without changing stored data."""
    presented = dict(item)
    presented.setdefault("source_name", "unknown")
    presented.setdefault("origin_kind", "unknown")
    presented.setdefault("delivery_mode", "unverified_history")
    presented.setdefault("fetched_at", "")
    presented.setdefault("external_id", "")
    presented.setdefault("provenance_verified", False)
    presented.setdefault("topic_category", "unknown")
    presented.setdefault("url", "")
    presented.setdefault("relevance_score", None)
    presented.setdefault("relevance_reasons", [])
    presented.setdefault("connector_rank", None)
    return presented


def _is_trusted_case(case: dict[str, Any]) -> bool:
    if not case["title"] or not case["source_id"]:
        return False
    resource_type = case["type"]
    source_name = str(case["source_name"]).lower()
    if source_name in {"", "fake", "unknown"}:
        return False
    if resource_type in _EXTERNAL_TYPES:
        return bool(
            case["provenance_verified"]
            and case["origin_kind"] in {"remote", "local_catalog"}
            and case["delivery_mode"] in {"live", "fresh_cache", "stale_cache", "local_metadata"}
            and str(case["url"]).startswith(("https://", "http://"))
        )
    if resource_type == "course_document":
        return bool(
            case["provenance_verified"]
            and case["origin_kind"] == "course_document"
            and case["delivery_mode"] in {"local_rag", "local_document"}
            and case["source_id"].startswith(("rag:", "document:"))
        )
    if resource_type == "competition":
        return bool(
            case["provenance_verified"]
            and source_name == "teacher_dataset"
            and case["origin_kind"] == "local_catalog"
            and case["delivery_mode"] == "local_dataset"
            and case["source_id"].startswith("competition:")
        )
    return False


def _topic_relevance(case: dict[str, Any], *, topic: str, current_plan: str) -> float:
    score, _ = _topic_relevance_details(case, topic=topic, current_plan=current_plan)
    return score


def _topic_relevance_details(case: dict[str, Any], *, topic: str, current_plan: str) -> tuple[float, list[str]]:
    query_profile = _anchor_profile(f"{topic} {current_plan}")
    if not _profile_non_generic(query_profile):
        return 0.0, []

    score = 0.0
    reasons: list[str] = []
    non_generic_matches: set[str] = set()
    for field_name, field_text in _case_relevance_fields(case).items():
        field_profile = _anchor_profile(field_text)
        field_weight = _FIELD_WEIGHTS[field_name]
        for category, category_weight in _ANCHOR_WEIGHTS.items():
            matches = sorted(query_profile[category] & field_profile[category])
            if not matches:
                continue
            if category != "generic":
                non_generic_matches.update(f"{category}:{match}" for match in matches)
            contribution = field_weight * category_weight * len(matches)
            score += contribution
            reasons.append(f"{field_name}:{category}={','.join(matches)}(+{contribution:.1f})")

    if not non_generic_matches:
        return 0.0, []
    if not _passes_specificity_gate(query_profile, non_generic_matches):
        return 0.0, []
    if case.get("delivery_mode") == "local_rag":
        score += 0.4
        reasons.append("delivery:local_rag(+0.4)")
    return score, reasons[:8]


def _case_relevance_fields(case: dict[str, Any]) -> dict[str, str]:
    metadata = case.get("metadata") or {}
    return {
        "title": str(case.get("title") or ""),
        "summary": str(case.get("summary") or ""),
        "tags": " ".join(str(tag) for tag in case.get("tags") or []),
        "module": str(metadata.get("module") or case.get("topic_category") or ""),
    }


def _profile_non_generic(profile: dict[str, set[str]]) -> set[str]:
    return set().union(profile["domain"], profile["entity"], profile["task"])


def _passes_specificity_gate(query_profile: dict[str, set[str]], matches: set[str]) -> bool:
    matched_domains = {value.split(":", 1)[1] for value in matches if value.startswith("domain:")}
    matched_entities = {value.split(":", 1)[1] for value in matches if value.startswith("entity:")}
    matched_tasks = {value.split(":", 1)[1] for value in matches if value.startswith("task:")}
    query_domains = query_profile["domain"]
    query_entities = query_profile["entity"]

    if {"blockchain", "covert_channel"}.issubset(query_domains):
        return {"blockchain", "covert_channel"}.issubset(matched_domains)
    if {"5g_nas", "5g_core"}.intersection(query_domains | query_entities):
        return bool(
            {"5g_nas", "5g_core"}.intersection(matched_domains)
            or {"nas", "open5gs"}.intersection(matched_entities)
        )
    if "llm" in query_domains and "vulnerability" in query_domains:
        return "vulnerability" in matched_domains and (
            "llm" in matched_domains
            or "vulnerability_analysis" in matched_tasks
            or "detection" in matched_tasks
            or bool(matched_entities)
        )
    if "network_security" in query_domains:
        return "network_security" in matched_domains and bool(matched_tasks or matched_entities)
    if query_entities:
        return bool(query_entities & matched_entities or matched_domains)
    return bool(matched_domains or matched_entities or matched_tasks)


def _anchor_profile(text: str) -> dict[str, set[str]]:
    lowered = f" {str(text or '').lower()} "
    profile: dict[str, set[str]] = {
        "domain": set(),
        "entity": set(),
        "task": set(),
        "generic": set(),
    }
    _add_alias_matches(profile["domain"], lowered, _DOMAIN_ALIASES)
    _add_alias_matches(profile["entity"], lowered, _ENTITY_ALIASES)
    _add_alias_matches(profile["task"], lowered, _TASK_ALIASES)

    for cve in re.findall(r"\bcve-\d{4}-\d{4,8}\b", lowered):
        profile["entity"].add(cve)
        profile["domain"].add("vulnerability")
    for standard in re.findall(r"\b(?:3gpp|ts\s*\d{2}\.\d{3})\b", lowered):
        profile["entity"].add(re.sub(r"\s+", " ", standard).strip())
        profile["domain"].add("standard")
    for repo in re.findall(r"(?<![a-z0-9_.-])([a-z0-9_.-]+/[a-z0-9_.-]+)(?![a-z0-9_.-])", lowered):
        profile["entity"].add(repo)

    tokens = {
        token.lower()
        for token in tokenize(lowered)
        if token.lower() not in _GENERIC_TERMS and re.search(r"[\w\u4e00-\u9fff]", token)
    }
    tokens.update(
        token
        for token in re.findall(r"[a-z0-9][a-z0-9._-]*", lowered)
        if token not in _GENERIC_TERMS
    )
    for token in tokens:
        if token in {"open5gs", "nvd"}:
            profile["entity"].add(token)
        elif token == "nas":
            profile["entity"].add("nas")
            profile["domain"].add("5g_nas")
        elif token == "llm":
            profile["domain"].add("llm")
        elif token == "blockchain":
            profile["domain"].add("blockchain")
        elif token in {"cve", "cwe"}:
            profile["domain"].add("vulnerability")
        elif _looks_like_specific_entity(token):
            profile["entity"].add(token)
    for token in re.findall(r"[a-z0-9][a-z0-9._-]*", lowered):
        if token in _GENERIC_TERMS:
            profile["generic"].add(token)
    return profile


def _looks_like_specific_entity(token: str) -> bool:
    if token in _GENERIC_TERMS or len(token) < 4:
        return False
    return bool(re.search(r"\d", token) or "_" in token or "." in token or "-" in token)


def _add_alias_matches(target: set[str], lowered: str, aliases: dict[str, tuple[str, ...]]) -> None:
    for canonical, values in aliases.items():
        for alias in values:
            needle = alias.lower()
            if needle in lowered:
                target.add(canonical)
                break


def _connector_rank(item: dict[str, Any], case: dict[str, Any], *, default: int) -> int:
    metadata = case.get("metadata") or {}
    for value in (
        case.get("connector_rank"),
        item.get("connector_rank"),
        item.get("rank"),
        metadata.get("connector_rank"),
        metadata.get("rank"),
    ):
        try:
            if value is not None and str(value).strip() != "":
                return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _case_identity(case: dict[str, Any]) -> tuple[str, str]:
    normalized_title = " ".join(str(case.get("title") or "").lower().split())
    return str(case.get("type") or "unknown"), normalized_title


def _semantic_terms(text: str) -> set[str]:
    lowered = " ".join(str(text or "").lower().split())
    terms = {
        token.lower()
        for token in tokenize(lowered)
        if token.lower() not in _GENERIC_TERMS and re.search(r"[\w\u4e00-\u9fff]", token)
    }
    terms.update(
        token
        for token in re.findall(r"[a-z0-9][a-z0-9._-]*", lowered)
        if token not in _GENERIC_TERMS
    )
    for canonical, aliases in _TOPIC_ALIASES:
        if any(alias in lowered for alias in aliases):
            terms.add(canonical)
    return terms


def _topic_category(topic: str, case: dict[str, Any]) -> str:
    terms = _semantic_terms(f"{topic} {case.get('title', '')} {' '.join(case.get('tags') or [])}")
    if "5g_nas" in terms:
        return "5G NAS"
    if "blockchain" in terms and "covert_channel" in terms:
        return "区块链隐蔽通信"
    if "llm" in terms and "vulnerability" in terms:
        return "LLM 漏洞检测"
    if "vulnerability" in terms:
        return "漏洞分析"
    if "standard" in terms:
        return "标准规范"
    return "其他"


def _dedupe_text(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
