"""Small, side-effect-free presentation mappings for the application shell."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


PRODUCT_NAME = "网络安全课程智能体"
PRODUCT_SUBTITLE = "课程问答、资料查证、实践指导与创新探索"

PAGE_LABELS = {
    "student": "课程学习",
    "innovation": "创新探索",
    "teacher": "教学管理",
    "tools": "运行状态",
    "about": "使用帮助",
}
ROLE_LABELS = {"student": "学生", "teacher": "教师", "admin": "管理员"}
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_COURSE_REFERENCE_STATUSES = frozenset({"allowed", "disabled_by_user", "blocked_by_policy", "revoked"})
_ERROR_LABELS = {
    "USER_DISABLED": "账号不可用，请联系管理员。",
    "TOKEN_REVOKED": "登录已失效，请重新登录。",
    "SESSION_ENDED": "学习会话已结束，请重新进入课程。",
    "REFERENCE_CONTEXT_CHANGED": "课程或资料状态已变化，请刷新后重试。",
    "MODEL_BUSY": "当前请求较多，请稍后重试。",
    "DOCUMENT_BUSY": "资料正在处理，请稍后重试。",
}
_TOOL_LABELS = {
    "course_retriever": "课程资料检索",
    "paper_search": "本地论文查询",
    "vuln_lookup": "本地漏洞查询",
    "standard_lookup": "本地标准查询",
    "mermaid_generator": "流程图生成",
    "paper_search_real": "公开论文查询",
    "vuln_lookup_real": "公开漏洞查询",
    "standard_lookup_real": "公开标准查询",
    "github_project_search": "开源项目查询",
}


def navigation_pages(role: str | None) -> list[str]:
    pages = ["student", "innovation"]
    if role in {"teacher", "admin"}:
        pages.extend(("teacher", "tools"))
    pages.append("about")
    return pages


def page_label(page_id: str) -> str:
    return PAGE_LABELS.get(page_id, "使用帮助")


def role_label(role: str | None) -> str:
    return ROLE_LABELS.get(role or "", "身份待核实")


def safe_request_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _REQUEST_ID.fullmatch(value) else None


def safe_course_reference_status(value: Any) -> str | None:
    return value if isinstance(value, str) and value in _COURSE_REFERENCE_STATUSES else None


def safe_error_message(
    *, status_code: int | None = None, error_code: str | None = None,
    retry_after: int | None = None, request_id: str | None = None,
    course_reference_status: str | None = None,
) -> str:
    checked_reference_status = safe_course_reference_status(course_reference_status)
    if error_code == "MODEL_CONTEXT_NOT_AUTHORIZED" and checked_reference_status == "disabled_by_user":
        message = "本次未允许课程片段用于模型回答。"
    elif error_code == "MODEL_CONTEXT_NOT_AUTHORIZED" and checked_reference_status == "blocked_by_policy":
        message = "当前课程未获许可，不能将课程片段交给外部模型。"
    elif error_code == "MODEL_CONTEXT_NOT_AUTHORIZED" and checked_reference_status == "revoked":
        message = "课程资料授权已失效，请刷新后重试。"
    elif isinstance(error_code, str) and error_code in _ERROR_LABELS:
        message = _ERROR_LABELS[error_code]
    elif status_code == 401:
        message = "登录已失效，请重新登录。"
    elif status_code == 403:
        message = "当前账号无权执行此操作。"
    elif status_code == 404:
        message = "所需内容未找到或已不可用。"
    elif status_code == 409:
        message = "当前状态已变化，请刷新后重试。"
    elif status_code == 422:
        message = "输入内容不符合要求，请检查后重试。"
    elif status_code == 429:
        message = "请求过于频繁，请稍后重试。"
    elif status_code in {502, 503, 504}:
        message = "服务暂不可用，请稍后重试。"
    else:
        message = "操作未能完成，请稍后重试或联系管理员。"
    if status_code == 429 and isinstance(retry_after, int) and 0 < retry_after <= 86400:
        message += f" 建议等待 {retry_after} 秒。"
    checked_id = safe_request_id(request_id)
    if checked_id:
        message += f" 反馈编号：{checked_id}。"
    return message


def health_rows(status: Any) -> list[dict[str, str]]:
    data = status if isinstance(status, dict) else {}
    rows = []
    for field, label in (("sqlite", "资料记录"), ("chroma", "课程检索"), ("mcp", "工具连接")):
        item = data.get(field)
        state = item.get("status") if isinstance(item, dict) else None
        state_label = {
            "ok": "正常", "loading": "处理中", "fallback": "降级运行",
            "warning": "需要关注", "error": "暂不可用",
        }.get(state, "状态待核实")
        rows.append({"组件": label, "状态": state_label})
    return rows


def tool_rows(tools: Any) -> list[dict[str, str]]:
    if not isinstance(tools, list):
        return []
    rows = []
    for item in tools:
        if not isinstance(item, dict) or item.get("name") not in _TOOL_LABELS:
            continue
        enabled = item.get("enabled")
        rows.append({
            "工具": _TOOL_LABELS[item["name"]],
            "启用状态": "已启用" if enabled is True else "未启用" if enabled is False else "状态待核实",
        })
    return rows


def model_status(status: Any) -> str:
    data = status if isinstance(status, dict) else {}
    return {"deepseek": "模型服务", "template": "本地模板"}.get(data.get("provider"), "生成方式待核实")


def event_label(value: Any) -> str:
    return {
        "PAGE_VIEW": "查看页面", "LOGIN": "登录", "RESOURCE_SEARCH": "搜索资料",
        "RESOURCE_OPEN": "查看资料", "PRACTICE_GUIDE_GENERATED": "生成实践指导",
        "INNOVATION_SUGGESTION_GENERATED": "生成创新建议",
    }.get(value, "其他学习事件")


def module_label(value: Any) -> str:
    return {
        "course_qa": "课程问答", "practice": "实践指导", "innovation": "创新探索",
        "resource": "资料查证", "general": "通用问答",
    }.get(value, "其他模块")


def progress_label(value: Any) -> str:
    normalized = normalize_progress_status(value)
    return {
        "not_started": "未开始", "in_progress": "进行中", "completed": "已完成",
        "stuck": "遇到困难", "active": "待处理", "reported": "待处理", "confirmed": "已确认",
        "resolved": "已解决", "ignored": "已忽略",
    }.get(normalized, "状态待核实")


def normalize_progress_status(value: Any) -> str:
    """Normalize only bounded string status values; malformed DTO values stay unknown."""
    if type(value) is not str:
        return ""
    normalized = value.strip().casefold()
    return normalized if 0 < len(normalized) <= 32 else ""


def agent_action_label(value: Any) -> str:
    return {
        "RETRIEVE": "检索资料", "ANSWER": "形成回答", "ASK_USER": "请求补充",
        "CALL_TOOL": "调用工具", "ABSTAIN": "拒绝回答", "CONTINUE": "继续处理",
    }.get(value, "其他处理环节")


def agent_stop_label(value: Any) -> str:
    return {
        "ANSWERED": "已回答", "ABSTAINED": "已拒答", "MAX_STEPS": "达到步骤上限",
        "REQUIRES_INPUT": "等待补充", "COMPLETED": "已完成", "FAILED": "处理失败",
        "LIMIT_REACHED": "达到上限", "CANCELLED": "已取消", "TIMEOUT": "处理超时",
    }.get(value, "其他结束状态")


def teacher_tool_label(value: Any) -> str:
    return _TOOL_LABELS.get(value, "其他工具")


def delivery_label(value: Any) -> str:
    return {
        "course_published": "已发布课程资料", "local_rag": "当前课程资料",
        "live": "本次联网获取", "fresh_cache": "有效缓存",
        "stale_cache": "过期缓存（请核对）", "local": "本地资料",
        "cache_first": "课程资料与已核验缓存", "failed": "暂不可用",
    }.get(str(value or "").lower(), "状态待核实")


def lookup_label(value: Any) -> str:
    return {
        "FOUND": "已找到", "SUCCESS": "已找到", "PARTIAL": "部分结果",
        "NOT_FOUND": "未找到", "UNAVAILABLE": "暂不可用", "FAILED": "暂不可用",
        "available": "当前可用", "unavailable": "暂不可用",
    }.get(str(value or ""), "状态待核实")


def source_type_label(value: Any) -> str:
    return {
        "course_document": "课程文档", "paper": "论文", "standard": "标准",
        "vulnerability": "漏洞", "project": "开源项目", "competition": "竞赛题目",
    }.get(str(value or "").lower(), "资料类型待核实")


def source_name_label(value: Any) -> str:
    return {
        "semantic_scholar": "Semantic Scholar", "arxiv": "arXiv",
        "nvd": "NVD", "github": "GitHub", "rfc_editor": "RFC Editor",
        "standard_source": "本地标准目录", "course_document": "当前课程资料",
        "local_catalog": "本地目录", "competition": "教师维护的竞赛资料",
    }.get(str(value or "").lower(), "来源待核实")


def reference_status_label(value: Any) -> str:
    return {
        "used": "已使用当前课程参考", "none": "课程内未找到可用参考",
        "disabled": "本次未选择课程参考", "blocked_by_policy": "课程政策禁止外发参考",
        "unavailable": "课程参考暂不可用", "allowed": "本次允许使用课程参考",
        "disabled_by_user": "本次未选择课程参考", "revoked": "课程参考已撤回",
    }.get(str(value or "").lower(), "参考状态待核实")


@dataclass(frozen=True)
class HotspotTopic:
    cluster_id: int
    label: str
    representative_question: str
    count: int
    approved: bool
    recent_question_at: str | None
    members: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class HotspotView:
    status: str
    message: str | None
    topic_count: int | None
    pending_count: int | None
    approved_count: int | None
    exportable_count: int | None
    noise_question_count: int | None
    noise_members: tuple[dict[str, Any], ...]
    topics: tuple[HotspotTopic, ...]
    faq_by_cluster: dict[int, dict[str, Any]]


def _hotspot_unavailable(status: str, message: str) -> HotspotView:
    return HotspotView(status, message, None, None, None, None, None, (), (), {})


def build_hotspot_view_model(summary: Any) -> HotspotView:
    """Project one authorized summary; never repair inconsistent cluster/FAQ reads."""
    if not isinstance(summary, dict):
        return _hotspot_unavailable("unavailable", "热点资料暂不可用，请稍后刷新")
    clusters = summary.get("clusters")
    faq = summary.get("faq")
    cluster_items = clusters.get("items") if isinstance(clusters, dict) else None
    faq_items = faq.get("items") if isinstance(faq, dict) else None
    if not isinstance(cluster_items, list) or not isinstance(faq_items, list):
        return _hotspot_unavailable("unavailable", "热点资料暂不可用，请稍后刷新")

    topics: list[HotspotTopic] = []
    by_id: dict[int, HotspotTopic] = {}
    noise_count = 0
    noise_members: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for item in cluster_items:
        if not isinstance(item, dict):
            return _hotspot_unavailable("inconsistent", "资料状态暂不同步，请刷新")
        cluster_id = item.get("id")
        members = item.get("members")
        if (not isinstance(cluster_id, int) or isinstance(cluster_id, bool) or cluster_id <= 0
                or cluster_id in seen_ids or not isinstance(members, list)
                or not isinstance(item.get("approved"), bool)
                or not isinstance(item.get("is_noise"), bool)):
            return _hotspot_unavailable("inconsistent", "资料状态暂不同步，请刷新")
        seen_ids.add(cluster_id)
        if item["is_noise"]:
            if any(not isinstance(member, dict) for member in members):
                return _hotspot_unavailable("inconsistent", "资料状态暂不同步，请刷新")
            noise_count += len(members)
            noise_members.extend({"question": member.get("question"), "created_at": member.get("created_at")}
                                 for member in members if isinstance(member, dict))
            continue
        if not members:
            continue
        if any(not isinstance(member, dict) for member in members):
            return _hotspot_unavailable("inconsistent", "资料状态暂不同步，请刷新")
        timestamps = [member.get("created_at") for member in members if isinstance(member.get("created_at"), str)]
        topic = HotspotTopic(
            cluster_id=cluster_id,
            label=str(item.get("label") or "未命名主题"),
            representative_question=str(item.get("representative_question") or ""),
            count=len(members),
            approved=item["approved"],
            recent_question_at=max(timestamps) if timestamps else None,
            members=tuple({
                "question_id": member.get("question_id"),
                "question": member.get("question"),
                "created_at": member.get("created_at"),
                "is_representative": member.get("is_representative") is True,
            } for member in members),
        )
        topics.append(topic)
        by_id[cluster_id] = topic

    faq_by_cluster: dict[int, dict[str, Any]] = {}
    for item in faq_items:
        if not isinstance(item, dict):
            return _hotspot_unavailable("inconsistent", "资料状态暂不同步，请刷新")
        cluster_id = item.get("cluster_id")
        topic = by_id.get(cluster_id) if isinstance(cluster_id, int) else None
        citations = item.get("citations")
        if not topic or not topic.approved or not isinstance(citations, list) or not citations:
            return _hotspot_unavailable("inconsistent", "资料状态暂不同步，请刷新")
        for field, expected in (("label", topic.label), ("representative_question", topic.representative_question),
                                ("count", topic.count)):
            if item.get(field) != expected:
                return _hotspot_unavailable("inconsistent", "资料状态暂不同步，请刷新")
        if not isinstance(item.get("answer"), str) or not item["answer"].strip():
            return _hotspot_unavailable("inconsistent", "资料状态暂不同步，请刷新")
        citation_by_id: dict[str, dict[str, Any]] = {}
        for citation in citations:
            if (not isinstance(citation, dict) or not isinstance(citation.get("source_id"), str)
                    or not citation["source_id"].strip() or not isinstance(citation.get("snippet"), str)
                    or not citation["snippet"].strip()):
                return _hotspot_unavailable("inconsistent", "资料状态暂不同步，请刷新")
            existing = citation_by_id.get(citation["source_id"])
            if existing is not None and existing != citation:
                return _hotspot_unavailable("inconsistent", "资料状态暂不同步，请刷新")
            citation_by_id[citation["source_id"]] = citation
        if cluster_id in faq_by_cluster and faq_by_cluster[cluster_id] != item:
            return _hotspot_unavailable("inconsistent", "资料状态暂不同步，请刷新")
        faq_by_cluster[cluster_id] = item.copy()

    topics.sort(key=lambda topic: (-topic.count, topic.cluster_id))
    return HotspotView(
        status="ready", message=None,
        topic_count=len(topics),
        pending_count=sum(not topic.approved for topic in topics),
        approved_count=sum(topic.approved for topic in topics),
        exportable_count=len(faq_by_cluster),
        noise_question_count=noise_count,
        noise_members=tuple(noise_members),
        topics=tuple(topics), faq_by_cluster=faq_by_cluster,
    )
