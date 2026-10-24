"""Student-facing Streamlit page."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import streamlit as st

from frontend import api_client
from frontend.api_client import APIClientError
from frontend.presentation import (
    PRODUCT_NAME, PRODUCT_SUBTITLE, agent_action_label, agent_stop_label,
    delivery_label, lookup_label, module_label, reference_status_label,
    source_name_label, source_type_label, teacher_tool_label,
)
from frontend.session_scope import ensure_student_scope

TOPICS = {
    "课程智能体": "这个课程智能体能帮我做什么？",
    "5G NAS": "5G NAS 认证流程有哪些关键风险点？",
    "区块链隐蔽通信": "我想做基于区块链的隐蔽通信，怎么设计实验？",
    "LLM 漏洞检测": "帮我找和 LLM 漏洞检测相关的研究方向。",
    "漏洞样例": "有没有可以作为漏洞检测样例的 CVE？",
    "标准样例": "3GPP TS 24.501 里 NAS 安全相关内容在哪里？",
}

STATUS_LABELS = {
    "COMPLETED": "已完成",
    "REQUIRES_INPUT": "等待补充",
    "ABSTAINED": "已拒答",
    "LIMIT_REACHED": "达到上限",
    "FAILED": "失败",
    "RUNNING": "运行中",
}

MAX_CHAT_HISTORY = 5


def render() -> None:
    if not _auth_token():
        st.info("请先登录后使用学生端。")
        return
    if _must_change_password():
        st.warning("首次登录需要先修改默认密码，完成后才能使用学生端。")
        return
    _render_header()
    _render_backend_status()
    scope = ensure_student_scope()
    if scope is None:
        return
    course_id, class_id = scope
    use_course_references = st.checkbox(
        "允许本次请求将我选择的最小课程片段交给外部模型处理",
        value=False,
        key="use_course_references",
        help="仅在课程负责人已将该课程加入外发许可清单时生效；问题文本和选中片段会由外部模型处理。",
    )

    tabs = st.tabs(["使用说明", "课程有据问答", "通用学习问答", "课程资料库", "实践指导", "实验任务"])
    with tabs[0]:
        _render_course_intro()
    with tabs[1]:
        _render_topic_navigation()
        _render_chat(course_id, class_id, use_course_references)
    with tabs[2]:
        _render_general_chat(course_id, class_id, use_course_references)
    with tabs[3]:
        _render_resource_portal(course_id, class_id)
    with tabs[4]:
        _render_practice_guide(course_id, class_id, use_course_references)
        st.divider()
        _render_practice_draft(course_id, class_id, use_course_references)
    with tabs[5]:
        _render_assignments(course_id, class_id)


def _render_header() -> None:
    st.title(PRODUCT_NAME)
    st.caption(PRODUCT_SUBTITLE)


def _render_course_intro() -> None:
    st.subheader("使用说明")
    st.markdown(
        """
        **你可以这样使用智能体：**

        - 在“课程有据问答”中提出课程概念、论文方向、标准条款位置和实验设计问题。
        - 在“课程资料库”中检索课程文档、论文、标准、漏洞、开源项目和竞赛题目。
        - 在“实践指导”中按“研究目标 -> 方法设计 -> 工具选择 -> 实验步骤 -> 结果分析 -> 创新点”的框架生成方案。
        - 在“实验任务”中选择当前步骤、更新进度、上报卡点；问答和实践指导会携带当前实验步骤上下文。
        - 当回答证据不足时，系统会拒答或追问，不会编造论文、CVE 或标准条款。
        """
    )

    st.subheader("课程简介")
    st.write("本课程围绕 5G NAS、区块链隐蔽通信、LLM 漏洞检测与安全评测等专题，训练学生使用课程资料、完成可复现实验并提出创新方案。")

    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            """
            **课程目标**

            - 理解协议、标准、漏洞和论文资料之间的证据关系。
            - 学会使用本地知识库和外部资料接口做可定位检索。
            - 能把研究问题拆解为目标、方法、工具、实验和分析。
            - 在安全边界内提出可验证的创新点。
            """
        )
    with cols[1]:
        st.markdown(
            """
            **考核方式**

            - 过程表现：资料检索、提问质量、进度记录和卡点反馈。
            - 实验报告：实验设计、证据引用、结果分析和复现说明。
            - 创新方案：问题价值、验证假设、实验方法、风险控制和引用来源。
            """
        )

    st.subheader("专题示例")
    st.write("5G NAS 认证风险、基于区块链的隐蔽通信、LLM 漏洞检测、标准条款定位、CVE 样例分析、开源安全项目复现。")

    st.subheader("网络安全实验边界")
    st.warning(
        "本系统只支持授权教学、资料检索、防御分析和实验设计。不得请求对公网目标的攻击步骤、凭据窃取、持久化控制、绕过检测或真实利用链执行。"
    )


def _render_backend_status() -> None:
    try:
        status = api_client.system_status()
    except APIClientError:
        st.error("学习服务暂不可用，请稍后重试或联系课程管理员。")
        return
    if status.get("fallback_active") is True:
        st.warning("当前为本地模板生成，不能视为外部模型回答；以每次回答显示的身份为准。")
    elif status.get("provider") == "deepseek":
        st.info("已选择外部模型服务；是否实际完成生成，请以本次回答状态为准。")
    else:
        st.info("生成方式待核实；请以本次回答状态为准。")


def _render_topic_navigation() -> None:
    st.subheader("专题导航")
    selected_topic = st.radio("选择课程专题", list(TOPICS), horizontal=True, label_visibility="collapsed")
    st.session_state.setdefault("question_input", TOPICS[selected_topic])

    col_a, col_b = st.columns([0.72, 0.28])
    with col_a:
        st.info(TOPICS[selected_topic])
    with col_b:
        if st.button("填入问答框", use_container_width=True):
            st.session_state.question_input = TOPICS[selected_topic]


def _render_course_selector() -> int | None:
    token = _auth_token()
    if not token:
        return None
    try:
        courses = api_client.courses(token)
    except APIClientError as exc:
        st.error(str(exc))
        return None
    if not courses:
        st.warning("当前账号没有可访问的课程。")
        return None
    options = {int(item["id"]): item for item in courses}
    current = st.session_state.get("selected_course_id")
    labels = list(options)
    default_index = next(
        (index for index, item_id in enumerate(labels) if item_id == current),
        0,
    )
    selected_course_id = st.selectbox(
        "当前课程", labels, index=default_index, key="student_course_selector",
        format_func=lambda item_id: str(options[item_id].get("name") or options[item_id].get("slug") or "未命名课程"),
    )
    if current is not None and int(current) != selected_course_id:
        for key in (
            "last_chat_result",
            "last_guide_result",
            "last_practice_draft_result",
            "chat_result_history",
            "resource_search_payload",
            "resource_search_fingerprint",
            "resource_page",
            "assignment_context",
            "innovation_session",
        ):
            st.session_state.pop(key, None)
    st.session_state.selected_course_id = selected_course_id
    return selected_course_id


def _render_chat(course_id: int, class_id: int | None, use_course_references: bool) -> None:
    st.subheader("课程有据问答")
    with st.form("chat_form", clear_on_submit=False):
        question = st.text_area("输入课程问题", key="question_input", height=120, placeholder="例如：5G NAS 认证流程有哪些关键风险点？")
        submitted = st.form_submit_button("提交问题", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("请输入一个问题。")
            return
        try:
            with st.spinner("正在查找资料并形成回答…"):
                result = api_client.chat(
                    question.strip(),
                    token=_auth_token(),
                    client_session_id=_client_session_id(),
                    assignment_id=_assignment_context().get("assignment_id"),
                    step_id=_assignment_context().get("step_id"),
                    course_id=course_id,
                    class_id=class_id,
                    use_course_references=use_course_references,
                )
        except APIClientError as exc:
            st.error(str(exc))
            return
        st.session_state.last_chat_result = result
        _remember_answer_result(result, course_id)

    result = st.session_state.get("last_chat_result")
    if result:
        _render_answer(result)
        _render_continue_if_needed(result)
    _render_answer_history(course_id, active_run_id=result.get("run_id") if result else None)


def _render_general_chat(course_id: int, class_id: int | None, use_course_references: bool) -> None:
    st.subheader("通用学习问答")
    st.caption(
        "你的问题会发送给外部模型。只有课程已获许可且你勾选本次使用时，系统才会附加最小课程片段；请勿自行粘贴未获许可的敏感正文。"
    )
    with st.form("general_chat_form", clear_on_submit=False):
        question = st.text_area(
            "输入学习问题",
            key="general_question_input",
            height=120,
            placeholder="例如：请用直观方式解释勾股定理。",
        )
        submitted = st.form_submit_button("生成通用回答", use_container_width=True)
    if submitted:
        if not question.strip():
            st.warning("请输入一个问题。")
            return
        try:
            with st.spinner("外部模型正在生成回答..."):
                result = api_client.general_chat(
                    _auth_token(),
                    question.strip(),
                    course_id=course_id,
                    class_id=class_id,
                    use_course_references=use_course_references,
                    client_session_id=_client_session_id(),
                )
        except APIClientError as exc:
            st.error(str(exc))
            return
        st.session_state.last_general_chat_result = result

    result = st.session_state.get("last_general_chat_result")
    if not result:
        return
    provider = str(result.get("provider") or "").lower()
    if result.get("status") == "refused":
        status = "安全拒绝"
    elif result.get("status") == "answered" and provider == "deepseek" and result.get("answer_kind") == "model_generated":
        status = "外部模型生成"
    elif result.get("status") == "answered" and provider == "template":
        status = "本地模板回答"
    else:
        status = "回答身份待核实"
    st.caption(f"回答身份：{status}｜参考：{reference_status_label(result.get('reference_status'))}")
    st.text(f"模型：{_safe_model_name(result.get('model'))}")
    st.text(str(result.get("answer") or ""))
    materials = result.get("reference_materials") or []
    if materials:
        with st.expander("本次供模型参考的课程片段", expanded=False):
            for item in materials:
                st.text(f"标题：{item.get('title') or '课程参考'}")
                st.text(str(item.get("excerpt") or ""))
                st.caption("仅作模型上下文，未经逐句核验。")
    for warning in result.get("warnings") or []:
        st.text(f"提醒：{warning}")


def _render_continue_if_needed(result: dict[str, Any]) -> None:
    if result.get("run_status") != "REQUIRES_INPUT" or not result.get("run_id"):
        return
    st.info("需要补充信息后继续本次回答。")
    st.text(str(result.get("pending_question") or "请补充问题背景。"))
    with st.form("continue_form", clear_on_submit=True):
        user_input = st.text_area("补充信息", height=90)
        submitted = st.form_submit_button("继续本次回答", use_container_width=True)
    if not submitted:
        return
    if not user_input.strip():
        st.warning("请输入补充信息。")
        return
    try:
        with st.spinner("正在继续原任务..."):
            continued = api_client.continue_agent_run(
                result["run_id"],
                user_input.strip(),
                token=_auth_token(),
                client_session_id=_client_session_id(),
                assignment_id=result.get("assignment_id") or _assignment_context().get("assignment_id"),
                step_id=result.get("step_id") or _assignment_context().get("step_id"),
                use_course_references=bool(st.session_state.get("use_course_references", False)),
            )
    except APIClientError as exc:
        st.error(str(exc))
        return
    st.session_state.last_chat_result = continued
    _remember_answer_result(continued, st.session_state.get("selected_course_id"))
    st.rerun()


def _render_practice_guide(course_id: int, class_id: int | None, use_course_references: bool) -> None:
    st.subheader("实践指导")
    with st.form("practice_form", clear_on_submit=False):
        topic = st.text_input("输入实践专题", value="基于区块链的隐蔽通信", placeholder="例如：基于区块链的隐蔽通信")
        submitted = st.form_submit_button("生成实践方案", use_container_width=True)

    if submitted:
        if not topic.strip():
            st.warning("请输入一个实践专题。")
            return
        try:
            with st.spinner("正在生成结构化实践方案..."):
                result = api_client.practice_guide(
                    topic.strip(),
                    token=_auth_token(),
                    client_session_id=_client_session_id(),
                    assignment_id=_assignment_context().get("assignment_id"),
                    step_id=_assignment_context().get("step_id"),
                    course_id=course_id,
                    class_id=class_id,
                    use_course_references=use_course_references,
                )
        except APIClientError as exc:
            st.error(str(exc))
            return
        st.session_state.last_guide_result = result
        _remember_answer_result(result, course_id)

    result = st.session_state.get("last_guide_result")
    if result:
        _render_answer(result)
        _render_continue_if_needed(result)


def _render_practice_draft(course_id: int, class_id: int | None, use_course_references: bool) -> None:
    st.subheader("探索草案")
    st.caption(
        "表单内容会发送给外部模型；系统附加课程片段仅在课程允许且你本次勾选后发送。"
        "请勿自行粘贴未获许可的敏感正文。草案是待验证建议，不代表实验结果或资料已经证明方案正确。"
    )
    with st.form("practice_draft_form", clear_on_submit=False):
        topic = st.text_input("探索主题", key="practice_draft_topic")
        objective = st.text_area("学习目标", key="practice_draft_objective", height=90)
        environment = st.text_area("隔离实验环境", key="practice_draft_environment", height=80)
        tools_text = st.text_input("可用工具（用逗号分隔，最多10项）", key="practice_draft_tools")
        external_query = st.text_input(
            "可选公开资料查询（留空则不查询外部来源）",
            key="practice_draft_external_query",
        )
        external_type = st.selectbox(
            "公开资料类型",
            ["paper", "vulnerability", "project", "standard"],
            key="practice_draft_external_type",
        )
        allow_external = st.checkbox(
            "允许联网查询并确认以上查询词",
            value=False,
            key="practice_draft_allow_external",
            help="默认关闭；只有勾选后，系统才会按上方明确查询词访问已启用的公开资料来源。",
        )
        submitted = st.form_submit_button("生成待验证探索草案", use_container_width=True)
    if submitted:
        tools = [item.strip() for item in tools_text.replace("，", ",").split(",") if item.strip()]
        if not topic.strip() or not objective.strip() or not environment.strip():
            st.warning("请完整填写主题、学习目标和隔离实验环境。")
            return
        queries = _practice_draft_external_queries(
            allowed=allow_external,
            resource_type=external_type,
            query=external_query,
        )
        try:
            with st.spinner("正在核对背景资料并生成待验证草案..."):
                result = api_client.practice_draft(
                    _auth_token(),
                    topic=topic.strip(),
                    objective=objective.strip(),
                    lab_environment=environment.strip(),
                    available_tools=tools,
                    course_id=course_id,
                    class_id=class_id,
                    use_course_references=use_course_references,
                    external_queries=queries,
                    client_session_id=_client_session_id(),
                )
        except APIClientError as exc:
            st.error(str(exc))
            return
        st.session_state.last_practice_draft_result = result
        st.session_state.practice_draft_result_scope = _practice_draft_scope_tag(course_id, class_id)

    result = _practice_draft_result_for_scope(course_id=course_id, class_id=class_id)
    if not result:
        return
    _render_practice_draft_result(result)


def _practice_draft_external_queries(
    *,
    allowed: bool,
    resource_type: str,
    query: str,
) -> list[dict[str, str]]:
    normalized = str(query or "").strip()
    if not allowed or not normalized:
        return []
    return [{"resource_type": str(resource_type), "query": normalized}]


def _practice_draft_scope_tag(course_id: int, class_id: int | None) -> dict[str, Any]:
    user = st.session_state.get("current_user")
    user_id = user.get("id") if isinstance(user, Mapping) else None
    return {
        "user_id": user_id,
        "course_id": int(course_id),
        "class_id": int(class_id) if class_id is not None else None,
        "mode": "practice_draft",
        "client_session_id": str(st.session_state.get("client_session_id") or ""),
    }


def _practice_draft_result_for_scope(*, course_id: int, class_id: int | None) -> dict[str, Any] | None:
    result = st.session_state.get("last_practice_draft_result")
    tag = st.session_state.get("practice_draft_result_scope")
    expected = _practice_draft_scope_tag(course_id, class_id)
    if not isinstance(result, dict) or not isinstance(tag, dict) or tag != expected:
        st.session_state.pop("last_practice_draft_result", None)
        st.session_state.pop("practice_draft_result_scope", None)
        return None
    return result


def _render_practice_draft_result(result: Mapping[str, Any]) -> None:
    st.text(str(result.get("proposal_label") or "模型提出的待验证教学草案"))
    left, right = st.columns(2)
    with left:
        st.markdown("#### 来源原文摘录")
        quotes = result.get("background_quotes") or []
        if not quotes:
            st.text("本次没有选用经服务器核对的背景摘录。")
        for item in quotes:
            if not isinstance(item, Mapping):
                continue
            st.text(f"来源：{item.get('source_title') or '背景资料'}")
            st.text(str(item.get("quote_text") or ""))
            st.text(f"来源状态：{delivery_label(item.get('delivery_mode'))}")
            safe_url = _safe_external_url(item.get("source_url"))
            if safe_url:
                st.link_button("打开来源", safe_url)
    with right:
        st.markdown("#### 待验证教学草案")
        if result.get("status") == "refused":
            st.text(str(result.get("refusal_reason") or "模型基于安全边界拒绝了本次请求。"))
        elif result.get("status") in {"grounded_draft", "proposal_only"}:
            plan = result.get("proposed_plan") or {}
            if not isinstance(plan, Mapping):
                plan = {}
            st.text(f"目标：{plan.get('objective') or '-'}")
            st.text(f"实验环境：{plan.get('lab_environment') or '-'}")
            st.text(f"基线与对照：{plan.get('baseline_and_control') or '-'}")
            for variable in plan.get("variables") or []:
                st.text(f"变量：{variable}")
            for index, step in enumerate(plan.get("steps") or [], start=1):
                if not isinstance(step, Mapping):
                    continue
                st.text(f"步骤 {index}：{step.get('action') or '-'}")
                st.text(f"预期观察：{step.get('expected_observation') or '-'}")
                st.text(f"检查方式：{step.get('check') or '-'}")
            for metric in plan.get("evaluation_metrics") or []:
                st.text(f"评估指标：{metric}")
            for observation in plan.get("expected_observations") or []:
                st.text(f"总体预期观察：{observation}")
            for risk in plan.get("risks") or []:
                st.text(f"风险：{risk}")
            for unknown in plan.get("unknowns") or []:
                st.text(f"待确认项：{unknown}")
        else:
            st.text("草案状态待核实，不能视为已生成。")
    for item in result.get("reference_statuses") or []:
        if not isinstance(item, Mapping):
            continue
        st.text(
            f"参考来源：{item.get('source') or '来源待核实'}；"
            f"查证：{lookup_label(item.get('lookup_state'))}；"
            f"资料：{delivery_label(item.get('delivery_mode'))}"
        )


def _render_assignments(course_id: int, class_id: int | None) -> None:
    st.subheader("实验任务")
    token = _auth_token()
    if not token:
        st.info("请先登录后查看实验任务和进度。")
        return
    try:
        payload = api_client.assignments(token, course_id=course_id, class_id=class_id)
    except APIClientError as exc:
        st.error(str(exc))
        return
    assignments = payload.get("assignments", [])
    if not assignments:
        st.info("暂无实验任务。")
        return

    assignment_options = {int(item["id"]): item for item in assignments}
    selected_id = st.selectbox("选择实验", list(assignment_options),
                               format_func=lambda item_id: str(assignment_options[item_id].get("title") or "未命名实验"))
    assignment = assignment_options[selected_id]
    steps = assignment.get("steps", [])
    if not steps:
        st.warning("该实验还没有步骤。")
        return

    progress = assignment.get("progress") or {}
    current_step_id = int(progress.get("step_id") or steps[0]["id"])
    step_options = {int(item["id"]): item for item in steps}
    default_step_index = next((index for index, item in enumerate(steps) if int(item["id"]) == current_step_id), 0)
    selected_step_id = st.selectbox("当前步骤", list(step_options), index=default_step_index,
                                   format_func=lambda step_id: str(step_options[step_id].get("title") or "未命名步骤"))
    selected_step = step_options[selected_step_id]
    st.session_state.assignment_context = {
        "assignment_id": int(assignment["id"]),
        "step_id": int(selected_step["id"]),
        "module": selected_step.get("module") or assignment.get("module") or "",
    }

    st.text(str(assignment.get("description") or ""))
    for step in steps:
        marker = "当前" if int(step["id"]) == int(selected_step["id"]) else f"步骤 {step.get('step_index')}"
        st.text(f"{marker}：{step.get('title') or '未命名步骤'}")
        st.text(str(step.get("description") or ""))

    with st.form("assignment_progress_form", clear_on_submit=False):
        status_options = {"进行中": "in_progress", "已完成": "completed", "暂停": "paused", "需要帮助": "stuck"}
        current_status = str(progress.get("status") or "in_progress")
        status_labels = list(status_options)
        default_status_index = next((index for index, label in enumerate(status_labels) if status_options[label] == current_status), 0)
        status_label = st.selectbox("进度状态", status_labels, index=default_status_index)
        percent = st.slider("完成度", 0, 100, int(progress.get("progress_percent") or 0))
        note = st.text_area("进度备注", value=str(progress.get("note") or ""), height=80, max_chars=500)
        submitted = st.form_submit_button("更新我的进度", use_container_width=True)
    if submitted:
        try:
            api_client.update_assignment_progress(
                token,
                int(assignment["id"]),
                step_id=int(selected_step["id"]),
                status=status_options[status_label],
                progress_percent=int(percent),
                note=note,
                client_session_id=_client_session_id(),
            )
        except APIClientError as exc:
            st.error(str(exc))
        else:
            st.success("进度已更新，并写入学习轨迹。")
            st.rerun()

    with st.form("stuck_report_form", clear_on_submit=True):
        reason = st.text_area("主动上报卡点", placeholder="说明你卡在哪一步、尝试过什么、希望获得什么帮助。", height=90, max_chars=800)
        reported = st.form_submit_button("上报卡点", use_container_width=True)
    if reported:
        if not reason.strip():
            st.warning("请填写卡点原因。")
            return
        try:
            api_client.report_stuck(
                token,
                assignment_id=int(assignment["id"]),
                step_id=int(selected_step["id"]),
                reason=reason.strip(),
                module=str(selected_step.get("module") or assignment.get("module") or ""),
                client_session_id=_client_session_id(),
            )
        except APIClientError as exc:
            st.error(str(exc))
        else:
            st.success("卡点已上报，教师端会立即看到。")


def _render_resource_portal(course_id: int, class_id: int | None) -> None:
    st.subheader("课程资料库")
    token = _auth_token()
    if not token:
        st.info("请先登录后浏览课程资料。")
        return
    type_options = {"全部": "", "课程文档": "course_document", "论文": "paper", "标准": "standard", "漏洞": "vulnerability", "开源项目": "project", "竞赛题目": "competition"}
    cols = st.columns([0.26, 0.18, 0.16, 0.14, 0.14, 0.12])
    keyword = cols[0].text_input("关键词", key="resource_keyword")
    topic = cols[1].text_input("专题", key="resource_topic")
    type_label = cols[2].selectbox("资料类型", list(type_options), key="resource_type")
    source = cols[3].text_input("来源", key="resource_source")
    use_year_from = cols[4].checkbox("起始年份", value=False, key="resource_use_year_from")
    use_year_to = cols[5].checkbox("结束年份", value=False, key="resource_use_year_to")
    year_from = st.number_input("起始年份数值", min_value=1990, max_value=2100, value=2020, step=1, disabled=not use_year_from)
    year_to = st.number_input("结束年份数值", min_value=1990, max_value=2100, value=2026, step=1, disabled=not use_year_to)
    page_size = st.select_slider("每页数量", options=[5, 10, 20, 50], value=10)
    page = int(st.session_state.get("resource_page", 1))
    current_fingerprint = _resource_query_fingerprint(
        keyword=keyword,
        topic=topic,
        resource_type=type_options[type_label],
        source=source,
        year_from=int(year_from) if use_year_from else None,
        year_to=int(year_to) if use_year_to else None,
        page=page,
        page_size=int(page_size),
        course_id=course_id,
        class_id=class_id,
    )
    stored_fingerprint = st.session_state.get("resource_search_fingerprint")
    if st.session_state.get("resource_search_payload") is not None and stored_fingerprint != current_fingerprint:
        st.session_state.pop("resource_search_payload", None)
        st.session_state.pop("resource_search_fingerprint", None)

    if st.button("按需联网搜索", use_container_width=True):
        st.session_state.resource_page = 1
        page = 1
        current_fingerprint = _resource_query_fingerprint(
            keyword=keyword,
            topic=topic,
            resource_type=type_options[type_label],
            source=source,
            year_from=int(year_from) if use_year_from else None,
            year_to=int(year_to) if use_year_to else None,
            page=page,
            page_size=int(page_size),
            course_id=course_id,
            class_id=class_id,
        )
        try:
            st.session_state.resource_search_payload = _normalize_remote_resource_payload(
                api_client.search_resources(
                    token,
                    query=keyword.strip(),
                    resource_type=type_options[type_label],
                    module=topic.strip(),
                    page=1,
                    page_size=int(page_size),
                    connector_name=_resource_connector_name(
                        type_options[type_label],
                        keyword,
                        rfc_enabled=_rfc_editor_enabled(),
                    ),
                )
            )
            st.session_state.resource_search_fingerprint = current_fingerprint
        except APIClientError as exc:
            st.session_state.pop("resource_search_payload", None)
            st.session_state.pop("resource_search_fingerprint", None)
            st.error(str(exc))
    try:
        payload = api_client.resource_portal(
            token,
            keyword=keyword.strip(),
            topic=topic.strip(),
            resource_type=type_options[type_label],
            year_from=int(year_from) if use_year_from else None,
            year_to=int(year_to) if use_year_to else None,
            source=source.strip(),
            page=page,
            page_size=int(page_size),
            client_session_id=_client_session_id(),
            course_id=course_id,
        )
    except APIClientError as exc:
        st.error(str(exc))
        return
    payload = _select_resource_payload(
        payload,
        st.session_state.get("resource_search_payload"),
        st.session_state.get("resource_search_fingerprint"),
        current_fingerprint,
    )

    total = payload.get("total")
    total_label = f"共 {total} 条资料" if isinstance(total, int) and total >= 0 else "资料数量待核实"
    st.caption(
        f"{total_label} | "
        f"查证状态：{_resource_status_label(payload)} | "
        f"资料状态：{_resource_delivery_label(payload.get('delivery_mode') or payload.get('cache_mode'))}"
    )
    if payload.get("can_manage"):
        st.info("如需更新公开资料，请使用教师端资料管理功能；学生页面只展示当前可用资料。")
    items = payload.get("items", [])
    if not items:
        st.info("暂无匹配资料。断网或外部服务不可用时，可以继续使用本地课程文档和已有缓存。")
        return
    for index, item in enumerate(items, start=1):
        _render_resource_item(item, index=index, token=token, course_id=course_id, class_id=class_id)
    nav_cols = st.columns(3)
    if nav_cols[0].button("上一页", disabled=page <= 1):
        st.session_state.resource_page = max(1, page - 1)
        st.rerun()
    nav_cols[1].caption(f"第 {page} 页")
    if nav_cols[2].button("下一页", disabled=not isinstance(total, int) or page * int(page_size) >= total):
        st.session_state.resource_page = page + 1
        st.rerun()


def _render_resource_item(item: Mapping[str, Any], *, index: int, token: str,
                          course_id: int, class_id: int | None) -> None:
    title = item.get("title") or "标题待核实"
    with st.expander(f"资料 {index}", expanded=False):
        st.text(f"标题：{title}")
        st.text(str(item.get("summary") or "暂无摘要。"))
        st.text(f"类型：{source_type_label(item.get('type'))}")
        st.text(f"专题：{item.get('topic_category') or '未标注'}")
        st.text(f"日期：{item.get('date_label') or item.get('year') or '未标注'}")
        if item.get("raw_summary") and item.get("raw_summary") != item.get("summary"):
            with st.expander("原始来源摘要", expanded=False):
                st.text(str(item.get("raw_summary")))
        st.text(f"来源：{source_name_label(item.get('source_name') or item.get('source'))}")
        st.text(f"来源类型：{_resource_origin_label(item.get('origin_kind'))}")
        st.text(f"资料状态：{_resource_delivery_label(item.get('delivery_mode') or item.get('cache_status'))}")
        st.text(f"获取时间：{item.get('fetched_at') or '未记录'}")
        st.text(f"标签：{', '.join(str(tag) for tag in item.get('tags') or []) or '无'}")
        safe_url = _safe_external_url(item.get("url"))
        if safe_url:
            st.link_button("打开外部链接", safe_url, use_container_width=False)
        if st.button("记录打开", key=f"open_{item.get('resource_id')}", use_container_width=False):
            try:
                api_client.open_resource(
                    token,
                    item["resource_id"],
                    client_session_id=_client_session_id(),
                    metadata={"title": title, "type": item.get("type"), "source": item.get("source")},
                    course_id=course_id,
                    class_id=class_id,
                )
            except APIClientError as exc:
                st.error(str(exc))
            else:
                st.success("已记录资料查看。")


def _resource_connector_name(
    resource_type: str,
    query: str = "",
    *,
    rfc_enabled: bool = False,
) -> str | None:
    if resource_type == "standard":
        is_rfc = bool(re.search(r"\bRFC[\s-]*\d+\b", query, flags=re.IGNORECASE))
        return "rfc_editor" if is_rfc and rfc_enabled else "standard_source"
    return {"vulnerability": "nvd", "project": "github"}.get(resource_type)


def _rfc_editor_enabled() -> bool:
    try:
        payload = api_client.connectors_status()
    except APIClientError:
        return False
    return any(
        str(item.get("name") or "") == "rfc_editor" and bool(item.get("enabled"))
        for item in payload.get("connectors") or []
        if isinstance(item, dict)
    )


def _safe_external_url(value: object) -> str:
    text = str(value or "").strip()
    if not text or re.search(r"[\s\\\x00-\x1f]", text):
        return ""
    try:
        parsed = urlparse(text)
    except ValueError:
        return ""
    return (
        text
        if parsed.scheme.lower() in {"http", "https"}
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        else ""
    )


def _resource_status_label(payload: Mapping[str, Any]) -> str:
    state = str(payload.get("lookup_state") or payload.get("status") or "").upper()
    delivery = str(payload.get("delivery_mode") or payload.get("cache_mode") or "").lower()
    if state == "PARTIAL" or payload.get("incomplete_results") is True:
        return "部分结果"
    if state == "NOT_FOUND":
        return "未找到"
    if state in {"UNAVAILABLE", "FAILED"}:
        return "暂不可用"
    if delivery == "live":
        return "实时结果"
    if delivery == "fresh_cache":
        return "新鲜缓存"
    if delivery == "stale_cache":
        return "过期缓存（上游不可用）"
    if state in {"FOUND", "SUCCESS"}:
        return "已找到"
    return "状态待核实"


def _resource_origin_label(value: object) -> str:
    return {
        "course_document": "当前课程资料",
        "remote": "公开远程来源",
        "local_catalog": "本地目录",
    }.get(str(value or "").lower(), "来源待核实")


def _resource_delivery_label(value: object) -> str:
    return delivery_label(value)


def _resource_query_fingerprint(
    *,
    keyword: str,
    topic: str,
    resource_type: str,
    source: str,
    year_from: int | None,
    year_to: int | None,
    page: int,
    page_size: int,
    course_id: int | None,
    class_id: int | None,
) -> tuple[str, str, str, str, int | None, int | None, int, int, int | None, int | None]:
    normalize = lambda value: " ".join(str(value or "").strip().lower().split())
    return (
        normalize(keyword),
        normalize(topic),
        normalize(resource_type),
        normalize(source),
        year_from,
        year_to,
        int(page),
        int(page_size),
        int(course_id) if course_id is not None else None,
        int(class_id) if class_id is not None else None,
    )


def _select_resource_payload(
    portal_payload: dict,
    remote_payload: dict | None,
    stored_fingerprint: object,
    current_fingerprint: tuple[str, str, str, str, int | None, int | None, int, int, int | None, int | None],
) -> dict:
    if remote_payload is None or stored_fingerprint != current_fingerprint:
        return portal_payload

    _, _, resource_type, source, year_from, year_to, page, page_size, _, _ = current_fingerprint
    items = []
    for item in remote_payload.get("items") or []:
        item_type = str(item.get("type") or "").strip().lower()
        item_source = str(item.get("source_name") or item.get("source") or "").strip().lower()
        item_year = _resource_item_year(item)
        if resource_type and item_type != resource_type:
            continue
        if source and item_source != source:
            continue
        if year_from is not None and (item_year is None or item_year < year_from):
            continue
        if year_to is not None and (item_year is None or item_year > year_to):
            continue
        items.append(item)
    return {
        **remote_payload,
        "items": items,
        "total": len(items),
        "page": page,
        "page_size": page_size,
    }


def _resource_item_year(item: dict) -> int | None:
    value = item.get("year") or item.get("published_at") or item.get("date_label")
    if value is None:
        return None
    text = str(value).strip()
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else None


def _normalize_remote_resource_payload(payload: dict) -> dict:
    items = []
    delivery_mode = payload.get("delivery_mode") or ("fresh_cache" if payload.get("cache_hit") else "unknown")
    origin_kind = payload.get("origin_kind") or "unknown"
    for resource in payload.get("resources") or []:
        if not resource.get("provenance_verified"):
            continue
        item_delivery_mode = resource.get("delivery_mode") or delivery_mode
        item_origin_kind = resource.get("origin_kind") or origin_kind
        resource_id = f"external:{resource.get('id')}" if resource.get("id") else (
            f"external:{resource.get('source_name')}:{resource.get('external_id')}"
        )
        resource_type = resource.get("resource_type") or "unknown"
        title = resource.get("title") or resource.get("external_id") or "标题待核实"
        abstract = resource.get("abstract") or ""
        source = resource.get("source_name") or payload.get("connector") or "external"
        items.append(
            {
                **resource,
                "resource_id": resource_id,
                "type": resource_type,
                "summary": abstract or "暂无摘要。",
                "raw_summary": abstract,
                "type_label": source_type_label(resource_type),
                "topic_category": resource.get("module") or "-",
                "date_label": resource.get("published_at") or "日期未知",
                "source": source,
                "cache_status": item_delivery_mode,
                "origin_kind": item_origin_kind,
                "delivery_mode": item_delivery_mode,
                "updated_at": resource.get("fetched_at") or "",
                "title": title,
                "tags": resource.get("tags") or [],
                "url": _safe_external_url(resource.get("url")),
            }
        )
    return {
        "items": items,
        "total": len(items),
        "page": 1,
        "page_size": len(items) or 1,
        "cache_mode": delivery_mode,
        "status": payload.get("status"),
        "lookup_state": payload.get("lookup_state"),
        "source_attempts": payload.get("source_attempts") or [],
        "incomplete_results": payload.get("incomplete_results"),
    }


def _remember_answer_result(result: dict[str, Any], course_id: int | None) -> None:
    run_id = result.get("run_id")
    if not run_id:
        return
    entry = dict(result)
    entry["_course_id"] = int(course_id) if course_id is not None else None
    history = [
        item
        for item in list(st.session_state.get("chat_result_history") or [])
        if item.get("run_id") != run_id
    ]
    history.append(entry)
    st.session_state.chat_result_history = history[-MAX_CHAT_HISTORY:]


def _answer_history_for_course(course_id: int | None) -> list[dict[str, Any]]:
    expected_course_id = int(course_id) if course_id is not None else None
    return [
        item
        for item in list(st.session_state.get("chat_result_history") or [])
        if item.get("_course_id") == expected_course_id
    ]


def _render_answer_history(course_id: int | None, active_run_id: str | None = None) -> None:
    history = [
        item
        for item in _answer_history_for_course(course_id)
        if item.get("run_id") != active_run_id
    ]
    if not history:
        return
    with st.expander("最近回答历史", expanded=False):
        for index, item in enumerate(reversed(history[-MAX_CHAT_HISTORY:]), start=1):
            st.text(f"历史回答 {index}")
            st.caption(_execution_status_text(item))
            answer = str(item.get("answer") or item.get("pending_question") or "")
            st.text(answer)


def _execution_status_text(result: dict[str, Any]) -> str:
    provider = str(result.get("provider") or "").lower()
    generation = {"deepseek": "外部模型", "template": "本地模板"}.get(provider, "生成方式待核实")
    fallback = result.get("fallback_used")
    fallback_label = "已降级" if fallback is True else "未降级" if fallback is False else "降级状态待核实"
    return "｜".join((
        f"回答状态：{STATUS_LABELS.get(result.get('run_status'), '状态待核实')}",
        f"生成方式：{generation}",
        f"资料：{delivery_label(result.get('delivery_mode'))}",
        fallback_label,
    ))


def _render_answer(result: dict[str, Any]) -> None:
    run_status = result.get("run_status")
    status_col, confidence_col, module_col = st.columns(3)
    status_col.metric("回答状态", STATUS_LABELS.get(run_status, "状态待核实"))
    confidence = result.get("confidence")
    confidence_col.metric("参考置信度", f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else "暂无数据")
    module_col.metric("问题类别", module_label(result.get("module")))
    st.caption(_execution_status_text(result))
    if result.get("model"):
        st.text(f"模型：{_safe_model_name(result.get('model'))}")
    st.caption("资料查找成功不等于回答成功；以本次回答状态和引用为准。")
    answer = str(result.get("answer") or result.get("pending_question") or "")
    if run_status == "REQUIRES_INPUT":
        st.info("需要补充信息。")
        st.text(answer)
    elif result.get("abstained") or run_status == "ABSTAINED":
        st.warning("当前证据不足，系统已拒答。")
        st.text(answer)
    elif run_status == "LIMIT_REACHED":
        st.warning("已达到本次处理上限。")
        st.text(answer)
    elif run_status == "FAILED":
        st.error("本次处理未完成。")
        st.text(answer)
    elif run_status == "COMPLETED":
        st.text(answer)
    else:
        st.info("回答状态待核实。")
        st.text(answer)
    _render_selected_tool(result)
    _render_tool_calls(result.get("tool_calls", []))
    _render_citations(result.get("citations", []))
    _render_feedback(result)
    _render_agent_trace(result.get("run_id"))


def _render_selected_tool(result: dict[str, Any]) -> None:
    tool = result.get("tool")
    if isinstance(tool, Mapping):
        st.text(f"建议使用：{teacher_tool_label(tool.get('name'))}")
        if tool.get("description"):
            st.text(str(tool["description"]))


def _render_tool_calls(tool_calls: list[dict[str, Any]]) -> None:
    if not tool_calls:
        return
    with st.expander("工具调用摘要", expanded=False):
        for item in tool_calls:
            result = _normalize_tool_result(item)
            data = result.get("data") if isinstance(result.get("data"), Mapping) else {}
            success = _coerce_bool(_first_present(item.get("success"), result.get("success")))
            fallback_used = _coerce_bool(
                _first_present(item.get("fallback_used"), result.get("fallback_used"), data.get("fallback_used"))
            )
            status = "成功" if success is True else "未完成" if success is False else "状态待核实"
            source = "本地降级" if fallback_used is True else "已连接资料工具" if fallback_used is False else "调用方式待核实"
            mode = _first_present(item.get("delivery_mode"), result.get("delivery_mode"), data.get("delivery_mode"))
            st.text(f"{teacher_tool_label(item.get('tool_name'))}：{status}；{source}；{delivery_label(mode)}")


def _normalize_tool_result(item: Mapping[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {}
    result = item.get("result")
    if isinstance(result, Mapping):
        return dict(result)
    result_json = item.get("result_json")
    if isinstance(result_json, Mapping):
        return dict(result_json)
    if isinstance(result_json, str):
        raw = result_json.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _render_citations(citations: list[dict[str, Any]]) -> None:
    st.subheader("引用来源")
    if not citations:
        st.caption("本次回答没有可展示的引用来源。")
        return
    for index, citation in enumerate(citations, 1):
        source_id = str(citation.get("source_id") or "").strip()
        marker = f"[{source_id}]" if re.fullmatch(r"S[1-9][0-9]*", source_id) else str(index)
        with st.expander(f"引用 {marker}", expanded=index == 1):
            st.text(f"标题：{citation.get('title') or '标题待核实'}")
            st.text(_format_location(citation))
            st.text(str(citation.get("snippet") or ""))
            safe_url = _safe_external_url(citation.get("source_url"))
            if safe_url:
                st.link_button("打开公开来源", safe_url)


def _render_agent_trace(run_id: str | None) -> None:
    if not run_id:
        return
    with st.expander("本次处理过程", expanded=False):
        try:
            run = api_client.get_agent_run(run_id, token=_auth_token())
        except APIClientError as exc:
            st.error(str(exc))
            return
        st.text(f"结束状态：{agent_stop_label(run.get('stop_reason'))}")
        st.text(f"处理步骤：{len(run.get('steps') or [])}")
        for index, step in enumerate(run.get("steps") or [], start=1):
            with st.expander(f"处理步骤 {index}", expanded=False):
                st.text(f"环节：{agent_action_label(step.get('action'))}")
                st.text(f"状态：{STATUS_LABELS.get(step.get('status'), '状态待核实')}")


def _render_feedback(result: dict[str, Any]) -> None:
    question_id = result.get("question_id")
    if not question_id:
        return
    with st.expander("评价本次回答", expanded=False):
        with st.form(f"feedback_{question_id}", clear_on_submit=True):
            rating = st.slider("评分", min_value=1, max_value=5, value=4)
            helpful = st.radio("是否有帮助", ["有帮助", "没有帮助", "不确定"], horizontal=True)
            comment = st.text_input("补充说明", max_chars=300)
            submitted = st.form_submit_button("提交评价", use_container_width=True)
        if submitted:
            helpful_value = True if helpful == "有帮助" else False if helpful == "没有帮助" else None
            try:
                api_client.submit_feedback(int(question_id), int(rating), helpful=helpful_value, comment=comment, token=_auth_token(), client_session_id=_client_session_id())
            except APIClientError as exc:
                st.error(str(exc))
                return
            st.success("评价已提交。")


def _format_location(citation: dict[str, Any]) -> str:
    parts = [f"来源类型：{source_type_label(citation.get('source_type'))}"]
    if citation.get("page") is not None:
        parts.append(f"页码：{citation.get('page')}")
    if citation.get("section"):
        parts.append(f"章节：{citation.get('section')}")
    if citation.get("external_id"):
        parts.append(f"公开编号：{citation.get('external_id')}")
    if citation.get("origin_kind"):
        parts.append(f"来源：{_resource_origin_label(citation.get('origin_kind'))}")
    if citation.get("delivery_mode"):
        parts.append(f"资料状态：{delivery_label(citation.get('delivery_mode'))}")
    return " | ".join(parts)


def _safe_model_name(value: object) -> str:
    text = str(value or "")
    return text if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", text) else "身份待核实"


def _auth_token() -> str | None:
    return st.session_state.get("auth_token")


def _must_change_password() -> bool:
    return bool(st.session_state.get("current_user", {}).get("must_change_password"))


def _client_session_id() -> str | None:
    return st.session_state.get("client_session_id")


def _assignment_context() -> dict[str, int | str]:
    return st.session_state.get("assignment_context") or {}


if __name__ == "__main__":
    render()
