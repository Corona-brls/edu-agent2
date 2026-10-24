"""Innovation coaching Streamlit page."""

from __future__ import annotations

import streamlit as st
import re
from urllib.parse import urlsplit

from frontend import api_client
from frontend.api_client import APIClientError
from frontend.presentation import (
    delivery_label, lookup_label, reference_status_label, source_name_label, source_type_label,
)
from frontend.session_scope import ensure_student_scope


def render() -> None:
    st.title("创新启发")
    st.caption("围绕专题、当前方案、课程证据和本地工具，形成可验证的创新方向。")
    token = st.session_state.get("auth_token")
    if not token:
        _clear_open_session()
        st.info("请先登录后使用创新启发会话。")
        return
    scope = ensure_student_scope()
    if scope is None:
        return
    course_id, class_id = scope
    page_scope = (st.session_state.get("current_user", {}).get("id"), course_id, class_id)
    previous_page_scope = st.session_state.get("innovation_session_scope")
    if st.session_state.get("innovation_session") and previous_page_scope != page_scope:
        _clear_open_session()

    st.subheader("打开已有创新会话")
    try:
        summaries = api_client.list_innovation_sessions(
            token, course_id=course_id, class_id=class_id,
        ).get("items", [])
    except APIClientError as exc:
        _clear_open_session()
        st.error(str(exc))
        return
    summary_by_id = {
        int(item["id"]): item for item in summaries
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    if st.session_state.get("innovation_resume_selection") not in summary_by_id:
        st.session_state.pop("innovation_resume_selection", None)
    selected_id = st.selectbox(
        "当前课班中的历史会话",
        [None, *summary_by_id],
        format_func=lambda value: "请选择会话" if value is None else _resume_label(summary_by_id[value]),
        key="innovation_resume_selection",
    )
    if st.button("打开所选会话", disabled=selected_id is None):
        if selected_id is None:
            st.info("请先选择要打开的会话。")
        else:
            try:
                opened = api_client.get_innovation_session(token, int(selected_id))
                if (
                    int(opened.get("id")) != int(selected_id)
                    or int(opened.get("user_id")) != int(page_scope[0])
                    or int(opened.get("course_id")) != course_id
                    or opened.get("class_id") != class_id
                ):
                    raise ValueError("会话范围不匹配")
            except (APIClientError, ValueError, TypeError):
                _clear_open_session()
                st.error("无法打开当前会话，请刷新后重试。")
                return
            _clear_suggestion_state()
            st.session_state.innovation_session = opened
            st.session_state.innovation_session_scope = page_scope

    with st.form("innovation_start", clear_on_submit=False):
        topic = st.text_input("研究专题", value="基于区块链的隐蔽通信")
        current_plan = st.text_area(
            "当前方案",
            value="计划在交易备注字段中编码少量信号，并比较吞吐量和可检测性。",
            height=120,
        )
        submitted = st.form_submit_button("开始创新会话", use_container_width=True)

    if submitted:
        try:
            result = api_client.start_innovation_session(
                token,
                topic=topic,
                current_plan=current_plan,
                client_session_id=st.session_state.get("client_session_id"),
                course_id=course_id,
                class_id=class_id,
            )
        except APIClientError as exc:
            st.error(str(exc))
            return
        _clear_suggestion_state()
        st.session_state.innovation_session = result
        st.session_state.innovation_session_scope = page_scope

    session = st.session_state.get("innovation_session")
    if not session:
        return

    current_scope = _suggestion_scope(
        user_id=st.session_state.get("current_user", {}).get("id"),
        course_id=course_id,
        class_id=class_id,
        client_session_id=st.session_state.get("client_session_id"),
        innovation_session_id=session.get("id"),
    )
    _clear_stale_suggestion_state(current_scope)

    st.subheader("来源案例")
    for case in session.get("cases", []):
        _render_case_card(case)

    st.subheader("当前方案限制")
    for item in session.get("limitations", []):
        st.text(str(item))

    st.subheader("启发式问题")
    open_questions = [item for item in session.get("questions", []) if not item.get("answer")]
    if open_questions:
        question = open_questions[0]
        st.text(str(question["question"]))
        with st.form(f"innovation_answer_{question['id']}", clear_on_submit=True):
            answer = st.text_area("你的回答", height=110)
            answered = st.form_submit_button("提交回答并继续", use_container_width=True)
        if answered:
            try:
                updated = api_client.answer_innovation_question(
                    token,
                    int(session["id"]),
                    question_id=int(question["id"]),
                    answer=answer,
                    client_session_id=st.session_state.get("client_session_id"),
                )
            except APIClientError as exc:
                st.error(str(exc))
                return
            st.session_state.innovation_session = updated
            st.rerun()
    else:
        st.success("当前启发问题已回答，可以生成候选创新方向。")

    if st.button("生成候选创新方向", use_container_width=True):
        try:
            finalized = api_client.finalize_innovation_session(
                token,
                int(session["id"]),
                client_session_id=st.session_state.get("client_session_id"),
            )
        except APIClientError as exc:
            st.error(str(exc))
            return
        st.session_state.innovation_session = finalized
        st.rerun()

    st.subheader("模型建议")
    st.text("专题、当前方案和必要回答会发送给外部模型；建议均为待验证内容。")
    use_course_references = st.checkbox(
        "本次明确允许附加符合课程 allowlist 的当前课程案例",
        value=bool(st.session_state.get("innovation_suggestion_use_course_references", False)),
        key="innovation_suggestion_use_course_references",
    )
    if st.button("生成模型建议", key="innovation_suggestion_button", use_container_width=True):
        try:
            suggestion = api_client.innovation_suggestions(
                token,
                int(session["id"]),
                client_session_id=st.session_state.get("client_session_id"),
                use_course_references=use_course_references,
            )
        except APIClientError as exc:
            st.error(str(exc))
        else:
            st.session_state.innovation_suggestion_result = suggestion
            st.session_state.innovation_suggestion_result_scope = current_scope

    suggestion_result = st.session_state.get("innovation_suggestion_result")
    if suggestion_result and st.session_state.get("innovation_suggestion_result_scope") == current_scope:
        _render_suggestion_result(suggestion_result, cases=session.get("cases") or [])

    ideas = session.get("final_ideas", [])
    if ideas:
        st.subheader("候选创新方向")
        for idea in ideas:
            _render_template_idea(idea)


def _render_suggestion_result(result: dict, *, cases: list[dict] | None = None) -> None:
    case_titles = {
        str(case.get("source_id")): str(case.get("title") or "标题待核实")
        for case in cases or [] if isinstance(case, dict) and case.get("source_id")
    }
    st.text(str(result.get("proposal_label") or "模型提出的待验证创新建议"))
    st.text(f"生成方式：{_generation_label(result.get('generation_mode'))}")
    st.text(f"建议状态：{_suggestion_status_label(result.get('status'))}")
    st.text(f"课程参考状态：{reference_status_label(result.get('course_reference_status'))}")
    collection = {"not_recorded": "历史收集过程未记录", "recorded": "本次收集过程已记录"}
    st.text(f"案例收集状态：{collection.get(result.get('collection_status'), '状态待核实')}")
    for warning in result.get("warnings") or []:
        st.text(f"提示：{warning}")
    if result.get("status") == "refused":
        st.text(str(result.get("refusal_reason") or "模型拒绝了本次请求。"))
    for index, item in enumerate(result.get("suggestions") or [], start=1):
        st.text(f"建议 {index}")
        st.text(f"待验证假设：{item.get('hypothesis') or '-'}")
        st.text(f"对照方案：{item.get('comparison') or '-'}")
        for metric in item.get("metrics") or []:
            st.text(f"评价指标：{metric}")
        st.text(f"下一问题：{item.get('next_question') or '-'}")
        for risk in item.get("risks") or []:
            st.text(f"风险：{risk}")
        for source_index, source_id in enumerate(item.get("background_source_ids") or [], start=1):
            st.text(f"背景案例 {source_index}：{case_titles.get(str(source_id), '来源标题待核实')}")
    for source_index, status in enumerate(result.get("source_statuses") or [], start=1):
        title = case_titles.get(str(status.get("source_id")), source_type_label(status.get("source_type")))
        st.text(
            f"背景案例 {source_index}（{title}）：{lookup_label(status.get('status'))}；"
            f"{delivery_label(status.get('delivery_mode'))}"
        )


def _render_case_card(case: dict) -> None:
    source_name = source_name_label(case.get("source_name"))
    with st.expander("案例详情", expanded=False):
        st.text(f"标题：{case.get('title') or '-'}")
        st.text(f"类型：{source_type_label(case.get('type'))} / 来源：{source_name}")
        st.text(str(case.get("summary") or "暂无摘要。"))
        safe_url = _safe_external_url(case.get("url"))
        if safe_url:
            st.link_button("来源链接", safe_url)
        st.text(" | ".join(_case_caption_parts(case)))


def _render_template_idea(idea: dict) -> None:
    with st.expander("候选方向详情", expanded=True):
        is_template = idea.get("generation_mode") == "template"
        identity = "模板启发" if is_template else "生成方式待核实"
        if not is_template:
            label = "待验证方向（生成方式待核实）"
        elif idea.get("label") == "案例背景／待验证方向":
            label = "案例背景／待验证方向"
        else:
            label = "模板启发／待验证方向"
        st.text(f"标题：{idea.get('title') or '-'}")
        st.text(f"生成方式：{identity}")
        st.text(f"性质：{label}")
        st.text(f"待验证假设：{idea.get('hypothesis')}")
        st.text(f"实验方法：{idea.get('experiment_method')}")
        st.text("风险：")
        for risk in idea.get("risks", []):
            st.text(f"- {risk}")
        st.text("案例背景：")
        evidence = idea.get("evidence", [])
        if evidence:
            for item in evidence:
                st.text(f"- {item.get('title')} ({source_type_label(item.get('type'))} / {source_name_label(item.get('source'))})")
        else:
            st.text("暂无可定位案例背景；方向仍待验证。")


def _clear_stale_suggestion_state(current_scope: tuple[object, ...]) -> None:
    previous_scope = st.session_state.get("innovation_suggestion_result_scope")
    if previous_scope is not None and previous_scope != current_scope:
        _clear_suggestion_state()


def _clear_suggestion_state() -> None:
    st.session_state.pop("innovation_suggestion_result", None)
    st.session_state.pop("innovation_suggestion_result_scope", None)
    st.session_state["innovation_suggestion_use_course_references"] = False


def _clear_open_session() -> None:
    st.session_state.pop("innovation_session", None)
    st.session_state.pop("innovation_session_scope", None)
    st.session_state.pop("innovation_resume_selection", None)
    _clear_suggestion_state()


def _resume_label(item: dict) -> str:
    status = {"ACTIVE": "进行中", "FINALIZED": "已形成候选"}.get(item.get("status"), "状态待核实")
    title = str(item.get("topic") or "未命名专题").strip()[:80]
    when = str(item.get("updated_at") or item.get("created_at") or "时间未记录")[:25]
    return f"{title} · {status} · {when}"


def _suggestion_scope(
    *,
    user_id: object,
    course_id: object,
    class_id: object,
    client_session_id: object,
    innovation_session_id: object,
) -> tuple[object, ...]:
    return (
        user_id,
        course_id,
        class_id,
        "innovation_suggestion",
        client_session_id,
        innovation_session_id,
    )


def _safe_external_url(value: object) -> str:
    text = str(value or "").strip()
    if not text or re.search(r"[\s\\\x00-\x1f]", text):
        return ""
    try:
        parsed = urlsplit(text)
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return ""
    if parsed.username is not None or parsed.password is not None:
        return ""
    return text


def _case_caption_parts(case: dict) -> list[str]:
    parts = [
        f"资料状态：{delivery_label(case.get('delivery_mode'))}",
        f"获取时间：{case.get('fetched_at') or '未记录'}",
    ]
    return parts


def _generation_label(value: object) -> str:
    return {"deepseek": "外部模型", "template": "本地模板"}.get(value, "生成方式待核实")


def _suggestion_status_label(value: object) -> str:
    return {"proposed": "待验证建议", "refused": "已拒绝生成"}.get(value, "状态待核实")
