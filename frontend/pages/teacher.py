"""Teacher-facing Streamlit analytics and knowledge management page."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import streamlit as st

from frontend import api_client
from frontend.api_client import APIClientError
from frontend.presentation import (
    agent_action_label, agent_stop_label, build_hotspot_view_model,
    event_label as _event_label, health_rows,
    model_status, module_label as _module_label, page_label,
    normalize_progress_status as _normalize_progress_status,
    progress_label as _progress_label, teacher_tool_label, tool_rows,
)


def render() -> None:
    st.header("教学管理")
    st.text("查看教学概览、整理热点问题，并管理授权课程资料。")

    if not _auth_token():
        st.info("请先登录教师或管理员账号。")
        return
    if not _has_management_access():
        st.warning("当前账号无权访问教学管理。")
        return
    if st.button("刷新当前页面", use_container_width=False):
        st.cache_data.clear()
        st.rerun()

    scope = _select_teacher_scope(_auth_token())
    if scope is None:
        return
    course_id, class_id = scope
    tabs = st.tabs(["教学概览", "热点问题库", "课程资料", "外部资料更新", "运行状态"])
    with tabs[0]:
        _render_learning_analytics(course_id=course_id, class_id=class_id)
    with tabs[1]:
        _render_hotspot_library(_auth_token(), course_id=course_id, class_id=class_id)
    with tabs[2]:
        _render_knowledge_management()
    with tabs[3]:
        _render_resource_sync()
    with tabs[4]:
        _render_system_status()


def _select_teacher_scope(token: str) -> tuple[int, int | None] | None:
    try:
        courses = api_client.courses(token)
    except APIClientError as exc:
        st.error(str(exc))
        _clear_hotspot_state()
        return None
    course_options = {int(item["id"]): item for item in courses if isinstance(item, dict) and item.get("id")}
    if not course_options:
        st.info("当前账号没有可管理的课程。")
        _clear_hotspot_state()
        return None
    if st.session_state.get("teacher_hotspot_course_selector") not in (None, *course_options):
        st.session_state.pop("teacher_hotspot_course_selector", None)
    course_id = st.selectbox(
        "当前课程", list(course_options), index=None, placeholder="请选择授权课程",
        format_func=lambda cid: str(course_options[cid].get("name") or course_options[cid].get("slug") or "未命名课程"),
        key="teacher_hotspot_course_selector",
    )
    if course_id is None:
        _clear_hotspot_state()
        return None
    if st.session_state.get("teacher_hotspot_course_bound") != int(course_id):
        st.session_state.pop("teacher_hotspot_class_selector", None)
        _clear_hotspot_state()
        st.session_state["teacher_hotspot_course_bound"] = int(course_id)
    try:
        classes = api_client.classes(token, course_id=int(course_id))
    except APIClientError as exc:
        st.error(str(exc))
        _clear_hotspot_state()
        return None
    class_options = {int(item["id"]): item for item in classes if isinstance(item, dict) and item.get("id")}
    if st.session_state.get("teacher_hotspot_class_selector") not in (None, *class_options):
        st.session_state.pop("teacher_hotspot_class_selector", None)
    class_id = st.selectbox(
        "当前班级", [None, *class_options], index=0,
        format_func=lambda cid: "全部授权班级" if cid is None else str(class_options[cid].get("name") or "未命名班级"),
        key="teacher_hotspot_class_selector",
    )
    scope = (st.session_state.get("current_user", {}).get("id"), int(course_id), class_id)
    if st.session_state.get("teacher_hotspot_scope") != scope:
        _clear_hotspot_state()
        st.session_state["teacher_hotspot_scope"] = scope
    return int(course_id), int(class_id) if class_id is not None else None


def _clear_hotspot_state() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith(("teacher_hotspot_selected_", "teacher_hotspot_filter_", "teacher_hotspot_prepared_",
                                "teacher_hotspot_confirm_", "teacher_hotspot_rename_", "teacher_hotspot_approve_",
                                "teacher_hotspot_merge_", "teacher_hotspot_split_")):
            st.session_state.pop(key, None)
    st.session_state.pop("teacher_hotspot_scope", None)


def _has_management_access() -> bool:
    return bool(_auth_token()) and _current_role() in {"teacher", "admin"}


def _render_learning_analytics(*, course_id: int, class_id: int | None) -> None:
    _render_phase3_dashboard(course_id=course_id, class_id=class_id)
    if _current_role() != "admin":
        return
    token = _auth_token()
    try:
        analytics = _load_analytics(token)
        faq = _load_faq(token)
    except APIClientError as exc:
        st.error(str(exc))
        return

    st.subheader("管理员全局统计（不受上方课程与班级选择影响）")
    _render_core_metrics(analytics)
    _render_phase2_metrics(analytics)
    left_col, right_col = st.columns([0.52, 0.48], gap="large")
    with left_col:
        _render_module_distribution(analytics)
        _render_agent_paths(analytics)
        _render_recent_questions(analytics)
    with right_col:
        _render_module_stuck_points(analytics)
        _render_assignment_stuck_analysis()
        _render_feedback_metrics(analytics)
        _render_faq(faq)


def _render_phase3_dashboard(*, course_id: int, class_id: int | None) -> None:
    st.subheader("教学概览")
    token = _auth_token()
    if not token:
        st.info("登录教师或管理员账号后可查看当前课程与班级的教学统计。")
        return

    with st.expander("筛选条件", expanded=True):
        cols = st.columns(2)
        start_at = cols[0].text_input("开始时间", value="", placeholder="2026-09-01T00:00:00")
        end_at = cols[1].text_input("结束时间", value="", placeholder="2026-09-30T23:59:59")

    try:
        summary = api_client.teacher_phase3_summary(
            token,
            course_id=course_id,
            class_id=class_id,
            start_at=start_at.strip() or None,
            end_at=end_at.strip() or None,
        )
    except APIClientError as exc:
        st.error(str(exc))
        return

    access = summary.get("access", {})
    st.caption("访问次数仅统计页面查看，不包含系统内部处理步骤。")
    cols = st.columns(5)
    cols[0].metric("独立活跃学生", int(access.get("unique_active_students", 0)))
    cols[1].metric("访问次数", int(access.get("visit_count", 0)))
    cols[2].metric("活跃天数", int(access.get("active_days", 0)))
    cols[3].metric("资料搜索", int(summary.get("resources", {}).get("search_count", 0)))
    cols[4].metric("资料打开", int(summary.get("resources", {}).get("open_count", 0)))
    st.text(f"最近活跃：{access.get('recent_active_at') or '暂无信息'}")

    event_cols = st.columns(2)
    with event_cols[0]:
        _table_if_any("页面分布", [
            {"页面": page_label(name), "次数": count}
            for name, count in access.get("page_distribution", {}).items() if isinstance(count, int)
        ])
        _table_if_any("学生事件分布", [
            {"事件": _event_label(name), "次数": count}
            for name, count in summary.get("event_distribution", {}).items() if isinstance(count, int)
        ])
    with event_cols[1]:
        qa = summary.get("qa", {})
        qa_cols = st.columns(2)
        qa_cols[0].metric("问答次数", qa.get("question_count", "暂无数据"))
        qa_cols[1].metric("实践指导次数", summary.get("practice", {}).get("guide_count", "暂无数据"))
        _table_if_any("问答模块分布", [
            {"模块": _module_label(name), "次数": count}
            for name, count in qa.get("module_distribution", {}).items() if isinstance(count, int)
        ])

    _render_student_timeline_panel(token)

    _table_if_any("实验进度状态", [
        {"状态": _progress_label(name), "条数": count}
        for name, count in summary.get("assignments", {}).get("by_status", {}).items() if isinstance(count, int)
    ])
    _table_if_any("最近实验进度", [
        {"任务": item.get("assignment_title") or "未命名任务", "步骤": item.get("step_title") or "暂无信息",
         "状态": _progress_label(item.get("status")), "更新时间": item.get("updated_at") or "暂无信息"}
        for item in summary.get("assignments", {}).get("recent", []) if isinstance(item, dict)
    ])
    stuck_items = summary.get("stuck", {}).get("items", [])
    _table_if_any("卡点分析", [
        {"所在步骤": item.get("step_title") or "暂无信息", "学生说明": item.get("reason") or "暂无信息",
         "处理状态": _progress_label(item.get("status"))}
        for item in stuck_items if isinstance(item, dict)
    ])
    _render_stuck_review_actions(token, stuck_items, key_prefix="phase3")


def _render_hotspot_library(token: str, *, course_id: int, class_id: int | None) -> None:
    st.subheader("热点问题库")
    st.text("系统按相似提问自动归类，供教师整理；以下为当前课程/班级累计问题，不受教学概览时间范围影响。")
    try:
        summary = api_client.teacher_phase3_summary(token, course_id=course_id, class_id=class_id)
    except APIClientError as exc:
        st.error(str(exc))
        _clear_hotspot_state()
        return
    view = build_hotspot_view_model(summary)
    if view.status != "ready":
        st.warning(view.message or "热点资料暂不可用，请稍后刷新")
        return

    cols = st.columns(4)
    for col, label, count in zip(
        cols,
        ("热点主题", "待教师审核", "已审核主题", "本次可导出常见问答"),
        (view.topic_count, view.pending_count, view.approved_count, view.exportable_count),
        strict=True,
    ):
        col.metric(label, count)
    st.caption("已审核但暂不可导出的主题仍需核查资料；审核并不等于正式资料发布。")

    keyword = st.text_input("筛选主题或代表问题", key="teacher_hotspot_filter_keyword").strip().casefold()
    review_filter = st.selectbox(
        "审核状态", ("all", "pending", "approved"),
        format_func=lambda value: {"all": "全部", "pending": "待审核", "approved": "已审核"}[value],
        key="teacher_hotspot_filter_review",
    )
    visible = [
        item for item in view.topics
        if (not keyword or keyword in item.label.casefold() or keyword in item.representative_question.casefold())
        and (review_filter == "all" or item.approved == (review_filter == "approved"))
    ]
    st.caption(f"当前显示 {len(visible)} 类；顶部概览保持当前授权范围的累计口径。")
    if view.topics:
        chart_rows = [{"主题": f"主题 {item.cluster_id}", "提问条数": item.count} for item in view.topics[:10]]
        st.caption("当前课程/班级累计提问条数（前10类）")
        st.bar_chart(pd.DataFrame(chart_rows), x="主题", y="提问条数", use_container_width=True)
    if not visible:
        st.info("当前筛选条件下没有热点主题。")
    for item in visible:
        with st.expander(f"主题 {item.cluster_id}", expanded=False):
            st.text(item.label)
            st.text(f"代表问题：{item.representative_question or '暂无信息'}")
            st.text(f"提问条数：{item.count}；审核状态：{'已审核' if item.approved else '待审核'}")
            st.text(f"最近提问：{item.recent_question_at or '暂无信息'}")
            st.dataframe(pd.DataFrame([
                {"问题": member.get("question") or "暂无信息", "提问时间": member.get("created_at") or "暂无信息",
                 "代表问题": "是" if member.get("is_representative") else "否"}
                for member in item.members
            ]), use_container_width=True, hide_index=True)
            faq = view.faq_by_cluster.get(item.cluster_id)
            if faq:
                st.text("本次可导出的常见问答：")
                st.text(str(faq.get("answer") or "暂无答案"))
                for citation in faq["citations"]:
                    if isinstance(citation, dict):
                        st.text(str(citation.get("title") or "资料来源"))
                        st.text(str(citation.get("snippet") or "暂无摘录"))
            else:
                st.info("尚未形成可导出的常见问答。")
            _render_hotspot_actions(token, item, view.topics)

    st.caption(f"待归类问题：{view.noise_question_count} 条，不计入热点主题。")
    if view.noise_members:
        st.dataframe(pd.DataFrame([
            {"问题": member.get("question") or "暂无信息", "提问时间": member.get("created_at") or "暂无信息"}
            for member in view.noise_members
        ]), use_container_width=True, hide_index=True)
    st.text("导出范围：当前课程/班级已审核且有引用的常见问答；当前筛选不会改变导出范围。")
    _render_phase3_faq_downloads(token, course_id=course_id, class_id=class_id)


def _render_phase3_faq_downloads(token: str, *, course_id: int | None, class_id: int | None) -> None:
    try:
        faq_payload = api_client.teacher_question_cluster_faq_json(token, course_id=course_id, class_id=class_id)
        json_bytes = json.dumps(faq_payload, ensure_ascii=False, indent=2).encode("utf-8")
    except APIClientError as exc:
        st.error(str(exc))
        json_bytes = b""
    st.download_button(
        "导出已审核常见问答（JSON）",
        data=json_bytes,
        file_name="phase3_faq.json",
        mime="application/json",
        disabled=not bool(json_bytes),
        use_container_width=True,
    )
    try:
        csv_bytes = api_client.teacher_phase3_faq_csv(token, course_id=course_id, class_id=class_id)
    except APIClientError as exc:
        st.error(str(exc))
        csv_bytes = b""
    st.download_button(
        "导出已审核常见问答（CSV）",
        data=csv_bytes,
        file_name="phase3_faq.csv",
        mime="text/csv",
        disabled=not bool(csv_bytes),
        use_container_width=True,
    )


def _render_student_timeline_panel(token: str) -> None:
    st.subheader("学生学习轨迹")
    student_id = int(st.number_input("学生编号", min_value=1, step=1, value=1))
    limit = int(st.number_input("最多显示条数", min_value=1, max_value=500, step=10, value=50))
    if not st.button("查看学习轨迹", use_container_width=True):
        return
    try:
        payload = api_client.teacher_phase3_student_timeline(token, student_id, limit=limit)
    except APIClientError:
        st.error("无法查看该学生的学习轨迹，请核对课程范围与学生编号。")
        return
    events = payload.get("events", [])
    if not events:
        st.info("该学生暂无学习轨迹。")
        return
    rows = [
        {
            "时间": item.get("created_at") or item.get("timestamp") or item.get("event_time") or item.get("updated_at") or "暂无信息",
            "事件": _event_label(item.get("event_type")),
            "页面": page_label(item.get("page_name") or item.get("page")),
            "课程编号": item.get("course_id"),
            "班级编号": item.get("class_id"),
            "说明": str(item.get("summary") or ""),
        }
        for item in events
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_hotspot_actions(token: str, topic: Any, topics: tuple[Any, ...]) -> None:
    cluster_id = topic.cluster_id
    with st.form(f"teacher_hotspot_rename_cluster_{cluster_id}"):
        label = st.text_input("修改主题名称", value=topic.label, key=f"teacher_hotspot_rename_label_{cluster_id}")
        confirmed = st.checkbox("确认修改主题名称", key=f"teacher_hotspot_confirm_rename_{cluster_id}")
        if st.form_submit_button("保存名称", use_container_width=True):
            if confirmed and label.strip():
                _call_cluster_action(lambda: api_client.rename_question_cluster(token, cluster_id, label.strip()), "主题名称已更新。")
            else:
                st.warning("请填写名称并确认操作。")
    if not topic.approved:
        with st.form(f"teacher_hotspot_approve_cluster_{cluster_id}"):
            confirmed = st.checkbox("确认审核此主题（不发布课程资料）", key=f"teacher_hotspot_confirm_approve_{cluster_id}")
            if st.form_submit_button("审核此主题", use_container_width=True):
                if confirmed:
                    _call_cluster_action(lambda: api_client.approve_question_cluster(token, cluster_id), "主题已审核。")
                else:
                    st.warning("请先确认审核操作。")

    with st.expander("高级整理操作", expanded=False):
        other_topics = {item.cluster_id: item for item in topics if item.cluster_id != cluster_id}
        if other_topics:
            with st.form(f"teacher_hotspot_merge_cluster_{cluster_id}"):
                sources = st.multiselect(
                    "并入此主题的来源主题", list(other_topics),
                    format_func=lambda cid: f"主题 {cid}：{other_topics[cid].label}",
                    key=f"teacher_hotspot_merge_sources_{cluster_id}",
                )
                merged_label = st.text_input("合并后的主题名称", value=topic.label, key=f"teacher_hotspot_merge_label_{cluster_id}")
                confirmed = st.checkbox("确认合并所选主题", key=f"teacher_hotspot_confirm_merge_{cluster_id}")
                if st.form_submit_button("合并主题", use_container_width=True):
                    if sources and confirmed:
                        _call_cluster_action(
                            lambda: api_client.merge_question_clusters(
                                token, target_cluster_id=cluster_id, source_cluster_ids=list(sources),
                                label=merged_label.strip() or None,
                            ), "主题已合并。",
                        )
                    else:
                        st.warning("请选择来源主题并确认合并。")
        members = {member["question_id"]: member for member in topic.members if isinstance(member.get("question_id"), int)}
        if members:
            with st.form(f"teacher_hotspot_split_cluster_{cluster_id}"):
                question_ids = st.multiselect(
                    "移入新主题的问题", list(members),
                    format_func=lambda qid: str(members[qid].get("question") or f"问题 {qid}"),
                    key=f"teacher_hotspot_split_members_{cluster_id}",
                )
                split_label = st.text_input("新主题名称", value="新主题", key=f"teacher_hotspot_split_label_{cluster_id}")
                confirmed = st.checkbox("确认拆分所选问题", key=f"teacher_hotspot_confirm_split_{cluster_id}")
                if st.form_submit_button("拆分问题", use_container_width=True):
                    if question_ids and split_label.strip() and confirmed:
                        _call_cluster_action(
                            lambda: api_client.split_question_cluster(
                                token, cluster_id, question_ids=list(question_ids), label=split_label.strip(),
                            ), "问题已拆分为新主题。",
                        )
                    else:
                        st.warning("请选择问题、填写新名称并确认拆分。")


def _render_knowledge_management() -> None:
    token = _auth_token()
    if not token:
        st.info("请先登录教师或管理员账号后再管理知识库。")
        return
    st.subheader("上传知识库文档")
    try:
        courses = api_client.courses(token)
    except APIClientError as exc:
        st.error(str(exc))
        return
    if not courses:
        st.warning("当前账号没有可管理的课程。")
        return
    course_options = {int(item["id"]): item for item in courses if isinstance(item, dict) and item.get("id")}
    if not course_options:
        st.warning("当前账号没有可管理的课程。")
        return
    course_id = st.selectbox(
        "知识库课程",
        list(course_options),
        format_func=lambda cid: str(course_options[cid].get("name") or course_options[cid].get("slug") or "未命名课程"),
        key="teacher_knowledge_course_selector",
    )
    uploaded_files = st.file_uploader(
        "支持 PDF、DOCX、Markdown 和 TXT，单文件最大 25MB",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        max_upload_size=25,
    )
    st.caption("上传后先进入“待索引”；索引成功仍保持未发布，只有授权教师明确发布后学生才可见。")
    if uploaded_files and st.button("上传并切片", use_container_width=True):
        for file in uploaded_files:
            try:
                result = api_client.upload_knowledge(
                    file.name,
                    file.getvalue(),
                    token=token,
                    course_id=course_id,
                )
            except APIClientError as exc:
                st.error(str(exc))
                continue
            if result.get("duplicate"):
                st.info("检测到重复内容，已复用现有文档。")
            else:
                st.success("已记录文档并完成切片；请核对索引与发布状态。")
            st.caption(
                f"索引状态：{_index_status_label(result.get('index_status', 'PENDING'))}；"
                "已索引仅表示技术处理完成，明确发布后学生才可使用。"
            )
        st.cache_data.clear()

    try:
        payload = api_client.knowledge_documents(token=token, course_id=course_id)
    except APIClientError as exc:
        st.error(str(exc))
        return

    _render_document_table(payload.get("documents", []))
    _render_index_jobs(payload.get("index_jobs", []))


def _render_document_table(documents: list[dict[str, Any]]) -> None:
    st.subheader("课程文档")
    if not documents:
        st.info("暂无已上传文档。")
        return

    st.dataframe(pd.DataFrame(_document_status_rows(documents)), use_container_width=True, hide_index=True)

    st.markdown("#### 文档操作")
    for doc in documents:
        with st.expander(f"文档 {doc['id']}", expanded=False):
            st.text(str(doc.get("filename") or "未命名文件"))
            st.text(f"索引状态：{_index_status_label(doc.get('index_status'))}；学生可见性：{_publication_status_label(doc)}")
            st.text(f"最近索引任务：{_index_status_label(doc.get('latest_job_status'))}")
            latest_error = _latest_job_error(doc.get("jobs", []))
            if latest_error:
                st.error("最近索引任务失败，请核对资料后重试。")
            col_a, col_b, col_c = st.columns(3)
            if col_a.button("手动重建索引", key=f"reindex_{doc['id']}", use_container_width=True):
                _call_reindex(int(doc["id"]), _auth_token())
            if doc.get("published_to_students"):
                if col_b.button("撤回发布", key=f"unpublish_{doc['id']}", use_container_width=True):
                    _call_document_action(api_client.unpublish_document, int(doc["id"]), "已撤回学生可见性。")
            elif col_b.button(
                "发布给学生",
                key=f"publish_{doc['id']}",
                disabled=doc.get("index_status") != "INDEXED",
                use_container_width=True,
            ):
                _call_document_action(api_client.publish_document, int(doc["id"]), "文档已发布给本课程学生。")
            if col_c.button("删除文档", key=f"delete_{doc['id']}", use_container_width=True):
                _call_delete(int(doc["id"]), _auth_token())


def _render_index_jobs(jobs: list[dict[str, Any]]) -> None:
    st.subheader("索引任务")
    if not jobs:
        st.info("暂无索引任务。")
        return
    df = pd.DataFrame(
        [
            {
                "任务编号": item.get("id"),
                "文档编号": item.get("document_id"),
                "文件名": item.get("filename"),
                "状态": _index_status_label(item.get("status")),
                "问题说明": "索引失败" if item.get("status") == "FAILED" else "-",
                "更新时间": item.get("updated_at"),
            }
            for item in jobs
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    if "状态" in df and not df[df["状态"] == "索引失败"].empty:
        st.error("存在失败索引任务，请展开课程文档查看失败原因。")


def _document_status_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        action = document.get("publication_last_action") or {}
        rows.append(
            {
                "文档编号": document.get("id"),
                "文件名": document.get("filename"),
                "格式": document.get("file_type"),
                "切片数": document.get("chunk_count"),
                "索引状态": _index_status_label(document.get("index_status")),
                "学生可见性": _publication_status_label(document),
                "发布人": document.get("published_by") or "-",
                "发布时间": document.get("published_at") or "-",
                "最近操作": _publication_action_label(action.get("action")),
                "最近操作时间": action.get("at") or "-",
                "更新时间": document.get("updated_at") or "-",
            }
        )
    return rows


def _index_status_label(status: Any) -> str:
    return {
        "PENDING": "待索引",
        "PARSING": "解析中",
        "CHUNKING": "切片中",
        "EMBEDDING": "建立索引中",
        "INDEXED": "已索引",
        "FAILED": "索引失败",
    }.get(str(status or "").upper(), "状态未知")


def _publication_status_label(document: dict[str, Any]) -> str:
    return "已发布" if document.get("published_to_students") is True else "未发布"


def _publication_action_label(action: Any) -> str:
    return {
        "publish": "发布",
        "unpublish": "撤回发布",
        "reindex": "重建索引",
        "index_failed": "索引失败",
    }.get(str(action or ""), "-")


def _resource_sync_connector_options() -> dict[str, str]:
    return {
        "arxiv": "paper",
        "semantic_scholar": "paper",
        "nvd": "vulnerability",
        "rfc_editor": "standard",
        "standard_source": "standard",
        "github": "project",
    }


def _render_resource_sync() -> None:
    token = _auth_token()
    if not token:
        st.info("请先登录教师或管理员账号。")
        return
    st.subheader("外部资料更新")
    try:
        status_payload = api_client.connectors_status()
        runs_payload = api_client.connector_sync_runs(token, limit=20)
    except APIClientError as exc:
        st.error(str(exc))
        return

    connectors = status_payload.get("connectors", [])
    source_labels = {"arxiv": "arXiv 论文", "semantic_scholar": "Semantic Scholar 论文",
                     "nvd": "NVD 漏洞", "rfc_editor": "RFC 标准", "standard_source": "本地标准", "github": "GitHub 项目"}
    if connectors:
        rows = [
            {"资料来源": source_labels[item["name"]],
             "启用状态": "已启用" if item.get("enabled") is True else "未启用"}
            for item in connectors if isinstance(item, dict) and item.get("name") in source_labels
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    connector_options = _resource_sync_connector_options()
    with st.form("resource_sync_form"):
        cols = st.columns([0.24, 0.26, 0.36, 0.14])
        connector = cols[0].selectbox("资料来源", list(connector_options), index=0,
                                      format_func=lambda value: source_labels.get(value, "来源待核实"))
        type_options = ["paper", "vulnerability", "standard", "project"]
        resource_type = cols[1].selectbox(
            "资料类型",
            type_options,
            index=type_options.index(connector_options[connector]),
            format_func=lambda value: {"paper": "论文", "vulnerability": "漏洞", "standard": "标准", "project": "开源项目"}[value],
        )
        query = cols[2].text_input("查询词", value="")
        page_size = cols[3].number_input("结果数量", min_value=1, max_value=10, value=3, step=1)
        submitted = st.form_submit_button("同步资料", use_container_width=True)
    if submitted:
        try:
            result = api_client.sync_connector(
                token,
                connector,
                query=query,
                resource_type=resource_type,
                page_size=int(page_size),
            )
        except APIClientError as exc:
            st.error(str(exc))
        else:
            if result.get("status") == "SUCCESS":
                st.success(f"资料同步完成，获得 {result.get('result_count', 0)} 条。")
            else:
                st.warning("资料同步未完成，请稍后查看同步记录或重试。")
            st.cache_data.clear()
            st.rerun()

    runs = runs_payload.get("items", [])
    if runs:
        st.subheader("最近同步记录")
        st.dataframe(pd.DataFrame([
            {"资料来源": source_labels.get(item.get("connector_name") or item.get("source_name"), "来源待核实"),
             "状态": {"SUCCESS": "已完成", "FAILED": "失败", "RUNNING": "处理中", "PENDING": "等待中"}.get(item.get("status"), "状态待核实"),
             "数量": item.get("result_count") if isinstance(item.get("result_count"), int) else "暂无数据",
             "更新时间": item.get("updated_at") or "暂无信息"}
            for item in runs if isinstance(item, dict)
        ]), use_container_width=True, hide_index=True)
    else:
        st.info("暂无资料同步记录。")


def _render_system_status() -> None:
    try:
        status = api_client.system_status()
        tools = api_client.get_tools()
    except APIClientError as exc:
        st.error(str(exc))
        return

    cols = st.columns(3)
    cols[0].metric("生成方式", model_status(status))
    cols[1].metric("模型服务", "已配置" if status.get("deepseek_configured") else "未配置")
    cols[2].metric("本地降级", "已启用" if status.get("fallback_active") else "未启用")
    if not status.get("deepseek_configured"):
        st.warning("模型服务暂未配置，部分问答将使用本地生成方式。")
    st.dataframe(pd.DataFrame(health_rows(status)), use_container_width=True, hide_index=True)
    rows = tool_rows(tools)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("暂无可展示的工具状态。")


def _render_core_metrics(analytics: dict[str, Any]) -> None:
    total = int(analytics.get("total_questions", 0))
    abstained = int(analytics.get("abstained_count", 0))
    average_confidence = float(analytics.get("average_confidence", 0.0))
    abstain_rate = float(analytics.get("abstain_rate", 0.0))
    cols = st.columns(4)
    cols[0].metric("总提问数", total)
    cols[1].metric("拒答次数", abstained)
    cols[2].metric("拒答率", f"{abstain_rate:.0%}")
    cols[3].metric("平均置信度", f"{average_confidence:.2f}")


def _render_phase2_metrics(analytics: dict[str, Any]) -> None:
    model = analytics.get("model_metrics", {})
    agent = analytics.get("agent_metrics", {})
    retrieval = analytics.get("retrieval_metrics", {})
    feedback = analytics.get("feedback_metrics", {})
    cols = st.columns(6)
    cols[0].metric("DeepSeek 成功率", f"{float(model.get('success_rate', 0.0)):.0%}")
    cols[1].metric("模型平均耗时", f"{float(model.get('average_latency_ms', 0.0)):.0f} ms")
    cols[2].metric("本地降级次数", int(model.get("fallback_count", 0)))
    cols[3].metric("平均处理步骤", f"{float(agent.get('average_steps', 0.0)):.2f}")
    cols[4].metric("知识库命中率", f"{float(retrieval.get('knowledge_hit_rate', 0.0)):.0%}")
    cols[5].metric("平均评分", f"{float(feedback.get('average_rating', 0.0)):.2f}")


def _render_module_distribution(analytics: dict[str, Any]) -> None:
    st.subheader("模块分布")
    distribution = analytics.get("module_distribution", {})
    if not distribution:
        st.info("暂无模块统计。学生提问后这里会显示分布。")
        return
    df = pd.DataFrame([{"模块": _module_label(module), "提问数": count}
                       for module, count in distribution.items() if isinstance(count, int)])
    if df.empty:
        st.info("暂无可展示的模块统计。")
        return
    df = df.sort_values("提问数", ascending=False)
    st.bar_chart(df, x="模块", y="提问数", use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_agent_paths(analytics: dict[str, Any]) -> None:
    st.subheader("处理路径统计")
    agent = analytics.get("agent_metrics", {})
    tools = analytics.get("tool_metrics", {})
    cols = st.columns(2)
    cols[0].metric("等待补充次数", int(agent.get("requires_input_count", 0)))
    cols[1].metric("步骤上限次数", int(agent.get("step_limit_reached_count", 0)))
    _table_if_any("最常选择的动作", [
        {"处理环节": agent_action_label(item.get("action")), "次数": item.get("count")}
        for item in agent.get("top_actions", []) if isinstance(item, dict)
    ])
    _table_if_any("最常调用的工具", [
        {"工具": teacher_tool_label(item.get("tool_name")), "次数": item.get("count")}
        for item in tools.get("top_tools", []) if isinstance(item, dict)
    ])
    _table_if_any("停止原因", [
        {"状态": agent_stop_label(item.get("reason")), "次数": item.get("count")}
        for item in agent.get("stop_reasons", []) if isinstance(item, dict)
    ])


def _render_recent_questions(analytics: dict[str, Any]) -> None:
    st.subheader("最近问题")
    recent = analytics.get("recent_questions", [])
    if not recent:
        st.info("暂无最近问题。")
        return
    df = pd.DataFrame([
        {"问题预览": item.get("question") or "暂无信息", "模块": _module_label(item.get("module")),
         "是否拒答": "是" if item.get("abstained") is True else "否" if item.get("abstained") is False else "暂无信息"}
        for item in recent if isinstance(item, dict)
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_module_stuck_points(analytics: dict[str, Any]) -> None:
    st.subheader("模块卡点")
    rows = analytics.get("module_stuck_points", [])
    if not rows:
        st.info("暂无明显模块卡点。")
        return
    st.dataframe(pd.DataFrame([
        {"模块": _module_label(item.get("module")), "卡点问题": item.get("stuck_count") if isinstance(item.get("stuck_count"), int) else "暂无数据",
         "模块总问题": item.get("total") if isinstance(item.get("total"), int) else "暂无数据"}
        for item in rows if isinstance(item, dict)
    ]), use_container_width=True, hide_index=True)


def _render_assignment_stuck_analysis() -> None:
    st.subheader("实验步骤卡点")
    token = _auth_token()
    if not token:
        st.info("登录教师账号后可查看实验步骤卡点。")
        return
    try:
        payload = api_client.teacher_stuck_analysis(token)
    except APIClientError as exc:
        st.error(str(exc))
        return
    rows = payload.get("items", [])
    if not rows:
        st.info("暂无主动上报或疑似实验卡点。")
        return
    st.dataframe(pd.DataFrame([
        {"所在步骤": item.get("step_title") or "暂无信息", "学生说明": item.get("reason") or "暂无信息",
         "处理状态": _progress_label(item.get("status"))}
        for item in rows if isinstance(item, dict)
    ]), use_container_width=True, hide_index=True)
    _render_stuck_review_actions(token, rows, key_prefix="legacy")


def _render_stuck_review_actions(token: str, rows: list[dict[str, Any]], *, key_prefix: str) -> None:
    for item in rows:
        projection = _stuck_review_projection(item)
        if projection is None:
            continue
        report_id, actions = projection
        with st.expander(f"处理卡点 {report_id}", expanded=False):
            st.text(f"处理状态：{_progress_label(item.get('status'))}")
            st.text(str(item.get("reason") or "暂无学生说明"))
            evidence = item.get("evidence")
            if isinstance(evidence, dict):
                for field, label in (("step", "所在步骤"), ("summary", "依据摘要"), ("source", "依据来源")):
                    if evidence.get(field):
                        st.text(f"{label}：{evidence[field]}")
            if not actions:
                continue
            cols = st.columns(len(actions))
            for col, (label, action) in zip(cols, actions, strict=True):
                if col.button(label, key=f"{key_prefix}_stuck_{report_id}_{action}", use_container_width=True):
                    try:
                        api_client.update_stuck_report(token, report_id, action)
                    except APIClientError as exc:
                        st.error(str(exc))
                    else:
                        st.success("卡点状态已更新。")
                        st.cache_data.clear()
                        st.rerun()


def _stuck_review_projection(item: dict[str, Any]) -> tuple[int, tuple[tuple[str, str], ...]] | None:
    """Return the only write actions allowed by the current trusted report status."""
    raw_report_id = item.get("id")
    if type(raw_report_id) is int:
        report_id = raw_report_id
    elif type(raw_report_id) is str and re.fullmatch(r"[1-9][0-9]*", raw_report_id):
        try:
            report_id = int(raw_report_id)
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        return None
    if not 1 <= report_id <= 9_223_372_036_854_775_807:
        return None

    status = _normalize_progress_status(item.get("status"))
    actions: dict[str, tuple[tuple[str, str], ...]] = {
        "active": (("确认", "confirm"), ("忽略", "ignore"), ("标记已解决", "resolve")),
        "confirmed": (("标记已解决", "resolve"),),
    }
    return report_id, actions.get(status, ())


def _call_cluster_action(action: Any, success_message: str) -> None:
    try:
        action()
    except APIClientError as exc:
        st.error(str(exc))
    else:
        st.success(success_message)
        st.cache_data.clear()
        st.rerun()


def _render_feedback_metrics(analytics: dict[str, Any]) -> None:
    st.subheader("学生反馈")
    feedback = analytics.get("feedback_metrics", {})
    cols = st.columns(3)
    cols[0].metric("反馈数", int(feedback.get("count", 0)))
    cols[1].metric("平均评分", f"{float(feedback.get('average_rating', 0.0)):.2f}")
    cols[2].metric("有帮助率", f"{float(feedback.get('helpful_rate', 0.0)):.0%}")
    _table_if_any("最近反馈", [
        {"评分": item.get("rating") if isinstance(item.get("rating"), int) else "暂无数据",
         "说明": item.get("comment") or "暂无信息"}
        for item in feedback.get("recent", []) if isinstance(item, dict)
    ])


def _render_faq(faq: dict[str, Any]) -> None:
    st.subheader("常见问答概览")
    items = faq.get("items", [])
    if not items:
        st.info("暂无 FAQ。")
        return
    df = pd.DataFrame([
        {"代表问题": item.get("representative_question") or "暂无信息", "次数": item.get("count"),
         "模块": _module_label(item.get("module"))}
        for item in items if isinstance(item, dict)
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.info("按课程/班级导出已审核常见问答，请使用“热点问题库”中的受权下载入口。")
    top_item = items[0]
    st.text(f"当前最高频问题：{top_item.get('representative_question') or '暂无信息'}")
    st.caption(f"提问次数：{top_item.get('count', '暂无数据')}；模块：{_module_label(top_item.get('module'))}")


def _table_if_any(title: str, rows: list[dict[str, Any]]) -> None:
    if rows:
        st.markdown(title)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _dict_rows(values: dict[str, Any], key_name: str, value_name: str) -> list[dict[str, Any]]:
    return [{key_name: key, value_name: value} for key, value in values.items()]


def _call_reindex(document_id: int, token: str | None = None) -> None:
    try:
        result = api_client.reindex_document(document_id, token=token)
    except APIClientError as exc:
        st.error(str(exc))
        st.cache_data.clear()
        return
    st.success("索引重建已提交。")
    st.cache_data.clear()
    st.rerun()


def _call_document_action(action: Any, document_id: int, message: str) -> None:
    try:
        result = action(document_id, token=_auth_token())
    except APIClientError as exc:
        st.error(str(exc))
        st.cache_data.clear()
        return
    st.success(message)
    st.cache_data.clear()
    st.rerun()


def _call_delete(document_id: int, token: str | None = None) -> None:
    try:
        result = api_client.delete_document(document_id, token=token)
    except APIClientError as exc:
        st.error(str(exc))
        st.cache_data.clear()
        return
    st.success("文档已删除。")
    st.cache_data.clear()
    st.rerun()


def _latest_job_error(jobs: list[dict[str, Any]]) -> str | None:
    for job in jobs:
        if job.get("status") == "FAILED" and job.get("error_message"):
            return str(job["error_message"])
    return None


def _safe_csv_bytes(df: pd.DataFrame) -> bytes:
    safe_df = df.map(_escape_csv_formula)
    return safe_df.to_csv(index=False).encode("utf-8-sig")


def _escape_csv_formula(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def _auth_token() -> str | None:
    return st.session_state.get("auth_token")


def _current_role() -> str:
    return str(st.session_state.get("current_user", {}).get("role", "")).lower()


@st.cache_data(ttl=5)
def _load_analytics(token: str | None = None) -> dict[str, Any]:
    return api_client.teacher_analytics(token=token)


@st.cache_data(ttl=5)
def _load_faq(token: str | None = None) -> dict[str, Any]:
    return api_client.teacher_faq(token=token)


if __name__ == "__main__":
    render()
