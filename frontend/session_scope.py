"""Shared authenticated course/class selection and learning-session binding."""

from __future__ import annotations

from uuid import uuid4

import streamlit as st

from frontend import api_client
from frontend.api_client import APIClientError


SCOPE_DEPENDENT_KEYS = (
    "last_chat_result",
    "last_general_chat_result",
    "last_guide_result",
    "last_practice_draft_result",
    "practice_draft_result_scope",
    "chat_result_history",
    "resource_search_payload",
    "resource_search_fingerprint",
    "resource_page",
    "resource_keyword",
    "resource_topic",
    "resource_type",
    "resource_source",
    "resource_use_year_from",
    "resource_use_year_to",
    "assignment_context",
    "innovation_session",
    "innovation_session_scope",
    "innovation_resume_selection",
    "innovation_suggestion_result",
    "innovation_suggestion_result_scope",
    "innovation_suggestion_use_course_references",
    "use_course_references",
    "question_input",
    "general_question_input",
    "practice_draft_topic",
    "practice_draft_objective",
    "practice_draft_environment",
    "practice_draft_tools",
    "practice_draft_external_query",
    "practice_draft_external_type",
    "practice_draft_allow_external",
)


def ensure_student_scope() -> tuple[int, int | None] | None:
    token = st.session_state.get("auth_token")
    if not token:
        _discard_learning_session(None)
        return None
    try:
        available_courses = api_client.courses(token)
    except APIClientError as exc:
        _discard_learning_session(token)
        st.error(str(exc))
        return None
    if not available_courses:
        _discard_learning_session(token)
        st.warning("当前账号没有可访问的课程。")
        return None

    course_options = {int(item["id"]): item for item in available_courses}
    selected_course = _explicit_choice(
        "当前课程",
        course_options,
        state_key="selected_course_id",
        widget_key="student_course_selector",
        render_label=lambda item: str(item.get('name') or item.get('slug') or '未命名课程'),
    )
    if selected_course is None:
        _discard_learning_session(token)
        st.info("请选择课程后再进入学习流程。")
        return None

    try:
        available_classes = api_client.classes(token, course_id=selected_course)
    except APIClientError as exc:
        _discard_learning_session(token)
        st.error(str(exc))
        return None
    class_options = {int(item["id"]): item for item in available_classes}
    selected_class: int | None = None
    if class_options:
        selected_class = _explicit_choice(
            "当前班级",
            class_options,
            state_key="selected_class_id",
            widget_key="student_class_selector",
            render_label=lambda item: str(item.get('name') or item.get('slug') or '未命名班级'),
        )
        if selected_class is None:
            _discard_learning_session(token)
            st.info("当前课程包含多个可访问班级，请明确选择班级。")
            return None

    scope = (selected_course, selected_class)
    previous_scope = st.session_state.get("learning_session_scope")
    if previous_scope != scope or not st.session_state.get("client_session_id"):
        _discard_learning_session(token)
        session_id = f"streamlit-{uuid4()}"
        try:
            api_client.create_learning_session(
                token,
                session_id,
                course_id=selected_course,
                class_id=selected_class,
            )
        except APIClientError as exc:
            st.error(str(exc))
            return None
        st.session_state.client_session_id = session_id
        st.session_state.learning_session_scope = scope
    return scope


def _explicit_choice(label_text, options, *, state_key, widget_key, render_label):
    current = st.session_state.get(state_key)
    try:
        current_id = int(current) if current is not None else None
    except (TypeError, ValueError):
        current_id = None
    valid_current = current_id if current_id in options else None
    if len(options) == 1:
        selected = next(iter(options))
        # Course/class names are user-controlled: display them without Markdown.
        st.text(f"{label_text}：{render_label(options[selected])}")
    else:
        selected = st.selectbox(
            label_text,
            [None, *options.keys()],
            index=([None, *options.keys()].index(valid_current) if valid_current is not None else 0),
            format_func=lambda value: "请选择" if value is None else render_label(options[value]),
            key=widget_key,
        )
    if current is not None and selected != current_id:
        _clear_scope_dependent_state()
        if state_key == "selected_course_id":
            st.session_state.pop("selected_class_id", None)
            st.session_state.pop("student_class_selector", None)
    if selected is None:
        st.session_state.pop(state_key, None)
    else:
        st.session_state[state_key] = int(selected)
    return selected


def _clear_scope_dependent_state() -> None:
    for key in SCOPE_DEPENDENT_KEYS:
        st.session_state.pop(key, None)


def _discard_learning_session(token: str | None) -> None:
    previous_id = st.session_state.get("client_session_id")
    if token and previous_id:
        try:
            api_client.end_learning_session(token, previous_id)
        except APIClientError:
            pass
    st.session_state.client_session_id = ""
    st.session_state.pop("learning_session_scope", None)
    _clear_scope_dependent_state()
