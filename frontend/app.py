import pandas as pd
import streamlit as st

from frontend import api_client
from frontend.api_client import APIClientError
from frontend.pages import innovation, student, teacher
from frontend.presentation import (
    PRODUCT_NAME,
    PRODUCT_SUBTITLE,
    health_rows,
    model_status,
    navigation_pages,
    page_label,
    role_label,
    safe_error_message,
    tool_rows,
)


st.set_page_config(page_title=PRODUCT_NAME, page_icon="🎓", layout="wide")


AUTH_SESSION_KEYS = ("auth_token", "current_user", "client_session_id")
USER_SCOPED_SESSION_KEYS = (
    "last_chat_result",
    "last_guide_result",
    "last_practice_draft_result",
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
    "selected_course_id",
    "selected_class_id",
    "student_class_selector",
    "learning_session_scope",
    "use_course_references",
    "student_course_selector",
    "innovation_course_selector",
    "teacher_knowledge_course_selector",
    "question_input",
    "practice_draft_topic",
    "practice_draft_objective",
    "practice_draft_environment",
    "practice_draft_tools",
    "practice_draft_external_query",
    "practice_draft_external_type",
    "practice_draft_allow_external",
    "practice_draft_result_scope",
    "app_navigation_page",
)
USER_SCOPED_SESSION_PREFIXES = (
    "teacher_hotspot_",
    "rename_label_",
    "merge_sources_",
    "merge_label_",
    "split_members_",
    "split_label_",
)


def main() -> None:
    _render_login()
    if not st.session_state.get("auth_token"):
        st.header(PRODUCT_NAME)
        st.text(PRODUCT_SUBTITLE)
        st.info("请先在侧边栏登录。")
        return
    if _must_change_password():
        st.warning("首次登录必须先修改默认密码，修改完成后才能使用课程智能体功能。")
        return
    st.sidebar.title(PRODUCT_NAME)
    pages = _navigation_pages()
    if st.session_state.get("app_navigation_page") not in pages:
        st.session_state["app_navigation_page"] = pages[0]
    page = st.sidebar.radio(
        "页面",
        pages,
        label_visibility="collapsed",
        format_func=page_label,
        key="app_navigation_page",
    )

    try:
        if page == "student":
            student.render()
            _track_page_view("student")
        elif page == "innovation":
            innovation.render()
            _track_page_view("innovation")
        elif page == "teacher":
            _track_page_view("teacher")
            teacher.render()
        elif page == "tools" and _current_role() in {"teacher", "admin"}:
            _track_page_view("tools")
            render_tools()
        else:
            _track_page_view("about")
            render_about()
    except APIClientError as exc:
        st.error(str(exc))
    except Exception:
        st.error(safe_error_message())


def _navigation_pages() -> list[str]:
    return navigation_pages(_current_role())


def _current_role() -> str:
    return str(st.session_state.get("current_user", {}).get("role", "")).lower()


def _must_change_password() -> bool:
    return bool(st.session_state.get("auth_token")) and bool(
        st.session_state.get("current_user", {}).get("must_change_password")
    )


def _render_login() -> None:
    st.sidebar.subheader("账号登录")
    if st.session_state.get("auth_token"):
        user = st.session_state.get("current_user", {})
        st.sidebar.text(str(user.get("display_name") or "未设置姓名"))
        st.sidebar.caption(role_label(user.get("role")))
        if user.get("must_change_password"):
            _render_change_password_form()
        if st.sidebar.button("退出登录", use_container_width=True):
            token = st.session_state.get("auth_token")
            session_id = st.session_state.get("client_session_id")
            if token and session_id:
                try:
                    api_client.end_learning_session(token, session_id)
                except APIClientError:
                    pass
            if token:
                try:
                    api_client.logout(token)
                except APIClientError:
                    pass
            _clear_auth_session()
            st.rerun()
        return

    with st.sidebar.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", use_container_width=True)
    if not submitted:
        return
    try:
        payload = api_client.login(username.strip(), password)
        token = payload["access_token"]
    except APIClientError as exc:
        st.sidebar.error(str(exc))
        return
    except Exception:
        st.sidebar.error(safe_error_message())
        return
    _clear_user_scoped_session()
    st.session_state.auth_token = token
    st.session_state.current_user = payload.get("user", {})
    st.session_state.client_session_id = ""
    st.sidebar.success("登录成功")
    st.rerun()


def _render_change_password_form() -> None:
    st.sidebar.warning("首次登录请修改密码，完成后即可使用课程功能。")
    with st.sidebar.form("change_password_form"):
        current_password = st.text_input("当前密码", type="password")
        new_password = st.text_input("新密码", type="password")
        confirm_password = st.text_input("确认新密码", type="password")
        submitted = st.form_submit_button("修改密码", use_container_width=True)
    if not submitted:
        return
    if new_password != confirm_password:
        st.sidebar.error("两次输入的新密码不一致。")
        return
    token = st.session_state.get("auth_token")
    if not token:
        st.sidebar.error("登录已失效，请重新登录。")
        _clear_auth_session()
        return
    try:
        changed = api_client.change_password(token, current_password, new_password)
        refreshed_token = changed.get("access_token") or token
    except APIClientError as exc:
        st.sidebar.error(str(exc))
        return
    except Exception:
        st.sidebar.error(safe_error_message())
        return
    st.session_state.auth_token = refreshed_token
    st.session_state.current_user = changed.get(
        "user",
        {**st.session_state.get("current_user", {}), "must_change_password": False},
    )
    st.session_state.client_session_id = ""
    st.sidebar.success("密码修改成功，下次请使用新密码登录。")
    st.rerun()


def _clear_auth_session() -> None:
    _clear_user_scoped_session()
    for key in AUTH_SESSION_KEYS:
        st.session_state.pop(key, None)


def _clear_user_scoped_session() -> None:
    for key in list(st.session_state.keys()):
        key_text = str(key)
        if key_text in USER_SCOPED_SESSION_KEYS or any(
            key_text.startswith(prefix) for prefix in USER_SCOPED_SESSION_PREFIXES
        ):
            st.session_state.pop(key, None)


def _track_page_view(page_name: str) -> None:
    token = st.session_state.get("auth_token")
    session_id = st.session_state.get("client_session_id")
    if not token or not session_id:
        return
    selected_course_id = st.session_state.get("selected_course_id")
    selected_class_id = st.session_state.get("selected_class_id")
    bound_scope = st.session_state.get("learning_session_scope")
    if selected_course_id is None or not isinstance(bound_scope, (tuple, list)) or len(bound_scope) != 2:
        return
    try:
        selected_course = int(selected_course_id)
        selected_class = int(selected_class_id) if selected_class_id is not None else None
        bound_course = int(bound_scope[0])
        bound_class = int(bound_scope[1]) if bound_scope[1] is not None else None
    except (TypeError, ValueError):
        return
    if (bound_course, bound_class) != (selected_course, selected_class):
        return
    try:
        api_client.heartbeat_learning_session(token, session_id)
        api_client.record_learning_event(
            token,
            session_id,
            "PAGE_VIEW",
            page_name=page_name,
            dedupe_key=f"{session_id}:PAGE_VIEW:{page_name}",
            course_id=selected_course,
            class_id=selected_class,
        )
    except APIClientError:
        return


def render_tools() -> None:
    st.header("运行状态")
    try:
        status = api_client.system_status()
        tools = api_client.get_tools()
    except APIClientError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error(safe_error_message())
        return

    st.metric("生成方式", model_status(status))
    st.dataframe(pd.DataFrame(health_rows(status)), use_container_width=True, hide_index=True)
    rows = tool_rows(tools)
    st.metric("可识别工具数", len(rows))
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("暂无可展示的工具状态。")


def render_about() -> None:
    st.header("使用帮助")
    st.markdown(
        """
        在“课程学习”中选择课程和班级，再进行课程问答、资料查证或实践指导。
        “创新探索”可帮助整理待验证的想法；教师可在“教学管理”中查看和管理授权资料。

        请核对回答所列来源和引用。模型生成内容及外部资料可能不完整或过时，
        重要结论应以课程要求和权威资料复核。课程内容是否用于模型参考，
        以当前页面明确的选择和课程授权为准。
        """
    )


if __name__ == "__main__":
    main()
