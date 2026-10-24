"""Application-shell presentation and safe error projection tests."""

from __future__ import annotations

import requests
import pytest
import streamlit as st
from copy import deepcopy

from frontend import api_client
from frontend import app
from frontend import presentation as view
from frontend import session_scope


def test_navigation_uses_stable_ids_and_role_specific_entries():
    assert view.navigation_pages("student") == ["student", "innovation", "about"]
    assert view.navigation_pages("teacher") == ["student", "innovation", "teacher", "tools", "about"]
    assert view.navigation_pages("admin") == view.navigation_pages("teacher")
    assert view.page_label("tools") == "运行状态"
    assert view.role_label("student") == "学生"
    assert view.role_label("teacher") == "教师"
    assert view.role_label("admin") == "管理员"


def test_progress_label_projects_active_and_terminal_stuck_states_safely():
    assert view.progress_label("active") == "待处理"
    assert view.progress_label("reported") == "待处理"
    assert view.progress_label("confirmed") == "已确认"
    assert view.progress_label("resolved") == "已解决"
    assert view.progress_label("ignored") == "已忽略"
    assert view.progress_label("unknown") == "状态待核实"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("active", "待处理"), (" ACTIVE ", "待处理"), ("AcTiVe", "待处理"),
        ("reported", "待处理"), (" Reported ", "待处理"),
        ("confirmed", "已确认"), (" Confirmed ", "已确认"),
        ("resolved", "已解决"), (" RESOLVED ", "已解决"),
        ("ignored", "已忽略"), (" IgNoReD ", "已忽略"),
        ("", "状态待核实"), ("unknown", "状态待核实"), (None, "状态待核实"),
        (True, "状态待核实"), (17, "状态待核实"), ([], "状态待核实"), ({}, "状态待核实"),
    ],
)
def test_progress_label_normalizes_only_bounded_string_states(value, expected):
    assert view.progress_label(value) == expected


def test_status_projection_uses_only_allowlisted_fields_and_safe_states():
    malicious = "![track](https://evil.test/x) token=secret C:/private Traceback"
    status = {
        "provider": "deepseek", "model": malicious,
        "sqlite": {"status": "ok", "detail": malicious},
        "chroma": {"status": "error", "detail": malicious},
        "mcp": {"status": "fallback", "detail": malicious},
    }
    tools = [
        {"name": "course_retriever", "enabled": True, "description": malicious, "mcp_error": malicious},
        {"name": "paper_search_real", "enabled": False, "status": malicious},
        {"name": malicious, "enabled": True},
    ]
    assert view.model_status(status) == "模型服务"
    assert view.health_rows(status) == [
        {"组件": "资料记录", "状态": "正常"},
        {"组件": "课程检索", "状态": "暂不可用"},
        {"组件": "工具连接", "状态": "降级运行"},
    ]
    rows = view.tool_rows(tools)
    assert len(rows) == 2
    assert rows[0] == {"工具": "课程资料检索", "启用状态": "已启用"}
    assert malicious not in str(rows + view.health_rows(status))
    assert view.health_rows({})[0]["状态"] == "状态待核实"
    assert view.health_rows({"mcp": {"status": "loading"}})[2]["状态"] == "处理中"
    assert view.health_rows({"mcp": {"status": "warning"}})[2]["状态"] == "需要关注"
    assert view.tool_rows([]) == []


def test_client_error_rejects_upstream_text_and_validates_feedback_id():
    malicious = "https://internal.test/x Authorization: Bearer secret\nTraceback <img src=x>"
    error = api_client.APIClientError(
        malicious, status_code=429, error_code="UNTRUSTED_CODE",
        request_id="bad request/<script>", retry_after=7,
    )
    assert error.status_code == 429
    assert error.error_code == "UNTRUSTED_CODE"
    assert error.retry_after == 7
    assert error.request_id is None
    assert "7 秒" in str(error)
    assert malicious not in str(error)
    assert "<script>" not in str(error)
    assert "req-7" in str(api_client.APIClientError(malicious, request_id="req-7"))
    assert view.safe_request_id("x" * 100) is None
    assert "操作未能完成" in view.safe_error_message()
    assert "服务暂不可用" in view.safe_error_message(status_code=503)


def test_model_context_policy_errors_preserve_only_safe_reference_status():
    expected = {
        "disabled_by_user": "本次未允许课程片段用于模型回答。",
        "blocked_by_policy": "当前课程未获许可，不能将课程片段交给外部模型。",
        "revoked": "课程资料授权已失效，请刷新后重试。",
    }
    for status, message in expected.items():
        error = api_client.APIClientError(
            "untrusted detail Authorization: secret",
            status_code=403,
            error_code="MODEL_CONTEXT_NOT_AUTHORIZED",
            course_reference_status=status,
        )
        assert error.course_reference_status == status
        assert str(error) == message
        assert "secret" not in str(error)

    contradictory = api_client.APIClientError(
        "must not leak",
        status_code=403,
        error_code="MODEL_CONTEXT_NOT_AUTHORIZED",
        course_reference_status="allowed",
    )
    assert contradictory.course_reference_status == "allowed"
    assert str(contradictory) == "当前账号无权执行此操作。"

    forged = api_client.APIClientError(
        "private raw detail",
        status_code=403,
        error_code="MODEL_CONTEXT_NOT_AUTHORIZED",
        course_reference_status="client_forged_status",
    )
    assert forged.course_reference_status is None
    assert str(forged) == "当前账号无权执行此操作。"
    assert "private" not in str(forged)


def test_http_client_extracts_policy_status_without_exposing_detail(monkeypatch):
    response = requests.Response()
    response.status_code = 403
    response.url = "http://internal.test/private"
    response._content = (
        b'{"error_code":"MODEL_CONTEXT_NOT_AUTHORIZED",'
        b'"course_reference_status":"disabled_by_user",'
        b'"message":"Authorization: Bearer secret http://internal.test/private"}'
    )
    monkeypatch.setattr(api_client.requests, "request", lambda *_args, **_kwargs: response)

    try:
        api_client.health()
        assert False, "expected APIClientError"
    except api_client.APIClientError as error:
        assert error.status_code == 403
        assert error.error_code == "MODEL_CONTEXT_NOT_AUTHORIZED"
        assert error.course_reference_status == "disabled_by_user"
        assert str(error) == "本次未允许课程片段用于模型回答。"
        assert "Authorization" not in str(error)
        assert "internal.test" not in str(error)


def test_client_http_and_network_errors_never_echo_body_or_url(monkeypatch):
    response = requests.Response()
    response.status_code = 429
    response.url = "http://internal.test/private"
    response._content = b'{"error_code":"MODEL_BUSY","message":"Authorization: secret",' \
                        b'"request_id":"req-42"}'
    response.headers["Retry-After"] = "3"
    monkeypatch.setattr(api_client.requests, "request", lambda *_args, **_kwargs: response)
    try:
        api_client.health()
        assert False, "expected APIClientError"
    except api_client.APIClientError as error:
        assert error.status_code == 429
        assert error.retry_after == 3
        assert error.request_id == "req-42"
        assert "secret" not in str(error)
        assert "internal.test" not in str(error)

    response.status_code = 503
    response._content = b"Traceback Authorization: secret"
    try:
        api_client.health()
        assert False, "expected APIClientError"
    except api_client.APIClientError as error:
        assert "Traceback" not in str(error)
        assert "secret" not in str(error)

    response.status_code = 200
    response._content = b"not JSON; Authorization secret"
    try:
        api_client.health()
        assert False, "expected APIClientError"
    except api_client.APIClientError as error:
        assert "Authorization" not in str(error)
        assert "secret" not in str(error)

    def fail_request(*_args, **_kwargs):
        raise requests.ConnectionError("http://internal.test Authorization secret")

    monkeypatch.setattr(api_client.requests, "request", fail_request)
    try:
        api_client.health()
        assert False, "expected APIClientError"
    except api_client.APIClientError as error:
        assert "internal.test" not in str(error)
        assert "secret" not in str(error)


def test_app_navigation_does_not_restore_admin_tools_for_student(monkeypatch):
    st.session_state.clear()
    st.session_state.auth_token = "student-token"
    st.session_state.current_user = {"role": "student"}
    st.session_state.app_navigation_page = "tools"
    chosen = []
    monkeypatch.setattr(app, "_render_login", lambda: None)
    monkeypatch.setattr(app.st.sidebar, "title", lambda *_args, **_kwargs: None)

    def radio(_label, options, **kwargs):
        chosen.append((list(options), kwargs["format_func"]("student")))
        return st.session_state.app_navigation_page

    monkeypatch.setattr(app.st.sidebar, "radio", radio)
    monkeypatch.setattr(app.student, "render", lambda: None)
    monkeypatch.setattr(app, "_track_page_view", lambda page: chosen.append(page))
    monkeypatch.setattr(app, "render_tools", lambda: (_ for _ in ()).throw(AssertionError("tools opened")))
    app.main()
    assert chosen == [(["student", "innovation", "about"], "课程学习"), "student"]
    assert st.session_state.app_navigation_page == "student"
    app._clear_auth_session()
    assert "app_navigation_page" not in st.session_state


def test_account_name_is_plain_text_and_tools_never_render_payload(monkeypatch):
    st.session_state.clear()
    st.session_state.auth_token = "token"
    st.session_state.current_user = {"role": "admin", "display_name": "![x](https://evil.test) <b>Admin</b>"}
    plain = []
    captions = []
    frames = []
    monkeypatch.setattr(app.st.sidebar, "subheader", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st.sidebar, "text", lambda value: plain.append(value))
    monkeypatch.setattr(app.st.sidebar, "caption", lambda value: captions.append(value))
    monkeypatch.setattr(app.st.sidebar, "button", lambda *_args, **_kwargs: False)
    app._render_login()
    assert plain == ["![x](https://evil.test) <b>Admin</b>"]
    assert captions == ["管理员"]

    secret = "Authorization Bearer secret Traceback C:/private"
    monkeypatch.setattr(app.api_client, "system_status", lambda: {
        "provider": "template", "sqlite": {"status": "ok", "detail": secret},
        "chroma": {"status": "error", "detail": secret},
        "mcp": {"status": "fallback", "detail": secret},
    })
    monkeypatch.setattr(app.api_client, "get_tools", lambda: [
        {"name": "course_retriever", "enabled": True, "description": secret},
    ])
    monkeypatch.setattr(app.st, "header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st, "metric", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st, "dataframe", lambda data, **_kwargs: frames.append(data))
    app.render_tools()
    assert len(frames) == 2
    assert secret not in " ".join(frame.to_string() for frame in frames)


def test_single_course_or_class_name_is_plain_text(monkeypatch):
    st.session_state.clear()
    plain = []
    monkeypatch.setattr(session_scope.st, "text", lambda value: plain.append(value))
    monkeypatch.setattr(
        session_scope.st, "caption",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unsafe caption")),
    )
    hostile = "<img src=x> ![tracking](https://evil.test) " + "a" * 300
    assert session_scope._explicit_choice(
        "当前课程", {4: {"id": 4, "name": hostile}},
        state_key="selected_course_id", widget_key="course_selector",
        render_label=lambda item: item["name"],
    ) == 4
    assert plain == [f"当前课程：{hostile}"]


def test_logout_clears_navigation_and_ends_learning_session(monkeypatch):
    st.session_state.clear()
    st.session_state.auth_token = "token"
    st.session_state.current_user = {"role": "admin", "display_name": "管理员"}
    st.session_state.client_session_id = "session-1"
    st.session_state.app_navigation_page = "tools"
    calls = []
    monkeypatch.setattr(app.st.sidebar, "subheader", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st.sidebar, "text", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st.sidebar, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st.sidebar, "button", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(app.api_client, "end_learning_session", lambda token, session: calls.append(("end", token, session)))
    monkeypatch.setattr(app.api_client, "logout", lambda token: calls.append(("logout", token)))
    monkeypatch.setattr(app.st, "rerun", lambda: None)
    app._render_login()
    assert calls == [("end", "token", "session-1"), ("logout", "token")]
    assert "app_navigation_page" not in st.session_state
    assert "auth_token" not in st.session_state


def test_teacher_and_admin_navigation_keeps_internal_page_view_names(monkeypatch):
    for role in ("teacher", "admin"):
        st.session_state.clear()
        st.session_state.auth_token = "token"
        st.session_state.current_user = {"role": role}
        st.session_state.app_navigation_page = "tools"
        seen = []
        monkeypatch.setattr(app, "_render_login", lambda: None)
        monkeypatch.setattr(app.st.sidebar, "title", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            app.st.sidebar, "radio",
            lambda _label, pages, **kwargs: seen.append((list(pages), kwargs["format_func"]("tools"))) or "tools",
        )
        monkeypatch.setattr(app, "render_tools", lambda: seen.append("rendered"))
        monkeypatch.setattr(app, "_track_page_view", lambda page: seen.append(page))
        app.main()
        assert seen == [(view.navigation_pages(role), "运行状态"), "tools", "rendered"]


def test_prelogin_description_and_password_form_are_chinese(monkeypatch):
    st.session_state.clear()
    seen = []
    monkeypatch.setattr(app, "_render_login", lambda: None)
    monkeypatch.setattr(app.st, "header", lambda text: seen.append(text))
    monkeypatch.setattr(app.st, "text", lambda text: seen.append(text))
    monkeypatch.setattr(app.st, "info", lambda text: seen.append(text))
    app.main()
    assert seen == [view.PRODUCT_NAME, view.PRODUCT_SUBTITLE, "请先在侧边栏登录。"]

    class Form:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(app.st.sidebar, "warning", lambda text: seen.append(text))
    monkeypatch.setattr(app.st.sidebar, "form", lambda *_args: Form())
    monkeypatch.setattr(app.st, "text_input", lambda label, **_kwargs: seen.append(label) or "")
    monkeypatch.setattr(app.st, "form_submit_button", lambda label, **_kwargs: seen.append(label) or False)
    app._render_change_password_form()
    assert seen[-5:] == [
        "首次登录请修改密码，完成后即可使用课程功能。",
        "当前密码", "新密码", "确认新密码", "修改密码",
    ]


def test_unknown_tool_failure_projects_safe_error(monkeypatch):
    errors = []
    monkeypatch.setattr(app.st, "header", lambda *_args: None)
    monkeypatch.setattr(app.st, "error", lambda value: errors.append(value))
    monkeypatch.setattr(
        app.api_client, "system_status",
        lambda: (_ for _ in ()).throw(RuntimeError("Traceback: Authorization secret https://internal.test")),
    )
    app.render_tools()
    assert errors == [view.safe_error_message()]


def test_unknown_page_failure_never_renders_exception_text(monkeypatch):
    st.session_state.clear()
    st.session_state.auth_token = "token"
    st.session_state.current_user = {"role": "student"}
    errors = []
    monkeypatch.setattr(app, "_render_login", lambda: None)
    monkeypatch.setattr(app.st.sidebar, "title", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st.sidebar, "radio", lambda *_args, **_kwargs: "student")
    monkeypatch.setattr(
        app.student, "render",
        lambda: (_ for _ in ()).throw(RuntimeError("Authorization secret https://internal.test Traceback")),
    )
    monkeypatch.setattr(app.st, "error", lambda value: errors.append(value))
    app.main()
    assert errors == [view.safe_error_message()]


def _hotspot_summary():
    return {
        "clusters": {"items": [
            {"id": 11, "label": "同名", "approved": False, "is_noise": False,
             "representative_question": "问题一", "members": [{"question_id": 101, "question": "问题一", "created_at": "2026-10-01", "is_representative": True}]},
            {"id": 12, "label": "同名", "approved": True, "is_noise": False,
             "representative_question": "问题二", "members": [
                 {"question_id": 102, "question": "问题二", "created_at": "2026-10-02"},
                 {"question_id": 103, "question": "问题三", "created_at": "2026-10-03"},
             ]},
            {"id": 13, "label": "待核查", "approved": True, "is_noise": False,
             "representative_question": "问题四", "members": [{"question_id": 104, "question": "问题四"}]},
            {"id": 14, "label": "噪声", "approved": False, "is_noise": True,
             "members": [{"question_id": 105}, {"question_id": 106}]},
            {"id": 15, "label": "空簇", "approved": True, "is_noise": False, "members": []},
        ]},
        "faq": {"needs_review_count": 0, "items": [
            {"cluster_id": 12, "label": "同名", "representative_question": "问题二", "count": 2,
             "answer": "回答", "citations": [{"source_id": "S1", "snippet": "原文"}]},
        ]},
    }


def test_hotspot_view_uses_members_and_valid_faq_not_old_review_count():
    summary = _hotspot_summary()
    untouched = deepcopy(summary)
    result = view.build_hotspot_view_model(summary)
    assert result.status == "ready"
    assert (result.topic_count, result.pending_count, result.approved_count, result.exportable_count) == (3, 1, 2, 1)
    assert result.noise_question_count == 2
    assert [topic.cluster_id for topic in result.topics] == [12, 11, 13]
    assert result.topics[0].recent_question_at == "2026-10-03"
    assert [topic.cluster_id for topic in result.topics if topic.label == "同名"] == [12, 11]
    assert summary == untouched


def test_hotspot_inconsistent_faq_never_claims_exportable_or_zero():
    for bad_faq in (
        {"cluster_id": 11, "answer": "越权", "citations": [{"source_id": "S1"}]},
        {"cluster_id": 12, "answer": "无引用", "citations": []},
        {"cluster_id": 999, "answer": "无簇", "citations": [{"source_id": "S1"}]},
        {"answer": "无编号", "citations": [{"source_id": "S1"}]},
    ):
        summary = _hotspot_summary()
        summary["faq"]["items"] = [bad_faq]
        result = view.build_hotspot_view_model(summary)
        assert result.status == "inconsistent"
        assert result.exportable_count is None
        assert result.message == "资料状态暂不同步，请刷新"
    assert view.build_hotspot_view_model({}).topic_count is None


def test_hotspot_duplicate_faq_identical_dedupes_conflict_warns_then_refresh_recovers():
    summary = _hotspot_summary()
    original = deepcopy(summary["faq"]["items"][0])
    summary["faq"]["items"].append(deepcopy(original))
    assert view.build_hotspot_view_model(summary).exportable_count == 1
    summary["faq"]["items"][1]["answer"] = "冲突回答"
    assert view.build_hotspot_view_model(summary).status == "inconsistent"
    summary["faq"]["items"] = [original]
    assert view.build_hotspot_view_model(summary).status == "ready"


def test_hotspot_faq_identity_and_citation_mismatch_is_not_exportable():
    for change in (
        {"count": 99},
        {"representative_question": "另一问题"},
        {"label": "另一主题"},
        {"citations": [{"source_id": "S1"}]},
        {"citations": [None]},
        {"citations": [{"source_id": "S1", "snippet": "甲"}, {"source_id": "S1", "snippet": "乙"}]},
    ):
        summary = _hotspot_summary()
        summary["faq"]["items"][0].update(change)
        result = view.build_hotspot_view_model(summary)
        assert result.status == "inconsistent"
        assert result.exportable_count is None


def test_hotspot_empty_and_unavailable_are_distinct():
    empty = view.build_hotspot_view_model({"clusters": {"items": []}, "faq": {"items": []}})
    assert empty.status == "ready"
    assert (empty.topic_count, empty.pending_count, empty.approved_count,
            empty.exportable_count, empty.noise_question_count) == (0, 0, 0, 0, 0)
    unavailable = view.build_hotspot_view_model(None)
    assert unavailable.status == "unavailable"
    assert unavailable.exportable_count is None
