from __future__ import annotations

import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from frontend import app as frontend_app
from frontend import session_scope
from frontend.api_client import APIClientError
from frontend.pages import student

STUDENT_PAGE = Path(__file__).resolve().parents[1] / "frontend" / "pages" / "student.py"


class _MetricColumn:
    def metric(self, *_args, **_kwargs):
        return None

    def caption(self, *_args, **_kwargs):
        return None


class _Expander:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def test_render_citations_uses_server_source_id_without_positional_rebinding(monkeypatch):
    labels: list[str] = []
    monkeypatch.setattr(
        student.st,
        "expander",
        lambda label, **_kwargs: labels.append(str(label)) or _Expander(),
    )
    monkeypatch.setattr(student.st, "subheader", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student.st, "write", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student.st, "link_button", lambda *_args, **_kwargs: None)

    student._render_citations(
        [
            {
                "source_id": "S2",
                "title": "Uploaded evidence",
                "snippet": "exact uploaded sentence",
                "document_id": 4,
                "chunk_id": 9,
            }
        ]
    )

    assert labels == ["引用 [S2]"]


def test_standard_resource_connector_selects_remote_rfc_and_local_3gpp():
    assert student._resource_connector_name("standard", "RFC 9000", rfc_enabled=True) == "rfc_editor"
    assert student._resource_connector_name("standard", "RFC 9000", rfc_enabled=False) == "standard_source"
    assert student._resource_connector_name("standard", "3GPP TS 24.501", rfc_enabled=True) == "standard_source"


def test_general_learning_form_submits_once_and_shows_reference_identity(monkeypatch):
    st.session_state.clear()
    st.session_state.auth_token = "token"
    st.session_state.client_session_id = "session-general"
    calls: list[dict] = []
    messages: list[str] = []
    monkeypatch.setattr(student.st, "subheader", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student.st, "caption", lambda message, **_kwargs: messages.append(str(message)))
    monkeypatch.setattr(student.st, "form", lambda *_args, **_kwargs: _Expander())
    monkeypatch.setattr(student.st, "text_area", lambda *_args, **_kwargs: "解释欧姆定律")
    monkeypatch.setattr(student.st, "form_submit_button", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(student.st, "spinner", lambda *_args, **_kwargs: _Expander())
    monkeypatch.setattr(student.st, "info", lambda message, **_kwargs: messages.append(str(message)))
    monkeypatch.setattr(student.st, "text", lambda message, **_kwargs: messages.append(str(message)))
    monkeypatch.setattr(student.st, "write", lambda message, **_kwargs: messages.append(str(message)))
    monkeypatch.setattr(student.st, "warning", lambda message, **_kwargs: messages.append(str(message)))
    monkeypatch.setattr(student.st, "error", lambda message, **_kwargs: messages.append(str(message)))
    monkeypatch.setattr(student.st, "expander", lambda *_args, **_kwargs: _Expander())
    monkeypatch.setattr(
        student.api_client,
        "general_chat",
        lambda token, question, **kwargs: calls.append(
            {"token": token, "question": question, **kwargs}
        )
        or {
            "status": "answered",
            "answer_kind": "model_generated",
            "provider": "deepseek",
            "answer": "电压等于电流与电阻的乘积。",
            "model": "deepseek-test",
            "reference_status": "used",
            "reference_materials": [
                {"title": "电路教材", "excerpt": "U=IR", "usage": "context_only"}
            ],
            "warnings": [],
        },
    )

    student._render_general_chat(3, 4, True)

    assert calls == [
        {
            "token": "token",
            "question": "解释欧姆定律",
            "course_id": 3,
            "class_id": 4,
            "use_course_references": True,
            "client_session_id": "session-general",
        }
    ]
    assert any("已使用当前课程参考" in item for item in messages)
    assert any("外部模型生成" in item for item in messages)
    assert any("电压等于" in item for item in messages)


def test_remote_resource_payload_keeps_lookup_status_and_only_safe_links():
    payload = student._normalize_remote_resource_payload(
        {
            "connector": "rfc_editor",
            "status": "PARTIAL",
            "lookup_state": "PARTIAL",
            "source_attempts": [{"source_name": "rfc_editor", "status": "PARTIAL"}],
            "incomplete_results": True,
            "resources": [
                {
                    "id": 1,
                    "external_id": "RFC-9000",
                    "resource_type": "standard",
                    "source_name": "rfc_editor",
                    "title": "QUIC",
                    "url": "javascript:alert(1)",
                    "provenance_verified": True,
                }
            ],
        }
    )

    assert payload["lookup_state"] == "PARTIAL"
    assert payload["source_attempts"][0]["source_name"] == "rfc_editor"
    assert payload["incomplete_results"] is True
    assert payload["items"][0]["url"] == ""


def test_render_citations_legacy_missing_source_id_uses_positional_label(monkeypatch):
    labels: list[str] = []
    monkeypatch.setattr(
        student.st,
        "expander",
        lambda label, **_kwargs: labels.append(str(label)) or _Expander(),
    )
    monkeypatch.setattr(student.st, "subheader", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student.st, "write", lambda *_args, **_kwargs: None)

    student._render_citations([{"title": "Legacy evidence", "snippet": "legacy"}])

    assert labels == ["引用 1"]


def test_student_unknown_resource_state_is_not_claimed_verified():
    assert "待核实" in student._resource_status_label({"lookup_state": "UNEXPECTED"})
    assert "待核实" in student._resource_delivery_label("unexpected")
    assert student._safe_model_name("deepseek-flash") == "deepseek-flash"
    assert student._safe_model_name("http://internal.test/model") == "身份待核实"


def test_remote_resource_fingerprint_is_scoped_to_class():
    fields = dict(keyword="RFC 9000", topic="", resource_type="standard", source="",
                  year_from=None, year_to=None, page=1, page_size=10, course_id=7)
    first = student._resource_query_fingerprint(**fields, class_id=11)
    second = student._resource_query_fingerprint(**fields, class_id=12)
    assert first != second


def test_student_answer_and_citation_dynamic_text_use_plain_components(monkeypatch):
    plain: list[str] = []
    labels: list[str] = []
    monkeypatch.setattr(student.st, "columns", lambda *_args, **_kwargs: [_MetricColumn() for _ in range(3)])
    monkeypatch.setattr(student.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student.st, "text", lambda value: plain.append(str(value)))
    monkeypatch.setattr(student.st, "markdown", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dynamic markdown")))
    monkeypatch.setattr(student.st, "expander", lambda label, **_kwargs: labels.append(str(label)) or _Expander())
    monkeypatch.setattr(student, "_render_selected_tool", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student, "_render_tool_calls", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student, "_render_feedback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student, "_render_agent_trace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student.st, "subheader", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student.st, "link_button", lambda *_args, **_kwargs: None)
    answer = "![pixel](https://evil.test/pixel) <img src=x>"
    student._render_answer({"run_status": "COMPLETED", "answer": answer,
                            "citations": [{"source_id": "S2", "title": answer, "snippet": answer,
                                           "document_id": 17, "chunk_id": 23}]})
    assert plain.count(answer) >= 2
    assert labels == ["引用 [S2]"]
    assert "document_id" not in "\n".join(plain)


def test_student_resource_card_keeps_literal_excerpt_and_safe_link_only(monkeypatch):
    labels: list[str] = []
    plain: list[str] = []
    links: list[str] = []
    opened: list[tuple] = []
    monkeypatch.setattr(student.st, "expander", lambda label, **_kwargs: labels.append(str(label)) or _Expander())
    monkeypatch.setattr(student.st, "text", lambda value: plain.append(str(value)))
    monkeypatch.setattr(student.st, "link_button", lambda _label, url, **_kwargs: links.append(url))
    monkeypatch.setattr(student.st, "button", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(student.st, "write", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dynamic write")))
    monkeypatch.setattr(student.st, "caption", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dynamic caption")))
    monkeypatch.setattr(student.api_client, "open_resource", lambda *args, **kwargs: opened.append((args, kwargs)))
    monkeypatch.setattr(student.st, "success", lambda *_args: None)
    hostile = "![pixel](https://evil.test/pixel) <img src=x>\n  原文"
    item = {"resource_id": "external:42", "type": "paper", "title": hostile,
            "summary": hostile, "source_name": "arxiv", "delivery_mode": "fresh_cache",
            "url": "javascript:alert(1)", "tags": [hostile]}

    student._render_resource_item(item, index=1, token="token", course_id=8, class_id=9)

    assert labels == ["资料 1"]
    assert any(hostile in value for value in plain)
    assert "external:42" not in "\n".join(plain)
    assert links == []
    assert opened[0][0][1] == "external:42"
    assert opened[0][1]["course_id"] == 8 and opened[0][1]["class_id"] == 9
    item["url"] = "https://arxiv.org/abs/2601.00001"
    student._render_resource_item(item, index=2, token="token", course_id=8, class_id=9)
    assert links == ["https://arxiv.org/abs/2601.00001"]


def test_student_backend_status_does_not_show_startup_command_or_component_enums(monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(student.api_client, "system_status", lambda: {
        "provider": "template", "fallback_active": True,
        "sqlite": {"status": "error", "detail": "C:/secret/app.db"},
    })
    monkeypatch.setattr(student.st, "warning", lambda value: shown.append(str(value)))
    monkeypatch.setattr(student.st, "info", lambda value: shown.append(str(value)))
    student._render_backend_status()
    assert "本地模板" in "\n".join(shown)
    assert "C:/secret" not in "\n".join(shown)
    assert "SQLite" not in "\n".join(shown)


def test_experiment_task_uses_stable_ids_and_plain_user_text(monkeypatch):
    st.session_state.clear()
    st.session_state.auth_token = "token"
    hostile = "<img src=x> ![pixel](https://evil.test/pixel)"
    plain: list[str] = []
    labels: list[str] = []
    monkeypatch.setattr(student.st, "subheader", lambda *_args: None)
    monkeypatch.setattr(student.st, "text", lambda value: plain.append(str(value)))
    monkeypatch.setattr(student.st, "caption", lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic caption")))
    monkeypatch.setattr(student.st, "write", lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic write")))
    monkeypatch.setattr(student.st, "form", lambda *_args, **_kwargs: _Expander())
    monkeypatch.setattr(student.st, "slider", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(student.st, "text_area", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(student.st, "form_submit_button", lambda *_args, **_kwargs: False)
    def selectbox(_label, options, **kwargs):
        if kwargs.get("format_func"):
            labels.append(kwargs["format_func"](options[0]))
        return options[0]
    monkeypatch.setattr(student.st, "selectbox", selectbox)
    monkeypatch.setattr(student.api_client, "assignments", lambda *_args, **_kwargs: {
        "assignments": [{"id": 37, "title": hostile, "description": hostile,
                         "steps": [{"id": 41, "step_index": 1, "title": hostile,
                                    "description": hostile}]}],
    })
    student._render_assignments(7, 8)
    assert labels == [hostile, hostile]
    assert plain.count(hostile) == 2
    assert st.session_state.assignment_context["assignment_id"] == 37
    assert st.session_state.assignment_context["step_id"] == 41


def test_portal_remote_click_once_and_class_change_forgets_old_result(monkeypatch):
    st.session_state.clear()
    st.session_state.auth_token = "token"
    searches: list[dict] = []
    clicks = iter([True, False])
    class Column:
        def text_input(self, *_args, **_kwargs):
            return "RFC 9000"
        def selectbox(self, *_args, **_kwargs):
            return "全部"
        def checkbox(self, *_args, **_kwargs):
            return False
        def button(self, *_args, **_kwargs):
            return False
        def caption(self, *_args, **_kwargs):
            return None
    monkeypatch.setattr(student.st, "columns", lambda count: [Column() for _ in range(6 if isinstance(count, list) else count)])
    monkeypatch.setattr(student.st, "subheader", lambda *_args: None)
    monkeypatch.setattr(student.st, "number_input", lambda *_args, **_kwargs: 2020)
    monkeypatch.setattr(student.st, "select_slider", lambda *_args, **_kwargs: 10)
    monkeypatch.setattr(student.st, "button", lambda *_args, **_kwargs: next(clicks))
    monkeypatch.setattr(student.st, "caption", lambda *_args: None)
    monkeypatch.setattr(student.st, "info", lambda *_args: None)
    monkeypatch.setattr(student, "_rfc_editor_enabled", lambda: False)
    monkeypatch.setattr(student.api_client, "search_resources", lambda *_args, **kwargs: searches.append(kwargs) or {
        "lookup_state": "NOT_FOUND", "resources": [], "source_attempts": [],
    })
    monkeypatch.setattr(student.api_client, "resource_portal", lambda *_args, **_kwargs: {
        "items": [], "total": 0, "lookup_state": "NOT_FOUND",
    })
    student._render_resource_portal(7, 8)
    assert len(searches) == 1
    assert st.session_state.get("resource_search_payload") is not None
    student._render_resource_portal(7, 9)
    assert len(searches) == 1
    assert "resource_search_payload" not in st.session_state


def test_portal_committed_keyword_rerun_clears_old_remote_result_without_new_search(monkeypatch):
    st.session_state.clear()
    st.session_state.auth_token = "token"
    current = {"keyword": "alpha"}
    remote_clicks = iter([True, False])
    searches: list[dict] = []
    rendered: list[str] = []

    class Column:
        def text_input(self, label, *_args, **_kwargs):
            return current["keyword"] if label == "关键词" else ""

        def selectbox(self, *_args, **_kwargs):
            return "论文"

        def checkbox(self, *_args, **_kwargs):
            return False

        def button(self, *_args, **_kwargs):
            return False

        def caption(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(student.st, "columns", lambda count: [Column() for _ in range(6 if isinstance(count, list) else count)])
    monkeypatch.setattr(student.st, "subheader", lambda *_args: None)
    monkeypatch.setattr(student.st, "number_input", lambda *_args, **_kwargs: 2020)
    monkeypatch.setattr(student.st, "select_slider", lambda *_args, **_kwargs: 10)
    monkeypatch.setattr(student.st, "button", lambda *_args, **_kwargs: next(remote_clicks))
    monkeypatch.setattr(student.st, "caption", lambda *_args: None)
    monkeypatch.setattr(student.st, "info", lambda *_args: None)
    monkeypatch.setattr(student, "_rfc_editor_enabled", lambda: False)
    monkeypatch.setattr(
        student.api_client,
        "search_resources",
        lambda *_args, **kwargs: searches.append(kwargs)
        or {
            "lookup_state": "FOUND",
            "resources": [
                {
                    "id": 1,
                    "external_id": "alpha-id",
                    "resource_type": "paper",
                    "source_name": "arxiv",
                    "title": "A-REMOTE-MARKER",
                    "abstract": "alpha",
                    "provenance_verified": True,
                }
            ],
        },
    )
    monkeypatch.setattr(
        student.api_client,
        "resource_portal",
        lambda *_args, **kwargs: {
            "items": ([{"resource_id": 2, "title": "B-PORTAL-MARKER"}] if kwargs["keyword"] == "beta" else []),
            "total": 1 if kwargs["keyword"] == "beta" else 0,
            "lookup_state": "FOUND",
        },
    )
    monkeypatch.setattr(
        student,
        "_render_resource_item",
        lambda item, **_kwargs: rendered.append(str(item.get("title"))),
    )

    student._render_resource_portal(7, 8)
    first_render_end = len(rendered)
    assert rendered == ["A-REMOTE-MARKER"]
    assert len(searches) == 1

    current["keyword"] = "beta"  # Simulates a committed text_input value on a new rerun.
    student._render_resource_portal(7, 8)

    assert len(searches) == 1
    assert rendered[first_render_end:] == ["B-PORTAL-MARKER"]
    assert "A-REMOTE-MARKER" not in rendered[first_render_end:]
    assert "resource_search_payload" not in st.session_state
    assert "resource_search_fingerprint" not in st.session_state


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("topic", "new topic"),
        ("resource_type", "standard"),
        ("source", "rfc_editor"),
        ("year_from", 2021),
        ("year_to", 2025),
        ("page_size", 20),
        ("course_id", 8),
        ("class_id", 9),
    ],
)
def test_remote_resource_fingerprint_invalidates_every_committed_scope_field(field, changed):
    base = dict(
        keyword="Alpha", topic="topic", resource_type="paper", source="arxiv",
        year_from=2020, year_to=2026, page=1, page_size=10, course_id=7, class_id=8,
    )
    modified = {**base, field: changed}
    assert student._resource_query_fingerprint(**base) != student._resource_query_fingerprint(**modified)


def test_remote_resource_fingerprint_normalizes_case_and_whitespace():
    base = dict(
        keyword=" Alpha  Query ", topic=" Topic ", resource_type=" PAPER ", source=" ArXiv ",
        year_from=2020, year_to=2026, page=1, page_size=10, course_id=7, class_id=8,
    )
    equivalent = {**base, "keyword": "alpha query", "topic": "topic", "resource_type": "paper", "source": "arxiv"}
    assert student._resource_query_fingerprint(**base) == student._resource_query_fingerprint(**equivalent)


def _capture_tool_call_summary(monkeypatch, item: dict) -> str:
    captured: list[str] = []
    monkeypatch.setattr(student.st, "expander", lambda *_args, **_kwargs: _Expander())
    monkeypatch.setattr(student.st, "text", lambda value: captured.append(str(value)))
    monkeypatch.setattr(student.st, "write", lambda *_args: (_ for _ in ()).throw(AssertionError("raw tool result")))

    student._render_tool_calls([item])

    assert len(captured) == 1
    return captured[0]


def test_anonymous_main_short_circuits_before_business_pages(monkeypatch):
    st.session_state.clear()
    messages: list[str] = []
    monkeypatch.setattr(frontend_app, "_render_login", lambda: None)
    monkeypatch.setattr(frontend_app.st, "info", lambda message: messages.append(str(message)))
    monkeypatch.setattr(
        frontend_app.st.sidebar,
        "title",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("navigation must not render")),
    )
    monkeypatch.setattr(
        frontend_app.student,
        "render",
        lambda: (_ for _ in ()).throw(AssertionError("student page must not render")),
    )

    frontend_app.main()

    assert messages
    assert "登录" in messages[0]


def test_must_change_password_blocks_business_pages(monkeypatch):
    st.session_state.clear()
    st.session_state.auth_token = "token"
    st.session_state.current_user = {"role": "student", "must_change_password": True}
    warnings: list[str] = []
    monkeypatch.setattr(frontend_app, "_render_login", lambda: None)
    monkeypatch.setattr(frontend_app.st, "warning", lambda message: warnings.append(str(message)))
    monkeypatch.setattr(
        frontend_app.st.sidebar,
        "title",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("navigation must not render")),
    )

    frontend_app.main()

    assert warnings


def test_clear_auth_session_removes_all_user_scoped_state():
    st.session_state.clear()
    keys = {
        "auth_token": "token",
        "current_user": {"role": "student"},
        "client_session_id": "session",
        "last_chat_result": {"run_id": "old"},
        "last_guide_result": {"run_id": "guide"},
        "chat_result_history": [{"run_id": "old"}],
        "resource_search_payload": {"items": []},
        "resource_search_fingerprint": ("old",),
        "resource_page": 3,
        "assignment_context": {"assignment_id": 1},
        "innovation_session": {"id": 1},
        "innovation_session_scope": (1, 2, 3),
        "innovation_resume_selection": 1,
        "innovation_suggestion_result": {"status": "proposed"},
        "selected_course_id": 7,
        "teacher_knowledge_course_selector": "course",
        "rename_label_12": "old label",
    }
    for key, value in keys.items():
        st.session_state[key] = value

    frontend_app._clear_auth_session()

    for key in keys:
        assert key not in st.session_state


def test_switching_accounts_clears_previous_user_visible_results():
    st.session_state.clear()
    st.session_state.auth_token = "student-a-token"
    st.session_state.current_user = {"username": "student_a"}
    st.session_state.last_chat_result = {"answer": "student A answer"}
    st.session_state.chat_result_history = [{"run_id": "a-run", "answer": "student A answer"}]
    st.session_state.resource_search_payload = {"items": [{"title": "student A resource"}]}

    frontend_app._clear_auth_session()
    st.session_state.auth_token = "student-b-token"
    st.session_state.current_user = {"username": "student_b"}

    assert "last_chat_result" not in st.session_state
    assert "chat_result_history" not in st.session_state
    assert "resource_search_payload" not in st.session_state
    assert st.session_state.current_user["username"] == "student_b"


def test_student_deep_link_requires_login(monkeypatch):
    st.session_state.clear()
    messages: list[str] = []
    monkeypatch.setattr(student.st, "info", lambda message: messages.append(str(message)))
    monkeypatch.setattr(
        student,
        "_render_header",
        lambda: (_ for _ in ()).throw(AssertionError("student body must not render")),
    )

    student.render()

    assert messages
    assert "登录" in messages[0]


def test_student_deep_link_script_requires_login_with_apptest(monkeypatch):
    monkeypatch.setattr(
        student.api_client,
        "system_status",
        lambda: (_ for _ in ()).throw(AssertionError("anonymous page must not call business API")),
    )

    at = AppTest.from_file(STUDENT_PAGE).run(timeout=10)

    assert len(at.info) == 1
    assert at.info[0].value
    assert len(at.exception) == 0


def test_student_deep_link_blocks_must_change_password(monkeypatch):
    st.session_state.clear()
    st.session_state.auth_token = "token"
    st.session_state.current_user = {"must_change_password": True}
    warnings: list[str] = []
    monkeypatch.setattr(student.st, "warning", lambda message: warnings.append(str(message)))
    monkeypatch.setattr(
        student,
        "_render_header",
        lambda: (_ for _ in ()).throw(AssertionError("student body must not render")),
    )

    student.render()

    assert warnings


def test_student_deep_link_script_blocks_must_change_password_with_apptest(monkeypatch):
    monkeypatch.setattr(
        student.api_client,
        "system_status",
        lambda: (_ for _ in ()).throw(AssertionError("password gate must not call business API")),
    )
    at = AppTest.from_file(STUDENT_PAGE)
    at.session_state["auth_token"] = "token"
    at.session_state["current_user"] = {"must_change_password": True}

    at.run(timeout=10)

    assert len(at.warning) == 1
    assert len(at.exception) == 0


def test_student_deep_link_script_renders_logged_in_page_with_apptest(monkeypatch):
    created: list[tuple[str, str, int, int | None]] = []
    monkeypatch.setattr(student.api_client, "system_status", lambda: {"fallback_active": False, "model": "test-model"})
    monkeypatch.setattr(student.api_client, "courses", lambda token: [{"id": 1, "name": "Demo", "slug": "demo"}])
    monkeypatch.setattr(student.api_client, "classes", lambda token, course_id: [{"id": 2, "course_id": 1, "name": "A"}])
    monkeypatch.setattr(
        student.api_client,
        "create_learning_session",
        lambda token, session_id, course_id, class_id=None: created.append(
            (token, session_id, course_id, class_id)
        ) or {"client_session_id": session_id},
    )
    monkeypatch.setattr(student.api_client, "resource_portal", lambda *args, **kwargs: {"items": [], "total": 0})
    monkeypatch.setattr(student.api_client, "assignments", lambda *args, **kwargs: {"assignments": []})
    at = AppTest.from_file(STUDENT_PAGE)
    at.session_state["auth_token"] = "token"
    at.session_state["current_user"] = {"must_change_password": False, "role": "student"}

    at.run(timeout=10)

    assert len(at.title) >= 1
    assert len(at.exception) == 0
    assert created and created[0][0] == "token"
    assert created[0][2:] == (1, 2)


def test_student_multicourse_requires_explicit_selection_before_session(monkeypatch):
    created: list[dict] = []
    monkeypatch.setattr(student.api_client, "system_status", lambda: {"fallback_active": True})
    monkeypatch.setattr(
        student.api_client,
        "courses",
        lambda token: [
            {"id": 1, "name": "Course A", "slug": "a"},
            {"id": 2, "name": "Course B", "slug": "b"},
        ],
    )
    monkeypatch.setattr(
        student.api_client,
        "create_learning_session",
        lambda *args, **kwargs: created.append(kwargs),
    )
    at = AppTest.from_file(STUDENT_PAGE)
    at.session_state["auth_token"] = "token"
    at.session_state["current_user"] = {"must_change_password": False, "role": "student"}

    at.run(timeout=10)

    assert created == []
    assert any("请选择课程" in item.value for item in at.info)
    assert len(at.exception) == 0


def test_student_same_course_multiclass_requires_explicit_class(monkeypatch):
    created: list[dict] = []
    monkeypatch.setattr(student.api_client, "system_status", lambda: {"fallback_active": True})
    monkeypatch.setattr(student.api_client, "courses", lambda token: [{"id": 1, "name": "Course", "slug": "c"}])
    monkeypatch.setattr(
        student.api_client,
        "classes",
        lambda token, course_id: [
            {"id": 10, "course_id": 1, "name": "Class A"},
            {"id": 11, "course_id": 1, "name": "Class B"},
        ],
    )
    monkeypatch.setattr(
        student.api_client,
        "create_learning_session",
        lambda *args, **kwargs: created.append(kwargs),
    )
    at = AppTest.from_file(STUDENT_PAGE)
    at.session_state["auth_token"] = "token"
    at.session_state["current_user"] = {"must_change_password": False, "role": "student"}

    at.run(timeout=10)

    assert created == []
    assert any("明确选择班级" in item.value for item in at.info)
    assert len(at.exception) == 0


def test_student_deep_link_script_requires_login_after_logout_with_apptest():
    at = AppTest.from_file(STUDENT_PAGE)
    at.session_state["auth_token"] = "old-token"
    at.session_state["current_user"] = {"must_change_password": False, "role": "student"}
    del at.session_state["auth_token"]
    del at.session_state["current_user"]

    at.run(timeout=10)

    assert len(at.info) == 1
    assert at.info[0].value


def test_answer_history_is_limited_course_scoped_and_updates_continue():
    st.session_state.clear()
    for index in range(6):
        student._remember_answer_result(
            {
                "run_id": f"run-{index}",
                "answer": f"answer {index}",
                "provider": "deepseek",
                "model": "deepseek-flash",
                "fallback_used": False,
                "delivery_mode": "local_rag",
                "run_status": "COMPLETED",
            },
            course_id=1,
        )
    student._remember_answer_result(
        {
            "run_id": "run-5",
            "answer": "continued answer",
            "provider": "template",
            "model": "local-template",
            "fallback_used": True,
            "delivery_mode": "offline_fallback",
            "run_status": "COMPLETED",
        },
        course_id=1,
    )
    student._remember_answer_result({"run_id": "course-2", "answer": "other course"}, course_id=2)

    course_one = student._answer_history_for_course(1)

    assert len(st.session_state.chat_result_history) == 5
    assert [item["run_id"] for item in course_one] == ["run-2", "run-3", "run-4", "run-5"]
    assert course_one[-1]["answer"] == "continued answer"
    assert student._answer_history_for_course(2)[0]["run_id"] == "course-2"


def test_render_answer_exposes_per_response_execution_status(monkeypatch):
    captions: list[str] = []
    plain: list[str] = []
    monkeypatch.setattr(student.st, "columns", lambda *_args, **_kwargs: [_MetricColumn() for _ in range(3)])
    monkeypatch.setattr(student.st, "caption", lambda message: captions.append(str(message)))
    monkeypatch.setattr(student.st, "text", lambda message: plain.append(str(message)))
    monkeypatch.setattr(student.st, "markdown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student, "_render_selected_tool", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student, "_render_tool_calls", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student, "_render_citations", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student, "_render_feedback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(student, "_render_agent_trace", lambda *_args, **_kwargs: None)

    student._render_answer(
        {
            "run_id": "run-1",
            "trace_id": "trace-1",
            "run_status": "COMPLETED",
            "confidence": 0.8,
            "module": "NAS",
            "answer": "answer",
            "provider": "deepseek",
            "model": "deepseek-flash",
            "fallback_used": False,
            "delivery_mode": "local_rag",
        }
    )

    combined = "\n".join(captions)
    assert "生成方式：外部模型" in combined
    assert "资料：当前课程资料" in combined
    assert "未降级" in combined
    assert "模型：deepseek-flash" in plain
    assert "run-1" not in combined and "trace-1" not in combined


def test_tool_call_summary_reads_structured_result(monkeypatch):
    summary = _capture_tool_call_summary(
        monkeypatch,
        {
            "tool_name": "paper_search_real",
            "success": 0,
            "fallback_used": 0,
            "latency_ms": 12.3,
            "result": {"success": False, "error_code": "TOOL_TIMEOUT", "error_message": "timed out"},
        },
    )

    assert "公开论文查询：未完成" in summary
    assert "已连接资料工具" in summary
    assert "timed out" not in summary


def test_tool_call_summary_reads_result_json_string(monkeypatch):
    summary = _capture_tool_call_summary(
        monkeypatch,
        {
            "tool_name": "paper_search_real",
            "success": 0,
            "fallback_used": 0,
            "latency_ms": 4.2,
            "result_json": json.dumps({"success": False, "error_code": "TOOL_TIMEOUT"}),
        },
    )

    assert "未完成" in summary


def test_tool_call_summary_reads_result_json_mapping(monkeypatch):
    summary = _capture_tool_call_summary(
        monkeypatch,
        {
            "tool_name": "paper_search_real",
            "success": 0,
            "result_json": {"success": False, "error_code": "TOOL_TIMEOUT"},
        },
    )

    assert "未完成" in summary


@pytest.mark.parametrize("raw", [None, "", "{not-json", "[{\"error_code\": \"TOOL_TIMEOUT\"}]"])
def test_tool_call_summary_ignores_invalid_result_json(monkeypatch, raw):
    summary = _capture_tool_call_summary(
        monkeypatch,
        {
            "tool_name": "paper_search_real",
            "success": 0,
            "fallback_used": 0,
            "latency_ms": 1.0,
            "result_json": raw,
        },
    )

    assert "未完成" in summary


def test_tool_call_summary_shows_connector_service_error(monkeypatch):
    summary = _capture_tool_call_summary(
        monkeypatch,
        {
            "tool_name": "paper_search_real",
            "success": 0,
            "result_json": json.dumps(
                {
                    "success": False,
                    "data": {"error_code": "CONNECTOR_SERVICE_ERROR", "delivery_mode": "failed"},
                }
            ),
        },
    )

    assert "未完成" in summary
    assert "暂不可用" in summary


def test_tool_call_summary_preserves_successful_live_result(monkeypatch):
    summary = _capture_tool_call_summary(
        monkeypatch,
        {
            "tool_name": "paper_search_real",
            "success": 1,
            "fallback_used": 0,
            "result_json": json.dumps(
                {
                    "success": True,
                    "data": {"delivery_mode": "live", "fallback_used": False},
                }
            ),
        },
    )

    assert "成功" in summary
    assert "本次联网获取" in summary
    assert "已连接资料工具" in summary


def _seed_old_frontend_scope() -> None:
    st.session_state.clear()
    st.session_state.auth_token = "token"
    st.session_state.client_session_id = "session-a"
    st.session_state.learning_session_scope = (1, 10)
    st.session_state.selected_course_id = 1
    st.session_state.selected_class_id = 10
    st.session_state.last_chat_result = {"answer": "course A"}
    st.session_state.assignment_context = {"assignment_id": 1}


def test_scope_cancel_ends_and_forgets_old_session(monkeypatch):
    _seed_old_frontend_scope()
    ended: list[str] = []
    created: list[tuple[str, int, int | None]] = []
    monkeypatch.setattr(
        session_scope.api_client,
        "courses",
        lambda _token: [
            {"id": 1, "name": "A", "slug": "a"},
            {"id": 2, "name": "B", "slug": "b"},
        ],
    )
    monkeypatch.setattr(session_scope.st, "selectbox", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        session_scope.api_client,
        "end_learning_session",
        lambda _token, session_id: ended.append(session_id) or {},
    )
    monkeypatch.setattr(
        session_scope.api_client,
        "classes",
        lambda _token, course_id: [{"id": 20, "course_id": course_id, "name": "B1"}],
    )
    monkeypatch.setattr(
        session_scope.api_client,
        "create_learning_session",
        lambda _token, session_id, course_id, class_id=None: created.append(
            (session_id, course_id, class_id)
        ) or {},
    )

    assert session_scope.ensure_student_scope() is None
    assert ended == ["session-a"]
    assert not st.session_state.get("client_session_id")
    assert "learning_session_scope" not in st.session_state
    assert "last_chat_result" not in st.session_state
    assert "assignment_context" not in st.session_state

    monkeypatch.setattr(session_scope.st, "selectbox", lambda *_args, **_kwargs: 2)
    assert session_scope.ensure_student_scope() == (2, 20)
    assert created and created[0][0] != "session-a"
    assert st.session_state.learning_session_scope == (2, 20)


def test_invalidated_scope_rebinds_after_end_failure_without_reusing_old_id(monkeypatch):
    _seed_old_frontend_scope()
    created: list[tuple[str, int, int | None]] = []
    monkeypatch.setattr(
        session_scope.api_client,
        "courses",
        lambda _token: [{"id": 2, "name": "B", "slug": "b"}],
    )
    monkeypatch.setattr(
        session_scope.api_client,
        "classes",
        lambda _token, course_id: [{"id": 20, "course_id": course_id, "name": "B1"}],
    )
    monkeypatch.setattr(
        session_scope.api_client,
        "end_learning_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(APIClientError("end failed")),
    )
    monkeypatch.setattr(
        session_scope.api_client,
        "create_learning_session",
        lambda _token, session_id, course_id, class_id=None: created.append(
            (session_id, course_id, class_id)
        ) or {},
    )

    assert session_scope.ensure_student_scope() == (2, 20)
    assert created and created[0][0] != "session-a"
    assert st.session_state.client_session_id == created[0][0]
    assert st.session_state.learning_session_scope == (2, 20)
    assert "last_chat_result" not in st.session_state


def test_scope_lookup_failure_clears_old_session_and_results(monkeypatch):
    _seed_old_frontend_scope()
    ended: list[str] = []
    monkeypatch.setattr(
        session_scope.api_client,
        "courses",
        lambda _token: (_ for _ in ()).throw(APIClientError("unauthorized")),
    )
    monkeypatch.setattr(
        session_scope.api_client,
        "end_learning_session",
        lambda _token, session_id: ended.append(session_id) or {},
    )

    assert session_scope.ensure_student_scope() is None
    assert ended == ["session-a"]
    assert not st.session_state.get("client_session_id")
    assert "learning_session_scope" not in st.session_state
    assert "last_chat_result" not in st.session_state


def test_page_view_requires_session_scope_to_match_current_selection(monkeypatch):
    _seed_old_frontend_scope()
    st.session_state.selected_course_id = 2
    st.session_state.selected_class_id = 20
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        frontend_app.api_client,
        "heartbeat_learning_session",
        lambda *_args, **_kwargs: calls.append(("heartbeat", "called")),
    )
    monkeypatch.setattr(
        frontend_app.api_client,
        "record_learning_event",
        lambda *_args, **_kwargs: calls.append(("event", "called")),
    )

    frontend_app._track_page_view("student")

    assert calls == []


def test_page_view_ignores_malformed_dirty_scope_without_error(monkeypatch):
    _seed_old_frontend_scope()
    st.session_state.learning_session_scope = ("not-a-course", {"bad": "class"})
    calls: list[str] = []
    monkeypatch.setattr(
        frontend_app.api_client,
        "heartbeat_learning_session",
        lambda *_args, **_kwargs: calls.append("heartbeat"),
    )
    monkeypatch.setattr(
        frontend_app.api_client,
        "record_learning_event",
        lambda *_args, **_kwargs: calls.append("event"),
    )

    frontend_app._track_page_view("student")

    assert calls == []


def test_page_view_records_after_exact_session_scope_binding(monkeypatch):
    _seed_old_frontend_scope()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        frontend_app.api_client,
        "heartbeat_learning_session",
        lambda _token, session_id: calls.append(("heartbeat", session_id)),
    )
    monkeypatch.setattr(
        frontend_app.api_client,
        "record_learning_event",
        lambda _token, session_id, event_type, **kwargs: calls.append(
            ("event", (session_id, event_type, kwargs["course_id"], kwargs["class_id"]))
        ),
    )

    frontend_app._track_page_view("student")

    assert calls == [
        ("heartbeat", "session-a"),
        ("event", ("session-a", "PAGE_VIEW", 1, 10)),
    ]


def test_practice_draft_result_renderer_uses_plain_text_and_displays_complete_plan(monkeypatch):
    plain: list[str] = []
    unsafe_components: list[tuple[str, str]] = []
    links: list[tuple[str, str]] = []

    class _Column:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(student.st, "columns", lambda *_args, **_kwargs: [_Column(), _Column()])
    monkeypatch.setattr(student.st, "text", lambda value, **_kwargs: plain.append(str(value)))
    for name in ("write", "warning", "caption", "info"):
        monkeypatch.setattr(
            student.st,
            name,
            lambda value, _name=name, **_kwargs: unsafe_components.append((_name, str(value))),
        )
    monkeypatch.setattr(student.st, "markdown", lambda value, **_kwargs: None)
    monkeypatch.setattr(
        student.st,
        "link_button",
        lambda label, url, **_kwargs: links.append((str(label), str(url))),
    )

    hostile = "[track](javascript:alert(1)) <img src=x> # fake heading"
    result = {
        "proposal_label": hostile,
        "status": "grounded_draft",
        "background_quotes": [
            {
                "source_title": hostile,
                "quote_text": hostile,
                "delivery_mode": hostile,
                "source_url": "https://user:secret@example.test/path",
            },
            {
                "source_title": "safe title",
                "quote_text": "safe quote",
                "delivery_mode": "live",
                "source_url": "https://example.test/source",
            },
        ],
        "proposed_plan": {
            "objective": hostile,
            "lab_environment": hostile,
            "baseline_and_control": hostile,
            "variables": ["variable-a"],
            "steps": [
                {"action": "action-a", "expected_observation": "observation-a", "check": "check-a"},
                {"action": "action-b", "expected_observation": "observation-b", "check": "check-b"},
                {"action": "action-c", "expected_observation": "observation-c", "check": "check-c"},
            ],
            "evaluation_metrics": ["metric-a"],
            "expected_observations": ["expected-a"],
            "risks": [hostile],
            "unknowns": ["unknown-a"],
        },
        "reference_statuses": [
            {"source": hostile, "lookup_state": hostile, "delivery_mode": hostile}
        ],
    }

    student._render_practice_draft_result(result)

    rendered = "\n".join(plain)
    for expected in (
        hostile,
        "variable-a",
        "action-a",
        "observation-a",
        "check-a",
        "metric-a",
        "expected-a",
        "unknown-a",
    ):
        assert expected in rendered
    assert unsafe_components == []
    assert links == [("打开来源", "https://example.test/source")]


def test_practice_draft_external_consent_and_result_scope_are_fail_closed() -> None:
    assert student._practice_draft_external_queries(
        allowed=False,
        resource_type="paper",
        query="public paper",
    ) == []
    assert student._practice_draft_external_queries(
        allowed=True,
        resource_type="paper",
        query="public paper",
    ) == [{"resource_type": "paper", "query": "public paper"}]

    st.session_state.clear()
    st.session_state.current_user = {"id": 9}
    st.session_state.client_session_id = "session-a"
    st.session_state.last_practice_draft_result = {"status": "proposal_only"}
    st.session_state.practice_draft_result_scope = {
        "user_id": 8,
        "course_id": 1,
        "class_id": 2,
        "mode": "practice_draft",
        "client_session_id": "session-a",
    }

    assert student._practice_draft_result_for_scope(course_id=1, class_id=2) is None
    assert "last_practice_draft_result" not in st.session_state
    assert "practice_draft_result_scope" not in st.session_state
    assert "practice_draft_allow_external" in session_scope.SCOPE_DEPENDENT_KEYS
    assert "practice_draft_result_scope" in session_scope.SCOPE_DEPENDENT_KEYS


def test_logout_clears_all_practice_draft_inputs_consent_result_and_scope() -> None:
    st.session_state.clear()
    draft_keys = {
        "practice_draft_topic": "topic",
        "practice_draft_objective": "objective",
        "practice_draft_environment": "environment",
        "practice_draft_tools": "Python",
        "practice_draft_external_query": "public query",
        "practice_draft_external_type": "paper",
        "practice_draft_allow_external": True,
        "last_practice_draft_result": {"status": "proposal_only"},
        "practice_draft_result_scope": {"user_id": 1},
    }
    st.session_state.auth_token = "token"
    st.session_state.current_user = {"id": 1}
    st.session_state.client_session_id = "session"
    for key, value in draft_keys.items():
        st.session_state[key] = value

    frontend_app._clear_auth_session()

    for key in draft_keys:
        assert key not in st.session_state
