import pytest
import streamlit as st

from frontend import api_client
from frontend import app as frontend_app
from frontend.pages import teacher
from frontend.presentation import build_hotspot_view_model


class _StuckExpander:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _StuckButtonColumn:
    def __init__(self, buttons: list[dict], clicked_action: str | None = None):
        self.buttons = buttons
        self.clicked_action = clicked_action

    def button(self, label, *, key, **_kwargs):
        action = str(key).rsplit("_", 1)[-1]
        self.buttons.append({"label": label, "key": key, "action": action})
        return action == self.clicked_action


def _capture_stuck_actions(monkeypatch, item, *, key_prefix="phase3", clicked_action=None, api_error=None):
    buttons: list[dict] = []
    calls: list[tuple] = []
    clears: list[bool] = []
    reruns: list[bool] = []
    successes: list[str] = []
    errors: list[str] = []
    texts: list[str] = []
    monkeypatch.setattr(teacher.st, "expander", lambda *_args, **_kwargs: _StuckExpander())
    monkeypatch.setattr(teacher.st, "text", lambda value, *_args, **_kwargs: texts.append(str(value)))
    monkeypatch.setattr(
        teacher.st,
        "columns",
        lambda count: [_StuckButtonColumn(buttons, clicked_action) for _ in range(count)],
    )
    monkeypatch.setattr(teacher.st, "success", lambda message: successes.append(str(message)))
    monkeypatch.setattr(teacher.st, "error", lambda message: errors.append(str(message)))
    monkeypatch.setattr(teacher.st.cache_data, "clear", lambda: clears.append(True))
    monkeypatch.setattr(teacher.st, "rerun", lambda: reruns.append(True))

    def update(*args):
        calls.append(args)
        if api_error is not None:
            raise api_error

    monkeypatch.setattr(teacher.api_client, "update_stuck_report", update)
    teacher._render_stuck_review_actions("teacher-token", [item], key_prefix=key_prefix)
    return {"buttons": buttons, "calls": calls, "clears": clears, "reruns": reruns,
            "successes": successes, "errors": errors, "texts": texts}


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("active", ["confirm", "ignore", "resolve"]),
        (" ACTIVE ", ["confirm", "ignore", "resolve"]),
        ("confirmed", ["resolve"]),
        ("ignored", []),
        ("resolved", []),
        ("unknown", []),
        ("", []),
        (None, []),
    ],
)
@pytest.mark.parametrize("key_prefix", ["phase3", "legacy"])
def test_stuck_review_actions_follow_current_status(monkeypatch, status, expected, key_prefix):
    captured = _capture_stuck_actions(
        monkeypatch,
        {"id": 17, "status": status, "reason": "plain reason"},
        key_prefix=key_prefix,
    )
    assert [item["action"] for item in captured["buttons"]] == expected
    assert all(item["key"].startswith(f"{key_prefix}_stuck_17_") for item in captured["buttons"])
    assert captured["calls"] == []


def test_stuck_suspected_without_report_id_has_no_actions(monkeypatch):
    captured = _capture_stuck_actions(
        monkeypatch,
        {"status": "active", "reason": "suspected only"},
    )
    assert captured["buttons"] == []
    assert captured["calls"] == []


def test_stuck_allowed_action_success_calls_once_clears_and_reruns(monkeypatch):
    captured = _capture_stuck_actions(
        monkeypatch,
        {"id": 23, "status": "confirmed", "reason": "plain reason"},
        key_prefix="legacy",
        clicked_action="resolve",
    )
    assert [item["action"] for item in captured["buttons"]] == ["resolve"]
    assert captured["calls"] == [("teacher-token", 23, "resolve")]
    assert captured["clears"] == [True]
    assert captured["reruns"] == [True]
    assert len(captured["successes"]) == 1
    assert captured["errors"] == []


def test_stuck_allowed_action_error_has_no_clear_or_rerun(monkeypatch):
    captured = _capture_stuck_actions(
        monkeypatch,
        {"id": 24, "status": "active", "reason": "plain reason"},
        clicked_action="confirm",
        api_error=api_client.APIClientError("操作未完成"),
    )
    assert captured["calls"] == [("teacher-token", 24, "confirm")]
    assert captured["clears"] == []
    assert captured["reruns"] == []
    assert captured["successes"] == []
    assert len(captured["errors"]) == 1
    assert captured["errors"][0]


def test_stuck_action_keys_do_not_collide_between_reports(monkeypatch):
    buttons: list[dict] = []
    monkeypatch.setattr(teacher.st, "expander", lambda *_args, **_kwargs: _StuckExpander())
    monkeypatch.setattr(teacher.st, "text", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        teacher.st,
        "columns",
        lambda count: [_StuckButtonColumn(buttons) for _ in range(count)],
    )
    teacher._render_stuck_review_actions(
        "teacher-token",
        [{"id": 30, "status": "confirmed"}, {"id": 31, "status": "confirmed"}],
        key_prefix="phase3",
    )
    assert [item["key"] for item in buttons] == [
        "phase3_stuck_30_resolve",
        "phase3_stuck_31_resolve",
    ]


@pytest.mark.parametrize(
    "report_id",
    [
        17.9, 17.0, float("inf"), float("-inf"), float("nan"), True, False, 0, -1,
        "17.9", "1e2", "+17", "-17", " 17 ", "017", [], {},
        9_223_372_036_854_775_808, "9223372036854775808",
    ],
)
@pytest.mark.parametrize("key_prefix", ["phase3", "legacy"])
def test_stuck_malformed_report_id_never_renders_or_calls(monkeypatch, report_id, key_prefix):
    captured = _capture_stuck_actions(
        monkeypatch,
        {"id": report_id, "status": "active", "reason": "plain reason"},
        key_prefix=key_prefix,
    )
    assert captured["buttons"] == []
    assert captured["calls"] == []
    assert captured["clears"] == []
    assert captured["reruns"] == []


@pytest.mark.parametrize("report_id", [17, "17", 9_223_372_036_854_775_807, "9223372036854775807"])
def test_stuck_canonical_positive_report_id_remains_actionable(monkeypatch, report_id):
    captured = _capture_stuck_actions(
        monkeypatch,
        {"id": report_id, "status": "confirmed", "reason": "plain reason"},
    )
    assert [item["action"] for item in captured["buttons"]] == ["resolve"]
    assert captured["calls"] == []


@pytest.mark.parametrize(
    ("status", "label", "actions"),
    [
        (" ACTIVE ", "待处理", ["confirm", "ignore", "resolve"]),
        ("Reported", "待处理", []),
        (" Confirmed ", "已确认", ["resolve"]),
        ("RESOLVED", "已解决", []),
        (" Ignored ", "已忽略", []),
        ("unknown", "状态待核实", []),
        (None, "状态待核实", []),
        (True, "状态待核实", []),
        ([], "状态待核实", []),
        ({}, "状态待核实", []),
    ],
)
def test_stuck_status_label_and_actions_share_normalization(monkeypatch, status, label, actions):
    captured = _capture_stuck_actions(
        monkeypatch,
        {"id": 17, "status": status, "reason": "plain reason"},
    )
    assert teacher._progress_label(status) == label
    assert [item["action"] for item in captured["buttons"]] == actions
    assert captured["calls"] == []


@pytest.mark.parametrize("key_prefix", ["phase3", "legacy"])
def test_confirmed_stuck_card_displays_trusted_status_label(monkeypatch, key_prefix):
    captured = _capture_stuck_actions(
        monkeypatch,
        {"id": 17, "status": " Confirmed ", "reason": "plain reason"},
        key_prefix=key_prefix,
    )

    assert "处理状态：已确认" in captured["texts"]
    assert [item["action"] for item in captured["buttons"]] == ["resolve"]
    assert captured["calls"] == []


def test_teacher_page_access_uses_logged_in_role(monkeypatch):
    monkeypatch.setattr(
        teacher.st.sidebar,
        "selectbox",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("role selectbox must not be used")),
    )
    st.session_state.clear()

    assert teacher._has_management_access() is False

    st.session_state.auth_token = "token"
    st.session_state.current_user = {"role": "student"}
    assert teacher._has_management_access() is False

    st.session_state.current_user = {"role": "teacher"}
    assert teacher._has_management_access() is True

    st.session_state.current_user = {"role": "admin"}
    assert teacher._has_management_access() is True


def test_sidebar_navigation_hides_teacher_entry_for_non_management_users():
    st.session_state.clear()
    assert "teacher" not in frontend_app._navigation_pages()
    assert "tools" not in frontend_app._navigation_pages()

    st.session_state.current_user = {"role": "student"}
    assert "teacher" not in frontend_app._navigation_pages()
    assert "tools" not in frontend_app._navigation_pages()

    st.session_state.current_user = {"role": "teacher"}
    assert "teacher" in frontend_app._navigation_pages()
    assert "tools" in frontend_app._navigation_pages()

    st.session_state.current_user = {"role": "admin"}
    assert "teacher" in frontend_app._navigation_pages()
    assert "tools" in frontend_app._navigation_pages()


def test_clear_auth_session_removes_token_user_and_client_session():
    st.session_state.clear()
    st.session_state.auth_token = "token"
    st.session_state.current_user = {"role": "teacher"}
    st.session_state.client_session_id = "session"

    frontend_app._clear_auth_session()

    assert "auth_token" not in st.session_state
    assert "current_user" not in st.session_state
    assert "client_session_id" not in st.session_state


def test_teacher_api_client_requires_real_token_for_management_requests():
    protected_calls = [
        lambda: api_client.teacher_analytics(),
        lambda: api_client.teacher_faq(),
        lambda: api_client.knowledge_documents(),
        lambda: api_client.upload_knowledge("demo.md", b"# demo"),
        lambda: api_client.reindex_document(1),
        lambda: api_client.publish_document(1),
        lambda: api_client.unpublish_document(1),
        lambda: api_client.delete_document(1),
    ]

    for call in protected_calls:
        with pytest.raises(api_client.APIClientError, match="登录已失效"):
            call()


def test_teacher_knowledge_upload_prompt_matches_backend_limit():
    source = teacher.__file__
    assert source
    text = open(source, encoding="utf-8").read()
    assert "25MB" in text


def test_teacher_resource_sync_sources_include_remote_and_local_standard_options():
    options = teacher._resource_sync_connector_options()

    assert options["rfc_editor"] == "standard"
    assert options["standard_source"] == "standard"


def test_teacher_document_status_projection_is_chinese_and_keeps_two_states_separate():
    rows = teacher._document_status_rows(
        [
            {
                "id": 9,
                "filename": "network_basics_v2.md",
                "file_type": ".md",
                "chunk_count": 3,
                "index_status": "INDEXED",
                "published_to_students": False,
                "published_by": None,
                "published_at": None,
                "publication_last_action": {"action": "reindex", "at": "2026-10-18T00:00:00Z"},
                "updated_at": "2026-10-18T00:00:00Z",
            }
        ]
    )

    assert rows == [
        {
            "文档编号": 9,
            "文件名": "network_basics_v2.md",
            "格式": ".md",
            "切片数": 3,
            "索引状态": "已索引",
            "学生可见性": "未发布",
            "发布人": "-",
            "发布时间": "-",
            "最近操作": "重建索引",
            "最近操作时间": "2026-10-18T00:00:00Z",
            "更新时间": "2026-10-18T00:00:00Z",
        }
    ]


def test_teacher_api_client_sends_bearer_token(monkeypatch):
    calls = []

    class FakeResponse:
        content = b""

        def raise_for_status(self):
            return None

        def json(self):
            return {"documents": [], "index_jobs": []}

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    api_client.knowledge_documents(token="teacher-token")

    assert calls[0]["headers"] == {"Authorization": "Bearer teacher-token"}
    assert "X-User-Role" not in calls[0]["headers"]


def test_teacher_cluster_management_api_client_uses_existing_protected_endpoints(monkeypatch):
    calls = []

    class FakeResponse:
        content = b""

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    api_client.rename_question_cluster("teacher-token", 12, "NAS FAQ")
    api_client.merge_question_clusters(
        "teacher-token",
        target_cluster_id=12,
        source_cluster_ids=[13, 14],
        label="Merged FAQ",
    )
    api_client.split_question_cluster("teacher-token", 12, question_ids=[101], label="Split FAQ")
    api_client.update_stuck_report("teacher-token", 7, "confirm")

    assert [(call["method"], call["url"].replace(api_client.API_BASE_URL, "")) for call in calls] == [
        ("PATCH", "/api/teacher/question-clusters/12"),
        ("POST", "/api/teacher/question-clusters/merge"),
        ("POST", "/api/teacher/question-clusters/12/split"),
        ("POST", "/api/teacher/stuck-reports/7/action"),
    ]
    assert all(call["headers"] == {"Authorization": "Bearer teacher-token"} for call in calls)
    assert calls[0]["json"] == {"label": "NAS FAQ"}
    assert calls[1]["json"] == {
        "target_cluster_id": 12,
        "source_cluster_ids": [13, 14],
        "label": "Merged FAQ",
    }
    assert calls[2]["json"] == {"question_ids": [101], "label": "Split FAQ"}
    assert calls[3]["json"] == {"action": "confirm"}


def test_teacher_phase3_faq_clients_send_scope_and_bearer_token(monkeypatch):
    calls = []

    class FakeResponse:
        content = b"csv"

        def raise_for_status(self):
            return None

        def json(self):
            return {"items": []}

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    payload = api_client.teacher_question_cluster_faq_json("teacher-token", course_id=3, class_id=4)
    csv_bytes = api_client.teacher_phase3_faq_csv("teacher-token", course_id=3, class_id=4)

    assert payload == {"items": []}
    assert csv_bytes == b"csv"
    assert [(call["method"], call["url"].replace(api_client.API_BASE_URL, "")) for call in calls] == [
        ("GET", "/api/teacher/question-clusters/faq-export"),
        ("GET", "/api/teacher/phase3/faq-export.csv"),
    ]
    assert all(call["headers"] == {"Authorization": "Bearer teacher-token"} for call in calls)
    assert all(call["params"] == {"course_id": 3, "class_id": 4} for call in calls)


def test_teacher_timeline_panel_success_renders_events(monkeypatch):
    captured = {}

    def fake_timeline(token, student_id, *, limit=100):
        captured["request"] = {"token": token, "student_id": student_id, "limit": limit}
        return {
            "events": [
                {
                    "created_at": "2026-10-04T10:00:00",
                    "event_type": "PAGE_VIEW",
                    "page_name": "student",
                    "course_id": 1,
                    "class_id": 2,
                    "module": "course_intro",
                }
            ]
        }

    monkeypatch.setattr(api_client, "teacher_phase3_student_timeline", fake_timeline)
    monkeypatch.setattr(teacher.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        teacher.st,
        "number_input",
        lambda label, **kwargs: 17 if label == "学生编号" else 25,
    )
    monkeypatch.setattr(teacher.st, "button", lambda *args, **kwargs: True)
    monkeypatch.setattr(teacher.st, "dataframe", lambda df, **kwargs: captured.setdefault("rows", df.to_dict("records")))

    teacher._render_student_timeline_panel("teacher-token")

    assert captured["request"] == {"token": "teacher-token", "student_id": 17, "limit": 25}
    assert captured["rows"][0]["事件"] == "查看页面"
    assert captured["rows"][0]["页面"] == "课程学习"
    assert captured["rows"][0]["课程编号"] == 1
    assert captured["rows"][0]["班级编号"] == 2


def test_teacher_timeline_panel_403_shows_safe_error(monkeypatch):
    errors = []

    def fake_timeline(*args, **kwargs):
        raise api_client.APIClientError("Traceback: token=secret")

    monkeypatch.setattr(api_client, "teacher_phase3_student_timeline", fake_timeline)
    monkeypatch.setattr(teacher.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(teacher.st, "number_input", lambda *args, **kwargs: 1)
    monkeypatch.setattr(teacher.st, "button", lambda *args, **kwargs: True)
    monkeypatch.setattr(teacher.st, "error", lambda message: errors.append(message))

    teacher._render_student_timeline_panel("teacher-token")

    assert errors == ["无法查看该学生的学习轨迹，请核对课程范围与学生编号。"]
    assert "Traceback" not in errors[0]
    assert "secret" not in errors[0]


def test_teacher_phase3_faq_downloads_include_json_and_preserve_csv(monkeypatch):
    downloads = []
    requests = []

    def fake_json(token, *, course_id=None, class_id=None):
        requests.append(("json", token, course_id, class_id))
        return {"items": [{"label": "NAS FAQ"}]}

    def fake_csv(token, *, course_id=None, class_id=None):
        requests.append(("csv", token, course_id, class_id))
        return b"cluster_id,label\n1,NAS FAQ\n"

    monkeypatch.setattr(api_client, "teacher_question_cluster_faq_json", fake_json)
    monkeypatch.setattr(api_client, "teacher_phase3_faq_csv", fake_csv)
    monkeypatch.setattr(teacher.st, "download_button", lambda label, **kwargs: downloads.append({"label": label, **kwargs}))

    teacher._render_phase3_faq_downloads("teacher-token", course_id=7, class_id=8)

    assert requests == [("json", "teacher-token", 7, 8), ("csv", "teacher-token", 7, 8)]
    assert downloads[0]["file_name"] == "phase3_faq.json"
    assert downloads[0]["mime"] == "application/json"
    assert b"NAS FAQ" in downloads[0]["data"]
    assert downloads[1]["file_name"] == "phase3_faq.csv"
    assert downloads[1]["mime"] == "text/csv"


def test_hotspot_page_uses_one_authorized_unfiltered_summary_and_safe_topic_text(monkeypatch):
    calls = []
    labels = []
    plain = []
    metrics = []

    class Box:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def metric(self, label, value):
            metrics.append((label, value))

    hostile = "![track](https://evil.test/x) <img src=x>"
    summary = {
        "clusters": {"items": [{"id": 7, "label": hostile, "approved": True, "is_noise": False,
                                 "representative_question": "什么是 NAS？",
                                 "members": [{"question_id": 71, "question": "什么是 NAS？", "created_at": "2026-10-19"}]}]},
        "faq": {"needs_review_count": 0, "items": [{"cluster_id": 7, "label": hostile,
                                                   "representative_question": "什么是 NAS？", "count": 1,
                                                   "answer": hostile,
                                                   "citations": [{"source_id": "S1", "title": hostile, "snippet": hostile}]}]},
    }
    monkeypatch.setattr(api_client, "teacher_phase3_summary", lambda token, **kwargs: calls.append((token, kwargs)) or summary)
    monkeypatch.setattr(teacher.st, "subheader", lambda *_args: None)
    monkeypatch.setattr(teacher.st, "caption", lambda *_args: None)
    monkeypatch.setattr(teacher.st, "text", lambda value: plain.append(str(value)))
    monkeypatch.setattr(teacher.st, "columns", lambda count: [Box() for _ in range(count)])
    monkeypatch.setattr(teacher.st, "text_input", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(teacher.st, "selectbox", lambda *_args, **_kwargs: "all")
    monkeypatch.setattr(teacher.st, "bar_chart", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(teacher.st, "dataframe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(teacher.st, "expander", lambda label, **_kwargs: labels.append(label) or Box())
    monkeypatch.setattr(teacher, "_render_hotspot_actions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(teacher, "_render_phase3_faq_downloads", lambda *_args, **_kwargs: None)

    teacher._render_hotspot_library("teacher-token", course_id=2, class_id=3)

    assert calls == [("teacher-token", {"course_id": 2, "class_id": 3})]
    assert metrics == [("热点主题", 1), ("待教师审核", 0), ("已审核主题", 1), ("本次可导出常见问答", 1)]
    assert labels == ["主题 7"]
    assert any(hostile in item for item in plain)


def test_hotspot_inconsistent_same_response_hides_counts_and_download(monkeypatch):
    warnings = []
    summary = {
        "clusters": {"items": [{"id": 7, "label": "主题", "approved": False, "is_noise": False,
                                 "representative_question": "问题", "members": [{"question_id": 71, "question": "问题"}]}]},
        "faq": {"items": [{"cluster_id": 7, "label": "主题", "representative_question": "问题",
                           "count": 1, "answer": "不一致", "citations": [{"source_id": "S1", "snippet": "摘录"}]}]},
    }
    monkeypatch.setattr(api_client, "teacher_phase3_summary", lambda *_args, **_kwargs: summary)
    monkeypatch.setattr(teacher.st, "subheader", lambda *_args: None)
    monkeypatch.setattr(teacher.st, "text", lambda *_args: None)
    monkeypatch.setattr(teacher.st, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(teacher.st, "columns", lambda *_args: pytest.fail("inconsistent counts must not render"))
    monkeypatch.setattr(teacher, "_render_phase3_faq_downloads", lambda *_args, **_kwargs: pytest.fail("inconsistent FAQ must not prepare download"))
    teacher._render_hotspot_library("token", course_id=2, class_id=None)
    assert warnings == ["资料状态暂不同步，请刷新"]


def test_hotspot_download_rejected_scope_never_replays_old_bytes(monkeypatch):
    downloads = []
    monkeypatch.setattr(teacher.st, "download_button", lambda label, **kwargs: downloads.append(kwargs))
    monkeypatch.setattr(teacher.st, "error", lambda *_args: None)
    monkeypatch.setattr(api_client, "teacher_question_cluster_faq_json", lambda *_args, **_kwargs: {"items": [{"cluster_id": 1}]})
    monkeypatch.setattr(api_client, "teacher_phase3_faq_csv", lambda *_args, **_kwargs: b"old data")
    teacher._render_phase3_faq_downloads("token", course_id=1, class_id=2)
    assert downloads[0]["disabled"] is False and downloads[1]["data"] == b"old data"

    def forbidden(*_args, **_kwargs):
        raise api_client.APIClientError("forbidden", status_code=403)

    monkeypatch.setattr(api_client, "teacher_question_cluster_faq_json", forbidden)
    monkeypatch.setattr(api_client, "teacher_phase3_faq_csv", forbidden)
    teacher._render_phase3_faq_downloads("token", course_id=3, class_id=4)
    assert downloads[2]["disabled"] is True and downloads[2]["data"] == b""
    assert downloads[3]["disabled"] is True and downloads[3]["data"] == b""


def test_hotspot_write_requires_confirmation_and_uses_stable_id(monkeypatch):
    cluster = build_hotspot_view_model({
        "clusters": {"items": [{"id": 9, "label": "旧名", "approved": True, "is_noise": False,
                                 "members": [{"question_id": 90, "question": "示例问题"}]}]},
        "faq": {"items": []},
    }).topics[0]
    calls = []
    confirmed = {"value": False}

    class Box:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(teacher.st, "form", lambda *_args: Box())
    monkeypatch.setattr(teacher.st, "expander", lambda *_args, **_kwargs: Box())
    monkeypatch.setattr(teacher.st, "text_input", lambda *_args, **_kwargs: "新名")
    monkeypatch.setattr(teacher.st, "checkbox", lambda *_args, **_kwargs: confirmed["value"])
    monkeypatch.setattr(teacher.st, "form_submit_button", lambda label, **_kwargs: label == "保存名称")
    monkeypatch.setattr(teacher.st, "multiselect", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(teacher.st, "warning", lambda *_args: None)
    monkeypatch.setattr(teacher, "_call_cluster_action", lambda action, _message: action())
    monkeypatch.setattr(api_client, "rename_question_cluster", lambda *args: calls.append(args))
    teacher._render_hotspot_actions("token", cluster, (cluster,))
    assert calls == []
    confirmed["value"] = True
    teacher._render_hotspot_actions("token", cluster, (cluster,))
    assert calls == [("token", 9, "新名")]


@pytest.mark.parametrize("submitted,expected", [
    ("审核此主题", ("approve", "token", 9)),
    ("合并主题", ("merge", "token", 9, (10,), "新主题")),
    ("拆分问题", ("split", "token", 9, (90,), "新主题")),
])
def test_hotspot_other_writes_require_confirmation_and_keep_stable_ids(monkeypatch, submitted, expected):
    topics = build_hotspot_view_model({
        "clusters": {"items": [
            {"id": 9, "label": "旧主题", "approved": False, "is_noise": False,
             "members": [{"question_id": 90, "question": "原问题"}]},
            {"id": 10, "label": "来源主题", "approved": False, "is_noise": False,
             "members": [{"question_id": 100, "question": "另一个问题"}]},
        ]},
        "faq": {"items": []},
    }).topics
    target = next(item for item in topics if item.cluster_id == 9)
    calls = []
    confirmed = {"value": False}

    class Box:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(teacher.st, "form", lambda *_args: Box())
    monkeypatch.setattr(teacher.st, "expander", lambda *_args, **_kwargs: Box())
    monkeypatch.setattr(teacher.st, "text_input", lambda *_args, **_kwargs: "新主题")
    monkeypatch.setattr(teacher.st, "checkbox", lambda *_args, **_kwargs: confirmed["value"])
    monkeypatch.setattr(teacher.st, "form_submit_button", lambda label, **_kwargs: label == submitted)
    monkeypatch.setattr(teacher.st, "multiselect", lambda label, *_args, **_kwargs: [10] if "来源主题" in label else [90])
    monkeypatch.setattr(teacher.st, "warning", lambda *_args: None)
    monkeypatch.setattr(teacher, "_call_cluster_action", lambda action, _message: action())
    monkeypatch.setattr(api_client, "approve_question_cluster", lambda *args: calls.append(("approve", *args)))
    monkeypatch.setattr(api_client, "merge_question_clusters", lambda token, *, target_cluster_id, source_cluster_ids, label:
                        calls.append(("merge", token, target_cluster_id, tuple(source_cluster_ids), label)))
    monkeypatch.setattr(api_client, "split_question_cluster", lambda token, cluster_id, *, question_ids, label:
                        calls.append(("split", token, cluster_id, tuple(question_ids), label)))

    teacher._render_hotspot_actions("token", target, topics)
    assert calls == []
    confirmed["value"] = True
    teacher._render_hotspot_actions("token", target, topics)
    assert calls == [expected]


def test_hotspot_scope_switch_and_authorization_failure_clear_unsubmitted_forms(monkeypatch):
    st.session_state.clear()
    st.session_state.current_user = {"id": 5, "role": "teacher"}
    st.session_state.teacher_hotspot_course_bound = 1
    st.session_state.teacher_hotspot_scope = (5, 1, 2)
    st.session_state.teacher_hotspot_rename_label_9 = "旧课程待提交名称"
    st.session_state.teacher_hotspot_merge_sources_9 = [10]
    st.session_state.teacher_hotspot_prepared_csv = b"old"
    monkeypatch.setattr(api_client, "courses", lambda _token: [
        {"id": 1, "name": "旧课程"}, {"id": 2, "name": "新课程"},
    ])
    monkeypatch.setattr(api_client, "classes", lambda _token, *, course_id: [
        {"id": 3, "name": "新班级"}] if course_id == 2 else [])
    monkeypatch.setattr(teacher.st, "selectbox", lambda label, *_args, **_kwargs: 2 if label == "当前课程" else 3)

    assert teacher._select_teacher_scope("token") == (2, 3)
    assert st.session_state.teacher_hotspot_scope == (5, 2, 3)
    assert "teacher_hotspot_rename_label_9" not in st.session_state
    assert "teacher_hotspot_merge_sources_9" not in st.session_state
    assert "teacher_hotspot_prepared_csv" not in st.session_state

    monkeypatch.setattr(api_client, "courses", lambda _token: [])
    monkeypatch.setattr(teacher.st, "info", lambda *_args: None)
    assert teacher._select_teacher_scope("token") is None
    assert "teacher_hotspot_scope" not in st.session_state
    st.session_state.clear()


def test_admin_overview_faq_does_not_make_an_unauthorized_local_export(monkeypatch):
    displayed = []
    monkeypatch.setattr(teacher.st, "subheader", lambda *_args: None)
    monkeypatch.setattr(teacher.st, "dataframe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(teacher.st, "text", lambda *_args: None)
    monkeypatch.setattr(teacher.st, "caption", lambda *_args: None)
    monkeypatch.setattr(teacher.st, "info", lambda message: displayed.append(message))
    monkeypatch.setattr(teacher.st, "download_button", lambda *_args, **_kwargs: pytest.fail("local FAQ export must not render"))
    teacher._render_faq({"items": [{"representative_question": "Q", "count": 1, "module": "course_qa"}]})
    assert any("热点问题库" in message for message in displayed)


def test_admin_agent_statistics_use_real_fields_and_chinese_whitelist(monkeypatch):
    tables = []
    monkeypatch.setattr(teacher.st, "subheader", lambda *_args: None)
    monkeypatch.setattr(teacher.st, "columns", lambda count: [type("Box", (), {"metric": lambda *_args: None})() for _ in range(count)])
    monkeypatch.setattr(teacher, "_table_if_any", lambda title, rows: tables.append((title, rows)))
    teacher._render_agent_paths({
        "agent_metrics": {"top_actions": [{"action": "RETRIEVE", "count": 2}],
                          "stop_reasons": [{"reason": "ANSWERED", "count": 1}]},
        "tool_metrics": {"top_tools": [{"tool_name": "course_retriever", "count": 3}]},
    })
    assert tables == [
        ("最常选择的动作", [{"处理环节": "检索资料", "次数": 2}]),
        ("最常调用的工具", [{"工具": "课程资料检索", "次数": 3}]),
        ("停止原因", [{"状态": "已回答", "次数": 1}]),
    ]
