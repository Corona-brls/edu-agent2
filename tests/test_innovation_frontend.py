from __future__ import annotations

from frontend.pages import innovation
from frontend.api_client import APIClientError


def test_innovation_case_caption_shows_relevance_fields():
    parts = innovation._case_caption_parts(
        {
            "source_id": "paper:nas",
            "origin_kind": "remote",
            "delivery_mode": "fresh_cache",
            "fetched_at": "2026-10-01T00:00:00Z",
            "relevance_score": 9.2,
            "relevance_reasons": ["title:domain=5g_nas(+9.0)", "tags:entity=nas(+7.7)"],
            "connector_rank": 2,
        }
    )

    caption = " | ".join(parts)
    assert "资料状态：有效缓存" in caption
    assert "获取时间：2026-10-01T00:00:00Z" in caption
    assert "source_id=" not in caption
    assert "relevance_score=" not in caption


def test_innovation_case_caption_handles_legacy_cases_without_new_fields():
    parts = innovation._case_caption_parts(
        {
            "source_id": "paper:legacy",
            "origin_kind": "unknown",
            "delivery_mode": "unverified_history",
            "fetched_at": "",
        }
    )

    caption = " | ".join(parts)
    assert "状态待核实" in caption
    assert "paper:legacy" not in caption
    assert "relevance_score" not in caption
    assert "relevance_reasons" not in caption
    assert "connector_rank" not in caption


def test_suggestion_renderer_uses_plain_text_for_all_dynamic_fields(monkeypatch):
    rendered = []
    monkeypatch.setattr(innovation.st, "text", lambda value: rendered.append(str(value)))
    monkeypatch.setattr(
        innovation.st,
        "markdown",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dynamic markdown is forbidden")),
    )
    monkeypatch.setattr(
        innovation.st,
        "write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dynamic write is forbidden")),
    )

    innovation._render_suggestion_result(
        {
            "proposal_label": "[追踪链接](https://evil.test/pixel)",
            "generation_mode": "deepseek",
            "status": "proposed",
            "suggestions": [
                {
                    "hypothesis": "<img src=x onerror=alert(1)>",
                    "comparison": "javascript:alert(1)",
                    "metrics": ["检测率"],
                    "next_question": "下一步？",
                    "risks": ["仅隔离环境"],
                    "background_source_ids": ["rag:1:2"],
                }
            ],
            "source_statuses": [
                {
                    "source_id": "rag:1:2",
                    "status": "available",
                    "delivery_mode": "local_rag",
                }
            ],
        }
    )

    joined = "\n".join(rendered)
    assert "<img" in joined
    assert "javascript:" in joined
    assert "背景案例 1：来源标题待核实" in joined
    assert "背景案例 1（资料类型待核实）：当前可用；当前课程资料" in joined
    assert "rag:1:2" not in joined


def test_suggestion_scope_includes_user_course_class_learning_and_innovation_session():
    first = innovation._suggestion_scope(
        user_id=1,
        course_id=2,
        class_id=3,
        client_session_id="learning-a",
        innovation_session_id=4,
    )
    assert first == (1, 2, 3, "innovation_suggestion", "learning-a", 4)
    assert first != innovation._suggestion_scope(
        user_id=1,
        course_id=2,
        class_id=3,
        client_session_id="learning-a",
        innovation_session_id=5,
    )


def test_innovation_safe_url_rejects_credentials_and_non_http():
    assert innovation._safe_external_url("https://example.test/source") == "https://example.test/source"
    assert innovation._safe_external_url("https://user:pass@example.test/source") == ""
    assert innovation._safe_external_url("javascript:alert(1)") == ""


def test_case_and_template_titles_use_fixed_expanders_and_plain_text(monkeypatch):
    labels = []
    rendered = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(innovation.st, "expander", lambda label, **_kwargs: labels.append(label) or Context())
    monkeypatch.setattr(innovation.st, "text", lambda value: rendered.append(str(value)))
    monkeypatch.setattr(innovation.st, "link_button", lambda *_args, **_kwargs: None)
    malicious = "[追踪](https://evil.test/pixel)<img src=x>"
    innovation._render_case_card(
        {
            "title": malicious,
            "type": "paper",
            "source_name": "semantic_scholar",
            "summary": malicious,
            "url": "javascript:alert(1)",
            "source_id": "paper:x",
        }
    )
    innovation._render_template_idea(
        {
            "title": malicious,
            "label": malicious,
            "hypothesis": malicious,
            "experiment_method": malicious,
            "risks": [malicious],
            "evidence": [{"title": malicious, "type": "paper", "source": "safe"}],
        }
    )
    assert labels == ["案例详情", "候选方向详情"]
    assert malicious in "\n".join(rendered)


def test_template_idea_renderer_uses_trusted_identity_not_historical_label(monkeypatch):
    rendered: list[str] = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(innovation.st, "expander", lambda *_args, **_kwargs: Context())
    monkeypatch.setattr(innovation.st, "text", lambda value: rendered.append(str(value)))
    idea = {
        "generation_mode": "template",
        "label": "模型建议",
        "title": "历史候选",
        "hypothesis": "尚待验证",
        "experiment_method": "隔离对照",
        "risks": [],
        "evidence": [],
    }
    innovation._render_template_idea(idea)
    assert "生成方式：模板启发" in rendered
    assert "性质：模板启发／待验证方向" in rendered
    assert not any("性质：模型建议" in item for item in rendered)
    rendered.clear()
    innovation._render_template_idea({**idea, "generation_mode": "unknown"})
    assert "生成方式：生成方式待核实" in rendered
    assert not any("性质：模型建议" in item for item in rendered)
    rendered.clear()
    innovation._render_template_idea({**idea, "label": "案例背景／待验证方向"})
    assert "生成方式：模板启发" in rendered
    assert "性质：案例背景／待验证方向" in rendered


def test_suggestion_renderer_includes_policy_collection_and_warnings(monkeypatch):
    rendered = []
    monkeypatch.setattr(innovation.st, "text", lambda value: rendered.append(str(value)))
    innovation._render_suggestion_result(
        {
            "status": "refused",
            "refusal_reason": "待补充",
            "course_reference_status": "blocked_by_policy",
            "collection_status": "not_recorded",
            "warnings": ["课程案例未发送"],
            "suggestions": [],
            "source_statuses": [],
        }
    )
    joined = "\n".join(rendered)
    assert "课程参考状态：课程政策禁止外发参考" in joined
    assert "案例收集状态：历史收集过程未记录" in joined
    assert "提示：课程案例未发送" in joined


def test_scope_mismatch_clears_suggestion_checkbox(monkeypatch):
    state = {
        "innovation_suggestion_result": {"status": "proposed"},
        "innovation_suggestion_result_scope": (1, 2, 3, "innovation_suggestion", "old", 4),
        "innovation_suggestion_use_course_references": True,
    }
    monkeypatch.setattr(innovation.st, "session_state", state)
    innovation._clear_stale_suggestion_state((1, 2, 3, "innovation_suggestion", "new", 4))
    assert "innovation_suggestion_result" not in state
    assert "innovation_suggestion_result_scope" not in state
    assert state["innovation_suggestion_use_course_references"] is False


def test_innovation_unknown_identity_never_looks_successful(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr(innovation.st, "text", lambda value: rendered.append(str(value)))
    innovation._render_suggestion_result({
        "generation_mode": "mystery", "status": "mystery",
        "course_reference_status": "mystery", "collection_status": "mystery",
        "source_statuses": [{"source_id": "rag:12:34", "status": "mystery", "delivery_mode": "mystery"}],
    })
    joined = "\n".join(rendered)
    assert "生成方式待核实" in joined
    assert "建议状态：状态待核实" in joined
    assert "参考状态待核实" in joined
    assert "rag:12:34" not in joined


def test_innovation_source_identity_uses_current_case_title_as_plain_text(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr(innovation.st, "text", lambda value: rendered.append(str(value)))
    title = "<img src=x> ![pixel](https://evil.test/pixel)"
    innovation._render_suggestion_result(
        {"generation_mode": "deepseek", "status": "proposed", "collection_status": "recorded",
         "course_reference_status": "allowed",
         "suggestions": [{"background_source_ids": ["rag:7:9"]}],
         "source_statuses": [{"source_id": "rag:7:9", "status": "available",
                              "source_type": "course_document", "delivery_mode": "local_rag"}]},
        cases=[{"source_id": "rag:7:9", "title": title}],
    )
    joined = "\n".join(rendered)
    assert title in joined
    assert "rag:7:9" not in joined
    assert "本次收集过程已记录" in joined
    assert "本次允许使用课程参考" in joined


def _resume_page_fixture(monkeypatch, *, get_session):
    class State(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    state = State(auth_token="isolated-token", current_user={"id": 1}, client_session_id="learning-a")
    rendered: list[str] = []
    calls: list[tuple] = []
    monkeypatch.setattr(innovation.st, "session_state", state)
    monkeypatch.setattr(innovation, "ensure_student_scope", lambda: (2, 3))
    for name in ("title", "caption", "subheader", "text", "success", "info", "warning", "error"):
        monkeypatch.setattr(innovation.st, name, lambda value, **_kwargs: rendered.append(str(value)))
    monkeypatch.setattr(innovation.st, "form", lambda *_args, **_kwargs: Context())
    monkeypatch.setattr(innovation.st, "expander", lambda *_args, **_kwargs: Context())
    monkeypatch.setattr(innovation.st, "text_input", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(innovation.st, "text_area", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(innovation.st, "form_submit_button", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(innovation.st, "button", lambda label, **_kwargs: label == "打开所选会话")
    monkeypatch.setattr(innovation.st, "checkbox", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(innovation.st, "selectbox", lambda *_args, **_kwargs: 7)
    monkeypatch.setattr(innovation.st, "link_button", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(innovation.st, "rerun", lambda: None)
    monkeypatch.setattr(
        innovation.api_client,
        "list_innovation_sessions",
        lambda token, **scope: calls.append(("list", token, scope)) or {
            "items": [{"id": 7, "topic": "历史模板", "status": "FINALIZED", "created_at": "2026-10-19", "updated_at": "2026-10-19"}]
        },
        raising=False,
    )
    monkeypatch.setattr(
        innovation.api_client,
        "get_innovation_session",
        lambda token, session_id: calls.append(("get", token, session_id)) or get_session(),
        raising=False,
    )
    return state, rendered, calls


def test_student_can_open_historical_template_session_after_page_refresh(monkeypatch):
    def historical():
        return {
            "id": 7, "user_id": 1, "course_id": 2, "class_id": 3,
            "status": "FINALIZED", "cases": [], "limitations": [], "questions": [],
            "final_ideas": [{
                "title": "历史候选", "label": "模板启发／待验证方向", "generation_mode": "template",
                "hypothesis": "待验证", "experiment_method": "隔离对照", "risks": [], "evidence": [],
            }],
        }

    state, rendered, calls = _resume_page_fixture(monkeypatch, get_session=historical)
    innovation.render()
    assert calls == [
        ("list", "isolated-token", {"course_id": 2, "class_id": 3}),
        ("get", "isolated-token", 7),
    ]
    assert state["innovation_session"]["id"] == 7
    assert "生成方式：模板启发" in rendered
    assert "性质：模板启发／待验证方向" in rendered
    assert not any("性质：模型建议" in text for text in rendered)


def test_failed_historical_get_clears_old_session_and_suggestions(monkeypatch):
    state, _, calls = _resume_page_fixture(
        monkeypatch,
        get_session=lambda: (_ for _ in ()).throw(APIClientError("denied", status_code=403)),
    )
    state["innovation_session"] = {"id": 6, "final_ideas": [{"label": "旧结果"}]}
    state["innovation_suggestion_result"] = {"status": "proposed"}
    state["innovation_suggestion_result_scope"] = (1, 2, 3, "innovation_suggestion", "learning-a", 6)
    state["innovation_suggestion_use_course_references"] = True
    innovation.render()
    assert ("get", "isolated-token", 7) in calls
    assert "innovation_session" not in state
    assert "innovation_suggestion_result" not in state
    assert "innovation_suggestion_result_scope" not in state
    assert state["innovation_suggestion_use_course_references"] is False


def test_historical_session_scope_change_clears_previous_result_before_list(monkeypatch):
    state, _, calls = _resume_page_fixture(monkeypatch, get_session=lambda: {})
    state["innovation_session"] = {"id": 6, "final_ideas": [{"label": "旧结果"}]}
    state["innovation_session_scope"] = (999, 2, 3)
    state["innovation_suggestion_result"] = {"status": "proposed"}
    state["innovation_suggestion_use_course_references"] = True
    monkeypatch.setattr(innovation.st, "button", lambda *_args, **_kwargs: False)
    innovation.render()
    assert calls == [("list", "isolated-token", {"course_id": 2, "class_id": 3})]
    assert "innovation_session" not in state
    assert "innovation_suggestion_result" not in state
    assert state["innovation_suggestion_use_course_references"] is False


def test_historical_session_list_error_clears_previous_result(monkeypatch):
    state, _, _ = _resume_page_fixture(monkeypatch, get_session=lambda: {})
    state["innovation_session"] = {"id": 6}
    state["innovation_session_scope"] = (1, 2, 3)
    state["innovation_suggestion_result"] = {"status": "proposed"}
    monkeypatch.setattr(
        innovation.api_client, "list_innovation_sessions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(APIClientError("denied", status_code=403)),
    )
    innovation.render()
    assert "innovation_session" not in state
    assert "innovation_suggestion_result" not in state
