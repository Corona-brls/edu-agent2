import pytest
from pydantic import ValidationError

from backend.agent.schemas import ActionDecision, AgentAction, ToolSpec
from backend.llm.deepseek_provider import DeepSeekProvider

from tests.test_llm_provider import FakeClient, FakeCompletions


@pytest.mark.parametrize(
    "payload,expected",
    [
        (
            '{"action":"retrieve","arguments":{"query":"NAS"},"reason":"需要先检索","task_complete":false}',
            AgentAction.RETRIEVE,
        ),
        (
            '{"action":"call_tool","tool_name":"course_retriever","arguments":{"query":"NAS"},"reason":"调用课程检索","task_complete":false}',
            AgentAction.CALL_TOOL,
        ),
        (
            '{"action":"ask_user","arguments":{"question":"请补充实验环境"},"reason":"信息不足","task_complete":false}',
            AgentAction.ASK_USER,
        ),
        (
            '{"action":"answer","arguments":{"answer":"已有证据"},"reason":"可以回答","task_complete":true}',
            AgentAction.ANSWER,
        ),
        (
            '{"action":"abstain","arguments":{"message":"证据不足"},"reason":"无证据","task_complete":true}',
            AgentAction.ABSTAIN,
        ),
    ],
)
def test_deepseek_decides_all_allowed_actions(monkeypatch, payload, expected):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = DeepSeekProvider(client=FakeClient(FakeCompletions([payload])))

    decision = provider.decide_next_action(
        state_summary="学生问题：NAS",
        available_tools=[
            ToolSpec(
                name="course_retriever",
                description="检索课程知识库",
                network=False,
                read_only=True,
            )
        ],
    )

    assert decision.action == expected


def test_tool_spec_accepts_policy_fields_from_gateway_catalog():
    spec = ToolSpec.model_validate(
        {
            "name": "course_retriever",
            "description": "检索课程知识库",
            "input_schema": {"type": "object"},
            "risk": "LOW",
            "network": False,
            "read_only": True,
            "enabled": True,
        }
    )

    assert spec.network is False
    assert spec.read_only is True


def test_action_decision_rejects_unknown_action():
    with pytest.raises(ValidationError):
        ActionDecision.model_validate(
            {
                "action": "delete_file",
                "arguments": {},
                "reason": "bad",
                "task_complete": False,
            }
        )


def test_action_decision_rejects_call_tool_without_tool_name():
    with pytest.raises(ValidationError):
        ActionDecision.model_validate(
            {
                "action": "call_tool",
                "arguments": {"query": "NAS"},
                "reason": "missing tool",
                "task_complete": False,
            }
        )


def test_action_decision_rejects_invalid_arguments():
    with pytest.raises(ValidationError):
        ActionDecision.model_validate(
            {
                "action": "retrieve",
                "arguments": ["not", "a", "dict"],
                "reason": "bad args",
                "task_complete": False,
            }
        )


def test_action_decision_rejects_ask_user_without_question():
    with pytest.raises(ValidationError):
        ActionDecision.model_validate(
            {
                "action": "ask_user",
                "arguments": {},
                "reason": "missing question",
                "task_complete": False,
            }
        )


def test_deepseek_only_returns_decision_without_executing_tool(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    fake = FakeCompletions(
        [
            '{"action":"call_tool","tool_name":"course_retriever","arguments":{"query":"NAS"},"reason":"需要工具","task_complete":false}'
        ]
    )
    provider = DeepSeekProvider(client=FakeClient(fake))

    decision = provider.decide_next_action(
        state_summary="需要课程检索",
        available_tools=[ToolSpec(name="course_retriever", description="检索课程知识库")],
    )

    assert decision.action == AgentAction.CALL_TOOL
    assert fake.calls[0]["messages"]
    assert len(fake.calls) == 1
