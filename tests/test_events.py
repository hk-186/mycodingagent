# -*- coding: utf-8 -*-
"""结构化事件流 + 三重预算测试（阶段 2）。

用 FakeAgent 模拟 agent.stream(stream_mode="updates") 的产出，
验证 events.iter_task_events 的事件转换、预算拦截与 /resume 传参。
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from mycodingagent import config
from mycodingagent.events import (
    ASSISTANT_MESSAGE,
    ERROR,
    STOP_COMPLETED,
    STOP_INTERRUPTED,
    STOP_RECURSION_LIMIT,
    STOP_STEP_LIMIT,
    STOP_TOKEN_LIMIT,
    SUMMARY,
    TOOL_END,
    TOOL_START,
    iter_task_events,
    render_event,
)


# ============================================================
# 测试替身：模拟 LangGraph 图的 updates 流
# ============================================================
class FakeAgent:
    """按预设序列产出 updates，可在末尾抛出指定异常。"""

    def __init__(self, updates, error=None):
        self._updates = list(updates)
        self._error = error
        self.last_payload = None
        self.last_config = None
        self.last_stream_mode = None

    def stream(self, payload, config=None, stream_mode=None):
        self.last_payload = payload
        self.last_config = config
        self.last_stream_mode = stream_mode
        yield from self._updates
        if self._error is not None:
            raise self._error


def model_update(*messages):
    """模型节点更新。"""
    return {"model": {"messages": list(messages)}}


def tools_update(*messages):
    """工具节点更新。"""
    return {"tools": {"messages": list(messages)}}


def ai(content="", tool_calls=None, total_tokens=None):
    kwargs = {}
    if total_tokens is not None:
        kwargs["usage_metadata"] = {
            "input_tokens": total_tokens - 1,
            "output_tokens": 1,
            "total_tokens": total_tokens,
        }
    return AIMessage(content=content, tool_calls=tool_calls or [], **kwargs)


def tool_call(name="execute", args=None, id="c1"):
    return {"name": name, "args": args or {}, "id": id, "type": "tool_call"}


def tool_msg(content="ok", name="execute", id="c1", status="success"):
    return ToolMessage(content=content, name=name, tool_call_id=id, status=status)


def collect(agent, **kwargs):
    return list(iter_task_events(agent, thread_id="t", **kwargs))


# ============================================================
# 事件转换
# ============================================================
def test_event_sequence_normal_round():
    """一轮正常任务：工具调用 → 工具结果 → 最终回答 → summary。"""
    agent = FakeAgent([
        model_update(ai(tool_calls=[tool_call(id="c1")], total_tokens=15)),
        tools_update(tool_msg("4 passed", id="c1")),
        model_update(ai("任务完成，测试通过", total_tokens=25)),
    ])
    events = collect(agent, user_input="修复失败的测试")

    assert [e.type for e in events] == [TOOL_START, TOOL_END, ASSISTANT_MESSAGE, SUMMARY]
    assert events[0].name == "execute"
    assert events[1].text == "4 passed"
    assert events[1].is_error is False
    assert events[2].text == "任务完成，测试通过"

    summary = events[-1]
    assert summary.steps == 2  # 两次模型调用
    assert summary.tokens == 40  # 15 + 25
    assert summary.stopped_reason == STOP_COMPLETED


def test_stream_args_for_new_task():
    agent = FakeAgent([])
    collect(agent, user_input="你好")
    assert agent.last_payload == {"messages": [{"role": "user", "content": "你好"}]}
    assert agent.last_stream_mode == "updates"
    assert agent.last_config["configurable"] == {"thread_id": "t"}
    assert agent.last_config["recursion_limit"] == config.AGENT_RECURSION_LIMIT


def test_resume_passes_none_payload():
    """/resume 续跑：payload 必须是 None（从 checkpoint 断点继续）。"""
    agent = FakeAgent([])
    collect(agent, user_input=None)
    assert agent.last_payload is None


def test_tool_end_error_status():
    agent = FakeAgent([
        model_update(ai(tool_calls=[tool_call(id="c1")])),
        tools_update(tool_msg("已拒绝执行", id="c1", status="error")),
    ])
    events = collect(agent, user_input="x")
    assert events[1].is_error is True
    assert "工具失败" in render_event(events[1])


def test_content_blocks_extracted():
    """content 为 blocks 列表时提取 text 块。"""
    msg = ai(content=[{"type": "text", "text": "你好"}, {"type": "text", "text": "世界"}])
    agent = FakeAgent([model_update(msg)])
    events = collect(agent, user_input="x")
    assert events[0].type == ASSISTANT_MESSAGE
    assert events[0].text == "你好\n世界"


def test_non_dict_and_none_updates_skipped():
    agent = FakeAgent([{"model": None}, {"other": "junk"}, model_update(ai("hi"))])
    events = collect(agent, user_input="x")
    assert [e.type for e in events] == [ASSISTANT_MESSAGE, SUMMARY]


def test_message_without_tool_calls_but_text_only_reports_text():
    """带工具调用时只报 tool_start，不重复输出正文（与旧 CLI 行为一致）。"""
    msg = ai(content="我先看看文件", tool_calls=[tool_call(name="ls", id="c1")])
    agent = FakeAgent([model_update(msg)])
    events = collect(agent, user_input="x")
    assert [e.type for e in events] == [TOOL_START, SUMMARY]
    assert events[0].name == "ls"


# ============================================================
# 三重预算
# ============================================================
def test_step_limit_stops_with_pending_work(monkeypatch):
    """步数达到上限且还有待执行的工具调用 → 安全停下。"""
    monkeypatch.setattr(config, "AGENT_MAX_STEPS", 2)
    agent = FakeAgent([
        model_update(ai(tool_calls=[tool_call(id="c1")], total_tokens=10)),
        tools_update(tool_msg(id="c1")),
        model_update(ai(tool_calls=[tool_call(id="c2")], total_tokens=10)),
        tools_update(tool_msg(id="c2")),
        model_update(ai(tool_calls=[tool_call(id="c3")])),  # 不应被消费到
    ])
    events = collect(agent, user_input="x")

    errors = [e for e in events if e.type == ERROR]
    assert len(errors) == 1
    assert "步数上限" in errors[0].text
    assert "/resume" in errors[0].text

    summary = events[-1]
    assert summary.steps == 2
    assert summary.stopped_reason == STOP_STEP_LIMIT


def test_final_answer_at_step_limit_completes():
    """第 N 步模型直接给出最终回答（无待执行工具）→ 不算超限，正常完成。"""
    monkeypatch_max = 2
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(config, "AGENT_MAX_STEPS", monkeypatch_max)
    try:
        agent = FakeAgent([
            model_update(ai(tool_calls=[tool_call(id="c1")])),
            tools_update(tool_msg(id="c1")),
            model_update(ai("完成了")),  # 步数恰好到上限但是最终回答
        ])
        events = collect(agent, user_input="x")
        assert events[-1].stopped_reason == STOP_COMPLETED
        assert all(e.type != ERROR for e in events)
    finally:
        monkeypatch.undo()


def test_token_budget_stops(monkeypatch):
    monkeypatch.setattr(config, "AGENT_TOKEN_BUDGET", 20)
    agent = FakeAgent([
        model_update(ai(tool_calls=[tool_call(id="c1")], total_tokens=15)),
        tools_update(tool_msg(id="c1")),
        model_update(ai(tool_calls=[tool_call(id="c2")], total_tokens=15)),  # 30 ≥ 20
    ])
    events = collect(agent, user_input="x")

    errors = [e for e in events if e.type == ERROR]
    assert len(errors) == 1
    assert "token" in errors[0].text.lower()
    summary = events[-1]
    assert summary.tokens == 30
    assert summary.stopped_reason == STOP_TOKEN_LIMIT


def test_recursion_limit_becomes_error_event():
    """LangGraph 抛 GraphRecursionError → 转成可恢复的 error 事件。"""
    agent = FakeAgent(
        [model_update(ai(tool_calls=[tool_call(id="c1")]))],
        error=GraphRecursionError("recursion limit reached"),
    )
    events = collect(agent, user_input="x")

    errors = [e for e in events if e.type == ERROR]
    assert len(errors) == 1
    assert "recursion_limit" in errors[0].text
    assert events[-1].stopped_reason == STOP_RECURSION_LIMIT


def test_keyboard_interrupt_becomes_error_event():
    agent = FakeAgent(
        [model_update(ai(tool_calls=[tool_call(id="c1")]))],
        error=KeyboardInterrupt(),
    )
    events = collect(agent, user_input="x")
    assert events[-1].stopped_reason == STOP_INTERRUPTED
    assert any("/resume" in e.text for e in events if e.type == ERROR)


# ============================================================
# 渲染
# ============================================================
def test_render_event_formats():
    from mycodingagent.events import AgentEvent

    assert "[工具调用]" in render_event(AgentEvent(type=TOOL_START, name="ls"))
    assert "助手>" in render_event(AgentEvent(type=ASSISTANT_MESSAGE, text="hi"))
    assert "[!]" in render_event(AgentEvent(type=ERROR, text="oops"))
    assert "工具失败" in render_event(AgentEvent(type=TOOL_END, is_error=True))
    summary_line = render_event(AgentEvent(type=SUMMARY, steps=3, tokens=100))
    assert "本轮统计" in summary_line and "3 步" in summary_line
