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


# ============================================================
# 阶段 3：INTERRUPT 事件 / detect_pending_interrupt / resume_decisions
# ============================================================
from mycodingagent.events import (  # noqa: E402 — 局部 import 避免污染顶部
    INTERRUPT,
    STOP_PENDING_INTERRUPT,
    _action_request_to_dict,
    _hitl_request_action_requests,
    detect_pending_interrupt,
)
from langgraph.types import Command  # noqa: E402


class FakeSnapshot:
    """模拟 langgraph StateSnapshot：只暴露 interrupts 属性。"""

    def __init__(self, interrupts=()):
        self.interrupts = tuple(interrupts)


class FakeInterrupt:
    """模拟 langgraph Interrupt：value 是 HITLRequest dict。"""

    def __init__(self, value, id="i1"):
        self.value = value
        self.id = id


class FakeStateAgent(FakeAgent):
    """扩展 FakeAgent，多一个 get_state() 用于检测 pending interrupt。"""

    def __init__(self, updates, error=None, state_interrupts=()):
        super().__init__(updates, error)
        self._state_interrupts = tuple(state_interrupts)
        self.last_get_state_cfg = None

    def get_state(self, cfg):
        self.last_get_state_cfg = cfg
        return FakeSnapshot(interrupts=self._state_interrupts)


def _hitl_request(action_name="execute", args=None, description=""):
    """构造一个 HITLRequest dict。"""
    return {
        "action_requests": [
            {
                "name": action_name,
                "args": args or {},
                "description": description,
            }
        ],
        "review_configs": [],
    }


# ---- detect_pending_interrupt 纯函数 ----
def test_detect_pending_interrupt_returns_first():
    hitl = _hitl_request()
    snap = FakeSnapshot(interrupts=[
        FakeInterrupt(value=hitl, id="i1"),
        FakeInterrupt(value=hitl, id="i2"),
    ])
    agent = type("A", (), {"get_state": lambda self, cfg: snap})()
    result = detect_pending_interrupt(agent, {})
    assert result is not None
    assert result.id == "i1"


def test_detect_pending_interrupt_empty_returns_none():
    snap = FakeSnapshot(interrupts=())
    agent = type("A", (), {"get_state": lambda self, cfg: snap})()
    assert detect_pending_interrupt(agent, {}) is None


def test_detect_pending_interrupt_handles_exception():
    class BoomAgent:
        def get_state(self, cfg):
            raise RuntimeError("nope")
    assert detect_pending_interrupt(BoomAgent(), {}) is None


def test_detect_pending_interrupt_handles_missing_get_state():
    """agent 没有 get_state 方法时不抛异常，返回 None。"""
    agent = object()
    assert detect_pending_interrupt(agent, {}) is None


# ---- _action_request_to_dict / _hitl_request_action_requests ----
def test_action_request_to_dict_from_dict():
    ar = {"name": "execute", "args": {"command": "ls"}, "description": "test"}
    d = _action_request_to_dict(ar)
    assert d == {"name": "execute", "args": {"command": "ls"}, "description": "test"}


def test_action_request_to_dict_handles_missing_fields():
    d = _action_request_to_dict({})
    assert d == {"name": "", "args": {}, "description": ""}


def test_action_request_to_dict_from_object():
    class Obj:
        name = "execute"
        args = {"command": "ls"}
        description = "test"
    d = _action_request_to_dict(Obj())
    assert d["name"] == "execute"
    assert d["args"] == {"command": "ls"}
    assert d["description"] == "test"


def test_hitl_request_action_requests_from_dict():
    hitl = _hitl_request(action_name="git_commit", args={"message": "fix"}, description="提交")
    assert len(_hitl_request_action_requests(FakeInterrupt(value=hitl))) == 1
    ar = _hitl_request_action_requests(FakeInterrupt(value=hitl))[0]
    assert ar["name"] == "git_commit"
    assert ar["args"] == {"message": "fix"}


def test_hitl_request_action_requests_empty_value():
    """value 为 None 或缺 action_requests 时返回空列表。"""
    assert _hitl_request_action_requests(FakeInterrupt(value=None)) == []
    assert _hitl_request_action_requests(FakeInterrupt(value={})) == []


# ---- iter_task_events 检测 pending interrupt 后产出 INTERRUPT ----
def test_pending_interrupt_produces_interrupt_event_not_summary():
    """stream 自然结束后检测到 pending interrupt → 产出 INTERRUPT，不产 summary。"""
    hitl = _hitl_request(
        action_name="execute",
        args={"command": "rm -rf /"},
        description="即将执行：rm -rf /",
    )
    agent = FakeStateAgent(
        updates=[model_update(ai("好的，开始"))],
        state_interrupts=[FakeInterrupt(value=hitl, id="int-1")],
    )
    events = collect(agent, user_input="x")
    assert [e.type for e in events] == [ASSISTANT_MESSAGE, INTERRUPT]
    ev = events[-1]
    assert ev.type == INTERRUPT
    assert ev.interrupt_id == "int-1"
    assert len(ev.action_requests) == 1
    assert ev.action_requests[0]["name"] == "execute"
    assert ev.action_requests[0]["args"]["command"] == "rm -rf /"
    assert "即将执行" in ev.text


def test_no_pending_interrupt_produces_summary():
    """stream 结束 + 无 pending interrupt → 正常产出 summary。"""
    agent = FakeStateAgent(
        updates=[model_update(ai("完成", total_tokens=10))],
        state_interrupts=[],
    )
    events = collect(agent, user_input="x")
    assert events[-1].type == SUMMARY


def test_multiple_action_requests_in_interrupt_event():
    """多个 action_request 都应出现在 INTERRUPT 事件中。"""
    hitl = {
        "action_requests": [
            {"name": "execute", "args": {"command": "rm x"}, "description": "删 x"},
            {"name": "git_commit", "args": {"message": "fix"}, "description": "提交"},
        ],
        "review_configs": [],
    }
    agent = FakeStateAgent(
        updates=[model_update(ai("ok"))],
        state_interrupts=[FakeInterrupt(value=hitl, id="i-multi")],
    )
    events = collect(agent, user_input="x")
    ev = events[-1]
    assert ev.type == INTERRUPT
    assert len(ev.action_requests) == 2
    assert ev.action_requests[0]["name"] == "execute"
    assert ev.action_requests[1]["name"] == "git_commit"


# ---- resume_decisions 参数 ----
def test_resume_decisions_payload_is_command():
    """resume_decisions 传入 → payload 是 Command(resume={"decisions": ...})。"""
    agent = FakeAgent([])  # 立即结束
    collect(agent, resume_decisions=[{"type": "approve"}])
    payload = agent.last_payload
    assert isinstance(payload, Command)
    resume = getattr(payload, "resume", None)
    assert resume is not None
    assert resume["decisions"] == [{"type": "approve"}]


def test_resume_decisions_reject_with_message():
    agent = FakeAgent([])
    collect(agent, resume_decisions=[{"type": "reject", "message": "不行"}])
    payload = agent.last_payload
    assert isinstance(payload, Command)
    assert payload.resume["decisions"] == [{"type": "reject", "message": "不行"}]


def test_resume_decisions_passes_config():
    """resume 时 config 仍包含 thread_id 和 recursion_limit。"""
    agent = FakeAgent([])
    collect(agent, resume_decisions=[{"type": "approve"}])
    assert agent.last_config["configurable"]["thread_id"] == "t"
    assert agent.last_config["recursion_limit"] == config.AGENT_RECURSION_LIMIT


def test_resume_decisions_exclusive_with_user_input():
    """resume_decisions 与 user_input 互斥；resume 优先（不构造 messages payload）。"""
    agent = FakeAgent([])
    collect(agent, user_input="不应使用", resume_decisions=[{"type": "approve"}])
    payload = agent.last_payload
    assert isinstance(payload, Command)
    # 不应是新 user input 的 dict 形式
    assert not (isinstance(payload, dict) and "messages" in payload)


# ---- render_event INTERRUPT 渲染 ----
def test_render_interrupt_event():
    from mycodingagent.events import AgentEvent
    ev = AgentEvent(
        type=INTERRUPT,
        text="即将执行 rm -rf /",
        action_requests=[
            {
                "name": "execute",
                "args": {"command": "rm -rf /"},
                "description": "即将执行：rm -rf /",
            }
        ],
        interrupt_id="int-1",
    )
    rendered = render_event(ev)
    assert "[审批请求]" in rendered
    assert "execute" in rendered
    assert "rm -rf /" in rendered
    assert "/approve" in rendered
    assert "/reject" in rendered
    assert "/respond" in rendered
    assert "1 个待决策" in rendered


def test_render_interrupt_empty_action_requests():
    from mycodingagent.events import AgentEvent
    ev = AgentEvent(type=INTERRUPT, text="")
    rendered = render_event(ev)
    assert "[审批请求]" in rendered
    assert "无 action_request" in rendered


def test_render_interrupt_multiple_action_requests():
    from mycodingagent.events import AgentEvent
    ev = AgentEvent(
        type=INTERRUPT,
        text="",
        action_requests=[
            {"name": "execute", "args": {"command": "rm x"}, "description": ""},
            {"name": "git_commit", "args": {"message": "fix"}, "description": "提交"},
        ],
    )
    rendered = render_event(ev)
    assert "1." in rendered and "2." in rendered
    assert "execute" in rendered
    assert "git_commit" in rendered
    assert "2 个待决策" in rendered


def test_summary_with_pending_interrupt_reason():
    from mycodingagent.events import AgentEvent
    line = render_event(AgentEvent(type=SUMMARY, stopped_reason=STOP_PENDING_INTERRUPT))
    assert "等待 HITL 决策" in line
