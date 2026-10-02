# -*- coding: utf-8 -*-
"""TuiTaskRunner：线程化事件消费、审批等待与取消（FakeAgent，不调真实 LLM）。"""
import time
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from mycodingagent.tui.runner import (
    AWAITING_DECISION,
    IDLE,
    RUNNING,
    TuiTaskRunner,
)


class FakeAgent:
    """rounds: [(updates, interrupts)]；协议同 events.iter_task_events。"""

    def __init__(self, rounds):
        self._rounds = rounds
        self._i = 0
        self.resume_payloads = []

    def stream(self, payload, config=None, stream_mode=None):
        if isinstance(payload, dict) and "resume" in str(payload):
            self.resume_payloads.append(payload)
        updates, _ = self._rounds[self._i]
        return iter(updates)

    def get_state(self, cfg):
        _, interrupts = self._rounds[self._i]
        self._i += 1
        return SimpleNamespace(interrupts=interrupts)


def _ai(text="done", tool_calls=None, tokens=10):
    return AIMessage(
        content=text,
        tool_calls=tool_calls or [],
        usage_metadata={"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
    )


def _tool():
    return ToolMessage(content="ok", name="execute", tool_call_id="tc1")


def _interrupt(name="execute"):
    return SimpleNamespace(
        value={"action_requests": [{"name": name, "args": {}, "description": "desc"}]},
        id="int-1",
    )


def _wait_for(states, target, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if states and states[-1][0] == target:
            return
        time.sleep(0.01)
    raise AssertionError(f"未等到状态 {target}，实际：{states}")


def _make(agent):
    events, states = [], []
    runner = TuiTaskRunner(
        agent,
        on_event=events.append,
        on_state=lambda s, info=None: states.append((s, info)),
    )
    return runner, events, states


def test_runner_completes_and_idle():
    agent = FakeAgent([
        ([{"n": {"messages": [_ai("完成")]}}], ()),
    ])
    runner, events, states = _make(agent)

    runner.start(thread_id="t1", user_input="干活")
    _wait_for(states, IDLE)

    assert [s[0] for s in states] == [RUNNING, IDLE]
    assert events  # 至少有 summary 事件
    assert not runner.busy


def test_runner_waits_for_decision_then_resumes():
    agent = FakeAgent([
        ([{"n": {"messages": [_ai(tool_calls=[{"name": "execute", "args": {}, "id": "tc1"}])]}}],
         (_interrupt(),)),
        ([{"n": {"messages": [_tool()]}}, {"n": {"messages": [_ai("done")]}}], ()),
    ])
    runner, events, states = _make(agent)

    runner.start(thread_id="t1", user_input="干活")
    _wait_for(states, AWAITING_DECISION)

    assert runner.busy
    runner.submit_decisions([{"type": "approve"}])
    _wait_for(states, IDLE)

    assert states[-1][0] == IDLE
    assert not runner.busy


def test_runner_cancel_returns_idle():
    agent = FakeAgent([
        ([{"n": {"messages": [_ai(tool_calls=[{"name": "execute", "args": {}, "id": "tc1"}])]}}],
         (_interrupt(),)),
    ])
    runner, events, states = _make(agent)

    runner.start(thread_id="t1", user_input="干活")
    _wait_for(states, AWAITING_DECISION)
    runner.submit_decisions(None)
    _wait_for(states, IDLE)

    assert isinstance(states[-1][1], str)
    assert "取消" in states[-1][1]
    assert not runner.busy


def test_runner_start_while_busy_rejected():
    agent = FakeAgent([
        ([{"n": {"messages": [_ai(tool_calls=[{"name": "execute", "args": {}, "id": "tc1"}])]}}],
         (_interrupt(),)),
    ])
    runner, _, _ = _make(agent)
    runner.start(thread_id="t1", user_input="干活")
    try:
        while not runner.busy:
            time.sleep(0.005)
        try:
            runner.start(thread_id="t1", user_input="再来")
        except RuntimeError:
            return
        raise AssertionError("应当拒绝重复启动")
    finally:
        runner.submit_decisions(None)
