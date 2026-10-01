# -*- coding: utf-8 -*-
"""eval runner：自动 approve 循环与单任务沙箱流程（FakeAgent，不调真实 LLM）。"""
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from mycodingagent import config
from mycodingagent.eval.loader import EvalTask
from mycodingagent.eval.runner import (
    STOP_TIMEOUT,
    _configure_langsmith,
    _restore_langsmith_env,
    auto_approve_loop,
    run_one,
)


# ============================================================
# FakeAgent：按预排剧本响应 stream / get_state（协议同 events.iter_task_events）
# ============================================================
class FakeAgent:
    """rounds: list of (updates, interrupts)。每轮 stream 产出 updates，
    stream 结束后 get_state 返回 interrupts（空 tuple 表示无审批中断）。"""

    def __init__(self, rounds):
        self._rounds = rounds
        self._i = 0
        self.payloads = []  # 记录每次 stream 收到的 payload

    def stream(self, payload, config=None, stream_mode=None):
        self.payloads.append(payload)
        updates, _ = self._rounds[self._i]
        return iter(updates)

    def get_state(self, cfg):
        _, interrupts = self._rounds[self._i]
        self._i += 1
        return SimpleNamespace(interrupts=interrupts)


def _ai(text="", tool_calls=None, tokens=10):
    return AIMessage(
        content=text,
        tool_calls=tool_calls or [],
        usage_metadata={"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
    )


def _tool(name="execute"):
    return ToolMessage(content="ok", name=name, tool_call_id="tc1")


def _interrupt(name, args=None):
    return SimpleNamespace(
        value={"action_requests": [{"name": name, "args": args or {}, "description": "desc"}]},
        id=f"int-{name}",
    )


def _task(tmp_path, graders=None, reply="代答文案"):
    fixture = tmp_path / "fixture"
    fixture.mkdir(exist_ok=True)
    return EvalTask(
        name="demo",
        type="bugfix",
        prompt="做任务",
        task_dir=tmp_path,
        fixture_dir=fixture,
        graders=graders if graders is not None else [],
        timeout_seconds=600,
        ask_user_reply=reply,
    )


# ============================================================
# auto_approve_loop
# ============================================================
def test_loop_completes_without_interrupt(tmp_path):
    agent = FakeAgent([
        ([
            {"n": {"messages": [_ai(tool_calls=[{"name": "execute", "args": {}, "id": "tc1"}])]}},
            {"n": {"messages": [_tool()]}},
            {"n": {"messages": [_ai("任务完成汇报", tokens=20)]}},
        ], ()),
    ])
    outcome = auto_approve_loop(agent, _task(tmp_path), deadline=time.monotonic() + 60)
    assert outcome.stopped_reason == "completed"
    assert outcome.final_report == "任务完成汇报"
    assert outcome.steps == 2  # 两条 AI 消息
    assert outcome.tokens == 30


def test_loop_auto_approves_execute_interrupt(tmp_path):
    agent = FakeAgent([
        ([{"n": {"messages": [_ai(tool_calls=[{"name": "execute", "args": {"command": "x"}, "id": "tc1"}])]}}],
         (_interrupt("execute"),)),
        ([{"n": {"messages": [_tool()]}}, {"n": {"messages": [_ai("done")]}}], ()),
    ])
    outcome = auto_approve_loop(agent, _task(tmp_path), deadline=time.monotonic() + 60)
    assert outcome.stopped_reason == "completed"
    # 第二轮 payload 必须是 Command(resume={"decisions": [{"type": "approve"}]})
    resume = agent.payloads[1]
    assert isinstance(resume, Command)
    assert resume.resume["decisions"] == [{"type": "approve"}]


def test_loop_auto_responds_ask_user(tmp_path):
    agent = FakeAgent([
        ([{"n": {"messages": [_ai(tool_calls=[{"name": "ask_user", "args": {}, "id": "tc1"}])]}}],
         (_interrupt("ask_user"),)),
        ([{"n": {"messages": [_tool("ask_user")]}}, {"n": {"messages": [_ai("done")]}}], ()),
    ])
    outcome = auto_approve_loop(
        agent, _task(tmp_path, reply="按最合理的来"), deadline=time.monotonic() + 60
    )
    assert outcome.stopped_reason == "completed"
    decisions = agent.payloads[1].resume["decisions"]
    assert decisions == [{"type": "respond", "message": "按最合理的来"}]


def test_loop_timeout_gives_up(tmp_path):
    agent = FakeAgent([
        ([{"n": {"messages": [_ai("hi")]}}], ()),
    ])
    outcome = auto_approve_loop(agent, _task(tmp_path), deadline=time.monotonic() - 1)
    assert outcome.stopped_reason == STOP_TIMEOUT
    assert "时限" in outcome.error


class _BlockingAgent(FakeAgent):
    """stream 产出事件后长时间无动静（模拟子代理执行期间主图无事件）。"""

    def stream(self, payload, config=None, stream_mode=None):
        self.payloads.append(payload)
        updates, _ = self._rounds[self._i]

        def gen():
            yield from updates
            time.sleep(30)  # 阻塞：远超测试用的 deadline

        return gen()


def test_loop_timeout_fires_even_when_stream_blocked(tmp_path):
    """事件流阻塞时 wall-clock 超时也必须生效（子线程消费 + 限时收事件）。"""
    agent = _BlockingAgent([
        ([{"n": {"messages": [_ai("开始干活")]}}], ()),
    ])
    started = time.monotonic()
    outcome = auto_approve_loop(agent, _task(tmp_path), deadline=started + 2)
    assert outcome.stopped_reason == STOP_TIMEOUT
    assert time.monotonic() - started < 10  # 不会等满 sleep(30)


# ============================================================
# run_one 完整流程（真实沙箱 + git + command grader，FakeAgent）
# ============================================================
def test_run_one_pass_and_restore_project_dir(tmp_path):
    task = _task(tmp_path, graders=[{"type": "file_exists", "path": "a.py"}])
    task.fixture_dir.joinpath("a.py").write_text("x = 1\n", encoding="utf-8")
    agent = FakeAgent([([{"n": {"messages": [_ai("搞定")]}}], ())])

    original_dir = config.PROJECT_DIR
    result = run_one(task, build_agent=lambda *a, **kw: agent)

    assert result.passed
    assert result.name == "demo"
    assert result.final_report == "搞定"
    assert result.tokens == 10
    assert config.PROJECT_DIR == original_dir  # 全局状态已恢复
    assert result.elapsed_seconds >= 0


def test_run_one_grader_fail(tmp_path):
    task = _task(tmp_path, graders=[{"type": "file_exists", "path": "missing.py"}])
    agent = FakeAgent([([{"n": {"messages": [_ai("done")]}}], ())])
    result = run_one(task, build_agent=lambda *a, **kw: agent)
    assert not result.passed
    assert result.grader_results[0].type == "file_exists"


def test_run_one_restores_project_dir_on_grader_error(tmp_path):
    task = _task(tmp_path, graders=[{"type": "unknown-grader"}])
    agent = FakeAgent([([{"n": {"messages": [_ai("done")]}}], ())])
    original_dir = config.PROJECT_DIR
    with pytest.raises(Exception):
        run_one(task, build_agent=lambda *a, **kw: agent)
    assert config.PROJECT_DIR == original_dir


def test_run_one_overrides_max_retries_and_restores(tmp_path):
    """eval 期间 LLM_MAX_RETRIES 被临时覆盖为 EVAL_LLM_MAX_RETRIES，结束恢复。"""
    task = _task(tmp_path, graders=[{"type": "file_exists", "path": "a.py"}])
    task.fixture_dir.joinpath("a.py").write_text("x = 1\n", encoding="utf-8")
    agent = FakeAgent([([{"n": {"messages": [_ai("done")]}}], ())])

    seen = {}

    def spy_build(*a, **kw):
        seen["retries"] = config.LLM_MAX_RETRIES  # build_agent 时读取生效值
        return agent

    original = config.LLM_MAX_RETRIES
    result = run_one(task, build_agent=spy_build)

    assert seen["retries"] == config.EVAL_LLM_MAX_RETRIES
    assert config.LLM_MAX_RETRIES == original  # 结束后恢复
    assert result.passed


def test_langsmith_disabled_leaves_env_untouched(tmp_path, monkeypatch):
    """默认不开 LangSmith：不改 LANGCHAIN_* 环境变量。"""
    monkeypatch.setattr(config, "EVAL_LANGSMITH", False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    assert _configure_langsmith(_task(tmp_path)) == {}
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_langsmith_enabled_sets_project_and_restores(tmp_path, monkeypatch):
    """开启时强制 tracing=true 并设置 project；用后恢复原 env。"""
    monkeypatch.setattr(config, "EVAL_LANGSMITH", True)
    monkeypatch.setattr(config, "EVAL_LANGSMITH_PROJECT", "eval-proj")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "old-proj")

    previous = _configure_langsmith(_task(tmp_path))
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "eval-proj"
    assert previous["__project__"] == "eval-proj"

    _restore_langsmith_env(previous)
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert os.environ["LANGCHAIN_PROJECT"] == "old-proj"


def test_run_one_timeout_not_passed_even_if_graders_ok(tmp_path):
    """超时任务即使文件已改对（grader 全过）也不算 pass——流程没走完。

    时限给 8s：全量跑负载高时沙箱准备（git init）也可能花掉 1-2s，
    时限太小会在进入事件循环前就过期，走不到 timeout 路径。
    """
    task = _task(tmp_path, graders=[{"type": "file_exists", "path": "a.py"}])
    task.fixture_dir.joinpath("a.py").write_text("x = 1\n", encoding="utf-8")
    # 事件流阻塞（模拟子代理长调用），触发 wall-clock 超时
    agent = _BlockingAgent([([{"n": {"messages": [_ai("开始")]}}], ())])
    task = EvalTask(**{**task.__dict__, "timeout_seconds": 8})
    result = run_one(task, build_agent=lambda *a, **kw: agent)
    assert not result.passed
    assert result.stopped_reason == STOP_TIMEOUT
    assert result.elapsed_seconds < 30  # 没等满 _BlockingAgent 的 sleep
    assert result.grader_results[0].passed  # 文件检查确实过了
