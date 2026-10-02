# -*- coding: utf-8 -*-
"""AgentApp：Textual App 级测试（asyncio.run + run_test，无需 pytest-asyncio）。"""
import asyncio
import time
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage
from textual.widgets import Input, RichLog

from mycodingagent.tui.app import AgentApp
from mycodingagent.tui.runner import AWAITING_DECISION, IDLE


class FakeAgent:
    """rounds: [(updates, interrupts)]。"""

    def __init__(self, rounds):
        self._rounds = rounds
        self._i = 0

    def stream(self, payload, config=None, stream_mode=None):
        updates, _ = self._rounds[self._i]
        return iter(updates)

    def get_state(self, cfg):
        _, interrupts = self._rounds[self._i]
        self._i += 1
        return SimpleNamespace(interrupts=interrupts)


def _ai(text="done", tool_calls=None, reasoning=None):
    kwargs = {}
    if reasoning:
        kwargs["additional_kwargs"] = {"reasoning_content": reasoning}
    return AIMessage(
        content=text,
        tool_calls=tool_calls or [],
        usage_metadata={"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
        **kwargs,
    )


def _interrupt(name="execute"):
    return SimpleNamespace(
        value={"action_requests": [{"name": name, "args": {}, "description": "desc"}]},
        id="int-1",
    )


def _tool():
    return ToolMessage(content="ok", name="execute", tool_call_id="tc1")


def _log_text(app):
    return "\n".join(strip.text for strip in app.query_one("#log", RichLog).lines)


def _submit(app, text):
    inp = app.query_one("#input", Input)
    app.on_input_submitted(Input.Submitted(inp, text))


async def _wait_idle(app, pilot, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app._mode == IDLE:
            return
        await pilot.pause(0.05)
    raise AssertionError("任务未在时限内回到 idle")


def test_app_help():
    async def scenario():
        app = AgentApp(FakeAgent([]), backend=SimpleNamespace())
        async with app.run_test() as pilot:
            _submit(app, "/help")
            await pilot.pause()
            assert "斜杠命令" in _log_text(app)

    asyncio.run(scenario())


def test_app_unknown_command():
    async def scenario():
        app = AgentApp(FakeAgent([]), backend=SimpleNamespace())
        async with app.run_test() as pilot:
            _submit(app, "/nope")
            await pilot.pause()
            assert "未知命令" in _log_text(app)

    asyncio.run(scenario())


def test_app_runs_task_to_completion():
    agent = FakeAgent([
        ([{"n": {"messages": [_ai("搞定")]}}], ()),
    ])

    async def scenario():
        app = AgentApp(agent, backend=SimpleNamespace())
        async with app.run_test() as pilot:
            _submit(app, "帮我修个 bug")
            # 先确认任务确实启动（初始 mode 也是 idle，不能直接等 idle）
            while app._mode == IDLE:
                await pilot.pause(0.01)
            await _wait_idle(app, pilot)
            text = _log_text(app)
            assert "你> 帮我修个 bug" in text
            assert "本轮统计" in text

    asyncio.run(scenario())


def test_app_reasoning_collapsed_by_default_and_toggle():
    """思考过程默认折叠为占位行，按 t 展开全文，再按 t 收回。"""
    agent = FakeAgent([
        ([{"n": {"messages": [_ai("最终答案", reasoning="让我想想这个任务怎么做")]} }], ()),
    ])

    async def scenario():
        app = AgentApp(agent, backend=SimpleNamespace())
        async with app.run_test() as pilot:
            _submit(app, "做个任务")
            while app._mode == IDLE:
                await pilot.pause(0.01)
            await _wait_idle(app, pilot)
            text = _log_text(app)
            # 默认折叠：占位行 + 不含全文
            assert "已折叠" in text
            assert "让我想想这个任务怎么做" not in text
            # 展开
            app.action_toggle_reasoning()
            await pilot.pause()
            text = _log_text(app)
            assert "让我想想这个任务怎么做" in text
            # 再收起
            app.action_toggle_reasoning()
            await pilot.pause()
            text = _log_text(app)
            assert "已折叠" in text
            assert "让我想想这个任务怎么做" not in text

    asyncio.run(scenario())


def test_app_approval_flow():
    agent = FakeAgent([
        ([{"n": {"messages": [_ai(tool_calls=[{"name": "execute", "args": {}, "id": "tc1"}])]}}],
         (_interrupt(),)),
        ([{"n": {"messages": [_tool()]}}, {"n": {"messages": [_ai("done")]}}], ()),
    ])

    async def scenario():
        app = AgentApp(agent, backend=SimpleNamespace())
        async with app.run_test() as pilot:
            _submit(app, "跑个命令")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and app._mode != AWAITING_DECISION:
                await pilot.pause(0.02)
            assert app._mode == AWAITING_DECISION

            # 无效决策不推进
            _submit(app, "/乱输")
            await pilot.pause()
            assert app._mode == AWAITING_DECISION
            assert "无效决策" in _log_text(app)

            # 批准后跑完
            _submit(app, "/approve")
            await _wait_idle(app, pilot)

    asyncio.run(scenario())


def test_app_cancel_approval():
    agent = FakeAgent([
        ([{"n": {"messages": [_ai(tool_calls=[{"name": "execute", "args": {}, "id": "tc1"}])]}}],
         (_interrupt(),)),
    ])

    async def scenario():
        app = AgentApp(agent, backend=SimpleNamespace())
        async with app.run_test() as pilot:
            _submit(app, "跑个命令")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and app._mode != AWAITING_DECISION:
                await pilot.pause(0.02)
            _submit(app, "/cancel")
            await _wait_idle(app, pilot)
            assert "取消" in _log_text(app)

    asyncio.run(scenario())


# ============================================================
# 会话管理命令（文本渲染）
# ============================================================
class StatefulAgent:
    """按 tid 持有 StateSnapshot 的假 agent（list_sessions 用）。"""

    def __init__(self, states, default_rounds=None):
        self._states = states

    def get_state(self, cfg):
        tid = cfg["configurable"]["thread_id"]
        return self._states.get(tid)

    def stream(self, payload, config=None, stream_mode=None):
        return iter([])


class FakeCheckpointer:
    def __init__(self, conn):
        self.conn = conn


class FakeStore:
    def __init__(self, items):
        self._items = items

    def search(self, namespace):
        return list(self._items)


def _state(msg_count, next_nodes=()):
    from langchain_core.messages import HumanMessage

    messages = [HumanMessage(content="你好")] * msg_count
    return SimpleNamespace(
        values={"messages": messages},
        next=tuple(next_nodes),
        metadata={"step": 3},
        created_at="2026-10-01T10:00:00",
    )


def _checkpointer(thread_ids):
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE checkpoints (thread_id TEXT)")
    conn.execute("CREATE TABLE writes (thread_id TEXT)")
    for tid in thread_ids:
        conn.execute("INSERT INTO checkpoints (thread_id) VALUES (?)", (tid,))
    return FakeCheckpointer(conn)


def test_app_sessions_without_checkpointer():
    async def scenario():
        app = AgentApp(StatefulAgent({}), backend=SimpleNamespace())
        async with app.run_test() as pilot:
            _submit(app, "/sessions")
            await pilot.pause()
            assert "没有任何历史会话" in _log_text(app)

    asyncio.run(scenario())


def test_app_sessions_lists_threads():
    from mycodingagent.tui.app import SessionSelectScreen

    states = {
        "main": _state(2),
        "sess-b": _state(1),
    }
    agent = StatefulAgent(states)

    async def scenario():
        app = AgentApp(
            agent,
            backend=SimpleNamespace(),
            checkpointer=_checkpointer(["main", "sess-b"]),
        )
        async with app.run_test() as pilot:
            _submit(app, "/sessions")
            await pilot.pause()
            assert isinstance(app.screen, SessionSelectScreen)

    asyncio.run(scenario())


def test_app_session_picker_dismiss_switches():
    from mycodingagent.tui.app import SessionSelectScreen

    states = {"main": _state(2), "sess-b": _state(1)}
    agent = StatefulAgent(states)

    async def scenario():
        app = AgentApp(
            agent,
            backend=SimpleNamespace(),
            checkpointer=_checkpointer(["main", "sess-b"]),
        )
        async with app.run_test() as pilot:
            _submit(app, "/sessions")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SessionSelectScreen)
            # 直接触发 dismiss，模拟用户选中 sess-b
            screen.dismiss("sess-b")
            await pilot.pause()
            assert app._session_id == "sess-b"
            assert "已切换到会话" in _log_text(app)

    asyncio.run(scenario())


def test_app_new_session():
    async def scenario():
        app = AgentApp(StatefulAgent({}), backend=SimpleNamespace())
        async with app.run_test() as pilot:
            _submit(app, "/new my-work")
            await pilot.pause()
            assert app._session_id == "my-work"
            assert "my-work" in _log_text(app)

    asyncio.run(scenario())


def test_app_delete_opens_picker():
    from mycodingagent.tui.app import SessionSelectScreen

    states = {"main": _state(2), "sess-b": _state(1)}
    agent = StatefulAgent(states)
    cp = _checkpointer(["main", "sess-b"])

    async def scenario():
        app = AgentApp(
            agent,
            backend=SimpleNamespace(),
            checkpointer=cp,
        )
        async with app.run_test() as pilot:
            _submit(app, "/delete")
            await pilot.pause()
            assert isinstance(app.screen, SessionSelectScreen)
            assert "删除" in app.screen._title
            # dismiss 选中 sess-b
            app.screen.dismiss("sess-b")
            await pilot.pause()
            assert "已删除会话: sess-b" in _log_text(app)

    asyncio.run(scenario())
    rows = cp.conn.execute(
        "SELECT DISTINCT thread_id FROM checkpoints"
    ).fetchall()
    assert [r[0] for r in rows] == ["main"]


def test_app_delete_by_index():
    states = {"main": _state(2), "sess-b": _state(1)}
    agent = StatefulAgent(states)
    cp = _checkpointer(["main", "sess-b"])

    async def scenario():
        app = AgentApp(
            agent,
            backend=SimpleNamespace(),
            checkpointer=cp,
        )
        async with app.run_test() as pilot:
            _submit(app, "/delete 1")  # candidates 排除当前会话 main，序号 1 = sess-b
            await pilot.pause()
            assert app._session_id == "main"  # 当前会话不变
            assert "已删除会话: sess-b" in _log_text(app)

    asyncio.run(scenario())
    # 真实落库验证：checkpoints 里 sess-b 已删除
    rows = cp.conn.execute(
        "SELECT DISTINCT thread_id FROM checkpoints"
    ).fetchall()
    assert [r[0] for r in rows] == ["main"]


def test_app_delete_current_session_blocked():
    agent = StatefulAgent({"main": _state(1)})

    async def scenario():
        app = AgentApp(
            agent,
            backend=SimpleNamespace(),
            checkpointer=_checkpointer(["main"]),
        )
        async with app.run_test() as pilot:
            _submit(app, "/delete main")
            await pilot.pause()
            assert "不能删除当前会话" in _log_text(app)

    asyncio.run(scenario())


def test_app_delete_not_found():
    agent = StatefulAgent({"main": _state(1)})

    async def scenario():
        app = AgentApp(
            agent,
            backend=SimpleNamespace(),
            checkpointer=_checkpointer(["main"]),
        )
        async with app.run_test() as pilot:
            _submit(app, "/delete nope")
            await pilot.pause()
            assert "找不到会话" in _log_text(app)

    asyncio.run(scenario())


def test_app_memory():
    store = FakeStore([SimpleNamespace(key="name", value={"value": "测试用户"})])

    async def scenario():
        app = AgentApp(StatefulAgent({}), backend=SimpleNamespace(), store=store)
        async with app.run_test() as pilot:
            _submit(app, "/memory")
            await pilot.pause()
            text = _log_text(app)
            assert "测试用户" in text and "name" in text

    asyncio.run(scenario())


def test_app_history():
    agent = StatefulAgent({"main": _state(4)})

    async def scenario():
        app = AgentApp(agent, backend=SimpleNamespace())
        async with app.run_test() as pilot:
            _submit(app, "/history")
            await pilot.pause()
            assert "4 条消息" in _log_text(app)

    asyncio.run(scenario())


def test_app_session_cmds_blocked_while_running():
    from mycodingagent.tui.runner import RUNNING

    async def scenario():
        app = AgentApp(StatefulAgent({}), backend=SimpleNamespace())
        async with app.run_test() as pilot:
            app._mode = RUNNING
            _submit(app, "/sessions")
            await pilot.pause()
            assert "任务执行中" in _log_text(app)

    asyncio.run(scenario())
