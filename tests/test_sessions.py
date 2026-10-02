# -*- coding: utf-8 -*-
"""sessions 模块测试：会话列表查询、序号解析、格式化输出。

测试策略：
- resolve_switch_arg / format_sessions 是纯函数，直接构造 SessionInfo 列表测；
- list_sessions 涉及 sqlite + get_state 回调，用内存 sqlite + Mock 回调测。
"""

import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from mycodingagent.sessions import (
    SessionInfo,
    delete_session,
    format_sessions,
    list_sessions,
    resolve_switch_arg,
)


# ============================================================
# 辅助：构造一个假 StateSnapshot（duck typing 即可）
# ============================================================
def _fake_state(
    *,
    next_: tuple = (),
    messages: list | None = None,
    step: int = 0,
    created_at: str = "2026-09-30 12:00:00",
):
    """模拟 langgraph.types.StateSnapshot 的最小可用子集。"""
    return SimpleNamespace(
        values={"messages": messages or []},
        next=next_,
        metadata={"step": step},
        created_at=created_at,
    )


class HumanMessage:
    """模拟 langchain HumanMessage（sessions.py 用 __class__.__name__ 判断类型）。"""
    def __init__(self, content):
        self.content = content


class AIMessage:
    """模拟 langchain AIMessage。"""
    def __init__(self, content):
        self.content = content


def _fake_human_msg(content: str) -> HumanMessage:
    return HumanMessage(content)


# ============================================================
# resolve_switch_arg：序号 / 精确 / 前缀匹配
# ============================================================
class TestResolveSwitchArg:
    @pytest.fixture
    def sessions(self):
        return [
            SessionInfo("main", "2026-09-30 12:00", 10, 5, "waiting", "你好"),
            SessionInfo("session-20260928-214144", "2026-09-28 21:42", 3, 2, "interrupted", "修 bug"),
            SessionInfo("session-20260929-211548", "2026-09-29 21:15", 5, 4, "waiting", "加函数"),
        ]

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            ("1", "main"),
            ("2", "session-20260928-214144"),
            ("3", "session-20260929-211548"),
        ],
    )
    def test_index_match(self, sessions, arg, expected):
        assert resolve_switch_arg(arg, sessions) == expected

    @pytest.mark.parametrize("arg", ["0", "4", "99", "-1"])
    def test_index_out_of_range(self, sessions, arg):
        # 负数 arg.isdigit() 为 False，会落到精确匹配 → 也返回 None
        assert resolve_switch_arg(arg, sessions) is None

    def test_exact_match(self, sessions):
        assert resolve_switch_arg("main", sessions) == "main"
        assert resolve_switch_arg("session-20260928-214144", sessions) == "session-20260928-214144"

    def test_prefix_unique_match(self, sessions):
        # "session-2026092" 同时匹配两个 → None
        assert resolve_switch_arg("session-2026092", sessions) is None
        # "session-20260928" 唯一匹配
        assert resolve_switch_arg("session-20260928", sessions) == "session-20260928-214144"

    def test_empty_arg(self, sessions):
        assert resolve_switch_arg("", sessions) is None
        assert resolve_switch_arg("   ", sessions) is None

    def test_no_match(self, sessions):
        assert resolve_switch_arg("nonexistent", sessions) is None

    def test_empty_sessions(self):
        assert resolve_switch_arg("1", []) is None
        assert resolve_switch_arg("main", []) is None


# ============================================================
# format_sessions：空列表 / 多会话格式
# ============================================================
class TestFormatSessions:
    def test_empty(self):
        assert format_sessions([]) == "[没有任何历史会话]"

    def test_single_session(self):
        s = [SessionInfo("main", "2026-09-30 12:00", 5, 3, "waiting", "你好")]
        out = format_sessions(s)
        assert "历史会话：" in out
        assert "main" in out
        assert "等待输入" in out
        assert "你好" in out

    def test_multiple_sessions_with_status_labels(self):
        sessions = [
            SessionInfo("main", "2026-09-30 12:00", 5, 3, "waiting", "你好"),
            SessionInfo("bug-fix", "2026-09-29 10:00", 10, 6, "interrupted", "修 utils.py"),
            SessionInfo("empty", "2026-09-28 09:00", 0, 0, "empty", ""),
        ]
        out = format_sessions(sessions)
        assert "1. main" in out
        assert "2. bug-fix" in out
        assert "3. empty" in out
        assert "中断" in out
        assert "等待输入" in out
        assert "空" in out
        assert "(无消息)" in out  # empty 会话的预览


# ============================================================
# list_sessions：sqlite + get_state 回调集成
# ============================================================
class TestListSessions:
    @pytest.fixture
    def conn(self):
        """构造一个内存 sqlite，模拟 checkpoints 表结构。"""
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
            """
        )
        return conn

    def _insert_thread(self, conn, thread_id):
        """往 checkpoints 表插一个 thread_id。"""
        conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id) VALUES (?, '', ?)",
            (thread_id, f"ckpt-{thread_id}"),
        )
        conn.commit()

    def test_empty_db(self, conn):
        """空 checkpoint 表 → 空列表。"""
        sessions = list_sessions(conn, lambda tid: None)
        assert sessions == []

    def test_skip_thread_without_state(self, conn):
        """get_state 返回 None → 跳过该 thread。"""
        self._insert_thread(conn, "ghost")

        def get_state(tid):
            return None

        sessions = list_sessions(conn, get_state)
        assert sessions == []

    def test_skip_thread_with_none_values(self, conn):
        """state.values 为 None → 跳过。"""
        self._insert_thread(conn, "main")

        def get_state(tid):
            return SimpleNamespace(values=None, next=(), metadata=None, created_at=None)

        sessions = list_sessions(conn, get_state)
        assert sessions == []

    def test_multiple_threads_sorted(self, conn):
        """多个 thread_id 按字典序返回。"""
        self._insert_thread(conn, "session-20260929")
        self._insert_thread(conn, "main")
        self._insert_thread(conn, "session-20260928")

        states = {
            "main": _fake_state(next_=(), messages=[_fake_human_msg("你好")], step=5),
            "session-20260928": _fake_state(next_=("agent",), messages=[_fake_human_msg("修 bug")], step=3),
            "session-20260929": _fake_state(next_=(), messages=[], step=0),
        }

        sessions = list_sessions(conn, lambda tid: states.get(tid))
        assert [s.thread_id for s in sessions] == ["main", "session-20260928", "session-20260929"]

    def test_status_classification(self, conn):
        """next 非空=中断；next 空 + 有消息=等待；next 空 + 无消息=空。"""
        self._insert_thread(conn, "interrupted")
        self._insert_thread(conn, "waiting")
        self._insert_thread(conn, "empty")

        states = {
            "interrupted": _fake_state(next_=("agent",), messages=[_fake_human_msg("x")], step=1),
            "waiting": _fake_state(next_=(), messages=[_fake_human_msg("x")], step=1),
            "empty": _fake_state(next_=(), messages=[], step=0),
        }

        sessions = list_sessions(conn, lambda tid: states.get(tid))
        by_id = {s.thread_id: s for s in sessions}
        assert by_id["interrupted"].status == "interrupted"
        assert by_id["waiting"].status == "waiting"
        assert by_id["empty"].status == "empty"

    def test_last_user_msg_preview(self, conn):
        """最后一条 HumanMessage 内容被截前 40 字符。"""
        self._insert_thread(conn, "main")
        long_text = "x" * 100

        states = {
            "main": _fake_state(
                next_=(),
                messages=[
                    _fake_human_msg("旧问题"),
                    _fake_human_msg(long_text),
                ],
                step=2,
            ),
        }

        sessions = list_sessions(conn, lambda tid: states.get(tid))
        assert sessions[0].last_user_msg == "x" * 40  # 截断到 40 字符

    def test_last_user_msg_skips_ai_messages(self, conn):
        """最后一条是 AI 消息时，应跳过它找到前面的 HumanMessage。"""
        self._insert_thread(conn, "main")

        states = {
            "main": _fake_state(
                next_=(),
                messages=[
                    _fake_human_msg("用户问"),
                    AIMessage("AI 回答"),
                ],
                step=1,
            ),
        }

        sessions = list_sessions(conn, lambda tid: states.get(tid))
        assert sessions[0].last_user_msg == "用户问"

    def test_step_from_metadata(self, conn):
        """step 从 metadata.step 读取。"""
        self._insert_thread(conn, "main")
        states = {"main": _fake_state(next_=(), messages=[_fake_human_msg("x")], step=42)}
        sessions = list_sessions(conn, lambda tid: states.get(tid))
        assert sessions[0].step == 42

    def test_step_missing_metadata(self, conn):
        """metadata 为 None → step=0。"""
        self._insert_thread(conn, "main")
        state = SimpleNamespace(
            values={"messages": [_fake_human_msg("x")]},
            next=(),
            metadata=None,
            created_at="2026-09-30 12:00:00",
        )
        sessions = list_sessions(conn, lambda tid: state)
        assert sessions[0].step == 0


# ============================================================
# delete_session：同时清理 checkpoints 和 writes 两张表
# ============================================================
class TestDeleteSession:
    @pytest.fixture
    def conn(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT)"
        )
        conn.execute(
            "CREATE TABLE writes (thread_id TEXT, task_id TEXT)"
        )
        for tid in ("main", "sess-b"):
            conn.execute(
                "INSERT INTO checkpoints VALUES (?, ?)", (tid, f"ckpt-{tid}")
            )
            conn.execute(
                "INSERT INTO writes VALUES (?, ?)", (tid, f"task-{tid}")
            )
        conn.commit()
        return conn

    def test_delete_existing(self, conn):
        assert delete_session(conn, "sess-b") is True
        remaining = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints"
            ).fetchall()
        }
        remaining_writes = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT thread_id FROM writes"
            ).fetchall()
        }
        assert remaining == {"main"}
        assert remaining_writes == {"main"}

    def test_delete_nonexistent(self, conn):
        assert delete_session(conn, "ghost") is False
        # 其他会话不受影响
        remaining = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints"
            ).fetchall()
        }
        assert remaining == {"main", "sess-b"}
