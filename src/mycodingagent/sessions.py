# -*- coding: utf-8 -*-
"""
会话管理
========
列出历史会话、解析 /switch 参数、删除会话。
thread_id 列表通过直接查 SqliteSaver 的 checkpoints 表获得
（LangGraph 没有提供"列出所有 thread_id"的公开 API）。
会话状态通过 agent.get_state() 拿 StateSnapshot 判断。
删除会话需同时清理 checkpoints 和 writes 两张表。
"""

import sqlite3
from dataclasses import dataclass
from typing import Any, Callable


# ============================================================
# 数据结构
# ============================================================
@dataclass
class SessionInfo:
    """单个历史会话的概要信息。"""
    thread_id: str          # 会话 ID
    last_updated: str       # 最后更新时间（state.created_at，ISO 字符串）
    step: int               # 最后一步的步数（来自 metadata.step）
    msg_count: int          # 消息总数
    status: str             # interrupted / waiting / empty
    last_user_msg: str      # 最后一条用户消息的预览（前 40 字符）


# 状态 → 中文标签
_STATUS_LABEL = {
    "interrupted": "中断",
    "waiting": "等待输入",
    "empty": "空",
}


# ============================================================
# 查询会话列表
# ============================================================
def list_sessions(
    conn: sqlite3.Connection,
    get_state: Callable[[str], Any],
) -> list[SessionInfo]:
    """
    列出所有历史会话。

    Args:
        conn: SqliteSaver 的 sqlite3.Connection（agent.checkpointer.conn）
        get_state: 接受 thread_id 返回 StateSnapshot 的回调
                   （等价于 lambda tid: agent.get_state({"configurable": {"thread_id": tid}})）

    Returns:
        按 thread_id 字典序排序的 SessionInfo 列表。
        没有 state 的 thread（理论上不会出现）会被跳过。
    """
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")
    thread_ids = [row[0] for row in cur.fetchall()]

    sessions: list[SessionInfo] = []
    for tid in thread_ids:
        state = get_state(tid)
        if state is None or state.values is None:
            continue

        messages = state.values.get("messages", [])
        msg_count = len(messages)

        # 找最后一条用户消息
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.__class__.__name__ == "HumanMessage":
                content = msg.content
                if not isinstance(content, str):
                    content = str(content)
                last_user_msg = content[:40]
                break

        # 判断状态：next 非空 = 有待执行的节点（中断）；否则按消息数判断
        if state.next:
            status = "interrupted"
        elif msg_count == 0:
            status = "empty"
        else:
            status = "waiting"

        # metadata.step 是步数；created_at 是 ISO 时间字符串
        step = 0
        if state.metadata:
            step = int(state.metadata.get("step", 0))
        last_updated = state.created_at or "未知"

        sessions.append(SessionInfo(
            thread_id=tid,
            last_updated=last_updated,
            step=step,
            msg_count=msg_count,
            status=status,
            last_user_msg=last_user_msg,
        ))

    return sessions


# ============================================================
# 格式化显示
# ============================================================
def format_sessions(sessions: list[SessionInfo]) -> str:
    """格式化会话列表用于 CLI 输出。"""
    if not sessions:
        return "[没有任何历史会话]"

    lines = ["历史会话："]
    for i, s in enumerate(sessions, 1):
        status_label = _STATUS_LABEL.get(s.status, s.status)
        preview = s.last_user_msg or "(无消息)"
        lines.append(
            f"  {i}. {s.thread_id}"
            f"  ({s.last_updated}, {s.step} 步, {s.msg_count} 条消息, {status_label})"
        )
        lines.append(f"      最后: {preview}")
    return "\n".join(lines)


# ============================================================
# /switch 参数解析
# ============================================================
def resolve_switch_arg(arg: str, sessions: list[SessionInfo]) -> str | None:
    """
    解析 /switch 参数：支持序号（1/2/3）或 thread_id 字符串（含前缀模糊匹配）。

    Returns:
        匹配到的 thread_id；未匹配返回 None。
    """
    arg = arg.strip()
    if not arg:
        return None

    # 1) 纯数字 → 按序号解析（1-based）
    if arg.isdigit():
        idx = int(arg)
        if 1 <= idx <= len(sessions):
            return sessions[idx - 1].thread_id
        return None

    # 2) 精确匹配 thread_id
    for s in sessions:
        if s.thread_id == arg:
            return arg

    # 3) 前缀模糊匹配（如 "session-2026" 匹配 "session-20260928-214144"）
    matches = [s.thread_id for s in sessions if s.thread_id.startswith(arg)]
    if len(matches) == 1:
        return matches[0]

    return None


# ============================================================
# 删除会话
# ============================================================
def delete_session(conn: sqlite3.Connection, thread_id: str) -> bool:
    """删除指定会话的全部 checkpoint 记录（checkpoints + writes 两张表）。

    Returns:
        是否删到了记录（thread_id 不存在时为 False）。
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    deleted = cur.rowcount > 0
    conn.commit()
    return deleted
