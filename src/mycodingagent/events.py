# -*- coding: utf-8 -*-
"""
结构化事件流（阶段 2）
======================
把 LangGraph 的流式更新（stream_mode="updates"）转换为统一的 AgentEvent
事件序列：tool_start / tool_end / assistant_message / error / summary。

CLI 只负责渲染事件；后续 TUI / API(SSE) 可直接复用同一事件流（缺口 C12）。

本层同时执行阶段 2 的三重预算，超限安全停下并产出可恢复的 error 事件：
    1. recursion_limit：传给 LangGraph，超限抛 GraphRecursionError；
    2. 步数上限：统计本轮 AI 消息数，达到 AGENT_MAX_STEPS 且还有待执行
       的工具调用时停止（模型给出最终回答不算违规）；
    3. token 预算：累计本轮 AI 消息 usage_metadata.total_tokens。

超限/中断时图状态保留在 checkpoint（state.next 非空），配合 /resume 续跑。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from mycodingagent import config

logger = logging.getLogger(__name__)

# ============================================================
# 事件类型与停止原因常量
# ============================================================
TOOL_START = "tool_start"
TOOL_END = "tool_end"
ASSISTANT_MESSAGE = "assistant_message"
ERROR = "error"
SUMMARY = "summary"

STOP_COMPLETED = "completed"              # 任务自然完成
STOP_STEP_LIMIT = "step_limit"            # 达到步数上限
STOP_TOKEN_LIMIT = "token_limit"          # 达到 token 预算
STOP_RECURSION_LIMIT = "recursion_limit"  # 达到 LangGraph recursion_limit
STOP_INTERRUPTED = "interrupted"          # 用户 Ctrl+C 中断
STOP_ERROR = "error"                      # 执行异常


@dataclass
class AgentEvent:
    """结构化事件：CLI / TUI / API 共用的最小事件单元。"""

    type: str
    name: str = ""            # 工具名（tool_start / tool_end）
    args_preview: str = ""    # 工具参数预览（tool_start）
    text: str = ""            # 正文（assistant_message / error）
    is_error: bool = False    # 工具是否执行失败（tool_end）
    steps: int = 0            # 本轮模型步数（summary）
    tokens: int = 0           # 本轮累计 token（summary）
    stopped_reason: str = ""  # 停止原因（summary）

    def to_dict(self) -> dict[str, Any]:
        """序列化为 plain dict（为 API/SSE 输出做准备）。"""
        return asdict(self)


# ============================================================
# 消息解析辅助函数
# ============================================================
def _message_text(content: Any) -> str:
    """提取消息文本：str 直接返回；content blocks 列表拼接 text 块。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _usage_total_tokens(msg: AIMessage) -> int:
    """读取 AI 消息的 usage_metadata.total_tokens（无计量时返回 0）。"""
    usage = getattr(msg, "usage_metadata", None)
    if not usage:
        return 0
    try:
        return int(usage.get("total_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "..."


# ============================================================
# 核心：任务执行 → 事件流（含预算执行）
# ============================================================
def iter_task_events(
    agent: Any,
    *,
    thread_id: str,
    user_input: str | None = None,
) -> Iterator[AgentEvent]:
    """运行一轮任务并产出结构化事件。

    Args:
        agent: build_deep_agent 构建的 LangGraph 图。
        thread_id: 会话 ID（对话历史经 checkpointer 持久化）。
        user_input: 用户任务文本；None 表示从 checkpoint 断点续跑（/resume），
            此时图从中断处继续，不需要新输入。

    Yields:
        AgentEvent 序列，最后一个事件恒为 summary（含步数/token/停止原因）。
    """
    cfg = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": config.AGENT_RECURSION_LIMIT,
    }
    payload = (
        None
        if user_input is None
        else {"messages": [{"role": "user", "content": user_input}]}
    )

    steps = 0
    tokens = 0
    stopped_reason = STOP_COMPLETED
    stream = agent.stream(payload, config=cfg, stream_mode="updates")

    try:
        while True:
            # ---- 取下一个更新，把各类"停止"转成事件 ----
            try:
                update = next(stream)
            except StopIteration:
                break
            except GraphRecursionError:
                stopped_reason = STOP_RECURSION_LIMIT
                yield AgentEvent(
                    type=ERROR,
                    text=(
                        f"已达到 recursion_limit（{config.AGENT_RECURSION_LIMIT} 步），"
                        "任务已安全暂停。输入 /resume 可从断点继续（预算重新计数）。"
                    ),
                )
                break
            except KeyboardInterrupt:
                stopped_reason = STOP_INTERRUPTED
                yield AgentEvent(
                    type=ERROR,
                    text="已被用户中断（Ctrl+C）。输入 /resume 可从断点继续。",
                )
                break

            if not update:
                continue

            # ---- 把节点更新里的消息逐个转换为事件 ----
            pending_work = False  # 本更新后是否还有待执行的工具调用
            for node_delta in update.values():
                if not isinstance(node_delta, dict):
                    continue
                for msg in node_delta.get("messages") or []:
                    if isinstance(msg, AIMessage):
                        steps += 1
                        tokens += _usage_total_tokens(msg)
                        for tc in msg.tool_calls or []:
                            yield AgentEvent(
                                type=TOOL_START,
                                name=str(tc.get("name", "")),
                                args_preview=_truncate(str(tc.get("args", "")), 120),
                            )
                        # 与旧版 CLI 行为一致：带工具调用的消息只报工具调用，
                        # 最终回答（无 tool_calls 的 AI 消息）才输出正文
                        if not msg.tool_calls:
                            text = _message_text(msg.content).strip()
                            if text:
                                yield AgentEvent(type=ASSISTANT_MESSAGE, text=text)
                        pending_work = pending_work or bool(msg.tool_calls)
                    elif isinstance(msg, ToolMessage):
                        yield AgentEvent(
                            type=TOOL_END,
                            name=str(getattr(msg, "name", "") or ""),
                            text=_truncate(_message_text(msg.content), 150),
                            is_error=getattr(msg, "status", "") == "error",
                        )

            # ---- 三重预算中的两项：步数 / token（有 pending 工作才拦截，
            #      模型正在给最终回答时让它自然收尾）----
            if pending_work and steps >= config.AGENT_MAX_STEPS:
                stopped_reason = STOP_STEP_LIMIT
                yield AgentEvent(
                    type=ERROR,
                    text=(
                        f"已达每任务步数上限（{config.AGENT_MAX_STEPS} 步），"
                        "任务已安全暂停。输入 /resume 可从断点继续（预算重新计数）。"
                    ),
                )
                break
            if pending_work and tokens >= config.AGENT_TOKEN_BUDGET:
                stopped_reason = STOP_TOKEN_LIMIT
                yield AgentEvent(
                    type=ERROR,
                    text=(
                        f"已达每任务 token 预算（{config.AGENT_TOKEN_BUDGET}），"
                        "任务已安全暂停。输入 /resume 可从断点继续（预算重新计数）。"
                    ),
                )
                break
    except Exception as e:  # noqa: BLE001 — 未知异常转成 error 事件而非让 CLI 崩溃
        logger.exception("任务执行异常")
        stopped_reason = STOP_ERROR
        yield AgentEvent(
            type=ERROR,
            text=f"任务执行异常（{type(e).__name__}）：{e}",
        )
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    yield AgentEvent(
        type=SUMMARY,
        steps=steps,
        tokens=tokens,
        stopped_reason=stopped_reason,
    )


# ============================================================
# 渲染：事件 → 终端文本（CLI 专用；TUI/API 不经过这里）
# ============================================================
def render_event(ev: AgentEvent) -> str:
    """把事件渲染成终端文本行。"""
    if ev.type == TOOL_START:
        return f"  [工具调用] {ev.name}({ev.args_preview})"
    if ev.type == TOOL_END:
        tag = "工具失败" if ev.is_error else "工具结果"
        return f"  [{tag}] {ev.name}: {ev.text}"
    if ev.type == ASSISTANT_MESSAGE:
        return f"助手> {ev.text}"
    if ev.type == ERROR:
        return f"[!] {ev.text}"
    if ev.type == SUMMARY:
        reason_label = {
            STOP_COMPLETED: "完成",
            STOP_STEP_LIMIT: "步数超限",
            STOP_TOKEN_LIMIT: "token 超限",
            STOP_RECURSION_LIMIT: "递归超限",
            STOP_INTERRUPTED: "用户中断",
            STOP_ERROR: "异常",
        }.get(ev.stopped_reason, ev.stopped_reason)
        return f"[本轮统计] {ev.steps} 步 / {ev.tokens} tokens / {reason_label}"
    return f"[{ev.type}] {ev.text}"
