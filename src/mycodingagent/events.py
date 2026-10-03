# -*- coding: utf-8 -*-
"""
结构化事件流（阶段 2 起，阶段 3 扩展）
======================================
把 LangGraph 的流式更新（stream_mode="updates"）转换为统一的 AgentEvent
事件序列：tool_start / tool_end / assistant_message / error / summary / interrupt。

CLI 只负责渲染事件；后续 TUI / API(SSE) 可直接复用同一事件流（缺口 C12）。

本层同时执行阶段 2 的三重预算，超限安全停下并产出可恢复的 error 事件：
    1. recursion_limit：传给 LangGraph，超限抛 GraphRecursionError；
    2. 步数上限：统计本轮 AI 消息数，达到 AGENT_MAX_STEPS 且还有待执行
       的工具调用时停止（模型给出最终回答不算违规）；
    3. token 预算：累计本轮 AI 消息 usage_metadata.total_tokens。

超限/中断时图状态保留在 checkpoint（state.next 非空），配合 /resume 续跑。

阶段 3 扩展：
- `INTERRUPT` 事件：deepagents `HumanInTheLoopMiddleware` 通过
  `langgraph.types.interrupt(HITLRequest)` 暂停图后，本层用
  `agent.get_state(config).interrupts` 检测并产出 INTERRUPT 事件；
- `resume_decisions` 参数：CLI 收集用户 Decision 列表后传回，本层用
  `Command(resume=HITLResponse(decisions=...))` 让图从断点继续。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

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
INTERRUPT = "interrupt"  # 阶段 3：HITL 审批请求

STOP_COMPLETED = "completed"              # 任务自然完成
STOP_STEP_LIMIT = "step_limit"            # 达到步数上限
STOP_TOKEN_LIMIT = "token_limit"          # 达到 token 预算
STOP_RECURSION_LIMIT = "recursion_limit"  # 达到 LangGraph recursion_limit
STOP_INTERRUPTED = "interrupted"          # 用户 Ctrl+C 中断
STOP_ERROR = "error"                      # 执行异常
STOP_PENDING_INTERRUPT = "pending_interrupt"  # 阶段 3：等待 HITL 决策


@dataclass
class AgentEvent:
    """结构化事件：CLI / TUI / API 共用的最小事件单元。"""

    type: str
    name: str = ""            # 工具名（tool_start / tool_end）
    args_preview: str = ""    # 工具参数预览（tool_start）
    text: str = ""            # 正文（assistant_message / error / interrupt description）
    is_error: bool = False    # 工具是否执行失败（tool_end）
    steps: int = 0            # 本轮模型步数（summary）
    tokens: int = 0           # 本轮累计 token（summary）
    stopped_reason: str = ""  # 停止原因（summary）
    # 阶段 3：INTERRUPT 事件携带的待审批 action_requests 列表。
    # 每个元素是 {"name": str, "args": dict, "description": str}；
    # CLI 用它渲染审批提示，收集等长 decisions 列表回传。
    action_requests: list[dict[str, Any]] = field(default_factory=list)
    interrupt_id: str = ""    # INTERRUPT 事件对应的 LangGraph Interrupt.id

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


def _friendly_llm_error(e: BaseException) -> str | None:
    """识别常见 LLM API 错误并翻译成用户可操作的提示；无法识别返回 None。

    openai SDK 的错误对象带 status_code；DeepSeek 等服务商还会把业务错误
    （如余额不足）包在 HTTP 500 里，所以先按 message 内容匹配再按状态码。
    """
    status = getattr(e, "status_code", None)
    text = str(e)
    lower = text.lower()
    if "insufficient balance" in lower:
        return (
            "模型服务提示账户余额不足。请前往服务商平台充值后，"
            "输入 /resume 从断点继续。"
        )
    if status == 401 or "invalid api key" in lower or "incorrect api key" in lower:
        return "API Key 无效或未生效（401）。请检查 .env 中的 API_KEY 配置。"
    if status == 403:
        return "服务拒绝访问（403）：当前 Key 可能未开通该模型或存在区域限制。"
    if status == 404 or "model_not_found" in lower:
        return "模型或端点不存在（404）。请检查 .env 中的 MODEL_NAME 与 BASE_URL。"
    if status == 429 or "rate limit" in lower:
        return "触发限流（429）。稍等片刻后输入 /resume 从断点继续。"
    if "timed out" in lower or "timeout" in lower:
        return "模型请求超时：服务响应过慢或网络不稳，稍后输入 /resume 重试。"
    if "connection error" in lower or "getaddrinfo failed" in lower or "failed to establish" in lower:
        return "无法连接模型服务。请检查网络与 .env 中的 BASE_URL 配置。"
    return None


# ============================================================
# 阶段 3：HITL interrupt 检测
# ============================================================
def detect_pending_interrupt(agent: Any, cfg: dict[str, Any]) -> Any:
    """检查图当前是否有未解决的 HITL interrupt 待审批。

    通过 `agent.get_state(cfg).interrupts` 读取（LangGraph 1.x 稳定 API）。
    返回第一个未解决的 `Interrupt` 对象（其 `.value` 是 HITLRequest）；
    没有 interrupt 时返回 None。

    抽成纯函数便于单测：FakeAgent 暴露 `get_state()` 返回 mock StateSnapshot。
    """
    try:
        snapshot = agent.get_state(cfg)
    except Exception:  # noqa: BLE001
        return None
    interrupts = getattr(snapshot, "interrupts", ()) or ()
    return interrupts[0] if interrupts else None


def _action_request_to_dict(action_request: Any) -> dict[str, Any]:
    """把 ActionRequest TypedDict 转成纯 dict（便于序列化与渲染）。"""
    if isinstance(action_request, dict):
        return {
            "name": action_request.get("name", ""),
            "args": action_request.get("args", {}) or {},
            "description": action_request.get("description", "") or "",
        }
    # 容错：dict-like 对象
    return {
        "name": getattr(action_request, "name", "") or "",
        "args": dict(getattr(action_request, "args", {}) or {}),
        "description": getattr(action_request, "description", "") or "",
    }


def _hitl_request_action_requests(interrupt_obj: Any) -> list[dict[str, Any]]:
    """从 Interrupt.value（HITLRequest）提取 action_requests 列表。"""
    value = getattr(interrupt_obj, "value", None)
    if value is None:
        return []
    # HITLRequest 是 TypedDict {"action_requests": list[ActionRequest], "review_configs": ...}
    if isinstance(value, dict):
        raw = value.get("action_requests", []) or []
    else:
        raw = getattr(value, "action_requests", []) or []
    return [_action_request_to_dict(ar) for ar in raw]


# ============================================================
# 核心：任务执行 → 事件流（含预算执行 + HITL）
# ============================================================
def iter_task_events(
    agent: Any,
    *,
    thread_id: str,
    user_input: str | None = None,
    resume_decisions: list[dict[str, Any]] | None = None,
) -> Iterator[AgentEvent]:
    """运行一轮任务并产出结构化事件。

    Args:
        agent: build_deep_agent 构建的 LangGraph 图。
        thread_id: 会话 ID（对话历史经 checkpointer 持久化）。
        user_input: 用户任务文本；None 表示从 checkpoint 断点续跑（/resume），
            此时图从中断处继续，不需要新输入。
        resume_decisions: 阶段 3 HITL 回答。CLI 收集到用户对每个 action_request
            的 Decision（dict 形式：{"type": "approve"/"reject"/"respond", ...}）
            后传入；本函数把它包成 `Command(resume=HITLResponse(decisions=...))`
            让图从 interrupt 断点继续。`resume_decisions` 与 `user_input` 互斥。

    Yields:
        AgentEvent 序列。常规结束恒为 summary；如果 stream 自然结束后检测到
        pending interrupt，则产出 INTERRUPT 事件后直接返回（不产出 summary，
        因为任务尚未真正结束，等 CLI 收集 decisions 后再调一次本函数 resume）。
    """
    cfg = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": config.AGENT_RECURSION_LIMIT,
    }

    # 阶段 3：构造 payload——三选一
    if resume_decisions is not None:
        # HITL resume：用 Command(resume=...) 让图从 interrupt 继续
        payload: Any = Command(resume={"decisions": list(resume_decisions)})
    elif user_input is None:
        # /resume 续跑：从 checkpoint 的 state.next 继续
        payload = None
    else:
        # 新一轮用户输入
        payload = {"messages": [{"role": "user", "content": user_input}]}

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
                        # 思考过程（DeepSeek reasoning_content）优先展示
                        reasoning = (
                            (msg.additional_kwargs or {}).get("reasoning_content")
                            or ""
                        ).strip()
                        if reasoning:
                            yield AgentEvent(
                                type=ASSISTANT_MESSAGE,
                                name="reasoning",
                                text=reasoning,
                            )
                        # 先输出正文（思考过程），再报工具调用——
                        # 带 tool_calls 的消息正文是模型调用工具前的推理，
                        # 展示出来便于观察决策过程
                        text = _message_text(msg.content).strip()
                        if text:
                            yield AgentEvent(type=ASSISTANT_MESSAGE, text=text)
                        for tc in msg.tool_calls or []:
                            yield AgentEvent(
                                type=TOOL_START,
                                name=str(tc.get("name", "")),
                                args_preview=_truncate(str(tc.get("args", "")), 120),
                            )
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
        stopped_reason = STOP_ERROR
        friendly = _friendly_llm_error(e)
        if friendly is not None:
            # 已识别的 API 错误：屏上给可操作提示，日志留一行原始错误，
            # 避免整页 traceback 刷屏
            logger.warning("已识别的 LLM API 错误：%s", e)
            yield AgentEvent(type=ERROR, text=friendly)
        else:
            logger.exception("任务执行异常")
            yield AgentEvent(
                type=ERROR,
                text=f"任务执行异常（{type(e).__name__}）：{e}",
            )
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    # 阶段 3：stream 自然结束后检测是否处于 HITL interrupt 待审批状态。
    # 若是，产出 INTERRUPT 事件（不产出 summary，等 CLI resume 后再收尾）。
    if stopped_reason == STOP_COMPLETED:
        pending = detect_pending_interrupt(agent, cfg)
        if pending is not None:
            action_requests = _hitl_request_action_requests(pending)
            # 拼一段综合描述给 CLI 渲染时使用
            descriptions = [ar.get("description", "") for ar in action_requests]
            combined = "\n---\n".join(d for d in descriptions if d)
            yield AgentEvent(
                type=INTERRUPT,
                text=combined,
                action_requests=action_requests,
                interrupt_id=str(getattr(pending, "id", "") or ""),
            )
            return  # 不产出 summary

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
        prefix = "思考> " if ev.name == "reasoning" else "助手> "
        return f"{prefix}{ev.text}"
    if ev.type == ERROR:
        return f"[!] {ev.text}"
    if ev.type == INTERRUPT:
        lines = ["[审批请求]"]
        for i, ar in enumerate(ev.action_requests, 1):
            name = ar.get("name", "")
            args = ar.get("args", {})
            desc = ar.get("description", "")
            lines.append(f"  {i}. 工具: {name}")
            if desc:
                lines.append(f"     {desc}")
            if args:
                args_str = _truncate(str(args), 200)
                lines.append(f"     参数: {args_str}")
        if not ev.action_requests:
            lines.append("  (无 action_request 信息)")
        lines.append(
            "请决策：/approve | /reject <reason> | /respond <text>"
            f"  ({len(ev.action_requests)} 个待决策)"
        )
        return "\n".join(lines)
    if ev.type == SUMMARY:
        reason_label = {
            STOP_COMPLETED: "完成",
            STOP_STEP_LIMIT: "步数超限",
            STOP_TOKEN_LIMIT: "token 超限",
            STOP_RECURSION_LIMIT: "递归超限",
            STOP_INTERRUPTED: "用户中断",
            STOP_ERROR: "异常",
            STOP_PENDING_INTERRUPT: "等待 HITL 决策",
        }.get(ev.stopped_reason, ev.stopped_reason)
        return f"[本轮统计] {ev.steps} 步 / {ev.tokens} tokens / {reason_label}"
    return f"[{ev.type}] {ev.text}"
