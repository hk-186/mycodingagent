# -*- coding: utf-8 -*-
"""
命令行入口
==========
运行方式：
    mycodingagent                 # pyproject 注册的命令
    python -m mycodingagent       # 等价写法

会话命令：
    /sessions             列出所有历史会话
    /switch <id|序号>     切换到指定会话（序号来自 /sessions 列表）
    /new [名字]           开启新会话（旧会话历史保留在磁盘）
    /resume               从最近一次中断/超限的断点继续当前会话的任务
    /memory               查看长期记忆（store 中保存的用户信息）
    /history              查看当前会话的消息统计
    /plan <task>          进入 Plan 模式执行任务（先输出计划等审批）
    /approve              审批通过当前 INTERRUPT 事件的所有 action_request
    /reject <reason>      拒绝当前 INTERRUPT，reason 可选
    /respond <text>       用 respond 决策回答 ask_user 工具的问题
    /exit                 退出
"""

import argparse
import logging
import os
from datetime import datetime

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from mycodingagent import config
from mycodingagent.agent import build_deep_agent
from mycodingagent.events import (
    INTERRUPT,
    SUMMARY,
    AgentEvent,
    iter_task_events,
    render_event,
)

logger = logging.getLogger(__name__)


# ============================================================
# 阶段 3：HITL 决策收集
# 把用户的 CLI 输入解析成 deepagents `Decision` dict 形式。
# 单 action_request 场景：用户输入单条决策；
# 多 action_request 场景：逐个询问，构造等长 decisions 列表。
# ============================================================
def _parse_decision_input(text: str) -> dict | None:
    """把用户输入解析成 Decision dict，无效返回 None。

    支持形式：
      /approve             -> {"type": "approve"}
      /reject [<reason>]   -> {"type": "reject", "message": <reason>} (reason 可选)
      /respond <text>      -> {"type": "respond", "message": <text>}
      a / r <reason> / rr <text>   —— 简写
    """
    text = text.strip()
    if not text:
        return None
    # 命令形式
    if text.startswith("/"):
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower()
        if cmd == "/approve":
            return {"type": "approve"}
        if cmd == "/reject":
            return {"type": "reject", "message": arg.strip()} if arg.strip() else {"type": "reject"}
        if cmd == "/respond":
            if not arg.strip():
                print("[/respond 需要回答内容，例如：/respond 是的，请按方案 A]")
                return None
            return {"type": "respond", "message": arg.strip()}
        print(f"[未知决策命令 {cmd}，可用：/approve /reject [<reason>] /respond <text>]")
        return None
    # 简写形式
    low = text.lower()
    if low in {"a", "y", "yes"}:
        return {"type": "approve"}
    if low in {"r", "n", "no"}:
        return {"type": "reject"}
    if low.startswith("rr "):
        return {"type": "respond", "message": text[3:].strip()}
    if low.startswith("r "):
        return {"type": "reject", "message": text[2:].strip()}
    # 默认当成 respond（适合 ask_user 场景）
    return {"type": "respond", "message": text}


def _collect_decisions_for_interrupt(ev: AgentEvent) -> list[dict] | None:
    """对 INTERRUPT 事件的 action_requests 逐个收集用户决策。

    返回与 action_requests 等长的 decisions 列表；
    用户输入空行 / 无效 / Ctrl+C 返回 None 表示取消本轮。
    """
    n = len(ev.action_requests)
    if n == 0:
        return []
    decisions: list[dict] = []
    for i, ar in enumerate(ev.action_requests, 1):
        name = ar.get("name", "")
        desc = ar.get("description", "")
        print(f"\n[{i}/{n}] 工具: {name}")
        if desc:
            print(desc)
        while True:
            try:
                user_input = input("决策> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return None
            if user_input in {"cancel", "abort", "/cancel"}:
                return None
            decision = _parse_decision_input(user_input)
            if decision is not None:
                decisions.append(decision)
                break
            # 无效输入：循环再问
    return decisions


# ============================================================
# 运行一轮任务：消费结构化事件流并渲染到终端（阶段 2 + 阶段 3 HITL）
# 事件转换与三重预算（recursion_limit / 步数 / token）都在
# events.iter_task_events 内完成；HITL interrupt 在本函数循环处理：
#   渲染 → 遇 INTERRUPT 停下 → 收集 decisions → 调 iter_task_events(resume_decisions=...)
#   循环直到收到 summary 事件。
# ============================================================
def run_task(
    agent,
    user_input: str | None,
    thread_id: str = "main",
    *,
    resume: bool = False,
    plan_mode: bool = False,
) -> None:
    shown = "(从上次断点继续)" if resume else user_input
    title = "续跑>" if resume else ("Plan>" if plan_mode else "任务>")
    print(f"\n{'='*60}\n{title} {shown}  [会话: {thread_id}]\n{'='*60}")

    # 第一轮参数
    next_user_input: str | None = None if resume else user_input
    next_resume_decisions: list[dict] | None = None

    try:
        while True:
            # 事件流：渲染并检测 INTERRUPT
            interrupt_event: AgentEvent | None = None
            got_summary = False
            for event in iter_task_events(
                agent,
                thread_id=thread_id,
                user_input=next_user_input,
                resume_decisions=next_resume_decisions,
            ):
                print(render_event(event), flush=True)
                if event.type == INTERRUPT:
                    interrupt_event = event
                elif event.type == SUMMARY:
                    got_summary = True

            # 没遇到 INTERRUPT 且收到 summary：本轮结束
            if interrupt_event is None:
                if not got_summary:
                    # 兜底：事件流没有正常收尾
                    print("[!] 任务流未产出 summary 事件（异常）")
                break

            # 遇到 INTERRUPT：收集用户决策
            decisions = _collect_decisions_for_interrupt(interrupt_event)
            if decisions is None:
                print("[已取消本轮审批，输入 /resume 可从断点继续]")
                break
            if len(decisions) != len(interrupt_event.action_requests):
                print(
                    f"[决策数量不匹配：需要 {len(interrupt_event.action_requests)} 个，"
                    f"收到 {len(decisions)} 个。审批未提交，输入 /resume 可重新审批]"
                )
                break

            # propose_plan 被 approve 后解除 Plan 模式
            if plan_mode and any(
                ar.get("name") == "propose_plan" for ar in interrupt_event.action_requests
            ):
                if any(d.get("type") == "approve" for d in decisions):
                    config.set_plan_mode(False)
                    plan_mode = False
                    print("[Plan 模式已解除，开始执行]")

            # 下一轮：用 decisions resume，不再传 user_input
            next_user_input = None
            next_resume_decisions = decisions
    except KeyboardInterrupt:
        print("\n[已中断] 输入 /resume 可从断点继续。")


# ============================================================
# 交互模式
# ============================================================
def chat(agent, store, checkpointer) -> None:
    thread_id = "main"  # 默认会话固定 id：重启进程后可接着聊（短期记忆持久化）
    print(f"模型: {config.MODEL_NAME} | 会话: {thread_id} | 项目目录: {config.PROJECT_DIR}")
    print(f"审批模式: {config.APPROVAL_MODE} | Plan 模式: {'开启' if config.PLAN_MODE else '关闭'}")
    print("输入 /help 查看命令，/exit 退出\n")

    # 工具函数：根据 thread_id 拿 StateSnapshot
    def get_state(tid: str):
        return agent.get_state({"configurable": {"thread_id": tid}})

    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, _, arg = user_input.partition(" ")
            if cmd == "/exit":
                break
            elif cmd == "/help":
                print("  /sessions             列出所有历史会话")
                print("  /switch <id|序号>     切换到指定会话")
                print("  /new [名字]           开启新会话（旧会话历史保留）")
                print("  /resume               从断点继续当前会话中断的任务")
                print("  /memory               查看长期记忆中保存的用户信息")
                print("  /history              查看当前会话的消息统计")
                print("  /plan <task>          进入 Plan 模式执行任务")
                print("  /approve              在 INTERRUPT 状态下审批通过")
                print("  /reject [<reason>]    拒绝当前审批，可附原因")
                print("  /respond <text>       用 respond 决策回答 ask_user 工具的问题")
                print("  /exit                 退出")
            elif cmd == "/resume":
                state = get_state(thread_id)
                if not state.next:
                    print("[当前会话没有可恢复的中断任务]")
                else:
                    pending = ", ".join(state.next)
                    print(f"[从断点继续，待执行节点: {pending}]")
                    run_task(agent, None, thread_id, resume=True)
            elif cmd == "/new":
                thread_id = arg.strip() or f"session-{datetime.now():%Y%m%d-%H%M%S}"
                print(f"[已切换到新会话: {thread_id}]")
            elif cmd == "/sessions":
                from mycodingagent.sessions import list_sessions, format_sessions
                sessions = list_sessions(checkpointer.conn, get_state)
                print(format_sessions(sessions))
            elif cmd == "/switch":
                from mycodingagent.sessions import list_sessions, resolve_switch_arg
                sessions = list_sessions(checkpointer.conn, get_state)
                new_tid = resolve_switch_arg(arg, sessions)
                if new_tid is None:
                    print(f"[找不到会话: {arg or '(空)'}，输入 /sessions 查看可用会话]")
                else:
                    thread_id = new_tid
                    state = get_state(thread_id)
                    n = len(state.values.get("messages", [])) if state.values else 0
                    status = "中断" if state.next else "等待输入"
                    print(f"[已切换到会话: {thread_id}]  ({n} 条消息, {status})")
            elif cmd == "/memory":
                items = store.search(("users",))
                if not items:
                    print("[长期记忆为空]")
                for item in items:
                    print(f"  {item.key} = {item.value['value']}")
            elif cmd == "/history":
                state = agent.get_state({"configurable": {"thread_id": thread_id}})
                n = len(state.values.get("messages", [])) if state.values else 0
                print(
                    f"[会话 {thread_id} 共 {n} 条消息（已持久化到 {config.CHECKPOINT_DB}）]"
                )
            elif cmd == "/plan":
                task = arg.strip()
                if not task:
                    print("[/plan 需要任务说明，例如：/plan 给 utils.py 加 is_palindrome 函数]")
                else:
                    config.set_plan_mode(True)
                    try:
                        run_task(agent, task, thread_id, plan_mode=True)
                    finally:
                        # 兜底：若任务结束（含异常）Plan 模式仍开启则关闭
                        if config.PLAN_MODE:
                            config.set_plan_mode(False)
            else:
                print(f"[未知命令 {cmd}，输入 /help 查看命令]")
            continue

        run_task(agent, user_input, thread_id)


def main() -> None:
    # 命令行参数：--project 指定工作目标项目目录（阶段 1 缺口 C3）
    parser = argparse.ArgumentParser(
        prog="mycodingagent",
        description="自研 AI coding agent",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="工作目标项目目录（默认 workspace/，或环境变量 AGENT_PROJECT_DIR）",
    )
    args = parser.parse_args()
    if args.project:
        try:
            config.set_project_dir(args.project)
        except NotADirectoryError as e:
            print(str(e))
            return

    # logging：级别走 LOG_LEVEL 环境变量（默认 WARNING，调试时设 INFO/DEBUG）
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "WARNING").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 修复计划 E4：提示的环境变量名与 config 实际读取的一致
    if not config.API_KEY:
        print("未检测到 AGICTO_API_KEY！请先设置环境变量（模板见 .env.example）。")
        return

    # SqliteSaver = 短期记忆（对话历史）  SqliteStore = 长期记忆（用户信息）
    # with 同时管理两个 sqlite 连接的生命周期
    saver_cm = SqliteSaver.from_conn_string(config.CHECKPOINT_DB)
    store_cm = SqliteStore.from_conn_string(config.STORE_DB)
    with saver_cm as checkpointer, store_cm as store:
        agent = build_deep_agent(checkpointer, store)
        chat(agent, store, checkpointer)

    print("再见！对话历史、长期记忆和文件都已保存，下次启动依然有效。")


if __name__ == "__main__":
    main()
