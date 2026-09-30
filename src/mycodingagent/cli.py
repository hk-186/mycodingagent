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
from mycodingagent.events import iter_task_events, render_event

logger = logging.getLogger(__name__)


# ============================================================
# 运行一轮任务：消费结构化事件流并渲染到终端（阶段 2）
# 事件转换与三重预算（recursion_limit / 步数 / token）都在
# events.iter_task_events 内完成，这里只做展示。
# ============================================================
def run_task(
    agent,
    user_input: str | None,
    thread_id: str = "main",
    *,
    resume: bool = False,
) -> None:
    shown = "(从上次断点继续)" if resume else user_input
    title = "续跑>" if resume else "任务>"
    print(f"\n{'='*60}\n{title} {shown}  [会话: {thread_id}]\n{'='*60}")
    try:
        for event in iter_task_events(
            agent,
            thread_id=thread_id,
            user_input=None if resume else user_input,
        ):
            print(render_event(event), flush=True)
    except KeyboardInterrupt:
        print("\n[已中断] 输入 /resume 可从断点继续。")


# ============================================================
# 交互模式
# ============================================================
def chat(agent, store, checkpointer) -> None:
    thread_id = "main"  # 默认会话固定 id：重启进程后可接着聊（短期记忆持久化）
    print(f"模型: {config.MODEL_NAME} | 会话: {thread_id} | 项目目录: {config.PROJECT_DIR}")
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
