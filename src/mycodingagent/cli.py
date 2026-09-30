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

logger = logging.getLogger(__name__)


# ============================================================
# 运行一轮任务并实时打印执行过程（模型思考 / 工具调用 / 工具结果）
# ============================================================
def run_task(agent, user_input: str, thread_id: str = "main") -> None:
    print(f"\n{'='*60}\n任务> {user_input}  [会话: {thread_id}]\n{'='*60}")
    cfg = {"configurable": {"thread_id": thread_id}}

    # 记录调用前的消息数，用于跳过"历史消息"
    state = agent.get_state(cfg)
    n_before = len(state.values.get("messages", [])) if state.values else 0

    # stream_mode="values" 每步产出完整状态，用 printed 做增量截取，
    # 只打印"本轮新增"的消息：
    # 人类消息 → AI 消息(可能含 tool_calls) → 工具结果消息 → ... → 最终 AI 回答
    printed = n_before + 1  # 跳过历史消息和本轮刚加入的用户消息
    for snapshot in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=cfg,  # 传 thread_id → 对话历史自动读写 checkpointer（短期记忆）
        stream_mode="values",
    ):
        for msg in snapshot["messages"][printed:]:
            printed += 1
            role = msg.__class__.__name__
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    args_text = str(tc["args"])
                    if len(args_text) > 100:
                        args_text = args_text[:100] + "..."
                    print(f"  [工具调用] {tc['name']}({args_text})", flush=True)
            elif role == "ToolMessage":
                preview = (msg.content or "")[:150].replace("\n", " ")
                suffix = "..." if len(msg.content or "") > 150 else ""
                print(f"  [工具结果] {preview}{suffix}", flush=True)
            elif msg.content:
                print(f"助手> {msg.content}", flush=True)


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
                print("  /memory               查看长期记忆中保存的用户信息")
                print("  /history              查看当前会话的消息统计")
                print("  /exit                 退出")
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
