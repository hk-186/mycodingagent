
# -*- coding: utf-8 -*-
"""
Deep Agents 完整示例
====================
对照学习：langchain_agent.py 用的是 create_agent（基础款 Agent），
本文件演示 deepagents 库的 create_deep_agent（深度任务款 Agent）。

  能力：深度任务（文件系统 + 子代理）/ 多轮对话 / 联网搜索 / 短期 + 长期记忆
  短期记忆：SqliteSaver —— 对话历史按 thread_id 持久化到磁盘，重启进程不丢失
  长期记忆：SqliteStore —— 用户信息跨会话共享（与 langchain_agent.py 同一套方案）
  联网：   TavilySearch（设置了 TAVILY_API_KEY 时自动启用）

为什么需要 Deep Agent？
    基础款 Agent 把所有对话历史都堆在一个上下文窗口里，任务一长
    （几十轮、读几十份资料）就会撑爆上下文、token 贵、模型容易忘目标。
    Deep Agent 在 LangGraph 之上预设了两个深度任务机制：

    1. 虚拟文件系统（本示例演示）
       Agent 自带 write_file / read_file / ls / edit_file / glob / grep 等
       工具，把中间资料和最终成果写进磁盘目录（deep_agent_workspace/），
       而不是全部塞在上下文里——像人做调研时记笔记，而不是全背下来。

    2. 子代理 subagents（本示例演示）
       注册带独立工具/独立上下文的子代理后，主 Agent 会多一个 task 工具，
       可以把子任务"派"给子代理，子代理在自己的上下文里干活，
       最后只把结论交回主 Agent——像主编派记者分头采访。

    （另有 skills 按需加载技能、summarization 长程摘要等高级能力，本示例不展开。
      记忆采用"双轨制"：行为规则放 memory=["/AGENTS.md"]——文件内容每轮注入
      system prompt，相当于随身员工手册；事实数据放 SqliteStore——save/recall
      工具按需查询的档案柜，与 langchain_agent.py 同款。两者可共存。）

运行：
    python deep_agent.py            # 交互模式（自己出题）
会话命令：
    /new [名字]  开启新会话（旧会话历史保留在磁盘）
    /memory      查看长期记忆（store 中保存的用户信息）
    /history     查看当前会话的消息统计
    /exit        退出

依赖：
    pip install deepagents
    （模型接入与 langchain_agent.py 完全相同：Dashscope 兼容端点）

可观测：
    与 create_agent 一样，create_deep_agent 内置 LangSmith 回调，
    设好环境变量后可在网页查看「主 Agent → task → 子 Agent」的完整执行树。
"""

import ast
import operator
import os
from datetime import datetime
from pathlib import Path

# ============================================================
# LangSmith 追踪配置（与 langchain_agent.py 相同的接入方式）
# 密钥 LANGSMITH_API_KEY、区域 LANGSMITH_ENDPOINT 仍读系统环境变量
# ============================================================
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "deep-agent-demo")
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://apac.api.smith.langchain.com")

# ============================================================
# 模型配置（与 langchain_agent.py 保持一致：Dashscope 兼容端点）
# ============================================================
# API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
# BASE_URL = "https://ws-l878zsxzmsqhd8uv.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
# MODEL_NAME = "deepseek-v4.1-flash"
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Agicto 配置
API_KEY = os.getenv("AGICTO_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
BASE_URL = "https://api.agicto.cn/v1"
MODEL_NAME = "deepseek-v4-flash"

CHECKPOINT_DB = "deep_agent_checkpoint.sqlite"  # 短期记忆：对话历史
STORE_DB = "deep_agent_memory.sqlite"  # 长期记忆：用户信息

# Agent 的"笔记本"目录：虚拟文件系统里写的文件都会落在这里
WORKSPACE_DIR = Path(__file__).parent / "deep_agent_workspace"

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_store  # 工具运行时由 agent 注入 store
from langgraph.store.sqlite import SqliteStore

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.7,
    )


# ============================================================
# 主 Agent 的自定义工具：获取当前时间
# （文件系统工具不用自己定义——create_deep_agent 已内置）
# ============================================================
@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


# ============================================================
# 子代理专用工具：安全计算器
# （ast 白名单解析，只允许四则运算，防止代码注入）
# ============================================================
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("表达式中含有不允许的元素")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，支持 + - * / // % ** 和括号。输入示例：(128 + 64) * 3"""
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


# ============================================================
# 长期记忆的读写工具（信息存入 SqliteStore，跨会话共享）
# 与 langchain_agent.py 完全相同的实现：get_store() 在工具运行时
# 取出 create_deep_agent(store=...) 注入的 store 来用
# ============================================================
@tool
def save_user_info(key: str, value: str) -> str:
    """保存用户的长期信息。key 是信息类别（如 name、age、hobby），value 是具体内容"""
    store = get_store()
    store.put(("users",), key, {"value": value})
    return f"已记住：{key} = {value}"


@tool
def recall_user_info(key: str) -> str:
    """回忆用户的长期信息。key 是信息类别（如 name、age、hobby），不确定时可用 recall_user_info_list"""
    store = get_store()
    item = store.get(("users",), key)
    if item is None:
        # 精确 key 没找到时，本地模糊匹配一遍，提升召回率
        hits = [
            i for i in store.search(("users",)) if key in i.key or key in str(i.value)
        ]
        if hits:
            return "；".join(f"{h.key} = {h.value['value']}" for h in hits)
        return f"没有找到 {key} 相关的记忆"
    return f"{key} = {item.value['value']}"


@tool
def recall_user_info_list() -> str:
    """列出所有已保存的用户长期信息"""
    store = get_store()
    items = store.search(("users",))
    if not items:
        return "还没有保存任何用户信息"
    return "；".join(f"{i.key} = {i.value['value']}" for i in items)


# ============================================================
# 构建 Deep Agent
# ============================================================
def build_deep_agent(checkpointer, store):
    """
    create_deep_agent 与 create_agent 的关键差异：

    - backend：文件系统后端。FilesystemBackend(root_dir=...) 把 Agent 的
      文件读写限制在该目录内，跑完可以直接去目录里翻 Agent 写的文件。
    - subagents：子代理列表（TypedDict 字典形式）。注册后主 Agent 自动
      获得 task 工具，按子代理的 description 决定何时委派。
    - 工具继承：主 Agent 自动拥有 ls/read_file/write_file/edit_file/
      glob/grep/task 等内置工具，无需手动传入。
    - checkpointer / store：与 create_agent 同名同义——短期/长期记忆，
      直接把 Sqlite 连接传进来即可，用法和 langchain_agent.py 完全一致。
    - memory：AGENTS.md 风格的规则记忆。文件内容在启动时整段注入 system
      prompt，Agent 学到新偏好后自己用 edit_file 写回——"员工手册"模式，
      适合存行为规则；事实性数据仍走 store（见上面的记忆工具）。
    """
    llm = get_llm()

    # 确保工作目录存在（Agent 往里写文件时才有落脚点）
    WORKSPACE_DIR.mkdir(exist_ok=True)

    # 主 Agent 的工具：时间 + 长期记忆读写（+ 设置了密钥时的联网搜索）
    tools = [
        get_current_time,
        save_user_info,
        recall_user_info,
        recall_user_info_list,
    ]
    search_hint = ""
    if TAVILY_API_KEY:
        from langchain_tavily import TavilySearch

        tools.append(TavilySearch(max_results=3))
        search_hint = "- web_search：回答时效性/事实性问题前先用它查最新资料\n"
    else:
        print("[提示] 未设置 TAVILY_API_KEY，联网搜索功能未启用")

    # 子代理：专门负责数学计算，拥有独立的工具集和系统提示词
    # 主 Agent 看不到 calculate 的内部过程，只收到子代理返回的结论
    calculator_subagent = {
        "name": "calculator",
        "description": (
            "数学计算专家。任何需要精确计算的问题都委派给它，"
            "例如四则运算、百分比、乘方。输入完整的数学表达式。"
        ),
        "tools": [calculate],
        "model": llm,
        "system_prompt": (
            "你是数学计算专家。收到表达式后必须调用 calculate 工具计算，"
            "禁止心算。用中文简要返回计算结果。"
        ),
    }

    agent = create_deep_agent(
        model=llm,
        tools=tools,
        backend=FilesystemBackend(root_dir=str(WORKSPACE_DIR)),
        subagents=[calculator_subagent],   # 注册子代理 → 主 Agent 获得 task 工具
        system_prompt=(
            "你是一个擅长完成深度任务的智能助手。严格遵守以下工具使用规则：\n"
            + search_hint
            + "- 任何数学计算都不要心算，使用 task 工具委派给 calculator 子代理\n"
            "- 需要时间信息时调用 get_current_time\n"
            "- save_user_info：用户提到自己的个人信息（名字、年龄、偏好等）时，"
            "每一项分别调用一次保存\n"
            "- recall_user_info：需要了解用户信息时调用，宁可多查也不要凭空猜测；"
            "不确定 key 时先用 recall_user_info_list 列出全部\n"
            "- 调研过程和最终成果用 write_file 写入文件保存（放在当前工作目录）\n"
            "- 完成后用中文简要汇报：做了什么、结论是什么、文件保存在哪里"
        ),
        memory=["/AGENTS.md"],  # 规则型记忆：AGENTS.md 内容每轮注入 system prompt
        checkpointer=checkpointer,  # 短期记忆：对话历史按 thread_id 持久化
        store=store,  # 长期记忆：save/recall_user_info 工具读写它
    )
    return agent


# ============================================================
# 运行一轮任务并实时打印执行过程（模型思考 / 工具调用 / 工具结果）
# ============================================================
def run_task(agent, user_input: str, thread_id: str = "main") -> None:
    print(f"\n{'='*60}\n任务> {user_input}  [会话: {thread_id}]\n{'='*60}")
    config = {"configurable": {"thread_id": thread_id}}

    # 记录调用前的消息数，用于跳过"历史消息"
    state = agent.get_state(config)
    n_before = len(state.values.get("messages", [])) if state.values else 0

    # 用 stream 代替 invoke：每完成一个步骤（模型回复 / 工具执行）就实时打印，
    # 不用等整个任务跑完。stream_mode="values" 每步产出完整状态，
    # 用 printed 做增量截取，只打印"本轮新增"的消息：
    # 人类消息 → AI 消息(可能含 tool_calls) → 工具结果消息 → ... → 最终 AI 回答
    printed = n_before + 1  # 跳过历史消息和本轮刚加入的用户消息
    for snapshot in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,  # 传 thread_id → 对话历史自动读写 checkpointer（短期记忆）
        stream_mode="values",
    ):
        for msg in snapshot["messages"][printed:]:
            printed += 1
            role = msg.__class__.__name__
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    args = tc["args"]
                    # 参数可能很长（write_file 的内容），打印时截断方便观察
                    args_text = str(args)
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
def chat(agent, store) -> None:
    thread_id = "main"  # 默认会话固定 id：重启进程后可接着聊（短期记忆持久化）
    print(f"模型: {MODEL_NAME} | 会话: {thread_id} | 工作目录: {WORKSPACE_DIR}")
    print("输入 /help 查看命令，/exit 退出\n")

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
                print("  /new [名字]  开启新会话（旧会话历史保留）")
                print("  /memory      查看长期记忆中保存的用户信息")
                print("  /history     查看当前会话的消息统计")
                print("  /exit        退出")
            elif cmd == "/new":
                thread_id = arg.strip() or f"session-{datetime.now():%Y%m%d-%H%M%S}"
                print(f"[已切换到新会话: {thread_id}]")
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
                    f"[会话 {thread_id} 共 {n} 条消息（已持久化到 {CHECKPOINT_DB}）]"
                )
            else:
                print(f"[未知命令 {cmd}，输入 /help 查看命令]")
            continue

        run_task(agent, user_input, thread_id)


def main():
    if not API_KEY:
        print("未检测到 AGICTO_API_KEY！请先设置环境变量。")
        return

    # SqliteSaver = 短期记忆（对话历史）  SqliteStore = 长期记忆（用户信息）
    # with 同时管理两个 sqlite 连接的生命周期（与 langchain_agent.py 相同）
    saver_cm = SqliteSaver.from_conn_string(CHECKPOINT_DB)
    store_cm = SqliteStore.from_conn_string(STORE_DB)
    with saver_cm as checkpointer, store_cm as store:
        agent = build_deep_agent(checkpointer, store)
        chat(agent, store)

    print("再见！对话历史、长期记忆和文件都已保存，下次启动依然有效。")


if __name__ == "__main__":
    main()
