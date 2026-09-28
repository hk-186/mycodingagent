# -*- coding: utf-8 -*-
"""
Agent 构建
==========
模型客户端、工具集、子代理与记忆的接线逻辑，
从 deep_agent.py 迁移并按集中配置改造。
"""

import logging
from datetime import datetime

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.config import get_store  # 工具运行时由 agent 注入 store

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from mycodingagent import config
from mycodingagent.tools.calculator import calculate

logger = logging.getLogger(__name__)


def get_llm() -> ChatOpenAI:
    """构建 LLM 客户端：temperature / timeout / 重试次数走集中配置（修复计划 E7）。"""
    return ChatOpenAI(
        model=config.MODEL_NAME,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
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
# 长期记忆的读写工具（信息存入 SqliteStore，跨会话共享）
# get_store() 在工具运行时取出 create_deep_agent(store=...) 注入的 store
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
    create_deep_agent 关键参数：

    - backend：FilesystemBackend 把 Agent 的文件读写限制在工作目录内
    - subagents：注册后主 Agent 自动获得 task 工具，按 description 决定何时委派
    - checkpointer / store：短期记忆（对话历史）/ 长期记忆（用户信息）
    - memory：AGENTS.md 风格的规则记忆，内容每轮注入 system prompt
    """
    llm = get_llm()

    # 确保工作目录存在（Agent 往里写文件时才有落脚点）
    config.WORKSPACE_DIR.mkdir(exist_ok=True)

    # 主 Agent 的工具：时间 + 长期记忆读写（+ 设置了密钥时的联网搜索）
    tools = [
        get_current_time,
        save_user_info,
        recall_user_info,
        recall_user_info_list,
    ]
    search_hint = ""
    if config.TAVILY_API_KEY:
        from langchain_tavily import TavilySearch

        tools.append(TavilySearch(max_results=3))
        search_hint = "- web_search：回答时效性/事实性问题前先用它查最新资料\n"
    else:
        logger.info("未设置 TAVILY_API_KEY，联网搜索功能未启用")

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
        backend=FilesystemBackend(root_dir=str(config.WORKSPACE_DIR)),
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
