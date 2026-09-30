# -*- coding: utf-8 -*-
"""
个人级长期记忆工具（跨项目共享）
=================================
信息存入 SqliteStore 的命名空间 ("users",)，跨项目共享：姓名、邮箱、爱好、
沟通风格等用户长期个人信息。与项目级记忆（tools/project_memory.py，按项目
命名空间隔离）对称。

写入时记录为「key：value」组合文本并对其建立向量索引（key 也能被语义命中），
recall 的检索顺序：精确 key → key 子串匹配兜底 → 语义向量检索。
"""

from langchain_core.tools import tool
from langgraph.config import get_store

from mycodingagent.tools import memory_common

# 个人信息固定命名空间：所有项目共享同一份用户画像。
_USER_NAMESPACE = ("users",)


@tool
def save_user_info(key: str, value: str) -> str:
    """保存用户的长期个人信息（跨项目共享）。key 是信息类别（如 name、email、hobby），value 是具体内容"""
    store = get_store()
    store.put(_USER_NAMESPACE, key, memory_common.make_record(key, value))
    return f"已记住：{key} = {value}"


@tool
def recall_user_info(query: str) -> str:
    """回忆用户的长期个人信息，可用精确类别名或自然语言描述（语义检索）。

    项目相关的约定请用 recall_project_fact。
    """
    store = get_store()
    # 1) 精确按 key 取
    item = store.get(_USER_NAMESPACE, query)
    if item is not None:
        return f"{query} = {memory_common.record_value(item.value)}"
    # 2) key 子串匹配兜底：query 里夹带了明确 key（如「用户的姓名 name」）
    all_items = store.search(_USER_NAMESPACE)
    key_hits = memory_common.match_record_keys(query, all_items)
    if key_hits:
        return memory_common.format_items(key_hits)
    # 3) 语义索引检索（向量同时编码了 key 与 value）
    hits = store.search(_USER_NAMESPACE, query=query, limit=5)
    if hits:
        return memory_common.format_items(hits)
    return f"没有找到与「{query}」相关的记忆"


@tool
def recall_user_info_list() -> str:
    """列出所有已保存的用户长期信息（显式列举，非模糊查询）"""
    store = get_store()
    items = store.search(_USER_NAMESPACE)
    if not items:
        return "还没有保存任何用户信息"
    return memory_common.format_items(items)
