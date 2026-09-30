# -*- coding: utf-8 -*-
"""
项目级长期记忆工具（阶段 5）
============================
与个人全局记忆（save_user_info，命名空间 ("users",)）不同，本模块的记忆
按项目命名空间隔离（config.project_namespace()）：测试命令、代码风格、
架构决策等项目约定只在本项目可见，/cd 切换项目后自动指向新命名空间。

查询走 store 语义索引（C13）：save 时对 ["value"] 字段建立向量索引，
recall 用自然语言语义检索，不再做全量扫描。
"""

from langchain_core.tools import tool
from langgraph.config import get_store

from mycodingagent import config


@tool
def save_project_fact(key: str, value: str) -> str:
    """保存「当前项目」的约定或事实（仅本项目可见）。

    key 是约定类别（如 test_command、code_style、architecture、deploy_notes），
    value 是具体内容。个人属性（姓名/邮箱等）请用 save_user_info。
    """
    store = get_store()
    store.put(config.project_namespace(), key, {"value": value}, index=["value"])
    return f"已记住本项目约定：{key} = {value}"


@tool
def recall_project_fact(query: str) -> str:
    """语义检索「当前项目」的约定。可用自然语言描述想找的内容。

    例如 query 可以是「这个项目怎么跑测试」「代码风格有什么要求」。
    """
    store = get_store()
    items = store.search(config.project_namespace(), query=query, limit=5)
    if not items:
        return f"本项目没有找到与「{query}」相关的记忆"
    return "；".join(f"{item.key} = {item.value['value']}" for item in items)


@tool
def list_project_facts() -> str:
    """列出当前项目已保存的全部约定。"""
    store = get_store()
    items = store.search(config.project_namespace())
    if not items:
        return "本项目还没有保存任何约定"
    return "；".join(f"{item.key} = {item.value['value']}" for item in items)
