# -*- coding: utf-8 -*-
"""
长期记忆共享约定（阶段 5 修复）
================================

个人记忆（agent.py，命名空间 ("users",)）与项目记忆（tools/project_memory.py，
命名空间 config.project_namespace()）共用同一套 key-value 结构与检索策略。

修复的问题：
    原先向量只编码 value 内容（text_fields=["value"]），store 的 key 不参与
    嵌入。当 value 是 "Kevin"、邮箱这类短专有名词时，自然语言查询（如
    「用户的姓名 name」）与它的语义距离很远，反而更容易命中大段中文描述，
    导致语义检索错位。

方案：
    1. 写入时在记录里额外存一份「key：value」组合文本（INDEX_TEXT_FIELD），
       向量索引该字段——使 key（name、test_command 等）也能被语义命中；
    2. 检索时在语义查询前先做 key 子串匹配兜底：query 中直接出现某个已知
       key 就精确返回，不依赖语义分数。

记录结构：
    {"value": <真实内容 str>, "index_text": "<key>：<value>"}
真实内容永远取 record["value"]，index_text 仅用于向量索引。
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol

# 供向量索引的组合文本字段名（全局 index config 的 text_fields 指向它）。
INDEX_TEXT_FIELD = "index_text"
# 真实内容字段名。
VALUE_FIELD = "value"


class _ItemLike(Protocol):
    """search 返回 Item 的最小结构（便于类型标注与单测替身）。"""

    key: str
    value: dict[str, Any]


def make_record(key: str, value: str) -> dict[str, str]:
    """构造写入 store 的记录：真实内容 + 含 key 的组合索引文本。"""
    return {
        VALUE_FIELD: value,
        INDEX_TEXT_FIELD: f"{key}：{value}",
    }


def record_value(record: dict[str, Any]) -> str:
    """从记录中取真实内容（容错：缺失 index_text 的旧结构也能读）。"""
    return str(record.get(VALUE_FIELD, ""))


def match_record_keys(query: str, items: Iterable[_ItemLike]) -> list[_ItemLike]:
    """key 子串匹配兜底：query 中直接出现某 item 的 key 即命中。

    大小写不敏感。用于自然语言 query 里夹带了明确 key 的情形，例如
    query="用户的姓名 name" 命中 key="name"。保持 items 原有顺序。
    """
    q = (query or "").lower()
    if not q:
        return []
    return [item for item in items if item.key and item.key.lower() in q]


def format_items(items: Iterable[_ItemLike]) -> str:
    """把条目拼成 'key = value；...' 的展示文本（只取真实内容字段）。"""
    return "；".join(f"{item.key} = {record_value(item.value)}" for item in items)
