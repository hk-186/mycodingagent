# -*- coding: utf-8 -*-
"""memory_common 纯函数 + 个人记忆工具 recall_user_info 检索顺序测试。

覆盖本次修复（key 不参与向量导致语义错位）：
- make_record / record_value / match_record_keys / format_items；
- recall_user_info 的优先级：精确 key → key 子串匹配兜底 → 语义检索 → 无结果。
"""

import pytest

from mycodingagent.tools import memory_common
from mycodingagent.tools.memory_common import INDEX_TEXT_FIELD, VALUE_FIELD
from mycodingagent.tools.user_memory import (
    recall_user_info,
    recall_user_info_list,
    save_user_info,
)


# ============================================================
# 纯函数
# ============================================================
def test_make_record_contains_value_and_index_text():
    rec = memory_common.make_record("name", "Kevin")
    assert rec[VALUE_FIELD] == "Kevin"
    assert rec[INDEX_TEXT_FIELD] == "name：Kevin"


def test_record_value_reads_value_field():
    assert memory_common.record_value({VALUE_FIELD: "x"}) == "x"


def test_record_value_tolerates_legacy_record():
    # 旧结构（无 index_text）也能读
    assert memory_common.record_value({VALUE_FIELD: "legacy"}) == "legacy"


def test_match_record_keys_case_insensitive():
    items = [_SimpleItem("name", {}), _SimpleItem("email", {})]
    hits = memory_common.match_record_keys("用户的 NAME 是什么", items)
    assert [h.key for h in hits] == ["name"]


def test_match_record_keys_no_match_returns_empty():
    items = [_SimpleItem("name", {})]
    assert memory_common.match_record_keys("无关问题", items) == []


def test_match_record_keys_empty_query():
    items = [_SimpleItem("name", {})]
    assert memory_common.match_record_keys("", items) == []


def test_match_record_keys_preserves_order():
    items = [_SimpleItem("a", {}), _SimpleItem("b", {}), _SimpleItem("c", {})]
    hits = memory_common.match_record_keys("b 和 a", items)
    assert [h.key for h in hits] == ["a", "b"]


def test_format_items_uses_value_only():
    items = [
        _SimpleItem("name", {VALUE_FIELD: "Kevin", INDEX_TEXT_FIELD: "name：Kevin"}),
        _SimpleItem("learning", {VALUE_FIELD: "L", INDEX_TEXT_FIELD: "learning：L"}),
    ]
    out = memory_common.format_items(items)
    assert out == "name = Kevin；learning = L"


class _SimpleItem:
    def __init__(self, key, value):
        self.key = key
        self.value = value


# ============================================================
# FakeStore：个人记忆工具的检索顺序集成
# ============================================================
class _Item:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class FakeStore:
    def __init__(self):
        self._items: dict[tuple, _Item] = {}

    def put(self, namespace, key, value, index=None):  # noqa: ARG002
        self._items[(tuple(namespace), key)] = _Item(key, value)

    def get(self, namespace, key):
        return self._items.get((tuple(namespace), key))

    def search(self, namespace, query=None, limit=10, **_kw):
        items = [v for (ns, _k), v in self._items.items() if ns == tuple(namespace)]
        if query is not None:
            q = query.lower()

            def score(item):
                text = (
                    item.value.get(INDEX_TEXT_FIELD, "")
                    + " "
                    + item.value.get(VALUE_FIELD, "")
                ).lower()
                return sum(1 for tok in q.split() if tok in text)

            items = [i for i in items if score(i) > 0]
            items.sort(key=score, reverse=True)
        return items[:limit]


@pytest.fixture()
def fake_store(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr("mycodingagent.tools.user_memory.get_store", lambda: store)
    return store


def _seed(store):
    save_user_info.invoke({"key": "name", "value": "Kevin"})
    save_user_info.invoke({"key": "learning", "value": "正在学习 LangChain"})


def test_recall_exact_key(fake_store):
    _seed(fake_store)
    out = recall_user_info.invoke({"query": "name"})
    assert out == "name = Kevin"


def test_recall_key_substring_fallback(fake_store):
    # query 是一整句、夹带了 key —— 精确 get 失败，应由 key 匹配兜底命中
    _seed(fake_store)
    out = recall_user_info.invoke({"query": "用户的姓名 name"})
    assert "name = Kevin" in out


def test_recall_semantic_fallback(fake_store):
    # query 不含任何 key 子串 —— 走语义检索
    _seed(fake_store)
    out = recall_user_info.invoke({"query": "学习 langchain 的情况"})
    assert "learning = 正在学习 LangChain" in out
    assert "Kevin" not in out


def test_recall_no_result(fake_store):
    _seed(fake_store)
    out = recall_user_info.invoke({"query": "完全无关内容 xyz"})
    assert "没有找到" in out


def test_recall_list_and_empty(fake_store):
    assert "还没有" in recall_user_info_list.invoke({})
    _seed(fake_store)
    out = recall_user_info_list.invoke({})
    assert "name = Kevin" in out and "learning = 正在学习 LangChain" in out
