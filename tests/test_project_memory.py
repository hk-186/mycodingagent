# -*- coding: utf-8 -*-
"""项目级记忆（命名空间隔离 + 语义检索）测试（阶段 5）。

包含：
- config.project_namespace：项目 slug 与 /cd 跟随；
- 三个 project_memory 工具（FakeStore，验证命名空间隔离与语义查询）；
- SqliteStore + 确定性 FakeEmbeddings 的离线语义索引集成（不依赖网络）。
"""

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.sqlite import SqliteStore

from mycodingagent import config
from mycodingagent.tools.project_memory import (
    list_project_facts,
    recall_project_fact,
    save_project_fact,
)


# ============================================================
# config.project_namespace
# ============================================================
def test_project_namespace_uses_slug(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    ns = config.project_namespace()
    assert ns[0] == "projects" and ns[-1] == "facts"
    assert ns[1]  # 非空 slug
    assert ":" not in ns[1] and "/" not in ns[1] and "\\" not in ns[1]


def test_project_namespace_differs_per_project(monkeypatch, tmp_path):
    a = tmp_path / "alpha"
    b = tmp_path / "beta"
    a.mkdir(); b.mkdir()
    monkeypatch.setattr(config, "PROJECT_DIR", a)
    ns_a = config.project_namespace()
    monkeypatch.setattr(config, "PROJECT_DIR", b)
    assert config.project_namespace() != ns_a


# ============================================================
# FakeStore：模拟带命名空间与语义查询的 store
# ============================================================
class _Item:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class FakeStore:
    def __init__(self):
        self._items: dict[tuple, _Item] = {}
        self.last_index = None

    def put(self, namespace, key, value, index=None):
        self._items[(tuple(namespace), key)] = _Item(key, value)
        self.last_index = index

    def get(self, namespace, key):
        return self._items.get((tuple(namespace), key))

    def search(self, namespace, query=None, limit=10, **_kw):
        items = [v for (ns, _k), v in self._items.items() if ns == tuple(namespace)]
        if query is not None:
            qt = set(query.lower().split())

            def score(item):
                vt = set(item.value.get("value", "").lower().split()) | {item.key.lower()}
                return len(qt & vt)

            items = [i for i in items if score(i) > 0]
            items.sort(key=score, reverse=True)
        return items[:limit]


@pytest.fixture()
def fake_store(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(
        "mycodingagent.tools.project_memory.get_store", lambda: store
    )
    return store


# ============================================================
# 三个项目记忆工具
# ============================================================
def test_save_project_fact(fake_store):
    out = save_project_fact.invoke({"key": "test_command", "value": "pytest -q"})
    assert "已记住" in out
    # 依赖全局 text_fields=[index_text]，不再显式传 index
    assert fake_store.last_index is None
    # 落在项目命名空间且记录结构正确
    item = fake_store.get(config.project_namespace(), "test_command")
    assert item is not None
    assert item.value["value"] == "pytest -q"
    assert item.value["index_text"] == "test_command：pytest -q"


def test_recall_project_fact_semantic(fake_store):
    save_project_fact.invoke({"key": "test_command", "value": "run pytest quietly"})
    save_project_fact.invoke({"key": "deploy", "value": "deploy to production server"})
    out = recall_project_fact.invoke({"query": "pytest command"})
    assert "run pytest quietly" in out
    assert "production" not in out


def test_recall_project_fact_empty(fake_store):
    out = recall_project_fact.invoke({"query": "任何内容"})
    assert "没有找到" in out


def test_list_project_facts(fake_store):
    save_project_fact.invoke({"key": "a", "value": "aaa"})
    save_project_fact.invoke({"key": "b", "value": "bbb"})
    out = list_project_facts.invoke({})
    assert "aaa" in out and "bbb" in out


def test_list_project_facts_empty(fake_store):
    assert "还没有" in list_project_facts.invoke({})


def test_project_facts_isolated_by_namespace(fake_store, monkeypatch, tmp_path):
    """两个项目各存同名 key，互不串扰。"""
    proj_a = tmp_path / "a"; proj_b = tmp_path / "b"
    proj_a.mkdir(); proj_b.mkdir()

    monkeypatch.setattr(config, "PROJECT_DIR", proj_a)
    save_project_fact.invoke({"key": "style", "value": "A 项目约定"})

    monkeypatch.setattr(config, "PROJECT_DIR", proj_b)
    assert "还没有" in list_project_facts.invoke({})
    save_project_fact.invoke({"key": "style", "value": "B 项目约定"})

    monkeypatch.setattr(config, "PROJECT_DIR", proj_a)
    assert "A 项目约定" in list_project_facts.invoke({})


# ============================================================
# 离线语义索引集成：SqliteStore + 确定性 Embeddings（sqlite-vec）
# ============================================================
class _DeterministicEmbeddings(Embeddings):
    """按词哈希到固定维度的确定性向量（离线、可重复）。"""

    dims = 16

    @staticmethod
    def _tokens(text):
        return text.lower().split()

    def _vec(self, text):
        vec = [0.0] * self.dims
        for tok in self._tokens(text):
            idx = sum(ord(c) for c in tok) % self.dims
            vec[idx] = 1.0
        return vec

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_sqlite_semantic_index_roundtrip(tmp_path):
    from mycodingagent.memory_common import INDEX_TEXT_FIELD, make_record

    db = str(tmp_path / "mem.sqlite")
    with SqliteStore.from_conn_string(
        db,
        index={
            "dims": _DeterministicEmbeddings.dims,
            "embed": _DeterministicEmbeddings(),
            "text_fields": [INDEX_TEXT_FIELD],
        },
    ) as store:
        ns = ("projects", "x", "facts")
        store.put(
            ns, "tests",
            make_record("tests", "run tests with pytest command"),
            index=[INDEX_TEXT_FIELD],
        )
        store.put(
            ns, "deploy",
            make_record("deploy", "deploy app to production server"),
            index=[INDEX_TEXT_FIELD],
        )
        hits = store.search(ns, query="pytest command for tests", limit=2)
        assert hits, "语义查询应返回结果"
        assert hits[0].key == "tests"
