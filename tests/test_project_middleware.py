# -*- coding: utf-8 -*-
"""ProjectMemoryMiddleware（/cd 跟随重载 AGENTS.md）与 agent 构图测试（阶段 5）。"""

import pytest

from mycodingagent import config
from mycodingagent.agent import build_deep_agent
from mycodingagent.project_middleware import ProjectMemoryMiddleware
from mycodingagent.tools.shell import SafeShellBackend


# ============================================================
# Fake backend / response
# ============================================================
class _Response:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error


class _FakeBackend:
    def __init__(self, response):
        self._response = response
        self.calls = 0

    def download_files(self, paths):
        self.calls += 1
        return [self._response for _ in paths]


@pytest.fixture()
def middleware():
    return ProjectMemoryMiddleware(backend=_FakeBackend(
        _Response(content="# 项目规则".encode("utf-8"))
    ))


# ============================================================
# before_agent：首次加载 / 同目录缓存 / 换目录重载
# ============================================================
def test_first_load_returns_contents_and_path(middleware, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    out = middleware.before_agent({}, None, None)
    assert out["project_memory_path"] == str(tmp_path)
    assert "/AGENTS.md" in out["memory_contents"]
    assert "项目规则" in out["memory_contents"]["/AGENTS.md"]


def test_same_project_uses_cache(middleware, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    state = {"memory_contents": {"/AGENTS.md": "旧内容"},
             "project_memory_path": str(tmp_path)}
    assert middleware.before_agent(state, None, None) is None
    assert middleware._backend.calls == 0  # 不读盘


def test_project_switch_reloads(middleware, monkeypatch, tmp_path):
    proj_a = tmp_path / "a"; proj_b = tmp_path / "b"
    proj_a.mkdir(); proj_b.mkdir()
    monkeypatch.setattr(config, "PROJECT_DIR", proj_a)
    middleware.before_agent({}, None, None)
    calls_after_first = middleware._backend.calls

    monkeypatch.setattr(config, "PROJECT_DIR", proj_b)
    out = middleware.before_agent(
        {"memory_contents": {"/AGENTS.md": "A"}, "project_memory_path": str(proj_a)},
        None, None,
    )
    assert out is not None
    assert out["project_memory_path"] == str(proj_b)
    assert middleware._backend.calls == calls_after_first + 1


def test_file_not_found_skipped(tmp_path):
    mw = ProjectMemoryMiddleware(backend=_FakeBackend(
        _Response(error="file_not_found")
    ))
    out = mw.before_agent({}, None, None)
    assert out["memory_contents"] == {}


def test_other_error_raises():
    mw = ProjectMemoryMiddleware(backend=_FakeBackend(
        _Response(error="permission_denied")
    ))
    with pytest.raises(ValueError):
        mw.before_agent({}, None, None)


# ============================================================
# agent 构图：验证子代理 / middleware / 工具注册整体合法
# （create_deep_agent 只构图，不调用模型，离线可跑）
# ============================================================
def test_build_deep_agent_with_new_subagents(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    backend = SafeShellBackend(root_dir=str(tmp_path), inherit_env=True)
    agent = build_deep_agent(InMemorySaver(), InMemoryStore(), backend=backend)

    # 工具挂在 tools 节点的 ToolNode 上
    tool_names = set(agent.nodes["tools"].bound.tools_by_name)
    # 子代理统一入口 task（code-reviewer / test-writer 注册成功）
    assert "task" in tool_names
    # 阶段 5 新工具
    assert {"save_project_fact", "recall_project_fact", "list_project_facts"} <= tool_names
    # 阶段 3 工具仍在
    assert {"ask_user", "propose_plan", "git_commit"} <= tool_names
