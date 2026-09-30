# -*- coding: utf-8 -*-
"""ProjectMemoryMiddleware（/cd 跟随重载 AGENTS.md）与 agent 构图测试（阶段 5）。"""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from mycodingagent import config
from mycodingagent.agent import build_deep_agent
from mycodingagent.project_middleware import ProjectMemoryMiddleware
from mycodingagent.tools.shell import SafeShellBackend


class _FakeChatModel(BaseChatModel):
    """支持 bind_tools 的最小假模型：返回无工具调用的 AIMessage，让图直接结束。

    用于真实执行图，覆盖框架对 before_agent 节点的实际调用路径。
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="结束"))])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self):
        return "fake-chat"


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
    out = middleware.before_agent({}, None)
    assert out["project_memory_path"] == str(tmp_path)
    assert "/AGENTS.md" in out["memory_contents"]
    assert "项目规则" in out["memory_contents"]["/AGENTS.md"]


def test_same_project_uses_cache(middleware, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    state = {"memory_contents": {"/AGENTS.md": "旧内容"},
             "project_memory_path": str(tmp_path)}
    assert middleware.before_agent(state, None) is None
    assert middleware._backend.calls == 0  # 不读盘


def test_project_switch_reloads(middleware, monkeypatch, tmp_path):
    proj_a = tmp_path / "a"; proj_b = tmp_path / "b"
    proj_a.mkdir(); proj_b.mkdir()
    monkeypatch.setattr(config, "PROJECT_DIR", proj_a)
    middleware.before_agent({}, None)
    calls_after_first = middleware._backend.calls

    monkeypatch.setattr(config, "PROJECT_DIR", proj_b)
    out = middleware.before_agent(
        {"memory_contents": {"/AGENTS.md": "A"}, "project_memory_path": str(proj_a)},
        None,
    )
    assert out is not None
    assert out["project_memory_path"] == str(proj_b)
    assert middleware._backend.calls == calls_after_first + 1


def test_file_not_found_skipped(tmp_path):
    mw = ProjectMemoryMiddleware(backend=_FakeBackend(
        _Response(error="file_not_found")
    ))
    out = mw.before_agent({}, None)
    assert out["memory_contents"] == {}


def test_other_error_raises():
    mw = ProjectMemoryMiddleware(backend=_FakeBackend(
        _Response(error="permission_denied")
    ))
    with pytest.raises(ValueError):
        mw.before_agent({}, None)


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


# ============================================================
# 图级回归：真实执行框架的 before_agent 节点（旧的三参数签名会在此抛
# TypeError：missing positional argument '_cfg'）
# ============================================================
def test_before_agent_runs_inside_real_graph(tmp_path, monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    (tmp_path / "AGENTS.md").write_text("# 图级规则", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path.resolve())
    monkeypatch.setattr("mycodingagent.agent.get_llm", lambda: _FakeChatModel())

    backend = SafeShellBackend(root_dir=str(tmp_path), inherit_env=True)
    agent = build_deep_agent(InMemorySaver(), InMemoryStore(), backend=backend)

    cfg = {"configurable": {"thread_id": "graph-1"}, "recursion_limit": 10}
    result = agent.invoke({"messages": [("user", "hi")]}, cfg)
    assert result["messages"][-1].content == "结束"

    snap = agent.get_state(cfg)
    memory = snap.values.get("memory_contents") if snap.values else {}
    assert memory.get("/AGENTS.md") == "# 图级规则"
