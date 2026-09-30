# -*- coding: utf-8 -*-
"""CLI 决策解析与收集测试（阶段 3）。

覆盖 `cli._parse_decision_input` 与 `cli._collect_decisions_for_interrupt`
两个纯逻辑函数，不依赖真实 agent。
"""

import pytest

from mycodingagent.cli import _collect_decisions_for_interrupt, _parse_decision_input
from mycodingagent.events import AgentEvent, INTERRUPT


# ============================================================
# _parse_decision_input：approve / reject / respond / 简写 / 无效
# ============================================================
class TestParseDecisionInput:
    @pytest.mark.parametrize("text", ["/approve", "a", "y", "yes", "/APPROVE", "Y"])
    def test_approve_forms(self, text):
        assert _parse_decision_input(text) == {"type": "approve"}

    @pytest.mark.parametrize("text", ["/reject", "r", "n", "no"])
    def test_reject_without_reason(self, text):
        assert _parse_decision_input(text) == {"type": "reject"}

    def test_reject_with_reason_slash_form(self):
        assert _parse_decision_input("/reject 危险") == {"type": "reject", "message": "危险"}

    def test_reject_with_reason_short_form(self):
        # "r my reason" → reject + reason
        assert _parse_decision_input("r my reason") == {"type": "reject", "message": "my reason"}

    def test_respond_slash_form(self):
        assert _parse_decision_input("/respond 是的") == {"type": "respond", "message": "是的"}

    def test_respond_short_form(self):
        # "rr yes" → respond + "yes"（取 text[3:]，保留大小写）
        assert _parse_decision_input("rr yes") == {"type": "respond", "message": "yes"}

    def test_respond_default_when_plain_text(self):
        """非命令、非简写的纯文本默认当作 respond。"""
        assert _parse_decision_input("just text") == {"type": "respond", "message": "just text"}

    def test_respond_requires_text(self, capsys):
        assert _parse_decision_input("/respond") is None
        captured = capsys.readouterr()
        assert "/respond" in captured.out

    def test_unknown_command_returns_none(self, capsys):
        assert _parse_decision_input("/foobar") is None
        captured = capsys.readouterr()
        assert "未知决策命令" in captured.out

    @pytest.mark.parametrize("text", ["", "   ", "\t"])
    def test_empty_input_returns_none(self, text):
        assert _parse_decision_input(text) is None

    def test_case_insensitive_command(self):
        """命令部分大小写不敏感，但 respond 的 message 保留原大小写。"""
        assert _parse_decision_input("/Reject 风险") == {"type": "reject", "message": "风险"}
        assert _parse_decision_input("/REJECT 风险") == {"type": "reject", "message": "风险"}


# ============================================================
# _collect_decisions_for_interrupt：基于 monkeypatch input
# ============================================================
def _interrupt_event(action_requests):
    """构造 INTERRUPT AgentEvent。"""
    return AgentEvent(
        type=INTERRUPT,
        text="",
        action_requests=action_requests,
    )


def test_collect_single_action_request_approve(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "/approve")
    ev = _interrupt_event([
        {"name": "execute", "args": {"command": "rm x"}, "description": "删 x"}
    ])
    decisions = _collect_decisions_for_interrupt(ev)
    assert decisions == [{"type": "approve"}]


def test_collect_multiple_action_requests(monkeypatch):
    """两个 action_request：第一个 approve，第二个 reject reason。"""
    inputs = iter(["/approve", "/reject 不行"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    ev = _interrupt_event([
        {"name": "execute", "args": {"command": "rm x"}, "description": "删 x"},
        {"name": "git_commit", "args": {"message": "fix"}, "description": "提交"},
    ])
    decisions = _collect_decisions_for_interrupt(ev)
    assert len(decisions) == 2
    assert decisions[0] == {"type": "approve"}
    assert decisions[1] == {"type": "reject", "message": "不行"}


def test_collect_cancel_keyword_returns_none(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "cancel")
    ev = _interrupt_event([
        {"name": "execute", "args": {"command": "rm x"}, "description": "删 x"}
    ])
    assert _collect_decisions_for_interrupt(ev) is None


def test_collect_abort_keyword_returns_none(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "abort")
    ev = _interrupt_event([
        {"name": "execute", "args": {"command": "rm x"}, "description": "删 x"}
    ])
    assert _collect_decisions_for_interrupt(ev) is None


def test_collect_eof_returns_none(monkeypatch):
    """Ctrl+D / EOF 视为取消。"""
    def raise_eof(_):
        raise EOFError()
    monkeypatch.setattr("builtins.input", raise_eof)
    ev = _interrupt_event([
        {"name": "execute", "args": {"command": "rm x"}, "description": "删 x"}
    ])
    assert _collect_decisions_for_interrupt(ev) is None


def test_collect_keyboard_interrupt_returns_none(monkeypatch):
    """Ctrl+C 视为取消。"""
    def raise_kbi(_):
        raise KeyboardInterrupt()
    monkeypatch.setattr("builtins.input", raise_kbi)
    ev = _interrupt_event([
        {"name": "execute", "args": {"command": "rm x"}, "description": "删 x"}
    ])
    assert _collect_decisions_for_interrupt(ev) is None


def test_collect_invalid_then_valid_retries(monkeypatch):
    """无效输入后会再次询问直到拿到有效决策（非命令纯文本会被当 respond，所以用 /unknown 触发真正无效）。"""
    inputs = iter(["/unknown", "/approve"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    ev = _interrupt_event([
        {"name": "execute", "args": {"command": "rm x"}, "description": "删 x"}
    ])
    decisions = _collect_decisions_for_interrupt(ev)
    assert decisions == [{"type": "approve"}]


def test_collect_empty_action_requests_returns_empty_list(monkeypatch):
    """无 action_request 的 INTERRUPT 直接返回空 decisions 列表，不询问 input。"""
    called = {"count": 0}

    def fake_input(_):
        called["count"] += 1
        return "/approve"
    monkeypatch.setattr("builtins.input", fake_input)
    ev = _interrupt_event([])
    decisions = _collect_decisions_for_interrupt(ev)
    assert decisions == []
    assert called["count"] == 0  # 不应该调用 input


def test_collect_respond_for_ask_user(monkeypatch):
    """ask_user 工具的 respond 决策：用户输入文本作为 message。"""
    monkeypatch.setattr("builtins.input", lambda _: "请按方案 A 执行")
    ev = _interrupt_event([
        {"name": "ask_user", "args": {"question": "用哪个方案？"}, "description": "提问"}
    ])
    decisions = _collect_decisions_for_interrupt(ev)
    assert decisions == [{"type": "respond", "message": "请按方案 A 执行"}]
