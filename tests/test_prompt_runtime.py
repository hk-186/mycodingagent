# -*- coding: utf-8 -*-
"""动态 prompt 占位符替换与 RuntimePromptMiddleware 测试。"""

import pytest

from mycodingagent import config
from mycodingagent.prompt_runtime import (
    APPROVAL_MODE_PLACEHOLDER,
    PLAN_MODE_LABEL_PLACEHOLDER,
    PLAN_MODE_SUFFIX_PLACEHOLDER,
    PROJECT_DIR_PLACEHOLDER,
    RuntimePromptMiddleware,
    render_runtime_fields,
)

TEMPLATE = (
    "目录：" + PROJECT_DIR_PLACEHOLDER + "\n"
    "模式：" + APPROVAL_MODE_PLACEHOLDER + PLAN_MODE_LABEL_PLACEHOLDER + "\n"
    + PLAN_MODE_SUFFIX_PLACEHOLDER
)


# ============================================================
# render_runtime_fields：占位符替换
# ============================================================
def test_replaces_project_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    out = render_runtime_fields(TEMPLATE)
    assert str(tmp_path) in out
    assert PROJECT_DIR_PLACEHOLDER not in out


def test_replaces_approval_mode(monkeypatch):
    monkeypatch.setattr(config, "APPROVAL_MODE", "all")
    out = render_runtime_fields(TEMPLATE)
    assert "all" in out
    assert APPROVAL_MODE_PLACEHOLDER not in out


def test_plan_mode_off_label_and_suffix_empty(monkeypatch):
    monkeypatch.setattr(config, "PLAN_MODE", False)
    out = render_runtime_fields(TEMPLATE)
    assert "（已开启 Plan 模式）" not in out
    assert "propose_plan" not in out
    assert PLAN_MODE_LABEL_PLACEHOLDER not in out
    assert PLAN_MODE_SUFFIX_PLACEHOLDER not in out


def test_plan_mode_on_label_and_suffix_present(monkeypatch):
    monkeypatch.setattr(config, "PLAN_MODE", True)
    out = render_runtime_fields(TEMPLATE)
    assert "（已开启 Plan 模式）" in out
    assert "propose_plan" in out  # suffix 正文


def test_empty_text_passthrough():
    assert render_runtime_fields("") == ""


def test_text_without_placeholder_unchanged():
    text = "普通 prompt，没有占位符"
    assert render_runtime_fields(text) == text


# ============================================================
# RuntimePromptMiddleware：每次 wrap_model_call 替换 system message
# ============================================================
class _FakeSystemMessage:
    def __init__(self, text: str):
        self.text = text


class _FakeRequest:
    def __init__(self, text: str):
        self.system_message = _FakeSystemMessage(text)

    def override(self, *, system_message):
        new = _FakeRequest.__new__(_FakeRequest)
        new.system_message = system_message
        return new


def test_middleware_renders_system_message(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    seen: dict[str, str] = {}

    def handler(request):
        seen["text"] = request.system_message.text
        return "done"

    result = RuntimePromptMiddleware().wrap_model_call(
        _FakeRequest("目录 " + PROJECT_DIR_PLACEHOLDER), handler
    )
    assert result == "done"
    assert str(tmp_path) in seen["text"]
    assert PROJECT_DIR_PLACEHOLDER not in seen["text"]


def test_middleware_name():
    assert RuntimePromptMiddleware().name == "runtime_prompt"
