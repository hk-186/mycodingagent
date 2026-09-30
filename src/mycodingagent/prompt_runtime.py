# -*- coding: utf-8 -*-
"""
运行时动态 prompt（阶段 3 增强）
================================
system prompt 在 agent 构造时生成一次静态文本，但其中几项内容会在运行时
变化（工作目录 /cd、审批模式、Plan 模式）。本模块用「占位符 + 每次调
模型前替换」解决：

- prompt 模板中写入占位符：
    __PROJECT_DIR__        当前工作目标目录
    __APPROVAL_MODE__      审批模式（minimal / all / disabled）
    __PLAN_MODE_LABEL__    Plan 模式开启时的短标记
    __PLAN_MODE_SUFFIX__   Plan 模式开启时的整段额外指令（关闭时为空）
- `RuntimePromptMiddleware.wrap_model_call` 在每次调用模型前，把
  system message 里的占位符替换成 config 当前值。

替换是纯文本操作，不影响 summarization / memory 等其它中间件对 prompt
的修改（本中间件插入在核心栈之后、那些尾部中间件之前，且只替换占位符）。
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from mycodingagent import config

# 占位符
PROJECT_DIR_PLACEHOLDER = "__PROJECT_DIR__"
APPROVAL_MODE_PLACEHOLDER = "__APPROVAL_MODE__"
PLAN_MODE_LABEL_PLACEHOLDER = "__PLAN_MODE_LABEL__"
PLAN_MODE_SUFFIX_PLACEHOLDER = "__PLAN_MODE_SUFFIX__"

_PLAN_MODE_SUFFIX = (
    "\n## Plan 模式（已开启）\n"
    "- 当前处于 Plan 模式：所有写操作（execute 写命令、edit_file、write_file、"
    "git_commit）都会被自动拦截等待用户审批，不会真正执行。\n"
    "- 你的任务是：先调研代码现状（read_file / ls / grep / glob 只读工具），"
    "形成清晰的改动计划，然后调用 propose_plan 工具提交计划等用户审批。\n"
    "- propose_plan 的 plan 参数应包含：要改的文件列表、每个文件的改动要点、"
    "验证方式（跑哪个测试/脚本）、潜在风险。\n"
    "- 计划被 approve 后才能开始动手；被 reject 要根据理由重规划。\n"
)


def render_runtime_fields(text: str) -> str:
    """把 system prompt 文本中的占位符替换为 config 当前运行时值。

    纯函数，便于单测；无占位符时原样返回。
    """
    if not text:
        return text
    text = text.replace(PROJECT_DIR_PLACEHOLDER, str(config.PROJECT_DIR))
    text = text.replace(APPROVAL_MODE_PLACEHOLDER, config.APPROVAL_MODE)
    text = text.replace(
        PLAN_MODE_LABEL_PLACEHOLDER,
        "（已开启 Plan 模式）" if config.PLAN_MODE else "",
    )
    text = text.replace(
        PLAN_MODE_SUFFIX_PLACEHOLDER,
        _PLAN_MODE_SUFFIX if config.PLAN_MODE else "",
    )
    return text


class RuntimePromptMiddleware(AgentMiddleware):
    """每次调模型前替换 system message 中的运行时占位符。"""

    @property
    def name(self) -> str:
        return "runtime_prompt"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> Any:
        system_message = request.system_message
        if system_message is not None:
            rendered = render_runtime_fields(system_message.text)
            request = request.override(system_message=SystemMessage(content=rendered))
        return handler(request)
