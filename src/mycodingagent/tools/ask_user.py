# -*- coding: utf-8 -*-
"""
ask_user 工具（阶段 3 缺口 C8 人机协作层）
==========================================
`ask_user(question)`：信息不足时主动向用户澄清。
工具体本身不会被执行——`HumanInTheLoopMiddleware` 通过 `interrupt_on` 配置
拦截该工具的调用，把问题作为 `ActionRequest` 推给 CLI；用户用 `respond` 决策
回答后，回答文本作为 `ToolMessage.content` 返给模型，工具执行被跳过。

`return ""` 仅为满足 type 签名，运行时不会到达。
"""

from langchain_core.tools import tool


@tool
def ask_user(question: str) -> str:
    """信息不足时向用户提问，等待用户回答后再继续。

    何时调用：
    - 用户需求中有歧义（例如「加个新功能」但未说哪个）；
    - 缺少关键决策信息（目标文件位置、API 设计偏好等）；
    - 多个合理实现方案需要让用户选择。

    不要为已经能从代码/上下文确定的小事问用户。
    问题要具体、可一句话回答，不要开放式泛问。
    """
    return ""  # unreachable：被 interrupt_on 中间件拦截，由 respond 决策代答
