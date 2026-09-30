# -*- coding: utf-8 -*-
"""
propose_plan 工具（阶段 3 缺口 C6/C8）
======================================
`propose_plan(plan)`：Plan 模式下输出改动计划，等用户审批。
工具体本身不执行任何操作——`HumanInTheLoopMiddleware` 通过 `interrupt_on`
拦截该工具调用，把 plan 文本作为 `ActionRequest` 推给 CLI；用户审批后
（approve / reject），中间件决定是否把工具调用结果返给模型。

`return "计划已批准，开始执行"` 仅为满足 type 签名，运行时不会到达；
被 reject 时由中间件生成 error `ToolMessage` 引导模型重规划。
"""

from langchain_core.tools import tool


@tool
def propose_plan(plan: str) -> str:
    """向用户提交改动计划等待审批。仅在 Plan 模式下使用。

    plan 内容应包含：
    - 即将修改/新增的文件列表与各自改动要点；
    - 验证方式（跑哪个测试/脚本）；
    - 潜在风险或副作用（仅当有特别值得提的）。

    不要在非 Plan 模式下调用本工具；不要把已经完成的工作写成计划。
    """
    return "计划已批准，开始执行"  # unreachable：被 interrupt_on 中间件拦截
