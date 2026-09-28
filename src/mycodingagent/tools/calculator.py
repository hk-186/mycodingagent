# -*- coding: utf-8 -*-
"""
安全计算器工具
==============
ast 白名单解析，只允许四则运算，防止代码注入。
从 deep_agent.py 原样迁移，行为保持不变。
"""

import ast
import operator

from langchain_core.tools import tool

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("表达式中含有不允许的元素")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，支持 + - * / // % ** 和括号。输入示例：(128 + 64) * 3"""
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"
