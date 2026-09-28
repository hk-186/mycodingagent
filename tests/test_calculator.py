# -*- coding: utf-8 -*-
"""calculate 工具安全求值的首批单测（修复计划 E5：补零测试缺口）。"""

import ast

import pytest

from mycodingagent.tools.calculator import _safe_eval, calculate


def _eval_expr(expression: str):
    """辅助：按 calculate 内部的方式解析表达式再求值。"""
    return _safe_eval(ast.parse(expression, mode="eval").body)


class TestSafeEval:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("1 + 2", 3),
            ("(128 + 64) * 3", 576),
            ("2 ** 10", 1024),
            ("10 / 4", 2.5),
            ("10 // 3", 3),
            ("10 % 3", 1),
            ("-5 + 3", -2),
            ("+7 * 2", 14),
            ("1.5 * 4", 6.0),
            ("-(3 + 4) * 2", -14),
        ],
    )
    def test_arithmetic_ok(self, expression, expected):
        assert _eval_expr(expression) == expected

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('dir')",  # 函数调用注入
            "().__class__",  # 属性访问
            "1 + 'a'",  # 类型不匹配
            "'a' * 3",  # 字符串运算
            "[1, 2]",  # 非标量常量
            "lambda: 1",  # 非表达式节点
        ],
    )
    def test_rejects_unallowed(self, expression):
        with pytest.raises(ValueError):
            _eval_expr(expression)


class TestCalculateTool:
    def test_ok(self):
        result = calculate.invoke({"expression": "(128 + 64) * 3"})
        assert result == "(128 + 64) * 3 = 576"

    def test_division_by_zero_returns_message(self):
        result = calculate.invoke({"expression": "1 / 0"})
        assert result.startswith("计算失败：")

    def test_invalid_expression_returns_message(self):
        result = calculate.invoke({"expression": "__import__('os')"})
        assert result.startswith("计算失败：")

    def test_syntax_error_returns_message(self):
        result = calculate.invoke({"expression": "1 +"})
        assert result.startswith("计算失败：")
