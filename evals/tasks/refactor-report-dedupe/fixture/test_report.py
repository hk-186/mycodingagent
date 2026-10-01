# -*- coding: utf-8 -*-
from report import render_expenses, render_sales

ROWS = [
    {"name": "apple", "amount": 10},
    {"name": "banana", "amount": 20},
]


def test_render_sales():
    out = render_sales(ROWS)
    assert out.splitlines()[0] == "SALES REPORT"
    assert "apple: 10" in out
    assert "banana: 20" in out
    assert out.splitlines()[-1] == "TOTAL: 30"
    assert out.count("=" * 30) == 2


def test_render_expenses():
    out = render_expenses(ROWS)
    assert out.splitlines()[0] == "EXPENSE REPORT"
    assert out.splitlines()[-1] == "TOTAL: 30"


def test_empty_rows():
    assert "TOTAL: 0" in render_sales([])
