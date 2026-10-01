# -*- coding: utf-8 -*-
"""销售/费用报表渲染。"""


def render_sales(rows):
    """rows: list[dict]（含 name/amount），渲染纯文本销售报表。"""
    lines = ["SALES REPORT", "=" * 30]
    total = 0
    for r in rows:
        total += r["amount"]
        lines.append(f"{r['name']}: {r['amount']}")
    lines.append("=" * 30)
    lines.append(f"TOTAL: {total}")
    return "\n".join(lines)


def render_expenses(rows):
    """渲染纯文本费用报表，格式与销售报表一致，仅标题不同。"""
    lines = ["EXPENSE REPORT", "=" * 30]
    total = 0
    for r in rows:
        total += r["amount"]
        lines.append(f"{r['name']}: {r['amount']}")
    lines.append("=" * 30)
    lines.append(f"TOTAL: {total}")
    return "\n".join(lines)
