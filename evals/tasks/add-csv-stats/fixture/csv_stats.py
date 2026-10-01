# -*- coding: utf-8 -*-
"""CSV 统计工具。"""
import csv


def read_rows(path):
    """读取 CSV（首行为表头），返回 dict 列表。"""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
