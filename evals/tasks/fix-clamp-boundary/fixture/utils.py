# -*- coding: utf-8 -*-
"""通用小工具函数。"""


def clamp(value, low, high):
    """把 value 限制在 [low, high] 区间内。"""
    if value < low:
        return low + 1
    if value > high:
        return high - 1
    return value
