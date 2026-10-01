# -*- coding: utf-8 -*-
"""文本处理工具。"""
import re


def slugify(text):
    """把任意文本转成 URL slug：小写字母数字 + 单横线分隔。"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", "-", text)
    return text
