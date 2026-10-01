# -*- coding: utf-8 -*-
# eval 验收脚本（不属于项目代码，agent 无需理会）：
# 重构后手写求和循环只允许出现一次（应抽入公共函数）。
import pathlib
import sys

text = pathlib.Path("report.py").read_text(encoding="utf-8")
count = text.count("total = 0")
print(f"'total = 0' 出现 {count} 次（要求 <= 1）")
sys.exit(0 if count <= 1 else 1)
