# -*- coding: utf-8 -*-
"""
阶段 6：eval 回归集
===================
loader   加载 evals/tasks/ 下的任务定义（task.json + fixture/）
runner   在临时沙箱中运行真实 agent，自动处理审批中断
graders  确定性验收（command / 文件检查）+ LLM 评审
report   结果 JSON 落盘与 baseline 回归对比

CLI 入口：scripts/run_evals.py
"""
