# -*- coding: utf-8 -*-
"""
eval 回归集运行器（阶段 6）
============================
在隔离沙箱中运行真实 agent 完成任务，用确定性 grader + LLM 评审判定，
结果落盘 evals/results/<timestamp>.json，可与历史 baseline 对比回归。

用法（项目根目录，已 pip install -e . 且 .env 配好 AGICTO_API_KEY）：
    python scripts/run_evals.py                      # 跑全部任务
    python scripts/run_evals.py --task fix-clamp-boundary   # 只跑指定任务
    python scripts/run_evals.py --dry-run            # 只列出任务，不调 LLM
    python scripts/run_evals.py --baseline evals/results/eval-20261001-120000.json

exit code：全部通过为 0；有失败任务或相对 baseline 出现 pass→fail 回归为 1。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mycodingagent import config
from mycodingagent.eval import loader, report
from mycodingagent.eval.runner import run_one

# evals/ 目录锚定项目根目录（与 config.PROJECT_ROOT 一致）
TASKS_ROOT = config.PROJECT_ROOT / "evals" / "tasks"
RESULTS_DIR = config.PROJECT_ROOT / "evals" / "results"


def _make_progress(task_name: str):
    """终端实时进度：一行一个工具调用/审批事件。"""
    def show(ev) -> None:  # noqa: ANN001
        if ev.type == "tool_start":
            print(f"    [{task_name}] 工具调用: {ev.name}({ev.args_preview})", flush=True)
        elif ev.type == "tool_end":
            tag = "失败" if ev.is_error else "完成"
            print(f"    [{task_name}] 工具{tag}: {ev.name}", flush=True)
        elif ev.type == "interrupt":
            names = [ar.get("name", "") for ar in ev.action_requests]
            print(f"    [{task_name}] 审批自动通过: {', '.join(names)}", flush=True)
    return show


def main() -> int:
    parser = argparse.ArgumentParser(description="eval 回归集运行器（阶段 6）")
    parser.add_argument(
        "--task", action="append", default=None,
        help="只跑指定任务（可多次传入）；缺省跑全部",
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出任务，不执行")
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="与历史结果 JSON 对比回归",
    )
    parser.add_argument(
        "--out", type=Path, default=RESULTS_DIR,
        help=f"结果输出目录（默认 {RESULTS_DIR}）",
    )
    args = parser.parse_args()

    try:
        tasks = loader.load_tasks(TASKS_ROOT, only=args.task)
    except loader.TaskDefError as e:
        print(f"任务定义错误：{e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(loader.describe_tasks(tasks))
        return 0

    if not config.API_KEY:
        print("未检测到 AGICTO_API_KEY！请先设置环境变量（模板见 .env.example）。")
        return 1

    print(f"开始 eval：{len(tasks)} 个任务（模型 {config.MODEL_NAME}）")
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {task.name} [{task.type}] 超时 {task.timeout_seconds}s")
        result = run_one(task, progress=_make_progress(task.name))
        results.append(result)
        mark = "PASS" if result.passed else "FAIL"
        print(
            f"  => {mark}（{result.steps} 步 / {result.tokens} tokens / "
            f"{result.elapsed_seconds:.0f}s / {result.stopped_reason}）",
            flush=True,
        )

    path = report.save(results, args.out)
    print(report.format_summary(results))
    print(f"结果已保存：{path}")

    failed = any(not r.passed for r in results)
    regression = False
    if args.baseline:
        print(report.compare(args.baseline, results))
        regression = report.has_regression(args.baseline, results)
    return 1 if (failed or regression) else 0


if __name__ == "__main__":
    sys.exit(main())
