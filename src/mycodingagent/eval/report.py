# -*- coding: utf-8 -*-
"""
eval 结果落盘与回归对比
========================
- save()：把一轮结果写成 evals/results/<timestamp>.json，含环境 metadata
  （模型名、用户级记忆文件是否存在），供跨时间/跨机器对比溯源。
- compare()：与 baseline 对比，聚焦「稳定 pass→fail」回归与资源消耗变化。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from mycodingagent import config
from mycodingagent.eval.runner import TaskResult


def _run_metadata() -> dict[str, Any]:
    """本轮运行的环境信息（对比时溯源环境差异）。"""
    user_memory = Path.home() / ".mycodingagent" / "MEMORY.md"
    return {
        "model": config.MODEL_NAME,
        "temperature": config.LLM_TEMPERATURE,
        "review_model": config.EVAL_REVIEW_MODEL or config.MODEL_NAME,
        "user_memory_file": user_memory.is_file(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _task_to_dict(r: TaskResult) -> dict[str, Any]:
    """asdict 递归转换（GraderResult 同为 dataclass）；diff/final_report 保留全文供排查。"""
    return asdict(r)


def save(results: list[TaskResult], out_dir: Path) -> Path:
    """落盘结果 JSON，返回文件路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    payload = {
        "metadata": _run_metadata(),
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "total_tokens": sum(r.tokens for r in results),
            "total_seconds": round(sum(r.elapsed_seconds for r in results), 1),
        },
        "tasks": [_task_to_dict(r) for r in results],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def format_summary(results: list[TaskResult]) -> str:
    """终端汇总表。"""
    lines = ["", "=" * 72, f"{'任务':<28}{'结果':<8}{'步数':>6}{'tokens':>10}{'耗时':>8}  评审"]
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        review = next((g for g in r.grader_results if g.type == "llm_review"), None)
        review_str = f"{review.score}/5" if review and review.score else "-"
        stop_tag = "" if r.stopped_reason == "completed" else f"({r.stopped_reason})"
        lines.append(
            f"{r.name:<28}{(mark + stop_tag):<8}{r.steps:>6}{r.tokens:>10}"
            f"{r.elapsed_seconds:>7.0f}s  {review_str}"
        )
        if not r.passed:
            failed = [g for g in r.grader_results if not g.passed]
            for g in failed:
                first_line = g.detail.splitlines()[0] if g.detail else ""
                lines.append(f"    └ {g.type}: {first_line}")
            if r.error:
                lines.append(f"    └ 运行错误: {r.error.splitlines()[0]}")
    n_pass = sum(1 for r in results if r.passed)
    lines.append(
        f"合计：{n_pass}/{len(results)} 通过，"
        f"{sum(r.tokens for r in results)} tokens，"
        f"{sum(r.elapsed_seconds for r in results):.0f}s"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


def compare(baseline_path: Path, results: list[TaskResult]) -> str:
    """与 baseline 结果对比，返回可读的回归报告文本。"""
    try:
        base = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return f"[baseline 读取失败：{e}]"
    base_tasks = {t["name"]: t for t in base.get("tasks", [])}

    lines = [f"与 baseline 对比（{baseline_path.name}）："]
    regressions: list[str] = []
    fixed: list[str] = []
    for r in results:
        b = base_tasks.get(r.name)
        if b is None:
            lines.append(f"  + {r.name}: 新任务，{'PASS' if r.passed else 'FAIL'}")
            continue
        if b["passed"] and not r.passed:
            regressions.append(r.name)
        elif not b["passed"] and r.passed:
            fixed.append(r.name)
        d_tokens = r.tokens - b.get("tokens", 0)
        d_steps = r.steps - b.get("steps", 0)
        lines.append(
            f"  {'!' if (b['passed'] and not r.passed) else ' '} {r.name}: "
            f"{'PASS' if r.passed else 'FAIL'}（baseline {'PASS' if b['passed'] else 'FAIL'}）"
            f"  tokens {d_tokens:+d} / steps {d_steps:+d}"
        )
    missing = [n for n in base_tasks if n not in {r.name for r in results}]
    for n in missing:
        lines.append(f"  - {n}: baseline 中存在，本轮未运行")

    lines.append("")
    if regressions:
        lines.append(f"!! 回归（pass→fail）：{', '.join(regressions)}")
    else:
        lines.append("无 pass→fail 回归。")
    if fixed:
        lines.append(f"修复（fail→pass）：{', '.join(fixed)}")
    return "\n".join(lines)


def has_regression(baseline_path: Path, results: list[TaskResult]) -> bool:
    """供 exit code 判断：存在 pass→fail 回归返回 True。"""
    try:
        base = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    base_pass = {t["name"]: t["passed"] for t in base.get("tasks", [])}
    return any(
        base_pass.get(r.name) is True and not r.passed for r in results
    )
