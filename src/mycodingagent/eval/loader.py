# -*- coding: utf-8 -*-
"""
eval 任务定义加载与校验
========================
任务目录布局（evals/tasks/<name>/）：
    task.json   任务定义（name/type/prompt/graders 等）
    fixture/    初始项目文件，运行前原样复制进沙箱

task.json 字段：
    name             必填，必须与目录名一致（防复制错位）
    type             必填，bugfix / feature / refactor（仅用于报告分组统计）
    prompt           必填，原样作为 iter_task_events(user_input=...) 的输入
    timeout_seconds  可选，单任务 wall-clock 超时，缺省 config.EVAL_TASK_TIMEOUT
    ask_user_reply   可选，agent 调 ask_user 触发审批时的自动代答文本
    graders          必填，非空数组，元素见 graders.py 的类型约定
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mycodingagent import config

VALID_TASK_TYPES = {"bugfix", "feature", "refactor"}

#: agent 调 ask_user 且任务未配置 ask_user_reply 时的统一兜底代答
DEFAULT_ASK_USER_REPLY = "按你认为最合理的方式处理，不要追问。"


class TaskDefError(ValueError):
    """任务定义文件不合法。"""


@dataclass(frozen=True)
class EvalTask:
    """一个 eval 任务的完整定义。"""

    name: str
    type: str
    prompt: str
    task_dir: Path
    fixture_dir: Path
    graders: list[dict[str, Any]]
    timeout_seconds: int = 0  # 0 表示未显式配置，加载时填默认值
    ask_user_reply: str = DEFAULT_ASK_USER_REPLY


def _validate_task(data: dict[str, Any], task_dir: Path) -> None:
    """校验 task.json 的必填字段与取值合法性，不合法抛 TaskDefError。"""
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise TaskDefError(f"{task_dir}: name 缺失或不是非空字符串")
    if name != task_dir.name:
        raise TaskDefError(
            f"{task_dir}: name({name!r}) 与目录名({task_dir.name!r}) 不一致"
        )
    if data.get("type") not in VALID_TASK_TYPES:
        raise TaskDefError(
            f"{task_dir}: type 必须是 {sorted(VALID_TASK_TYPES)} 之一，"
            f"实际为 {data.get('type')!r}"
        )
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise TaskDefError(f"{task_dir}: prompt 缺失或不是非空字符串")
    graders = data.get("graders")
    if not isinstance(graders, list) or not graders:
        raise TaskDefError(f"{task_dir}: graders 必须是非空数组")
    for i, g in enumerate(graders):
        if not isinstance(g, dict) or not g.get("type"):
            raise TaskDefError(f"{task_dir}: graders[{i}] 缺少 type 字段")
    timeout = data.get("timeout_seconds", config.EVAL_TASK_TIMEOUT)
    if not isinstance(timeout, int) or timeout <= 0:
        raise TaskDefError(f"{task_dir}: timeout_seconds 必须是正整数")
    if not (task_dir / "fixture").is_dir():
        raise TaskDefError(f"{task_dir}: 缺少 fixture/ 子目录")


def load_task(task_dir: Path) -> EvalTask:
    """加载并校验单个任务目录。"""
    task_json = task_dir / "task.json"
    if not task_json.is_file():
        raise TaskDefError(f"{task_dir}: 缺少 task.json")
    try:
        data = json.loads(task_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise TaskDefError(f"{task_json}: JSON 解析失败：{e}") from e
    if not isinstance(data, dict):
        raise TaskDefError(f"{task_json}: 顶层必须是 JSON 对象")
    _validate_task(data, task_dir)
    return EvalTask(
        name=data["name"],
        type=data["type"],
        prompt=data["prompt"],
        task_dir=task_dir,
        fixture_dir=task_dir / "fixture",
        graders=list(data["graders"]),
        timeout_seconds=int(data.get("timeout_seconds", config.EVAL_TASK_TIMEOUT)),
        ask_user_reply=str(data.get("ask_user_reply") or DEFAULT_ASK_USER_REPLY),
    )


def load_tasks(
    tasks_root: Path,
    only: list[str] | None = None,
) -> list[EvalTask]:
    """加载 tasks_root 下全部任务（按目录名排序），可用 only 按名字过滤。

    only 中的名字若不存在抛 TaskDefError（防止打错名字静默跑空）。
    """
    if not tasks_root.is_dir():
        raise TaskDefError(f"任务根目录不存在：{tasks_root}")
    task_dirs = sorted(p for p in tasks_root.iterdir() if p.is_dir())
    if only:
        known = {p.name for p in task_dirs}
        unknown = [n for n in only if n not in known]
        if unknown:
            raise TaskDefError(
                f"找不到任务：{', '.join(unknown)}（可用：{', '.join(sorted(known))}）"
            )
        wanted = set(only)
        task_dirs = [p for p in task_dirs if p.name in wanted]
    return [load_task(p) for p in task_dirs]


def describe_tasks(tasks: list[EvalTask]) -> str:
    """--dry-run 用的任务清单文本。"""
    lines = [f"共 {len(tasks)} 个任务："]
    for t in tasks:
        grader_types = [g["type"] for g in t.graders]
        lines.append(
            f"  - {t.name} [{t.type}] 超时 {t.timeout_seconds}s，"
            f"graders: {', '.join(grader_types)}"
        )
    return "\n".join(lines)
