# -*- coding: utf-8 -*-
"""eval loader：task.json 加载、校验、--task 过滤。"""
import json

import pytest

from mycodingagent import config
from mycodingagent.eval import loader
from mycodingagent.eval.loader import TaskDefError, load_task, load_tasks


def _make_task(tmp_path, name="demo", *, data=None, with_fixture=True):
    """在 tmp_path 下构造一个任务目录并返回其路径。"""
    task_dir = tmp_path / name
    task_dir.mkdir(exist_ok=True)
    payload = {
        "name": name,
        "type": "bugfix",
        "prompt": "修复它",
        "graders": [{"type": "file_exists", "path": "a.py"}],
    }
    if data:
        payload.update(data)
    (task_dir / "task.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    if with_fixture:
        (task_dir / "fixture").mkdir(exist_ok=True)
    return task_dir


def test_load_valid_task(tmp_path):
    task = load_task(_make_task(tmp_path))
    assert task.name == "demo"
    assert task.type == "bugfix"
    assert task.timeout_seconds == config.EVAL_TASK_TIMEOUT  # 缺省填配置默认值
    assert task.fixture_dir.is_dir()


def test_name_must_match_dir(tmp_path):
    with pytest.raises(TaskDefError, match="不一致"):
        load_task(_make_task(tmp_path, "demo", data={"name": "other"}))


def test_invalid_type(tmp_path):
    with pytest.raises(TaskDefError, match="type"):
        load_task(_make_task(tmp_path, data={"type": "unknown"}))


def test_missing_prompt(tmp_path):
    with pytest.raises(TaskDefError, match="prompt"):
        load_task(_make_task(tmp_path, data={"prompt": "  "}))


def test_empty_graders(tmp_path):
    with pytest.raises(TaskDefError, match="graders"):
        load_task(_make_task(tmp_path, data={"graders": []}))


def test_grader_missing_type(tmp_path):
    with pytest.raises(TaskDefError, match="type"):
        load_task(_make_task(tmp_path, data={"graders": [{"run": "x"}]}))


def test_missing_fixture_dir(tmp_path):
    with pytest.raises(TaskDefError, match="fixture"):
        load_task(_make_task(tmp_path, with_fixture=False))


def test_bad_json(tmp_path):
    task_dir = tmp_path / "bad"
    task_dir.mkdir()
    (task_dir / "task.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(TaskDefError, match="JSON"):
        load_task(task_dir)


def test_load_tasks_filter_and_unknown(tmp_path):
    _make_task(tmp_path, "a")
    _make_task(tmp_path, "b")
    tasks = load_tasks(tmp_path, only=["b"])
    assert [t.name for t in tasks] == ["b"]
    with pytest.raises(TaskDefError, match="找不到任务"):
        load_tasks(tmp_path, only=["nope"])


def test_timeout_override(tmp_path):
    task = load_task(_make_task(tmp_path, data={"timeout_seconds": 30}))
    assert task.timeout_seconds == 30
    with pytest.raises(TaskDefError, match="timeout"):
        load_task(_make_task(tmp_path, data={"timeout_seconds": -1}))


def test_real_evals_tasks_all_valid():
    """项目自带 evals/tasks 下全部任务定义必须合法（防手误写坏）。"""
    tasks = load_tasks(config.PROJECT_ROOT / "evals" / "tasks")
    names = [t.name for t in tasks]
    assert len(tasks) >= 5
    assert len(names) == len(set(names))
