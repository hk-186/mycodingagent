# -*- coding: utf-8 -*-
"""eval report：结果落盘、汇总渲染、baseline 回归对比。"""
import json

from mycodingagent.eval import report
from mycodingagent.eval.graders import GraderResult
from mycodingagent.eval.runner import TaskResult


def _result(name, passed, tokens=100, steps=5, score=0):
    graders = [GraderResult(type="command", passed=passed, detail="ok")]
    if score:
        graders.append(GraderResult(type="llm_review", passed=score >= 3, score=score, detail="评"))
    return TaskResult(
        name=name, type="bugfix", passed=passed,
        grader_results=graders, steps=steps, tokens=tokens,
        elapsed_seconds=12.0, stopped_reason="completed",
        final_report="汇报", diff="diff",
    )


def test_save_writes_json_with_metadata(tmp_path):
    path = report.save([_result("t1", True)], tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["passed"] == 1
    assert data["summary"]["total_tokens"] == 100
    assert data["tasks"][0]["name"] == "t1"
    assert "model" in data["metadata"]
    assert "user_memory_file" in data["metadata"]


def test_format_summary_marks_fail_and_review(tmp_path):
    text = report.format_summary([
        _result("good-task", True, score=4),
        _result("bad-task", False),
    ])
    assert "PASS" in text and "FAIL" in text
    assert "4/5" in text
    assert "1/2 通过" in text


def _save_baseline(tmp_path, tasks):
    path = tmp_path / "baseline.json"
    payload = {
        "metadata": {},
        "summary": {},
        "tasks": [
            {"name": n, "passed": p, "tokens": t, "steps": s}
            for n, p, t, s in tasks
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_compare_detects_regression(tmp_path):
    baseline = _save_baseline(tmp_path, [("t1", True, 100, 5), ("t2", True, 80, 4)])
    results = [_result("t1", True, tokens=120), _result("t2", False)]
    text = report.compare(baseline, results)
    assert "回归（pass→fail）：t2" in text
    assert report.has_regression(baseline, results)


def test_compare_detects_fix_and_new_task(tmp_path):
    baseline = _save_baseline(tmp_path, [("t1", False, 100, 5)])
    results = [_result("t1", True), _result("t2", True)]
    text = report.compare(baseline, results)
    assert "修复（fail→pass）：t1" in text
    assert "新任务" in text
    assert not report.has_regression(baseline, results)


def test_compare_missing_baseline_file(tmp_path):
    text = report.compare(tmp_path / "gone.json", [_result("t1", True)])
    assert "读取失败" in text
    assert not report.has_regression(tmp_path / "gone.json", [_result("t1", False)])


def test_compare_token_delta(tmp_path):
    baseline = _save_baseline(tmp_path, [("t1", True, 100, 5)])
    results = [_result("t1", True, tokens=150, steps=7)]
    text = report.compare(baseline, results)
    assert "tokens +50" in text
    assert "steps +2" in text
