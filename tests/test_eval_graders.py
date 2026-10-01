# -*- coding: utf-8 -*-
"""eval graders：4 种确定性 grader + llm_review（mock 评审模型，不触网）。"""
import sys
from pathlib import Path

import pytest

from mycodingagent.eval import graders
from mycodingagent.eval.graders import (
    GradeContext,
    GraderError,
    GraderResult,
    run_graders,
)


def _ctx(tmp_path, **kw):
    return GradeContext(sandbox=tmp_path, **kw)


# ============================================================
# command grader（真实子进程，用 sys.executable 保证跨平台）
# ============================================================
def test_command_pass(tmp_path):
    r = run_graders(
        [{"type": "command", "run": f'"{sys.executable}" -c "pass"'}], _ctx(tmp_path)
    )
    assert r[0].passed


def test_command_exit_code_mismatch(tmp_path):
    r = run_graders(
        [{"type": "command", "run": f'"{sys.executable}" -c "exit(1)"'}],
        _ctx(tmp_path),
    )
    assert not r[0].passed
    assert "exit=1" in r[0].detail


def test_command_timeout(tmp_path):
    r = run_graders(
        [{
            "type": "command",
            "run": f'"{sys.executable}" -c "import time; time.sleep(10)"',
            "timeout_seconds": 1,
        }],
        _ctx(tmp_path),
    )
    assert not r[0].passed
    assert "超时" in r[0].detail


def test_command_runs_in_sandbox(tmp_path):
    (tmp_path / "marker.txt").write_text("hi", encoding="utf-8")
    code = "import pathlib; exit(0 if pathlib.Path('marker.txt').exists() else 1)"
    r = run_graders(
        [{"type": "command", "run": f'"{sys.executable}" -c "{code}"'}], _ctx(tmp_path)
    )
    assert r[0].passed


# ============================================================
# 文件检查 grader
# ============================================================
def test_file_exists(tmp_path):
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    assert run_graders([{"type": "file_exists", "path": "a.py"}], _ctx(tmp_path))[0].passed
    assert not run_graders(
        [{"type": "file_exists", "path": "b.py"}], _ctx(tmp_path)
    )[0].passed


def test_file_contains_plain_and_regex(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    raise ValueError\n", encoding="utf-8")
    assert run_graders(
        [{"type": "file_contains", "path": "a.py", "pattern": "raise ValueError"}],
        _ctx(tmp_path),
    )[0].passed
    assert run_graders(
        [{"type": "file_contains", "path": "a.py", "pattern": r"raise \w+Error", "regex": True}],
        _ctx(tmp_path),
    )[0].passed
    r = run_graders(
        [{"type": "file_contains", "path": "a.py", "pattern": "nope"}], _ctx(tmp_path)
    )
    assert not r[0].passed


def test_file_contains_missing_file(tmp_path):
    r = run_graders(
        [{"type": "file_contains", "path": "gone.py", "pattern": "x"}], _ctx(tmp_path)
    )
    assert not r[0].passed
    assert "不存在" in r[0].detail


def test_file_not_contains(tmp_path):
    (tmp_path / "a.py").write_text("bad_pattern here", encoding="utf-8")
    assert not run_graders(
        [{"type": "file_not_contains", "path": "a.py", "pattern": "bad_pattern"}],
        _ctx(tmp_path),
    )[0].passed
    assert run_graders(
        [{"type": "file_not_contains", "path": "a.py", "pattern": "clean"}],
        _ctx(tmp_path),
    )[0].passed
    # 文件不存在视为通过（无可禁内容）
    assert run_graders(
        [{"type": "file_not_contains", "path": "gone.py", "pattern": "x"}],
        _ctx(tmp_path),
    )[0].passed


def test_path_escape_rejected(tmp_path):
    with pytest.raises(GraderError, match="逃逸"):
        run_graders(
            [{"type": "file_contains", "path": "../outside.txt", "pattern": "x"}],
            _ctx(tmp_path),
        )


def test_unknown_grader_type(tmp_path):
    with pytest.raises(GraderError, match="未知 grader"):
        run_graders([{"type": "magic"}], _ctx(tmp_path))


# ============================================================
# 短路：确定性 grader 失败后不执行 llm_review（省钱）
# ============================================================
def test_short_circuit_skips_llm_review(tmp_path, monkeypatch):
    called = []

    def fake_review(spec, ctx):
        called.append(True)
        return GraderResult(type="llm_review", passed=True, score=5)

    monkeypatch.setitem(graders._GRADERS, "llm_review", fake_review)
    specs = [
        {"type": "command", "run": f'"{sys.executable}" -c "exit(1)"'},
        {"type": "llm_review", "criteria": "x"},
    ]
    results = run_graders(specs, _ctx(tmp_path))
    assert len(results) == 1
    assert not called


# ============================================================
# llm_review：mock ChatOpenAI
# ============================================================
class _FakeResp:
    def __init__(self, content):
        self.content = content


def _mock_llm(monkeypatch, reply):
    import langchain_openai

    class FakeChat:
        def __init__(self, **kw):
            pass

        def invoke(self, prompt):
            return _FakeResp(reply)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChat)


def test_llm_review_pass(tmp_path, monkeypatch):
    _mock_llm(monkeypatch, '{"score": 4, "comment": "改动干净"}')
    r = run_graders(
        [{"type": "llm_review", "criteria": "c", "pass_score": 3}],
        _ctx(tmp_path, diff="diff", final_report="汇报", task_prompt="任务"),
    )
    assert r[0].passed and r[0].score == 4
    assert "改动干净" in r[0].detail


def test_llm_review_below_pass_score(tmp_path, monkeypatch):
    _mock_llm(monkeypatch, '{"score": 2, "comment": "不达标"}')
    r = run_graders(
        [{"type": "llm_review", "criteria": "c", "pass_score": 3}], _ctx(tmp_path)
    )
    assert not r[0].passed and r[0].score == 2


def test_llm_review_json_fence(tmp_path, monkeypatch):
    _mock_llm(monkeypatch, '好的，结果如下：\n```json\n{"score": 5, "comment": "完美"}\n```')
    r = run_graders([{"type": "llm_review", "criteria": "c"}], _ctx(tmp_path))
    assert r[0].passed and r[0].score == 5


def test_llm_review_non_json_fails(tmp_path, monkeypatch):
    _mock_llm(monkeypatch, "我觉得做得不错")
    r = run_graders([{"type": "llm_review", "criteria": "c"}], _ctx(tmp_path))
    assert not r[0].passed
    assert "非 JSON" in r[0].detail


def test_llm_review_api_error_fails_gracefully(tmp_path, monkeypatch):
    import langchain_openai

    class BoomChat:
        def __init__(self, **kw):
            pass

        def invoke(self, prompt):
            raise ConnectionError("network down")

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", BoomChat)
    r = run_graders([{"type": "llm_review", "criteria": "c"}], _ctx(tmp_path))
    assert not r[0].passed and r[0].score == 0
    assert "调用失败" in r[0].detail
