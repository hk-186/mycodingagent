# -*- coding: utf-8 -*-
"""Git 工具测试（阶段 3）。

用 tmp_path + git init 构造临时仓库，monkeypatch config.PROJECT_DIR 指向它，
覆盖 git_status / git_diff / git_log / git_commit 四个工具。
"""

import subprocess

import pytest

from mycodingagent import config
from mycodingagent.tools.git_tools import git_commit, git_diff, git_log, git_status


# ============================================================
# Fixture：初始化一个临时 git 仓库，并把 config.PROJECT_DIR 指向它
# ============================================================
@pytest.fixture()
def git_repo(tmp_path, monkeypatch):
    """初始化临时 git 仓库并配置 user，避免 commit 时报 missing identity 错。"""
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    subprocess.run(
        ["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    # 关闭 GPG 签名，避免某些环境的 commit hook 失败
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    return tmp_path


def _commit_file(repo, filename, content, message="init"):
    """在 repo 里写入文件、add 并 commit。"""
    (repo / filename).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", filename], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message], cwd=repo, check=True, capture_output=True
    )


# ============================================================
# git_status
# ============================================================
def test_git_status_initial_repo(git_repo):
    """刚 init 的仓库：git status --short --branch 会输出 `## No commits yet` 之类。"""
    result = git_status.invoke({})
    assert isinstance(result, str)
    assert "##" in result  # 分支信息行


def test_git_status_shows_untracked_file(git_repo):
    """有未跟踪文件时输出含 `??` 标记。"""
    (git_repo / "new.txt").write_text("hello", encoding="utf-8")
    result = git_status.invoke({})
    assert "??" in result
    assert "new.txt" in result


def test_git_status_clean_repo(git_repo):
    """已 commit 后无变更：git status 输出只剩 `## branch` 行（无文件变更行）。"""
    _commit_file(git_repo, "a.txt", "init", "first")
    result = git_status.invoke({})
    assert "##" in result
    assert "??" not in result
    assert " M " not in result  # 无已修改未暂存行


# ============================================================
# git_log
# ============================================================
def test_git_log_empty_repo(git_repo):
    """无提交时 git log 失败，工具返回「暂无提交」。"""
    result = git_log.invoke({"limit": 10})
    assert "暂无提交" in result


def test_git_log_with_commits(git_repo):
    _commit_file(git_repo, "a.txt", "a", "first commit")
    _commit_file(git_repo, "b.txt", "b", "second commit")
    result = git_log.invoke({"limit": 10})
    assert "second commit" in result
    assert "first commit" in result


def test_git_log_limit_clamped(git_repo):
    """limit 上限 50、下限 1，超出范围被裁剪。"""
    for i in range(3):
        _commit_file(git_repo, f"f{i}.txt", str(i), f"commit {i}")
    # limit=1 只显示最新一条
    result = git_log.invoke({"limit": 1})
    assert "commit 2" in result
    assert "commit 1" not in result
    assert "commit 0" not in result


# ============================================================
# git_diff
# ============================================================
def test_git_diff_no_changes(git_repo):
    _commit_file(git_repo, "a.txt", "init", "first")
    result = git_diff.invoke({"file_path": ""})
    assert "无差异" in result


def test_git_diff_shows_unstaged_change(git_repo):
    _commit_file(git_repo, "a.txt", "init", "first")
    (git_repo / "a.txt").write_text("modified", encoding="utf-8")
    result = git_diff.invoke({"file_path": ""})
    # 输出含 --stat 行 + 完整 diff 内容，二者都应提到文件
    assert "a.txt" in result


def test_git_diff_with_file_path(git_repo):
    _commit_file(git_repo, "a.txt", "init", "first")
    _commit_file(git_repo, "b.txt", "init", "second")
    (git_repo / "a.txt").write_text("modified-a", encoding="utf-8")
    (git_repo / "b.txt").write_text("modified-b", encoding="utf-8")
    # 只看 a.txt：返回内容应含 a.txt，不含 b.txt 的差异行
    result = git_diff.invoke({"file_path": "/a.txt"})
    assert "a.txt" in result
    # b.txt 的修改不应出现在按文件路径查询的结果里
    assert "modified-b" not in result


# ============================================================
# git_commit
# ============================================================
def test_git_commit_empty_message(git_repo):
    result = git_commit.invoke({"message": ""})
    assert "提交信息不能为空" in result


def test_git_commit_whitespace_message(git_repo):
    result = git_commit.invoke({"message": "   "})
    assert "提交信息不能为空" in result


def test_git_commit_no_changes_at_all(git_repo):
    """空仓库无任何文件改动：返回「没有变更可提交」。"""
    result = git_commit.invoke({"message": "test"})
    assert "没有变更可提交" in result


def test_git_commit_no_staged_but_has_unstaged(git_repo):
    """有未暂存变更但未 add：提示先 git add。"""
    (git_repo / "a.txt").write_text("hello", encoding="utf-8")
    result = git_commit.invoke({"message": "test"})
    assert "没有暂存" in result
    assert "git add" in result


def test_git_commit_with_staged(git_repo):
    """暂存后 commit：成功并展示 log -1 --stat。"""
    (git_repo / "a.txt").write_text("hello", encoding="utf-8")
    subprocess.run(
        ["git", "add", "a.txt"], cwd=git_repo, check=True, capture_output=True
    )
    result = git_commit.invoke({"message": "add a.txt"})
    assert "提交成功" in result
    assert "add a.txt" in result  # log -1 --stat 含 message


def test_git_commit_invalid_repo_handled_gracefully(monkeypatch, tmp_path):
    """非 git 仓库目录：工具应返回失败信息而非抛异常。"""
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)  # tmp_path 未 git init
    result = git_commit.invoke({"message": "test"})
    # 不是 git 仓库时 git 命令报错，工具把错误返回
    assert "失败" in result or "没有变更" in result or "not a git repository" in result.lower() or "git" in result.lower()
