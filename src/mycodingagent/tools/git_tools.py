# -*- coding: utf-8 -*-
"""
Git 工具（阶段 3 缺口 C4）
=========================
四个 `@tool`：`git_status` / `git_diff` / `git_log` 只读直接执行；
`git_commit` 提交前展示 status+diff 摘要后由 `InterruptOnConfig` 触发审批。

所有命令通过 `subprocess.run(["git", ...], cwd=PROJECT_DIR, capture_output=True)`
执行，避免走 SafeShellBackend 黑名单逻辑（git 写命令不应被拦截，应由本层审批）。
"""

from __future__ import annotations

import subprocess

from langchain_core.tools import tool

from mycodingagent import config


def _run_git(args: list[str]) -> tuple[int, str, str]:
    """在项目目录运行 git 命令，返回 (exit_code, stdout, stderr)。"""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(config.PROJECT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except FileNotFoundError:
        return 1, "", "git 未安装或不在 PATH 中"
    except subprocess.TimeoutExpired:
        return 124, "", "git 命令超时"
    except Exception as e:  # noqa: BLE001 —— 与其他工具一致：异常转成失败结果
        return 1, "", f"git 命令执行异常（{type(e).__name__}）：{e}"


@tool
def git_status() -> str:
    """查看当前 Git 仓库的工作区状态（git status --short，含暂存与未暂存变更）。"""
    code, out, err = _run_git(["status", "--short", "--branch"])
    if code != 0:
        return f"git status 失败（exit {code}）：{err or out}"
    return out.strip() or "工作区干净（无变更）"


@tool
def git_diff(file_path: str = "") -> str:
    """查看 Git 工作区与暂存区的差异。file_path 为空时查看全部，否则只看该文件。

    Args:
        file_path: 可选，相对项目根目录的文件路径（虚拟路径，/ 开头）
    """
    args = ["diff", "--stat"]
    # 暂存区差异也一并展示（git diff HEAD 更完整）
    args = ["diff", "HEAD", "--stat"]
    if file_path:
        # 虚拟路径 / 开头去掉前导 /
        rel = file_path.lstrip("/")
        args.extend(["--", rel])
    code, out, err = _run_git(args)
    if code != 0:
        return f"git diff 失败（exit {code}）：{err or out}"
    if not out.strip():
        return "无差异"
    # 同时给出实际 diff 内容（截断到 4000 字符，避免 context 爆炸）
    code2, out2, _ = _run_git(["diff", "HEAD"] + (["--", file_path.lstrip("/")] if file_path else []))
    full = out2.strip()
    if len(full) > 4000:
        full = full[:4000] + "\n... (diff 已截断，完整差异请用 git diff HEAD 查看)"
    return f"{out.strip()}\n\n{full}"


@tool
def git_log(limit: int = 10) -> str:
    """查看最近 N 条 Git 提交记录（默认 10 条）。"""
    n = max(1, min(int(limit), 50))
    code, out, err = _run_git(["log", f"-{n}", "--oneline", "--decorate"])
    if code != 0:
        # 空仓库（无任何 commit）给友好提示，而不是失败信息
        combined = (err + out).lower()
        if "does not have any commits" in combined or "no commits" in combined:
            return "暂无提交"
        return f"git log 失败（exit {code}）：{err or out}"
    return out.strip() or "暂无提交"


@tool
def git_commit(message: str) -> str:
    """提交当前已暂存的变更到 Git。

    提交前会先展示 `git status` + `git diff` 摘要给用户审批；
    审批通过后才真正执行 `git commit -m <message>`。
    没有暂存变更时直接返回失败提示，不会创建空提交。

    Args:
        message: 提交信息（必填）
    """
    if not message or not message.strip():
        return "提交信息不能为空"

    # 先看是否有暂存变更
    code, out, _ = _run_git(["diff", "--cached", "--stat"])
    if code == 0 and not out.strip():
        # 没有暂存变更，尝试 git status 看是否有未暂存变更
        _, status_out, _ = _run_git(["status", "--short"])
        if not status_out.strip():
            return "没有变更可提交"
        return "没有暂存变更。请先用 git add 暂存需要提交的文件，再调用 git_commit。\n当前未暂存变更：\n" + status_out

    code, out, err = _run_git(["commit", "-m", message])
    if code != 0:
        return f"git commit 失败（exit {code}）：{err or out}"
    # 提交成功后展示最新的 log 一条作为确认
    _, log_out, _ = _run_git(["log", "-1", "--stat"])
    return f"提交成功\n{out.strip()}\n\n{log_out.strip()}"
