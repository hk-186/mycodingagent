# -*- coding: utf-8 -*-
"""
审批策略（阶段 3）
==================
本模块把 `permissions.classify_command` 的命令分类（block/needs_approval/allow）
与 `config.APPROVAL_MODE` / `config.PLAN_MODE` 组合，输出「是否需要触发 HITL 审批」
的布尔结论，供 deepagents `InterruptOnConfig.when` 谓词调用。

`when` 谓词由 `langchain.agents.middleware.human_in_the_loop.HumanInTheLoopMiddleware`
在每次模型 AIMessage 产出后、工具执行前调用，签名固定为
`(ToolCallRequest) -> bool`：返回 True 表示触发 interrupt，False 表示放行。
"""

from __future__ import annotations

from typing import Any

from mycodingagent import config
from mycodingagent.permissions import classify_command


def _is_write_shell_command(command: str) -> bool:
    """粗略判断 shell 命令是否是写操作（会改变文件系统状态）。

    阶段 3 APPROVAL_MODE=all 时，所有写命令都需要审批。读命令（ls/dir/cat/echo
    不带重定向等）直接放行。
    """
    if not command:
        return False
    # 用空格切第一个 token + 旗标，看是否在已知写命令集合
    # 写操作动词：rm/del/cp/mv/mkdir/touch/echo > /pip install/npm install/git commit 等
    write_verbs = {
        # 文件增删改
        "rm", "del", "erase", "rmdir", "rd", "cp", "copy", "mv", "move", "rename",
        "mkdir", "md", "touch", "tee",
        # 权限/属主
        "chmod", "chown", "icacls", "takeown", "attrib",
        # 包管理器（写）
        "pip", "pip3", "npm", "yarn", "pnpm", "winget", "apt", "brew",
        # Git 写操作（status/diff/log 不在此）
        # git 本身会按子命令判断，单独处理
    }
    # 提取首个 token（处理 sudo 前缀）
    tokens = command.replace("(", " ").split()
    if not tokens:
        return False
    # 跳过 sudo / cmd.exe 等前缀
    idx = 0
    while idx < len(tokens) and tokens[idx].lower() in {"sudo", "cmd", "cmd.exe", "powershell", "pwsh"}:
        idx += 1
    if idx >= len(tokens):
        return False
    first = tokens[idx].lower()
    # 处理 .exe 后缀
    if first.endswith(".exe"):
        first = first[:-4]
    if first == "git":
        # git 子命令判定
        git_args = tokens[idx + 1 :] if idx + 1 < len(tokens) else []
        if not git_args:
            return False
        sub = git_args[0].lower()
        # 只读子命令
        return sub not in {"status", "diff", "log", "show", "branch", "ls-files",
                           "ls-remote", "remote", "config", "-l", "blame", "shortlog",
                           "describe", "name-rev", "rev-parse", "stash", "tag"}
    if first in write_verbs:
        return True
    # 重定向写文件
    if ">" in command or ">>" in command:
        return True
    # PowerShell Set-Content / Out-File / Add-Content
    if first in {"set-content", "out-file", "add-content", "new-item", "set-itemproperty"}:
        return True
    return False


def should_interrupt_delete(tool_call_request: Any) -> bool:
    """`InterruptOnConfig.when` 谓词：delete 工具删除文件前触发审批。

    delete 是 deepagents 内置文件工具，不走 execute/shell 分类；它同样会
    永久删除文件，默认（minimal/all）与 Plan 模式都需要用户审批。仅在
    APPROVAL_MODE=disabled 且非 Plan 模式时放行。
    """
    tool_call = getattr(tool_call_request, "tool_call", None)
    if tool_call is None and isinstance(tool_call_request, dict):
        tool_call = tool_call_request.get("tool_call")
    if not tool_call or tool_call.get("name") != "delete":
        return False
    file_path = (tool_call.get("args") or {}).get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return False
    if config.PLAN_MODE:
        return True
    return config.APPROVAL_MODE != "disabled"


def should_interrupt_command(tool_call_request: Any) -> bool:
    """`InterruptOnConfig.when` 谓词：是否对这次 execute 工具调用触发审批。

    策略组合：

    1. `APPROVAL_MODE == "disabled"` —— 永不审批（除 plan_mode 全拦截）。
    2. `PLAN_MODE == True` —— Plan 模式下所有写命令一律审批，读命令放行。
    3. `APPROVAL_MODE == "all"` —— 所有写命令都审批（block 也走审批链路，
       CLI 端 reject 给模型「用户拒绝」语义）。
    4. `APPROVAL_MODE == "minimal"`（默认）—— 仅危险/灰区命令审批，
       黑名单也触发，由 CLI 端 reject 给出明确原因。
    """
    # tool_call_request 是 langchain ToolCallRequest TypedDict：
    #   {"tool_call": {"name": str, "args": dict, "id": str, "type": "tool_call"}, ...}
    tool_call = getattr(tool_call_request, "tool_call", None)
    if tool_call is None and isinstance(tool_call_request, dict):
        # dict 形式：用 .get 避免缺 key 抛 KeyError
        tool_call = tool_call_request.get("tool_call")
    if not tool_call:
        return False
    command = (tool_call.get("args") or {}).get("command", "")
    if not command or not isinstance(command, str):
        return False

    # Plan 模式：所有写命令都审批
    if config.PLAN_MODE and _is_write_shell_command(command):
        return True

    decision = classify_command(command)
    if config.APPROVAL_MODE == "disabled":
        # disabled 模式：除 plan_mode 全拦截外不审批；但黑名单仍由 SafeShellBackend 兜底拒绝
        return False
    if config.APPROVAL_MODE == "all":
        # all 模式：所有写命令都审批（block / needs_approval / write 命令统一审批）
        return decision.action != "allow" or _is_write_shell_command(command)
    # minimal 模式：危险 + 灰区审批；安全命令放行
    return decision.action in {"block", "needs_approval"}
