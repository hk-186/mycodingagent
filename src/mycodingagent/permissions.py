# -*- coding: utf-8 -*-
"""
危险命令拦截 + 命令分级（阶段 1 起，阶段 3 扩展）
=================================================
shell 执行前的命令分类层：

- `check_command(command) -> str | None`（阶段 1）：返回拒绝原因表示黑名单命中；
  `SafeShellBackend` 兜底用，确保即使审批被绕过也不会执行递归强删等操作。
- `classify_command(command) -> ApprovalDecision`（阶段 3）：分三档
    * "block"          —— 命中黑名单，直接拒绝（CLI 端 reject 给模型「用户拒绝」语义）
    * "needs_approval" —— 灰区命令（单文件删除、git reset --mixed 等），需用户审批
    * "allow"          —— 安全命令，直接放行

设计取向（对应开发计划"不可妥协的设计裁决"第 2 条）：
- 宁可误拦，不可漏拦。覆盖 Windows（cmd / PowerShell）与 bash 两类破坏性命令。
- 黑名单与灰区共用 `_DANGEROUS_PATTERNS`，灰区由 `_GREY_AREA_PATTERNS` 补充。
"""

import re
from dataclasses import dataclass


# 每条规则：(正则, 拒绝原因)。正则统一忽略大小写。
_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    # ---- 递归强删（bash）：组合旗标（-rf/-fr/-rfv）、分离旗标（-r … -f）、长旗标 ----
    (r"\brm\s+(-\w+\s+)*-\w*r\w*f", "递归强制删除（rm -rf 类）"),
    (r"\brm\s+(-\w+\s+)*-\w*f\w*r", "递归强制删除（rm -fr 类）"),
    (r"\brm\s+(-\w+\s+)*-\w*r\w*\s+(-\w+\s+)*-\w*f", "递归强制删除（rm -r … -f）"),
    (r"\brm\s+(-\w+\s+)*-\w*f\w*\s+(-\w+\s+)*-\w*r", "递归强制删除（rm -f … -r）"),
    (r"\brm\s+[^|;&]*--recursive[^|;&]*--force|\brm\s+[^|;&]*--force[^|;&]*--recursive",
     "递归强制删除（rm --recursive --force）"),
    (r"remove-item\s+[^|;&]*-recurse[^|;&]*-force|remove-item\s+[^|;&]*-force[^|;&]*-recurse",
     "PowerShell 递归强制删除（Remove-Item -Recurse -Force）"),
    (r"(?:^|[;&|]\s*)(?:del|rd|rmdir)\s+(?:/\w+\s+)*/\w*s\w*", "cmd 递归删除（del/rd/rmdir /s）"),
    # ---- 磁盘与文件系统破坏 ----
    (r"\bformat\s+\w:", "格式化磁盘"),
    (r"\bmkfs(?:\.\w+)?\b", "格式化文件系统（mkfs）"),
    (r"\bdd\s+[^|;&]*\bof=/dev/", "dd 直写块设备"),
    (r"\bdiskpart\b", "diskpart 磁盘分区操作"),
    # ---- 关机 / 重启 ----
    (r"\b(?:shutdown|poweroff|halt|reboot|restart-computer|stop-computer)\b", "关机/重启系统"),
    # ---- 远程脚本直接执行（命令注入的经典入口）----
    (r"\b(?:curl|wget|irm|iwr|invoke-webrequest)\b[^|;&]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|pwsh|powershell|iex|invoke-expression)\b",
     "管道执行远程脚本"),
    (r"\biex\s*\(\s*(?:iwr|irm|invoke-webrequest|invoke-restmethod)\b", "Invoke-Expression 执行远程内容"),
    # ---- 系统配置 ----
    (r"\breg\s+(?:delete|add)\s+hklm", "修改系统级注册表"),
    # ---- fork 炸弹 ----
    (r":\s*\(\s*\)\s*\{\s*:\s*\|", "fork 炸弹"),
    # ---- Git 破坏性操作（裁决 6：不做未告知的破坏性 Git 操作）----
    (r"\bgit\s+reset\s+--hard", "git reset --hard（丢弃全部未提交改动）"),
    (r"\bgit\s+clean\s+-\w*f", "git clean -f（强制删除未跟踪文件）"),
    (r"\bgit\s+push\s+(?:-\w+\s+)*-(?:f\b|f\w|-force)", "git push --force（强推覆盖远端）"),
]

# 灰区命令：未到黑名单但值得审批（用户拒绝后才不执行，approve 则放行）。
_GREY_AREA_PATTERNS: list[tuple[str, str]] = [
    # 单文件删除（rm 不带 -r/-f 旗标，仍可能误删用户文件）
    (r"\brm\s+(?!.*-\w*r\w*)(?!.*-\w*f\w*)", "删除文件（rm 单文件）"),
    # cmd 单文件删除
    (r"(?:^|[;&|]\s*)(?:del|erase)\s+(?!.*\s/\w*s\w*)", "删除文件（cmd del 单文件）"),
    (r"\bremove-item\b(?!.*-recurse)(?!.*-force)", "PowerShell 单文件删除"),
    # Git 混合重置：丢弃已暂存但未提交的改动
    (r"\bgit\s+reset\b(?!.*--hard)(?!.*--soft)", "git reset --mixed（丢弃暂存改动）"),
    # 修改文件权限/属主
    (r"\b(?:chmod|chown|icacls|takeown)\b", "修改文件权限或属主"),
    # 包管理器安装/卸载（影响系统状态）
    (r"\b(?:pip|pip3|npm|yarn|pnpm)\s+(?:uninstall|remove)\b", "卸载依赖包"),
    (r"\bwinget\s+(?:uninstall|remove)\b", "winget 卸载软件"),
    # 杀进程
    (r"\b(?:kill|killall|taskkill|stop-process)\b", "终止进程"),
]

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in _DANGEROUS_PATTERNS
]
_GREY_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in _GREY_AREA_PATTERNS
]


@dataclass(frozen=True)
class ApprovalDecision:
    """命令分类结果（阶段 3）。"""

    action: str   # "block" | "needs_approval" | "allow"
    reason: str    # 拒绝原因 / 审批提示 / 空


def check_command(command: str) -> str | None:
    """检查 shell 命令是否命中危险黑名单。

    返回拒绝原因（str）表示拒绝执行；返回 None 表示放行。
    注意：这是粗粒度的文本匹配，存在误拦可能——这是有意的取舍，
    误拦的命令可以让用户手动执行。
    """
    if not command:
        return None
    for pattern, reason in _RULES:
        if pattern.search(command):
            return reason
    return None


def classify_command(command: str) -> ApprovalDecision:
    """把 shell 命令分到三档：block / needs_approval / allow。

    - block：黑名单命中，CLI 端应当 reject 给模型「用户拒绝」语义
    - needs_approval：灰区，弹审批由用户决定
    - allow：常规安全命令直接放行

    注意：调用方在 APPROVAL_MODE=all 时所有非 allow 命令都应走 needs_approval，
    本函数只负责按命令文本分类，模式判断在 approvals.should_interrupt_command。
    """
    if not command:
        return ApprovalDecision(action="allow", reason="")
    for pattern, reason in _RULES:
        if pattern.search(command):
            return ApprovalDecision(action="block", reason=reason)
    for pattern, reason in _GREY_RULES:
        if pattern.search(command):
            return ApprovalDecision(action="needs_approval", reason=reason)
    return ApprovalDecision(action="allow", reason="")
