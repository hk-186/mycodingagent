# -*- coding: utf-8 -*-
"""
危险命令拦截（阶段 1，对应缺口 C1 的安全配套）
==============================================
shell 执行前的第一道防线：命中黑名单的命令直接拒绝，不进入执行环节。

设计取向（对应开发计划"不可妥协的设计裁决"第 2 条）：
- 宁可误拦，不可漏拦。按"初步拦截"定位，覆盖 Windows（cmd / PowerShell）
  与 bash 两类常见破坏性命令。
- 本模块只做拦截不做审批；人工审批 + diff 预览是阶段 3 的任务。
"""

import re

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

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in _DANGEROUS_PATTERNS
]


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
