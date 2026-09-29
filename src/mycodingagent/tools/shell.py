# -*- coding: utf-8 -*-
"""
Shell 执行（缺口 C1）
====================
在 deepagents 的 LocalShellBackend 外面加一层危险命令拦截，
作为 create_deep_agent 的 backend 注入：

- 父类已实现 cwd 锚定 root_dir、默认超时、输出截断、exit code 返回，
  并且 deepagents 检测到后端支持 execute 时会自动注册 execute 工具；
- 本类只在执行前多加一道 permissions.check_command 黑名单检查。
"""

from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from mycodingagent import permissions


class SafeShellBackend(LocalShellBackend):
    """带危险命令拦截的本地 shell 后端。

    LocalShellBackend 默认 env 为空（连 PATH 都没有），必须显式
    inherit_env=True 继承当前进程环境变量，否则任何命令都无法执行。
    """

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        reason = permissions.check_command(command)
        if reason:
            return ExecuteResponse(
                output=(
                    f"已拒绝执行：命令命中危险命令黑名单（{reason}）。\n"
                    "如确有需要，请直接向用户说明，让用户手动执行。"
                ),
                exit_code=None,
                truncated=False,
            )
        return super().execute(command, timeout=timeout)
