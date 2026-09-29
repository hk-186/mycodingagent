# -*- coding: utf-8 -*-
"""
Shell 执行（缺口 C1）
====================
在 deepagents 的 LocalShellBackend 外面加两层增强，
作为 create_deep_agent 的 backend 注入：

1. 危险命令拦截：执行前过 permissions.check_command 黑名单；
2. Windows 输出编码修复：cmd 内置命令（dir 等）输出 GBK，
   父类按 UTF-8 解码会乱码（"下午" → "涓嬪崍"），这里捕获原始字节，
   按 UTF-8 优先、GBK 回退解码；同时让子 Python 进程统一输出 UTF-8。

其余行为（cwd 锚定 root_dir、超时、exit code、输出截断）与父类保持一致。
"""

import subprocess
import sys

from deepagents.backends.local_shell import DEFAULT_EXECUTE_TIMEOUT, LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from mycodingagent import permissions

# Windows 控制台/命令常用代码页：UTF-8 失败后回退 GBK（cp936）
_FALLBACK_ENCODINGS = ("utf-8", "gbk") if sys.platform == "win32" else ("utf-8", "latin-1")


def _decode_output(data: bytes) -> str:
    """按候选编码依次尝试解码子进程输出，全部失败时用替换字符兜底。"""
    for encoding in _FALLBACK_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class SafeShellBackend(LocalShellBackend):
    """带危险命令拦截和输出编码修复的本地 shell 后端。

    LocalShellBackend 默认 env 为空（连 PATH 都没有），必须显式
    inherit_env=True 继承当前进程环境变量，否则任何命令都无法执行。
    """

    def __init__(
        self,
        root_dir=None,
        *,
        timeout: int = DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> None:
        super().__init__(
            root_dir,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            env=env,
            inherit_env=inherit_env,
        )
        # 让被执行的子 Python 进程用 UTF-8 输出，配合 _decode_output 优先 UTF-8
        self._env.setdefault("PYTHONIOENCODING", "utf-8")

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        # 第一道：危险命令黑名单
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

        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        # 第二道：捕获原始字节自行解码（父类 text=True 在中文 Windows 上
        # 会把 GBK 输出按 UTF-8 解成乱码）
        try:
            result = subprocess.run(  # noqa: S602
                command,
                check=False,
                shell=True,  # Intentional: designed for LLM-controlled shell execution
                capture_output=True,
                stdin=subprocess.DEVNULL,  # 防止读 stdin 的命令挂起
                timeout=effective_timeout,
                env=self._env,
                cwd=str(self.cwd),
                start_new_session=(sys.platform != "win32"),
            )

            output_parts = []
            stdout = _decode_output(result.stdout)
            stderr = _decode_output(result.stderr)
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.extend(f"[stderr] {line}" for line in stderr.strip().split("\n"))

            output = "\n".join(output_parts) if output_parts else "<no output>"

            truncated = False
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            if result.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            if timeout is not None:
                msg = (
                    f"Error: Command timed out after {effective_timeout} seconds "
                    "(custom timeout). The command may be stuck or require more time."
                )
            else:
                msg = (
                    f"Error: Command timed out after {effective_timeout} seconds. "
                    "For long-running commands, re-run using the timeout parameter."
                )
            return ExecuteResponse(output=msg, exit_code=124, truncated=False)
        except Exception as e:  # noqa: BLE001 — 与父类一致：异常转成统一结果而非抛出
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )
