# -*- coding: utf-8 -*-
"""
项目 AGENTS.md 加载中间件（阶段 5）
===================================
deepagents 自带的 `MemoryMiddleware` 会把首次加载的 memory 内容缓存在
图状态中（`memory_contents` 存在就不再加载）。运行时 `/cd` 切换项目目录后，
缓存仍是旧项目的 AGENTS.md，造成约定陈旧。

本中间件扩展它：状态中额外记录已加载内容对应的项目路径，
- 同项目：保持父类缓存行为，不重复读盘；
- 换项目（/cd 后）：重新加载新项目根目录的 AGENTS.md。

加载源固定为项目根 `/AGENTS.md`；文件缺失时自动跳过（父类行为）。
"""

from __future__ import annotations

from typing import Annotated, NotRequired

from langchain.agents.middleware.types import PrivateStateAttr
from deepagents.middleware.memory import MemoryMiddleware, MemoryState

from mycodingagent import config


class ProjectMemoryState(MemoryState):
    """扩展状态：记录 memory_contents 对应的项目目录。"""

    project_memory_path: NotRequired[Annotated[str, PrivateStateAttr]]


class ProjectMemoryMiddleware(MemoryMiddleware):
    """加载项目根 AGENTS.md，并在 /cd 切换项目后自动重载。"""

    state_schema = ProjectMemoryState

    def __init__(self, *, backend) -> None:
        super().__init__(backend=backend, sources=["/AGENTS.md"])

    def _parse_results(self, results) -> dict[str, str]:
        """解析 download_files 返回序列（file_not_found 自动跳过）。"""
        contents: dict[str, str] = {}
        for path, response in zip(self.sources, results, strict=True):
            if response.error is not None:
                if response.error == "file_not_found":
                    continue
                raise ValueError(f"Failed to download {path}: {response.error}")
            if response.content is not None:
                contents[path] = response.content.decode("utf-8")
        return contents

    def before_agent(self, state, runtime):  # noqa: ANN001, ARG001
        current = str(config.PROJECT_DIR)
        if "memory_contents" in state and state.get("project_memory_path") == current:
            return None  # 同项目：沿用缓存
        results = self._backend.download_files(list(self.sources))
        return {
            "memory_contents": self._parse_results(results),
            "project_memory_path": current,
        }

    async def abefore_agent(self, state, runtime):  # noqa: ANN001, ARG001
        current = str(config.PROJECT_DIR)
        if "memory_contents" in state and state.get("project_memory_path") == current:
            return None
        results = await self._backend.adownload_files(list(self.sources))
        return {
            "memory_contents": self._parse_results(results),
            "project_memory_path": current,
        }
