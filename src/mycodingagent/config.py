# -*- coding: utf-8 -*-
"""
集中配置
========
所有环境变量与默认值只在这一处定义，业务代码统一从 config 读取，
禁止再在源码中散落硬编码（修复计划 E3）。

优先级：环境变量 > 本文件默认值。
模板见项目根目录 .env.example。
"""

import os
from pathlib import Path

# 项目根目录（config.py 位于 src/mycodingagent/ 下，向上三级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ------------------------------------------------------------
# LangSmith 追踪：显式 opt-in（修复计划 E6）
# 只有用户主动设置 LANGSMITH_TRACING=true 才启用追踪，
# 并顺带补齐默认项目名与区域端点；不设置则完全不追踪。
# ------------------------------------------------------------
if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
    os.environ.setdefault("LANGSMITH_PROJECT", "mycodingagent")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://apac.api.smith.langchain.com")

# ------------------------------------------------------------
# 模型配置（修复计划 E3：不再硬编码在业务源码中）
# ------------------------------------------------------------
API_KEY = os.getenv("AGICTO_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.agicto.cn/v1")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-v4-flash")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# LLM 客户端参数（修复计划 E7：超时 / 重试 / coding 场景低温）
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ------------------------------------------------------------
# 存储路径：锚定项目根目录，不受启动时的工作目录影响
# ------------------------------------------------------------
CHECKPOINT_DB = str(PROJECT_ROOT / "agent_state.sqlite")  # 短期记忆：对话历史
STORE_DB = str(PROJECT_ROOT / "agent_memory.sqlite")  # 长期记忆：用户信息

# Agent 的"笔记本"目录：虚拟文件系统里写的文件都会落在这里
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
