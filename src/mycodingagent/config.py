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

from dotenv import load_dotenv

# 项目根目录（config.py 位于 src/mycodingagent/ 下，向上三级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ------------------------------------------------------------
# 加载项目根目录的 .env 文件（若存在；系统环境变量优先，不会被覆盖）。
# 必须在下方所有 os.getenv 之前执行。
# ------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

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
# 阶段 2：三重预算与验证闭环
# AGENT_RECURSION_LIMIT：LangGraph 图步数上限（模型节点/工具节点各算一步），
#   超限抛 GraphRecursionError，状态保留在 checkpoint，可用 /resume 续跑。
# AGENT_MAX_STEPS：每任务模型调用轮次上限（事件流层计数，超限安全停下）。
# AGENT_TOKEN_BUDGET：每任务累计 token 上限（按 AI 消息 usage_metadata 累计）。
# VERIFY_LOOP_MAX_ROUNDS：注入 system prompt 的「改完→跑测试→修复」最大轮数。
# ------------------------------------------------------------
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "80"))
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "40"))
AGENT_TOKEN_BUDGET = int(os.getenv("AGENT_TOKEN_BUDGET", "300000"))
VERIFY_LOOP_MAX_ROUNDS = int(os.getenv("VERIFY_LOOP_MAX_ROUNDS", "5"))

# ------------------------------------------------------------
# 存储路径：锚定项目根目录，不受启动时的工作目录影响
# ------------------------------------------------------------
CHECKPOINT_DB = str(PROJECT_ROOT / "agent_state.sqlite")  # 短期记忆：对话历史
STORE_DB = str(PROJECT_ROOT / "agent_memory.sqlite")  # 长期记忆：用户信息

# ------------------------------------------------------------
# 工作目标目录（阶段 1：从固定 workspace 改为可指定任意项目目录）
# 优先级：CLI --project 参数 > 环境变量 AGENT_PROJECT_DIR > 默认 workspace/
# CLI 通过 set_project_dir() 在构建 Agent 前切换。
# ------------------------------------------------------------
def _default_project_dir() -> Path:
    env_dir = os.getenv("AGENT_PROJECT_DIR", "")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return PROJECT_ROOT / "workspace"


PROJECT_DIR = _default_project_dir()


def set_project_dir(path: str | Path) -> None:
    """运行时切换工作目标目录（CLI --project 调用）。目录必须已存在。"""
    global PROJECT_DIR
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"项目目录不存在：{resolved}")
    PROJECT_DIR = resolved
