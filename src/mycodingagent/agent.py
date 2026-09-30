# -*- coding: utf-8 -*-
"""
Agent 构建
==========
模型客户端、工具集、子代理与记忆的接线逻辑，
从 deep_agent.py 迁移并按集中配置改造。
"""

import logging
import platform
import sys
from datetime import datetime

from langchain.agents.middleware import TodoListMiddleware  # write_todos 工具 + todos 状态
from langchain.agents.middleware import InterruptOnConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.config import get_store  # 工具运行时由 agent 注入 store

from deepagents import create_deep_agent

from mycodingagent import config
from mycodingagent.approvals import should_interrupt_command
from mycodingagent.tools.ask_user import ask_user
from mycodingagent.tools.calculator import calculate
from mycodingagent.tools.git_tools import git_commit, git_diff, git_log, git_status
from mycodingagent.tools.plan import propose_plan
from mycodingagent.tools.shell import SafeShellBackend

logger = logging.getLogger(__name__)


def get_llm() -> ChatOpenAI:
    """构建 LLM 客户端：temperature / timeout / 重试次数走集中配置（修复计划 E7）。"""
    return ChatOpenAI(
        model=config.MODEL_NAME,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
    )


# ============================================================
# 主 Agent 的自定义工具：获取当前时间
# （文件系统工具不用自己定义——create_deep_agent 已内置）
# ============================================================
@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


# ============================================================
# 长期记忆的读写工具（信息存入 SqliteStore，跨会话共享）
# get_store() 在工具运行时取出 create_deep_agent(store=...) 注入的 store
# ============================================================
@tool
def save_user_info(key: str, value: str) -> str:
    """保存用户的长期信息。key 是信息类别（如 name、age、hobby），value 是具体内容"""
    store = get_store()
    store.put(("users",), key, {"value": value})
    return f"已记住：{key} = {value}"


@tool
def recall_user_info(key: str) -> str:
    """回忆用户的长期信息。key 是信息类别（如 name、age、hobby），不确定时可用 recall_user_info_list"""
    store = get_store()
    item = store.get(("users",), key)
    if item is None:
        # 精确 key 没找到时，本地模糊匹配一遍，提升召回率
        hits = [
            i for i in store.search(("users",)) if key in i.key or key in str(i.value)
        ]
        if hits:
            return "；".join(f"{h.key} = {h.value['value']}" for h in hits)
        return f"没有找到 {key} 相关的记忆"
    return f"{key} = {item.value['value']}"


@tool
def recall_user_info_list() -> str:
    """列出所有已保存的用户长期信息"""
    store = get_store()
    items = store.search(("users",))
    if not items:
        return "还没有保存任何用户信息"
    return "；".join(f"{i.key} = {i.value['value']}" for i in items)


# ============================================================
# 构建 Deep Agent
# ============================================================
def _shell_environment_hint() -> str:
    """告诉模型 execute 实际运行的 shell 环境，避免按错平台写命令。

    subprocess(shell=True) 在 Windows 上走 cmd.exe，POSIX 上走 /bin/sh，
    两套语法不同（分隔符、列目录、环境变量等），必须显式告知。
    """
    if sys.platform == "win32":
        return (
            f"运行平台：Windows（{platform.release()}），execute 的命令由 cmd.exe 执行。\n"
            "- 多条命令用 && 连接，禁止用分号 ;\n"
            "- 列目录用 dir（不是 ls），查看当前目录用 cd（不是 pwd）\n"
            "- 路径用反斜杠或加引号；环境变量写作 %VAR%（不是 $VAR）\n"
            "- 递归删除目录属于危险操作会被拦截，确有需要请让用户手动执行\n"
        )
    return (
        f"运行平台：{platform.system()}，execute 的命令由 /bin/sh 执行。\n"
        "- 多条命令用 && 连接，列目录用 ls，查看当前目录用 pwd。\n"
    )


# ============================================================
# 阶段 3：审批请求的描述工厂（InterruptOnConfig.description callable）
# 把 tool_call 的 args 渲染成可读的审批摘要给用户看
# ============================================================
def _execute_description_factory(tool_call, state, runtime) -> str:  # noqa: ANN001
    """execute 工具审批摘要：展示待执行的 shell 命令。"""
    command = (tool_call.get("args") or {}).get("command", "")
    return f"即将执行 shell 命令：\n  {command}"


def _commit_description_factory(tool_call, state, runtime) -> str:  # noqa: ANN001
    """git_commit 审批摘要：展示 status + 已暂存 diff 摘要。"""
    import subprocess  # 局部导入避免顶部循环依赖

    message = (tool_call.get("args") or {}).get("message", "")
    parts = [f"提交信息：{message}"]

    # 取工作区状态
    try:
        status = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=str(config.PROJECT_DIR), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
        if status.returncode == 0 and status.stdout.strip():
            parts.append(f"\n当前工作区状态：\n{status.stdout.strip()}")
    except Exception:  # noqa: BLE001
        pass

    # 取已暂存 diff 摘要
    try:
        diff = subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            cwd=str(config.PROJECT_DIR), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
        if diff.returncode == 0 and diff.stdout.strip():
            parts.append(f"\n本次提交将包含的变更：\n{diff.stdout.strip()}")
        else:
            parts.append("\n（无已暂存变更——提交可能失败，建议先 git add）")
    except Exception:  # noqa: BLE001
        pass

    return "\n".join(parts)


def _plan_prompt_suffix() -> str:
    """Plan 模式开启时追加到 system prompt 的额外指令。"""
    if not config.PLAN_MODE:
        return ""
    return (
        "\n## Plan 模式（已开启）\n"
        "- 当前处于 Plan 模式：所有写操作（execute 写命令、edit_file、write_file、"
        "git_commit）都会被自动拦截等待用户审批，不会真正执行。\n"
        "- 你的任务是：先调研代码现状（read_file / ls / grep / glob 只读工具），"
        "形成清晰的改动计划，然后调用 propose_plan 工具提交计划等用户审批。\n"
        "- propose_plan 的 plan 参数应包含：要改的文件列表、每个文件的改动要点、"
        "验证方式（跑哪个测试/脚本）、潜在风险。\n"
        "- 计划被 approve 后才能开始动手；被 reject 要根据理由重规划。\n"
    )


def build_deep_agent(checkpointer, store):
    """
    create_deep_agent 关键参数：

    - backend：SafeShellBackend 把文件工具与 execute 的根目录锚定在项目目录，
      并在执行 shell 命令前做危险命令拦截（阶段 1）
    - middleware：TodoListMiddleware 提供 write_todos 工具（阶段 1 缺口 C6）
    - subagents：注册后主 Agent 自动获得 task 工具，按 description 决定何时委派
    - checkpointer / store：短期记忆（对话历史）/ 长期记忆（用户信息）
    - memory：AGENTS.md 风格的规则记忆，内容每轮注入 system prompt
    """
    llm = get_llm()

    # 确保默认工作目录存在（真实项目目录已存在，此调用对它是 no-op）
    config.PROJECT_DIR.mkdir(exist_ok=True)

    # 主 Agent 的工具：时间 + 长期记忆读写 + Git 工具 + ask_user + propose_plan
    # （+ 设置了密钥时的联网搜索）
    tools = [
        get_current_time,
        save_user_info,
        recall_user_info,
        recall_user_info_list,
        git_status,
        git_diff,
        git_log,
        git_commit,
        ask_user,
        propose_plan,
    ]
    search_hint = ""
    if config.TAVILY_API_KEY:
        from langchain_tavily import TavilySearch

        tools.append(TavilySearch(max_results=3))
        search_hint = "- web_search：回答时效性/事实性问题前先用它查最新资料\n"
    else:
        logger.info("未设置 TAVILY_API_KEY，联网搜索功能未启用")

    # 阶段 3：interrupt_on 配置——deepagents 原生 HITL 能力
    # - execute：危险/灰区命令（plan_mode=all 时所有写命令）触发审批
    # - git_commit：展示 status+diff 摘要后等审批
    # - ask_user：respond 决策让用户代答
    # - propose_plan：approve/reject 决策
    interrupt_on_config = {
        "execute": InterruptOnConfig(
            when=should_interrupt_command,
            allowed_decisions=["approve", "reject"],
            description=_execute_description_factory,
        ),
        "git_commit": InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            description=_commit_description_factory,
        ),
        "ask_user": InterruptOnConfig(
            allowed_decisions=["respond"],
        ),
        "propose_plan": InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
        ),
    }

    # 子代理：专门负责数学计算，拥有独立的工具集和系统提示词
    # 显式 interrupt_on={} 避免继承父级审批配置（Plan agent 验证建议）
    calculator_subagent = {
        "name": "calculator",
        "description": (
            "数学计算专家。任何需要精确计算的问题都委派给它，"
            "例如四则运算、百分比、乘方。输入完整的数学表达式。"
        ),
        "tools": [calculate],
        "model": llm,
        "system_prompt": (
            "你是数学计算专家。收到表达式后必须调用 calculate 工具计算，"
            "禁止心算。用中文简要返回计算结果。"
        ),
        "interrupt_on": {},  # 子代理禁用审批继承
    }

    agent = create_deep_agent(
        model=llm,
        tools=tools,
        backend=SafeShellBackend(root_dir=str(config.PROJECT_DIR), inherit_env=True),
        middleware=[TodoListMiddleware()],
        subagents=[calculator_subagent],   # 注册子代理 → 主 Agent 获得 task 工具
        interrupt_on=interrupt_on_config,
        system_prompt=(
            f"你是 mycodingagent，一个在用户真实仓库里工作的 coding agent。\n"
            f"当前工作目标目录：{config.PROJECT_DIR}\n"
            f"审批模式：{config.APPROVAL_MODE}"
            + ("（已开启 Plan 模式）" if config.PLAN_MODE else "")
            + "\n"
            "\n## 文件路径约定（重要）\n"
            "- ls / read_file / write_file / edit_file / glob / grep 一律使用"
            "以 / 开头的虚拟路径，/ 就代表上面的项目目录本身：\n"
            "  / 映射到项目根目录；/src/app.py 映射到 项目目录/src/app.py。\n"
            "- 禁止给这些工具传 D:\\... 这类 Windows 绝对路径（会直接报错）。\n"
            "- 不要再把项目目录名拼到虚拟路径后面：例如项目目录是 workspace 时，"
            "用 / 而不是 /workspace（后者会去找 workspace/workspace）。\n"
            "- execute 的 shell 命令每次都已在项目目录内运行，无需 cd。\n"
            + _shell_environment_hint()
            + search_hint
            + "\n## 编码工作守则\n"
            "1. 先读后改：修改任何文件前必须先 read_file 看清现状，"
            "禁止凭想象猜测文件内容。\n"
            "2. 不臆造路径：不确定文件位置时，先用 glob / grep / ls 定位，"
            "再动手。\n"
            "3. 最小修改：只改动完成任务必需的部分，不顺手重构、"
            "不改动无关的格式与注释。\n"
            "4. 用 edit_file 做精确替换（old_string 必须在文件中唯一，"
            "否则会报错）；只有新建文件才用 write_file。\n"
            "5. 改完自测：每次修改后，用 execute 运行相关脚本或测试，"
            "拿到 exit code 和报错后继续修复，直到验证通过为止。\n"
            "6. 危险命令（递归删除、格式化磁盘、关机等）会被系统直接拒绝，"
            "被拒绝时不要换写法绕过，向用户说明即可。\n"
            + f"\n## 验证闭环（最多 {config.VERIFY_LOOP_MAX_ROUNDS} 轮）\n"
            "1. 每完成一处代码修改，立即用 execute 运行测试或脚本验证，"
            "不要攒多个修改后一次性验证，否则出错时无法定位是哪处改坏的。\n"
            "2. 验证失败时：读 stderr 和 exit code 定位原因 → 修复 → 重跑，"
            "循环直到测试全部通过。每次重跑前确认上一处报错已真正消除。\n"
            f"3. 修复尝试超过 {config.VERIFY_LOOP_MAX_ROUNDS} 轮仍失败：停止尝试，"
            "向用户汇报已尝试的方案、当前报错全文与你的怀疑方向，请用户决策，"
            "禁止无休止地换写法重试。\n"
            "4. 运行测试前先确认测试框架与命令（pytest -q / python 脚本等），"
            "不确定时先 ls / glob / grep 看清项目结构再跑，不要臆造测试命令。\n"
            "\n## 任务管理\n"
            "- 多步任务先调用 write_todos 列出计划，每完成一步立即更新状态，"
            "全部完成后再汇报。\n"
            "- 完成后用中文简要汇报：改了哪些文件、如何验证、结果如何。\n"
            "\n## 人机协作（阶段 3）\n"
            "- execute 的危险/灰区命令（rm、git reset、批量删除、卸载包等）会"
            "被自动拦截等用户审批。被 reject 时表示用户拒绝该次操作，"
            "不要换写法绕过，按拒绝理由调整方案即可。\n"
            "- git_commit 会先展示 status+已暂存 diff 摘要给用户审批，"
            "approve 后才真正提交。提交前请确认变更已 git add 暂存。\n"
            "- 信息不足（如多个合理实现方案、缺关键参数）时调用 ask_user "
            "主动向用户提问，不要瞎猜。问题要具体、可一句话回答。\n"
            "- Plan 模式由用户用 CLI /plan 命令显式开启；开启时所有写操作"
            "自动拦截，你应该先调研代码后调用 propose_plan 提交计划等审批。\n"
            + _plan_prompt_suffix()
        ),
        memory=["/AGENTS.md"],  # 规则型记忆：AGENTS.md 内容每轮注入 system prompt
        checkpointer=checkpointer,  # 短期记忆：对话历史按 thread_id 持久化
        store=store,  # 长期记忆：save/recall_user_info 工具读写它
    )
    return agent
