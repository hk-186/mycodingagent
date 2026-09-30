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

from deepagents import create_deep_agent

from mycodingagent import config
from mycodingagent.approvals import should_interrupt_command
from mycodingagent.project_middleware import ProjectMemoryMiddleware
from mycodingagent.prompt_runtime import RuntimePromptMiddleware
from mycodingagent.tools.ask_user import ask_user
from mycodingagent.tools.calculator import calculate
from mycodingagent.tools.git_tools import git_commit, git_diff, git_log, git_status
from mycodingagent.tools.plan import propose_plan
from mycodingagent.tools.project_memory import (
    list_project_facts,
    recall_project_fact,
    save_project_fact,
)
from mycodingagent.tools.shell import SafeShellBackend
from mycodingagent.tools.user_memory import (
    recall_user_info,
    recall_user_info_list,
    save_user_info,
)

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


def build_deep_agent(checkpointer, store, backend=None):
    """
    create_deep_agent 关键参数：

    - backend：SafeShellBackend 把文件工具与 execute 的根目录锚定在项目目录，
      并在执行 shell 命令前做危险命令拦截（阶段 1）。可由外部传入（CLI /cd
      需要持有同一实例在运行时切换目录）；不传则内部构造一个。
    - middleware：TodoListMiddleware 提供 write_todos 工具（阶段 1 缺口 C6）；
      RuntimePromptMiddleware 在每次调模型前替换 prompt 中的运行时占位符。
    - subagents：注册后主 Agent 自动获得 task 工具，按 description 决定何时委派
    - checkpointer / store：短期记忆（对话历史）/ 长期记忆（用户信息）
    - memory：AGENTS.md 风格的规则记忆，内容每轮注入 system prompt
    """
    llm = get_llm()

    # 确保默认工作目录存在（真实项目目录已存在，此调用对它是 no-op）
    config.PROJECT_DIR.mkdir(exist_ok=True)

    # backend 可由外部注入（运行时 /cd 需要 CLI 持有同一实例）
    if backend is None:
        backend = SafeShellBackend(root_dir=str(config.PROJECT_DIR), inherit_env=True)

    # 主 Agent 的工具：时间 + 个人记忆 + 项目记忆 + Git 工具 + ask_user + propose_plan
    # （+ 设置了密钥时的联网搜索）
    tools = [
        get_current_time,
        calculate,
        save_user_info,
        recall_user_info,
        recall_user_info_list,
        save_project_fact,
        recall_project_fact,
        list_project_facts,
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

    # 阶段 5 子代理：code-reviewer（改动后审查）、test-writer（补测试）
    # 两个子代理都只读（不给写工具），输出结构化意见由主 Agent 决定是否采纳；
    # 显式 interrupt_on={} 避免继承父级审批配置。
    code_reviewer_subagent = {
        "name": "code-reviewer",
        "description": (
            "代码审查专家。主 Agent 完成代码改动并验证通过后，把本次改动的文件"
            "或 git diff 交给它审查：正确性、边界情况、可维护性、潜在 bug 与"
            "安全风险。输入应说明改了哪些文件、任务目标是什么。"
        ),
        "model": llm,
        "system_prompt": (
            "你是资深代码审查专家，只读不改。收到审查请求后：\n"
            "1. 用 read_file / grep / glob / ls 查看相关改动与上下文，必要时用 "
            "execute 运行测试验证；不要修改任何文件。\n"
            "2. 输出结构化中文审查意见，按严重度分组：\n"
            "   - 【必须修改】明确的 bug、安全问题、会导致失败的错误（给出文件与原因）\n"
            "   - 【建议改进】可维护性、可读性、设计取舍\n"
            "   - 【确认通过】没有严重问题时明确说明\n"
            "3. 只报告有依据的问题，禁止泛泛而谈或编造；没有问题就直接说通过。"
        ),
        "interrupt_on": {},
    }

    test_writer_subagent = {
        "name": "test-writer",
        "description": (
            "测试编写专家。需要为新功能/改动补充测试时委派给它：分析目标代码的"
            "行为与边界，给出应补充的测试用例清单和完整测试代码建议。"
        ),
        "model": llm,
        "system_prompt": (
            "你是测试工程师专家，只读分析并产出测试方案。收到请求后：\n"
            "1. 用 read_file / grep / glob 看清目标代码与现有测试的组织方式、"
            "框架与命名约定；不要修改任何文件。\n"
            "2. 输出中文测试方案：正常路径、边界值、异常路径各应覆盖哪些 case；\n"
            "3. 给出符合项目现有风格的完整测试代码建议，并说明放在哪个文件；\n"
            "4. 只针对本次改动相关的行为设计用例，不堆砌无关测试。"
        ),
        "interrupt_on": {},
    }

    agent = create_deep_agent(
        model=llm,
        tools=tools,
        backend=backend,
        middleware=[
            TodoListMiddleware(),
            ProjectMemoryMiddleware(backend=backend),  # AGENTS.md：/cd 后跟随重载
            RuntimePromptMiddleware(),
        ],
        subagents=[code_reviewer_subagent, test_writer_subagent],
        interrupt_on=interrupt_on_config,
        system_prompt=(
            "你是 mycodingagent，一个在用户真实仓库里工作的 coding agent。\n"
            "当前工作目标目录：__PROJECT_DIR__\n"
            "审批模式：__APPROVAL_MODE____PLAN_MODE_LABEL__\n"
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
            "__PLAN_MODE_SUFFIX__"
            "\n## 记忆与多代理协作（阶段 5）\n"
            "- 两层长期记忆：个人信息（姓名/邮箱等跨项目属性）用 "
            "save_user_info / recall_user_info；项目约定（测试命令、代码风格、"
            "架构决策）用 save_project_fact / recall_project_fact，项目约定按项目"
            "隔离，不要混用。\n"
            "- 发现项目特有的约定或用户纠正了你的工作方式时，主动用 "
            "save_project_fact 记下来，让后续会话能延续。\n"
            "- 项目根目录的 AGENTS.md 会自动加载（若存在）；把它视为项目规则"
            "参考，与用户明确指令冲突时以用户为准。\n"
            "- 代码改动并验证通过后，必须用 task 工具委派 code-reviewer 审查"
            "本次改动（说明任务目标与改动文件）；收到【必须修改】意见要处理后"
            "重新验证，【确认通过】后再向用户汇报。\n"
            "- 用户要求补测试，或改动缺少测试覆盖时，用 task 委派 test-writer "
            "获取测试方案，由你负责落地并跑通。\n"
        ),
        checkpointer=checkpointer,  # 短期记忆：对话历史按 thread_id 持久化
        store=store,  # 长期记忆：个人/项目记忆工具读写它
    )
    return agent
