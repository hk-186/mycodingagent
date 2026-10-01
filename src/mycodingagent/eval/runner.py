# -*- coding: utf-8 -*-
"""
eval 单任务运行器
=================
核心流程（run_one）：
    1. 复制 fixture 到 TemporaryDirectory 沙箱，git init + 初始提交
       （供 llm_review 拿 diff、让 agent 的 git 工具可用）；
    2. config.set_project_dir(沙箱)，构建真实 agent（InMemorySaver /
       InMemoryStore——不读写项目根目录的两个 sqlite，零污染）；
    3. auto_approve_loop：复刻 cli.run_task 的 INTERRUPT 循环，
       把人工决策换成自动决策（ask_user → respond，其余 → approve）；
    4. 在沙箱存活期内执行 grader，组装 TaskResult；
    5. finally 恢复 config.PROJECT_DIR。

超时为 wall-clock 硬上限：事件流在子线程消费、主线程限时收事件
（_events_with_deadline），即使流阻塞在子代理等长调用上也能按时放弃
（进行中的单次 LLM HTTP 调用由 LLM_TIMEOUT 兜底）。
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from mycodingagent import config
from mycodingagent.events import (
    ASSISTANT_MESSAGE,
    ERROR,
    INTERRUPT,
    SUMMARY,
    TOOL_END,
    TOOL_START,
    iter_task_events,
)
from mycodingagent.eval.graders import GradeContext, GraderResult, run_graders
from mycodingagent.eval.loader import EvalTask

logger = logging.getLogger(__name__)

STOP_TIMEOUT = "timeout"  # events.py 之外的 runner 侧停止原因：wall-clock 超时


@dataclass
class TaskOutcome:
    """agent 运行产物（grader 判定前）。"""

    final_report: str = ""
    steps: int = 0
    tokens: int = 0
    stopped_reason: str = ""
    error: str = ""


@dataclass
class TaskResult:
    """单任务的完整 eval 结果（report.py 序列化它落盘）。"""

    name: str
    type: str
    passed: bool
    grader_results: list[GraderResult] = field(default_factory=list)
    steps: int = 0
    tokens: int = 0
    elapsed_seconds: float = 0.0
    stopped_reason: str = ""
    error: str = ""
    final_report: str = ""
    diff: str = ""


# ============================================================
# 可选 LangSmith 观测层（默认关；只增强追踪/反馈，不参与判定）
# ============================================================
_LANGSMITH_ENV_KEYS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT",
)


def _configure_langsmith(task: EvalTask) -> dict[str, str | None]:
    """EVAL_LANGSMITH=1 时开启 LangSmith tracing；返回需恢复的 env 快照。"""
    if not config.EVAL_LANGSMITH:
        return {}
    previous = {key: os.environ.get(key) for key in _LANGSMITH_ENV_KEYS}
    project = (
        config.EVAL_LANGSMITH_PROJECT
        or os.environ.get("LANGCHAIN_PROJECT")
        or "mycodingagent-eval"
    )
    # 新版 SDK 读 LANGSMITH_*；LangChain 旧集成读 LANGCHAIN_*。两套都写，
    # 保证 langsmith 0.13+ 与旧版 langchain-core 都能触发详细 trace。
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project
    previous["__project__"] = project
    return previous


def _restore_langsmith_env(previous: dict[str, str | None]) -> None:
    for key in _LANGSMITH_ENV_KEYS:
        if key not in previous:
            continue
        value = previous[key]
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _emit_langsmith_feedback(
    task: EvalTask, result: TaskResult, project: str | None
) -> None:
    """把 eval 汇总写成一条 LangSmith run + feedback；失败只告警，不改判定。"""
    if not config.EVAL_LANGSMITH or not project:
        return
    try:
        from langsmith import Client, RunTree
    except Exception as e:  # noqa: BLE001 — LangSmith 是可选依赖
        logger.warning("EVAL_LANGSMITH 已开启但 langsmith 不可用：%s", e)
        return

    try:
        client = Client()
        # 新版 feedback API 要求 session_id（= project/session ID）；
        # RunTree.post 不会回填该字段，显式查一次 project。
        session_id = client.read_project(project_name=project).id
        run = RunTree(
            name=f"eval:{task.name}",
            run_type="chain",
            inputs={"prompt": task.prompt, "task_type": task.type},
            project_name=project,
            tags=["eval", task.name, task.type],
            metadata={
                "timeout_seconds": task.timeout_seconds,
                "stopped_reason": result.stopped_reason,
            },
        )
        run.post()
        run.end(
            outputs={
                "passed": result.passed,
                "stopped_reason": result.stopped_reason,
                "error": result.error,
                "steps": result.steps,
                "tokens": result.tokens,
                "elapsed_seconds": result.elapsed_seconds,
                "final_report": result.final_report,
                "diff": result.diff,
                "grader_results": [
                    {
                        "type": r.type,
                        "passed": r.passed,
                        "detail": r.detail,
                        "score": r.score,
                    }
                    for r in result.grader_results
                ],
            },
            error=result.error or None,
        )
        run.patch()
        client.create_feedback(
            run_id=run.id,
            key="eval_pass",
            score=1.0 if result.passed else 0.0,
            comment=result.error or result.stopped_reason,
            session_id=session_id,
        )
        for r in result.grader_results:
            if r.type == "llm_review":
                client.create_feedback(
                    run_id=run.id,
                    key="llm_review",
                    score=float(r.score),
                    comment=r.detail,
                    session_id=session_id,
                )
    except Exception as e:  # noqa: BLE001 — 观测层故障不拖垮 eval
        logger.warning("LangSmith 写入失败（不影响 eval 判定）：%s", e)


# ============================================================
# 沙箱内 git 辅助（grader 的 diff 来源；git 工具也需要仓库环境）
# ============================================================
def _git(sandbox: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(sandbox),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def _git_init_commit(sandbox: Path) -> None:
    """git init 并提交 fixture 初始状态；失败抛 RuntimeError（git 是硬依赖）。

    先写 .gitignore 排除 __pycache__/ 等运行产物，避免 agent 跑测试后
    生成的缓存文件混入 git diff 干扰 llm_review。
    """
    (sandbox / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n.pytest_cache/\n", encoding="utf-8"
    )
    for args in (
        ("init", "-q"),
        ("add", "-A"),
        ("-c", "user.email=eval@mycodingagent.local", "-c", "user.name=eval",
         "commit", "-q", "--allow-empty", "-m", "fixture 初始状态"),
    ):
        proc = _git(sandbox, *args)
        if proc.returncode != 0:
            raise RuntimeError(
                f"沙箱 git {' '.join(args)} 失败：{proc.stderr.strip()}"
            )


def _git_diff(sandbox: Path) -> str:
    """取工作区相对初始提交的完整 diff（含未跟踪新文件内容）。"""
    # 先把新文件纳入索引再 diff HEAD，否则 untracked 文件不出现在 diff 里
    _git(sandbox, "add", "-A")
    proc = _git(sandbox, "diff", "HEAD")
    return proc.stdout if proc.returncode == 0 else ""


# ============================================================
# 自动 approve 事件循环（复刻 cli.run_task，人工决策 → 自动决策）
# ============================================================
def _events_with_deadline(
    agent: Any,
    *,
    thread_id: str,
    user_input: str | None,
    resume_decisions: list[dict] | None,
    deadline: float,
) -> Iterator[Any]:
    """在子线程消费 iter_task_events，主线程按 deadline 限时收事件。

    事件流可能长时间阻塞在 next(stream) 上（如子代理任务执行期间主图
    不产生事件），直接在事件循环里检查 wall-clock 永远不会触发。
    子线程 + 队列让超时在无事件时也能生效；放弃后 daemon 子线程随
    沙箱/InMemorySaver 一起被丢弃，无副作用。
    """
    q: queue.Queue = queue.Queue()

    def consume() -> None:
        try:
            for ev in iter_task_events(
                agent,
                thread_id=thread_id,
                user_input=user_input,
                resume_decisions=resume_decisions,
            ):
                q.put(ev)
        except Exception as e:  # noqa: BLE001 — 透传给主线程处理
            q.put(e)
        finally:
            q.put(None)  # 结束标记

    threading.Thread(target=consume, daemon=True).start()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return  # 调用方据 deadline 判定超时
        try:
            item = q.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def auto_approve_loop(
    agent: Any,
    task: EvalTask,
    *,
    deadline: float,
    progress: Callable[[Any], None] | None = None,
) -> TaskOutcome:
    """跑一轮任务直到 SUMMARY / 超时 / 错误。

    遇 INTERRUPT：ask_user 用 task.ask_user_reply 代答（respond），
    其余 action_request 一律 approve。
    """
    outcome = TaskOutcome()
    next_user_input: str | None = task.prompt
    next_decisions: list[dict] | None = None

    while True:
        interrupt_ev = None
        for ev in _events_with_deadline(
            agent,
            thread_id=f"eval-{task.name}",
            user_input=next_user_input,
            resume_decisions=next_decisions,
            deadline=deadline,
        ):
            # 用 >= 而非 >：_events_with_deadline 在 remaining<=0 时返回，
            # 即 monotonic() >= deadline 恒成立；Windows 定时器粒度约 15.6ms，
            # 恰好相等时用 > 会漏判超时
            if time.monotonic() >= deadline:
                break
            if progress is not None and ev.type in (TOOL_START, TOOL_END, INTERRUPT):
                progress(ev)
            if ev.type == INTERRUPT:
                interrupt_ev = ev
            elif ev.type == ASSISTANT_MESSAGE:
                outcome.final_report = ev.text  # 取最后一条汇报
            elif ev.type == SUMMARY:
                outcome.steps = ev.steps
                outcome.tokens = ev.tokens
                outcome.stopped_reason = ev.stopped_reason
            elif ev.type == ERROR:
                outcome.error = ev.text

        if time.monotonic() >= deadline:
            outcome.stopped_reason = STOP_TIMEOUT
            outcome.error = f"超过单任务时限（{task.timeout_seconds}s），已放弃"
            return outcome

        if interrupt_ev is None:
            # 无 SUMMARY 也无 INTERRUPT：异常路径，error 已记录
            if not outcome.stopped_reason:
                outcome.stopped_reason = "error"
            return outcome

        # 自动决策：与 action_requests 等长
        next_decisions = [
            {"type": "respond", "message": task.ask_user_reply}
            if ar.get("name") == "ask_user"
            else {"type": "approve"}
            for ar in interrupt_ev.action_requests
        ]
        next_user_input = None


# ============================================================
# 单任务入口
# ============================================================
def run_one(
    task: EvalTask,
    *,
    build_agent: Callable[..., Any] | None = None,
    progress: Callable[[Any], None] | None = None,
) -> TaskResult:
    """在隔离沙箱里运行单个 eval 任务并判定。

    build_agent 可注入（单测用 FakeAgent 工厂）；缺省为
    agent.build_deep_agent（真实 LLM）。
    """
    if build_agent is None:
        from mycodingagent.agent import build_deep_agent

        build_agent = build_deep_agent

    started = time.monotonic()
    original_project_dir = config.PROJECT_DIR
    # eval 求快速失败：端点抖动时高重试会把单次逻辑调用放大到
    # LLM_TIMEOUT×(1+retries)，子代理多轮累积易超单任务时限。
    # 临时覆盖 LLM_MAX_RETRIES（agent/llm_review 构建时读取），结束恢复。
    original_max_retries = config.LLM_MAX_RETRIES
    config.LLM_MAX_RETRIES = config.EVAL_LLM_MAX_RETRIES
    langsmith_env = _configure_langsmith(task)
    try:
        with tempfile.TemporaryDirectory(prefix="mca-eval-") as sandbox_str:
            sandbox = Path(sandbox_str)
            shutil.copytree(task.fixture_dir, sandbox, dirs_exist_ok=True)
            _git_init_commit(sandbox)

            config.set_project_dir(sandbox)
            try:
                from mycodingagent.tools.shell import SafeShellBackend

                backend = SafeShellBackend(
                    root_dir=str(sandbox), inherit_env=True
                )
                agent = build_agent(InMemorySaver(), InMemoryStore(), backend=backend)
                outcome = auto_approve_loop(
                    agent, task,
                    deadline=started + task.timeout_seconds,
                    progress=progress,
                )
                diff = _git_diff(sandbox)

                ctx = GradeContext(
                    sandbox=sandbox,
                    diff=diff,
                    final_report=outcome.final_report,
                    task_prompt=task.prompt,
                )
                grader_results = run_graders(task.graders, ctx)
            finally:
                config.set_project_dir(original_project_dir)
                config.LLM_MAX_RETRIES = original_max_retries
                _restore_langsmith_env(langsmith_env)

        # 判定语义：grader 全过且 agent 正常收尾（completed）才算 pass。
        # 超时/超限场景下文件可能已改对（grader 过），但 agent 没走完流程，
        # 这本身是回归集要暴露的信号（效率退化/API 抖动），不能混为 PASS。
        passed = (
            bool(grader_results)
            and all(r.passed for r in grader_results)
            and outcome.stopped_reason == "completed"
        )
        result = TaskResult(
            name=task.name,
            type=task.type,
            passed=passed,
            grader_results=grader_results,
            steps=outcome.steps,
            tokens=outcome.tokens,
            elapsed_seconds=round(time.monotonic() - started, 1),
            stopped_reason=outcome.stopped_reason,
            error=outcome.error,
            final_report=outcome.final_report,
            diff=diff,
        )
        _emit_langsmith_feedback(task, result, langsmith_env.get("__project__"))
        return result
    except Exception:  # noqa: BLE001 — 兜底恢复全局状态后重抛
        config.set_project_dir(original_project_dir)
        config.LLM_MAX_RETRIES = original_max_retries
        _restore_langsmith_env(langsmith_env)
        raise
