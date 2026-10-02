# -*- coding: utf-8 -*-
"""
TUI 任务执行器
==============
cli.run_task 的线程化版本：agent 事件流在后台线程消费，事件/状态变化
通过回调（App 侧用 call_from_thread 投递）渲染；遇 INTERRUPT 阻塞在
队列上等待用户在输入栏逐条提交决策，收齐后 resume。

状态机：running -> awaiting_decision（可反复）-> idle
取消（/cancel）或异常也回到 idle，由 App 决定提示文案。
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

from mycodingagent import config
from mycodingagent.events import (
    INTERRUPT,
    SUMMARY,
    AgentEvent,
    iter_task_events,
)

logger = logging.getLogger(__name__)

# 运行状态
IDLE = "idle"
RUNNING = "running"
AWAITING_DECISION = "awaiting_decision"


class TuiTaskRunner:
    """后台线程跑一轮 agent 任务。

    回调：
        on_event(ev)            每个 AgentEvent（含工具调用/消息/汇总）
        on_state(state, info)   状态切换；info 为 INTERRUPT 事件或错误串
    """

    def __init__(
        self,
        agent: Any,
        *,
        on_event: Callable[[AgentEvent], None],
        on_state: Callable[[str, Any], None],
    ) -> None:
        self._agent = agent
        self._on_event = on_event
        self._on_state = on_state
        self._decision_q: queue.Queue[list[dict] | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        """是否有任务在跑（含等待决策）。"""
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        *,
        thread_id: str,
        user_input: str | None,
        resume: bool = False,
        plan_mode: bool = False,
    ) -> None:
        """启动一轮任务；同一时间只允许一个。"""
        if self.busy:
            raise RuntimeError("已有任务在执行")
        # 清空可能的陈旧决策
        while not self._decision_q.empty():
            try:
                self._decision_q.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(
            target=self._run,
            kwargs={
                "thread_id": thread_id,
                "user_input": user_input,
                "resume": resume,
                "plan_mode": plan_mode,
            },
            daemon=True,
        )
        self._thread.start()

    def submit_decisions(self, decisions: list[dict] | None) -> None:
        """收齐的决策；None 表示取消本轮审批。"""
        self._decision_q.put(decisions)

    # ------------------------------------------------------------
    def _set_state(self, state: str, info: Any = None) -> None:
        self._on_state(state, info)

    def _run(
        self,
        *,
        thread_id: str,
        user_input: str | None,
        resume: bool,
        plan_mode: bool,
    ) -> None:
        self._set_state(RUNNING, user_input)
        next_user_input: str | None = None if resume else user_input
        next_decisions: list[dict] | None = None
        try:
            while True:
                interrupt_event: AgentEvent | None = None
                got_summary = False
                for ev in iter_task_events(
                    self._agent,
                    thread_id=thread_id,
                    user_input=next_user_input,
                    resume_decisions=next_decisions,
                ):
                    self._on_event(ev)
                    if ev.type == INTERRUPT:
                        interrupt_event = ev
                    elif ev.type == SUMMARY:
                        got_summary = True

                if interrupt_event is None:
                    if not got_summary:
                        logger.warning("任务流未产出 summary 事件")
                    self._set_state(IDLE)
                    return

                # 等 UI 收齐每个 action_request 的决策
                self._set_state(AWAITING_DECISION, interrupt_event)
                decisions = self._decision_q.get()
                if decisions is None:
                    self._set_state(IDLE, "已取消本轮审批，输入 /resume 可从断点继续")
                    return
                if len(decisions) != len(interrupt_event.action_requests):
                    self._set_state(
                        IDLE,
                        f"决策数量不匹配（需 {len(interrupt_event.action_requests)} 个），"
                        "输入 /resume 可重新审批",
                    )
                    return

                # propose_plan 批准后退出 Plan 模式（同 cli.run_task）
                if plan_mode and any(
                    ar.get("name") == "propose_plan"
                    for ar in interrupt_event.action_requests
                ):
                    if any(d.get("type") == "approve" for d in decisions):
                        config.set_plan_mode(False)
                        plan_mode = False

                next_user_input = None
                next_decisions = decisions
        except Exception as e:  # noqa: BLE001 — 回到 idle 并把错误显示出来
            logger.exception("TUI 任务执行失败")
            self._set_state(IDLE, f"任务执行出错：{e}")
