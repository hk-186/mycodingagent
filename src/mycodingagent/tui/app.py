# -*- coding: utf-8 -*-
"""
TUI 主界面（Textual）
=====================
布局：Header / 事件日志（RichLog）/ 状态栏 / 输入栏。

- 普通输入：作为任务交给后台 TuiTaskRunner 执行
- 审批中断：输入栏切到「决策」语义，逐条收集 /approve /reject /respond
  （复用 cli._parse_decision_input），收齐后一次性提交给 runner
- 支持斜杠命令：/help /pwd /cd /resume /plan /cancel /exit
  会话管理类命令（/sessions /switch /new 等）仍以 CLI 为准

runner 在后台线程，所有 UI 更新走 call_from_thread。
"""

from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Header, Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from mycodingagent import config
from mycodingagent.cli import _parse_decision_input, change_project_dir
from mycodingagent.events import ASSISTANT_MESSAGE, SUMMARY, AgentEvent, render_event
from mycodingagent.tui.runner import (
    AWAITING_DECISION,
    IDLE,
    RUNNING,
    TuiTaskRunner,
)

HELP_TEXT = """\
【斜杠命令】
  /help              显示本帮助
  /sessions          列出历史会话（弹窗选择，回车切换）
  /new [名字]        开启新会话（旧会话历史保留）
  /delete [id|序号]  删除指定会话（不带参数时弹窗选择，当前会话除外）
  /memory            查看长期记忆
  /history           查看当前会话消息统计
  /pwd               查看当前工作目录
  /cd <path>         切换工作目录（目录必须已存在）
  /resume            从断点继续当前会话中断的任务
  /plan <task>       进入 Plan 模式执行任务（先出计划等审批）
  /exit              退出

【审批决策】（等待决策时直接输入）
  /approve           审批通过（简写 a）
  /reject [<理由>]   拒绝（简写 r <理由>）
  /respond <内容>    代答 ask_user（简写 rr <内容>，或直接输入回答）
  /cancel            取消当前审批

【快捷键】
  t                  展开/收起模型的思考过程
  Ctrl+D             取消审批（等同 /cancel）
  Ctrl+C             退出
"""


class StatusBar(Static):
    """底部状态栏：模型 / 目录 / 审批 / 步数 token。"""


class SessionSelectScreen(ModalScreen[str | None]):
    """会话选择弹窗：键盘上下选、回车确认、Esc 取消。title 区分用途（切换/删除）。"""

    BINDINGS = [
        Binding("escape", "cancel", "取消", priority=True),
    ]

    CSS = """
    SessionSelectScreen {
        align: center middle;
    }
    #session-dialog {
        width: 90%;
        max-width: 90;
        height: 80%;
        max-height: 40;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    #session-title {
        height: 2;
        text-style: bold;
        content-align: center middle;
    }
    #session-list {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    """

    def __init__(self, sessions: list, title: str = "") -> None:
        super().__init__()
        self._sessions = sessions
        self._title = title or "选择会话（↑/↓ 移动，Enter 确认，Esc 取消）"

    def compose(self) -> ComposeResult:
        with Vertical(id="session-dialog"):
            yield Static(self._title, id="session-title")
            yield OptionList(
                *[
                    Option(
                        f"{idx + 1}. {s.status}  {s.thread_id}  "
                        f"({s.msg_count} 条, {s.step} 步)",
                        id=s.thread_id,
                    )
                    for idx, s in enumerate(self._sessions)
                ],
                id="session-list",
            )

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AgentApp(App):
    """MyCodingAgent 终端界面。"""

    TITLE = "MyCodingAgent"
    CSS = """
    Screen {
        layout: vertical;
    }
    #log {
        height: 4fr;
        min-height: 10;
        border: round $primary;
        padding: 0 1;
    }
    StatusBar {
        background: $boost;
        color: $text;
        padding: 0 1;
        height: 1;
    }
    #input-row {
        height: 3;
    }
    #prompt-label {
        width: auto;
        padding: 0 1;
    }
    #input {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "退出", priority=True),
        Binding("ctrl+d", "cancel_decision", "取消审批", show=False),
        Binding("t", "toggle_reasoning", "展开/收起思考", show=False),
    ]

    def __init__(
        self,
        agent: Any,
        backend: Any,
        *,
        thread_id: str = "main",
        store: Any = None,
        checkpointer: Any = None,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._backend = backend
        self._store = store
        self._checkpointer = checkpointer
        self._session_id = thread_id
        self._mode = IDLE
        self._decisions: list[dict] = []
        self._need_decisions = 0
        self._last_steps = 0
        self._last_tokens = 0
        # 日志项缓存：("event", AgentEvent) 或 ("line", 手写文本)。
        # 重放用于"展开/收起思考"后重渲染（RichLog 不支持删除单行）。
        self._log_items: list[tuple[str, Any]] = []
        self._show_reasoning = False
        self.runner = TuiTaskRunner(
            agent,
            on_event=self._post_event,
            on_state=self._post_state,
        )

    # ------------------------------------------------------------
    # 构图
    # ------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="log", highlight=False, markup=False, wrap=True)
        yield StatusBar(id="status")
        with Horizontal(id="input-row"):
            yield Static("你>", id="prompt-label")
            yield Input(id="input", placeholder="输入任务或 /help 查看命令")

    def on_mount(self) -> None:
        self._refresh_status()
        self._write(
            f"模型: {config.MODEL_NAME} | 审批: {config.APPROVAL_MODE} | "
            f"Plan: {'开' if config.PLAN_MODE else '关'}\n"
            "输入 /help 查看命令。"
        )
        self.query_one("#input", Input).focus()

    # ------------------------------------------------------------
    # 后台线程回调（由 runner 线程触发，转入 UI 线程执行）
    # ------------------------------------------------------------
    def _post_event(self, ev: AgentEvent) -> None:
        self.call_from_thread(self._handle_event, ev)

    def _post_state(self, state: str, info: Any = None) -> None:
        self.call_from_thread(self._handle_state, state, info)

    def _render_item(self, item: tuple[str, Any]) -> str:
        """日志项 → 显示文本。reasoning 事件在折叠态显示占位行。"""
        kind, payload = item
        if kind == "line":
            return str(payload)
        ev = payload
        if ev.type == ASSISTANT_MESSAGE and ev.name == "reasoning" and not self._show_reasoning:
            chars = len(ev.text)
            return f"思考> [已折叠 {chars} 字，按 t 展开]"
        return render_event(ev)

    def _handle_event(self, ev: AgentEvent) -> None:
        item = ("event", ev)
        self._log_items.append(item)
        self.query_one("#log", RichLog).write(self._render_item(item))
        if ev.type == SUMMARY:
            self._last_steps = ev.steps
            self._last_tokens = ev.tokens
            self._refresh_status()

    def action_toggle_reasoning(self) -> None:
        """t 键：展开/收起思考过程，重放日志重渲染。"""
        self._show_reasoning = not self._show_reasoning
        log = self.query_one("#log", RichLog)
        log.clear()
        for item in self._log_items:
            log.write(self._render_item(item))

    def _handle_state(self, state: str, info: Any = None) -> None:
        self._mode = state
        if state == AWAITING_DECISION and isinstance(info, AgentEvent):
            self._need_decisions = len(info.action_requests)
            self._decisions = []
            self._write(f"等待决策：共需 {self._need_decisions} 个（Ctrl+D 或 /cancel 取消）")
        elif state == IDLE and isinstance(info, str) and info:
            self._write(f"[{info}]")
        if state == IDLE and config.PLAN_MODE:
            # 兜底：任务结束（含取消/异常）时关闭 Plan 模式（同 CLI）
            config.set_plan_mode(False)
        self._refresh_prompt()
        self._refresh_status()

    # ------------------------------------------------------------
    # 输入处理
    # ------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.query_one("#input", Input).value = ""
        if not text:
            return

        if self._mode == AWAITING_DECISION:
            self._collect_decision(text)
            return

        if text.startswith("/"):
            self._handle_slash(text)
            return

        if self._mode == RUNNING:
            self._write("[任务执行中，请等待结束；若在等待决策会出现提示]")
            return

        self._start_task(user_input=text)

    def _collect_decision(self, text: str) -> None:
        if text in {"/cancel", "cancel"}:
            self.runner.submit_decisions(None)
            return
        decision = _parse_decision_input(text)
        if decision is None:
            self._write("[无效决策，可用：/approve | /reject <理由> | /respond <内容>]")
            return
        self._decisions.append(decision)
        got = len(self._decisions)
        if got < self._need_decisions:
            self._write(f"[{got}/{self._need_decisions}] 请继续输入下一条决策")
            return
        self.runner.submit_decisions(self._decisions)

    def _get_state(self, tid: str | None = None) -> Any:
        return self._agent.get_state(
            {"configurable": {"thread_id": tid or self._session_id}}
        )

    def _handle_slash(self, text: str) -> None:
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower()
        # 会话切换类命令只允许空闲时执行，避免与进行中的任务串台
        session_cmds = {
            "/sessions", "/new", "/delete", "/memory", "/history",
        }
        if cmd in session_cmds and self._mode != IDLE:
            self._write("任务执行中，会话管理命令请等任务结束后再用")
            return
        if cmd == "/exit":
            self.exit()
        elif cmd == "/help":
            self._write(HELP_TEXT)
        elif cmd == "/sessions":
            self._list_sessions()
        elif cmd == "/new":
            self._new_session(arg.strip())
        elif cmd == "/delete":
            self._delete_session(arg.strip())
        elif cmd == "/memory":
            self._show_memory()
        elif cmd == "/history":
            self._show_history()
        elif cmd == "/pwd":
            self._write(f"当前工作目录: {config.PROJECT_DIR}")
        elif cmd == "/cd":
            target = arg.strip()
            if not target:
                self._write(f"当前工作目录: {config.PROJECT_DIR}")
                return
            try:
                change_project_dir(self._backend, target)
            except NotADirectoryError as e:
                self._write(f"切换失败：{e}")
            else:
                self._write(f"工作目录已切换为：{config.PROJECT_DIR}")
                self._refresh_status()
        elif cmd == "/resume":
            state = self._get_state()
            if not state.next:
                self._write("当前会话没有可恢复的中断任务")
            else:
                self._start_task(user_input=None, resume=True)
        elif cmd == "/plan":
            task = arg.strip()
            if not task:
                self._write("/plan 需要任务说明，例如：/plan 给 utils.py 加 is_palindrome")
                return
            self._start_task(user_input=task, plan_mode=True)
        elif cmd == "/cancel":
            self._write("当前没有等待决策的审批")
        else:
            self._write(f"未知命令 {cmd}，输入 /help 查看可用命令")

    # ------------------------------------------------------------
    # 会话管理（文本渲染；弹窗选择后续增强）
    # ------------------------------------------------------------
    def _all_sessions(self) -> list:
        from mycodingagent.sessions import list_sessions

        if self._checkpointer is None:
            return []
        return list_sessions(self._checkpointer.conn, lambda tid: self._get_state(tid))

    def _list_sessions(self) -> None:
        sessions = self._all_sessions()
        if not sessions:
            self._write("没有任何历史会话")
            return
        self.push_screen(SessionSelectScreen(sessions), self._on_session_picked)

    def _on_session_picked(self, tid: str | None) -> None:
        if tid is not None:
            self._apply_session_switch(tid)

    def _apply_session_switch(self, new_tid: str) -> None:
        self._session_id = new_tid
        state = self._get_state()
        n = (
            len(state.values.get("messages", []))
            if state is not None and state.values
            else 0
        )
        status = "中断" if state is not None and state.next else "等待输入"
        self._last_steps = 0
        self._last_tokens = 0
        self._write(f"已切换到会话: {new_tid}  ({n} 条消息, {status})")
        self._refresh_status()

    def _new_session(self, name: str) -> None:
        from datetime import datetime

        self._session_id = name or f"session-{datetime.now():%Y%m%d-%H%M%S}"
        self._last_steps = 0
        self._last_tokens = 0
        self._write(f"已切换到新会话: {self._session_id}")
        self._refresh_status()

    def _delete_session(self, arg: str) -> None:
        from mycodingagent.sessions import delete_session, resolve_switch_arg

        if self._checkpointer is None:
            self._write("会话存储不可用，无法删除")
            return
        sessions = self._all_sessions()
        # 当前会话不可删除，从候选中排除
        candidates = [s for s in sessions if s.thread_id != self._session_id]
        if not arg:
            if not candidates:
                self._write("没有其他会话可删除")
                return
            self.push_screen(
                SessionSelectScreen(
                    candidates,
                    title="选择要删除的会话（↑/↓ 移动，Enter 删除，Esc 取消）",
                ),
                self._on_delete_picked,
            )
            return
        tid = resolve_switch_arg(arg, candidates)
        if tid is None:
            if resolve_switch_arg(arg, sessions) == self._session_id:
                self._write("不能删除当前会话，请先 /new 或切换到其他会话")
            else:
                self._write(f"找不到会话: {arg}，输入 /sessions 查看可用会话")
            return
        self._confirm_delete(tid)

    def _on_delete_picked(self, tid: str | None) -> None:
        if tid is not None:
            self._confirm_delete(tid)

    def _confirm_delete(self, tid: str) -> None:
        from mycodingagent.sessions import delete_session

        if delete_session(self._checkpointer.conn, tid):
            self._write(f"已删除会话: {tid}")
        else:
            self._write(f"会话无历史记录或已删除: {tid}")

    def _show_memory(self) -> None:
        if self._store is None:
            self._write("长期记忆不可用")
            return
        items = self._store.search(("users",))
        if not items:
            self._write("长期记忆为空")
            return
        self._write("长期记忆：")
        for item in items:
            self._write(f"  {item.key} = {item.value['value']}")

    def _show_history(self) -> None:
        state = self._get_state()
        n = (
            len(state.values.get("messages", []))
            if state is not None and state.values
            else 0
        )
        self._write(f"会话 {self._session_id} 共 {n} 条消息")

    # ------------------------------------------------------------
    def _start_task(
        self,
        *,
        user_input: str | None,
        resume: bool = False,
        plan_mode: bool = False,
    ) -> None:
        # 用户输入不属于 agent 事件流，需手动回显到日志区（/resume 无输入，跳过）
        if user_input:
            prefix = "计划> " if plan_mode else "你> "
            self._write(f"{prefix}{user_input}")
        if plan_mode:
            config.set_plan_mode(True)
        self.runner.start(
            thread_id=self._session_id,
            user_input=user_input,
            resume=resume,
            plan_mode=plan_mode,
        )

    def action_cancel_decision(self) -> None:
        if self._mode == AWAITING_DECISION:
            self.runner.submit_decisions(None)

    # ------------------------------------------------------------
    # UI 刷新
    # ------------------------------------------------------------
    def _refresh_prompt(self) -> None:
        label = self.query_one("#prompt-label", Static)
        inp = self.query_one("#input", Input)
        if self._mode == AWAITING_DECISION:
            label.update("决策>")
            inp.placeholder = "/approve | /reject <理由> | /respond <内容> | /cancel"
        elif self._mode == RUNNING:
            label.update("执行>")
            inp.placeholder = "任务执行中…"
        else:
            label.update("你>")
            inp.placeholder = "输入任务或 /help 查看命令"

    def _refresh_status(self) -> None:
        mode_text = {
            IDLE: "空闲",
            RUNNING: "执行中",
            AWAITING_DECISION: "等待决策",
        }.get(self._mode, self._mode)
        self.query_one("#status", Static).update(
            f"{config.MODEL_NAME} | 会话:{self._session_id} | {config.PROJECT_DIR} "
            f"| 审批:{config.APPROVAL_MODE} "
            f"| {mode_text} | {self._last_steps}步/{self._last_tokens}tok"
        )

    def _write(self, text: str) -> None:
        item = ("line", text)
        self._log_items.append(item)
        self.query_one("#log", RichLog).write(text)
