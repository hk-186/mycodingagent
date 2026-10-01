# -*- coding: utf-8 -*-
"""
eval 判定器（grader）
=====================
每个 grader 接收沙箱上下文，返回 GraderResult(passed, detail)。

支持的类型（task.json 的 graders 数组元素）：
    {"type": "command", "run": "...", "expect_exit_code": 0, "timeout_seconds": 120}
        在沙箱根目录 subprocess.run(shell=True) 执行，比对 exit code。
    {"type": "file_exists", "path": "相对沙箱根的路径"}
    {"type": "file_contains", "path": "...", "pattern": "...", "regex": false}
    {"type": "file_not_contains", "path": "...", "pattern": "...", "regex": false}
    {"type": "llm_review", "criteria": "...", "pass_score": 3}
        把任务 prompt + agent 最终汇报 + 沙箱 git diff 喂评审模型，
        要求输出 {"score": 1-5, "comment": "..."} JSON，score >= pass_score 算过。
        只在所有确定性 grader 通过后才执行（由 run_graders 保证顺序短路）。

约定：确定性 grader 是硬门槛；llm_review 只做质量加分项，防止评审模型
宽松把「没修对」判成 pass。
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mycodingagent import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraderResult:
    """单个 grader 的判定结果。"""

    type: str
    passed: bool
    detail: str = ""
    score: int = 0  # 仅 llm_review 使用（1-5），其余为 0


@dataclass
class GradeContext:
    """grader 执行上下文：沙箱目录 + agent 运行产物。"""

    sandbox: Path          # 沙箱根目录（grader 执行期间仍存活）
    diff: str = ""         # agent 改动相对初始 fixture 的 git diff
    final_report: str = "" # agent 最后一条 assistant 消息（任务汇报）
    task_prompt: str = ""  # 任务原始 prompt（供 llm_review 参考）
    extra: dict[str, Any] = field(default_factory=dict)


class GraderError(ValueError):
    """grader 配置不合法。"""


# ============================================================
# 确定性 grader
# ============================================================
def _grade_command(spec: dict[str, Any], ctx: GradeContext) -> GraderResult:
    run = spec.get("run")
    if not isinstance(run, str) or not run.strip():
        raise GraderError("command grader 缺少非空 run 字段")
    expect = int(spec.get("expect_exit_code", 0))
    timeout = int(spec.get("timeout_seconds", config.EVAL_COMMAND_TIMEOUT))
    try:
        proc = subprocess.run(
            run,
            shell=True,
            cwd=str(ctx.sandbox),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return GraderResult(
            type="command",
            passed=False,
            detail=f"命令超时（>{timeout}s）：{run}",
        )
    passed = proc.returncode == expect
    tail = (proc.stdout + proc.stderr).strip()[-500:]
    return GraderResult(
        type="command",
        passed=passed,
        detail=(
            f"exit={proc.returncode}（期望 {expect}）\n输出尾部：\n{tail}"
            if not passed
            else f"exit={proc.returncode}"
        ),
    )


def _resolve_sandbox_path(ctx: GradeContext, rel: str) -> Path:
    """把 grader 里的相对路径解析到沙箱内，拒绝逃逸沙箱的路径。"""
    p = (ctx.sandbox / rel).resolve()
    if not str(p).startswith(str(ctx.sandbox.resolve())):
        raise GraderError(f"路径逃逸沙箱：{rel!r}")
    return p


def _grade_file_exists(spec: dict[str, Any], ctx: GradeContext) -> GraderResult:
    rel = spec.get("path")
    if not isinstance(rel, str) or not rel.strip():
        raise GraderError("file_exists grader 缺少非空 path 字段")
    exists = _resolve_sandbox_path(ctx, rel).is_file()
    return GraderResult(
        type="file_exists",
        passed=exists,
        detail="" if exists else f"文件不存在：{rel}",
    )


def _match_file_content(spec: dict[str, Any], ctx: GradeContext) -> tuple[bool, str]:
    """返回 (是否命中, 错误详情)。文件不存在视为未命中。"""
    rel = spec.get("path")
    pattern = spec.get("pattern")
    if not isinstance(rel, str) or not rel.strip():
        raise GraderError(f"{spec.get('type')} grader 缺少非空 path 字段")
    if not isinstance(pattern, str) or not pattern:
        raise GraderError(f"{spec.get('type')} grader 缺少非空 pattern 字段")
    p = _resolve_sandbox_path(ctx, rel)
    if not p.is_file():
        return False, f"文件不存在：{rel}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if spec.get("regex"):
        try:
            hit = re.search(pattern, text) is not None
        except re.error as e:
            raise GraderError(f"非法正则 {pattern!r}：{e}") from e
    else:
        hit = pattern in text
    return hit, ""


def _grade_file_contains(spec: dict[str, Any], ctx: GradeContext) -> GraderResult:
    hit, err = _match_file_content(spec, ctx)
    return GraderResult(
        type="file_contains",
        passed=hit,
        detail="" if hit else (err or f"未命中：{spec.get('path')} 不含 {spec.get('pattern')!r}"),
    )


def _grade_file_not_contains(spec: dict[str, Any], ctx: GradeContext) -> GraderResult:
    hit, err = _match_file_content(spec, ctx)
    if err:
        # 文件不存在时 not_contains 视为通过（无可禁内容）
        return GraderResult(type="file_not_contains", passed=True, detail=f"（{err}，视为通过）")
    return GraderResult(
        type="file_not_contains",
        passed=not hit,
        detail="" if not hit else f"命中禁用内容：{spec.get('path')} 含 {spec.get('pattern')!r}",
    )


# ============================================================
# LLM 评审 grader
# ============================================================
_REVIEW_PROMPT = """你是一位严格的代码评审。请根据以下信息对 agent 完成的任务改动打分。

【任务目标】
{task_prompt}

【评审标准】
{criteria}

【agent 的最终汇报】
{final_report}

【agent 的代码改动（git diff）】
{diff}

请只输出一个 JSON 对象（不要输出其他内容）：
{{"score": <1-5 的整数>, "comment": "<中文简评，50 字以内>"}}
打分参考：5=完全达标且改动干净；4=达标但有小瑕疵；3=基本达标；2=部分达标；1=未达标。"""


def _grade_llm_review(spec: dict[str, Any], ctx: GradeContext) -> GraderResult:
    criteria = spec.get("criteria")
    if not isinstance(criteria, str) or not criteria.strip():
        raise GraderError("llm_review grader 缺少非空 criteria 字段")
    pass_score = int(spec.get("pass_score", config.EVAL_REVIEW_PASS_SCORE))

    # 评审模型：EVAL_REVIEW_MODEL 可覆盖，默认复用被评模型配置
    from langchain_openai import ChatOpenAI  # 局部导入：单测不触网

    model_name = config.EVAL_REVIEW_MODEL or config.MODEL_NAME
    llm = ChatOpenAI(
        model=model_name,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        temperature=0.0,  # 评审要稳定，强制低温
        timeout=config.LLM_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
    )
    prompt = _REVIEW_PROMPT.format(
        task_prompt=ctx.task_prompt,
        criteria=criteria,
        final_report=ctx.final_report or "（agent 未给出最终汇报）",
        diff=ctx.diff[:8000] if ctx.diff else "（无改动）",
    )
    try:
        resp = llm.invoke(prompt)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as e:  # noqa: BLE001 — 评审失败不等于任务失败，记 0 分不过
        logger.warning("llm_review 调用失败：%s", e)
        return GraderResult(
            type="llm_review", passed=False, score=0,
            detail=f"评审模型调用失败：{type(e).__name__}: {e}",
        )

    # 从回复中提取 JSON（模型可能包裹 ```json 围栏）
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        return GraderResult(
            type="llm_review", passed=False, score=0,
            detail=f"评审输出非 JSON：{raw[:200]}",
        )
    try:
        verdict = json.loads(m.group(0))
        score = int(verdict["score"])
        comment = str(verdict.get("comment", ""))
    except (ValueError, KeyError, TypeError) as e:
        return GraderResult(
            type="llm_review", passed=False, score=0,
            detail=f"评审 JSON 解析失败（{e}）：{raw[:200]}",
        )
    if not 1 <= score <= 5:
        return GraderResult(
            type="llm_review", passed=False, score=score,
            detail=f"评分越界（{score}，应 1-5）：{comment}",
        )
    return GraderResult(
        type="llm_review",
        passed=score >= pass_score,
        score=score,
        detail=f"{score}/5（及格 {pass_score}）：{comment}",
    )


# ============================================================
# 调度入口
# ============================================================
_GRADERS = {
    "command": _grade_command,
    "file_exists": _grade_file_exists,
    "file_contains": _grade_file_contains,
    "file_not_contains": _grade_file_not_contains,
    "llm_review": _grade_llm_review,
}


def run_graders(specs: list[dict[str, Any]], ctx: GradeContext) -> list[GraderResult]:
    """顺序执行全部 grader。

    确定性 grader 任一失败则短路（llm_review 是付费调用，硬门槛不过就不评）；
    llm_review 之间互不影响。返回与 specs 等长或更短（短路时）的结果列表。
    """
    results: list[GraderResult] = []
    for spec in specs:
        gtype = spec.get("type")
        fn = _GRADERS.get(gtype)
        if fn is None:
            raise GraderError(
                f"未知 grader 类型：{gtype!r}（支持：{sorted(_GRADERS)}）"
            )
        result = fn(spec, ctx)
        results.append(result)
        # 硬门槛短路：确定性 grader 失败，跳过后续（含付费的 llm_review）
        if not result.passed and gtype != "llm_review":
            break
    return results
