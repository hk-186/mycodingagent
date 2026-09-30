# -*- coding: utf-8 -*-
"""
一次性运维脚本：升级记忆格式并补齐语义向量
==========================================

背景：
    早期版本的向量只编码 value 内容（text_fields=["value"]），store 的 key
    不参与嵌入。当 value 是 "Kevin"、邮箱这类短专有名词时，自然语言查询
    （如「用户的姓名 name」）与它语义距离很远，容易错位命中其它长文本。

本脚本做两件事：
    1. 把记录升级为新结构：
       {"value": <内容>, "index_text": "<key>：<内容>"}，
       并清除该记录残留的旧字段（field_name != "index_text"）向量；
    2. 通过 SqliteStore 官方 put 重写，复用框架的序列化、外键与 embed 流程，
       对 index_text（含 key）重新生成向量。

判定一条记录需要处理：value 中缺 index_text，或该 (prefix,key) 还没有
field_name="index_text" 的向量。幂等可重复执行。

用法（项目根目录，已 pip install -e . 且 .env 配好 AGICTO_API_KEY）：
    python scripts/backfill_vectors.py --dry-run   # 只预览，不写入
    python scripts/backfill_vectors.py             # 交互确认后执行
    python scripts/backfill_vectors.py --yes       # 跳过确认直接执行
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from langchain_openai import OpenAIEmbeddings
from langgraph.store.sqlite import SqliteStore

from mycodingagent import config, memory_common
from mycodingagent.memory_common import INDEX_TEXT_FIELD, VALUE_FIELD


def plan_upgrades(db_path: str) -> tuple[int, list[tuple[str, str, str]]]:
    """返回 (主表总记录数, 待处理 [(prefix, key, 真实内容), ...])。

    用独立短连接只读扫描；调用方在打开可写 SqliteStore 前关闭它，避免锁冲突。
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT s.prefix, s.key, s.value,
                   EXISTS (
                       SELECT 1 FROM store_vectors v
                       WHERE v.prefix = s.prefix AND v.key = s.key
                         AND v.field_name = ?
                   ) AS has_new_vector
            FROM store s
            """,
            (INDEX_TEXT_FIELD,),
        ).fetchall()
    finally:
        conn.close()

    pending: list[tuple[str, str, str]] = []
    for prefix, key, raw_value, has_new_vector in rows:
        try:
            record = json.loads(raw_value)
        except (ValueError, TypeError):
            print(
                f"  [跳过] {prefix}/{key}：value 不是合法 JSON 对象",
                file=sys.stderr,
            )
            continue
        if not isinstance(record, dict) or VALUE_FIELD not in record:
            print(
                f"  [跳过] {prefix}/{key}：记录缺少 {VALUE_FIELD!r} 字段",
                file=sys.stderr,
            )
            continue
        needs_upgrade = INDEX_TEXT_FIELD not in record
        if needs_upgrade or not has_new_vector:
            pending.append((prefix, key, str(record[VALUE_FIELD])))
    return len(rows), pending


def purge_legacy_vectors(db_path: str, pairs: list[tuple[str, str]]) -> None:
    """删除待处理记录残留的非 index_text 向量（用独立连接，避免与 store 写冲突）。"""
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "DELETE FROM store_vectors "
            "WHERE prefix = ? AND key = ? AND field_name <> ?",
            [(p, k, INDEX_TEXT_FIELD) for p, k in pairs],
        )
        conn.commit()
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="升级记忆格式（index_text 含 key）并补齐语义向量。"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印将要处理的记忆，不实际写入。"
    )
    parser.add_argument(
        "--yes", action="store_true", help="跳过交互确认直接执行。"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not config.API_KEY:
        print("未检测到 AGICTO_API_KEY，请先在 .env 配置（模板见 .env.example）。")
        return 1

    total, pending = plan_upgrades(config.STORE_DB)
    print(f"记忆主表共 {total} 条；待升级/补向量 {len(pending)} 条。")
    for prefix, key, _ in pending:
        print(f"  - namespace={tuple(prefix.split('.'))}  key={key}")

    if not pending:
        print("没有需要处理的记忆，结束。")
        return 0

    if args.dry_run:
        print("\n[dry-run] 未写入任何数据。去掉 --dry-run 以实际执行。")
        return 0

    if not args.yes:
        try:
            answer = input("\n确认升级以上记忆并重新生成向量？[y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in {"y", "yes"}:
            print("已取消，未做任何改动。")
            return 0

    # 先清旧向量（独立连接），再打开 store 重写
    purge_legacy_vectors(config.STORE_DB, [(p, k) for p, k, _ in pending])

    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
    )
    store_cm = SqliteStore.from_conn_string(
        config.STORE_DB,
        index={
            "dims": config.EMBEDDING_DIMS,
            "embed": embeddings,
            "text_fields": [INDEX_TEXT_FIELD],
        },
    )

    ok = 0
    failed = 0
    with store_cm as store:
        for prefix, key, content in pending:
            namespace = tuple(prefix.split("."))
            try:
                store.put(namespace, key, memory_common.make_record(key, content))
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  [失败] {namespace} / {key}：{exc}", file=sys.stderr)
            else:
                ok += 1
                print(f"  [OK] {namespace} / {key}")

    print(f"\n完成：成功 {ok} 条，失败 {failed} 条。")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
