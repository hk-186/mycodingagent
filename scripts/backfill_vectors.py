# -*- coding: utf-8 -*-
"""
一次性运维脚本：为历史记忆批量补语义向量
=========================================

背景：
    阶段 5 给 SqliteStore 配置了 sqlite-vec 语义索引。该索引只对「配置生效
    之后新写入/重新保存」的记忆自动 embed；在此之前已存在于 store 主表里的
    历史记忆没有向量，因而无法被自然语言语义查询召回（精确 get 不受影响）。

做法：
    扫描 store 主表全部 (prefix, key)，与 store_vectors 中已有向量的
    (prefix, key) 做差集；对每条缺失记录，读出原 value，通过 SqliteStore
    官方 put 重新写入——复用框架的序列化、外键与 embed 全流程，自动生成向量。
    不手工拼接向量 blob，避免格式/维度不一致。

注意：
    put 是 INSERT OR REPLACE，会刷新该条记录的 created_at / updated_at
    （value 内容不变）。这是一次性补数据的可接受代价。

用法（在项目根目录，已 pip install -e . 且 .env 配好 AGICTO_API_KEY）：
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

from mycodingagent import config


def find_missing_pairs(db_path: str) -> tuple[int, list[tuple[str, str, str]]]:
    """返回 (主表总记录数, 缺向量的 [(prefix, key, value_json), ...])。

    用独立短连接只读扫描，调用方负责在打开可写 SqliteStore 前关闭它，
    避免 SQLite 写锁冲突。
    """
    conn = sqlite3.connect(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM store").fetchone()[0]
        rows = conn.execute("SELECT prefix, key, value FROM store").fetchall()
        have = {
            (prefix, key)
            for prefix, key in conn.execute(
                "SELECT DISTINCT prefix, key FROM store_vectors"
            )
        }
    finally:
        conn.close()
    missing = [(p, k, v) for (p, k, v) in rows if (p, k) not in have]
    return total, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为历史记忆批量补语义向量（sqlite-vec）。"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要补的记忆，不实际写入。",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过交互确认直接执行。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not config.API_KEY:
        print("未检测到 AGICTO_API_KEY，请先在 .env 配置（模板见 .env.example）。")
        return 1

    total, missing = find_missing_pairs(config.STORE_DB)
    print(f"记忆主表共 {total} 条；其中缺向量 {len(missing)} 条。")
    for prefix, key, _ in missing:
        print(f"  - namespace={tuple(prefix.split('.'))}  key={key}")

    if not missing:
        print("没有需要补向量的记忆，结束。")
        return 0

    if args.dry_run:
        print("\n[dry-run] 未写入任何数据。去掉 --dry-run 以实际执行。")
        return 0

    if not args.yes:
        try:
            answer = input("\n确认对以上记忆重新写入并生成向量？[y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in {"y", "yes"}:
            print("已取消，未做任何改动。")
            return 0

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
            "text_fields": ["value"],
        },
    )

    ok = 0
    failed = 0
    # 与 cli.py 相同的方式打开 store（自动注册 sqlite-vec 扩展）
    with store_cm as store:
        for prefix, key, raw_value in missing:
            namespace = tuple(prefix.split("."))
            try:
                value = json.loads(raw_value)
                store.put(namespace, key, value)
            except Exception as exc:  # noqa: BLE001, PERF203
                failed += 1
                print(f"  [失败] {namespace} / {key}：{exc}", file=sys.stderr)
            else:
                ok += 1
                print(f"  [OK] {namespace} / {key}")

    print(f"\n完成：成功 {ok} 条，失败 {failed} 条。")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
