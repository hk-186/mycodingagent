# -*- coding: utf-8 -*-
"""
调试脚本：检查 agicto 中转账户是否欠费
=====================================
用 max_tokens=1 的最小请求探测，避免产生明显费用。

用法（PowerShell）：
    python scripts/check_agicto_balance.py
    python scripts/check_agicto_balance.py --base-url https://api.agicto.cn/v1 --model deepseek-v4-flash

环境变量：
    AGICTO_API_KEY   必需
    LLM_BASE_URL     可选，默认 https://api.agicto.cn/v1
    LLM_MODEL_NAME   可选，默认 deepseek-v4-flash
"""

import argparse
import os
import sys

import requests

DEFAULT_BASE_URL = "https://api.agicto.cn/v1"
DEFAULT_MODEL = "deepseek-v4-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 agicto 账户是否欠费")
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        help=f"API 端点（默认 {DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL),
        help=f"模型名（默认 {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--key",
        default=os.getenv("AGICTO_API_KEY", ""),
        help="API Key（默认取 AGICTO_API_KEY 环境变量）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.key:
        print("[X] 未提供 API Key：请设置 AGICTO_API_KEY 环境变量，或用 --key 传入")
        return 2

    url = args.base_url.rstrip("/") + "/chat/completions"
    print(f"端点 : {url}")
    print(f"模型 : {args.model}")
    print(f"Key  : {args.key[:8]}...（长度 {len(args.key)}）")
    print("-" * 50)

    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {args.key}"},
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"[X] 网络请求失败：{type(e).__name__}: {e}")
        return 2

    print(f"HTTP 状态码: {resp.status_code}")
    print(f"原始响应: {resp.text[:500]}")
    print("-" * 50)

    body_lower = resp.text.lower()
    if resp.status_code == 200:
        print("[OK] 账户可用：请求成功，未欠费")
        return 0
    if "insufficient balance" in body_lower:
        print("[!] 账户欠费：服务商返回 Insufficient Balance，请充值后重试")
        return 1
    if resp.status_code == 401:
        print("[!] Key 无效（401）：请检查 AGICTO_API_KEY")
        return 1
    if resp.status_code == 429:
        print("[!] 触发限流（429）：稍后再试，不一定是欠费")
        return 1
    print(f"[?] 其他错误（HTTP {resp.status_code}），请看上面的原始响应自行判断")
    return 1


if __name__ == "__main__":
    sys.exit(main())
