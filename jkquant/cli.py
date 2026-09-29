from __future__ import annotations

import argparse
import logging
from datetime import date

from .config import load_config
from .pipeline import run_daily, run_update


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _top_k(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 50:
        raise argparse.ArgumentTypeError("Top-K 必须在 1 到 50 之间")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="A股日频多因子选股系统")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    subparsers = parser.add_subparsers(dest="command", required=True)
    update = subparsers.add_parser("update", help="更新本地数据")
    update.add_argument("--config", default=argparse.SUPPRESS, help="配置文件路径")
    update.add_argument("--end", type=_date)
    daily = subparsers.add_parser("daily", help="更新数据并输出 Top-K")
    daily.add_argument("--config", default=argparse.SUPPRESS, help="配置文件路径")
    daily.add_argument("--end", type=_date)
    daily.add_argument("--top-k", type=_top_k, help="临时覆盖推荐数量（1-50）")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    if args.command == "update":
        store = run_update(config, args.end)
        print(f"数据目录: {store.root}")
        return
    if args.top_k is not None:
        config["strategy"]["top_k"] = args.top_k
    path, summary = run_daily(config, args.end)
    print(f"数据日期: {summary['trade_date']}")
    print(f"原始股票数: {summary['universe_count']}")
    print(f"有效股票数: {summary['eligible_count']}")
    print(f"Top-K 文件: {path}")
    print("数据异常: 无阻断性异常")


if __name__ == "__main__":
    main()
