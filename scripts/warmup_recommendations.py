from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from jkquant.config import load_config
from jkquant.pipeline import RECOMMENDATION_HISTORY_START, warm_recommendation_cache


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量预计算并缓存每日 Top-50 推荐")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--start", type=parse_date, default=RECOMMENDATION_HISTORY_START)
    parser.add_argument("--end", type=parse_date, help="默认截至本地最新交易日")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    summary = warm_recommendation_cache(load_config(args.config), args.start, args.end)
    print(
        "Top-50 预热完成："
        f"{summary['start_date']} 至 {summary['end_date']}，"
        f"共 {summary['trading_days']} 个交易日，"
        f"新增 {summary['newly_cached_days']} 日，"
        f"已存在 {summary['already_cached_days']} 日。"
    )


if __name__ == "__main__":
    main()
