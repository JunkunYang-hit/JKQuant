from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from jkquant.config import load_config
from jkquant.pipeline import (
    RECOMMENDATION_HISTORY_START, build_store, run_streak2_leader_backtest,
)


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="回测连续2次Top20领跑者全仓策略的完整区间和最近三个月",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--end", type=parse_date, help="结束日期，默认使用本地最新交易日")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    daily = build_store(config).load_daily()
    if daily.empty:
        raise RuntimeError("本地没有日线数据，请先运行每日更新")
    local_end = daily["trade_date"].max().date()
    end = min(args.end or local_end, local_end)
    recent_start = (pd.Timestamp(end) - pd.DateOffset(months=3)).date()
    runs = [
        ("完整历史", RECOMMENDATION_HISTORY_START),
        ("最近三个月", recent_start),
    ]
    for label, start in runs:
        output, metrics = run_streak2_leader_backtest(config, start, end)
        print(
            f"{label}: {start} 至 {end} | 累计收益 {metrics['cumulative_return']:.2%} | "
            f"最大回撤 {metrics['max_drawdown']:.2%} | 完整交易 {metrics['total_trade_count']} | "
            f"结果 {output}"
        )


if __name__ == "__main__":
    main()
