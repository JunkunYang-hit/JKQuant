from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from jkquant.config import load_config
from jkquant.pipeline import run_recent_signal_ensemble


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="回测八策略等资金联合组合最近若干月表现")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--end", type=parse_date)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    path, metrics = run_recent_signal_ensemble(load_config(args.config), args.months, args.end)
    print(f"八策略联合近期回测完成：{path}")
    print(f"累计收益：{metrics['cumulative_return']:.2%}")
    print(f"最大回撤：{metrics['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
