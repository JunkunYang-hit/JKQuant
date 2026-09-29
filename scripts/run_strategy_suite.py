from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from jkquant.config import load_config
from jkquant.pipeline import run_strategy_suite


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="执行当前七套研究策略并生成横向比较结果")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--start", type=parse_date)
    parser.add_argument("--end", type=parse_date)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    path, summaries = run_strategy_suite(load_config(args.config), args.start, args.end)
    print(f"已完成 {len(summaries)} 个策略，汇总文件: {path}")


if __name__ == "__main__":
    main()
