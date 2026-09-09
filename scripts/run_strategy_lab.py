from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from jkquant.config import load_config
from jkquant.pipeline import run_strategy_lab


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行最佳策略参数消融试验场")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--start", type=parse_date)
    parser.add_argument("--end", type=parse_date)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    progress_path, results = run_strategy_lab(load_config(args.config), args.start, args.end)
    print(f"策略试验完成：{len(results)} 组；进度文件：{progress_path}")
    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
