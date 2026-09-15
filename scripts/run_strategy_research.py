from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from jkquant.config import load_config
from jkquant.strategy_research import run_strategy_research


def main() -> None:
    parser = argparse.ArgumentParser(description="训练期选定低换手分散策略，验证近一年/三个月/一个月")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    output = run_strategy_research(load_config(args.config), args.end)
    print(f"研究报告已生成：{output / 'research.json'}")


if __name__ == "__main__":
    main()
