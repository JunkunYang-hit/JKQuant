from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from jkquant.config import load_config
from jkquant.pipeline import run_daily
from scripts.run_webui import main as run_webui


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_top_k(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 50:
        raise argparse.ArgumentTypeError("--top-k 必须在 1 到 50 之间")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="更新最新行情、生成当日 Top-K，并启动本地 WebUI"
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--top-k", type=parse_top_k, help="推荐数量（1-50）")
    parser.add_argument("--end", type=parse_date, help="数据截止日 YYYY-MM-DD；默认今天")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    top_k = args.top_k or int(config["strategy"]["top_k"])
    config["strategy"]["top_k"] = top_k
    os.environ["JKQUANT_TOP_K"] = str(top_k)

    report, summary = run_daily(config, args.end)
    print(f"数据日期: {summary['trade_date']}")
    print(f"推荐数量: Top-{top_k}")
    print(f"推荐文件: {report}")
    print("正在启动 WebUI；在本终端按 Ctrl+C 可关闭。")
    run_webui()


if __name__ == "__main__":
    main()
