from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from jkquant.config import load_config
from jkquant.data.fundamental_updater import update_fundamentals
from jkquant.data.tushare_provider import TushareProvider
from jkquant.pipeline import build_store


def main() -> None:
    parser = argparse.ArgumentParser(description="断点续传 Tushare 财务三表")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--max-stocks", type=int, default=None, help="试跑时限制股票数量")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config("config.yaml")
    load_dotenv(Path(config["_config_dir"]) / ".env")
    data = config["data"]
    provider = TushareProvider(
        token_env=data.get("token_env", "TUSHARE_TOKEN"),
        requests_per_minute=int(data.get("financial_requests_per_minute", 75)),
        max_retries=int(data.get("max_retries", 5)),
    )
    result = update_fundamentals(
        provider, build_store(config), date.fromisoformat(args.start),
        date.fromisoformat(args.end), args.max_stocks,
    )
    print(result)


if __name__ == "__main__":
    main()
