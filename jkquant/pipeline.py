from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .backtest.engine import run_backtest as execute_backtest
from .backtest.reporting import write_backtest_report
from .config import resolve_path
from .data.demo_provider import DemoProvider
from .data.storage import ParquetStore
from .data.tushare_provider import TushareProvider
from .data.updater import update_data
from .factors import calculate_factors
from .report import write_csv
from .strategy import select_stocks

LOGGER = logging.getLogger(__name__)


def build_provider(config: dict[str, Any]):
    data = config["data"]
    if data["provider"] == "demo":
        return DemoProvider(int(data["demo_stock_count"]), int(data["demo_seed"]))
    if data["provider"] == "tushare":
        load_dotenv(Path(config["_config_dir"]) / ".env")
        return TushareProvider(
            token_env=data.get("token_env", "TUSHARE_TOKEN"),
            requests_per_minute=int(data.get("requests_per_minute", 45)),
            max_retries=int(data.get("max_retries", 5)),
            retry_backoff_seconds=float(data.get("retry_backoff_seconds", 5)),
            request_timeout_seconds=int(data.get("request_timeout_seconds", 15)),
        )
    raise ValueError(f"不支持的数据源: {data['provider']}")


def build_store(config: dict[str, Any]) -> ParquetStore:
    root = resolve_path(config, config["data"]["cache_dir"])
    return ParquetStore(root / config["data"]["provider"])


def run_update(
    config: dict[str, Any], end_date: date | None = None, start_date: date | None = None
) -> ParquetStore:
    store = build_store(config)
    update_data(
        build_provider(config), store, end_date or date.today(), int(config["data"]["history_days"]),
        start_date=start_date, basic_refresh_days=int(config["data"].get("basic_refresh_days", 7)),
    )
    return store


def run_daily(config: dict[str, Any], end_date: date | None = None) -> tuple[Path, dict[str, int | str]]:
    store = run_update(config, end_date)
    daily = store.load_daily()
    factors = calculate_factors(daily)
    selection, summary = select_stocks(factors, store.load_basic(), config)
    if selection.empty:
        raise RuntimeError("过滤后没有足够数据生成选股结果，请检查配置和数据完整性")
    path = write_csv(selection, resolve_path(config, config["report"]["output_dir"]))
    LOGGER.info("报告已生成: %s", path)
    return path, summary


def run_historical_backtest(
    config: dict[str, Any], start_date: date | None = None, end_date: date | None = None
) -> tuple[dict[str, Path], dict[str, Any]]:
    settings = config["backtest"]
    start = start_date or date.fromisoformat(settings["start_date"])
    configured_end = settings.get("end_date")
    end = end_date or (date.fromisoformat(configured_end) if configured_end else date.today())
    if start >= end:
        raise ValueError("回测开始日期必须早于结束日期")
    # Extra observations are required before start for rolling factor windows.
    store = run_update(config, end_date=end, start_date=start - timedelta(days=120))
    daily = store.load_daily()
    daily = daily[daily["trade_date"].dt.date <= end]
    result = execute_backtest(daily, store.load_basic(), config, start, end)
    folder = resolve_path(config, settings["output_dir"]) / f"{start}_{end}"
    paths = write_backtest_report(result, folder)
    LOGGER.info("回测完成: %s", folder)
    return paths, result.metrics
