from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

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
        return TushareProvider()
    raise ValueError(f"不支持的数据源: {data['provider']}")


def build_store(config: dict[str, Any]) -> ParquetStore:
    return ParquetStore(resolve_path(config, config["data"]["cache_dir"]))


def run_update(config: dict[str, Any], end_date: date | None = None) -> ParquetStore:
    store = build_store(config)
    update_data(build_provider(config), store, end_date or date.today(), int(config["data"]["history_days"]))
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
