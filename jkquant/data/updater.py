from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from .provider import DataProvider
from .storage import ParquetStore

LOGGER = logging.getLogger(__name__)


def _download_range(
    provider: DataProvider, store: ParquetStore, start_date: date, end_date: date
) -> None:
    """Persist one month at a time so an interrupted initial load can resume."""
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=30), end_date)
        LOGGER.info("下载日线数据: %s 至 %s", chunk_start, chunk_end)
        store.save_daily(provider.daily(chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)


def _derive_basic_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Build minimal metadata when a 120-point account only has daily access."""
    first_seen = daily.groupby("ts_code", as_index=False)["trade_date"].min()
    first_seen = first_seen.rename(columns={"trade_date": "list_date"})
    first_seen["name"] = first_seen["ts_code"]
    first_seen["list_status"] = "L"
    first_seen["delist_date"] = pd.NaT
    first_seen["metadata_source"] = "daily_derived"
    return first_seen


def update_data(
    provider: DataProvider,
    store: ParquetStore,
    end_date: date,
    history_days: int,
    start_date: date | None = None,
    basic_refresh_days: int = 7,
    fetch_stock_basic: bool = True,
) -> None:
    if fetch_stock_basic:
        basic_is_fresh = store.basic_path.exists() and (
            date.today() - date.fromtimestamp(store.basic_path.stat().st_mtime)
        ).days < basic_refresh_days
        if basic_is_fresh:
            LOGGER.info("股票基础信息缓存仍在有效期内")
        else:
            LOGGER.info("更新股票基础信息")
            store.save_basic(provider.stock_basic())

    old = store.load_daily()
    if old.empty:
        requested_start = start_date or end_date - timedelta(days=max(history_days * 2, 365))
        _download_range(provider, store, requested_start, end_date)
    else:
        cached_start = old["trade_date"].min().date()
        cached_end = old["trade_date"].max().date()
        if start_date and start_date < cached_start:
            _download_range(provider, store, start_date, cached_start - timedelta(days=1))
        if cached_end < end_date:
            _download_range(provider, store, cached_end + timedelta(days=1), end_date)
        elif not start_date or start_date >= cached_start:
            LOGGER.info("本地日线数据已覆盖请求区间")

    if not fetch_stock_basic:
        LOGGER.info("120 积分模式：从日线缓存构造最小证券元数据")
        store.save_basic(_derive_basic_from_daily(store.load_daily()))
