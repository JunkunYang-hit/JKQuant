from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
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
    fetch_company_names: bool = False,
    name_refresh_days: int = 30,
    metadata_provider: DataProvider | None = None,
) -> None:
    if fetch_stock_basic:
        basic_is_fresh = store.basic_path.exists() and (
            date.today() - date.fromtimestamp(store.basic_path.stat().st_mtime)
        ).days < basic_refresh_days
        if basic_is_fresh:
            cached_basic = store.load_basic()
            basic_is_fresh = "industry" in cached_basic.columns and not (
                "metadata_source" in cached_basic.columns
                and cached_basic["metadata_source"].eq("daily_derived").any()
            )
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
        basic = _derive_basic_from_daily(store.load_daily())
        names_are_fresh = store.names_path.exists() and (
            date.today() - date.fromtimestamp(store.names_path.stat().st_mtime)
        ).days < name_refresh_days
        if fetch_company_names and not names_are_fresh:
            LOGGER.info("通过备用元数据源更新证券中文简称")
            if metadata_provider is None:
                raise RuntimeError("已启用证券名称更新，但没有配置元数据源")
            store.save_names(metadata_provider.company_names())
        names = store.load_names()
        if not names.empty:
            basic = basic.merge(names, on="ts_code", how="left", suffixes=("", "_company"))
            basic["name"] = basic["name_company"].fillna(basic["name"])
            basic = basic.drop(columns=["name_company"])
        store.save_basic(basic)


def update_enriched_data(
    provider: DataProvider, store: ParquetStore, start_date: date, end_date: date,
    benchmark_code: str = "510300.SH",
) -> dict[str, int]:
    """Incrementally persist the 2000-point low-frequency and macro datasets."""
    daily = store.load_daily()
    dates = sorted(
        value.date() for value in pd.to_datetime(daily["trade_date"].unique())
        if start_date <= value.date() <= end_date
    )
    counts: dict[str, int] = {}
    for endpoint in ("daily_basic", "stk_limit"):
        cached = store.load_market_dataset("limit" if endpoint == "stk_limit" else endpoint)
        cached_dates = set(cached["trade_date"].dt.date) if not cached.empty else set()
        missing = [value for value in dates if value not in cached_dates]
        frames = []
        def fetch(trade_date: date) -> pd.DataFrame:
            return provider.market_by_trade_date(endpoint, trade_date)  # type: ignore[attr-defined]

        with ThreadPoolExecutor(max_workers=6, thread_name_prefix=f"tushare-{endpoint}") as executor:
            fetched = executor.map(fetch, missing)
            for index, frame in enumerate(fetched, start=1):
                if not frame.empty:
                    frames.append(frame)
                if index % 20 == 0 or index == len(missing):
                    LOGGER.info("%s 更新进度: %d/%d", endpoint, index, len(missing))
        if frames:
            store.save_market_dataset(
                "limit" if endpoint == "stk_limit" else endpoint,
                pd.concat(frames, ignore_index=True),
            )
        counts[endpoint] = len(missing)

    period_dates = {
        "weekly": list(pd.Series(pd.to_datetime(dates)).groupby(
            pd.Series(pd.to_datetime(dates)).dt.to_period("W-FRI")
        ).max().dt.date),
        "monthly": list(pd.Series(pd.to_datetime(dates)).groupby(
            pd.Series(pd.to_datetime(dates)).dt.to_period("M")
        ).max().dt.date),
    }
    for endpoint, expected_dates in period_dates.items():
        cached = store.load_market_dataset(endpoint)
        cached_dates = set(cached["trade_date"].dt.date) if not cached.empty else set()
        missing = [value for value in expected_dates if value not in cached_dates]
        frames = []
        for trade_date in missing:
            frame = provider.market_by_trade_date(endpoint, trade_date)  # type: ignore[attr-defined]
            if not frame.empty:
                frames.append(frame)
        if frames:
            store.save_market_dataset(endpoint, pd.concat(frames, ignore_index=True))
        counts[endpoint] = len(missing)

    benchmark = provider.benchmark_daily(benchmark_code, start_date, end_date)  # type: ignore[attr-defined]
    store.save_market_dataset("benchmark", benchmark)
    counts["benchmark"] = len(benchmark)
    for endpoint in ("cn_gdp", "cn_cpi", "cn_ppi", "cn_m", "cn_pmi", "sf_month"):
        try:
            frame = provider.macro_dataset(endpoint)  # type: ignore[attr-defined]
            store.save_macro(endpoint, frame)
            counts[endpoint] = len(frame)
        except Exception as exc:
            LOGGER.warning("宏观数据 %s 更新失败，保留已有缓存: %s", endpoint, exc)
            counts[endpoint] = 0
    return counts
