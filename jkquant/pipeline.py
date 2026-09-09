from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from .backtest.engine import run_backtest as execute_backtest
from .backtest.reporting import write_backtest_report
from .backtest.strategy_suite import (
    BASE_STRATEGIES, STRATEGIES, run_event_strategy, write_strategy_result,
    write_suite_index,
)
from .config import resolve_path
from .data.akshare_provider import AkshareMetadataProvider
from .data.selection_cache import SelectionCache, strategy_key
from .data.demo_provider import DemoProvider
from .data.storage import ParquetStore
from .data.tushare_provider import TushareProvider
from .data.updater import update_data
from .factors import calculate_factors
from .report import write_csv
from .strategy import select_stocks

LOGGER = logging.getLogger(__name__)
CACHE_TOP_K = 50
RECOMMENDATION_HISTORY_START = date(2025, 9, 1)


def _top50_config(config: dict[str, Any]) -> dict[str, Any]:
    return {**config, "strategy": {**config["strategy"], "top_k": CACHE_TOP_K}}


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
    provider = build_provider(config)
    is_demo = config["data"]["provider"] == "demo"
    metadata_provider = None
    if not is_demo and config["data"].get("name_provider") == "akshare":
        metadata_provider = AkshareMetadataProvider()
    update_data(
        provider, store, end_date or date.today(), int(config["data"]["history_days"]),
        start_date=start_date, basic_refresh_days=int(config["data"].get("basic_refresh_days", 7)),
        fetch_stock_basic=is_demo or bool(config["data"].get("fetch_stock_basic", True)),
        fetch_company_names=(not is_demo) and bool(config["data"].get("fetch_company_names", False)),
        name_refresh_days=int(config["data"].get("name_refresh_days", 30)),
        metadata_provider=metadata_provider,
    )
    return store


def run_daily(config: dict[str, Any], end_date: date | None = None) -> tuple[Path, dict[str, int | str]]:
    store = run_update(config, end_date)
    daily = store.load_daily()
    factors = calculate_factors(daily)
    requested_top_k = int(config["strategy"]["top_k"])
    cache_config = _top50_config(config)
    top50, summary = select_stocks(
        factors, store.load_basic(), cache_config, top_k=CACHE_TOP_K,
        use_current_metadata=True,
    )
    if top50.empty:
        raise RuntimeError("过滤后没有足够数据生成选股结果，请检查配置和数据完整性")
    selection = top50.head(requested_top_k).copy()
    path = write_csv(selection, resolve_path(config, config["report"]["output_dir"]))
    SelectionCache(store.root / "selection_results.sqlite3").put(
        strategy_key(cache_config), factors["trade_date"].max().date(), top50
    )
    LOGGER.info("报告已生成: %s", path)
    return path, summary


def available_selection_dates(config: dict[str, Any]) -> list[date]:
    daily = build_store(config).load_daily()
    if daily.empty:
        return []
    earliest = daily["trade_date"].min() + pd.Timedelta(days=120)
    values = sorted(daily.loc[daily["trade_date"].ge(earliest), "trade_date"].unique())
    return [pd.Timestamp(value).date() for value in values]


def selection_for_date(
    config: dict[str, Any], selected_date: date
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a cached historical Top-K or calculate it from a bounded window."""
    started = time.perf_counter()
    store = build_store(config)
    cache = SelectionCache(store.root / "selection_results.sqlite3")
    requested_top_k = int(config["strategy"]["top_k"])
    cache_config = _top50_config(config)
    key = strategy_key(cache_config)
    cached = cache.get(key, selected_date)
    if cached is not None:
        return cached.head(requested_top_k).copy(), {
            "cached": True, "elapsed_seconds": time.perf_counter() - started,
        }
    daily = store.load_daily()
    target = pd.Timestamp(selected_date)
    if not daily["trade_date"].eq(target).any():
        raise ValueError(f"{selected_date} 不是本地缓存中的交易日")
    window = daily[daily["trade_date"].between(target - pd.Timedelta(days=200), target)]
    factors = calculate_factors(window)
    latest_date = daily["trade_date"].max().date()
    top50, summary = select_stocks(
        factors, store.load_basic(), cache_config, top_k=CACHE_TOP_K,
        use_current_metadata=selected_date == latest_date,
    )
    if top50.empty:
        raise RuntimeError(f"{selected_date} 没有足够数据生成候选")
    cache.put(key, selected_date, top50)
    summary.update({"cached": False, "elapsed_seconds": time.perf_counter() - started})
    return top50.head(requested_top_k).copy(), summary


def recommendation_history_stats(
    config: dict[str, Any], selected_date: date,
    start_date: date = RECOMMENDATION_HISTORY_START,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Summarize Top-20 streaks and Top-50 counts from persisted daily results."""
    store = build_store(config)
    cache = SelectionCache(store.root / "selection_results.sqlite3")
    daily = store.load_daily()
    if selected_date < start_date:
        return {}, {"cached_days": 0, "expected_days": 0}
    effective_start = start_date
    expected_dates = [
        pd.Timestamp(value).date() for value in sorted(daily["trade_date"].unique())
        if effective_start <= pd.Timestamp(value).date() <= selected_date
    ]
    key = strategy_key(_top50_config(config))
    history = cache.history(key, effective_start, selected_date)
    if history.empty:
        return {}, {"cached_days": 0, "expected_days": len(expected_dates)}

    total_counts = history.groupby("ts_code").size().astype(int).to_dict()
    by_date = {
        trade_date: day.set_index("ts_code")["rank"].astype(int).to_dict()
        for trade_date, day in history.groupby("trade_date")
    }
    current = by_date.get(selected_date, {})
    current_top20 = {code for code, rank in current.items() if rank <= 20}
    streaks = {code: 0 for code in current_top20}
    for code in current_top20:
        for trade_date in reversed(expected_dates):
            if by_date.get(trade_date, {}).get(code, CACHE_TOP_K + 1) <= 20:
                streaks[code] += 1
            else:
                break
    stats = {
        code: {
            "consecutive_top20": int(streaks.get(code, 0)),
            "top50_count": int(count),
        }
        for code, count in total_counts.items()
    }
    return stats, {
        "cached_days": len(by_date), "expected_days": len(expected_dates),
    }


def warm_recommendation_cache(
    config: dict[str, Any], start_date: date = RECOMMENDATION_HISTORY_START,
    end_date: date | None = None,
) -> dict[str, int | str]:
    """Precompute one canonical Top-50 result for every local trading day."""
    store = build_store(config)
    daily = store.load_daily()
    if daily.empty:
        raise RuntimeError("本地没有日线数据，请先运行每日更新")
    last_date = min(end_date or date.today(), daily["trade_date"].max().date())
    dates = [
        pd.Timestamp(value) for value in sorted(daily["trade_date"].unique())
        if start_date <= pd.Timestamp(value).date() <= last_date
    ]
    if not dates:
        raise ValueError("指定区间没有本地交易日")
    cache_config = _top50_config(config)
    key = strategy_key(cache_config)
    cache = SelectionCache(store.root / "selection_results.sqlite3")
    cached = cache.cached_dates(key, start_date, last_date)
    missing = [value for value in dates if value.date() not in cached]
    if missing:
        factor_start = missing[0] - pd.Timedelta(days=200)
        source = daily[daily["trade_date"].between(factor_start, dates[-1])]
        factors = calculate_factors(source)
        basic = store.load_basic()
        latest_date = daily["trade_date"].max().date()
        for index, trade_date in enumerate(missing, start=1):
            day = factors[factors["trade_date"].eq(trade_date)]
            top50, _ = select_stocks(
                day, basic, cache_config, top_k=CACHE_TOP_K,
                use_current_metadata=trade_date.date() == latest_date,
            )
            cache.put(key, trade_date.date(), top50)
            if index == 1 or index % 10 == 0 or index == len(missing):
                LOGGER.info("Top-50 预热进度: %d/%d (%s)", index, len(missing), trade_date.date())
    return {
        "start_date": start_date.isoformat(), "end_date": last_date.isoformat(),
        "trading_days": len(dates), "newly_cached_days": len(missing),
        "already_cached_days": len(dates) - len(missing),
    }


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


def run_strategy_suite(
    config: dict[str, Any], start_date: date | None = None, end_date: date | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    settings = config.get("strategy_suite", {})
    start = start_date or date.fromisoformat(settings.get("start_date", "2025-09-01"))
    configured_end = settings.get("end_date")
    requested_end = end_date or (
        date.fromisoformat(configured_end) if configured_end else date.today()
    )
    store = run_update(config, end_date=requested_end, start_date=start - timedelta(days=200))
    daily = store.load_daily()
    end = min(requested_end, daily["trade_date"].max().date())
    if start >= end:
        raise ValueError("多策略回测开始日期必须早于本地最新交易日")
    warm_recommendation_cache(config, start, end)
    cache = SelectionCache(store.root / "selection_results.sqlite3")
    rankings = cache.history(strategy_key(_top50_config(config)), start, end)
    expected_dates = {
        pd.Timestamp(value).date() for value in daily["trade_date"].unique()
        if start <= pd.Timestamp(value).date() <= end
    }
    cached_dates = set(rankings["trade_date"].unique())
    if cached_dates != expected_dates:
        missing = sorted(expected_dates - cached_dates)
        raise RuntimeError(f"Top-50 信号缓存不完整，缺少 {len(missing)} 个交易日")
    basic = store.load_basic()
    names = (
        basic.dropna(subset=["name"]).drop_duplicates("ts_code")
        .set_index("ts_code")["name"].astype(str).to_dict()
        if "name" in basic else {}
    )
    costs = config["backtest"]["cost"]
    initial_cash = float(config["backtest"]["initial_cash"])
    take_profit = float(settings.get("take_profit", 0.20))
    record_profit = float(settings.get("record_profit", 0.20))
    root = resolve_path(
        config, settings.get("output_dir", "backtests/strategy_suite")
    ) / f"{start}_{end}"
    root.mkdir(parents=True, exist_ok=True)
    st_codes: set[str] = set()
    if config.get("market", {}).get("exclude_st", True) and "name" in basic:
        st_mask = basic["name"].fillna("").astype(str).str.upper().str.contains("ST")
        st_codes = set(basic.loc[st_mask, "ts_code"].astype(str))
        daily = daily[~daily["ts_code"].isin(st_codes)].copy()
        rankings = rankings[~rankings["ts_code"].isin(st_codes)].copy()
        LOGGER.info("策略回测按当前证券简称近似排除 ST：%d 只", len(st_codes))
    summaries: list[dict[str, Any]] = []
    baseline = execute_backtest(daily, basic, config, start, end)
    baseline.metrics.update({
        "strategy_id": "baseline_top10_3d",
        "strategy_name": "基准：Top10线性权重，每3日调仓",
        "strategy_description": "固定持有Top10并按排名线性分配权重，每3个交易日重新选股调仓。",
        "threshold_enabled": False, "take_profit_count": 0,
        "crossed_20_count": 0, "completed_trades": int(len(baseline.trades)),
        "total_trade_count": None, "profitable_trade_count": None,
        "losing_trade_count": None, "flat_trade_count": None,
        "open_positions": int(baseline.daily["holdings"].iloc[-1]),
        "average_holding_days": 0.0, "profitable_trade_rate": None,
        "st_filter_mode": "current_name_approximation",
        "excluded_st_count": len(st_codes),
    })
    baseline_folder = root / "baseline_top10_3d"
    write_strategy_result(
        baseline_folder, baseline.daily, baseline.trades, pd.DataFrame(), baseline.metrics,
    )
    summaries.append({
        "strategy_id": "baseline_top10_3d", "name": baseline.metrics["strategy_name"],
        "description": baseline.metrics["strategy_description"],
        "folder": "baseline_top10_3d", "metrics": baseline.metrics,
    })
    LOGGER.info(
        "基准回测完成: %s | 累计收益 %.2f%%",
        baseline.metrics["strategy_name"], baseline.metrics["cumulative_return"] * 100,
    )
    for spec in STRATEGIES:
        result, trades, events, metrics = run_event_strategy(
            daily, rankings, names, spec, start, end, initial_cash, costs,
            take_profit=take_profit, record_profit=record_profit,
        )
        metrics["threshold_enabled"] = True
        metrics["st_filter_mode"] = "current_name_approximation"
        metrics["excluded_st_count"] = len(st_codes)
        folder = root / spec.strategy_id
        write_strategy_result(folder, result, trades, events, metrics)
        summaries.append({
            "strategy_id": spec.strategy_id, "name": spec.name,
            "description": spec.description, "folder": spec.strategy_id,
            "metrics": metrics,
        })
        LOGGER.info("策略回测完成: %s | 累计收益 %.2f%%", spec.name, metrics["cumulative_return"] * 100)
    sweep_rows: list[dict[str, Any]] = []
    raw_thresholds = settings.get("take_profit_sweep", [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, None])
    thresholds = [None if value is None else float(value) for value in raw_thresholds]
    for spec in BASE_STRATEGIES:
        for threshold in thresholds:
            _, sweep_trades, _, sweep_metrics = run_event_strategy(
                daily, rankings, names, spec, start, end, initial_cash, costs,
                take_profit=threshold, record_profit=threshold,
            )
            sweep_rows.append({
                "strategy_id": spec.strategy_id,
                "strategy_name": spec.name,
                "take_profit_threshold": threshold,
                "threshold_label": "不设止盈" if threshold is None else f"{threshold:.0%}",
                "cumulative_return": sweep_metrics["cumulative_return"],
                "annualized_return": sweep_metrics["annualized_return"],
                "sharpe_ratio": sweep_metrics["sharpe_ratio"],
                "max_drawdown": sweep_metrics["max_drawdown"],
                "total_trade_count": sweep_metrics["total_trade_count"],
                "profitable_trade_rate": sweep_metrics["profitable_trade_rate"],
                "average_winner_return": sweep_metrics["average_winner_return"],
                "average_loser_return": sweep_metrics["average_loser_return"],
                "take_profit_count": sweep_metrics["take_profit_count"],
                "open_positions": int(sweep_trades["status"].eq("持有中").sum()) if not sweep_trades.empty else 0,
            })
    sweep = pd.DataFrame(sweep_rows)
    sweep_path = root / "take_profit_sweep.csv"
    sweep.to_csv(sweep_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    threshold_summary = []
    for label, group in sweep.groupby("threshold_label", sort=False):
        threshold_summary.append({
            "threshold": label,
            "mean_cumulative_return": float(group["cumulative_return"].mean()),
            "median_cumulative_return": float(group["cumulative_return"].median()),
            "positive_strategy_count": int(group["cumulative_return"].gt(0).sum()),
            "mean_max_drawdown": float(group["max_drawdown"].mean()),
        })
    index_path = write_suite_index(root, start, end, summaries, {
        "st_filter": {"mode": "current_name_approximation", "excluded_count": len(st_codes)},
        "trading_constraints": "开盘涨停不买、开盘跌停不卖",
        "take_profit_sweep_file": sweep_path.name,
        "take_profit_sweep_summary": threshold_summary,
    })
    LOGGER.info("多策略回测汇总: %s", index_path)
    return index_path, summaries
