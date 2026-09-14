from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from .backtest.engine import run_backtest as execute_backtest
from .backtest.metrics import calculate_metrics
from .backtest.reporting import write_backtest_report
from .backtest.strategy_lab import run_experiments
from .backtest.strategy_suite import (
    BASE_STRATEGIES, STRATEGIES, StrategySpec, run_event_strategy, write_strategy_result,
    write_suite_index,
)
from .config import resolve_path
from .data.akshare_provider import AkshareMetadataProvider
from .data.selection_cache import SelectionCache, strategy_key
from .data.demo_provider import DemoProvider
from .data.storage import ParquetStore
from .data.tushare_provider import TushareProvider
from .data.updater import update_data, update_enriched_data
from .factors import calculate_factors
from .report import write_csv
from .signal_statistics import streak2_leader_continuation
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
    if not is_demo and bool(config["data"].get("fetch_enriched_data", True)):
        daily = store.load_daily()
        if not daily.empty:
            update_enriched_data(
                provider, store, daily["trade_date"].min().date(), end_date or date.today(),
                config.get("backtest", {}).get("benchmark", "510300.SH"),
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
    report_dir = resolve_path(config, config["report"]["output_dir"])
    profile_id = config["strategy"].get("profile_id")
    if profile_id:
        report_dir = report_dir / str(profile_id)
    path = write_csv(selection, report_dir)
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
    current_top5 = {code for code, rank in current.items() if rank <= 5}
    current_top20 = {code for code, rank in current.items() if rank <= 20}
    current_top50 = set(current)
    streaks5 = {code: 0 for code in current_top5}
    streaks = {code: 0 for code in current_top20}
    streaks50 = {code: 0 for code in current_top50}
    for code in current_top5:
        for trade_date in reversed(expected_dates):
            if by_date.get(trade_date, {}).get(code, CACHE_TOP_K + 1) <= 5:
                streaks5[code] += 1
            else:
                break
    for code in current_top20:
        for trade_date in reversed(expected_dates):
            if by_date.get(trade_date, {}).get(code, CACHE_TOP_K + 1) <= 20:
                streaks[code] += 1
            else:
                break
    for code in current_top50:
        for trade_date in reversed(expected_dates):
            if code in by_date.get(trade_date, {}):
                streaks50[code] += 1
            else:
                break
    stats = {
        code: {
            "consecutive_top5": int(streaks5.get(code, 0)),
            "consecutive_top20": int(streaks.get(code, 0)),
            "consecutive_top50": int(streaks50.get(code, 0)),
            "top50_count": int(count),
        }
        for code, count in total_counts.items()
    }
    return stats, {
        "cached_days": len(by_date), "expected_days": len(expected_dates),
    }


def top20_streak2_leader_probability(
    config: dict[str, Any], selected_date: date,
    start_date: date = RECOMMENDATION_HISTORY_START,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Historical continuation rate for the best-ranked exact two-day Top20 streak."""
    store = build_store(config)
    daily = store.load_daily()
    trading_dates = [
        pd.Timestamp(value).date() for value in sorted(daily["trade_date"].unique())
        if start_date <= pd.Timestamp(value).date() <= selected_date
    ]
    cache = SelectionCache(store.root / "selection_results.sqlite3")
    rankings = cache.history(
        strategy_key(_top50_config(config)), start_date, selected_date,
    )
    summary, events = streak2_leader_continuation(rankings, trading_dates, top_n=20)
    basic = store.load_basic().drop_duplicates("ts_code")
    names = basic.set_index("ts_code")["name"].fillna("").astype(str).to_dict()
    candidate = summary.get("current_candidate")
    if candidate:
        candidate["name"] = names.get(candidate["ts_code"], "")
    if not events.empty:
        events["name"] = events["ts_code"].map(names).fillna("")
    summary["cached_trading_dates"] = int(rankings["trade_date"].nunique()) if not rankings.empty else 0
    summary["expected_trading_dates"] = len(trading_dates)
    return summary, events


def best_strategy_recommendations(
    config: dict[str, Any], selected_date: date, top_n: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build next-open candidates supported by the best backtested event strategies."""
    suite_root = resolve_path(
        config, config.get("strategy_suite", {}).get("output_dir", "backtests/strategy_suite")
    )
    suite_paths = sorted(suite_root.glob("*/suite.json"), reverse=True)
    if not suite_paths:
        return pd.DataFrame(), {"reason": "尚无多策略回测结果", "strategies": []}
    suite_path = suite_paths[0]
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    spec_by_id = {spec.strategy_id: spec for spec in STRATEGIES}
    ranked = sorted(
        (
            item for item in suite.get("strategies", [])
            if item.get("strategy_id") in spec_by_id
        ),
        key=lambda item: float(item["metrics"]["cumulative_return"]),
        reverse=True,
    )[:top_n]
    if not ranked:
        return pd.DataFrame(), {"reason": "回测结果中没有可用于当日信号的策略", "strategies": []}

    signal_config = {
        **config,
        "strategy": {**config["strategy"], "top_k": CACHE_TOP_K},
    }
    top50, calculation = selection_for_date(signal_config, selected_date)
    stats, coverage = recommendation_history_stats(config, selected_date)
    rows: list[dict[str, Any]] = []
    for item in ranked:
        spec = spec_by_id[item["strategy_id"]]
        rank_limit = spec.fallback_entry_rank or spec.entry_rank
        candidates: list[pd.Series] = []
        for _, stock in top50.sort_values("rank").iterrows():
            code = str(stock["ts_code"])
            rank = int(stock["rank"])
            if rank > rank_limit:
                continue
            streak = 1
            if spec.consecutive_rank is not None:
                streak = int(stats.get(code, {}).get(f"consecutive_top{spec.consecutive_rank}", 0))
            if streak >= spec.consecutive_days:
                candidates.append(stock)
        if spec.fallback_entry_rank is not None:
            candidates = candidates[:1]
        for stock in candidates:
            code = str(stock["ts_code"])
            rows.append({
                "ts_code": code,
                "name": str(stock.get("name", code)),
                "rank": int(stock["rank"]),
                "total_score": float(stock["total_score"]),
                "consecutive_top20": int(stats.get(code, {}).get("consecutive_top20", 0)),
                "consecutive_top50": int(stats.get(code, {}).get("consecutive_top50", 0)),
                "strategy_id": spec.strategy_id,
                "strategy_name": spec.name,
                "strategy_return": float(item["metrics"]["cumulative_return"]),
            })
    strategy_meta = [{
        "strategy_id": item["strategy_id"],
        "name": item["name"],
        "cumulative_return": float(item["metrics"]["cumulative_return"]),
    } for item in ranked]
    if not rows:
        return pd.DataFrame(), {
            "reason": "最佳五策略在所选日期均没有满足入场条件的标的",
            "strategies": strategy_meta, "suite": suite_path.parent.name,
            "coverage": coverage, "calculation": calculation,
        }
    detail = pd.DataFrame(rows)
    recommendations = detail.groupby(["ts_code", "name"], as_index=False).agg(
        rank=("rank", "min"),
        total_score=("total_score", "max"),
        consecutive_top20=("consecutive_top20", "max"),
        consecutive_top50=("consecutive_top50", "max"),
        strategy_support_count=("strategy_id", "nunique"),
        supporting_strategies=("strategy_name", lambda values: "；".join(dict.fromkeys(values))),
        best_supporting_return=("strategy_return", "max"),
        mean_supporting_return=("strategy_return", "mean"),
    )
    recommendations = recommendations.sort_values(
        ["strategy_support_count", "rank", "total_score"],
        ascending=[False, True, False], kind="stable",
    ).reset_index(drop=True)
    recommendations.insert(0, "joint_rank", recommendations.index + 1)
    return recommendations, {
        "reason": "", "strategies": strategy_meta, "suite": suite_path.parent.name,
        "coverage": coverage, "calculation": calculation,
    }


def combined_signal_definitions(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return five best lab variants plus the best three retained suite strategies."""
    definitions: list[dict[str, Any]] = []
    lab_root = resolve_path(
        config, config.get("strategy_lab", {}).get("output_dir", "backtests/strategy_lab")
    )
    lab_paths = []
    for candidate in sorted(lab_root.glob("*/results.csv"), reverse=True):
        progress_path = candidate.parent / "progress.json"
        if not progress_path.exists() or json.loads(progress_path.read_text(encoding="utf-8")).get("status") == "completed":
            lab_paths.append(candidate)
    if lab_paths:
        lab = pd.read_csv(lab_paths[0]).sort_values("cumulative_return", ascending=False).head(5)
        for _, row in lab.iterrows():
            definitions.append({
                "strategy_id": f"lab_{row['experiment_id']}", "source": "策略试验场前五",
                "name": str(row["strategy_name"]), "entry_rank": int(row["entry_rank"]),
                "consecutive_rank": int(row["entry_rank"]), "consecutive_days": 3,
                "exit_rank": int(row["exit_rank"]),
                "confirmation_days": int(row["confirmation_days"]),
                "take_profit": float(row["take_profit"]),
                "historical_return": float(row["cumulative_return"]),
            })
    suite_root = resolve_path(
        config, config.get("strategy_suite", {}).get("output_dir", "backtests/strategy_suite")
    )
    suite_paths = sorted(suite_root.glob("*/suite.json"), reverse=True)
    if suite_paths:
        suite = json.loads(suite_paths[0].read_text(encoding="utf-8"))
        spec_by_id = {spec.strategy_id: spec for spec in STRATEGIES}
        ranked = sorted(
            (item for item in suite.get("strategies", []) if item.get("strategy_id") in spec_by_id),
            key=lambda item: float(item["metrics"]["cumulative_return"]), reverse=True,
        )[:3]
        for item in ranked:
            spec = spec_by_id[item["strategy_id"]]
            definitions.append({
                "strategy_id": spec.strategy_id, "source": "原联合推荐前三",
                "name": spec.name, "entry_rank": spec.entry_rank,
                "consecutive_rank": spec.consecutive_rank,
                "consecutive_days": spec.consecutive_days, "exit_rank": spec.exit_rank,
                "confirmation_days": spec.exit_confirmation_days,
                "take_profit": float(item["metrics"].get("take_profit_threshold") or 0.20),
                "historical_return": float(item["metrics"]["cumulative_return"]),
            })
    return definitions


def _ranking_context(
    config: dict[str, Any], selected_date: date,
) -> tuple[list[date], dict[date, dict[str, int]]]:
    store = build_store(config)
    cache = SelectionCache(store.root / "selection_results.sqlite3")
    history = cache.history(
        strategy_key(_top50_config(config)), RECOMMENDATION_HISTORY_START, selected_date,
    )
    dates = sorted(history["trade_date"].unique()) if not history.empty else []
    by_date = {
        trade_date: day.set_index("ts_code")["rank"].astype(int).to_dict()
        for trade_date, day in history.groupby("trade_date")
    }
    return dates, by_date


def _consecutive_inside(
    dates: list[date], by_date: dict[date, dict[str, int]], code: str, threshold: int,
) -> int:
    count = 0
    for trade_date in reversed(dates):
        if by_date.get(trade_date, {}).get(code, CACHE_TOP_K + 1) <= threshold:
            count += 1
        else:
            break
    return count


def _consecutive_outside(
    dates: list[date], by_date: dict[date, dict[str, int]], code: str, threshold: int,
) -> int:
    count = 0
    for trade_date in reversed(dates):
        if by_date.get(trade_date, {}).get(code, CACHE_TOP_K + 1) > threshold:
            count += 1
        else:
            break
    return count


def combined_signal_recommendations(
    config: dict[str, Any], selected_date: date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate current entry candidates from the fixed eight-strategy signal set."""
    definitions = combined_signal_definitions(config)
    if len(definitions) != 8:
        return pd.DataFrame(), {
            "reason": f"需要5个试验场策略和3个原策略，当前只找到{len(definitions)}个。",
            "strategies": definitions,
        }
    signal_config = {**config, "strategy": {**config["strategy"], "top_k": CACHE_TOP_K}}
    top50, calculation = selection_for_date(signal_config, selected_date)
    dates, by_date = _ranking_context(config, selected_date)
    rows: list[dict[str, Any]] = []
    for definition in definitions:
        required_rank = definition["consecutive_rank"]
        for _, stock in top50.loc[top50["rank"].le(definition["entry_rank"])].iterrows():
            code = str(stock["ts_code"])
            streak = (
                1 if required_rank is None
                else _consecutive_inside(dates, by_date, code, int(required_rank))
            )
            if streak < int(definition["consecutive_days"]):
                continue
            rows.append({
                "ts_code": code, "name": str(stock.get("name", code)),
                "rank": int(stock["rank"]), "total_score": float(stock["total_score"]),
                "close": float(stock["close"]),
                "consecutive_entry_days": streak, "strategy_id": definition["strategy_id"],
                "strategy_name": definition["name"], "strategy_source": definition["source"],
                "strategy_return": definition["historical_return"],
            })
    metadata = {"reason": "", "strategies": definitions, "calculation": calculation}
    if not rows:
        metadata["reason"] = "八策略在所选日期均没有满足新开仓条件的股票。"
        return pd.DataFrame(), metadata
    detail = pd.DataFrame(rows)
    recommendations = detail.groupby(["ts_code", "name"], as_index=False).agg(
        rank=("rank", "min"), total_score=("total_score", "max"), close=("close", "max"),
        consecutive_entry_days=("consecutive_entry_days", "max"),
        strategy_support_count=("strategy_id", "nunique"),
        supporting_strategies=("strategy_name", lambda values: "；".join(dict.fromkeys(values))),
        best_supporting_return=("strategy_return", "max"),
    ).sort_values(
        ["strategy_support_count", "rank", "total_score"], ascending=[False, True, False],
        kind="stable",
    ).reset_index(drop=True)
    recommendations.insert(0, "joint_rank", recommendations.index + 1)
    return recommendations, metadata


def stock_signal_reminders(
    config: dict[str, Any], selected_date: date, ts_code: str,
    entry_price: float, price_stop_loss: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate eight strategy exits, profit targets and an auxiliary price stop."""
    definitions = combined_signal_definitions(config)
    dates, by_date = _ranking_context(config, selected_date)
    current_rank = by_date.get(selected_date, {}).get(ts_code, CACHE_TOP_K + 1)
    store = build_store(config)
    daily = store.load_daily()
    market = daily[
        daily["trade_date"].dt.date.eq(selected_date) & daily["ts_code"].eq(ts_code)
    ]
    if market.empty:
        raise ValueError(f"{selected_date} 没有 {ts_code} 的日线行情")
    row = market.iloc[-1]
    day_high, day_low, close = float(row["high"]), float(row["low"]), float(row["close"])
    price_stop = entry_price * (1 - price_stop_loss)
    price_stop_met = day_low <= price_stop
    records = []
    for definition in definitions:
        required_rank = definition["consecutive_rank"]
        entry_streak = (
            1 if required_rank is None
            else _consecutive_inside(dates, by_date, ts_code, int(required_rank))
        )
        entry_met = current_rank <= definition["entry_rank"] and entry_streak >= definition["consecutive_days"]
        exit_streak = _consecutive_outside(dates, by_date, ts_code, definition["exit_rank"])
        rank_exit_met = exit_streak >= definition["confirmation_days"]
        profit_price = entry_price * (1 + definition["take_profit"])
        profit_met = day_high >= profit_price
        records.append({
            "strategy_name": definition["name"], "source": definition["source"],
            "historical_return": definition["historical_return"], "current_rank": current_rank,
            "entry_condition": entry_met, "entry_streak": entry_streak,
            "take_profit_rate": definition["take_profit"], "take_profit_price": profit_price,
            "take_profit_met": profit_met, "exit_rank": definition["exit_rank"],
            "required_exit_confirmations": definition["confirmation_days"],
            "current_exit_streak": exit_streak, "rank_exit_met": rank_exit_met,
            "signal": "止盈" if profit_met else "策略退出" if rank_exit_met else "价格止损预警" if price_stop_met else "买入条件满足" if entry_met else "继续观察",
        })
    return pd.DataFrame(records), {
        "trade_date": selected_date, "ts_code": ts_code, "open": float(row["open"]),
        "high": day_high, "low": day_low, "close": close, "entry_price": entry_price,
        "price_change_from_entry": close / entry_price - 1, "price_stop_loss": price_stop_loss,
        "price_stop": price_stop, "price_stop_met": price_stop_met,
    }


def run_recent_signal_ensemble(
    config: dict[str, Any], months: int = 3, end_date: date | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Backtest the eight signal strategies as equal-capital independent sleeves."""
    if months <= 0:
        raise ValueError("回测月数必须大于0")
    definitions = combined_signal_definitions(config)
    if len(definitions) != 8:
        raise RuntimeError(f"八策略定义不完整，当前找到 {len(definitions)} 套")
    store = build_store(config)
    daily = store.load_daily()
    if daily.empty:
        raise RuntimeError("本地没有日线数据")
    end = min(end_date or date.today(), daily["trade_date"].max().date())
    start = (pd.Timestamp(end) - pd.DateOffset(months=months)).date()
    signal_start = start - timedelta(days=30)
    warm_recommendation_cache(config, signal_start, end)
    cache = SelectionCache(store.root / "selection_results.sqlite3")
    rankings = cache.history(strategy_key(_top50_config(config)), signal_start, end)
    basic = store.load_basic()
    names = (
        basic.dropna(subset=["name"]).drop_duplicates("ts_code")
        .set_index("ts_code")["name"].astype(str).to_dict()
        if "name" in basic else {}
    )
    if config.get("market", {}).get("exclude_st", True) and "name" in basic:
        st_mask = basic["name"].fillna("").astype(str).str.upper().str.contains("ST")
        st_codes = set(basic.loc[st_mask, "ts_code"].astype(str))
        daily = daily[~daily["ts_code"].isin(st_codes)].copy()
        rankings = rankings[~rankings["ts_code"].isin(st_codes)].copy()

    initial_cash = float(config["backtest"]["initial_cash"])
    sleeve_cash = initial_cash / len(definitions)
    component_daily: list[pd.DataFrame] = []
    component_summaries: list[dict[str, Any]] = []
    benchmark = store.load_market_dataset("benchmark")
    limits = store.load_market_dataset("limit")
    for definition in definitions:
        spec = StrategySpec(
            strategy_id=definition["strategy_id"], name=definition["name"],
            description=f"八策略联合中的独立资金子账户：{definition['source']}",
            entry_rank=int(definition["entry_rank"]), exit_rank=int(definition["exit_rank"]),
            consecutive_rank=(
                int(definition["consecutive_rank"])
                if definition["consecutive_rank"] is not None else None
            ),
            consecutive_days=int(definition["consecutive_days"]), weighting="equal",
            exit_confirmation_days=int(definition["confirmation_days"]),
        )
        result, trades, _, metrics = run_event_strategy(
            daily, rankings, names, spec, start, end, sleeve_cash,
            config["backtest"]["cost"], take_profit=float(definition["take_profit"]),
            record_profit=float(definition["take_profit"]),
            benchmark_daily=benchmark,
            limit_daily=limits,
        )
        component = result[[
            "trade_date", "equity_value", "benchmark_return", "turnover",
            "transaction_cost", "rebalanced", "holdings",
        ]].copy().set_index("trade_date")
        component.columns = pd.MultiIndex.from_product([[definition["strategy_id"]], component.columns])
        component_daily.append(component)
        component_summaries.append({
            **definition,
            "recent_cumulative_return": metrics["cumulative_return"],
            "recent_annualized_return": metrics["annualized_return"],
            "recent_sharpe_ratio": metrics["sharpe_ratio"],
            "recent_max_drawdown": metrics["max_drawdown"],
            "recent_trade_count": metrics["total_trade_count"],
            "recent_trade_win_rate": metrics["profitable_trade_rate"],
            "recent_open_positions": int(trades["status"].eq("持有中").sum()) if not trades.empty else 0,
        })

    combined = pd.concat(component_daily, axis=1).sort_index()
    value_columns = [column for column in combined if column[1] == "equity_value"]
    equity_value = combined[value_columns].sum(axis=1)
    previous_value = equity_value.shift(1).fillna(initial_cash)
    net_return = equity_value / previous_value - 1
    benchmark_columns = [column for column in combined if column[1] == "benchmark_return"]
    benchmark_return = combined[benchmark_columns[0]]
    rebalanced_columns = [column for column in combined if column[1] == "rebalanced"]
    holding_columns = [column for column in combined if column[1] == "holdings"]
    turnover_value = pd.Series(0.0, index=combined.index)
    cost_value = pd.Series(0.0, index=combined.index)
    for definition in definitions:
        strategy_id = definition["strategy_id"]
        prior_sleeve_value = combined[(strategy_id, "equity_value")].shift(1).fillna(sleeve_cash)
        turnover_value += combined[(strategy_id, "turnover")] * prior_sleeve_value
        cost_value += combined[(strategy_id, "transaction_cost")] * prior_sleeve_value
    ensemble = pd.DataFrame({
        "trade_date": combined.index, "signal_date": combined.index,
        "gross_return": (equity_value + cost_value) / previous_value - 1,
        "net_return": net_return, "benchmark_return": benchmark_return,
        "turnover": turnover_value / previous_value,
        "transaction_cost": cost_value / previous_value,
        "holdings": combined[holding_columns].sum(axis=1),
        "rebalanced": combined[rebalanced_columns].any(axis=1),
        "equity_value": equity_value,
    }).reset_index(drop=True)
    ensemble["equity"] = ensemble["equity_value"] / initial_cash
    ensemble["benchmark_equity"] = (1 + ensemble["benchmark_return"]).cumprod()
    ensemble["drawdown"] = ensemble["equity"] / ensemble["equity"].cummax() - 1
    metrics = calculate_metrics(ensemble)
    metrics.update({
        "strategy_id": "eight_strategy_equal_sleeves_recent",
        "strategy_name": "八策略联合（等资金子账户）最近三个月",
        "strategy_description": "八套策略各使用1/8初始资金并独立执行，组合权益为八个子账户之和。",
        "requested_months": months, "start_date": start.isoformat(), "end_date": end.isoformat(),
        "component_count": len(definitions), "selection_bias_warning": (
            "八套策略由包含本区间的历史结果筛选，本结果是近期稳定性复测，不是严格样本外测试。"
        ),
    })
    output = resolve_path(config, "backtests/signal_ensemble") / f"{start}_{end}"
    output.mkdir(parents=True, exist_ok=True)
    daily_path = output / "daily.csv"
    components_path = output / "components.csv"
    metrics_path = output / "metrics.json"
    ensemble.to_csv(daily_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    pd.DataFrame(component_summaries).sort_values(
        "recent_cumulative_return", ascending=False,
    ).to_csv(components_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics_path, metrics
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
    result = execute_backtest(
        daily, store.load_basic(), config, start, end,
        store.load_market_dataset("benchmark"), store.load_market_dataset("limit"),
    )
    folder_name = f"{start}_{end}"
    profile_id = config["strategy"].get("profile_id")
    if profile_id:
        folder_name = f"{profile_id}_{folder_name}"
    folder = resolve_path(config, settings["output_dir"]) / folder_name
    paths = write_backtest_report(result, folder)
    LOGGER.info("回测完成: %s", folder)
    return paths, result.metrics


def run_streak2_leader_backtest(
    config: dict[str, Any], start_date: date | None = None, end_date: date | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run the exact-two-day Top20 leader strategy against local point-in-time signals."""
    store = build_store(config)
    daily = store.load_daily()
    if daily.empty:
        raise RuntimeError("本地没有日线数据，请先运行每日更新")
    local_end = daily["trade_date"].max().date()
    end = min(end_date or local_end, local_end)
    start = start_date or RECOMMENDATION_HISTORY_START
    if start >= end:
        raise ValueError("回测开始日期必须早于本地最新交易日")
    warm_recommendation_cache(config, RECOMMENDATION_HISTORY_START, end)
    cache = SelectionCache(store.root / "selection_results.sqlite3")
    rankings = cache.history(
        strategy_key(_top50_config(config)), RECOMMENDATION_HISTORY_START, end,
    )
    basic = store.load_basic()
    names = (
        basic.dropna(subset=["name"]).drop_duplicates("ts_code")
        .set_index("ts_code")["name"].astype(str).to_dict()
        if "name" in basic else {}
    )
    excluded_st_count = 0
    if config.get("market", {}).get("exclude_st", True) and "name" in basic:
        st_mask = basic["name"].fillna("").astype(str).str.upper().str.contains("ST")
        st_codes = set(basic.loc[st_mask, "ts_code"].astype(str))
        excluded_st_count = len(st_codes)
        daily = daily[~daily["ts_code"].isin(st_codes)].copy()
        rankings = rankings[~rankings["ts_code"].isin(st_codes)].copy()
    spec = next(
        item for item in STRATEGIES
        if item.strategy_id == "s14_top20_exact2_leader_full"
    )
    result, trades, events, metrics = run_event_strategy(
        daily, rankings, names, spec, start, end,
        float(config["backtest"]["initial_cash"]), config["backtest"]["cost"],
        take_profit=0.32, record_profit=0.32,
        benchmark_daily=store.load_market_dataset("benchmark"),
        limit_daily=store.load_market_dataset("limit"),
    )
    metrics.update({
        "threshold_enabled": True,
        "st_filter_mode": "current_name_approximation",
        "excluded_st_count": excluded_st_count,
        "signal_execution": "收盘确认信号，下一交易日开盘执行",
    })
    output = resolve_path(config, "backtests/streak2_leader") / f"{start}_{end}"
    write_strategy_result(output, result, trades, events, metrics)
    LOGGER.info("连续2次Top20领跑者策略回测完成: %s", output)
    return output, metrics


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
    benchmark = store.load_market_dataset("benchmark")
    limits = store.load_market_dataset("limit")
    baseline = execute_backtest(daily, basic, config, start, end, benchmark, limits)
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
        effective_take_profit = (
            spec.fixed_take_profit if spec.fixed_take_profit is not None else take_profit
        )
        result, trades, events, metrics = run_event_strategy(
            daily, rankings, names, spec, start, end, initial_cash, costs,
            take_profit=effective_take_profit, record_profit=effective_take_profit,
            benchmark_daily=benchmark,
            limit_daily=limits,
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
                take_profit=threshold, record_profit=threshold, benchmark_daily=benchmark,
                limit_daily=limits,
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


def run_strategy_lab(
    config: dict[str, Any], start_date: date | None = None, end_date: date | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Run the best-strategy ablation grid entirely from local cached data."""
    settings = config.get("strategy_lab", {})
    suite_settings = config.get("strategy_suite", {})
    start = start_date or date.fromisoformat(
        settings.get("start_date", suite_settings.get("start_date", "2025-09-01"))
    )
    configured_end = settings.get("end_date")
    requested_end = end_date or (
        date.fromisoformat(configured_end) if configured_end else date.today()
    )
    store = build_store(config)
    daily = store.load_daily()
    if daily.empty:
        raise RuntimeError("本地没有日线数据，请先运行每日更新")
    end = min(requested_end, daily["trade_date"].max().date())
    if start >= end:
        raise ValueError("策略试验场开始日期必须早于本地最新交易日")
    warm_recommendation_cache(config, start, end)
    cache = SelectionCache(store.root / "selection_results.sqlite3")
    rankings = cache.history(strategy_key(_top50_config(config)), start, end)
    expected_dates = {
        pd.Timestamp(value).date() for value in daily["trade_date"].unique()
        if start <= pd.Timestamp(value).date() <= end
    }
    if set(rankings["trade_date"].unique()) != expected_dates:
        raise RuntimeError("策略试验场所需的 Top-50 历史信号缓存不完整")
    basic = store.load_basic()
    names = (
        basic.dropna(subset=["name"]).drop_duplicates("ts_code")
        .set_index("ts_code")["name"].astype(str).to_dict()
        if "name" in basic else {}
    )
    if config.get("market", {}).get("exclude_st", True) and "name" in basic:
        st_mask = basic["name"].fillna("").astype(str).str.upper().str.contains("ST")
        st_codes = set(basic.loc[st_mask, "ts_code"].astype(str))
        daily = daily[~daily["ts_code"].isin(st_codes)].copy()
        rankings = rankings[~rankings["ts_code"].isin(st_codes)].copy()
    cost_key = hashlib.sha256(
        json.dumps(config["backtest"]["cost"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    output = resolve_path(
        config, settings.get("output_dir", "backtests/strategy_lab")
    ) / f"{start}_{end}_cost-{cost_key}"
    return run_experiments(
        daily, rankings, names, start, end,
        float(config["backtest"]["initial_cash"]), config["backtest"]["cost"],
        settings, output, benchmark_daily=store.load_market_dataset("benchmark"),
        limit_daily=store.load_market_dataset("limit"),
    )
