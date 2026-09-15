from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest.research_engine import ResearchSpec, run_research_backtest
from .backtest.strategy_suite import write_strategy_result
from .config import resolve_path
from .pipeline import build_store

LOGGER = logging.getLogger(__name__)

# Fixed before evaluating the requested year. The later year never selects weights.
PROFILES = {
    "balanced": {"low_vol": .40, "value": .30, "momentum": .20, "liquidity": .10},
    "defensive": {"low_vol": .60, "value": .20, "momentum": .10, "liquidity": .10},
    "value": {"low_vol": .30, "value": .50, "momentum": .10, "liquidity": .10},
    "momentum": {"low_vol": .30, "value": .10, "momentum": .50, "liquidity": .10},
}
PROFILE_NAMES = {
    "balanced": "均衡低波价值", "defensive": "低波防守", "value": "价值优先", "momentum": "中期动量",
}
SOURCES = [
    {"title": "Volatility Managed Portfolios", "url": "https://www.nber.org/papers/w22208"},
    {"title": "Momentum Crashes", "url": "https://www.nber.org/papers/w20439"},
    {"title": "Understanding Defensive Equity", "url": "https://www.aqr.com/insights/research/white-papers/understanding-defensive-equity"},
    {"title": "Trading Costs", "url": "https://www.aqr.com/insights/research/working-paper/trading-costs"},
]


def prepare_research_signals(
    daily: pd.DataFrame, valuations: pd.DataFrame, basic: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Daily cross-sectional scores; valuations must match the signal date exactly."""
    codes = daily["ts_code"].astype(str)
    main_board = codes.str.match(r"^(600|601|603|605)\d{3}\.SH$|^(000|001|002|003)\d{3}\.SZ$")
    frame = daily.loc[main_board].sort_values(["ts_code", "trade_date"]).copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    group = frame.groupby("ts_code", sort=False)
    frame["history_count"] = group.cumcount() + 1
    frame["return"] = pd.to_numeric(frame["pct_chg"], errors="coerce") / 100
    frame["price_index"] = (1 + frame["return"]).groupby(frame["ts_code"]).cumprod()
    price = frame.groupby("ts_code", sort=False)["price_index"]
    # Skip the last five sessions to reduce overlap with very short-term chasing.
    frame["momentum_63_5"] = price.shift(5) / price.shift(63) - 1
    frame["volatility"] = frame.groupby("ts_code", sort=False)["return"].transform(
        lambda values: values.rolling(60, min_periods=60).std()
    ) * np.sqrt(252)
    frame["mean_amount"] = group["amount"].transform(lambda values: values.rolling(20).mean())
    frame["return_5"] = price.pct_change(5, fill_method=None)
    value = valuations[["trade_date", "ts_code", "pe_ttm", "pb"]].copy()
    value["trade_date"] = pd.to_datetime(value["trade_date"])
    if value.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("每日估值有重复键，不能进行研究回测")
    frame = frame.merge(value, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    metadata = basic.drop_duplicates("ts_code").set_index("ts_code")
    frame["name"] = frame["ts_code"].map(metadata["name"]).fillna("")
    frame["industry"] = frame["ts_code"].map(metadata.get("industry", pd.Series(dtype=str))).fillna("未知行业")
    frame["list_date"] = pd.to_datetime(frame["ts_code"].map(metadata["list_date"]), errors="coerce")
    valid = (
        frame["history_count"].ge(120)
        & (frame["trade_date"] - frame["list_date"]).dt.days.ge(180)
        & ~frame["name"].str.upper().str.contains("ST|退", regex=True)
        & frame["pe_ttm"].between(.01, 80) & frame["pb"].between(.01, 10)
        & frame["mean_amount"].ge(100_000) & frame["amount"].ge(20_000)
        & frame["vol"].gt(0) & frame["close"].ge(3)
        & frame["volatility"].between(.05, .80)
        & frame["return_5"].lt(.20)
    )
    eligible = frame.loc[valid].dropna(subset=["momentum_63_5", "volatility"]).copy()
    def pct_rank(series: pd.Series) -> pd.Series:
        return series.groupby(eligible["trade_date"]).rank(pct=True)
    eligible["low_vol"] = pct_rank(-eligible["volatility"])
    eligible["value"] = .5 * pct_rank(1 / eligible["pe_ttm"]) + .5 * pct_rank(1 / eligible["pb"])
    eligible["momentum"] = pct_rank(eligible["momentum_63_5"])
    eligible["liquidity"] = pct_rank(eligible["mean_amount"])
    results = {}
    for profile, weights in PROFILES.items():
        scored = eligible[["trade_date", "ts_code", "name", "industry", "volatility", "pe_ttm", "pb", "momentum_63_5", "mean_amount", "close"]].copy()
        scored["score"] = sum(eligible[key] * weight for key, weight in weights.items())
        scored = scored.sort_values(["trade_date", "score", "ts_code"], ascending=[True, False, True])
        scored["rank"] = scored.groupby("trade_date").cumcount() + 1
        results[profile] = scored[scored["rank"].le(100)].reset_index(drop=True)
    return results, {
        "daily_rows": len(daily), "main_board_rows": len(frame), "eligible_rows": len(eligible),
        "valuation_coverage": float(frame["pe_ttm"].notna().mean()),
        "eligible_dates": int(eligible["trade_date"].nunique()),
        "first_eligible_date": str(eligible["trade_date"].min().date()),
        "last_eligible_date": str(eligible["trade_date"].max().date()),
        "st_filter": "按当前名称排除ST及退市名称，存在历史状态/幸存者偏差",
        "universe": "沪深主板；上市180天且120条观测；盈利PE≤80、PB≤10；20日均成交额≥1亿元；5日涨幅<20%；年化波动5%～80%",
    }


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_strategy_research(config: dict[str, Any], end_date: date | None = None) -> Path:
    store = build_store(config)
    daily = store.load_daily()
    if daily.empty:
        raise ValueError("本地没有日线数据，请先更新行情")
    end = min(end_date or date.today(), daily["trade_date"].max().date())
    year_start = (pd.Timestamp(end) - pd.DateOffset(years=1)).date()
    train_end = year_start - pd.Timedelta(days=1)
    train_start = max(date(2025, 1, 2), (daily["trade_date"].min() + pd.Timedelta(days=180)).date())
    if (pd.Timestamp(train_end) - pd.Timestamp(train_start)).days < 120:
        raise ValueError("近一年之前的训练历史不足120天，不能进行时间分离验证")
    daily = daily[daily["trade_date"].dt.date.le(end)].copy()
    valuations = store.load_market_dataset("daily_basic")
    benchmark = store.load_market_dataset("benchmark")
    limits = store.load_market_dataset("limit")
    basic = store.load_basic()
    names = basic.drop_duplicates("ts_code").set_index("ts_code")["name"].fillna("").to_dict()
    if benchmark.empty or valuations.empty:
        raise ValueError("缺少沪深300ETF或历史每日估值数据，不能完成本次研究")
    evaluation_dates = set(daily.loc[daily["trade_date"].dt.date.between(train_start, end), "trade_date"])
    if not evaluation_dates.issubset(set(benchmark["trade_date"])):
        raise ValueError("沪深300ETF未覆盖完整验证区间，请先更新基准数据")
    LOGGER.info("生成4组预设评分：低波、价值、中期动量、流动性")
    signals, coverage = prepare_research_signals(daily, valuations, basic)
    settings = config.get("strategy_research", {})
    costs = dict(config["backtest"]["cost"])
    initial_cash = float(config["backtest"]["initial_cash"])
    base_spec = ResearchSpec(
        strategy_id="robust_research", name="低波价值与中期动量组合",
        description="分散持有最多10只；周度补仓；Top20进入、连续两次跌出Top40退出；单股买入最多10%，组合目标仓位80%；8%收盘止损、12%收盘移动止盈。",
        use_market_regime=False, corporate_action_mode="reinvest_proxy",
    )
    protocol = {
        "engine_version": 1, "profiles": PROFILES, "spec": asdict(base_spec), "costs": costs,
        "train_start": train_start, "train_end": train_end,
        "evaluation_start": year_start, "end": end,
        "selection_rule": "仅按训练期累计收益 + 1.5×最大回撤 - 0.1×年化波动评分；至少5笔已平仓交易；不按后续一年收益选优",
        "data": coverage, "sources": SOURCES,
        "validation_scope": "回顾性时间分离验证；这段历史已被先前策略研究使用，不能称为从未看过的真正样本外数据",
        "corporate_action_model": "持仓按前次收盘/当日除权昨收比例调整经济份额，近似分红再投资；未还原现金到账日、送转登记和红利税",
    }
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True, default=str).encode()).hexdigest()[:10]
    root = resolve_path(config, settings.get("output_dir", "backtests/strategy_research")) / f"{end}_{signature}"
    root.mkdir(parents=True, exist_ok=True)
    _json(root / "protocol.json", protocol)
    training = []
    for profile in PROFILES:
        spec = replace(base_spec, strategy_id=f"research_{profile}", name=PROFILE_NAMES[profile])
        result, trades, events, metrics = run_research_backtest(
            daily, signals[profile], benchmark, names, spec, train_start, train_end,
            initial_cash, costs, limit_daily=limits,
        )
        score = metrics["cumulative_return"] + 1.5 * metrics["max_drawdown"] - .1 * metrics["annualized_volatility"]
        row = {"profile": profile, "name": spec.name, "selection_score": score, **metrics}
        training.append(row)
        write_strategy_result(root / "training" / profile, result, trades, events, metrics)
        LOGGER.info("训练 %s：收益 %.2f%%，回撤 %.2f%%，评分 %.4f", spec.name, metrics["cumulative_return"] * 100, metrics["max_drawdown"] * 100, score)
    training.sort(key=lambda item: item["selection_score"], reverse=True)
    qualified = [item for item in training if item.get("total_trade_count", 0) >= 5]
    if not qualified:
        raise ValueError("训练期没有至少5笔完成交易的候选，请增加历史覆盖")
    selected = qualified[0]["profile"]
    selected_spec = replace(base_spec, strategy_id=f"research_{selected}", name=PROFILE_NAMES[selected] + "（时间分离验证）")
    _json(root / "selection.json", {"selected_profile": selected, "training": training})
    LOGGER.info("训练选定 %s，权重冻结，开始验证", selected_spec.name)
    horizons = [("近一年", "year", year_start), ("近三个月", "quarter", (pd.Timestamp(end) - pd.DateOffset(months=3)).date()), ("近一个月", "month", (pd.Timestamp(end) - pd.DateOffset(months=1)).date())]
    periods = []
    for label, key, start in horizons:
        result, trades, events, metrics = run_research_backtest(
            daily, signals[selected], benchmark, names, selected_spec, start, end,
            initial_cash, costs, limit_daily=limits,
        )
        metrics.update({"threshold_enabled": True, "research_period": label, "research_profile": selected})
        write_strategy_result(root / key, result, trades, events, metrics)
        periods.append({"label": label, "key": key, "start_date": str(start), "end_date": str(end), "metrics": metrics})
        LOGGER.info("%s：累计 %.2f%%，回撤 %.2f%%", label, metrics["cumulative_return"] * 100, metrics["max_drawdown"] * 100)
    # One change at a time, reported without selecting the final profile on these results.
    variants = {
        "ma60_regime": replace(selected_spec, use_market_regime=True),
        "no_stop": replace(selected_spec, stop_loss=None, trailing_stop=None),
        "tight_exit": replace(selected_spec, exit_rank=20, exit_confirmation_days=1),
        "double_cost": selected_spec,
        "raw_prices": replace(selected_spec, corporate_action_mode="raw"),
    }
    ablations = []
    for key, spec in variants.items():
        run_costs = {name: value * 2 for name, value in costs.items()} if key == "double_cost" else costs
        result, trades, events, metrics = run_research_backtest(
            daily, signals[selected], benchmark, names, spec, year_start, end,
            initial_cash, run_costs, limit_daily=limits,
        )
        write_strategy_result(root / "checks" / key, result, trades, events, metrics)
        ablations.append({"check": key, **metrics})
    current = signals[selected]
    current = current[current["trade_date"].eq(current["trade_date"].max())].head(20)
    current.to_csv(root / "candidates.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_name": selected_spec.name, "selected_profile": selected,
        "weights": PROFILES[selected], "spec": asdict(selected_spec),
        "periods": periods, "training": training, "checks": ablations,
        "protocol": protocol,
        "return_definition": "各窗口以相同初始资金独立空仓起跑，历史信号保留暖场；期末持仓按收盘价估值，不强制平仓",
    }
    _json(root / "research.json", summary)
    return root
