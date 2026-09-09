from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..factors import calculate_factors
from ..strategy import select_stocks
from .metrics import calculate_metrics
from .trading_rules import is_open_limit_down, is_open_limit_up


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]


def _weighted_return(weights: dict[str, float], returns: pd.Series) -> float:
    return float(sum(weight * float(returns.get(code, 0.0)) for code, weight in weights.items()))


def _target_weights(codes: list[str], method: str) -> dict[str, float]:
    if not codes:
        return {}
    if method == "equal":
        return {code: 1.0 / len(codes) for code in codes}
    if method == "rank_linear":
        raw = list(range(len(codes), 0, -1))
        total = float(sum(raw))
        return {code: weight / total for code, weight in zip(codes, raw, strict=True)}
    raise ValueError(f"不支持的回测权重方法: {method}")


def run_backtest(
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    config: dict[str, Any],
    start_date: date,
    end_date: date,
) -> BacktestResult:
    """Run a lagged-signal, configurable-weight, long-only Top-K backtest.

    A signal formed after day T's close is first traded at T+1's open. On a
    rebalance day, old holdings earn the overnight leg and new holdings earn
    the open-to-close leg. This prevents using T+1 prices in T's selection.
    """
    factors = calculate_factors(daily)
    dates = sorted(pd.Timestamp(value) for value in factors["trade_date"].unique())
    dates = [value for value in dates if pd.Timestamp(start_date) <= value <= pd.Timestamp(end_date)]
    if len(dates) < 2:
        raise ValueError("回测区间内至少需要两个交易日")

    settings = config["backtest"]
    top_k = int(settings["top_k"])
    rebalance_days = int(settings["rebalance_days"])
    weighting = str(settings.get("weighting", "equal"))
    costs = settings["cost"]
    holdings: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    all_dates = sorted(pd.Timestamp(value) for value in factors["trade_date"].unique())
    date_position = {value: index for index, value in enumerate(all_dates)}
    for step, trade_date in enumerate(dates):
        global_index = date_position[trade_date]
        if global_index == 0:
            continue
        signal_date = all_dates[global_index - 1]
        market_today = factors[factors["trade_date"].eq(trade_date)].set_index("ts_code")
        close_return = market_today["close"] / market_today["pre_close"] - 1
        benchmark_return = float(close_return.replace([np.inf, -np.inf], np.nan).dropna().mean())
        turnover = buy = sell = cost_rate = 0.0
        blocked_buys = blocked_sells = 0
        rebalanced = step % rebalance_days == 0

        if rebalanced:
            signal = factors[factors["trade_date"].eq(signal_date)]
            selection, _ = select_stocks(signal, basic, config, top_k=top_k)
            codes = [code for code in selection["ts_code"] if code in market_today.index]
            tradable_codes = []
            for code in codes:
                if code not in holdings and is_open_limit_up(
                    code, float(market_today.at[code, "open"]),
                    float(market_today.at[code, "pre_close"]),
                ):
                    blocked_buys += 1
                    continue
                tradable_codes.append(code)
            raw_target = _target_weights(tradable_codes, weighting) if tradable_codes else holdings.copy()
            locked = {
                code: weight for code, weight in holdings.items()
                if raw_target.get(code, 0.0) < weight and code in market_today.index
                and is_open_limit_down(
                    code, float(market_today.at[code, "open"]),
                    float(market_today.at[code, "pre_close"]),
                )
            }
            blocked_sells = len(locked)
            remaining = max(0.0, 1.0 - sum(locked.values()))
            adjustable = {code: weight for code, weight in raw_target.items() if code not in locked}
            adjustable_total = sum(adjustable.values())
            target = locked.copy()
            if adjustable_total > 0:
                target.update({code: weight / adjustable_total * remaining for code, weight in adjustable.items()})
            changes = {code: target.get(code, 0.0) - holdings.get(code, 0.0) for code in set(target) | set(holdings)}
            buy = sum(max(value, 0.0) for value in changes.values())
            sell = sum(max(-value, 0.0) for value in changes.values())
            turnover = buy + sell
            cost_rate = (
                buy * (float(costs["commission_buy"]) + float(costs["slippage"]))
                + sell * (
                    float(costs["commission_sell"])
                    + float(costs["stamp_tax"])
                    + float(costs["slippage"])
                )
            )
            overnight = market_today["open"] / market_today["pre_close"] - 1
            intraday = market_today["close"] / market_today["open"] - 1
            gross_return = (1 + _weighted_return(holdings, overnight)) * (
                1 + _weighted_return(target, intraday)
            ) - 1
            holdings = target
            trades.append({
                "trade_date": trade_date, "signal_date": signal_date,
                "holdings": len(holdings), "buy_turnover": buy,
                "sell_turnover": sell, "cost_rate": cost_rate,
                "codes": ",".join(holdings),
                "weights": ",".join(f"{code}:{weight:.6f}" for code, weight in holdings.items()),
            })
        else:
            gross_return = _weighted_return(holdings, close_return)

        net_return = (1 + gross_return) * (1 - cost_rate) - 1
        records.append({
            "trade_date": trade_date,
            "signal_date": signal_date,
            "gross_return": gross_return,
            "net_return": net_return,
            "benchmark_return": benchmark_return,
            "turnover": turnover,
            "transaction_cost": cost_rate,
            "holdings": len(holdings),
            "rebalanced": rebalanced,
            "limit_up_buy_blocked": blocked_buys,
            "limit_down_sell_blocked": blocked_sells,
        })

    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("没有生成可用的回测记录")
    result["equity"] = (1 + result["net_return"]).cumprod()
    result["equity_value"] = result["equity"] * float(settings["initial_cash"])
    result["benchmark_equity"] = (1 + result["benchmark_return"]).cumprod()
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1
    metrics = calculate_metrics(result)
    metrics.update({
        "limit_up_buy_blocked_count": int(result["limit_up_buy_blocked"].sum()),
        "limit_down_sell_blocked_count": int(result["limit_down_sell_blocked"].sum()),
    })
    return BacktestResult(result, pd.DataFrame(trades), metrics)
