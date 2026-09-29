from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from jkquant.low_position_pool import build_low_position_features, screen_low_position_breakouts

from .benchmark import benchmark_return_map
from .metrics import calculate_metrics
from .trading_rules import affordable_buy_notional, is_open_limit_down, is_open_limit_up, transaction_fee


@dataclass(frozen=True)
class LowPositionStrategySpec:
    strategy_id: str
    name: str
    description: str
    max_positions: int


LOW_POSITION_STRATEGIES = [
    LowPositionStrategySpec(
        "smallcap_focus_top1",
        "小盘重点观察·单股重仓",
        "收盘确认重点观察后，下一交易日从异动强度最高者中最多持有1只；25%止盈、10%止损，连续2日未上涨则次日开盘退出，法定长假前清仓。",
        1,
    ),
    LowPositionStrategySpec(
        "smallcap_focus_top2",
        "小盘重点观察·双股重仓",
        "收盘确认重点观察后，下一交易日从异动强度最高者中最多持有2只，每只目标约占权益一半；25%止盈、10%止损，连续2日未上涨则次日开盘退出，法定长假前清仓。",
        2,
    ),
]


def build_priority_signal_history(
    daily: pd.DataFrame,
    valuation: pd.DataFrame,
    basic: pd.DataFrame,
    settings: dict[str, Any],
    market_settings: dict[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Build end-of-day priority signals once per historical trading day."""
    features = build_low_position_features(
        daily,
        int(settings.get("lookback_days", 250)),
        int(settings.get("consolidation_days", 20)),
    )
    dates = [
        pd.Timestamp(value)
        for value in sorted(features["trade_date"].unique())
        if start_date <= pd.Timestamp(value).date() <= end_date
    ]
    records: list[pd.DataFrame] = []
    for trade_date in dates:
        priority, _ = screen_low_position_breakouts(
            daily,
            valuation,
            basic,
            settings,
            trade_date,
            market_settings,
            prepared_features=features,
        )
        if priority.empty:
            continue
        day = priority.copy()
        day["signal_date"] = trade_date
        day["signal_rank"] = np.arange(1, len(day) + 1)
        records.append(day)
    if not records:
        return pd.DataFrame(columns=["signal_date", "signal_rank", "ts_code", "alert_score"])
    return pd.concat(records, ignore_index=True)


def _statutory_preholiday_dates(trading_dates: list[pd.Timestamp]) -> set[date]:
    """Find the last local trading day before a named statutory holiday."""
    try:
        from chinese_calendar import get_holiday_detail
    except ImportError:
        return set()
    result: set[date] = set()
    for current, following in zip(trading_dates, trading_dates[1:]):
        cursor = current.date() + timedelta(days=1)
        while cursor < following.date():
            try:
                is_holiday, holiday_name = get_holiday_detail(cursor)
            except (NotImplementedError, ValueError):
                break
            if is_holiday and holiday_name is not None:
                result.add(current.date())
                break
            cursor += timedelta(days=1)
    return result


def run_low_position_strategy(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    names: dict[str, str],
    spec: LowPositionStrategySpec,
    start_date: date,
    end_date: date,
    initial_cash: float,
    costs: dict[str, float],
    take_profit: float = 0.40,
    stop_loss: float = 0.10,
    non_up_exit_days: int = 2,
    benchmark_daily: pd.DataFrame | None = None,
    limit_daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Backtest close-confirmed small-cap alerts with next-open entries."""
    market_data = daily[daily["trade_date"].dt.date.between(start_date, end_date)].copy()
    if limit_daily is not None and not limit_daily.empty:
        limits = limit_daily[["trade_date", "ts_code", "up_limit", "down_limit"]].copy()
        market_data = market_data.merge(limits, on=["trade_date", "ts_code"], how="left")
    dates = sorted(pd.Timestamp(value) for value in market_data["trade_date"].unique())
    if not dates:
        raise ValueError("小盘重点观察回测区间没有日线数据")
    signal_map = {
        pd.Timestamp(signal_date).date(): day.sort_values(["alert_score", "signal_rank"], ascending=[False, True])
        for signal_date, day in signals.groupby("signal_date")
    }
    preholiday_dates = _statutory_preholiday_dates(dates)
    benchmark_returns = benchmark_return_map(benchmark_daily)
    buy_commission = float(costs["commission_buy"])
    sell_commission = float(costs["commission_sell"])
    min_commission = float(costs.get("min_commission", 0.0))
    stamp_tax = float(costs["stamp_tax"])
    slippage = float(costs["slippage"])
    cash = previous_equity = float(initial_cash)
    positions: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    daily_records: list[dict[str, Any]] = []
    trade_sequence = 0
    blocked_buys = blocked_sells_total = corporate_action_adjustments = 0

    def sell(code: str, position: dict[str, Any], day: pd.Timestamp, price: float, reason: str) -> float:
        nonlocal cash
        gross = float(position["shares"] * price)
        fee = transaction_fee(
            gross,
            sell_commission,
            min_commission=min_commission,
            stamp_tax_rate=stamp_tax,
            slippage_rate=slippage,
        )
        cash += gross - fee
        trades.append({
            "strategy_id": spec.strategy_id,
            "trade_id": position["trade_id"],
            "ts_code": code,
            "name": names.get(code, ""),
            "entry_date": position["entry_date"],
            "exit_date": day.date(),
            "holding_period": f"{position['entry_date']} 至 {day.date()}",
            "holding_trading_days": int(position["holding_days"]),
            "entry_price": float(position["entry_price"]),
            "entry_price_unadjusted": float(position["entry_price_unadjusted"]),
            "exit_price": float(price),
            "price_change": float(price / position["entry_price"] - 1),
            "net_return": float((gross - fee) / position["cost_basis"] - 1),
            "max_gain": float(position["max_price"] / position["entry_price"] - 1),
            "max_drawdown_during_holding": float(position["min_price"] / position["entry_price"] - 1),
            "exit_reason": reason,
            "crossed20_date": None,
            "status": "已平仓",
        })
        del positions[code]
        return fee

    for index, trade_date in enumerate(dates):
        market = market_data[market_data["trade_date"].eq(trade_date)].set_index("ts_code")
        signal_date = dates[index - 1].date() if index else trade_date.date()
        day_signals = signal_map.get(signal_date, pd.DataFrame()) if index else pd.DataFrame()
        transaction_cost = traded_gross = 0.0
        changed = False
        blocked_today = 0
        blocked_exit_codes: set[str] = set()

        for code, position in list(positions.items()):
            position["holding_days"] += 1
            if code not in market.index:
                continue
            pre_close = float(market.at[code, "pre_close"])
            previous_close = float(position["last_price"])
            if np.isfinite(pre_close) and pre_close > 0 and abs(previous_close - pre_close) > max(
                0.0051, previous_close * 0.00001,
            ):
                ratio = previous_close / pre_close
                position["shares"] *= ratio
                for field in ("entry_price", "last_price", "max_price", "min_price"):
                    position[field] /= ratio
                corporate_action_adjustments += 1

        # Gaps through a risk boundary are handled at the opening price.
        for code in list(positions):
            if code not in market.index:
                continue
            position = positions[code]
            open_price = float(market.at[code, "open"])
            pre_close = float(market.at[code, "pre_close"])
            stop_price = float(position["entry_price"] * (1 - stop_loss))
            take_price = float(position["entry_price"] * (1 + take_profit))
            if position.get("pending_non_up_exit", False):
                exact_down = float(market.at[code, "down_limit"]) if "down_limit" in market and pd.notna(market.at[code, "down_limit"]) else None
                if is_open_limit_down(code, open_price, pre_close, exact_down):
                    blocked_today += 1
                    blocked_sells_total += 1
                    blocked_exit_codes.add(code)
                    continue
                fee = sell(code, position, trade_date, open_price, f"连续{non_up_exit_days}日未上涨")
            elif open_price <= stop_price:
                exact_down = float(market.at[code, "down_limit"]) if "down_limit" in market and pd.notna(market.at[code, "down_limit"]) else None
                if is_open_limit_down(code, open_price, pre_close, exact_down):
                    blocked_today += 1
                    blocked_sells_total += 1
                    blocked_exit_codes.add(code)
                    continue
                fee = sell(code, position, trade_date, open_price, f"亏损达到{stop_loss:.0%}止损")
            elif open_price >= take_price:
                fee = sell(code, position, trade_date, open_price, f"盈利达到{take_profit:.0%}止盈")
            else:
                continue
            traded_gross += float(position["shares"] * open_price)
            transaction_cost += fee
            changed = True

        # Calendar risk is known before the session: do not open fresh positions.
        if trade_date.date() not in preholiday_dates and not day_signals.empty:
            candidates = [
                str(code) for code in day_signals["ts_code"]
                if str(code) not in positions and str(code) in market.index
            ]
            slots = max(0, spec.max_positions - len(positions))
            for code in candidates[:slots]:
                if cash <= min_commission:
                    break
                open_price = float(market.at[code, "open"])
                pre_close = float(market.at[code, "pre_close"])
                exact_up = float(market.at[code, "up_limit"]) if "up_limit" in market and pd.notna(market.at[code, "up_limit"]) else None
                if open_price <= 0 or is_open_limit_up(code, open_price, pre_close, exact_up):
                    blocked_buys += 1
                    continue
                open_equity = cash + sum(
                    float(position["shares"] * float(market.at[held, "open"]))
                    for held, position in positions.items() if held in market.index
                )
                cash_budget = min(cash, open_equity / spec.max_positions)
                gross_limit = affordable_buy_notional(
                    cash_budget,
                    buy_commission,
                    min_commission=min_commission,
                    slippage_rate=slippage,
                )
                shares = int(gross_limit // (open_price * 100)) * 100
                if shares < 100:
                    continue
                gross = float(shares * open_price)
                fee = transaction_fee(
                    gross,
                    buy_commission,
                    min_commission=min_commission,
                    slippage_rate=slippage,
                )
                if gross + fee > cash + 1e-9:
                    continue
                trade_sequence += 1
                cash -= gross + fee
                positions[code] = {
                    "trade_id": f"{spec.strategy_id}-{trade_sequence:06d}",
                    "shares": shares,
                    "entry_date": trade_date.date(),
                    "entry_price": open_price,
                    "entry_price_unadjusted": open_price,
                    "cost_basis": gross + fee,
                    "holding_days": 1,
                    "last_price": open_price,
                    "max_price": open_price,
                    "min_price": open_price,
                    "non_up_streak": 0,
                    "pending_non_up_exit": False,
                }
                traded_gross += gross
                transaction_cost += fee
                changed = True

        # With daily bars the order of a same-day stop and target is unknown;
        # use the conservative convention that the stop is hit first.
        for code in list(positions):
            if code not in market.index:
                continue
            if code in blocked_exit_codes:
                positions[code]["last_price"] = float(market.at[code, "close"])
                continue
            position = positions[code]
            high = float(market.at[code, "high"])
            low = float(market.at[code, "low"])
            open_price = float(market.at[code, "open"])
            stop_price = float(position["entry_price"] * (1 - stop_loss))
            take_price = float(position["entry_price"] * (1 + take_profit))
            position["max_price"] = max(float(position["max_price"]), high)
            position["min_price"] = min(float(position["min_price"]), low)
            if low <= stop_price:
                execution = min(open_price, stop_price)
                fee = sell(code, position, trade_date, execution, f"亏损达到{stop_loss:.0%}止损")
            elif high >= take_price:
                execution = max(open_price, take_price)
                fee = sell(code, position, trade_date, execution, f"盈利达到{take_profit:.0%}止盈")
            else:
                position["last_price"] = float(market.at[code, "close"])
                continue
            traded_gross += float(position["shares"] * execution)
            transaction_cost += fee
            changed = True

        for code, position in positions.items():
            if code not in market.index:
                continue
            close_price = float(market.at[code, "close"])
            pre_close = float(market.at[code, "pre_close"])
            if close_price <= pre_close:
                position["non_up_streak"] = int(position.get("non_up_streak", 0)) + 1
            else:
                position["non_up_streak"] = 0
            if int(position["non_up_streak"]) >= non_up_exit_days:
                position["pending_non_up_exit"] = True
            position["last_price"] = close_price

        if trade_date.date() in preholiday_dates:
            for code in list(positions):
                if code not in market.index:
                    continue
                position = positions[code]
                close_price = float(market.at[code, "close"])
                fee = sell(code, position, trade_date, close_price, "法定节假日前清仓")
                traded_gross += float(position["shares"] * close_price)
                transaction_cost += fee
                changed = True

        equity = cash + sum(
            float(position["shares"] * position["last_price"]) for position in positions.values()
        )
        net_return = equity / previous_equity - 1 if previous_equity else 0.0
        gross_return = (equity + transaction_cost) / previous_equity - 1 if previous_equity else 0.0
        daily_records.append({
            "trade_date": trade_date,
            "signal_date": pd.Timestamp(signal_date),
            "gross_return": gross_return,
            "net_return": net_return,
            "benchmark_return": float(benchmark_returns.get(trade_date, 0.0)),
            "turnover": traded_gross / previous_equity if previous_equity else 0.0,
            "transaction_cost": transaction_cost / previous_equity if previous_equity else 0.0,
            "holdings": len(positions),
            "rebalanced": changed,
            "cash": cash,
            "equity_value": equity,
            "limit_up_buy_blocked": 0,
            "limit_down_sell_blocked": blocked_today,
        })
        previous_equity = equity

    result = pd.DataFrame(daily_records)
    result["equity"] = result["equity_value"] / float(initial_cash)
    result["benchmark_equity"] = (1 + result["benchmark_return"]).cumprod()
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1
    final_date = dates[-1]
    for code, position in positions.items():
        price = float(position["last_price"])
        trades.append({
            "strategy_id": spec.strategy_id,
            "trade_id": position["trade_id"],
            "ts_code": code,
            "name": names.get(code, ""),
            "entry_date": position["entry_date"],
            "exit_date": final_date.date(),
            "holding_period": f"{position['entry_date']} 至 {final_date.date()}",
            "holding_trading_days": int(position["holding_days"]),
            "entry_price": float(position["entry_price"]),
            "entry_price_unadjusted": float(position["entry_price_unadjusted"]),
            "exit_price": price,
            "price_change": float(price / position["entry_price"] - 1),
            "net_return": float(position["shares"] * price / position["cost_basis"] - 1),
            "max_gain": float(position["max_price"] / position["entry_price"] - 1),
            "max_drawdown_during_holding": float(position["min_price"] / position["entry_price"] - 1),
            "exit_reason": "期末仍持有",
            "crossed20_date": None,
            "status": "持有中",
        })
    trades_frame = pd.DataFrame(trades)
    closed = trades_frame[trades_frame.get("status", pd.Series(dtype=str)).eq("已平仓")]
    profitable = closed["net_return"].gt(0) if "net_return" in closed else pd.Series(dtype=bool)
    losing = closed["net_return"].lt(0) if "net_return" in closed else pd.Series(dtype=bool)
    metrics = calculate_metrics(result)
    metrics.update({
        "strategy_id": spec.strategy_id,
        "strategy_name": spec.name,
        "strategy_description": spec.description,
        "initial_cash": float(initial_cash),
        "take_profit_threshold": take_profit,
        "stop_loss_threshold": stop_loss,
        "non_up_exit_days": non_up_exit_days,
        "threshold_enabled": True,
        "max_positions": spec.max_positions,
        "completed_trades": len(closed),
        "total_trade_count": len(closed),
        "profitable_trade_count": int(profitable.sum()),
        "losing_trade_count": int(losing.sum()),
        "flat_trade_count": int(len(closed) - profitable.sum() - losing.sum()),
        "profitable_trade_rate": float(profitable.mean()) if not profitable.empty else 0.0,
        "average_holding_days": float(closed["holding_trading_days"].mean()) if not closed.empty else 0.0,
        "average_winner_return": float(closed.loc[profitable, "net_return"].mean()) if profitable.any() else 0.0,
        "average_loser_return": float(closed.loc[losing, "net_return"].mean()) if losing.any() else 0.0,
        "take_profit_count": int(closed["exit_reason"].str.contains("止盈").sum()) if not closed.empty else 0,
        "stop_loss_count": int(closed["exit_reason"].str.contains("止损").sum()) if not closed.empty else 0,
        "non_up_exit_count": int(closed["exit_reason"].str.contains("未上涨").sum()) if not closed.empty else 0,
        "preholiday_exit_count": int(closed["exit_reason"].eq("法定节假日前清仓").sum()) if not closed.empty else 0,
        "open_positions": len(positions),
        "limit_up_buy_blocked_count": blocked_buys,
        "limit_down_sell_blocked_count": blocked_sells_total,
        "corporate_action_mode": "reinvest_proxy",
        "corporate_action_adjustment_count": corporate_action_adjustments,
        "take_profit_model": f"日内最高价达到{take_profit:.0%}止盈；若同日同时触发止损和止盈，保守按止损优先",
        "cost_model": dict(costs),
        "holiday_exit_dates": sorted(value.isoformat() for value in preholiday_dates),
    })
    return result, trades_frame, pd.DataFrame(), metrics
