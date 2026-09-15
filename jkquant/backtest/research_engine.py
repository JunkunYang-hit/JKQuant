"""Conservative, integer-share backtests for the strategy research workflow.

Signals and risk decisions are made from completed sessions only.  Intraday
high/low prices are deliberately not used to infer an executable order path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .metrics import calculate_metrics
from .trading_rules import (
    affordable_buy_notional, is_open_limit_down, is_open_limit_up, transaction_fee,
)


@dataclass(frozen=True)
class ResearchSpec:
    strategy_id: str
    name: str
    description: str
    max_positions: int = 10
    rebalance_days: int = 5
    entry_rank: int = 20
    exit_rank: int = 40
    exit_confirmation_days: int = 2
    max_weight: float = 0.10
    gross_exposure: float = 0.80
    stop_loss: float | None = 0.08
    trailing_stop: float | None = 0.12
    take_profit: float | None = None
    use_market_regime: bool = True
    market_ma_days: int = 60
    max_per_industry: int | None = 2
    corporate_action_mode: str = "raw"

    def __post_init__(self) -> None:
        if min(self.max_positions, self.rebalance_days, self.entry_rank,
               self.exit_rank, self.exit_confirmation_days, self.market_ma_days) < 1:
            raise ValueError("持仓数、排名和交易日参数必须为正整数")
        if not 0 < self.max_weight <= 1 or not 0 <= self.gross_exposure <= 1:
            raise ValueError("单股上限与总仓位必须在0～1之间")
        if self.exit_rank < self.entry_rank:
            raise ValueError("退出排名不能小于买入排名")
        for value in (self.stop_loss, self.trailing_stop):
            if value is not None and not 0 < value < 1:
                raise ValueError("止损和移动止盈比例必须在0～1之间")
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("止盈比例必须为正")
        if self.max_per_industry is not None and self.max_per_industry < 1:
            raise ValueError("行业持仓上限必须为正整数")
        if self.corporate_action_mode not in {"raw", "reinvest_proxy"}:
            raise ValueError("除权口径必须为raw或reinvest_proxy")


TRADE_COLUMNS = [
    "strategy_id", "trade_id", "ts_code", "name", "signal_date", "entry_date",
    "exit_date", "holding_period", "holding_trading_days", "entry_rank",
    "entry_price", "entry_price_unadjusted", "exit_price", "shares", "initial_shares",
    "cost_basis", "buy_fee", "sell_fee",
    "fees", "price_change", "net_return", "net_pnl", "max_gain",
    "max_drawdown_during_holding", "exit_reason", "crossed20_date", "status",
]
EVENT_COLUMNS = [
    "strategy_id", "trade_id", "event", "ts_code", "name", "entry_date",
    "event_date", "holding_period", "holding_trading_days", "entry_price",
    "trigger_price", "day_high", "price_change", "final_exit_date",
    "final_exit_price", "final_price_change", "final_net_return",
    "final_exit_reason", "final_status",
]


def _capped_inverse_volatility(
    candidates: list[tuple[str, float]], budget: float, single_cap: float,
) -> dict[str, float]:
    """Allocate a monetary budget with water-filled, capped inverse vol weights."""
    pending = {code: 1 / max(vol, 0.05) for code, vol in candidates}
    allocations: dict[str, float] = {}
    remaining = min(max(budget, 0.0), len(pending) * single_cap)
    while pending and remaining > 1e-8:
        denominator = sum(pending.values())
        proposal = {code: remaining * weight / denominator for code, weight in pending.items()}
        capped = [code for code, value in proposal.items() if value > single_cap + 1e-8]
        if not capped:
            allocations.update(proposal)
            break
        for code in capped:
            allocations[code] = single_cap
            remaining -= single_cap
            del pending[code]
    return allocations


def run_research_backtest(
    daily: pd.DataFrame, signals: pd.DataFrame, benchmark: pd.DataFrame,
    names: dict[str, str], spec: ResearchSpec, start_date: date, end_date: date,
    initial_cash: float, costs: dict[str, float], limit_daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run a long-only, buffered portfolio using T close signals at T+1 open.

    ``daily`` and ``signals`` should include warmup sessions before start_date.
    ``signals`` contains trade_date, ts_code, rank, annualized volatility and an
    optional eligible boolean.  Low-volatility allocation controls entry size;
    appreciating positions are not repeatedly trimmed to their initial weights.
    A weak market permits reductions only, with half the normal exposure cap.
    Prices must share one consistent basis; raw prices do not include dividends.
    """
    if initial_cash <= 0:
        raise ValueError("初始资金必须为正")
    data = daily.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    if data.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("日线行情存在重复的股票/日期")
    if limit_daily is not None and not limit_daily.empty:
        limits = limit_daily[["trade_date", "ts_code", "up_limit", "down_limit"]].copy()
        limits["trade_date"] = pd.to_datetime(limits["trade_date"]).dt.normalize()
        data = data.drop(columns=["up_limit", "down_limit"], errors="ignore").merge(
            limits, on=["trade_date", "ts_code"], how="left", validate="one_to_one",
        )
    signal_data = signals.copy()
    signal_data["trade_date"] = pd.to_datetime(signal_data["trade_date"]).dt.normalize()
    if signal_data.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("信号存在重复的股票/日期")
    if "eligible" not in signal_data:
        signal_data["eligible"] = True
    if "volatility" not in signal_data:
        signal_data["volatility"] = 0.30
    signal_data["volatility"] = pd.to_numeric(signal_data["volatility"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan,
    ).fillna(0.30).clip(lower=0.05)
    snapshots = {
        pd.Timestamp(day): group.set_index("ts_code").to_dict("index")
        for day, group in signal_data.groupby("trade_date", sort=False)
    }
    benchmark_data = benchmark.copy() if benchmark is not None else pd.DataFrame()
    if not benchmark_data.empty:
        benchmark_data["trade_date"] = pd.to_datetime(benchmark_data["trade_date"]).dt.normalize()
        benchmark_data = benchmark_data.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        benchmark_data["market_average"] = benchmark_data["close"].rolling(spec.market_ma_days).mean()
        benchmark_rows = benchmark_data.set_index("trade_date").to_dict("index")
    else:
        benchmark_rows = {}
    calendar = sorted(set(data["trade_date"]) | set(benchmark_rows))
    previous_dates = {day: calendar[index - 1] if index else None for index, day in enumerate(calendar)}
    dates = [day for day in calendar if pd.Timestamp(start_date) <= day <= pd.Timestamp(end_date)]
    if not dates:
        raise ValueError("回测区间没有交易日")
    # Only signaled securities can be held; retain their warmup prices and suspensions.
    data = data[data["ts_code"].isin(signal_data["ts_code"].unique())]
    markets = {pd.Timestamp(day): frame.set_index("ts_code") for day, frame in data.groupby("trade_date", sort=False)}
    buy_commission = float(costs.get("commission_buy", 0.0005))
    sell_commission = float(costs.get("commission_sell", 0.0005))
    min_commission = float(costs.get("min_commission", 5.0))
    stamp_tax = float(costs.get("stamp_tax", 0.0005))
    slippage = float(costs.get("slippage", 0.001))
    cash = previous_equity = float(initial_cash)
    benchmark_equity = 1.0
    benchmark_previous_close: float | None = None
    benchmark_started = False
    missing_benchmark_days = missing_signal_days = corporate_actions = adjusted_actions = 0
    positions: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    sequence = 0

    def market_row(market: pd.DataFrame, code: str) -> pd.Series | None:
        return market.loc[code] if code in market.index else None

    def valid_price(value: Any) -> bool:
        return value is not None and pd.notna(value) and np.isfinite(float(value)) and float(value) > 0

    def executable(row: pd.Series | None) -> bool:
        # Daily final volume is not observable at the opening auction.
        return row is not None and valid_price(row.get("open"))

    def position_record(code: str, position: dict[str, Any], trade_date: pd.Timestamp,
                        price: float, fee: float, reason: str, status: str) -> dict[str, Any]:
        gross = position["shares"] * price
        net_pnl = gross - fee - position["cost_basis"]
        return {
            "strategy_id": spec.strategy_id, "trade_id": position["trade_id"], "ts_code": code,
            "name": names.get(code, ""), "signal_date": position["signal_date"],
            "entry_date": position["entry_date"], "exit_date": trade_date.date(),
            "holding_period": f"{position['entry_date']} 至 {trade_date.date()}",
            "holding_trading_days": position["holding_days"], "entry_rank": position["entry_rank"],
            "entry_price": position["entry_price"], "exit_price": price,
            "entry_price_unadjusted": position["entry_price_unadjusted"],
            "shares": position["shares"], "initial_shares": position["initial_shares"],
            "cost_basis": position["cost_basis"],
            "buy_fee": position["buy_fee"], "sell_fee": fee, "fees": position["buy_fee"] + fee,
            "price_change": price / position["entry_price"] - 1,
            "net_return": net_pnl / position["cost_basis"], "net_pnl": net_pnl,
            "max_gain": position["max_price"] / position["entry_price"] - 1,
            "max_drawdown_during_holding": position["min_price"] / position["entry_price"] - 1,
            "exit_reason": reason, "crossed20_date": position["threshold_date"], "status": status,
        }

    for day_index, trade_date in enumerate(dates):
        market = markets.get(trade_date, pd.DataFrame())
        signal_date = previous_dates[trade_date]
        snapshot = snapshots.get(signal_date)
        if snapshot is None:
            missing_signal_days += 1
        ranks = {code: float(values["rank"]) for code, values in (snapshot or {}).items()}
        benchmark_signal = benchmark_rows.get(signal_date, {})
        market_average = benchmark_signal.get("market_average")
        market_risk_on = not spec.use_market_regime or (
            valid_price(market_average) and valid_price(benchmark_signal.get("close"))
            and float(benchmark_signal["close"]) >= float(market_average)
        )
        exposure_cap = spec.gross_exposure * (1.0 if market_risk_on else 0.5)
        traded_gross = day_fees = 0.0
        blocked_buys = blocked_sells = suspended_buys = suspended_sells = 0
        sold_today: set[str] = set()
        entry_weights: list[float] = []

        # Risk triggers refer exclusively to the last completed close.  Once a
        # triggered exit is blocked, its order remains pending until executable.
        for code, position in positions.items():
            position["holding_days"] += 1
            row = market_row(market, code)
            if row is not None and valid_price(row.get("pre_close")):
                difference = abs(float(row["pre_close"]) - position["last_price"])
                if difference > max(0.0051, position["last_price"] * 0.00001):
                    corporate_actions += 1
                    if spec.corporate_action_mode == "reinvest_proxy":
                        ratio = position["last_price"] / float(row["pre_close"])
                        position["shares"] *= ratio
                        for field in ("entry_price", "last_price", "max_price", "min_price"):
                            position[field] /= ratio
                        adjusted_actions += 1
            if snapshot is not None:
                position["exit_days"] = position["exit_days"] + 1 if ranks.get(code, np.inf) > spec.exit_rank else 0
            prior_return = position["last_price"] / position["entry_price"] - 1
            prior_drawdown = position["last_price"] / position["max_price"] - 1
            reason = None
            if spec.stop_loss is not None and prior_return <= -spec.stop_loss:
                reason = f"收盘亏损达到{spec.stop_loss:.0%}止损"
            elif spec.trailing_stop is not None and prior_drawdown <= -spec.trailing_stop:
                reason = f"收盘较持仓收盘高点回撤{spec.trailing_stop:.0%}"
            elif spec.take_profit is not None and prior_return >= spec.take_profit:
                reason = f"收盘盈利达到{spec.take_profit:.0%}止盈"
            elif position["exit_days"] >= spec.exit_confirmation_days:
                reason = f"连续{spec.exit_confirmation_days}次跌出Top{spec.exit_rank}"
            if reason is not None and not position["pending_exit"]:
                position["pending_exit"] = reason

        def open_value(code: str) -> float:
            row = market_row(market, code)
            price = float(row["open"]) if row is not None and valid_price(row.get("open")) else positions[code]["last_price"]
            return positions[code]["shares"] * price

        opening_equity = cash + sum(open_value(code) for code in positions)

        def execute_sell(code: str, reason: str) -> bool:
            nonlocal cash, day_fees, traded_gross, blocked_sells, suspended_sells
            position = positions[code]
            if position["entry_date"] >= trade_date.date():
                return False
            row = market_row(market, code)
            if not executable(row):
                suspended_sells += 1
                return False
            price = float(row["open"])
            pre_close = float(row["pre_close"]) if valid_price(row.get("pre_close")) else position["last_price"]
            exact_limit = float(row["down_limit"]) if valid_price(row.get("down_limit")) else None
            if is_open_limit_down(code, price, pre_close, exact_limit):
                blocked_sells += 1
                return False
            gross = position["shares"] * price
            fee = transaction_fee(gross, sell_commission, min_commission=min_commission,
                                  stamp_tax_rate=stamp_tax, slippage_rate=slippage)
            cash += gross - fee
            day_fees += fee
            traded_gross += gross
            trades.append(position_record(code, position, trade_date, price, fee, reason, "已平仓"))
            del positions[code]
            sold_today.add(code)
            return True

        attempted_sells: set[str] = set()
        for code in list(positions):
            if positions[code]["pending_exit"]:
                attempted_sells.add(code)
                execute_sell(code, positions[code]["pending_exit"])
        # Risk-off does not top up existing holdings or open new ones.  Whole
        # positions, weakest rank first, are reduced to the lower exposure cap.
        if not market_risk_on:
            for code in sorted(positions, key=lambda item: (ranks.get(item, np.inf), item), reverse=True):
                held_value = sum(open_value(item) for item in positions)
                if held_value <= max(opening_equity - day_fees, 0) * exposure_cap + 1e-8:
                    break
                if code not in attempted_sells:
                    positions[code]["pending_exit"] = "沪深300ETF弱于均线，降低总仓位"
                    execute_sell(code, positions[code]["pending_exit"])

        # Anchor the schedule to the complete market calendar so independently
        # reported horizons do not silently use different rebalance weekdays.
        calendar_index = calendar.index(trade_date)
        if calendar_index % spec.rebalance_days == 0 and market_risk_on and snapshot is not None:
            candidates: list[tuple[str, float]] = []
            available_slots = max(0, spec.max_positions - len(positions))
            industry_counts: dict[str, int] = {}
            for position in positions.values():
                industry = position.get("industry", "")
                if industry:
                    industry_counts[industry] = industry_counts.get(industry, 0) + 1
            for code, values in sorted(snapshot.items(), key=lambda item: (item[1]["rank"], item[0])):
                if len(candidates) >= available_slots:
                    break
                if code in positions or code in sold_today or float(values["rank"]) > spec.entry_rank:
                    continue
                if pd.isna(values["eligible"]) or not bool(values["eligible"]):
                    continue
                if "ST" in str(names.get(code, "")).upper():
                    continue
                industry = str(values.get("industry", "")) if pd.notna(values.get("industry", "")) else ""
                if industry and spec.max_per_industry is not None and industry_counts.get(industry, 0) >= spec.max_per_industry:
                    continue
                row = market_row(market, code)
                if not executable(row):
                    suspended_buys += 1
                    continue
                price = float(row["open"])
                if not valid_price(row.get("pre_close")):
                    suspended_buys += 1
                    continue
                exact_limit = float(row["up_limit"]) if valid_price(row.get("up_limit")) else None
                if is_open_limit_up(code, price, float(row["pre_close"]), exact_limit):
                    blocked_buys += 1
                    continue
                candidates.append((code, float(values["volatility"])))
                if industry:
                    industry_counts[industry] = industry_counts.get(industry, 0) + 1
            held_value = sum(open_value(code) for code in positions)
            current_equity = cash + held_value
            budget = min(cash, max(0, current_equity * exposure_cap - held_value))
            allocations = _capped_inverse_volatility(candidates, budget, current_equity * spec.max_weight)
            for code, _ in candidates:
                price = float(market.at[code, "open"])
                allocation = min(cash, allocations.get(code, 0))
                gross_limit = affordable_buy_notional(allocation, buy_commission,
                                                     min_commission=min_commission, slippage_rate=slippage)
                shares = int(np.floor((gross_limit + 1e-8) / (price * 100))) * 100
                if shares <= 0:
                    continue
                gross = shares * price
                fee = transaction_fee(gross, buy_commission, min_commission=min_commission, slippage_rate=slippage)
                if gross + fee > cash + 1e-8:
                    raise AssertionError("买入超出可用现金")
                sequence += 1
                cash -= gross + fee
                day_fees += fee
                traded_gross += gross
                entry_weights.append(gross / current_equity)
                positions[code] = {
                    "trade_id": f"{spec.strategy_id}-{sequence:06d}", "shares": shares,
                    "initial_shares": shares, "entry_price_unadjusted": price,
                    "signal_date": signal_date.date(), "entry_date": trade_date.date(),
                    "entry_price": price, "cost_basis": gross + fee, "buy_fee": fee,
                    "holding_days": 1, "threshold_date": None, "last_price": price,
                    "max_price": price, "min_price": price, "entry_rank": int(ranks[code]),
                    "industry": str(snapshot[code].get("industry", "")) if pd.notna(snapshot[code].get("industry", "")) else "",
                    "exit_days": 0, "pending_exit": None,
                }

        # Closing marks update the next session's decisions, never today's fills.
        for code, position in positions.items():
            row = market_row(market, code)
            if row is not None and valid_price(row.get("close")):
                price = float(row["close"])
                position["last_price"] = price
                position["max_price"] = max(position["max_price"], price)
                position["min_price"] = min(position["min_price"], price)
                if position["threshold_date"] is None and price >= position["entry_price"] * 1.20:
                    position["threshold_date"] = trade_date.date().isoformat()
                    events.append({
                        "strategy_id": spec.strategy_id, "trade_id": position["trade_id"],
                        "event": "收盘盈利达到20%", "ts_code": code, "name": names.get(code, ""),
                        "entry_date": position["entry_date"], "event_date": trade_date.date(),
                        "holding_period": f"{position['entry_date']} 至 {trade_date.date()}",
                        "holding_trading_days": position["holding_days"],
                        "entry_price": position["entry_price"], "trigger_price": position["entry_price"] * 1.20,
                        "day_high": price, "price_change": price / position["entry_price"] - 1,
                    })
        held_value = sum(position["shares"] * position["last_price"] for position in positions.values())
        equity = cash + held_value
        benchmark_row = benchmark_rows.get(trade_date)
        benchmark_return = 0.0
        if benchmark_row is not None and valid_price(benchmark_row.get("close")):
            close = float(benchmark_row["close"])
            if not benchmark_started:
                denominator = benchmark_row.get("open", benchmark_row.get("pre_close", close))
                denominator = float(denominator) if valid_price(denominator) else close
                benchmark_started = True
            else:
                denominator = benchmark_previous_close
            benchmark_return = close / denominator - 1
            benchmark_previous_close = close
            benchmark_equity *= 1 + benchmark_return
        else:
            missing_benchmark_days += 1
        records.append({
            "trade_date": trade_date, "signal_date": pd.Timestamp(signal_date) if signal_date else pd.NaT,
            "gross_return": (equity + day_fees) / previous_equity - 1,
            "net_return": equity / previous_equity - 1, "benchmark_return": benchmark_return,
            "turnover": traded_gross / previous_equity, "transaction_cost": day_fees / previous_equity,
            "transaction_cost_cash": day_fees, "holdings": len(positions), "rebalanced": traded_gross > 0,
            "cash": cash, "equity_value": equity, "equity": equity / initial_cash,
            "benchmark_equity": benchmark_equity, "gross_exposure": held_value / equity if equity > 0 else 0,
            "market_risk_on": market_risk_on, "target_exposure": exposure_cap,
            "limit_up_buy_blocked": blocked_buys, "limit_down_sell_blocked": blocked_sells,
            "suspension_buy_blocked": suspended_buys, "suspension_sell_blocked": suspended_sells,
            "maximum_entry_weight": max(entry_weights, default=0),
            "held_codes": ",".join(sorted(positions)),
        })
        previous_equity = equity

    result = pd.DataFrame(records)
    # Include initial capital in the high water mark so first-day losses count.
    result["drawdown"] = result["equity"] / result["equity"].cummax().clip(lower=1.0) - 1
    for code, position in positions.items():
        trades.append(position_record(code, position, dates[-1], position["last_price"], 0.0, "期末仍持有", "持有中"))
    trades_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    events_frame = pd.DataFrame(events)
    if not events_frame.empty:
        outcomes = trades_frame[["trade_id", "exit_date", "exit_price", "price_change", "net_return", "exit_reason", "status"]].rename(columns={
            "exit_date": "final_exit_date", "exit_price": "final_exit_price", "price_change": "final_price_change",
            "net_return": "final_net_return", "exit_reason": "final_exit_reason", "status": "final_status",
        })
        events_frame = events_frame.merge(outcomes, on="trade_id", how="left", validate="many_to_one")
    events_frame = events_frame.reindex(columns=EVENT_COLUMNS)
    closed = trades_frame.loc[trades_frame["status"].eq("已平仓")]
    metrics = calculate_metrics(result)
    metrics.update({
        "strategy_id": spec.strategy_id, "strategy_name": spec.name, "strategy_description": spec.description,
        "research_spec": asdict(spec), "start_date": dates[0].date().isoformat(), "end_date": dates[-1].date().isoformat(),
        "initial_cash": initial_cash, "final_equity": previous_equity,
        "take_profit_threshold": spec.take_profit, "exit_confirmation_days": spec.exit_confirmation_days,
        "capital_fraction_per_entry": None, "max_positions": spec.max_positions,
        "take_profit_count": int(closed["exit_reason"].str.contains("止盈", na=False).sum()),
        "crossed_20_count": len(events_frame), "completed_trades": len(closed), "total_trade_count": len(closed),
        "profitable_trade_count": int(closed["net_return"].gt(0).sum()),
        "losing_trade_count": int(closed["net_return"].lt(0).sum()),
        "flat_trade_count": int(closed["net_return"].eq(0).sum()), "open_positions": len(positions),
        "average_holding_days": float(closed["holding_trading_days"].mean()) if len(closed) else 0.0,
        "profitable_trade_rate": float(closed["net_return"].gt(0).mean()) if len(closed) else 0.0,
        "average_winner_return": float(closed.loc[closed["net_return"].gt(0), "net_return"].mean()) if closed["net_return"].gt(0).any() else 0.0,
        "average_loser_return": float(closed.loc[closed["net_return"].lt(0), "net_return"].mean()) if closed["net_return"].lt(0).any() else 0.0,
        "limit_up_buy_blocked_count": int(result["limit_up_buy_blocked"].sum()),
        "limit_down_sell_blocked_count": int(result["limit_down_sell_blocked"].sum()),
        "suspension_buy_blocked_count": int(result["suspension_buy_blocked"].sum()),
        "suspension_sell_blocked_count": int(result["suspension_sell_blocked"].sum()),
        "total_transaction_cost_cash": float(result["transaction_cost_cash"].sum()),
        "average_gross_exposure": float(result["gross_exposure"].mean()),
        "missing_signal_days": missing_signal_days, "missing_benchmark_days": missing_benchmark_days,
        "unadjusted_corporate_action_discrepancy_count": corporate_actions,
        "corporate_action_adjustment_count": adjusted_actions,
        "corporate_action_mode": spec.corporate_action_mode,
        "corporate_action_model": "以昨实际收盘/今pre_close比值调整既有持仓份额与价格基准；包含再投资假设，非真实分红到账/股份登记" if spec.corporate_action_mode == "reinvest_proxy" else "原始价格，不补分红或送转股",
        "take_profit_model": "前收盘触发止损/止盈，下个交易日开盘成交，遵守T+1及涨跌停限制",
        "execution_model": "前一交易日收盘信号，次日开盘成交；100股整数手；停牌保留上次估值；不推断日内成交路径",
        "weight_model": "逆波动率分配、单股买入金额上限；缓冲区持仓不微调；弱市只减不加，整笔退出至半仓上限",
        "benchmark_model": "沪深300ETF区间首日开盘买入并持有；价格收益，不含分红与交易成本",
        "cost_model": dict(costs),
    })
    return result, trades_frame, events_frame, metrics
