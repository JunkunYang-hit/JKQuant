from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import calculate_metrics
from .benchmark import benchmark_return_map
from .trading_rules import is_open_limit_down, is_open_limit_up


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    name: str
    description: str
    entry_rank: int
    exit_rank: int
    consecutive_rank: int | None
    consecutive_days: int
    weighting: str
    exit_confirmation_days: int = 1
    fallback_entry_rank: int | None = None
    capital_fraction_per_entry: float | None = None
    max_positions: int | None = None
    fixed_take_profit: float | None = None


STRATEGIES = [
    StrategySpec("s01_top10_exit20", "Top10比例买入，跌出Top20卖出", "按排名线性加权买入Top10，跌出Top20卖出。", 10, 20, None, 1, "rank_linear"),
    StrategySpec("s02_top5_exit20_confirm2", "Top5比例买入，跌出Top20卖出（两日确认）", "按排名线性加权买入Top5；首次跌出暂缓，连续两个信号日跌出Top20才卖出。", 5, 20, None, 1, "rank_linear", 2),
    StrategySpec("s03_top50_streak3", "连续3次Top50，跌出Top50卖出", "连续3个交易日进入Top50后等权买入，跌出Top50卖出。", 50, 50, 50, 3, "equal"),
    StrategySpec("s03_top50_streak3_confirm2", "连续3次Top50，跌出Top50卖出（两日确认）", "连续3个交易日进入Top50后等权买入；连续两个信号日跌出Top50才卖出。", 50, 50, 50, 3, "equal", 2),
    StrategySpec("s04_top50_streak2", "连续2次Top50，跌出Top50卖出", "连续2个交易日进入Top50后等权买入，跌出Top50卖出。", 50, 50, 50, 2, "equal"),
    StrategySpec("s04_top50_streak2_confirm2", "连续2次Top50，跌出Top50卖出（两日确认）", "连续2个交易日进入Top50后等权买入；连续两个信号日跌出Top50才卖出。", 50, 50, 50, 2, "equal", 2),
    StrategySpec("s05_top20_streak3", "连续3次Top20，跌出Top20卖出", "连续3个交易日进入Top20后等权买入，跌出Top20卖出。", 20, 20, 20, 3, "equal"),
    StrategySpec("s05_top20_streak3_confirm2", "连续3次Top20，跌出Top20卖出（两日确认）", "连续3个交易日进入Top20后等权买入；连续两个信号日跌出Top20才卖出。", 20, 20, 20, 3, "equal", 2),
    StrategySpec("s06_top20_streak2_confirm2", "连续2次Top20，跌出Top20卖出（两日确认）", "连续2个交易日进入Top20后等权买入；连续两个信号日跌出Top20才卖出。", 20, 20, 20, 2, "equal", 2),
    StrategySpec("s07_top10_streak2_exit20", "Top10且连续2次Top20，跌出Top20卖出", "按排名线性加权买入Top10且连续2次进入Top20的股票，跌出Top20卖出。", 10, 20, 20, 2, "rank_linear"),
    StrategySpec("s08_top5_streak2_exit20_confirm2", "Top5且连续2次Top20，跌出Top20卖出（两日确认）", "按排名线性加权买入Top5且连续2次进入Top20的股票；连续两个信号日跌出Top20才卖出。", 5, 20, 20, 2, "rank_linear", 2),
    StrategySpec("s10_top5_equal_exit10_confirm2", "当日Top5等权，跌出Top10卖出（两日确认）", "等权买入当日Top5；连续两个信号日跌出Top10才卖出。", 5, 10, None, 1, "equal", 2),
    StrategySpec("s12_top1_streak3_half", "Top1且连续3次Top20，半仓买入", "只买当日Top1且连续3次进入Top20的股票；每次使用账户权益50%，最多持有2只；跌出Top20或盈利30%卖出。", 1, 20, 20, 3, "equal", 1, None, 0.5, 2, 0.30),
    StrategySpec("s13_top1_fallback2_streak3_half", "Top1优先、Top2回退且连续3次Top20，半仓买入", "优先买当日Top1且连续3次进入Top20的股票；无可买Top1时检查Top2；每次使用账户权益50%，最多持有2只；跌出Top20或盈利30%卖出。", 1, 20, 20, 3, "equal", 1, 2, 0.5, 2, 0.30),
]
BASE_STRATEGIES = [spec for spec in STRATEGIES if not spec.strategy_id.endswith("_confirm2") and spec.fixed_take_profit is None]


def _entry_weights(candidates: list[tuple[str, int]], spec: StrategySpec) -> dict[str, float]:
    if not candidates:
        return {}
    if spec.weighting == "equal":
        return {code: 1 / len(candidates) for code, _ in candidates}
    raw = {code: float(spec.entry_rank + 1 - rank) for code, rank in candidates}
    total = sum(raw.values())
    return {code: value / total for code, value in raw.items()}


def _signal_snapshots(rankings: pd.DataFrame, spec: StrategySpec) -> dict[date, dict[str, Any]]:
    required_streaks: dict[str, int] = {}
    snapshots: dict[date, dict[str, Any]] = {}
    for trade_date, day in rankings.sort_values(["trade_date", "rank"]).groupby("trade_date"):
        ranks = day.set_index("ts_code")["rank"].astype(int).to_dict()
        if spec.consecutive_rank is None:
            required_streaks = {}
        else:
            current = {code for code, rank in ranks.items() if rank <= spec.consecutive_rank}
            required_streaks = {code: required_streaks.get(code, 0) + 1 for code in current}
        snapshots[pd.Timestamp(trade_date).date()] = {
            "ranks": ranks, "streaks": required_streaks.copy(),
        }
    return snapshots


def run_event_strategy(
    daily: pd.DataFrame, rankings: pd.DataFrame, names: dict[str, str], spec: StrategySpec,
    start_date: date, end_date: date, initial_cash: float, costs: dict[str, float],
    take_profit: float | None = 0.20, record_profit: float | None = 0.20,
    benchmark_daily: pd.DataFrame | None = None,
    limit_daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run one daily-signal/next-open strategy with opening price constraints."""
    if limit_daily is not None and not limit_daily.empty:
        limits = limit_daily[["trade_date", "ts_code", "up_limit", "down_limit"]].copy()
        daily = daily.merge(limits, on=["trade_date", "ts_code"], how="left")
    if spec.fixed_take_profit is not None:
        take_profit = spec.fixed_take_profit
        record_profit = spec.fixed_take_profit
    market_data = daily[daily["trade_date"].dt.date.between(start_date, end_date)].copy()
    dates = sorted(pd.Timestamp(value) for value in market_data["trade_date"].unique())
    if not dates:
        raise ValueError("回测区间没有日线数据")
    snapshots = _signal_snapshots(rankings, spec)
    cash = previous_equity = float(initial_cash)
    positions: dict[str, dict[str, Any]] = {}
    pending_exit: dict[str, int] = {}
    profit_blocked: set[str] = set()
    daily_records: list[dict[str, Any]] = []
    benchmark_returns = benchmark_return_map(benchmark_daily)
    trades: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    trade_sequence = 0
    buy_rate = float(costs["commission_buy"]) + float(costs["slippage"])
    sell_rate = float(costs["commission_sell"]) + float(costs["stamp_tax"]) + float(costs["slippage"])
    record_label = f"盈利达到{record_profit:.0%}" if record_profit is not None else ""
    take_profit_label = f"盈利达到{take_profit:.0%}止盈" if take_profit is not None else ""

    def record_threshold(code: str, position: dict[str, Any], event_date: pd.Timestamp, high: float) -> None:
        if record_profit is None or position["threshold_date"] is not None:
            return
        position["threshold_date"] = event_date.date().isoformat()
        events.append({
            "strategy_id": spec.strategy_id, "trade_id": position["trade_id"], "event": record_label,
            "ts_code": code, "name": names.get(code, ""), "entry_date": position["entry_date"],
            "event_date": event_date.date(), "holding_period": f"{position['entry_date']} 至 {event_date.date()}",
            "holding_trading_days": int(position["holding_days"]), "entry_price": float(position["entry_price"]),
            "trigger_price": float(position["entry_price"] * (1 + record_profit)), "day_high": high,
            "price_change": high / position["entry_price"] - 1,
        })

    def sell(code: str, position: dict[str, Any], trade_date: pd.Timestamp, price: float, reason: str) -> tuple[float, float]:
        nonlocal cash
        gross = float(position["shares"] * price)
        fee = gross * sell_rate
        cash += gross - fee
        net_return = (gross - fee) / float(position["cost_basis"]) - 1
        trades.append({
            "strategy_id": spec.strategy_id, "trade_id": position["trade_id"], "ts_code": code,
            "name": names.get(code, ""), "entry_date": position["entry_date"], "exit_date": trade_date.date(),
            "holding_trading_days": int(position["holding_days"]), "holding_period": f"{position['entry_date']} 至 {trade_date.date()}",
            "entry_price": float(position["entry_price"]), "exit_price": price,
            "price_change": price / float(position["entry_price"]) - 1, "net_return": net_return,
            "max_gain": position["max_price"] / float(position["entry_price"]) - 1,
            "max_drawdown_during_holding": position["min_price"] / float(position["entry_price"]) - 1,
            "exit_reason": reason, "crossed20_date": position["threshold_date"], "status": "已平仓",
        })
        del positions[code]
        pending_exit.pop(code, None)
        return gross, fee

    for index, trade_date in enumerate(dates):
        market = market_data[market_data["trade_date"].eq(trade_date)].set_index("ts_code")
        signal_date = dates[index - 1].date() if index > 0 else trade_date.date()
        signal = snapshots.get(signal_date, {"ranks": {}, "streaks": {}}) if index > 0 else {"ranks": {}, "streaks": {}}
        ranks: dict[str, int] = signal["ranks"]
        streaks: dict[str, int] = signal["streaks"]
        traded_gross = transaction_cost = 0.0
        blocked_buys = blocked_sells = 0
        changed = False

        for code in list(profit_blocked):
            if ranks.get(code, 51) > spec.exit_rank:
                profit_blocked.remove(code)
        for code, position in positions.items():
            position["holding_days"] += 1
            if code in market.index:
                open_price = float(market.at[code, "open"])
                position["max_price"] = max(position["max_price"], open_price)
                position["min_price"] = min(position["min_price"], open_price)

        if take_profit is not None:
            for code in list(positions):
                if code not in market.index or not np.isfinite(market.at[code, "open"]):
                    continue
                position = positions[code]
                open_price = float(market.at[code, "open"])
                if record_profit is not None and open_price >= position["entry_price"] * (1 + record_profit):
                    record_threshold(code, position, trade_date, open_price)
                if open_price >= position["entry_price"] * (1 + take_profit):
                    gross, fee = sell(code, position, trade_date, open_price, take_profit_label)
                    traded_gross += gross
                    transaction_cost += fee
                    profit_blocked.add(code)
                    changed = True

        for code in list(positions):
            if ranks.get(code, 51) <= spec.exit_rank:
                pending_exit.pop(code, None)
                continue
            pending_exit[code] = pending_exit.get(code, 0) + 1
            if pending_exit[code] < spec.exit_confirmation_days or code not in market.index:
                continue
            open_price = float(market.at[code, "open"])
            pre_close = float(market.at[code, "pre_close"])
            exact_down = float(market.at[code, "down_limit"]) if "down_limit" in market and pd.notna(market.at[code, "down_limit"]) else None
            if is_open_limit_down(code, open_price, pre_close, exact_down):
                blocked_sells += 1
                continue
            gross, fee = sell(code, positions[code], trade_date, open_price, f"连续{spec.exit_confirmation_days}次跌出Top{spec.exit_rank}")
            traded_gross += gross
            transaction_cost += fee
            changed = True

        candidates: list[tuple[str, int]] = []
        candidate_rank_limit = spec.fallback_entry_rank or spec.entry_rank
        for code, rank in sorted(ranks.items(), key=lambda item: item[1]):
            qualified_streak = spec.consecutive_rank is None or streaks.get(code, 0) >= spec.consecutive_days
            if not (rank <= candidate_rank_limit and qualified_streak and code not in positions and code not in profit_blocked and code in market.index):
                continue
            open_price = float(market.at[code, "open"])
            pre_close = float(market.at[code, "pre_close"])
            if open_price <= 0:
                continue
            exact_up = float(market.at[code, "up_limit"]) if "up_limit" in market and pd.notna(market.at[code, "up_limit"]) else None
            if is_open_limit_up(code, open_price, pre_close, exact_up):
                blocked_buys += 1
                continue
            candidates.append((code, rank))
        if spec.fallback_entry_rank is not None:
            candidates = candidates[:1]
        if spec.max_positions is not None:
            available_slots = max(0, spec.max_positions - len(positions))
            candidates = candidates[:available_slots]
        required_cash = (
            previous_equity * spec.capital_fraction_per_entry
            if spec.capital_fraction_per_entry is not None else previous_equity * 1e-8
        )
        if candidates and cash + 1e-8 >= required_cash:
            allocations = _entry_weights(candidates, spec)
            total_cash_budget = (
                required_cash if spec.capital_fraction_per_entry is not None else cash
            )
            gross_budget = total_cash_budget / (1 + buy_rate)
            for code, rank in candidates:
                trade_sequence += 1
                open_price = float(market.at[code, "open"])
                gross = gross_budget * allocations[code]
                fee = gross * buy_rate
                shares = gross / open_price
                cash -= gross + fee
                positions[code] = {
                    "trade_id": f"{spec.strategy_id}-{trade_sequence:06d}", "shares": shares,
                    "entry_date": trade_date.date(), "entry_price": open_price, "cost_basis": gross + fee,
                    "holding_days": 1, "threshold_date": None, "last_price": open_price,
                    "max_price": open_price, "min_price": open_price, "entry_rank": rank,
                }
                traded_gross += gross
                transaction_cost += fee
                changed = True
            cash = max(cash, 0.0)

        for code in list(positions):
            if code not in market.index:
                continue
            position = positions[code]
            high = float(market.at[code, "high"])
            low_price = float(market.at[code, "low"]) if "low" in market.columns else min(
                float(market.at[code, "open"]), float(market.at[code, "close"])
            )
            position["max_price"] = max(position["max_price"], high)
            position["min_price"] = min(position["min_price"], low_price)
            if record_profit is not None and high >= position["entry_price"] * (1 + record_profit):
                record_threshold(code, position, trade_date, high)
            if take_profit is not None and high >= position["entry_price"] * (1 + take_profit):
                execution = max(float(market.at[code, "open"]), position["entry_price"] * (1 + take_profit))
                gross, fee = sell(code, position, trade_date, execution, take_profit_label)
                traded_gross += gross
                transaction_cost += fee
                profit_blocked.add(code)
                changed = True
            elif code in positions:
                positions[code]["last_price"] = float(market.at[code, "close"])

        equity = cash + sum(float(position["shares"] * position["last_price"]) for position in positions.values())
        benchmark_return = float(benchmark_returns.get(trade_date, 0.0))
        net_return = equity / previous_equity - 1 if previous_equity else 0.0
        gross_return = (equity + transaction_cost) / previous_equity - 1 if previous_equity else 0.0
        daily_records.append({
            "trade_date": trade_date, "signal_date": pd.Timestamp(signal_date), "gross_return": gross_return,
            "net_return": net_return, "benchmark_return": benchmark_return,
            "turnover": traded_gross / previous_equity if previous_equity else 0.0,
            "transaction_cost": transaction_cost / previous_equity if previous_equity else 0.0,
            "holdings": len(positions), "rebalanced": changed, "cash": cash, "equity_value": equity,
            "limit_up_buy_blocked": blocked_buys, "limit_down_sell_blocked": blocked_sells,
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
            "strategy_id": spec.strategy_id, "trade_id": position["trade_id"], "ts_code": code,
            "name": names.get(code, ""), "entry_date": position["entry_date"], "exit_date": final_date.date(),
            "holding_period": f"{position['entry_date']} 至 {final_date.date()}",
            "holding_trading_days": int(position["holding_days"]), "entry_price": float(position["entry_price"]),
            "exit_price": price, "price_change": price / float(position["entry_price"]) - 1,
            "net_return": (position["shares"] * price) / float(position["cost_basis"]) - 1,
            "max_gain": position["max_price"] / float(position["entry_price"]) - 1,
            "max_drawdown_during_holding": position["min_price"] / float(position["entry_price"]) - 1,
            "exit_reason": "期末仍持有", "crossed20_date": position["threshold_date"], "status": "持有中",
        })
    trades_frame = pd.DataFrame(trades)
    events_frame = pd.DataFrame(events)
    if not events_frame.empty and not trades_frame.empty:
        outcomes = trades_frame[["trade_id", "exit_date", "exit_price", "price_change", "net_return", "exit_reason", "status"]].rename(columns={
            "exit_date": "final_exit_date", "exit_price": "final_exit_price", "price_change": "final_price_change",
            "net_return": "final_net_return", "exit_reason": "final_exit_reason", "status": "final_status",
        })
        events_frame = events_frame.merge(outcomes, on="trade_id", how="left")
    metrics = calculate_metrics(result)
    closed = trades_frame[trades_frame.get("status", pd.Series(dtype=str)).eq("已平仓")]
    metrics.update({
        "strategy_id": spec.strategy_id, "strategy_name": spec.name, "strategy_description": spec.description,
        "take_profit_threshold": take_profit, "exit_confirmation_days": spec.exit_confirmation_days,
        "capital_fraction_per_entry": spec.capital_fraction_per_entry,
        "max_positions": spec.max_positions,
        "take_profit_count": int(closed["exit_reason"].eq(take_profit_label).sum()) if not closed.empty else 0,
        "crossed_20_count": len(events_frame), "completed_trades": len(closed), "total_trade_count": len(closed),
        "profitable_trade_count": int(closed["net_return"].gt(0).sum()) if not closed.empty else 0,
        "losing_trade_count": int(closed["net_return"].lt(0).sum()) if not closed.empty else 0,
        "flat_trade_count": int(closed["net_return"].eq(0).sum()) if not closed.empty else 0,
        "open_positions": len(positions),
        "average_holding_days": float(closed["holding_trading_days"].mean()) if not closed.empty else 0.0,
        "profitable_trade_rate": float(closed["net_return"].gt(0).mean()) if not closed.empty else 0.0,
        "average_winner_return": float(closed.loc[closed["net_return"].gt(0), "net_return"].mean()) if closed["net_return"].gt(0).any() else 0.0,
        "average_loser_return": float(closed.loc[closed["net_return"].lt(0), "net_return"].mean()) if closed["net_return"].lt(0).any() else 0.0,
        "limit_up_buy_blocked_count": int(result["limit_up_buy_blocked"].sum()),
        "limit_down_sell_blocked_count": int(result["limit_down_sell_blocked"].sum()),
        "take_profit_model": "不设固定止盈" if take_profit is None else f"日内最高价触及{take_profit:.0%}时按目标价卖出；开盘跳空越过时按开盘价",
    })
    return result, trades_frame, events_frame, metrics


def write_strategy_result(output_dir: Path, daily: pd.DataFrame, trades: pd.DataFrame, events: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"daily": output_dir / "daily.csv", "trades": output_dir / "trades.csv", "events": output_dir / "threshold_events.csv", "metrics": output_dir / "metrics.json"}
    if events.empty and not len(events.columns):
        events = pd.DataFrame(columns=["strategy_id", "trade_id", "event", "ts_code", "name", "entry_date", "event_date", "holding_period", "holding_trading_days", "entry_price", "trigger_price", "day_high", "price_change", "final_exit_date", "final_exit_price", "final_price_change", "final_net_return", "final_exit_reason", "final_status"])
    daily.to_csv(paths["daily"], index=False, encoding="utf-8-sig", float_format="%.8f")
    trades.to_csv(paths["trades"], index=False, encoding="utf-8-sig", float_format="%.8f")
    events.to_csv(paths["events"], index=False, encoding="utf-8-sig", float_format="%.8f")
    paths["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def write_suite_index(output_dir: Path, start_date: date, end_date: date, summaries: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> Path:
    path = output_dir / "suite.json"
    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"), "start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "strategies": summaries}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
