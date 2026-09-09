from datetime import date

import pandas as pd

from jkquant.backtest.strategy_suite import STRATEGIES, run_event_strategy


def test_profit_threshold_is_recorded_and_sold() -> None:
    dates = pd.date_range("2025-09-01", periods=5, freq="B")
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ",
        "open": [100, 100, 110, 120, 100],
        "high": [105, 110, 121, 131, 105],
        "close": [100, 108, 120, 125, 102],
        "pre_close": [100, 100, 108, 120, 125],
    })
    rankings = pd.DataFrame({
        "trade_date": [value.date() for value in dates],
        "rank": 1, "ts_code": "000001.SZ",
    })
    costs = {"commission_buy": 0, "commission_sell": 0, "stamp_tax": 0, "slippage": 0}
    result, trades, events, metrics = run_event_strategy(
        daily, rankings, {"000001.SZ": "测试股票"}, STRATEGIES[0],
        date(2025, 9, 1), date(2025, 9, 5), 1_000_000, costs,
    )
    closed = trades[trades["status"].eq("已平仓")]
    assert len(events) == 1
    assert events.iloc[0]["event"] == "盈利达到20%"
    assert len(closed) == 1
    assert closed.iloc[0]["exit_reason"] == "盈利达到30%止盈"
    assert closed.iloc[0]["exit_price"] == 130
    assert metrics["profit_take_30_count"] == 1
    assert not result.empty


def test_consecutive_strategy_waits_for_three_signals() -> None:
    dates = pd.date_range("2025-09-01", periods=5, freq="B")
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ",
        "open": 100.0, "high": 101.0, "close": 100.0, "pre_close": 100.0,
    })
    rankings = pd.DataFrame({
        "trade_date": [value.date() for value in dates],
        "rank": 1, "ts_code": "000001.SZ",
    })
    costs = {"commission_buy": 0, "commission_sell": 0, "stamp_tax": 0, "slippage": 0}
    result, trades, _, metrics = run_event_strategy(
        daily, rankings, {}, STRATEGIES[4], date(2025, 9, 1), date(2025, 9, 5),
        1_000_000, costs,
    )
    assert result.loc[result["trade_date"].lt("2025-09-04"), "holdings"].eq(0).all()
    assert result.loc[result["trade_date"].eq("2025-09-04"), "holdings"].iloc[0] == 1
    assert trades.iloc[0]["entry_date"] == date(2025, 9, 4)
    assert metrics["open_positions"] == 1
