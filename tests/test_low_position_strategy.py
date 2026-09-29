from datetime import date

import pandas as pd
import pytest

from jkquant.backtest.low_position_strategy import (
    LOW_POSITION_STRATEGIES,
    _statutory_preholiday_dates,
    run_low_position_strategy,
)


def test_smallcap_strategy_uses_next_open_and_ten_percent_stop() -> None:
    dates = pd.bdate_range("2026-07-06", periods=3)
    daily = pd.DataFrame({
        "trade_date": dates,
        "ts_code": "600001.SH",
        "open": [10.0, 10.0, 9.5],
        "high": [10.2, 10.2, 9.8],
        "low": [9.9, 9.0, 9.2],
        "close": [10.0, 9.5, 9.4],
        "pre_close": [10.0, 10.0, 9.5],
    })
    signals = pd.DataFrame([{
        "signal_date": dates[0], "signal_rank": 1,
        "ts_code": "600001.SH", "alert_score": 90.0,
    }])
    costs = {
        "commission_buy": 0.0, "commission_sell": 0.0,
        "min_commission": 0.0, "stamp_tax": 0.0, "slippage": 0.0,
    }
    result, trades, _, metrics = run_low_position_strategy(
        daily,
        signals,
        {"600001.SH": "测试股票"},
        LOW_POSITION_STRATEGIES[0],
        dates[0].date(),
        dates[-1].date(),
        10_000,
        costs,
    )
    assert trades.iloc[0]["entry_date"] == dates[1].date()
    assert trades.iloc[0]["exit_date"] == dates[1].date()
    assert trades.iloc[0]["exit_reason"] == "亏损达到10%止损"
    assert trades.iloc[0]["exit_price"] == pytest.approx(9.0)
    assert result.iloc[-1]["equity"] == pytest.approx(0.9)
    assert metrics["stop_loss_count"] == 1


def test_statutory_holiday_marks_last_trading_day() -> None:
    dates = [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-09")]
    assert date(2026, 9, 30) in _statutory_preholiday_dates(dates)


def test_two_non_up_closes_exit_at_following_open() -> None:
    dates = pd.bdate_range("2026-07-06", periods=5)
    daily = pd.DataFrame({
        "trade_date": dates,
        "ts_code": "600001.SH",
        "open": [10.0, 10.0, 9.9, 9.8, 9.8],
        "high": [10.1, 10.1, 10.0, 9.9, 9.9],
        "low": [9.9, 9.8, 9.7, 9.7, 9.7],
        "close": [10.0, 9.9, 9.8, 9.8, 9.8],
        "pre_close": [10.0, 10.0, 9.9, 9.8, 9.8],
    })
    signals = pd.DataFrame([{
        "signal_date": dates[0], "signal_rank": 1,
        "ts_code": "600001.SH", "alert_score": 90.0,
    }])
    costs = {
        "commission_buy": 0.0, "commission_sell": 0.0,
        "min_commission": 0.0, "stamp_tax": 0.0, "slippage": 0.0,
    }
    _, trades, _, metrics = run_low_position_strategy(
        daily, signals, {}, LOW_POSITION_STRATEGIES[0],
        dates[0].date(), dates[-1].date(), 10_000, costs,
        take_profit=.25, stop_loss=.10, non_up_exit_days=2,
    )
    assert trades.iloc[0]["exit_date"] == dates[3].date()
    assert trades.iloc[0]["exit_reason"] == "连续2日未上涨"
    assert metrics["non_up_exit_count"] == 1
