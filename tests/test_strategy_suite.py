from datetime import date

import pandas as pd
import pytest

from jkquant.backtest.strategy_suite import BASE_STRATEGIES, STRATEGIES, run_event_strategy


def strategy(strategy_id: str):
    return next(item for item in STRATEGIES if item.strategy_id == strategy_id)


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
    assert closed.iloc[0]["exit_reason"] == "盈利达到20%止盈"
    assert closed.iloc[0]["exit_price"] == 120
    assert events.iloc[0]["final_exit_reason"] == "盈利达到20%止盈"
    assert events.iloc[0]["final_net_return"] == pytest.approx(0.2)
    assert metrics["take_profit_count"] == 1
    assert metrics["total_trade_count"] == 1
    assert metrics["profitable_trade_count"] == 1
    assert metrics["losing_trade_count"] == 0
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
        daily, rankings, {}, strategy("s05_top20_streak3"), date(2025, 9, 1), date(2025, 9, 5),
        1_000_000, costs,
    )
    assert result.loc[result["trade_date"].lt("2025-09-04"), "holdings"].eq(0).all()
    assert result.loc[result["trade_date"].eq("2025-09-04"), "holdings"].iloc[0] == 1
    assert trades.iloc[0]["entry_date"] == date(2025, 9, 4)
    assert metrics["open_positions"] == 1


def test_active_strategies_are_registered() -> None:
    specs = {spec.strategy_id: spec for spec in STRATEGIES}
    assert len(specs) == 14
    assert len(BASE_STRATEGIES) == 5
    assert "s11_top5_streak2_exit10_confirm2" not in specs
    assert specs["s12_top1_streak3_half"].capital_fraction_per_entry == 0.5
    assert specs["s12_top1_streak3_half"].fixed_take_profit == 0.30
    assert specs["s13_top1_fallback2_streak3_half"].fallback_entry_rank == 2


def test_limit_up_blocks_buy() -> None:
    dates = pd.date_range("2025-09-01", periods=3, freq="B")
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ",
        "open": [10.0, 11.0, 10.0], "high": [10.0, 11.0, 10.1],
        "low": [10.0, 11.0, 9.9], "close": [10.0, 11.0, 10.0],
        "pre_close": [10.0, 10.0, 11.0],
    })
    rankings = pd.DataFrame({"trade_date": [value.date() for value in dates], "rank": 1, "ts_code": "000001.SZ"})
    costs = {"commission_buy": 0, "commission_sell": 0, "stamp_tax": 0, "slippage": 0}
    result, _, _, metrics = run_event_strategy(
        daily, rankings, {}, BASE_STRATEGIES[0], dates[0].date(), dates[-1].date(), 1_000_000, costs,
    )
    assert result.iloc[1]["holdings"] == 0
    assert metrics["limit_up_buy_blocked_count"] == 1


def test_two_day_exit_confirmation_cancels_when_rank_recovers() -> None:
    dates = pd.date_range("2025-09-01", periods=6, freq="B")
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ", "open": 10.0, "high": 10.1,
        "low": 9.9, "close": 10.0, "pre_close": 10.0,
    })
    rankings = pd.DataFrame({
        "trade_date": [value.date() for value in dates], "rank": [1, 21, 1, 21, 21, 21],
        "ts_code": "000001.SZ",
    })
    costs = {"commission_buy": 0, "commission_sell": 0, "stamp_tax": 0, "slippage": 0}
    _, trades, _, _ = run_event_strategy(
        daily, rankings, {}, strategy("s02_top5_exit20_confirm2"), dates[0].date(), dates[-1].date(), 1_000_000, costs,
    )
    closed = trades[trades["status"].eq("已平仓")]
    assert len(closed) == 1
    assert closed.iloc[0]["exit_date"] == dates[-1].date()


def test_limit_down_delays_sell_until_next_tradable_open() -> None:
    dates = pd.date_range("2025-09-01", periods=4, freq="B")
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ",
        "open": [10.0, 10.0, 9.0, 9.2], "high": [10.1, 10.1, 9.0, 9.3],
        "low": [9.9, 9.9, 9.0, 9.1], "close": [10.0, 10.0, 9.0, 9.2],
        "pre_close": [10.0, 10.0, 10.0, 9.0],
    })
    rankings = pd.DataFrame({
        "trade_date": [value.date() for value in dates], "rank": [1, 21, 21, 21],
        "ts_code": "000001.SZ",
    })
    costs = {"commission_buy": 0, "commission_sell": 0, "stamp_tax": 0, "slippage": 0}
    _, trades, _, metrics = run_event_strategy(
        daily, rankings, {}, BASE_STRATEGIES[0], dates[0].date(), dates[-1].date(),
        1_000_000, costs, take_profit=None, record_profit=None,
    )
    closed = trades[trades["status"].eq("已平仓")]
    assert metrics["limit_down_sell_blocked_count"] == 1
    assert closed.iloc[0]["exit_date"] == dates[-1].date()


def test_fallback_strategy_buys_qualified_top2_with_half_equity() -> None:
    dates = pd.date_range("2025-09-01", periods=4, freq="B")
    codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    daily = pd.MultiIndex.from_product([dates, codes], names=["trade_date", "ts_code"]).to_frame(index=False)
    daily[["open", "high", "low", "close", "pre_close"]] = [10.0, 10.1, 9.9, 10.0, 10.0]
    rankings = pd.DataFrame({
        "trade_date": [dates[0].date(), dates[0].date(), dates[1].date(), dates[1].date(),
                       dates[2].date(), dates[2].date(), dates[3].date(), dates[3].date()],
        "ts_code": [codes[0], codes[1], codes[2], codes[1], codes[3], codes[1], codes[0], codes[1]],
        "rank": [1, 2, 1, 2, 1, 2, 1, 2],
    })
    costs = {"commission_buy": 0, "commission_sell": 0, "stamp_tax": 0, "slippage": 0}
    result, trades, _, metrics = run_event_strategy(
        daily, rankings, {}, strategy("s13_top1_fallback2_streak3_half"),
        dates[0].date(), dates[-1].date(), 1_000_000, costs,
    )
    assert result.iloc[-1]["holdings"] == 1
    assert result.iloc[-1]["cash"] == pytest.approx(500_000)
    assert trades.iloc[0]["ts_code"] == codes[1]
    assert metrics["take_profit_threshold"] == 0.30
