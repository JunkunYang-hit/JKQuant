from datetime import date

import pandas as pd
import pytest

from jkquant.backtest.strategy_suite import STRATEGIES, run_event_strategy


def strategy(strategy_id: str):
    return next(item for item in STRATEGIES if item.strategy_id == strategy_id)


def test_profit_threshold_is_recorded_and_sold() -> None:
    dates = pd.date_range("2025-09-01", periods=5, freq="B")
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ",
        "open": [100, 100, 100, 120, 100],
        "high": [105, 110, 101, 131, 105],
        "close": [100, 100, 100, 125, 102],
        "pre_close": [100, 100, 100, 100, 125],
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
        daily, rankings, {}, strategy("s03_top50_streak3_confirm2"), date(2025, 9, 1), date(2025, 9, 5),
        1_000_000, costs,
    )
    assert result.loc[result["trade_date"].lt("2025-09-04"), "holdings"].eq(0).all()
    assert result.loc[result["trade_date"].eq("2025-09-04"), "holdings"].iloc[0] == 1
    assert trades.iloc[0]["entry_date"] == date(2025, 9, 4)
    assert metrics["open_positions"] == 1


def test_ex_right_proxy_does_not_turn_split_into_a_loss() -> None:
    dates = pd.bdate_range("2025-09-01", periods=5)
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ",
        "open": [10., 10., 10., 5., 5.], "high": [10., 10., 10., 5., 5.],
        "low": [10., 10., 10., 5., 5.], "close": [10., 10., 10., 5., 5.],
        "pre_close": [10., 10., 10., 5., 5.],
    })
    rankings = pd.DataFrame({"trade_date": [value.date() for value in dates],
                             "rank": 1, "ts_code": "000001.SZ"})
    costs = {"commission_buy": 0, "commission_sell": 0, "stamp_tax": 0, "slippage": 0}
    result, trades, _, metrics = run_event_strategy(
        daily, rankings, {}, strategy("s04_top50_streak2"), dates[0].date(), dates[-1].date(),
        1_000_000, costs, take_profit=None, record_profit=None,
    )
    assert result.iloc[-1]["equity_value"] == pytest.approx(1_000_000)
    assert trades.iloc[0]["entry_price_unadjusted"] == 10
    assert trades.iloc[0]["entry_price"] == 5
    assert metrics["corporate_action_adjustment_count"] == 1


def test_active_strategies_are_registered() -> None:
    specs = {spec.strategy_id: spec for spec in STRATEGIES}
    assert len(specs) == 5
    assert set(specs) == {
        "s04_top50_streak2_confirm2", "s04_top50_streak2",
        "s14_top20_exact2_leader_full", "s03_top50_streak3_confirm2",
        "s06_top20_streak2_confirm2",
    }
    assert specs["s14_top20_exact2_leader_full"].exact_consecutive_days
    assert specs["s14_top20_exact2_leader_full"].fixed_take_profit == 0.32


def test_exact_two_day_leader_strategy_buys_only_best_new_streak() -> None:
    dates = pd.date_range("2025-09-01", periods=5, freq="B")
    codes = ["000001.SZ", "000002.SZ"]
    daily = pd.MultiIndex.from_product([dates, codes], names=["trade_date", "ts_code"]).to_frame(index=False)
    daily[["open", "high", "low", "close", "pre_close"]] = [10.0, 10.1, 9.9, 10.0, 10.0]
    rankings = pd.DataFrame([
        {"trade_date": day.date(), "ts_code": code, "rank": rank}
        for day, day_ranks in zip(dates, [(30, 2), (5, 2), (1, 2), (1, 2), (1, 2)])
        for code, rank in zip(codes, day_ranks)
    ])
    costs = {"commission_buy": 0, "commission_sell": 0, "stamp_tax": 0, "slippage": 0}
    result, trades, _, metrics = run_event_strategy(
        daily, rankings, {}, strategy("s14_top20_exact2_leader_full"),
        dates[2].date(), dates[-1].date(), 1_000_000, costs,
    )
    assert trades.iloc[0]["ts_code"] == codes[0]
    assert trades.iloc[0]["entry_date"] == dates[3].date()
    assert result.iloc[1]["cash"] == pytest.approx(0)
    assert metrics["open_positions"] == 1


def test_limit_up_blocks_buy() -> None:
    dates = pd.date_range("2025-09-01", periods=3, freq="B")
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ",
        "open": [10.0, 10.0, 11.0], "high": [10.0, 10.0, 11.0],
        "low": [10.0, 10.0, 11.0], "close": [10.0, 10.0, 11.0],
        "pre_close": [10.0, 10.0, 10.0],
    })
    rankings = pd.DataFrame({"trade_date": [value.date() for value in dates], "rank": 1, "ts_code": "000001.SZ"})
    costs = {"commission_buy": 0, "commission_sell": 0, "stamp_tax": 0, "slippage": 0}
    result, _, _, metrics = run_event_strategy(
        daily, rankings, {}, strategy("s04_top50_streak2"), dates[0].date(), dates[-1].date(), 1_000_000, costs,
    )
    assert result.iloc[1]["holdings"] == 0
    assert metrics["limit_up_buy_blocked_count"] == 1


def test_two_day_exit_confirmation_cancels_when_rank_recovers() -> None:
    dates = pd.date_range("2025-09-01", periods=7, freq="B")
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ", "open": 10.0, "high": 10.1,
        "low": 9.9, "close": 10.0, "pre_close": 10.0,
    })
    rankings = pd.DataFrame({
        "trade_date": [value.date() for value in dates], "rank": [1, 1, 21, 1, 21, 21, 21],
        "ts_code": "000001.SZ",
    })
    costs = {"commission_buy": 0, "commission_sell": 0, "stamp_tax": 0, "slippage": 0}
    _, trades, _, _ = run_event_strategy(
        daily, rankings, {}, strategy("s06_top20_streak2_confirm2"), dates[0].date(), dates[-1].date(), 1_000_000, costs,
    )
    closed = trades[trades["status"].eq("已平仓")]
    assert len(closed) == 1
    assert closed.iloc[0]["exit_date"] == dates[-1].date()


def test_limit_down_delays_sell_until_next_tradable_open() -> None:
    dates = pd.date_range("2025-09-01", periods=6, freq="B")
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ",
        "open": [10.0, 10.0, 10.0, 10.0, 9.0, 9.2],
        "high": [10.1, 10.1, 10.1, 10.1, 9.0, 9.3],
        "low": [9.9, 9.9, 9.9, 9.9, 9.0, 9.1],
        "close": [10.0, 10.0, 10.0, 10.0, 9.0, 9.2],
        "pre_close": [10.0, 10.0, 10.0, 10.0, 10.0, 9.0],
    })
    rankings = pd.DataFrame({
        "trade_date": [value.date() for value in dates], "rank": [1, 1, 21, 21, 21, 21],
        "ts_code": "000001.SZ",
    })
    costs = {"commission_buy": 0, "commission_sell": 0, "stamp_tax": 0, "slippage": 0}
    _, trades, _, metrics = run_event_strategy(
        daily, rankings, {}, strategy("s06_top20_streak2_confirm2"), dates[0].date(), dates[-1].date(),
        1_000_000, costs, take_profit=None, record_profit=None,
    )
    closed = trades[trades["status"].eq("已平仓")]
    assert metrics["limit_down_sell_blocked_count"] == 1
    assert closed.iloc[0]["exit_date"] == dates[-1].date()
