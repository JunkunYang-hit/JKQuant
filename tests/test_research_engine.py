from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from jkquant.backtest.research_engine import ResearchSpec, run_research_backtest


ZERO_COSTS = {
    "commission_buy": 0, "commission_sell": 0, "min_commission": 0,
    "stamp_tax": 0, "slippage": 0,
}


def _spec(**changes) -> ResearchSpec:
    base = ResearchSpec(
        "research", "测试", "测试", max_positions=1, rebalance_days=1,
        entry_rank=1, exit_rank=2, max_weight=1, gross_exposure=1,
        stop_loss=None, trailing_stop=None, use_market_regime=False,
        max_per_industry=None,
    )
    return replace(base, **changes)


def _benchmark(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"trade_date": dates, "open": 100.0, "close": 100.0})


def test_research_engine_uses_previous_signal_and_integer_board_lots() -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    codes = ["000001.SZ", "000002.SZ"]
    daily = pd.MultiIndex.from_product([dates, codes], names=["trade_date", "ts_code"]).to_frame(index=False)
    daily[["open", "high", "low", "close", "pre_close"]] = [10.0, 10.0, 10.0, 10.0, 10.0]
    signals = pd.DataFrame([
        {"trade_date": dates[0], "ts_code": codes[0], "rank": 1, "volatility": .2, "industry": "银行"},
        {"trade_date": dates[1], "ts_code": codes[1], "rank": 1, "volatility": .2, "industry": "消费"},
    ])
    result, trades, _, metrics = run_research_backtest(
        daily, signals, _benchmark(dates), {}, _spec(),
        dates[1].date(), dates[2].date(), 10_500, ZERO_COSTS,
    )
    assert trades.iloc[0]["ts_code"] == codes[0]
    assert trades.iloc[0]["entry_date"] == dates[1].date()
    assert trades.iloc[0]["initial_shares"] == 1000
    assert result.iloc[0]["cash"] == pytest.approx(500)
    assert metrics["missing_signal_days"] == 0


def test_close_stop_on_purchase_day_executes_next_open_under_t_plus_one() -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ",
        "open": [10.0, 10.0, 8.5], "high": [10.0, 10.0, 8.5],
        "low": [10.0, 8.8, 8.5], "close": [10.0, 8.8, 8.5],
        "pre_close": [10.0, 10.0, 8.8],
    })
    signals = pd.DataFrame({
        "trade_date": dates[:2], "ts_code": "000001.SZ", "rank": 1,
        "volatility": .2, "industry": "银行",
    })
    _, trades, _, _ = run_research_backtest(
        daily, signals, _benchmark(dates), {}, _spec(stop_loss=.08),
        dates[1].date(), dates[2].date(), 100_000, ZERO_COSTS,
    )
    assert trades.iloc[0]["entry_date"] == dates[1].date()
    assert trades.iloc[0]["exit_date"] == dates[2].date()
    assert "止损" in trades.iloc[0]["exit_reason"]


def test_corporate_action_proxy_keeps_value_across_split_like_price_reset() -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ",
        "open": [100.0, 100.0, 50.0], "high": [100.0, 100.0, 50.0],
        "low": [100.0, 100.0, 50.0], "close": [100.0, 100.0, 50.0],
        "pre_close": [100.0, 100.0, 50.0],
    })
    signals = pd.DataFrame({
        "trade_date": dates[:2], "ts_code": "000001.SZ", "rank": 1,
        "volatility": .2, "industry": "汽车",
    })
    result, trades, _, metrics = run_research_backtest(
        daily, signals, _benchmark(dates), {}, _spec(corporate_action_mode="reinvest_proxy"),
        dates[1].date(), dates[2].date(), 100_000, ZERO_COSTS,
    )
    assert result.iloc[-1]["equity"] == pytest.approx(1.0)
    assert trades.iloc[-1]["shares"] == pytest.approx(2000)
    assert metrics["corporate_action_adjustment_count"] == 1
