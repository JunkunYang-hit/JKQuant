from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from jkquant.diagnostics import calculate_factor_diagnostics


def test_factor_diagnostics_produces_ic_quantiles_and_correlations() -> None:
    dates = pd.bdate_range("2025-01-01", periods=90)
    rows = []
    codes = [f"{index:06d}.SZ" for index in range(1, 41)]
    for stock_index, code in enumerate(codes):
        strength = (stock_index - 20) / 20000
        close = 10.0
        for day_index, trade_date in enumerate(dates):
            daily_return = strength + np.sin(day_index / 5 + stock_index) * 0.001
            close *= 1 + daily_return
            rows.append({
                "trade_date": trade_date, "ts_code": code, "close": close,
                "pct_chg": daily_return * 100, "amount": 100_000 + stock_index * 1_000,
                "vol": 10_000,
            })
    daily = pd.DataFrame(rows)
    basic = pd.DataFrame({
        "ts_code": codes, "name": [f"股票{index}" for index in range(40)],
        "list_date": pd.Timestamp("2020-01-01"), "delist_date": pd.NaT,
    })
    factors = {
        "return_5d": {"direction": 1}, "return_20d": {"direction": 1},
        "close_ma20": {"direction": 1}, "ma5_ma20": {"direction": 1},
        "volatility_20d": {"direction": -1}, "max_drawdown_20d": {"direction": -1},
        "amount_mean_20d": {"direction": 1}, "amount_ratio_5_20": {"direction": 1},
    }
    config = {
        "market": {"exclude_st": True, "min_list_days": 120, "min_amount": 20_000},
        "strategy": {"factors": factors},
        "factor_diagnostics": {"horizons": [1, 5], "quantiles": 5},
    }
    result = calculate_factor_diagnostics(
        daily, basic, config, dates[30].date(), dates[70].date(),
    )
    assert len(result.summary) == 16
    assert set(result.daily_ic["horizon"]) == {1, 5}
    assert set(result.quantiles["quantile"]) == {1, 2, 3, 4, 5}
    assert result.correlations.shape == (8, 8)
    momentum = result.summary.query("factor == 'return_20d' and horizon == 5").iloc[0]
    assert momentum["mean_rank_ic"] > 0.2
    assert momentum["top_bottom_spread"] > 0
