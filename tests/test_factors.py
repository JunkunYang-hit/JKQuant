from datetime import date, timedelta

import pandas as pd

from jkquant.data.demo_provider import DemoProvider
from jkquant.factors import calculate_factors


def test_factors_only_use_current_and_past_rows() -> None:
    provider = DemoProvider(stock_count=4, seed=7)
    end = date(2025, 6, 30)
    daily = provider.daily(end - timedelta(days=80), end)
    full = calculate_factors(daily)
    cutoff = full["trade_date"].sort_values().unique()[-5]
    partial = calculate_factors(daily[daily["trade_date"] <= cutoff])
    full_at_cutoff = full[full["trade_date"] == cutoff].sort_values("ts_code")
    partial_at_cutoff = partial[partial["trade_date"] == cutoff].sort_values("ts_code")
    assert full_at_cutoff["return_20d"].tolist() == partial_at_cutoff["return_20d"].tolist()


def test_tushare_pct_chg_avoids_raw_ex_rights_price_jump() -> None:
    dates = pd.bdate_range("2025-01-01", periods=25)
    daily = pd.DataFrame({
        "trade_date": dates,
        "ts_code": "000001.SZ",
        "close": [10.0] * 20 + [5.0] * 5,
        "amount": 100000.0,
        "pct_chg": 0.0,
    })
    result = calculate_factors(daily)
    assert result["return_20d"].iloc[-1] == 0.0
    assert result["close_ma20"].iloc[-1] == 0.0
