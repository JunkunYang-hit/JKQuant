from datetime import date, timedelta

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

