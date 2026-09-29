import pandas as pd

from jkquant.backtest.benchmark import benchmark_return_map


def test_benchmark_pct_change_is_percentage() -> None:
    frame = pd.DataFrame({"trade_date": ["2026-09-01"], "pct_chg": [1.25]})
    values = benchmark_return_map(frame)
    assert values[pd.Timestamp("2026-09-01")] == 0.0125
