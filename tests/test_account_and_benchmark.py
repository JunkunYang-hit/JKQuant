from datetime import date

import pandas as pd

from jkquant.backtest.benchmark import benchmark_return_map
from jkquant.data.account_store import AccountStore


def test_account_is_persistent(tmp_path):
    path = tmp_path / "account.sqlite3"
    AccountStore(path).upsert("510300.SH", "沪深300ETF", 100, 3.9, date(2026, 9, 1))
    holding = AccountStore(path).get("510300.SH")
    assert holding is not None
    assert holding["quantity"] == 100
    AccountStore(path).delete("510300.SH")
    assert AccountStore(path).get("510300.SH") is None


def test_benchmark_pct_change_is_percentage():
    frame = pd.DataFrame({"trade_date": ["2026-09-01"], "pct_chg": [1.25]})
    values = benchmark_return_map(frame)
    assert values[pd.Timestamp("2026-09-01")] == 0.0125
