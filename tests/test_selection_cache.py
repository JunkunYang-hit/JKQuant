from datetime import date

import pandas as pd

from jkquant.data.selection_cache import SelectionCache


def test_selection_cache_round_trip(tmp_path) -> None:
    cache = SelectionCache(tmp_path / "results.sqlite3")
    frame = pd.DataFrame([
        {"rank": 1, "trade_date": "2025-01-02", "ts_code": "000001.SZ", "name": "平安银行", "total_score": 0.9},
        {"rank": 2, "trade_date": "2025-01-02", "ts_code": "600000.SH", "name": "浦发银行", "total_score": 0.8},
    ])
    cache.put("strategy-a", date(2025, 1, 2), frame)
    loaded = cache.get("strategy-a", date(2025, 1, 2))
    assert loaded is not None
    assert loaded["ts_code"].tolist() == ["000001.SZ", "600000.SH"]
    assert cache.get("strategy-b", date(2025, 1, 2)) is None
    assert cache.cached_dates("strategy-a", date(2025, 1, 1), date(2025, 1, 3)) == {
        date(2025, 1, 2)
    }
    history = cache.history("strategy-a", date(2025, 1, 1), date(2025, 1, 3))
    assert history[["rank", "ts_code"]].to_dict("records") == [
        {"rank": 1, "ts_code": "000001.SZ"},
        {"rank": 2, "ts_code": "600000.SH"},
    ]
