from __future__ import annotations

import pandas as pd

from jkquant.signal_statistics import streak2_leader_continuation


def test_streak2_leader_continuation_uses_exact_two_day_streak_and_next_day() -> None:
    dates = [value.date() for value in pd.bdate_range("2026-01-05", periods=7)]
    ranks = {
        dates[0]: {"A": 30, "B": 30, "X": 40},
        dates[1]: {"A": 10, "B": 12, "X": 40},
        dates[2]: {"A": 5, "B": 2, "C": 30, "X": 40},
        dates[3]: {"A": 8, "B": 25, "C": 10, "X": 40},
        dates[4]: {"A": 25, "C": 4, "X": 40},
        dates[5]: {"C": 3, "D": 10, "X": 40},
        dates[6]: {"C": 2, "D": 1, "X": 40},
    }
    rows = [
        {"trade_date": trade_date, "ts_code": code, "rank": rank}
        for trade_date, day in ranks.items() for code, rank in day.items()
    ]
    summary, events = streak2_leader_continuation(pd.DataFrame(rows), dates)
    assert summary["trials"] == 2
    assert summary["successes"] == 1
    assert summary["win_rate"] == 0.5
    assert events.iloc[0]["ts_code"] == "B"
    assert not bool(events.iloc[0]["success"])
    assert events.iloc[1]["ts_code"] == "C"
    assert bool(events.iloc[1]["success"])
    assert summary["current_candidate"] == {"ts_code": "D", "rank": 1}
