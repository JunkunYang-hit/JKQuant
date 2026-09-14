from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def streak2_leader_continuation(
    rankings: pd.DataFrame, trading_dates: list[date], top_n: int = 20,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate whether the best-ranked exact two-day streak survives a third day."""
    columns = [
        "signal_date", "next_trade_date", "ts_code", "signal_rank",
        "next_rank", "success",
    ]
    if rankings.empty or len(trading_dates) < 4:
        return {
            "trials": 0, "successes": 0, "win_rate": None,
            "current_candidate": None, "definition": f"恰好连续2次进入Top{top_n}者中当日排名最靠前的一只",
        }, pd.DataFrame(columns=columns)

    frame = rankings.copy()
    frame["trade_date"] = frame["trade_date"].map(lambda value: pd.Timestamp(value).date())
    by_date = {
        trade_date: day.set_index("ts_code")["rank"].astype(int).to_dict()
        for trade_date, day in frame.groupby("trade_date")
    }
    ordered_dates = sorted(dict.fromkeys(trading_dates))

    def leader_at(index: int) -> tuple[str, int] | None:
        current_date = ordered_dates[index]
        previous_date = ordered_dates[index - 1]
        before_previous = ordered_dates[index - 2]
        if not all(value in by_date for value in (current_date, previous_date, before_previous)):
            return None
        current = by_date[current_date]
        previous = by_date[previous_date]
        before = by_date[before_previous]
        candidates = [
            (str(code), int(rank)) for code, rank in current.items()
            if rank <= top_n and previous.get(code, top_n + 1) <= top_n
            and before.get(code, top_n + 1) > top_n
        ]
        return min(candidates, key=lambda item: (item[1], item[0])) if candidates else None

    events = []
    for index in range(2, len(ordered_dates) - 1):
        leader = leader_at(index)
        next_date = ordered_dates[index + 1]
        if leader is None or next_date not in by_date:
            continue
        code, signal_rank = leader
        next_rank = by_date[next_date].get(code)
        success = next_rank is not None and next_rank <= top_n
        events.append({
            "signal_date": ordered_dates[index], "next_trade_date": next_date,
            "ts_code": code, "signal_rank": signal_rank,
            "next_rank": int(next_rank) if next_rank is not None else None,
            "success": bool(success),
        })

    current = leader_at(len(ordered_dates) - 1)
    event_frame = pd.DataFrame(events, columns=columns)
    successes = int(event_frame["success"].sum()) if not event_frame.empty else 0
    trials = len(event_frame)
    return {
        "trials": trials,
        "successes": successes,
        "failures": trials - successes,
        "win_rate": successes / trials if trials else None,
        "current_candidate": (
            {"ts_code": current[0], "rank": current[1]} if current is not None else None
        ),
        "definition": f"恰好连续2次进入Top{top_n}者中当日排名最靠前的一只",
        "sample_start": event_frame["signal_date"].min() if trials else None,
        "sample_end": event_frame["signal_date"].max() if trials else None,
    }, event_frame
