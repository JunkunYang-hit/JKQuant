from __future__ import annotations

import pandas as pd

from jkquant.limit_up import build_board_history, promotion_summary


def test_streak_breaks_on_suspension_and_promotion_uses_next_market_day():
    days = pd.bdate_range("2025-01-06", periods=4)
    rows = []
    limits = []
    for code, present in {"A.SZ": [0, 1, 3], "B.SZ": [0, 1, 2, 3]}.items():
        for index in present:
            rows.append({"trade_date": days[index], "ts_code": code, "open": 10., "high": 11.,
                         "low": 10., "close": 11., "pre_close": 10., "pct_chg": 10.,
                         "vol": 100., "amount": 1_000_000.})
            limits.append({"trade_date": days[index], "ts_code": code, "up_limit": 11.})
    basic = pd.DataFrame({"ts_code": ["A.SZ", "B.SZ"], "name": ["A", "B"],
                          "industry": ["甲", "甲"]})
    boards = build_board_history(pd.DataFrame(rows), pd.DataFrame(limits), basic)
    a = boards[boards.ts_code.eq("A.SZ")].sort_values("trade_date")
    assert a.streak.tolist() == [1, 2, 1]
    early = promotion_summary(boards, days[1]).set_index("当前板数")
    assert early.loc[1, "样本数"] == 2
    assert early.loc[1, "晋级数"] == 2
    later = promotion_summary(boards, days[3]).set_index("当前板数")
    assert later.loc[2, "样本数"] == 2
    assert later.loc[2, "晋级数"] == 1


def test_st_board_excluded_by_limit_ratio():
    daily = pd.DataFrame([{"trade_date": "2025-01-06", "ts_code": "A.SZ", "open": 10.,
                           "high": 10.5, "low": 10., "close": 10.5, "pre_close": 10.,
                           "pct_chg": 5., "vol": 100., "amount": 1000.}])
    limits = pd.DataFrame([{"trade_date": "2025-01-06", "ts_code": "A.SZ", "up_limit": 10.5}])
    basic = pd.DataFrame([{"ts_code": "A.SZ", "name": "A", "industry": "甲"}])
    assert build_board_history(daily, limits, basic).empty


def test_star_market_is_not_a_board_candidate():
    daily = pd.DataFrame([{"trade_date": "2025-01-06", "ts_code": "688256.SH", "open": 10.,
                           "high": 12., "low": 10., "close": 12., "pre_close": 10.,
                           "pct_chg": 20., "vol": 100., "amount": 1000.}])
    limits = pd.DataFrame([{"trade_date": "2025-01-06", "ts_code": "688256.SH", "up_limit": 12.}])
    basic = pd.DataFrame([{"ts_code": "688256.SH", "name": "甲", "industry": "半导体",
                           "market": "科创板"}])
    assert build_board_history(daily, limits, basic).empty


def test_chinext_market_is_not_a_board_candidate():
    daily = pd.DataFrame([{"trade_date": "2025-01-06", "ts_code": "300139.SZ", "open": 10.,
                           "high": 12., "low": 10., "close": 12., "pre_close": 10.,
                           "pct_chg": 20., "vol": 100., "amount": 1000.}])
    limits = pd.DataFrame([{"trade_date": "2025-01-06", "ts_code": "300139.SZ", "up_limit": 12.}])
    basic = pd.DataFrame([{"ts_code": "300139.SZ", "name": "丙", "industry": "电子",
                           "market": "创业板"}])
    assert build_board_history(daily, limits, basic).empty
