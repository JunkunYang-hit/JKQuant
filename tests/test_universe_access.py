from __future__ import annotations

import pandas as pd

from jkquant.strategy import is_bse_market, is_chinext_market, is_star_market, select_stocks


def test_star_market_codes_and_metadata_are_filtered_before_ranking():
    codes = pd.Series(["688256.SH", "689009.SH", "600000.SH", "000001.SZ"])
    assert is_star_market(codes).tolist() == [True, True, False, False]
    assert is_chinext_market(pd.Series(["300139.SZ", "301123.SZ", "000001.SZ"])).tolist() == [True, True, False]
    assert is_bse_market(pd.Series(["920001.BJ", "830001.BJ", "600000.SH"])).tolist() == [True, True, False]
    factors = pd.DataFrame({
        "trade_date": [pd.Timestamp("2025-09-01")] * 4,
        "ts_code": ["688256.SH", "300139.SZ", "920001.BJ", "600000.SH"],
        "history_count": [200] * 4, "amount": [100_000] * 4,
        "vol": [1000] * 4, "close": [20., 30., 15., 10.], "factor": [10., 11., 12., 1.],
    })
    basic = pd.DataFrame({
        "ts_code": ["688256.SH", "300139.SZ", "920001.BJ", "600000.SH"], "name": ["甲", "丙", "丁", "乙"],
        "market": ["科创板", "创业板", "北交所", "主板"], "list_date": ["2020-01-01"] * 4,
        "delist_date": [None] * 4,
    })
    config = {"market": {"exclude_star_market": True, "exclude_chinext_market": True, "exclude_bse_market": True, "min_list_days": 120,
                         "min_amount": 0, "exclude_st": True},
              "strategy": {"top_k": 20, "factors": {"factor": {
                  "category": "momentum", "direction": 1, "weight": 1}},
                           "category_weights": {"momentum": 1}}}
    chosen, summary = select_stocks(factors, basic, config)
    assert chosen.ts_code.tolist() == ["600000.SH"]
    assert summary["eligible_count"] == 1
