from __future__ import annotations

import pandas as pd

from scripts.audit_worst_trades import audit_worst


def test_old_trade_is_flagged_but_adjusted_trade_is_not(tmp_path):
    daily = pd.DataFrame([
        {"ts_code": "A.SZ", "trade_date": "2025-01-06", "open": 10., "high": 10., "low": 10.,
         "close": 10., "pre_close": 10., "pct_chg": 0.},
        {"ts_code": "A.SZ", "trade_date": "2025-01-07", "open": 5., "high": 5., "low": 5.,
         "close": 5., "pre_close": 5., "pct_chg": 0.},
    ])
    old = tmp_path / "old" / "strategy" / "trades.csv"
    new = tmp_path / "new" / "strategy" / "trades.csv"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    base = {"ts_code": "A.SZ", "entry_date": "2025-01-06", "exit_date": "2025-01-07",
            "exit_price": 5., "net_return": -.501, "holding_trading_days": 2}
    pd.DataFrame([{**base, "entry_price": 10., "price_change": -.5}]).to_csv(old, index=False)
    pd.DataFrame([{**base, "entry_price": 5., "entry_price_unadjusted": 10.,
                   "price_change": 0.}]).to_csv(new, index=False)
    result, _ = audit_worst(daily, [old, new])
    assert result.iloc[0]["回测遗漏百分点"] == 50
    assert result.iloc[1]["回测遗漏百分点"] == 0
