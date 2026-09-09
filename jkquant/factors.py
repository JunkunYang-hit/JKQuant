from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_COLUMNS = [
    "return_5d", "return_20d", "close_ma20", "ma5_ma20",
    "volatility_20d", "max_drawdown_20d", "amount_mean_20d", "amount_ratio_5_20",
]


def calculate_factors(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate backward-looking price/volume factors without future data."""
    required = {"trade_date", "ts_code", "close", "amount"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"日线数据缺少字段: {sorted(missing)}")
    frame = daily.sort_values(["ts_code", "trade_date"]).copy()
    grouped = frame.groupby("ts_code", group_keys=False)
    frame["history_count"] = grouped.cumcount() + 1
    if "pct_chg" in frame:
        frame["return_1d"] = pd.to_numeric(frame["pct_chg"], errors="coerce") / 100
        # Tushare pct_chg uses the ex-rights pre_close. A cumulative index made
        # from it is suitable for price factors without requiring adj_factor.
        frame["factor_price"] = frame.groupby("ts_code")["return_1d"].transform(
            lambda values: (1 + values.fillna(0)).cumprod()
        )
    else:
        frame["return_1d"] = grouped["close"].pct_change()
        frame["factor_price"] = frame["close"]
    price_grouped = frame.groupby("ts_code", group_keys=False)["factor_price"]
    frame["return_5d"] = price_grouped.pct_change(5)
    frame["return_20d"] = price_grouped.pct_change(20)
    ma5 = price_grouped.transform(lambda x: x.rolling(5, min_periods=5).mean())
    ma20 = price_grouped.transform(lambda x: x.rolling(20, min_periods=20).mean())
    frame["close_ma20"] = frame["factor_price"] / ma20 - 1
    frame["ma5_ma20"] = ma5 / ma20 - 1
    frame["volatility_20d"] = grouped["return_1d"].transform(
        lambda x: x.rolling(20, min_periods=20).std()
    ) * np.sqrt(252)
    rolling_peak = price_grouped.transform(lambda x: x.rolling(20, min_periods=20).max())
    frame["max_drawdown_20d"] = 1 - frame["factor_price"] / rolling_peak
    amount5 = grouped["amount"].transform(lambda x: x.rolling(5, min_periods=5).mean())
    amount20 = grouped["amount"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    frame["amount_mean_20d"] = amount20
    frame["amount_ratio_5_20"] = amount5 / amount20
    frame[FACTOR_COLUMNS] = frame[FACTOR_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return frame
