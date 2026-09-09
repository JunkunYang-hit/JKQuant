from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .provider import DataProvider


class DemoProvider(DataProvider):
    """Deterministic synthetic data for installation checks, never for investment."""

    def __init__(self, stock_count: int = 100, seed: int = 20260909) -> None:
        self.stock_count = stock_count
        self.seed = seed

    def stock_basic(self) -> pd.DataFrame:
        codes = self._codes()
        return pd.DataFrame(
            {
                "ts_code": codes,
                "name": [f"样例股份{i:03d}" for i in range(1, self.stock_count + 1)],
                "list_date": pd.Timestamp("2010-01-01"),
                "list_status": "L",
            }
        )

    def daily(self, start_date: date, end_date: date) -> pd.DataFrame:
        # Generate from a fixed anchor so overlapping requests always return
        # identical prices; this makes incremental-cache tests meaningful.
        all_dates = pd.bdate_range("2005-01-03", end_date)
        wanted = all_dates >= pd.Timestamp(start_date)
        codes = self._codes()
        rng = np.random.default_rng(self.seed)
        rows: list[pd.DataFrame] = []
        for index, code in enumerate(codes):
            drift = (index / max(self.stock_count - 1, 1) - 0.45) * 0.0006
            returns = rng.normal(drift, 0.018, len(all_dates))
            close = (8 + index * 0.15) * np.exp(np.cumsum(returns))
            pre_close = np.r_[close[0] / np.exp(returns[0]), close[:-1]]
            open_ = pre_close * (1 + rng.normal(0, 0.005, len(all_dates)))
            high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.015, len(all_dates)))
            low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.015, len(all_dates)))
            vol = rng.lognormal(11.5, 0.45, len(all_dates))
            # Tushare: vol is in lots (100 shares), amount is in CNY thousands.
            amount = vol * (open_ + close) / 2 / 10
            rows.append(pd.DataFrame({
                "trade_date": all_dates[wanted], "ts_code": code, "open": open_[wanted],
                "high": high[wanted], "low": low[wanted], "close": close[wanted],
                "pre_close": pre_close[wanted], "vol": vol[wanted], "amount": amount[wanted],
            }))
        return pd.concat(rows, ignore_index=True)

    def _codes(self) -> list[str]:
        return [f"{600000 + i:06d}.SH" for i in range(self.stock_count)]
