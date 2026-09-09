from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataProvider(ABC):
    """Normalized end-of-day data source."""

    @abstractmethod
    def daily(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Return OHLCV rows with trade_date and ts_code."""

    @abstractmethod
    def stock_basic(self) -> pd.DataFrame:
        """Return ts_code, name, list_date and list_status."""

    def company_names(self) -> pd.DataFrame:
        """Return optional ts_code/name mapping for low-permission accounts."""
        return pd.DataFrame(columns=["ts_code", "name"])
