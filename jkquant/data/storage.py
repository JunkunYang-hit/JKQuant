from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.daily_path = root / "daily.parquet"
        self.basic_path = root / "stock_basic.parquet"
        self.names_path = root / "company_names.parquet"

    def load_daily(self) -> pd.DataFrame:
        if not self.daily_path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(self.daily_path)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        return frame

    def save_daily(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        old = self.load_daily()
        combined = pd.concat([old, frame], ignore_index=True)
        combined = combined.drop_duplicates(["trade_date", "ts_code"], keep="last")
        combined = combined.sort_values(["trade_date", "ts_code"])
        combined.to_parquet(self.daily_path, index=False)

    def load_basic(self) -> pd.DataFrame:
        return pd.read_parquet(self.basic_path)

    def save_basic(self, frame: pd.DataFrame) -> None:
        frame.to_parquet(self.basic_path, index=False)

    def load_names(self) -> pd.DataFrame:
        if not self.names_path.exists():
            return pd.DataFrame(columns=["ts_code", "name"])
        return pd.read_parquet(self.names_path)

    def save_names(self, frame: pd.DataFrame) -> None:
        if not frame.empty:
            frame.to_parquet(self.names_path, index=False)
