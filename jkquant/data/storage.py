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
        self.weekly_path = root / "weekly.parquet"
        self.monthly_path = root / "monthly.parquet"
        self.daily_basic_path = root / "daily_basic.parquet"
        self.limit_path = root / "stk_limit.parquet"
        self.benchmark_path = root / "benchmark_510300.parquet"
        self.macro_dir = root / "macro"
        self.fundamental_dir = root / "fundamentals"

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

    @staticmethod
    def _load_frame(path: Path, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        for column in date_columns:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame

    @classmethod
    def _merge_frame(
        cls, path: Path, frame: pd.DataFrame, keys: list[str],
        date_columns: tuple[str, ...] = (),
    ) -> None:
        if frame.empty:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        old = cls._load_frame(path, date_columns)
        combined = pd.concat([old, frame], ignore_index=True)
        combined = combined.drop_duplicates(keys, keep="last").sort_values(keys)
        combined.to_parquet(path, index=False)

    def save_market_dataset(self, name: str, frame: pd.DataFrame) -> None:
        path = getattr(self, f"{name}_path")
        self._merge_frame(path, frame, ["trade_date", "ts_code"], ("trade_date",))

    def load_market_dataset(self, name: str) -> pd.DataFrame:
        return self._load_frame(getattr(self, f"{name}_path"), ("trade_date",))

    def save_macro(self, name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.macro_dir.mkdir(parents=True, exist_ok=True)
        frame.drop_duplicates().to_parquet(self.macro_dir / f"{name}.parquet", index=False)

    def load_macro(self, name: str) -> pd.DataFrame:
        return self._load_frame(self.macro_dir / f"{name}.parquet")

    def save_fundamental(self, statement: str, ts_code: str, frame: pd.DataFrame) -> None:
        safe_code = ts_code.replace(".", "_")
        path = self.fundamental_dir / statement / f"{safe_code}.parquet"
        keys = [column for column in ("ts_code", "end_date", "ann_date", "report_type") if column in frame]
        if keys:
            self._merge_frame(path, frame, keys, ("ann_date", "f_ann_date", "end_date"))

    def load_fundamental(self, statement: str, ts_code: str) -> pd.DataFrame:
        safe_code = ts_code.replace(".", "_")
        return self._load_frame(
            self.fundamental_dir / statement / f"{safe_code}.parquet",
            ("ann_date", "f_ann_date", "end_date"),
        )
