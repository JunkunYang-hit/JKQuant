from __future__ import annotations

from datetime import date

import pandas as pd

from .provider import DataProvider


class AkshareMetadataProvider(DataProvider):
    """Use AKShare only for current A-share code/name metadata."""

    def daily(self, start_date: date, end_date: date) -> pd.DataFrame:
        raise NotImplementedError("AKShare 在本项目中只用于证券名称")

    def stock_basic(self) -> pd.DataFrame:
        raise NotImplementedError("AKShare 在本项目中只用于证券名称")

    def company_names(self) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("未安装 AKShare，请执行 pip install -e .") from exc
        frame = ak.stock_info_a_code_name().copy()
        if frame.empty or not {"code", "name"}.issubset(frame.columns):
            raise RuntimeError("AKShare 未返回有效的 A 股代码名称表")
        frame["code"] = frame["code"].astype(str).str.zfill(6)

        def to_ts_code(code: str) -> str:
            if code.startswith("6"):
                return f"{code}.SH"
            if code.startswith(("4", "8", "9")):
                return f"{code}.BJ"
            return f"{code}.SZ"

        frame["ts_code"] = frame["code"].map(to_ts_code)
        frame["name_source"] = "akshare_current"
        return frame[["ts_code", "name", "name_source"]].drop_duplicates("ts_code")
