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

    def intraday(self, ts_code: str, trade_date: date) -> pd.DataFrame:
        """Fetch one-minute bars for one of the most recent five trading days."""
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("未安装 AKShare，请执行 pip install -e .") from exc
        day = trade_date.strftime("%Y-%m-%d")
        frame = ak.stock_zh_a_hist_min_em(
            symbol=ts_code.split(".")[0],
            start_date=f"{day} 09:30:00",
            end_date=f"{day} 15:00:00",
            period="1",
            adjust="",
        ).copy()
        aliases = {
            "时间": "datetime", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "均价": "average",
        }
        frame = frame.rename(columns=aliases)
        required = {"datetime", "close", "volume"}
        if frame.empty or not required.issubset(frame.columns):
            return pd.DataFrame(columns=["datetime", "open", "close", "high", "low", "volume", "amount", "average"])
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        numeric = [column for column in ["open", "close", "high", "low", "volume", "amount", "average"] if column in frame]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
        return frame.dropna(subset=["datetime", "close"]).sort_values("datetime").reset_index(drop=True)
