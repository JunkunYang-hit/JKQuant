from __future__ import annotations

import os
from datetime import date

import pandas as pd

from .provider import DataProvider


class TushareProvider(DataProvider):
    def __init__(self, token: str | None = None) -> None:
        token = token or os.getenv("TUSHARE_TOKEN")
        if not token:
            raise RuntimeError("使用 Tushare 前请设置环境变量 TUSHARE_TOKEN")
        try:
            import tushare as ts
        except ImportError as exc:
            raise RuntimeError("未安装 tushare，请先执行 pip install -e .") from exc
        self.pro = ts.pro_api(token)

    def stock_basic(self) -> pd.DataFrame:
        frames = []
        for status in ("L", "D", "P"):
            frame = self.pro.stock_basic(
                exchange="", list_status=status,
                fields="ts_code,name,list_date,delist_date,list_status",
            )
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")
        result["list_date"] = pd.to_datetime(result["list_date"], format="%Y%m%d", errors="coerce")
        return result

    def daily(self, start_date: date, end_date: date) -> pd.DataFrame:
        calendar = self.pro.trade_cal(
            exchange="SSE",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        )
        open_dates = calendar.loc[calendar["is_open"].astype(str).eq("1"), "cal_date"]
        frames = [self.pro.daily(trade_date=value) for value in open_dates]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result["trade_date"] = pd.to_datetime(result["trade_date"], format="%Y%m%d")
        return result
