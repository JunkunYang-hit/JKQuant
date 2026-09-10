from __future__ import annotations

import os
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any, Callable

import pandas as pd

from .provider import DataProvider

LOGGER = logging.getLogger(__name__)


class TushareProvider(DataProvider):
    def __init__(
        self,
        token: str | None = None,
        token_env: str = "TUSHARE_TOKEN",
        requests_per_minute: int = 45,
        max_retries: int = 5,
        retry_backoff_seconds: float = 5,
        request_timeout_seconds: int = 15,
    ) -> None:
        token = token or os.getenv(token_env)
        if not token:
            raise RuntimeError(
                f"未找到 Tushare Token，请在项目根目录的 .env 中设置 {token_env}=你的Token"
            )
        try:
            import tushare as ts
        except ImportError as exc:
            raise RuntimeError("未安装 tushare，请先执行 pip install -e .") from exc
        self.pro = ts.pro_api(token, timeout=request_timeout_seconds)
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute 必须大于 0")
        self.min_interval = 60.0 / requests_per_minute
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._last_request_at = 0.0
        self._rate_lock = threading.Lock()

    def _call(self, function: Callable[..., pd.DataFrame], **kwargs: Any) -> pd.DataFrame:
        """Rate-limit every API call and retry temporary server/rate errors."""
        for attempt in range(self.max_retries + 1):
            # Reserve a request slot under a lock, then perform network I/O
            # outside the lock so slow responses do not reduce throughput.
            with self._rate_lock:
                wait = self.min_interval - (time.monotonic() - self._last_request_at)
                if wait > 0:
                    time.sleep(wait)
                self._last_request_at = time.monotonic()
            try:
                result = function(**kwargs)
                return result
            except Exception as exc:
                message = str(exc).lower()
                if any(marker in message for marker in ("权限", "积分", "token", "permission")):
                    raise RuntimeError(f"Tushare 权限错误: {exc}") from exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Tushare 请求重试 {self.max_retries} 次后失败: {exc}") from exc
                delay = self.retry_backoff_seconds * (2**attempt)
                LOGGER.warning("Tushare 请求失败，%.1f 秒后重试 (%d/%d): %s", delay, attempt + 1, self.max_retries, exc)
                time.sleep(delay)
        raise AssertionError("unreachable")

    def stock_basic(self) -> pd.DataFrame:
        frames = []
        for status in ("L", "D", "P"):
            frame = self._call(
                self.pro.stock_basic,
                exchange="", list_status=status,
                fields="ts_code,symbol,name,area,industry,market,exchange,fullname,enname,cnspell,list_date,delist_date,list_status",
            )
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")
        result["list_date"] = pd.to_datetime(result["list_date"], format="%Y%m%d", errors="coerce")
        result["delist_date"] = pd.to_datetime(result["delist_date"], format="%Y%m%d", errors="coerce")
        return result

    def daily(self, start_date: date, end_date: date) -> pd.DataFrame:
        # 低积分账户可能只能访问 daily，不能访问 trade_cal。按工作日请求，
        # 节假日返回空表；按交易日取全市场也不会触发单次 6000 行上限。
        dates = pd.bdate_range(start_date, end_date)
        frames = []
        def fetch(value: pd.Timestamp) -> pd.DataFrame:
            return self._call(self.pro.daily, trade_date=value.strftime("%Y%m%d"))

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="tushare") as executor:
            for index, frame in enumerate(executor.map(fetch, dates), start=1):
                if not frame.empty:
                    frames.append(frame)
                if index % 20 == 0 or index == len(dates):
                    LOGGER.info("Tushare 日线进度: %d/%d 个工作日", index, len(dates))
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result["trade_date"] = pd.to_datetime(result["trade_date"], format="%Y%m%d")
        return result

    @staticmethod
    def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in ("trade_date", "ann_date", "f_ann_date", "end_date", "cal_date"):
            if column in result:
                result[column] = pd.to_datetime(result[column], format="%Y%m%d", errors="coerce")
        return result

    def trade_calendar(self, start_date: date, end_date: date) -> pd.DataFrame:
        return self._normalize_dates(self._call(
            self.pro.trade_cal, exchange="SSE", start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"), is_open="1",
        ))

    def market_by_trade_date(self, endpoint: str, trade_date: date) -> pd.DataFrame:
        function = getattr(self.pro, endpoint)
        return self._normalize_dates(self._call(function, trade_date=trade_date.strftime("%Y%m%d")))

    def benchmark_daily(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        # fund_daily is the exchange-traded ETF OHLC endpoint. One 10-year range
        # remains below its row limit for a single ETF.
        return self._normalize_dates(self._call(
            self.pro.fund_daily, ts_code=ts_code, start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        ))

    def financial_statement(
        self, statement: str, ts_code: str, start_date: date, end_date: date,
    ) -> pd.DataFrame:
        if statement not in {"income", "balancesheet", "cashflow"}:
            raise ValueError(f"不支持的财务报表: {statement}")
        return self._normalize_dates(self._call(
            getattr(self.pro, statement), ts_code=ts_code,
            start_date=start_date.strftime("%Y%m%d"), end_date=end_date.strftime("%Y%m%d"),
        ))

    def macro_dataset(self, endpoint: str) -> pd.DataFrame:
        if endpoint not in {"cn_gdp", "cn_cpi", "cn_ppi", "cn_m", "cn_pmi", "sf_month"}:
            raise ValueError(f"不支持的宏观接口: {endpoint}")
        return self._normalize_dates(self._call(getattr(self.pro, endpoint)))
