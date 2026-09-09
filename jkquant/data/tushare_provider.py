from __future__ import annotations

import os
import logging
import time
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

    def _call(self, function: Callable[..., pd.DataFrame], **kwargs: Any) -> pd.DataFrame:
        """Rate-limit every API call and retry temporary server/rate errors."""
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            try:
                result = function(**kwargs)
                self._last_request_at = time.monotonic()
                return result
            except Exception as exc:
                self._last_request_at = time.monotonic()
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
                fields="ts_code,name,list_date,delist_date,list_status",
            )
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")
        result["list_date"] = pd.to_datetime(result["list_date"], format="%Y%m%d", errors="coerce")
        return result

    def daily(self, start_date: date, end_date: date) -> pd.DataFrame:
        # 低积分账户可能只能访问 daily，不能访问 trade_cal。按工作日请求，
        # 节假日返回空表；按交易日取全市场也不会触发单次 6000 行上限。
        dates = pd.bdate_range(start_date, end_date)
        frames = []
        for index, value in enumerate(dates, start=1):
            frame = self._call(self.pro.daily, trade_date=value.strftime("%Y%m%d"))
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
