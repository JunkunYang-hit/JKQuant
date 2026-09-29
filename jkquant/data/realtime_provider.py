from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from dotenv import load_dotenv

from .tdxquant_provider import TdxQuantClient, TdxQuantError


class RealtimeQuoteError(RuntimeError):
    pass


class RealtimeQuoteProvider(Protocol):
    source_name: str

    def snapshot(self, stock_list: list[str]) -> pd.DataFrame: ...


class TushareRealtimeProvider:
    """Tushare rt_k adapter normalized to daily units: lots and CNY thousands."""

    source_name = "Tushare实时日线"

    def __init__(self, token: str | None = None, token_env: str = "TUSHARE_TOKEN",
                 client: Any | None = None, batch_size: int = 300) -> None:
        token = token or os.getenv(token_env)
        if client is None and not token:
            raise RealtimeQuoteError(f"未找到 {token_env}")
        if client is None:
            try:
                import tushare as ts
            except ImportError as exc:
                raise RealtimeQuoteError("未安装 tushare") from exc
            client = ts.pro_api(token)
        self.client = client
        self.batch_size = batch_size

    def snapshot(self, stock_list: list[str]) -> pd.DataFrame:
        codes = list(dict.fromkeys(str(code) for code in stock_list if code))
        if not codes:
            return pd.DataFrame()
        frames = []
        fields = "ts_code,name,pre_close,high,open,low,close,vol,amount,trade_time"
        try:
            for start in range(0, len(codes), self.batch_size):
                query = ",".join(codes[start:start + self.batch_size])
                frame = self.client.rt_k(ts_code=query, fields=fields)
                if frame is not None and not frame.empty:
                    frames.append(frame)
        except Exception as exc:
            message = str(exc)
            if "权限" in message or "permission" in message.lower():
                raise RealtimeQuoteError(
                    "当前 Tushare 账号没有 rt_k 权限；2000积分不包含实时日线，需要单独开通“A股日线RT”。"
                ) from exc
            raise RealtimeQuoteError(f"Tushare实时行情请求失败：{message}") from exc
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="last")
        numeric = ["pre_close", "high", "open", "low", "close", "vol", "amount"]
        for column in numeric:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        # rt_k uses shares/CNY; JKQuant daily uses lots/CNY thousands.
        result["vol"] = result["vol"] / 100
        result["amount"] = result["amount"] / 1_000
        result["source"] = self.source_name
        result["fetched_at"] = datetime.now().isoformat(timespec="seconds")
        return result[result["ts_code"].isin(codes)].reset_index(drop=True)


class TdxRealtimeProvider:
    """TdxQuant local-client snapshot adapter with the same normalized units."""

    source_name = "通达信TdxQuant"

    def __init__(self, base_url: str = "http://127.0.0.1:17709/", timeout: float = 3,
                 client: TdxQuantClient | None = None) -> None:
        self.client = client or TdxQuantClient(base_url=base_url, timeout=timeout)

    def snapshot(self, stock_list: list[str]) -> pd.DataFrame:
        try:
            raw = self.client.snapshot(stock_list)
        except TdxQuantError as exc:
            raise RealtimeQuoteError(str(exc)) from exc
        if raw.empty:
            return raw
        result = raw.rename(columns={
            "LastClose": "pre_close", "Open": "open", "Max": "high", "Min": "low",
            "Now": "close", "Volume": "vol", "Amount": "amount", "RefreshTime": "trade_time",
        }).copy()
        for column in ("pre_close", "open", "high", "low", "close", "vol", "amount"):
            if column in result:
                result[column] = pd.to_numeric(result[column], errors="coerce")
        # Tdx snapshot uses lots and CNY ten-thousands. Daily cache uses lots and CNY thousands.
        result["amount"] = result["amount"] * 10
        result["source"] = self.source_name
        result["fetched_at"] = datetime.now().isoformat(timespec="seconds")
        return result


def build_realtime_provider(config: dict[str, Any], provider: str | None = None) -> RealtimeQuoteProvider:
    intraday = config.get("intraday", {})
    name = provider or str(intraday.get("provider", "tdxquant"))
    if name == "tdxquant":
        settings = intraday.get("tdxquant", {})
        return TdxRealtimeProvider(
            base_url=str(settings.get("base_url", "http://127.0.0.1:17709/")),
            timeout=float(settings.get("timeout_seconds", 3)),
        )
    if name == "tushare_rt":
        load_dotenv(Path(config["_config_dir"]) / ".env")
        settings = intraday.get("tushare_rt", {})
        return TushareRealtimeProvider(
            token_env=str(settings.get("token_env", config.get("data", {}).get("token_env", "TUSHARE_TOKEN"))),
            batch_size=int(settings.get("batch_size", 300)),
        )
    raise RealtimeQuoteError(f"不支持的实时行情源：{name}")
