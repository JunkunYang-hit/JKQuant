from __future__ import annotations

from typing import Any

import pandas as pd
import requests


class TdxQuantError(RuntimeError):
    pass


class TdxQuantClient:
    """Minimal client for the official TdxQuant local HTTP endpoint."""

    def __init__(
        self, base_url: str = "http://127.0.0.1:17709/", timeout: float = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        if session is None:
            self.session = requests.Session()
            # The endpoint is always a loopback service.  Ignoring HTTP(S)_PROXY
            # prevents corporate/system proxy settings from intercepting it.
            self.session.trust_env = False
        else:
            self.session = session
        self._request_id = 0

    def call(self, method: str, **params: Any) -> Any:
        self._request_id += 1
        try:
            response = self.session.post(
                self.base_url,
                json={"id": self._request_id, "method": method, "params": params},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise TdxQuantError(
                "无法连接通达信本地行情服务；请先启动支持 TQ 的通达信客户端并登录。"
            ) from exc
        if isinstance(payload, dict) and payload.get("error"):
            raise TdxQuantError(str(payload["error"]))
        result = payload.get("result", payload) if isinstance(payload, dict) else payload
        if isinstance(result, dict) and str(result.get("ErrorId", "0")) not in {"", "0"}:
            raise TdxQuantError(str(result.get("Msg") or result))
        return result

    @staticmethod
    def _value(result: Any) -> Any:
        if isinstance(result, dict) and "Value" in result:
            return result["Value"]
        return result

    def snapshot(self, stock_list: list[str]) -> pd.DataFrame:
        if not stock_list:
            return pd.DataFrame()
        try:
            result = self.call(
                "get_market_snapshot_batch", stock_list=stock_list, field_list=[], return_df=False,
            )
        except TdxQuantError:
            # Older tqcenter releases expose only the non-batch function.
            rows: dict[str, Any] = {}
            for code in stock_list:
                rows[code] = self._value(self.call("get_market_snapshot", stock_code=code))
            value = rows
        else:
            value = self._value(result)
        if not isinstance(value, dict):
            raise TdxQuantError("通达信快照返回格式无法识别")
        rows = []
        for code in stock_list:
            raw = value.get(code, {}) or {}
            rows.append({"ts_code": code, **raw})
        frame = pd.DataFrame(rows)
        numeric = [
            "LastClose", "Open", "Max", "Min", "Now", "Volume", "NowVol", "Amount",
            "Inside", "Outside", "Before5MinNow", "Average", "Zangsu", "ZAFPre3",
        ]
        for column in numeric:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if {"Now", "LastClose"}.issubset(frame.columns):
            frame["change_pct"] = frame["Now"].div(frame["LastClose"]).sub(1)
        return frame

    def intraday_bars(self, stock_code: str, count: int = 240, period: str = "1m") -> Any:
        """Get intraday bars; the raw official payload is kept to avoid lossy guessing."""
        return self.call(
            "get_market_data",
            field_list=["Open", "High", "Low", "Close", "Volume", "Amount"],
            stock_list=[stock_code],
            count=count,
            dividend_type="none",
            period=period,
        )
