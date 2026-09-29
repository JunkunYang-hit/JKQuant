from __future__ import annotations

import pandas as pd

from jkquant.data.realtime_provider import TdxRealtimeProvider, TushareRealtimeProvider
from jkquant.data.tdxquant_provider import TdxQuantClient
from jkquant.low_position_pool import evaluate_realtime_alerts


class _Tushare:
    def rt_k(self, ts_code: str, fields: str) -> pd.DataFrame:
        assert ts_code == "600001.SH"
        assert "trade_time" in fields
        return pd.DataFrame([{
            "ts_code": "600001.SH", "name": "样本", "pre_close": 10, "open": 10.1,
            "high": 11.05, "low": 10, "close": 11, "vol": 30_000, "amount": 330_000,
            "trade_time": "2026-09-29 10:30:00",
        }])


class _Tdx:
    def snapshot(self, stock_list: list[str]) -> pd.DataFrame:
        assert stock_list == ["600001.SH"]
        return pd.DataFrame([{
            "ts_code": "600001.SH", "LastClose": 10, "Open": 10.1, "Max": 11.05,
            "Min": 10, "Now": 11, "Volume": 300, "Amount": 33, "RefreshTime": "103000",
        }])


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "id": 1,
            "result": {"Value": {"600000.SH": {
                "Now": "9.07", "LastClose": "9.06", "Open": "9.05",
                "Max": "9.15", "Min": "9.00", "RefreshTime": "153058",
            }}},
        }


class _Session:
    def __init__(self) -> None:
        self.payload: dict | None = None

    def post(self, _url: str, json: dict, timeout: float) -> _Response:
        self.payload = json
        assert timeout == 1
        return _Response()


def test_realtime_providers_normalize_to_daily_units() -> None:
    tushare = TushareRealtimeProvider(client=_Tushare()).snapshot(["600001.SH"])
    assert tushare.iloc[0]["vol"] == 300  # shares -> lots
    assert tushare.iloc[0]["amount"] == 330  # CNY -> CNY thousands
    tdx = TdxRealtimeProvider(client=_Tdx()).snapshot(["600001.SH"])  # type: ignore[arg-type]
    assert tdx.iloc[0]["vol"] == 300
    assert tdx.iloc[0]["amount"] == 330  # CNY ten-thousands -> thousands


def test_tdxquant_http_snapshot_normalizes_values() -> None:
    session = _Session()
    client = TdxQuantClient(timeout=1, session=session)  # type: ignore[arg-type]
    result = client.snapshot(["600000.SH"])
    assert session.payload is not None
    assert session.payload["method"] == "get_market_snapshot_batch"
    assert result.iloc[0]["Now"] == 9.07
    assert result.iloc[0]["change_pct"] > 0


def test_realtime_bar_promotes_watch_pool_member_to_priority() -> None:
    dates = pd.bdate_range("2026-08-01", periods=20)
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "600001.SH", "vol": 100.0, "amount": 100.0,
    })
    pool = pd.DataFrame([{
        "ts_code": "600001.SH", "name": "样本", "industry": "制造", "circ_mv": 500.0,
        "pool_score": 80.0,
    }])
    quotes = pd.DataFrame([{
        "ts_code": "600001.SH", "pre_close": 10.0, "open": 10.1, "high": 11.05,
        "low": 10.0, "close": 11.0, "vol": 300.0, "amount": 300.0,
        "trade_time": "103000", "source": "测试", "fetched_at": "2026-09-29T10:30:00",
    }])
    settings = {
        "min_volume_ratio": 2, "min_amount_ratio": 1.8, "min_daily_return": .05,
        "min_body_pct": .04, "min_close_location": .75, "min_turnover_rate": 3,
        "max_turnover_rate": 25,
    }
    alerts, monitored = evaluate_realtime_alerts(pool, daily, quotes, settings)
    assert len(monitored) == 1
    assert alerts["ts_code"].tolist() == ["600001.SH"]
    assert monitored.iloc[0]["volume_ratio"] == 3
    assert monitored.iloc[0]["observation_level"] == "🔴 盘中重点观察"
    assert monitored.iloc[0]["volume_pace_ratio"] == 12


def test_realtime_pace_can_raise_an_early_warning_before_close() -> None:
    dates = pd.bdate_range("2026-08-01", periods=20)
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "600001.SH", "vol": 100.0, "amount": 100.0,
    })
    pool = pd.DataFrame([{
        "ts_code": "600001.SH", "name": "样本", "industry": "制造", "circ_mv": 500.0,
        "pool_score": 80.0,
    }])
    quotes = pd.DataFrame([{
        "ts_code": "600001.SH", "pre_close": 10.0, "open": 10.05, "high": 10.40,
        "low": 10.0, "close": 10.35, "vol": 40.0, "amount": 40.0,
        "trade_time": "103000", "source": "测试", "fetched_at": "2026-09-29T10:30:00",
    }])
    settings = {
        "min_volume_ratio": 2, "min_amount_ratio": 1.8, "min_daily_return": .05,
        "min_body_pct": .04, "min_close_location": .75, "min_turnover_rate": 3,
        "max_turnover_rate": 25, "live_min_elapsed_minutes": 30,
        "early_volume_pace": 1.5, "early_amount_pace": 1.4,
        "early_daily_return": .03, "early_body_pct": .02,
        "early_close_location": .65, "early_turnover_pace": 2,
    }
    alerts, monitored = evaluate_realtime_alerts(pool, daily, quotes, settings)
    assert alerts["ts_code"].tolist() == ["600001.SH"]
    assert not bool(monitored.iloc[0]["priority_alert"])
    assert bool(monitored.iloc[0]["early_warning"])
    assert monitored.iloc[0]["observation_level"] == "🟠 可能启动"
