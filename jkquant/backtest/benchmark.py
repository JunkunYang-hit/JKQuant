from __future__ import annotations

import pandas as pd


def benchmark_return_map(frame: pd.DataFrame | None) -> dict[pd.Timestamp, float]:
    if frame is None or frame.empty:
        return {}
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    if "pct_chg" in data:
        values = pd.to_numeric(data["pct_chg"], errors="coerce") / 100.0
    else:
        values = pd.to_numeric(data["close"], errors="coerce") / pd.to_numeric(
            data["pre_close"], errors="coerce"
        ) - 1
    return dict(zip(data["trade_date"], values.fillna(0.0), strict=True))
