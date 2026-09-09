from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def calculate_metrics(daily: pd.DataFrame) -> dict[str, Any]:
    returns = daily["net_return"].fillna(0.0)
    benchmark_returns = daily["benchmark_return"].fillna(0.0)
    periods = len(returns)
    years = periods / 252 if periods else 0
    cumulative = float(daily["equity"].iloc[-1] - 1) if periods else 0.0
    benchmark = float(daily["benchmark_equity"].iloc[-1] - 1) if periods else 0.0
    annualized = (1 + cumulative) ** (1 / years) - 1 if years > 0 and cumulative > -1 else -1.0
    volatility = float(returns.std(ddof=1) * math.sqrt(252)) if periods > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(252)) if returns.std(ddof=1) > 0 else 0.0
    yearly = (
        daily.set_index("trade_date")["net_return"]
        .groupby(lambda value: value.year)
        .apply(lambda values: float((1 + values).prod() - 1))
        .to_dict()
    )
    return {
        "annualized_return": float(annualized),
        "cumulative_return": cumulative,
        "benchmark_return": benchmark,
        "excess_return": cumulative - benchmark,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(daily["drawdown"].min()) if periods else 0.0,
        "annualized_volatility": volatility,
        "average_turnover": float(daily["turnover"].mean()) if periods else 0.0,
        "win_rate": float((returns > 0).mean()) if periods else 0.0,
        "trading_days": periods,
        "yearly_returns": {str(key): value for key, value in yearly.items()},
    }

