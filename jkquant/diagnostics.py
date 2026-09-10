from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from .factors import FACTOR_COLUMNS, calculate_factors


@dataclass
class FactorDiagnosticResult:
    summary: pd.DataFrame
    daily_ic: pd.DataFrame
    quantiles: pd.DataFrame
    correlations: pd.DataFrame
    metadata: dict[str, Any]


def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return np.nan
    return float(left.corr(right))


def _eligible_rows(
    daily: pd.DataFrame, basic: pd.DataFrame, config: dict[str, Any],
) -> pd.DataFrame:
    frame = calculate_factors(daily)
    grouped_price = frame.groupby("ts_code", group_keys=False)["factor_price"]
    horizons = [int(value) for value in config["factor_diagnostics"]["horizons"]]
    for horizon in horizons:
        frame[f"forward_return_{horizon}d"] = grouped_price.shift(-horizon) / frame["factor_price"] - 1

    metadata = [
        column for column in ("ts_code", "name", "list_date", "delist_date")
        if column in basic
    ]
    frame = frame.merge(basic[metadata].drop_duplicates("ts_code"), on="ts_code", how="left")
    market = config["market"]
    if market.get("exclude_st", True) and "name" in frame:
        frame = frame[~frame["name"].fillna("").str.upper().str.contains("ST")]
    if "delist_date" in frame:
        delist = pd.to_datetime(frame["delist_date"], errors="coerce")
        frame = frame[delist.isna() | delist.gt(frame["trade_date"])]
    if "list_date" in frame:
        list_days = frame["trade_date"] - pd.to_datetime(frame["list_date"], errors="coerce")
        frame = frame[list_days.dt.days.ge(int(market["min_list_days"]))]
    frame = frame[
        frame["amount"].gt(float(market["min_amount"]))
        & frame["vol"].gt(0)
    ]
    return frame.dropna(subset=FACTOR_COLUMNS).copy()


def calculate_factor_diagnostics(
    daily: pd.DataFrame, basic: pd.DataFrame, config: dict[str, Any],
    start_date: date, end_date: date,
) -> FactorDiagnosticResult:
    """Measure point-in-time factor efficacy against subsequent returns."""
    settings = config["factor_diagnostics"]
    horizons = [int(value) for value in settings["horizons"]]
    quantile_count = int(settings.get("quantiles", 5))
    if quantile_count < 2:
        raise ValueError("factor_diagnostics.quantiles 必须至少为 2")
    frame = _eligible_rows(daily, basic, config)
    frame = frame[frame["trade_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    if frame.empty:
        raise RuntimeError("诊断区间内没有满足条件的因子样本")

    directions = {
        name: int(config["strategy"]["factors"][name]["direction"])
        for name in FACTOR_COLUMNS
    }
    daily_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        target = f"forward_return_{horizon}d"
        horizon_frame = frame.dropna(subset=[target])
        for factor in FACTOR_COLUMNS:
            values = horizon_frame[["trade_date", factor, target]].dropna().copy()
            values["oriented_factor"] = values[factor] * directions[factor]
            values["factor_rank"] = values.groupby("trade_date")["oriented_factor"].rank(pct=True)
            values["quantile"] = np.minimum(
                np.ceil(values["factor_rank"] * quantile_count).astype(int), quantile_count,
            )
            for trade_date, day in values.groupby("trade_date", sort=True):
                if len(day) < max(20, quantile_count * 2):
                    continue
                daily_rows.append({
                    "trade_date": pd.Timestamp(trade_date), "factor": factor,
                    "horizon": horizon, "sample_count": len(day),
                    "ic": _safe_correlation(day["oriented_factor"], day[target]),
                    "rank_ic": _safe_correlation(day["oriented_factor"].rank(), day[target].rank()),
                })
            daily_quantiles = (
                values.groupby(["trade_date", "quantile"], as_index=False)[target].mean()
            )
            grouped_quantiles = daily_quantiles.groupby("quantile")[target].agg(["mean", "count"])
            observation_counts = values.groupby("quantile").size().to_dict()
            for quantile, row in grouped_quantiles.iterrows():
                quantile_rows.append({
                    "factor": factor, "horizon": horizon, "quantile": int(quantile),
                    "mean_forward_return": float(row["mean"]),
                    "trading_dates": int(row["count"]),
                    "sample_count": int(observation_counts.get(quantile, 0)),
                })

            factor_daily = pd.DataFrame(
                row for row in daily_rows
                if row["factor"] == factor and row["horizon"] == horizon
            )
            quantile_map = {
                int(row["quantile"]): float(row["mean_forward_return"])
                for row in quantile_rows
                if row["factor"] == factor and row["horizon"] == horizon
            }
            rank_ic = factor_daily["rank_ic"].dropna() if not factor_daily.empty else pd.Series(dtype=float)
            ic = factor_daily["ic"].dropna() if not factor_daily.empty else pd.Series(dtype=float)
            rank_std = float(rank_ic.std(ddof=1)) if len(rank_ic) > 1 else 0.0
            summary_rows.append({
                "factor": factor, "horizon": horizon,
                "trading_dates": int(rank_ic.size), "sample_count": int(len(values)),
                "mean_ic": float(ic.mean()) if not ic.empty else np.nan,
                "mean_rank_ic": float(rank_ic.mean()) if not rank_ic.empty else np.nan,
                "rank_ic_std": rank_std,
                "rank_ic_ir": float(rank_ic.mean() / rank_std) if rank_std > 0 else np.nan,
                "positive_rank_ic_rate": float(rank_ic.gt(0).mean()) if not rank_ic.empty else np.nan,
                "bottom_quantile_return": quantile_map.get(1, np.nan),
                "top_quantile_return": quantile_map.get(quantile_count, np.nan),
                "top_bottom_spread": quantile_map.get(quantile_count, np.nan) - quantile_map.get(1, np.nan),
            })

    correlation_sum: pd.DataFrame | None = None
    correlation_days = 0
    oriented = frame[["trade_date", *FACTOR_COLUMNS]].copy()
    for factor in FACTOR_COLUMNS:
        oriented[factor] *= directions[factor]
    for _, day in oriented.groupby("trade_date"):
        if len(day) < 20:
            continue
        ranked = day[FACTOR_COLUMNS].rank()
        variable_columns = ranked.columns[ranked.nunique(dropna=True).ge(2)]
        correlation = ranked[variable_columns].corr().reindex(
            index=FACTOR_COLUMNS, columns=FACTOR_COLUMNS,
        )
        correlation_sum = correlation if correlation_sum is None else correlation_sum.add(correlation, fill_value=0)
        correlation_days += 1
    correlations = (
        correlation_sum / correlation_days if correlation_sum is not None and correlation_days else
        pd.DataFrame(index=FACTOR_COLUMNS, columns=FACTOR_COLUMNS, dtype=float)
    )
    metadata = {
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizons": horizons, "quantiles": quantile_count,
        "factor_direction": directions, "eligible_rows": int(len(frame)),
        "correlation_dates": correlation_days,
        "st_filter_mode": "current_name_approximation",
        "method": "按交易日横截面计算；因子按策略方向统一后与未来收益相关",
    }
    return FactorDiagnosticResult(
        pd.DataFrame(summary_rows), pd.DataFrame(daily_rows),
        pd.DataFrame(quantile_rows), correlations, metadata,
    )


def write_factor_diagnostics(result: FactorDiagnosticResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    result.daily_ic.to_csv(output_dir / "daily_ic.csv", index=False, encoding="utf-8-sig")
    result.quantiles.to_csv(output_dir / "quantiles.csv", index=False, encoding="utf-8-sig")
    result.correlations.to_csv(output_dir / "correlations.csv", encoding="utf-8-sig")
    (output_dir / "metadata.json").write_text(
        json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return output_dir
