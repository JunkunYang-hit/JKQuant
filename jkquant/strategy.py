from __future__ import annotations

from typing import Any

import pandas as pd


def is_star_market(codes: pd.Series, markets: pd.Series | None = None) -> pd.Series:
    """STAR Market shares need a separate trading permission (688/689.SH)."""
    by_code = codes.fillna("").astype(str).str.match(r"^68[89]\d{3}\.SH$")
    if markets is None:
        return by_code
    return by_code | markets.fillna("").astype(str).eq("科创板")


def is_chinext_market(codes: pd.Series, markets: pd.Series | None = None) -> pd.Series:
    """ChiNext shares need a separate trading permission (300/301.SZ)."""
    by_code = codes.fillna("").astype(str).str.match(r"^30[01]\d{3}\.SZ$")
    if markets is None:
        return by_code
    return by_code | markets.fillna("").astype(str).eq("创业板")


def is_bse_market(codes: pd.Series, markets: pd.Series | None = None) -> pd.Series:
    """Beijing Stock Exchange shares require a separate investor permission."""
    by_code = codes.fillna("").astype(str).str.endswith(".BJ")
    if markets is None:
        return by_code
    return by_code | markets.fillna("").astype(str).isin(["北交所", "北证A股"])


def select_stocks(
    factors: pd.DataFrame,
    basic: pd.DataFrame,
    config: dict[str, Any],
    top_k: int | None = None,
    use_current_metadata: bool = False,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Filter and rank the latest available cross-section."""
    if factors.empty:
        raise ValueError("没有可用于选股的日线数据")
    selection_date = factors["trade_date"].max()
    latest = factors.loc[factors["trade_date"].eq(selection_date)].copy()
    universe_count = len(latest)
    latest = latest.merge(basic, on="ts_code", how="left")
    market = config["market"]
    if market.get("exclude_star_market", True):
        latest = latest[~is_star_market(latest["ts_code"], latest.get("market"))].copy()
    if market.get("exclude_chinext_market", True):
        latest = latest[~is_chinext_market(latest["ts_code"], latest.get("market"))].copy()
    if market.get("exclude_bse_market", True):
        latest = latest[~is_bse_market(latest["ts_code"], latest.get("market"))].copy()
    if market.get("exclude_st", True) and use_current_metadata:
        latest = latest[~latest["name"].fillna("").str.upper().str.contains("ST")]
    if "delist_date" in latest:
        delist_date = pd.to_datetime(latest["delist_date"], errors="coerce")
        latest = latest[delist_date.isna() | delist_date.gt(selection_date)]
    list_days = (selection_date - pd.to_datetime(latest["list_date"], errors="coerce")).dt.days
    derived = latest.get("metadata_source", pd.Series("tushare", index=latest.index)).eq("daily_derived")
    minimum_observations = int(int(market["min_list_days"]) * 5 / 7)
    enough_history = latest["history_count"].ge(minimum_observations)
    latest = latest[list_days.ge(int(market["min_list_days"])) | (derived & enough_history)]
    latest = latest[latest["amount"].gt(float(market["min_amount"]))]
    # Tushare 停牌日没有 daily 行；成交量为零也视为不可交易。
    latest = latest[latest["vol"].gt(0)]

    factor_config = config["strategy"]["factors"]
    required = list(factor_config)
    latest = latest.dropna(subset=required).copy()
    eligible_count = len(latest)
    category_columns: dict[str, list[tuple[str, float]]] = {}
    for factor, spec in factor_config.items():
        rank = latest[factor].rank(method="average", pct=True)
        score_column = f"{factor}_score"
        latest[score_column] = rank if int(spec["direction"]) == 1 else 1 - rank
        category_columns.setdefault(spec["category"], []).append((score_column, float(spec["weight"])))

    for category, columns in category_columns.items():
        weight_sum = sum(weight for _, weight in columns)
        latest[f"{category}_score"] = sum(latest[col] * weight for col, weight in columns) / weight_sum
    weights = config["strategy"]["category_weights"]
    latest["total_score"] = sum(latest[f"{name}_score"] * weight for name, weight in weights.items())
    limit = int(top_k or config["strategy"]["top_k"])
    latest = latest.sort_values("total_score", ascending=False).head(limit)
    latest.insert(0, "rank", range(1, len(latest) + 1))
    columns = ["rank", "trade_date", "ts_code", "name", "total_score"]
    columns += [f"{name}_score" for name in weights]
    columns += ["close", "amount"] + required
    summary: dict[str, int | str] = {
        "trade_date": selection_date.strftime("%Y-%m-%d"),
        "universe_count": universe_count,
        "eligible_count": eligible_count,
    }
    return latest[columns].reset_index(drop=True), summary
