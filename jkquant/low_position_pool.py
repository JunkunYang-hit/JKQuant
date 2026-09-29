"""Point-in-time screen for small-cap, low-position volume breakouts."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jkquant.strategy import is_bse_market, is_chinext_market, is_star_market


POOL_RULE_LABELS = {
    "small_cap_ok": "小盘",
    "not_extended_ok": "低位或此前未明显上涨",
    "consolidation_ok": "震荡收敛",
    "liquidity_ok": "具备基础流动性",
}


def _trading_minutes_elapsed(value: Any) -> float:
    """Return elapsed A-share continuous-auction minutes for an HHMMSS-like value."""
    digits = "".join(character for character in str(value) if character.isdigit())
    if len(digits) < 6:
        return 240.0
    hhmmss = digits[-6:]
    hour, minute = int(hhmmss[:2]), int(hhmmss[2:4])
    clock = hour * 60 + minute
    morning_open, morning_close = 9 * 60 + 30, 11 * 60 + 30
    afternoon_open, afternoon_close = 13 * 60, 15 * 60
    if clock <= morning_open:
        return 0.0
    if clock <= morning_close:
        return float(clock - morning_open)
    if clock < afternoon_open:
        return 120.0
    if clock <= afternoon_close:
        return float(120 + clock - afternoon_open)
    return 240.0


def build_low_position_features(daily: pd.DataFrame, lookback_days: int = 250,
                                consolidation_days: int = 20) -> pd.DataFrame:
    """Build features using only the current bar and information before it."""
    if daily.empty:
        return pd.DataFrame()
    required = {"trade_date", "ts_code", "open", "high", "low", "close", "pct_chg", "vol", "amount"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"日线缺少字段: {', '.join(sorted(missing))}")

    frame = daily.sort_values(["ts_code", "trade_date"]).copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    for column in ("open", "high", "low", "close", "pct_chg", "vol", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["daily_return"] = frame["pct_chg"].clip(lower=-99) / 100
    frame["return_index"] = (1 + frame["daily_return"].fillna(0)).groupby(frame["ts_code"]).cumprod()
    group = frame.groupby("ts_code", sort=False)
    prior_index = group["return_index"].shift()
    min_periods = max(60, lookback_days // 2)
    prior_low = group["return_index"].transform(
        lambda values: values.shift().rolling(lookback_days, min_periods=min_periods).min()
    )
    prior_high = group["return_index"].transform(
        lambda values: values.shift().rolling(lookback_days, min_periods=min_periods).max()
    )
    frame["low_position"] = (
        (prior_index - prior_low) / (prior_high - prior_low).replace(0, np.nan)
    ).clip(0, 1)
    frame["prior_return_60"] = prior_index / group["return_index"].shift(61) - 1
    frame["prior_return_120"] = prior_index / group["return_index"].shift(121) - 1
    frame["consolidation_range"] = group["return_index"].transform(
        lambda values: values.shift().rolling(consolidation_days, min_periods=consolidation_days).max()
        / values.shift().rolling(consolidation_days, min_periods=consolidation_days).min() - 1
    )
    frame["prior_volatility"] = group["daily_return"].transform(
        lambda values: values.shift().rolling(consolidation_days, min_periods=consolidation_days).std()
        * np.sqrt(252)
    )
    prior_volume = group["vol"].transform(
        lambda values: values.shift().rolling(consolidation_days, min_periods=consolidation_days).mean()
    )
    prior_amount = group["amount"].transform(
        lambda values: values.shift().rolling(consolidation_days, min_periods=consolidation_days).mean()
    )
    frame["volume_ratio"] = frame["vol"] / prior_volume.replace(0, np.nan)
    frame["previous_volume"] = group["vol"].shift()
    frame["previous_day_volume_ratio"] = frame["vol"] / frame["previous_volume"].replace(0, np.nan)
    frame["amount_ratio"] = frame["amount"] / prior_amount.replace(0, np.nan)
    frame["body_pct"] = (frame["close"] - frame["open"]) / frame["open"].replace(0, np.nan)
    candle_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["close_location"] = ((frame["close"] - frame["low"]) / candle_range).clip(0, 1).fillna(.5)
    frame["history_count"] = group.cumcount() + 1
    return frame


def screen_low_position_breakouts(
    daily: pd.DataFrame,
    valuation: pd.DataFrame,
    basic: pd.DataFrame,
    settings: dict[str, Any],
    selected_date: pd.Timestamp | str | None = None,
    market_settings: dict[str, Any] | None = None,
    prepared_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return today's priority alerts and the broader watch pool.

    Pool membership describes what was true before today's close. A volume/bullish
    breakout is an alert attached to a pool member, not a prerequisite for joining.
    """
    features = prepared_features if prepared_features is not None else build_low_position_features(
        daily, int(settings.get("lookback_days", 250)), int(settings.get("consolidation_days", 20)),
    )
    if features.empty:
        return features, features
    target = pd.Timestamp(selected_date) if selected_date is not None else features["trade_date"].max()
    day = features[features["trade_date"].eq(target)].copy()
    if day.empty:
        return day, day

    values = valuation.copy()
    if not values.empty:
        values["trade_date"] = pd.to_datetime(values["trade_date"])
        value_columns = [
            column for column in ("trade_date", "ts_code", "circ_mv", "total_mv", "turnover_rate", "volume_ratio")
            if column in values
        ]
        values = values.loc[values["trade_date"].eq(target), value_columns].drop_duplicates("ts_code")
        day = day.merge(values, on=["trade_date", "ts_code"], how="left", suffixes=("", "_tushare"))
    info_columns = [column for column in ("ts_code", "name", "industry", "market", "list_date", "delist_date") if column in basic]
    day = day.merge(basic[info_columns].drop_duplicates("ts_code"), on="ts_code", how="left")
    if "name" not in day:
        day["name"] = ""
    if "industry" not in day:
        day["industry"] = "未分类"
    if "market" not in day:
        day["market"] = ""
    if "circ_mv" not in day:
        day["circ_mv"] = np.nan
    if "turnover_rate" not in day:
        day["turnover_rate"] = np.nan
    for column in ("circ_mv", "turnover_rate"):
        day[column] = pd.to_numeric(day[column], errors="coerce")

    market = market_settings or {}
    eligible = ~day["name"].fillna("").str.contains("ST|退", case=False, regex=True)
    if settings.get("main_board_only", True):
        eligible &= day["market"].fillna("").astype(str).eq("主板")
        eligible &= day["ts_code"].fillna("").astype(str).str.endswith((".SH", ".SZ"))
    if market.get("exclude_star_market", True):
        eligible &= ~is_star_market(day["ts_code"], day["market"])
    if market.get("exclude_chinext_market", True):
        eligible &= ~is_chinext_market(day["ts_code"], day["market"])
    if market.get("exclude_bse_market", True):
        eligible &= ~is_bse_market(day["ts_code"], day["market"])
    eligible &= day["history_count"].ge(int(settings.get("min_history_days", 120)))
    eligible &= day["close"].between(
        float(settings.get("min_price", 3.0)),
        float(settings.get("max_price", 25.0)),
        inclusive="both",
    )
    day = day[eligible].copy()

    min_cap = float(settings.get("min_circ_mv_yi", 10)) * 10_000
    max_cap = float(settings.get("max_circ_mv_yi", 100)) * 10_000
    day["small_cap_ok"] = day["circ_mv"].between(min_cap, max_cap, inclusive="both")
    day["low_position_ok"] = day["low_position"].le(float(settings.get("max_low_position", .30)))
    day["not_risen_ok"] = (
        day["prior_return_60"].le(float(settings.get("max_prior_return_60", .10)))
        & day["prior_return_120"].le(float(settings.get("max_prior_return_120", .20)))
    )
    # A narrow, previously flat stock may sit near the top of its tiny annual range.
    # It remains eligible when the absolute prior returns show that it has not run up.
    day["not_extended_ok"] = day["low_position_ok"] | day["not_risen_ok"]
    day["consolidation_ok"] = (
        day["consolidation_range"].le(float(settings.get("max_consolidation_range", .20)))
        & day["prior_volatility"].le(float(settings.get("max_prior_volatility", .40)))
    )
    day["liquidity_ok"] = (
        day["amount"].ge(float(settings.get("min_amount", market.get("min_amount", 20_000))))
        & day["turnover_rate"].between(
            float(settings.get("pool_min_turnover_rate", .5)),
            float(settings.get("max_turnover_rate", 25)), inclusive="both",
        )
    )
    day["volume_surge_ok"] = (
        day["volume_ratio"].ge(float(settings.get("min_volume_ratio", 2.0)))
        & day["amount_ratio"].ge(float(settings.get("min_amount_ratio", 1.8)))
    )
    day["previous_day_volume_ok"] = day["previous_day_volume_ratio"].ge(
        float(settings.get("min_previous_day_volume_ratio", 1.0))
    )
    day["large_bull_ok"] = (
        day["daily_return"].ge(float(settings.get("min_daily_return", .05)))
        & day["body_pct"].ge(float(settings.get("min_body_pct", .04)))
    )
    day["close_strong_ok"] = day["close_location"].ge(float(settings.get("min_close_location", .75)))
    day["breakout_turnover_ok"] = day["turnover_rate"].between(
        float(settings.get("min_turnover_rate", 3)), float(settings.get("max_turnover_rate", 25)),
        inclusive="both",
    )
    rule_columns = list(POOL_RULE_LABELS)
    day["rule_hits"] = day[rule_columns].sum(axis=1).astype(int)
    day["matched_rules"] = day.apply(
        lambda row: "、".join(label for column, label in POOL_RULE_LABELS.items() if bool(row[column])), axis=1,
    )
    day["failed_rules"] = day.apply(
        lambda row: "、".join(label for column, label in POOL_RULE_LABELS.items() if not bool(row[column])), axis=1,
    )
    day["priority_alert"] = (
        day[rule_columns].all(axis=1) & day["volume_surge_ok"] & day["previous_day_volume_ok"] & day["large_bull_ok"]
        & day["close_strong_ok"] & day["breakout_turnover_ok"]
    )
    day["observation_level"] = np.where(day["priority_alert"], "🔴 重点观察", "备选观察")

    cap_score = (1 - (day["circ_mv"] - min_cap) / max(max_cap - min_cap, 1)).clip(0, 1)
    low_score = (1 - day["low_position"] / max(float(settings.get("max_low_position", .30)), .01)).clip(0, 1)
    return_60_score = (
        1 - day["prior_return_60"].clip(lower=0)
        / max(float(settings.get("max_prior_return_60", .10)), .01)
    ).clip(0, 1)
    return_120_score = (
        1 - day["prior_return_120"].clip(lower=0)
        / max(float(settings.get("max_prior_return_120", .20)), .01)
    ).clip(0, 1)
    not_extended_score = pd.concat(
        [low_score, (return_60_score + return_120_score) / 2], axis=1,
    ).max(axis=1)
    contraction_score = (
        1 - day["consolidation_range"] / max(float(settings.get("max_consolidation_range", .20)), .01)
    ).clip(0, 1)
    liquidity_score = (
        day["amount"] / max(float(settings.get("min_amount", 20_000)), 1) / 3
    ).clip(0, 1)
    day["pool_score"] = 100 * (
        .20 * cap_score + .35 * not_extended_score + .30 * contraction_score + .15 * liquidity_score
    )
    volume_score = (day["volume_ratio"] / max(float(settings.get("min_volume_ratio", 2)), .1) / 2).clip(0, 1)
    candle_score = (day["daily_return"].clip(lower=0) / .10).clip(0, 1)
    day["alert_score"] = 100 * (
        .35 * volume_score + .15 * (day["amount_ratio"] / 3).clip(0, 1)
        + .35 * candle_score + .15 * day["close_location"].clip(0, 1)
    )
    pool = day[day[rule_columns].all(axis=1)].copy()
    pool = pool.sort_values(
        ["priority_alert", "alert_score", "pool_score", "amount"], ascending=False,
    ).head(int(settings.get("max_pool_size", 300))).reset_index(drop=True)
    priority = pool[pool["priority_alert"]].copy().reset_index(drop=True)
    return priority, pool


def evaluate_realtime_alerts(
    pool: pd.DataFrame, daily: pd.DataFrame, quotes: pd.DataFrame, settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overlay an unfinished realtime daily bar on an existing end-of-day watch pool."""
    if pool.empty or quotes.empty:
        return pd.DataFrame(), pd.DataFrame()
    codes = set(pool["ts_code"].astype(str))
    history = daily[daily["ts_code"].astype(str).isin(codes)].sort_values(["ts_code", "trade_date"])
    recent = history.groupby("ts_code", as_index=False).tail(20)
    averages = recent.groupby("ts_code", as_index=False).agg(
        prior_vol_mean=("vol", "mean"), prior_amount_mean=("amount", "mean"),
        previous_volume=("vol", "last"),
    )
    quote_columns = [
        column for column in (
            "ts_code", "pre_close", "open", "high", "low", "close", "vol", "amount",
            "trade_time", "source", "fetched_at",
        ) if column in quotes
    ]
    live = pool.drop(columns=[
        column for column in (
            "pre_close", "open", "high", "low", "close", "vol", "amount", "trade_time",
            "source", "fetched_at", "volume_ratio", "amount_ratio", "body_pct", "close_location",
            "daily_return", "priority_alert", "observation_level", "alert_score",
        ) if column in pool
    ]).merge(quotes[quote_columns], on="ts_code", how="inner").merge(averages, on="ts_code", how="left")
    for column in ("pre_close", "open", "high", "low", "close", "vol", "amount"):
        live[column] = pd.to_numeric(live[column], errors="coerce")
    live["daily_return"] = live["close"] / live["pre_close"].replace(0, np.nan) - 1
    live["body_pct"] = (live["close"] - live["open"]) / live["open"].replace(0, np.nan)
    live["close_location"] = (
        (live["close"] - live["low"]) / (live["high"] - live["low"]).replace(0, np.nan)
    ).clip(0, 1).fillna(.5)
    live["volume_ratio"] = live["vol"] / live["prior_vol_mean"].replace(0, np.nan)
    live["amount_ratio"] = live["amount"] / live["prior_amount_mean"].replace(0, np.nan)
    live["previous_day_volume_ratio"] = live["vol"] / live["previous_volume"].replace(0, np.nan)
    live["trading_minutes"] = live.get("trade_time", pd.Series(index=live.index, dtype=object)).map(
        _trading_minutes_elapsed,
    )
    # Comparing morning cumulative volume with a full-day average would detect
    # breakouts too late.  Project the current trading pace to a full session,
    # while flooring elapsed time to suppress noisy opening-minute extrapolation.
    minimum_minutes = float(settings.get("live_min_elapsed_minutes", 30))
    live["trading_progress"] = live["trading_minutes"].clip(lower=minimum_minutes, upper=240) / 240
    live["volume_pace_ratio"] = live["volume_ratio"] / live["trading_progress"]
    live["amount_pace_ratio"] = live["amount_ratio"] / live["trading_progress"]
    # circ_mv is CNY ten-thousands and vol is lots: vol * pre_close / circ_mv gives turnover percent.
    live["turnover_rate"] = live["vol"] * live["pre_close"] / live["circ_mv"].replace(0, np.nan)
    live["turnover_pace"] = live["turnover_rate"] / live["trading_progress"]
    live["volume_surge_ok"] = (
        live["volume_pace_ratio"].ge(float(settings.get("min_volume_ratio", 2)))
        & live["amount_pace_ratio"].ge(float(settings.get("min_amount_ratio", 1.8)))
    )
    live["previous_day_volume_ok"] = live["previous_day_volume_ratio"].ge(
        float(settings.get("min_previous_day_volume_ratio", 1.0))
    )
    live["large_bull_ok"] = (
        live["daily_return"].ge(float(settings.get("min_daily_return", .05)))
        & live["body_pct"].ge(float(settings.get("min_body_pct", .04)))
    )
    live["close_strong_ok"] = live["close_location"].ge(float(settings.get("min_close_location", .75)))
    live["breakout_turnover_ok"] = (
        live["turnover_pace"].ge(float(settings.get("min_turnover_rate", 3)))
        & live["turnover_rate"].le(float(settings.get("max_turnover_rate", 25)))
    )
    live["priority_alert"] = (
        live["volume_surge_ok"] & live["previous_day_volume_ok"] & live["large_bull_ok"]
        & live["close_strong_ok"] & live["breakout_turnover_ok"]
    )
    live["early_warning"] = (
        live["volume_pace_ratio"].ge(float(settings.get("early_volume_pace", 1.5)))
        & live["amount_pace_ratio"].ge(float(settings.get("early_amount_pace", 1.4)))
        & live["daily_return"].ge(float(settings.get("early_daily_return", .03)))
        & live["body_pct"].ge(float(settings.get("early_body_pct", .02)))
        & live["close_location"].ge(float(settings.get("early_close_location", .65)))
        & live["turnover_pace"].ge(float(settings.get("early_turnover_pace", 2)))
    )
    live["observation_level"] = np.select(
        [live["priority_alert"], live["early_warning"]],
        ["🔴 盘中重点观察", "🟠 可能启动"],
        default="盘中观察",
    )
    volume_score = (
        live["volume_pace_ratio"] / max(float(settings.get("min_volume_ratio", 2)), .1) / 2
    ).clip(0, 1)
    candle_score = (live["daily_return"].clip(lower=0) / .10).clip(0, 1)
    live["alert_score"] = 100 * (
        .35 * volume_score + .15 * (live["amount_pace_ratio"] / 3).clip(0, 1)
        + .35 * candle_score + .15 * live["close_location"]
    )
    live = live.sort_values(
        ["priority_alert", "early_warning", "alert_score", "pool_score"], ascending=False,
    ).reset_index(drop=True)
    alerts = live[live["priority_alert"] | live["early_warning"]].copy().reset_index(drop=True)
    return alerts, live
