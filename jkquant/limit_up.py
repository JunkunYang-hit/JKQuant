"""Daily-close limit-up board observations, not intraday order-book signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from jkquant.strategy import is_chinext_market, is_star_market


def build_board_history(
    daily: pd.DataFrame, limits: pd.DataFrame, basic: pd.DataFrame,
    valuation: pd.DataFrame | None = None,
    exclude_star_market: bool = True,
    exclude_chinext_market: bool = True,
) -> pd.DataFrame:
    """Return sealed-at-close boards with streaks and point-in-time price features.

    A suspension breaks a streak. Current stock names/industries are descriptive only;
    they must not be treated as historical industry/ST membership.
    """
    if daily.empty or limits.empty:
        return pd.DataFrame()
    dates = pd.Index(sorted(pd.to_datetime(daily.trade_date).unique()))
    columns = ["trade_date", "ts_code", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]
    prices = daily[[c for c in columns if c in daily]].copy()
    prices["trade_date"] = pd.to_datetime(prices.trade_date)
    prices = prices.sort_values(["ts_code", "trade_date"])
    group = prices.groupby("ts_code", sort=False)
    # pct_chg uses the ex-right pre-close; chaining it avoids raw-price jumps.
    daily_return = pd.to_numeric(prices.pct_chg, errors="coerce").fillna(0).clip(lower=-99) / 100
    prices["return_index"] = (1 + daily_return).groupby(prices.ts_code).cumprod()
    group = prices.groupby("ts_code", sort=False)
    prices["pre20_range"] = group.return_index.transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).max()
        / s.shift(1).rolling(20, min_periods=20).min() - 1
    )
    prior_low = group.return_index.transform(lambda s: s.shift(1).rolling(120, min_periods=60).min())
    prior_high = group.return_index.transform(lambda s: s.shift(1).rolling(120, min_periods=60).max())
    previous_index = group.return_index.shift()
    prices["low_position"] = ((previous_index - prior_low) / (prior_high - prior_low).replace(0, np.nan)).clip(0, 1)
    prior_vol = group.vol.transform(lambda s: s.shift(1).rolling(20, min_periods=20).mean())
    prices["volume_surge"] = prices.vol / prior_vol.replace(0, np.nan)
    prices["market_day"] = dates.get_indexer(prices.trade_date)

    limit_cols = limits[["trade_date", "ts_code", "up_limit"]].copy()
    limit_cols["trade_date"] = pd.to_datetime(limit_cols.trade_date)
    prices = prices.merge(limit_cols, on=["trade_date", "ts_code"], how="inner")
    for col in ("open", "high", "low", "close", "pre_close", "up_limit", "amount"):
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    # Exclude 5% ST boards and no-limit/invalid records. Ordinary/20%/30% boards remain.
    eligible = prices.up_limit.div(prices.pre_close).ge(1.07) & prices.amount.gt(0)
    sealed = prices.close.ge(prices.up_limit - 0.011) & prices.close.le(prices.up_limit + 0.011)
    boards = prices.loc[eligible & sealed].copy()
    if boards.empty:
        return boards
    info = basic[[c for c in ("ts_code", "name", "industry", "market") if c in basic]].drop_duplicates("ts_code")
    boards = boards.merge(info, on="ts_code", how="left")
    if "name" not in boards:
        boards["name"] = ""
    if "industry" not in boards:
        boards["industry"] = "未分类"
    boards = boards[~boards.name.fillna("").str.contains("ST|退", case=False, regex=True)].copy()
    if exclude_star_market:
        boards = boards[~is_star_market(boards.ts_code, boards.get("market"))].copy()
    if exclude_chinext_market:
        boards = boards[~is_chinext_market(boards.ts_code, boards.get("market"))].copy()
    boards["industry"] = boards.industry.fillna("未分类")
    boards["one_word"] = (boards.open.ge(boards.up_limit - .011) & boards.low.ge(boards.up_limit - .011))
    boards = boards.sort_values(["ts_code", "trade_date"])
    previous_board_day = boards.groupby("ts_code").market_day.shift()
    streak_start = previous_board_day.ne(boards.market_day - 1)
    boards["streak"] = boards.groupby(["ts_code", streak_start.groupby(boards.ts_code).cumsum()]).cumcount() + 1
    boards["industry_boards"] = boards.groupby(["trade_date", "industry"]).ts_code.transform("size")
    if valuation is not None and not valuation.empty:
        fields = [c for c in ("trade_date", "ts_code", "turnover_rate", "total_mv") if c in valuation]
        value = valuation[fields].copy()
        value["trade_date"] = pd.to_datetime(value.trade_date)
        boards = boards.merge(value, on=["trade_date", "ts_code"], how="left")
    if "turnover_rate" not in boards:
        boards["turnover_rate"] = np.nan
    return boards.sort_values(["trade_date", "streak", "amount"], ascending=[True, False, False]).reset_index(drop=True)


def promotion_summary(boards: pd.DataFrame, selected_date: pd.Timestamp) -> pd.DataFrame:
    """Historical next-trading-day 1→2 and 2→3 rates known by selected_date."""
    if boards.empty:
        return pd.DataFrame()
    cutoff = pd.Timestamp(selected_date)
    history = _labeled_history(boards, cutoff)
    result = history[history.streak.isin([1, 2])].groupby("streak").promoted.agg(["sum", "count", "mean"]).reset_index()
    return result.rename(columns={"streak": "当前板数", "sum": "晋级数", "count": "样本数", "mean": "历史晋级率"})


def _labeled_history(boards: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    history = boards[boards.trade_date.lt(cutoff)].copy()
    history["next_market_day"] = history.market_day + 1
    next_board = boards[["market_day", "ts_code", "streak"]].rename(
        columns={"market_day": "next_market_day", "streak": "next_streak"},
    )
    history = history.merge(next_board, on=["next_market_day", "ts_code"], how="left")
    history["promoted"] = history.next_streak.eq(history.streak + 1)
    return history


_FEATURES = ("pre20_range", "low_position", "volume_surge", "turnover_rate",
             "industry_boards", "one_word", "amount")


def _feature_array(frame: pd.DataFrame) -> np.ndarray:
    values = frame[list(_FEATURES)].astype(float).to_numpy(copy=True)
    values[:, 0] = np.clip(values[:, 0], 0, 2)
    values[:, 1] = np.clip(values[:, 1], 0, 1)
    values[:, 2] = np.log1p(np.clip(values[:, 2], 0, 50))
    values[:, 3] = np.clip(values[:, 3], 0, 60)
    values[:, 4] = np.log1p(np.clip(values[:, 4], 0, 100))
    values[:, 6] = np.log1p(np.clip(values[:, 6], 0, None))
    return values


def _fit_logit(train: pd.DataFrame):
    raw = _feature_array(train)
    median = np.nanmedian(raw, axis=0)
    median = np.nan_to_num(median, nan=0.0)
    raw = np.where(np.isfinite(raw), raw, median)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    x = np.column_stack([np.ones(len(raw)), (raw - mean) / scale])
    y = train.promoted.astype(float).to_numpy()
    prevalence = np.clip(y.mean(), .001, .999)
    coefficients = np.zeros(x.shape[1])
    coefficients[0] = np.log(prevalence / (1 - prevalence))
    penalty = np.diag([0.0] + [2.0] * (x.shape[1] - 1))
    for _ in range(15):
        probability = 1 / (1 + np.exp(-np.clip(x @ coefficients, -30, 30)))
        weights = np.clip(probability * (1 - probability), 1e-5, None)
        step = np.linalg.solve((x.T * weights) @ x + penalty,
                               x.T @ (y - probability) - penalty @ coefficients)
        coefficients += step
        if np.max(np.abs(step)) < 1e-6:
            break
    return median, mean, scale, coefficients


def _predict_logit(frame: pd.DataFrame, model) -> np.ndarray:
    median, mean, scale, coefficients = model
    raw = _feature_array(frame)
    raw = np.where(np.isfinite(raw), raw, median)
    x = np.column_stack([np.ones(len(raw)), (raw - mean) / scale])
    return 1 / (1 + np.exp(-np.clip(x @ coefficients, -30, 30)))


def promotion_candidates(boards: pd.DataFrame, selected_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward stage-specific ranking; report temporal holdout precision.

    The model sees only boards whose next-trading-day outcome is known before or
    on selected_date. A held-out most-recent quarter checks whether ranking helped.
    """
    cutoff = pd.Timestamp(selected_date)
    day = boards[boards.trade_date.eq(cutoff) & boards.streak.isin([1, 2])].copy()
    if day.empty:
        return day, pd.DataFrame()
    labeled = _labeled_history(boards, cutoff)
    day["estimated_promotion"] = np.nan
    validations = []
    for stage in (1, 2):
        history = labeled[labeled.streak.eq(stage)].sort_values("trade_date")
        selected = day.streak.eq(stage)
        if not selected.any() or len(history) < 100:
            continue
        dates = sorted(history.trade_date.unique())
        split = dates[max(int(len(dates) * .75), 1)]
        older, holdout = history[history.trade_date.lt(split)], history[history.trade_date.ge(split)].copy()
        if len(older) >= 75 and len(holdout) >= 20:
            holdout["estimate"] = _predict_logit(holdout, _fit_logit(older))
            holdout["top_fifth"] = holdout.groupby("trade_date").estimate.rank(pct=True).ge(.8)
            top = holdout[holdout.top_fifth]
            validations.append({"晋级": f"{stage}→{stage + 1}", "验证样本数": len(holdout),
                                "整体晋级率": float(holdout.promoted.mean()),
                                "每日评分前20%晋级率": float(top.promoted.mean()) if len(top) else np.nan,
                                "前20%样本数": len(top),
                                "验证起始日": str(pd.Timestamp(split).date())})
        day.loc[selected, "estimated_promotion"] = _predict_logit(day[selected], _fit_logit(history))
    return day, pd.DataFrame(validations)
