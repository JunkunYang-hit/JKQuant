from __future__ import annotations

import numpy as np
import pandas as pd

from jkquant.low_position_pool import screen_low_position_breakouts


def _market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = pd.bdate_range("2025-01-02", periods=261)
    base = np.concatenate([
        np.linspace(1.0, 1.30, 150),
        np.linspace(1.30, 1.01, 80),
        1.01 + np.sin(np.linspace(0, 2 * np.pi, 30)) * .006,
    ])
    base = np.r_[base, base[-1] * 1.08]
    rows = []
    for code in ("600001.SH", "600002.SH", "600003.SH"):
        price_path = np.linspace(1.0, 1.05, len(dates)) if code == "600003.SH" else base
        close = price_path * (10 if code != "600002.SH" else 20)
        returns = pd.Series(close).pct_change().fillna(0) * 100
        for index, day in enumerate(dates):
            signal = code != "600003.SH" and index == len(dates) - 1
            open_price = close[index - 1] * 1.01 if signal else close[index] * .998
            rows.append({
                "trade_date": day, "ts_code": code,
                "open": open_price, "high": close[index] * (1.005 if signal else 1.004),
                "low": min(open_price, close[index]) * .997, "close": close[index],
                "pct_chg": returns.iloc[index], "vol": 300 if signal else 100,
                "amount": 90_000 if signal else 30_000,
            })
    signal_date = dates[-1]
    valuation = pd.DataFrame([
        {"trade_date": signal_date, "ts_code": "600001.SH", "circ_mv": 500_000, "turnover_rate": 8.0},
        {"trade_date": signal_date, "ts_code": "600002.SH", "circ_mv": 5_000_000, "turnover_rate": 8.0},
        {"trade_date": signal_date, "ts_code": "600003.SH", "circ_mv": 400_000, "turnover_rate": 2.0},
    ])
    basic = pd.DataFrame([
        {"ts_code": "600001.SH", "name": "小盘样本", "industry": "制造", "market": "主板"},
        {"ts_code": "600002.SH", "name": "大盘样本", "industry": "制造", "market": "主板"},
        {"ts_code": "600003.SH", "name": "长期横盘", "industry": "制造", "market": "主板"},
    ])
    return pd.DataFrame(rows), valuation, basic, signal_date


def _settings() -> dict:
    return {
        "lookback_days": 250, "consolidation_days": 20, "min_history_days": 120,
        "min_circ_mv_yi": 10, "max_circ_mv_yi": 100, "max_low_position": .45,
        "min_price": 3.0, "max_price": 25.0,
        "max_prior_return_60": .10, "max_prior_return_120": .20,
        "max_consolidation_range": .20, "max_prior_volatility": .40,
        "min_volume_ratio": 2, "min_amount_ratio": 1.8,
        "min_daily_return": .05, "min_body_pct": .04, "min_close_location": .75,
        "pool_min_turnover_rate": .5, "min_turnover_rate": 3,
        "max_turnover_rate": 25, "min_amount": 20_000, "max_pool_size": 300,
    }


def test_low_position_pool_requires_small_cap_and_breakout_rules() -> None:
    daily, valuation, basic, signal_date = _market()
    priority, pool = screen_low_position_breakouts(
        daily, valuation, basic, _settings(), signal_date,
        {"exclude_star_market": True, "exclude_chinext_market": True},
    )
    assert priority["ts_code"].tolist() == ["600001.SH"]
    assert bool(priority.iloc[0]["volume_surge_ok"])
    assert set(pool["ts_code"]) == {"600001.SH", "600003.SH"}
    flat = pool[pool["ts_code"].eq("600003.SH")].iloc[0]
    assert not bool(flat["low_position_ok"])
    assert bool(flat["not_risen_ok"])
    assert not bool(flat["priority_alert"])
    assert "600002.SH" not in set(pool["ts_code"])


def test_later_prices_do_not_change_an_earlier_screen() -> None:
    daily, valuation, basic, signal_date = _market()
    earlier = signal_date - pd.offsets.BDay(1)
    earlier_values = valuation.assign(trade_date=earlier)
    first = screen_low_position_breakouts(daily, earlier_values, basic, _settings(), earlier)[1]
    changed = daily.copy()
    changed.loc[changed["trade_date"].eq(signal_date), "pct_chg"] = 99
    second = screen_low_position_breakouts(changed, earlier_values, basic, _settings(), earlier)[1]
    pd.testing.assert_frame_equal(first, second)


def test_watch_pool_keeps_only_prices_from_three_to_twenty_five() -> None:
    daily, valuation, basic, signal_date = _market()
    target = daily["trade_date"].eq(signal_date) & daily["ts_code"].eq("600003.SH")
    for price, expected in ((2.99, False), (3.0, True), (25.0, True), (25.01, False)):
        changed = daily.copy()
        changed.loc[target, "close"] = price
        pool = screen_low_position_breakouts(
            changed, valuation, basic, _settings(), signal_date,
            {"exclude_star_market": True, "exclude_chinext_market": True},
        )[1]
        assert ("600003.SH" in set(pool["ts_code"])) is expected


def test_priority_requires_volume_to_reach_previous_trading_day() -> None:
    daily, valuation, basic, signal_date = _market()
    code = daily["ts_code"].eq("600001.SH")
    history_dates = sorted(daily.loc[code, "trade_date"].unique())
    previous_twenty = history_dates[-21:-1]
    daily.loc[code & daily["trade_date"].isin(previous_twenty), "vol"] = 50
    daily.loc[code & daily["trade_date"].eq(previous_twenty[-1]), "vol"] = 1_000
    daily.loc[code & daily["trade_date"].eq(signal_date), "vol"] = 200
    priority, pool = screen_low_position_breakouts(
        daily, valuation, basic, {**_settings(), "min_previous_day_volume_ratio": 1.0}, signal_date,
    )
    sample = pool[pool["ts_code"].eq("600001.SH")].iloc[0]
    assert bool(sample["volume_surge_ok"])
    assert not bool(sample["previous_day_volume_ok"])
    assert "600001.SH" not in set(priority["ts_code"])


def test_watch_pool_can_be_restricted_to_permission_free_main_board() -> None:
    daily, valuation, basic, signal_date = _market()
    bj_daily = daily[daily["ts_code"].eq("600001.SH")].copy()
    bj_daily["ts_code"] = "920001.BJ"
    bj_value = valuation[valuation["ts_code"].eq("600001.SH")].copy()
    bj_value["ts_code"] = "920001.BJ"
    bj_basic = pd.DataFrame([{
        "ts_code": "920001.BJ", "name": "北交样本", "industry": "制造", "market": "北交所",
    }])
    priority, pool = screen_low_position_breakouts(
        pd.concat([daily, bj_daily], ignore_index=True),
        pd.concat([valuation, bj_value], ignore_index=True),
        pd.concat([basic, bj_basic], ignore_index=True),
        {**_settings(), "main_board_only": True},
        signal_date,
        {"exclude_star_market": True, "exclude_chinext_market": True, "exclude_bse_market": True},
    )
    assert "920001.BJ" not in set(pool["ts_code"])
    assert "920001.BJ" not in set(priority["ts_code"])
