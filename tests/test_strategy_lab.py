import json
from datetime import date

import pandas as pd

from jkquant.backtest.strategy_lab import experiment_grid, run_experiments


def test_default_experiment_grid_has_264_combinations() -> None:
    grid = experiment_grid({})
    assert len(grid) == 264
    assert {item["take_profit"] for item in grid} == {
        0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40,
    }
    assert {item["confirmation_days"] for item in grid} == {1, 2, 3}


def test_strategy_lab_runs_and_resumes_without_duplicates(tmp_path) -> None:
    dates = pd.date_range("2025-09-01", periods=5, freq="B")
    daily = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ", "open": 10.0,
        "high": 10.1, "low": 9.9, "close": 10.0, "pre_close": 10.0,
    })
    rankings = pd.DataFrame({
        "trade_date": [value.date() for value in dates], "ts_code": "000001.SZ", "rank": 24,
    })
    settings = {
        "take_profit_start": 0.20, "take_profit_end": 0.20, "take_profit_step": 0.02,
        "confirmation_days": [1], "entry_ranks": [25], "exit_ranks": [20],
    }
    costs = {"commission_buy": 0, "commission_sell": 0, "stamp_tax": 0, "slippage": 0}
    progress_path, first = run_experiments(
        daily, rankings, {}, dates[0].date(), dates[-1].date(), 1_000_000,
        costs, settings, tmp_path,
    )
    _, resumed = run_experiments(
        daily, rankings, {}, dates[0].date(), dates[-1].date(), 1_000_000,
        costs, settings, tmp_path,
    )
    assert len(first) == len(resumed) == 1
    assert first.iloc[0]["total_trade_count"] == 1
    assert json.loads(progress_path.read_text(encoding="utf-8"))["status"] == "completed"
