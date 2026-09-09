from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .strategy_suite import StrategySpec, run_event_strategy

LOGGER = logging.getLogger(__name__)


def experiment_grid(settings: dict[str, Any]) -> list[dict[str, Any]]:
    start = int(round(float(settings.get("take_profit_start", 0.20)) * 100))
    end = int(round(float(settings.get("take_profit_end", 0.40)) * 100))
    step = int(round(float(settings.get("take_profit_step", 0.02)) * 100))
    thresholds = [value / 100 for value in range(start, end + 1, step)]
    return [
        {
            "entry_rank": int(entry_rank), "exit_rank": int(exit_rank),
            "confirmation_days": int(confirmation), "take_profit": float(threshold),
        }
        for threshold, confirmation, entry_rank, exit_rank in product(
            thresholds,
            settings.get("confirmation_days", [1, 2, 3]),
            settings.get("entry_ranks", [20, 25]),
            settings.get("exit_ranks", [10, 15, 20, 25]),
        )
    ]


def experiment_id(item: dict[str, Any]) -> str:
    return (
        f"entry{item['entry_rank']}_exit{item['exit_rank']}_"
        f"confirm{item['confirmation_days']}_tp{int(round(item['take_profit'] * 100)):02d}"
    )


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_results(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.8f")
    temporary.replace(path)


def run_experiments(
    daily: pd.DataFrame,
    rankings: pd.DataFrame,
    names: dict[str, str],
    start_date: date,
    end_date: date,
    initial_cash: float,
    costs: dict[str, float],
    settings: dict[str, Any],
    output_dir: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Run/resume the strategy-lab grid and persist compact metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    progress_path = output_dir / "progress.json"
    grid = experiment_grid(settings)
    existing = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    existing = existing.drop(columns=["result_rank"], errors="ignore")
    completed_ids = set(existing.get("experiment_id", pd.Series(dtype=str)).astype(str))
    rows = existing.to_dict("records")
    started_at = datetime.now()

    def update(status: str, current: str = "", error: str = "") -> None:
        payload = {
            "status": status, "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
            "total": len(grid), "completed": len(rows), "current_experiment": current,
            "started_at": started_at.isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": (datetime.now() - started_at).total_seconds(), "error": error,
            "results_file": results_path.name,
        }
        _write_progress(progress_path, payload)
        if progress_callback:
            progress_callback(payload)

    update("running")
    try:
        for item in grid:
            item_id = experiment_id(item)
            if item_id in completed_ids:
                continue
            update("running", item_id)
            spec = StrategySpec(
                strategy_id=f"lab_{item_id}",
                name=(
                    f"连续3次Top{item['entry_rank']}｜跌出Top{item['exit_rank']}｜"
                    f"{item['confirmation_days']}次确认｜{item['take_profit']:.0%}止盈"
                ),
                description="当前最佳策略的参数消融实验。",
                entry_rank=item["entry_rank"], exit_rank=item["exit_rank"],
                consecutive_rank=item["entry_rank"], consecutive_days=3,
                weighting="equal", exit_confirmation_days=item["confirmation_days"],
            )
            run_started = time.perf_counter()
            _, trades, _, metrics = run_event_strategy(
                daily, rankings, names, spec, start_date, end_date, initial_cash, costs,
                take_profit=item["take_profit"], record_profit=item["take_profit"],
            )
            rows.append({
                "experiment_id": item_id, "strategy_name": spec.name,
                **item,
                "cumulative_return": metrics["cumulative_return"],
                "annualized_return": metrics["annualized_return"],
                "benchmark_return": metrics["benchmark_return"],
                "excess_return": metrics["excess_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "annualized_volatility": metrics["annualized_volatility"],
                "total_trade_count": metrics["total_trade_count"],
                "profitable_trade_rate": metrics["profitable_trade_rate"],
                "average_holding_days": metrics["average_holding_days"],
                "take_profit_count": metrics["take_profit_count"],
                "open_positions": int(trades["status"].eq("持有中").sum()) if not trades.empty else 0,
                "limit_up_buy_blocked_count": metrics["limit_up_buy_blocked_count"],
                "limit_down_sell_blocked_count": metrics["limit_down_sell_blocked_count"],
                "runtime_seconds": time.perf_counter() - run_started,
            })
            completed_ids.add(item_id)
            result = pd.DataFrame(rows).sort_values("cumulative_return", ascending=False)
            _write_results(results_path, result)
            if len(rows) == 1 or len(rows) % 5 == 0:
                LOGGER.info("策略试验场进度: %d/%d | 当前 %s", len(rows), len(grid), item_id)
        result = pd.DataFrame(rows).sort_values("cumulative_return", ascending=False).reset_index(drop=True)
        result.insert(0, "result_rank", result.index + 1)
        _write_results(results_path, result)
        update("completed")
        return progress_path, result
    except Exception as exc:
        update("failed", error=str(exc))
        raise
