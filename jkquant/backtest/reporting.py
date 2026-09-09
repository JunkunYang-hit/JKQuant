from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .engine import BacktestResult


def write_backtest_report(result: BacktestResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "daily.csv"
    trades_path = output_dir / "rebalances.csv"
    metrics_path = output_dir / "metrics.json"
    plot_path = output_dir / "equity_drawdown.png"
    result.daily.to_csv(daily_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    result.trades.to_csv(trades_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    metrics_path.write_text(json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[2, 1])
    axes[0].plot(result.daily["trade_date"], result.daily["equity"], label="Strategy")
    axes[0].plot(result.daily["trade_date"], result.daily["benchmark_equity"], label="Universe EW")
    axes[0].set_ylabel("Net value")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].fill_between(result.daily["trade_date"], result.daily["drawdown"], 0, alpha=0.4)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return {"daily": daily_path, "trades": trades_path, "metrics": metrics_path, "plot": plot_path}
