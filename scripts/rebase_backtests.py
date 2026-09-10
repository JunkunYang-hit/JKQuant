from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jkquant.backtest.benchmark import benchmark_return_map
from jkquant.backtest.metrics import calculate_metrics
from jkquant.config import load_config
from jkquant.pipeline import build_store


def main() -> None:
    config = load_config("config.yaml")
    benchmark = build_store(config).load_market_dataset("benchmark")
    mapping = benchmark_return_map(benchmark)
    updated = 0
    for daily_path in Path("backtests").glob("**/daily.csv"):
        metrics_path = daily_path.parent / "metrics.json"
        if not metrics_path.exists():
            continue
        daily = pd.read_csv(daily_path)
        dates = pd.to_datetime(daily["trade_date"])
        daily["trade_date"] = dates
        daily["benchmark_return"] = dates.map(mapping).fillna(0.0)
        daily["benchmark_equity"] = (1 + daily["benchmark_return"]).cumprod()
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig", float_format="%.8f")
        old_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        old_metrics.update(calculate_metrics(daily))
        old_metrics["benchmark_code"] = "510300.SH"
        old_metrics["benchmark_name"] = "沪深300ETF"
        metrics_path.write_text(json.dumps(old_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        updated += 1

    for suite_path in Path("backtests/strategy_suite").glob("*/suite.json"):
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        for item in suite.get("strategies", []):
            metrics_path = suite_path.parent / item["folder"] / "metrics.json"
            if metrics_path.exists():
                item["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        suite["benchmark_code"] = "510300.SH"
        suite["benchmark_name"] = "沪深300ETF"
        suite_path.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已将 {updated} 份回测逐日结果切换为 510300.SH 基准")


if __name__ == "__main__":
    main()
