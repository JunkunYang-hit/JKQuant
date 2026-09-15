"""Read-only, offline audit of cached prices and historical strategy results.

Usage: conda run -n jkquant python scripts/audit_strategy_evidence.py
Only known market datasets and result files are read; account/API secrets are not.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def describe(frame: pd.DataFrame) -> dict:
    result = {"rows": len(frame), "columns": list(frame.columns)}
    if "ts_code" in frame:
        result["stock_count"] = int(frame["ts_code"].nunique())
    if "trade_date" in frame:
        dates = pd.to_datetime(frame["trade_date"])
        result.update(start=str(dates.min().date()), end=str(dates.max().date()),
                      trading_days=int(dates.nunique()),
                      future_rows=int((dates > pd.Timestamp.today().normalize()).sum()))
    return result


def audit() -> dict:
    cache = ROOT / "data/cache/tushare"
    output: dict = {"datasets": {}, "strategy_results": {}}
    daily = pd.read_parquet(cache / "daily.parquet")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    price = daily.sort_values(["ts_code", "trade_date"])
    summary = describe(daily)
    summary["duplicate_keys"] = int(daily.duplicated(["trade_date", "ts_code"]).sum())
    summary["null_prices"] = daily[["open", "high", "low", "close", "pre_close"]].isna().sum().to_dict()
    summary["nonpositive_prices"] = int((daily[["open", "high", "low", "close", "pre_close"]] <= 0).any(axis=1).sum())
    bad_ohlc = (daily["high"] < daily[["open", "close", "low"]].max(axis=1) - 1e-6) | (daily["low"] > daily[["open", "close", "high"]].min(axis=1) + 1e-6)
    summary["invalid_ohlc_rows"] = int(bad_ohlc.sum())
    calculated_pct = (daily["close"] / daily["pre_close"] - 1) * 100
    if "pct_chg" in daily:
        summary["pct_chg_mismatch_gt_0_02_percentage_point"] = int((calculated_pct - daily["pct_chg"]).abs().gt(0.02).sum())
        summary["pct_chg_null_rows"] = int(daily["pct_chg"].isna().sum())
    summary["absolute_daily_return_gt_40pct"] = int(calculated_pct.abs().gt(40).sum())
    prev = price.groupby("ts_code")["close"].shift()
    gap = (price["pre_close"] / prev - 1).abs()
    summary["pre_close_previous_close_gap_gt_1pct"] = int(gap.gt(0.01).sum())
    summary["pre_close_previous_close_gap_gt_10pct"] = int(gap.gt(0.10).sum())
    summary["largest_preclose_gaps"] = price.assign(gap=gap).nlargest(8, "gap")[["trade_date", "ts_code", "pre_close", "close", "gap"]].to_dict("records")
    summary["largest_intraday_ranges"] = daily.assign(intraday_range=daily["high"] / daily["low"] - 1).nlargest(8, "intraday_range")[["trade_date", "ts_code", "open", "high", "low", "close", "intraday_range"]].to_dict("records")
    count = daily.groupby("trade_date").size()
    summary["daily_stock_count_quantiles"] = count.quantile([0, 0.1, 0.5, 0.9, 1]).to_dict()
    summary["last_dates_stock_count"] = {str(d.date()): int(n) for d, n in count.tail(8).items()}
    # A demo stream has high-precision synthetic OHLC and no pct_chg column.
    summary["close_more_than_2dp_rows"] = int((daily["close"] * 100 - (daily["close"] * 100).round()).abs().gt(1e-6).sum())
    summary["positive_amount_fraction"] = float(daily["amount"].gt(0).mean())
    output["datasets"]["daily"] = summary
    for name in ["stock_basic", "daily_basic", "weekly", "monthly", "stk_limit", "benchmark_510300"]:
        path = cache / f"{name}.parquet"
        if not path.exists():
            output["datasets"][name] = {"missing": True}
            continue
        frame = pd.read_parquet(path)
        detail = describe(frame)
        if name == "stock_basic":
            detail["list_status_counts"] = frame.get("list_status", pd.Series(dtype=str)).value_counts().to_dict()
            detail["daily_codes_missing_in_basic"] = sorted(set(daily.ts_code) - set(frame.ts_code))[:20]
        if name == "daily_basic":
            detail["valid_fraction"] = {
                column: {"nonnull": float(frame[column].notna().mean()), "positive": float(frame[column].gt(0).mean())}
                for column in ["pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "turnover_rate", "volume_ratio"]
            }
        if name == "benchmark_510300":
            frame = frame.sort_values("trade_date")
            detail["first_rows"] = frame.head(2).to_dict("records")
            detail["last_rows"] = frame.tail(2).to_dict("records")
            detail["market_dates_without_benchmark"] = len(set(daily.trade_date) - set(pd.to_datetime(frame.trade_date)))
        output["datasets"][name] = detail
    output["datasets"]["fundamentals_files"] = {name: len(list((cache / "fundamentals" / name).glob("*.parquet"))) for name in ["income", "balancesheet", "cashflow"]}
    output["datasets"]["adjustment_factor_files"] = [str(path.relative_to(ROOT)) for path in cache.rglob("*adj*")]
    output["datasets"]["macro"] = {path.stem: describe(pd.read_parquet(path)) for path in (cache / "macro").glob("*.parquet")}
    streak_path = ROOT / "backtests/streak2_leader/2025-09-01_2026-09-11"
    if streak_path.exists():
        trades = pd.read_csv(streak_path / "trades.csv")
        ledger = pd.read_csv(streak_path / "daily.csv")
        holding_parts = []
        for row in trades.itertuples():
            exit_date = row.exit_date if isinstance(row.exit_date, str) else str(daily.trade_date.max().date())
            holding_parts.append(price.loc[
                price.ts_code.eq(row.ts_code) & price.trade_date.gt(pd.Timestamp(row.entry_date))
                & price.trade_date.le(pd.Timestamp(exit_date))
            ].assign(adjustment_gap=gap))
        holding_days = pd.concat(holding_parts)
        significant_gaps = holding_days[holding_days.adjustment_gap.gt(.001)]
        output["streak2_loss_audit"] = {
            "worst_trades": trades.nsmallest(5, "net_return")[["ts_code", "entry_date", "exit_date", "net_return", "holding_trading_days"]].to_dict("records"),
            "worst_days": ledger.nsmallest(5, "net_return")[["trade_date", "net_return"]].to_dict("records"),
            "holding_gap_rows": significant_gaps[["trade_date", "ts_code", "pre_close", "adjustment_gap"]].to_dict("records"),
            "transaction_cost_note": "total_transaction_cost sums daily cost/previous_equity ratios; it is not cost/initial_cash.",
        }
    metrics_fields = ["cumulative_return", "benchmark_return", "max_drawdown", "average_turnover", "total_transaction_cost", "total_trade_count", "profitable_trade_rate", "average_holding_days", "cost_model"]
    for suite_path in sorted((ROOT / "backtests/strategy_suite").glob("*/suite.json")):
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        entries = []
        for item in suite.get("strategies", []):
            values = item.get("metrics", {})
            entries.append({"id": item["strategy_id"], **{key: values.get(key) for key in metrics_fields}})
        entries.sort(key=lambda x: x.get("cumulative_return") or -1, reverse=True)
        output["strategy_results"][str(suite_path.parent.relative_to(ROOT))] = {"strategy_count": len(entries), "best5": entries[:5], "worst3": entries[-3:]}
    for metric_path in sorted((ROOT / "backtests").glob("**/metrics.json")):
        if "strategy_suite" in metric_path.parts or "strategy_lab" in metric_path.parts:
            continue
        values = json.loads(metric_path.read_text(encoding="utf-8"))
        output["strategy_results"][str(metric_path.parent.relative_to(ROOT))] = {key: values.get(key) for key in metrics_fields}
    for path in sorted((ROOT / "backtests/strategy_lab").glob("*/results.csv")):
        frame = pd.read_csv(path)
        fields = ["experiment_id", "entry_rank", "exit_rank", "confirmation_days", "take_profit", "cumulative_return", "max_drawdown", "total_trade_count", "profitable_trade_rate", "average_holding_days"]
        fields = [column for column in fields if column in frame]
        output["strategy_results"][str(path.relative_to(ROOT))] = {
            "experiments": len(frame),
            "positive_fraction": float((frame.cumulative_return > 0).mean()),
            "return_quantiles": frame.cumulative_return.quantile([0, .25, .5, .75, 1]).to_dict(),
            "best5": frame.nlargest(5, "cumulative_return")[fields].to_dict("records"),
            "grouped_mean_returns": {column: frame.groupby(column).cumulative_return.mean().to_dict() for column in ["entry_rank", "exit_rank", "confirmation_days", "take_profit"]},
        }
    return output


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=True, indent=2, default=str, allow_nan=False))
