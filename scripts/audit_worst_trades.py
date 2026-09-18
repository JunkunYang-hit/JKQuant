"""Audit each available strategy's ten worst *closed* transactions offline.

Usage: conda run -n jkquant python scripts/audit_worst_trades.py
The ex-right calculation is a diagnostic proxy, not a broker cash-flow ledger.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def source_files() -> list[Path]:
    root = ROOT / "backtests"
    suite = sorted((root / "strategy_suite").glob("*/suite.json"))
    latest_suite = suite[-1].parent if suite else None
    paths = sorted(latest_suite.glob("*/trades.csv")) if latest_suite else []
    paths += sorted((root / "streak2_leader").glob("*/trades.csv"))[-1:]
    research = sorted((root / "strategy_research").glob("*/research.json"), key=lambda p: p.stat().st_mtime)
    if research:
        paths += sorted(research[-1].parent.glob("*/trades.csv"))
    return paths


def audit_worst(daily: pd.DataFrame, paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily[["ts_code", "trade_date", "close", "pre_close", "open", "high", "low", "pct_chg"]].copy()
    daily.trade_date = pd.to_datetime(daily.trade_date)
    daily = daily.sort_values(["ts_code", "trade_date"])
    daily["previous_raw_close"] = daily.groupby("ts_code").close.shift()
    daily["adjustment_ratio"] = daily.previous_raw_close / daily.pre_close
    price_groups = {code: group.set_index("trade_date") for code, group in daily.groupby("ts_code", sort=False)}
    records = []
    summaries = []
    for path in paths:
        trades = pd.read_csv(path)
        if not {"net_return", "entry_date", "exit_date", "ts_code"}.issubset(trades):
            continue
        trades["net_return"] = pd.to_numeric(trades.net_return, errors="coerce")
        trades = trades[trades.exit_date.notna() & trades.net_return.notna()].copy()
        strategy = path.parent.name
        family = path.parent.parent.name
        source = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        losses = trades[trades.net_return.lt(0)].nsmallest(10, "net_return")
        summaries.append({"策略": strategy, "结果类型": family, "交易总数": len(trades),
                          "亏损交易数": int(trades.net_return.lt(0).sum()),
                          "胜率": float(trades.net_return.gt(0).mean()) if len(trades) else np.nan,
                          "最差单笔净收益": float(trades.net_return.min()) if len(trades) else np.nan,
                          "来源": source})
        for row in losses.itertuples(index=False):
            entry, exit_ = pd.Timestamp(row.entry_date), pd.Timestamp(row.exit_date)
            price = price_groups.get(row.ts_code)
            period = price.loc[(price.index > entry) & (price.index <= exit_)] if price is not None else pd.DataFrame()
            changes = period.adjustment_ratio.dropna() if not period.empty else pd.Series(dtype=float)
            flagged = changes[(changes - 1).abs().gt(.01)]
            entry_price = float(getattr(row, "entry_price_unadjusted", row.entry_price))
            exit_price = float(row.exit_price) if "exit_price" in trades else np.nan
            raw = exit_price / entry_price - 1 if entry_price > 0 else np.nan
            proxy = (1 + raw) * float(changes.prod()) - 1 if pd.notna(raw) else np.nan
            reported_price_change = float(getattr(row, "price_change", np.nan))
            adjusted_in_engine = "entry_price_unadjusted" in trades
            records.append({"策略": strategy, "结果类型": family, "股票代码": row.ts_code,
                            "股票名称": getattr(row, "name", ""), "买入日": str(entry.date()),
                            "卖出日": str(exit_.date()),
                            "持有交易日": getattr(row, "holding_trading_days", np.nan),
                            "买入价": entry_price, "卖出价": exit_price,
                            "原始价差": raw, "回测口径价差": reported_price_change,
                            "回测净收益": float(row.net_return), "引擎已处理除权": adjusted_in_engine,
                            "除权参考价差": proxy, "除权影响百分点": (proxy - raw) * 100,
                            "回测遗漏百分点": (proxy - reported_price_change) * 100,
                            "持有期显著除权日数": len(flagged),
                            "显著除权日期": ",".join(str(d.date()) for d in flagged.index),
                            "退出原因": getattr(row, "exit_reason", ""),
                            "持有期最高浮盈": getattr(row, "max_gain", np.nan),
                            "持有期最大浮亏": getattr(row, "max_drawdown_during_holding", np.nan),
                            "来源": source})
    return pd.DataFrame(records), pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "backtests" / "loss_audit")
    args = parser.parse_args()
    prices = pd.read_parquet(ROOT / "data/cache/tushare/daily.parquet")
    worst, summary = audit_worst(prices, source_files())
    args.output.mkdir(parents=True, exist_ok=True)
    worst.to_csv(args.output / "worst_10_by_strategy.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    material = worst[worst["回测遗漏百分点"].abs().ge(3)]
    lines = ["# 各策略最差十笔已平仓交易诊断", "",
             "说明：按每笔回测净收益排序；原始价差为卖出价/买入价-1。除权参考价差用持有期"
             "逐日‘前实际收盘/当日除权前收’修正，仅是分红再投式诊断，不等于现金分红入账、"
             "送转后的整数股数或真实券商收益；净收益还含交易成本。", "",
             f"策略结果组数：{len(summary)}；列示亏损交易：{len(worst)}；"
             f"回测疑似遗漏除权修正达到3个百分点：{len(material)}。", "",
             "## 值得核查的除权交易", ""]
    if material.empty:
        lines.append("列示交易中没有回测遗漏幅度达到3个百分点的案例。")
    else:
        for record in material.to_dict("records"):
            lines.append(f"- {record['策略']} / {record['股票代码']} / {record['买入日']}→{record['卖出日']}："
                         f"回测净收益 {record['回测净收益']:.2%}，疑似遗漏 {record['回测遗漏百分点']:.2f} 个百分点；"
                         f"显著除权日 {record['显著除权日期']}。")
    lines += ["", "## 文件", "", "逐笔见 `worst_10_by_strategy.csv`；总体胜率与最差单笔见 `strategy_summary.csv`。",
              "旧策略套件未处理除权；新研究引擎使用再投资代理且其 entry_price 已调整，"
              "因此只看原始价差会重复计算除权。新旧引擎口径不同，不可将两者收益简单排序比较。"
              "基线三日调仓、八策略组合和策略试验场参数汇总缺少逐股开平仓账本，因此不在逐笔排名中。"]
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"审计 {len(summary)} 组策略、{len(worst)} 笔最差亏损交易；回测疑似遗漏除权≥3个百分点 {len(material)} 笔。")
    print(args.output)


if __name__ == "__main__":
    main()
