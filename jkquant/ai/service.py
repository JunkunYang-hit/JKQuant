from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ..config import resolve_path
from ..pipeline import build_store, recommendation_history_stats, selection_for_date
from .cache import AIAnalysisCache
from .deepseek import DeepSeekClient, DeepSeekError
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, user_prompt


def _json_value(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp, date)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    return value


def _latest_fundamental(store, statement: str, code: str, selected_date: date,
                        fields: list[str]) -> dict[str, Any] | None:
    frame = store.load_fundamental(statement, code)
    if frame.empty:
        return None
    target = pd.Timestamp(selected_date)
    if "ann_date" in frame:
        frame = frame[frame["ann_date"].notna() & frame["ann_date"].le(target)]
    if frame.empty:
        return None
    order = [column for column in ("end_date", "ann_date") if column in frame]
    row = frame.sort_values(order, ascending=False).iloc[0] if order else frame.iloc[-1]
    result: dict[str, Any] = {}
    for field in fields:
        if field not in row.index:
            continue
        value = _json_value(row[field])
        if value is not None and field not in {"ann_date", "end_date"}:
            value = round(float(value) / 100_000_000, 4)
        result[field] = value
    return result or None


def _macro_snapshot(store, selected_date: date) -> list[dict[str, Any]]:
    result = []
    for name in ("cn_gdp", "cn_cpi", "cn_ppi", "cn_m", "cn_pmi", "sf_month"):
        frame = store.load_macro(name)
        if frame.empty:
            continue
        time_column = next((c for c in ("month", "quarter", "ann_date", "trade_date") if c in frame), None)
        eligible = frame
        if time_column:
            raw = frame[time_column].astype(str).str.replace(r"\D", "", regex=True)
            cutoff = selected_date.strftime("%Y%m%d")
            if raw.str.len().max() == 6:
                cutoff = selected_date.strftime("%Y%m")
            eligible = frame.loc[raw.le(cutoff)].copy()
            if not eligible.empty:
                eligible["__ai_time_key"] = raw.loc[eligible.index]
                eligible = eligible.sort_values("__ai_time_key").drop(columns="__ai_time_key")
        if eligible.empty:
            continue
        row = eligible.iloc[-1]
        values = {str(k): _json_value(v) for k, v in row.items()}
        values = {k: v for k, v in values.items() if v is not None}
        result.append({"dataset": name, "latest_available": dict(list(values.items())[:12])})
    return result


def build_analysis_context(config: dict[str, Any], selected_date: date) -> dict[str, Any]:
    ai_config = {**config, "strategy": {**config["strategy"], "top_k": 20}}
    candidates, _ = selection_for_date(ai_config, selected_date)
    candidates = candidates.head(20).copy()
    store = build_store(config)
    basic = store.load_basic().drop_duplicates("ts_code")
    metadata_columns = [c for c in ("ts_code", "name", "industry", "area", "market") if c in basic]
    candidates = candidates.drop(columns=[c for c in ("name", "industry", "area", "market") if c in candidates], errors="ignore")
    candidates = candidates.merge(basic[metadata_columns], on="ts_code", how="left")

    daily_basic = store.load_market_dataset("daily_basic")
    if not daily_basic.empty:
        daily_basic = daily_basic[daily_basic["trade_date"].le(pd.Timestamp(selected_date))]
        daily_basic = daily_basic.sort_values("trade_date").groupby("ts_code", as_index=False).tail(1)
        valuation_columns = [c for c in (
            "ts_code", "trade_date", "turnover_rate", "volume_ratio", "pe_ttm", "pb",
            "ps_ttm", "dv_ttm", "total_mv", "circ_mv",
        ) if c in daily_basic]
        candidates = candidates.merge(daily_basic[valuation_columns], on="ts_code", how="left")

    streaks, coverage = recommendation_history_stats(config, selected_date)
    benchmark = store.load_market_dataset("benchmark")
    benchmark_return = None
    if not benchmark.empty:
        bounded = benchmark[benchmark["trade_date"].le(pd.Timestamp(selected_date))].sort_values("trade_date").tail(21)
        if len(bounded) > 1:
            benchmark_return = float(bounded["close"].iloc[-1] / bounded["close"].iloc[0] - 1)

    rows = []
    factor_fields = (
        "rank", "ts_code", "name", "industry", "area", "market", "close", "total_score",
        "momentum_score", "trend_score", "risk_score", "liquidity_score", "return_5d",
        "return_20d", "close_ma20", "ma5_ma20", "volatility_20d", "max_drawdown_20d",
        "amount_mean_20d", "amount_ratio_5_20", "turnover_rate", "volume_ratio", "pe_ttm",
        "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv",
    )
    for _, row in candidates.iterrows():
        code = str(row["ts_code"])
        item = {field: _json_value(row.get(field)) for field in factor_fields if field in row.index}
        item["rank"] = int(item["rank"])
        item["history"] = streaks.get(code, {})
        item["fundamentals"] = {
            "income": _latest_fundamental(
                store, "income", code, selected_date,
                ["ann_date", "end_date", "revenue", "total_revenue", "n_income_attr_p"],
            ),
            "balance_sheet": _latest_fundamental(
                store, "balancesheet", code, selected_date,
                ["ann_date", "end_date", "total_assets", "total_liab", "total_hldr_eqy_exc_min_int"],
            ),
            "cash_flow": _latest_fundamental(
                store, "cashflow", code, selected_date,
                ["ann_date", "end_date", "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act"],
            ),
        }
        rows.append(item)

    return {
        "analysis_date": selected_date.isoformat(),
        "candidate_count": len(rows),
        "selection_source": "固定量价多因子模型Top-20",
        "history_coverage": coverage,
        "benchmark": {"code": "510300.SH", "recent_20_trading_day_return": benchmark_return},
        "units": {
            "return/volatility/drawdown/ma_distance": "小数比例，0.1代表10%",
            "amount/amount_mean_20d": "千元", "total_mv/circ_mv": "万元",
            "turnover_rate/dv_ttm": "百分数，1代表1%", "financial_currency": "亿元",
        },
        "macro_note": "低频宏观背景，仅使用所选日期前的本地最新值，不是当日交易信号",
        "macro_snapshot": _macro_snapshot(store, selected_date),
        "candidates": rows,
        "known_limitations": [
            "未接入新闻、公告正文、研报、舆情和盘中数据",
            "历史ST状态仍不完整，股票简称和行业主要来自当前基础信息",
            "财务数据按公告日截断；未完成补库的报表显示为空",
        ],
    }


def _cache(config: dict[str, Any]) -> AIAnalysisCache:
    store = build_store(config)
    filename = config.get("ai", {}).get("cache_file", "ai_analysis.sqlite3")
    return AIAnalysisCache(store.root / filename)


def get_cached_analysis(config: dict[str, Any], selected_date: date) -> dict[str, Any] | None:
    return _cache(config).latest(selected_date)


def run_ai_analysis(config: dict[str, Any], selected_date: date, force: bool = False) -> dict[str, Any]:
    load_dotenv(resolve_path(config, ".env"))
    settings = config.get("ai", {})
    provider = str(settings.get("provider", "deepseek"))
    if provider != "deepseek":
        raise ValueError(f"暂不支持AI供应商：{provider}")
    context = build_analysis_context(config, selected_date)
    encoded = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    input_hash = hashlib.sha256(encoded).hexdigest()
    model = str(settings.get("model", "deepseek-v4-flash"))
    cache = _cache(config)
    if not force:
        cached = cache.get(selected_date, provider, model, PROMPT_VERSION, input_hash)
        if cached:
            cached.update({"provider": provider, "model": model, "prompt_version": PROMPT_VERSION,
                           "input_hash": input_hash})
            return cached
    client = DeepSeekClient(settings)
    analysis, usage = client.complete(
        SYSTEM_PROMPT, user_prompt(context), float(settings.get("temperature", 0.2)),
        int(settings.get("max_tokens", 6000)),
    )
    expected = [(int(item["rank"]), str(item["ts_code"]), str(item.get("name", "")))
                for item in context["candidates"]]
    received = [(int(item.get("rank", -1)), str(item.get("ts_code", "")), str(item.get("name", "")))
                for item in analysis.get("candidates", [])]
    if sorted(expected) != sorted(received):
        raise DeepSeekError("DeepSeek返回的候选数量或股票身份与输入Top-20不一致，结果未写入缓存")
    return cache.put(selected_date, provider, model, PROMPT_VERSION, input_hash, analysis, context, usage)
