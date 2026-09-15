from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate(config)
    config["_config_dir"] = str(config_path.resolve().parent)
    return config


def _validate(config: dict[str, Any]) -> None:
    required = {"data", "market", "strategy", "report"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"配置缺少节点: {sorted(missing)}")
    weights = config["strategy"]["category_weights"]
    if abs(sum(float(v) for v in weights.values()) - 1.0) > 1e-9:
        raise ValueError("strategy.category_weights 权重之和必须为 1")
    strategy_top_k = int(config["strategy"]["top_k"])
    if not 1 <= strategy_top_k <= 50:
        raise ValueError("strategy.top_k 必须在 1 到 50 之间")
    if "backtest" in config:
        backtest = config["backtest"]
        backtest_top_k = int(backtest["top_k"])
        if not 1 <= backtest_top_k <= 50:
            raise ValueError("backtest.top_k 必须在 1 到 50 之间")
        if int(backtest["rebalance_days"]) <= 0:
            raise ValueError("backtest.rebalance_days 必须大于 0")
        if backtest.get("weighting", "equal") not in {"equal", "rank_linear"}:
            raise ValueError("backtest.weighting 仅支持 equal 或 rank_linear")
        if float(backtest["initial_cash"]) <= 0:
            raise ValueError("backtest.initial_cash 必须大于 0")
        costs = backtest["cost"]
        if any(float(value) < 0 for value in costs.values()):
            raise ValueError("回测交易成本不能为负数")
    for profile_id, profile in config.get("strategy_profiles", {}).items():
        weights = profile.get("category_weights")
        if weights and abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
            raise ValueError(f"策略配置 {profile_id} 的类别权重之和必须为 1")
    if "strategy_suite" in config:
        suite = config["strategy_suite"]
        record_profit = float(suite.get("record_profit", 0.20))
        take_profit = float(suite.get("take_profit", 0.30))
        if not 0 < record_profit <= take_profit:
            raise ValueError("strategy_suite 必须满足 0 < record_profit <= take_profit")
    if "ai" in config:
        ai = config["ai"]
        if ai.get("provider", "deepseek") != "deepseek":
            raise ValueError("ai.provider 当前仅支持 deepseek")
        if ai.get("thinking", "disabled") not in {"enabled", "disabled"}:
            raise ValueError("ai.thinking 仅支持 enabled 或 disabled")
        temperature = float(ai.get("temperature", 0.2))
        if not 0 <= temperature <= 2:
            raise ValueError("ai.temperature 必须在 0 到 2 之间")
        if int(ai.get("max_tokens", 16000)) <= 0:
            raise ValueError("ai.max_tokens 必须大于 0")
        if int(ai.get("timeout_seconds", 180)) <= 0:
            raise ValueError("ai.timeout_seconds 必须大于 0")
        models = [str(value) for value in ai.get("models", [ai.get("model", "deepseek-flash")])]
        if not models or any(not value.strip() for value in models):
            raise ValueError("ai.models 必须是非空模型名列表")
        if str(ai.get("model", "deepseek-flash")) not in models:
            raise ValueError("ai.model 必须包含在 ai.models 中")


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_config_dir"]) / path


def apply_strategy_profile(config: dict[str, Any], profile_id: str) -> dict[str, Any]:
    """Return an isolated config with one named scoring profile applied."""
    profiles = config.get("strategy_profiles", {})
    if profile_id not in profiles:
        raise ValueError(f"未知策略配置：{profile_id}")
    result = deepcopy(config)
    profile = profiles[profile_id]
    if profile_id == "baseline" and not profile.get("category_weights") and not profile.get("factor_weights"):
        return result
    if "category_weights" in profile:
        result["strategy"]["category_weights"] = deepcopy(profile["category_weights"])
    for factor, weight in profile.get("factor_weights", {}).items():
        if factor not in result["strategy"]["factors"]:
            raise ValueError(f"策略配置 {profile_id} 包含未知因子：{factor}")
        result["strategy"]["factors"][factor]["weight"] = float(weight)
    result["strategy"]["profile_id"] = profile_id
    result["strategy"]["profile_name"] = profile.get("name", profile_id)
    return result
