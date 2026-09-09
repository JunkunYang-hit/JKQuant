from __future__ import annotations

from pathlib import Path
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
    if int(config["strategy"]["top_k"]) <= 0:
        raise ValueError("strategy.top_k 必须大于 0")
    if "backtest" in config:
        backtest = config["backtest"]
        if int(backtest["top_k"]) <= 0 or int(backtest["rebalance_days"]) <= 0:
            raise ValueError("backtest.top_k 和 rebalance_days 必须大于 0")
        if float(backtest["initial_cash"]) <= 0:
            raise ValueError("backtest.initial_cash 必须大于 0")
        costs = backtest["cost"]
        if any(float(value) < 0 for value in costs.values()):
            raise ValueError("回测交易成本不能为负数")


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_config_dir"]) / path
