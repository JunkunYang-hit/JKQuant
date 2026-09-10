from datetime import date
import json
from pathlib import Path

import pytest
import pandas as pd
import yaml

from jkquant.config import apply_strategy_profile, load_config
from jkquant.pipeline import (
    best_strategy_recommendations, combined_signal_definitions, run_daily,
    selection_for_date,
)


def test_demo_pipeline_creates_top_k_csv(tmp_path: Path) -> None:
    source = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    source["data"]["provider"] = "demo"
    source["data"]["cache_dir"] = str(tmp_path / "cache")
    source["report"]["output_dir"] = str(tmp_path / "reports")
    source["strategy"]["top_k"] = 10
    path = tmp_path / "test-config.yaml"
    path.write_text(yaml.safe_dump(source, allow_unicode=True), encoding="utf-8")
    report, summary = run_daily(load_config(path))
    assert report.exists()
    assert len(report.read_text(encoding="utf-8-sig").splitlines()) == 11
    assert "amount_100m" in report.read_text(encoding="utf-8-sig").splitlines()[0]
    assert summary["universe_count"] == 100
    source["strategy"]["top_k"] = 30
    path.write_text(yaml.safe_dump(source, allow_unicode=True), encoding="utf-8")
    cached, calculation = selection_for_date(
        load_config(path), pd.Timestamp(summary["trade_date"]).date()
    )
    assert calculation["cached"] is True
    assert len(cached) == 30


@pytest.mark.parametrize("section", ["strategy", "backtest"])
def test_top_k_is_limited_to_fifty(tmp_path: Path, section: str) -> None:
    source = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    source[section]["top_k"] = 51
    path = tmp_path / f"invalid-{section}.yaml"
    path.write_text(yaml.safe_dump(source, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="1 到 50"):
        load_config(path)


def test_best_strategy_recommendations_aggregate_entry_signals(tmp_path: Path, monkeypatch) -> None:
    suite_dir = tmp_path / "backtests" / "strategy_suite" / "2025-09-01_2026-09-09"
    suite_dir.mkdir(parents=True)
    strategy_ids = [
        "s05_top20_streak3_confirm2", "s04_top50_streak2_confirm2",
        "s05_top20_streak3", "s04_top50_streak2", "s02_top5_exit20_confirm2",
    ]
    suite = {
        "strategies": [
            {"strategy_id": strategy_id, "name": strategy_id,
             "metrics": {"cumulative_return": 0.5 - index * 0.1}}
            for index, strategy_id in enumerate(strategy_ids)
        ]
    }
    (suite_dir / "suite.json").write_text(
        __import__("json").dumps(suite, ensure_ascii=False), encoding="utf-8",
    )
    frame = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
        "name": ["甲", "乙", "丙"], "rank": [1, 2, 3],
        "total_score": [0.9, 0.8, 0.7],
    })
    stats = {
        "000001.SZ": {"consecutive_top20": 3, "consecutive_top50": 3},
        "000002.SZ": {"consecutive_top20": 3, "consecutive_top50": 3},
        "000003.SZ": {"consecutive_top20": 2, "consecutive_top50": 2},
    }
    monkeypatch.setattr("jkquant.pipeline.selection_for_date", lambda *_: (frame, {"cached": True}))
    monkeypatch.setattr(
        "jkquant.pipeline.recommendation_history_stats",
        lambda *_: (stats, {"cached_days": 3, "expected_days": 3}),
    )
    config = {
        "_config_dir": str(tmp_path), "strategy_suite": {"output_dir": "backtests/strategy_suite"},
        "strategy": {"top_k": 20},
    }
    result, metadata = best_strategy_recommendations(config, date(2026, 9, 9))
    assert len(metadata["strategies"]) == 5
    assert result.iloc[0]["strategy_support_count"] == 5
    assert result.loc[result["ts_code"].eq("000003.SZ"), "strategy_support_count"].iloc[0] == 3


def test_combined_signal_definitions_use_lab_five_and_suite_three(tmp_path: Path) -> None:
    lab_dir = tmp_path / "lab" / "2025-09-01_2026-09-09"
    suite_dir = tmp_path / "suite" / "2025-09-01_2026-09-09"
    lab_dir.mkdir(parents=True)
    suite_dir.mkdir(parents=True)
    pd.DataFrame([
        {
            "experiment_id": f"experiment_{index}", "strategy_name": f"试验{index}",
            "entry_rank": 20, "exit_rank": 20, "confirmation_days": 2,
            "take_profit": 0.20 + index / 100, "cumulative_return": 1 - index / 10,
        }
        for index in range(6)
    ]).to_csv(lab_dir / "results.csv", index=False)
    suite_ids = [
        "s05_top20_streak3_confirm2", "s04_top50_streak2_confirm2",
        "s05_top20_streak3", "s04_top50_streak2", "s02_top5_exit20_confirm2",
    ]
    (suite_dir / "suite.json").write_text(json.dumps({
        "strategies": [
            {
                "strategy_id": strategy_id, "name": strategy_id,
                "metrics": {"cumulative_return": 0.5 - index / 10, "take_profit_threshold": 0.20},
            }
            for index, strategy_id in enumerate(suite_ids)
        ]
    }, ensure_ascii=False), encoding="utf-8")
    config = {
        "_config_dir": str(tmp_path),
        "strategy_lab": {"output_dir": "lab"},
        "strategy_suite": {"output_dir": "suite"},
    }
    definitions = combined_signal_definitions(config)
    assert len(definitions) == 8
    assert sum(item["source"] == "策略试验场前五" for item in definitions) == 5
    assert sum(item["source"] == "原联合推荐前三" for item in definitions) == 3
    assert all(item["strategy_id"] != "s04_top50_streak2" for item in definitions)


def test_defensive_profile_does_not_mutate_baseline_config() -> None:
    config = load_config("config.yaml")
    baseline = apply_strategy_profile(config, "baseline")
    defensive = apply_strategy_profile(config, "recent_defensive")
    assert "profile_id" not in baseline["strategy"]
    assert defensive["strategy"]["category_weights"]["risk"] == 0.70
    assert defensive["strategy"]["factors"]["volatility_20d"]["weight"] == 0.90
    assert config["strategy"]["category_weights"]["risk"] == 0.20
