from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from jkquant.ai import run_ai_analysis
from jkquant.config import load_config
from jkquant.pipeline import available_selection_dates


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 DeepSeek 分析本地 Top-20 候选")
    parser.add_argument("--date", type=date.fromisoformat, help="分析日期 YYYY-MM-DD，默认本地最新交易日")
    parser.add_argument("--force", action="store_true", help="忽略同数据缓存并重新请求 DeepSeek")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.yaml")
    dates = available_selection_dates(config)
    if not dates:
        raise SystemExit("没有本地交易日数据，请先运行 python scripts/run_today.py")
    requested = args.date or dates[-1]
    eligible = [value for value in dates if value <= requested]
    if not eligible:
        raise SystemExit(f"{requested} 之前没有可用交易日")
    selected_date = eligible[-1]
    try:
        result = run_ai_analysis(config, selected_date, force=args.force)
    except Exception as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps({
        "analysis_date": selected_date.isoformat(), "model": result["model"],
        "cached": result["cached"], "created_at": result["created_at"],
        "overall_risk_level": result["analysis"].get("overall_risk_level"),
        "candidate_count": len(result["analysis"].get("candidates", [])),
        "usage": result.get("usage", {}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
