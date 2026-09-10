from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from jkquant.config import load_config
from jkquant.pipeline import run_factor_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="计算量价因子的IC、分层收益和相关性")
    parser.add_argument("--start", type=date.fromisoformat, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, help="结束日期 YYYY-MM-DD")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = run_factor_diagnostics(
        load_config(root / "config.yaml"), start_date=args.start, end_date=args.end,
    )
    summary = pd.read_csv(output / "summary.csv")
    top = summary.sort_values("mean_rank_ic", ascending=False).head(8)
    print(f"因子诊断已写入：{output}")
    print(top[["factor", "horizon", "mean_rank_ic", "positive_rank_ic_rate", "top_bottom_spread"]].to_string(index=False))


if __name__ == "__main__":
    main()
