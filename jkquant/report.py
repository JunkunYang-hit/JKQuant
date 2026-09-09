from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_csv(selection: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trade_date = pd.Timestamp(selection["trade_date"].iloc[0]).strftime("%Y-%m-%d")
    path = output_dir / f"{trade_date}.csv"
    selection.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
    return path

