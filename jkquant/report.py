from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_csv(selection: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trade_date = pd.Timestamp(selection["trade_date"].iloc[0]).strftime("%Y-%m-%d")
    path = output_dir / f"{trade_date}.csv"
    export = selection.copy()
    export["amount_100m"] = export.pop("amount") / 100_000
    if "amount_mean_20d" in export:
        export["amount_mean_20d_100m"] = export.pop("amount_mean_20d") / 100_000
    export.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
    return path
