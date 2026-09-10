from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from .storage import ParquetStore
from .tushare_provider import TushareProvider

LOGGER = logging.getLogger(__name__)
STATEMENTS = ("income", "balancesheet", "cashflow")


def update_fundamentals(
    provider: TushareProvider, store: ParquetStore, start_date: date,
    end_date: date, max_stocks: int | None = None,
) -> dict:
    """Download three statements stock-by-stock; existing files are resumable."""
    basic = store.load_basic()
    listed = basic[basic["list_status"].eq("L")]["ts_code"].astype(str).tolist()
    if max_stocks is not None:
        listed = listed[:max_stocks]
    progress_path = store.fundamental_dir / "progress.json"
    completed = skipped = failed = 0
    failures: list[dict[str, str]] = []
    total = len(listed) * len(STATEMENTS)
    status = {
        "status": "running", "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(), "total_tasks": total,
        "completed": 0, "skipped": 0, "failed": 0, "current_stock": "",
        "updated_at": datetime.now().isoformat(timespec="seconds"), "recent_failures": [],
    }
    for ts_code in listed:
        for statement in STATEMENTS:
            target = store.fundamental_dir / statement / f"{ts_code.replace('.', '_')}.parquet"
            if target.exists():
                skipped += 1
                continue
            try:
                frame = provider.financial_statement(statement, ts_code, start_date, end_date)
                if not frame.empty:
                    store.save_fundamental(statement, ts_code, frame)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(target, index=False)
                completed += 1
            except Exception as exc:  # continue the multi-hour batch and retain failures
                failed += 1
                failures.append({"ts_code": ts_code, "statement": statement, "error": str(exc)})
                LOGGER.warning("财务数据失败 %s %s: %s", ts_code, statement, exc)
            status = {
                "status": "running", "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(), "total_tasks": total,
                "completed": completed, "skipped": skipped, "failed": failed,
                "current_stock": ts_code, "updated_at": datetime.now().isoformat(timespec="seconds"),
                "recent_failures": failures[-20:],
            }
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    status["status"] = "completed"
    status["updated_at"] = datetime.now().isoformat(timespec="seconds")
    progress_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status
