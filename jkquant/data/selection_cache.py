from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def strategy_key(config: dict[str, Any]) -> str:
    payload = {"schema": 1, "market": config["market"], "strategy": config["strategy"]}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class SelectionCache:
    """SQLite cache for small, indexed Top-K result sets."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS selection_results (
                    strategy_key TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    row_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (strategy_key, trade_date, rank)
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_selection_date ON selection_results(trade_date)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def get(self, key: str, trade_date: date) -> pd.DataFrame | None:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT row_json FROM selection_results
                   WHERE strategy_key = ? AND trade_date = ? ORDER BY rank""",
                (key, trade_date.isoformat()),
            ).fetchall()
        if not rows:
            return None
        return pd.DataFrame([json.loads(row[0]) for row in rows])

    def put(self, key: str, trade_date: date, frame: pd.DataFrame) -> None:
        records = json.loads(frame.to_json(orient="records", date_format="iso"))
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM selection_results WHERE strategy_key = ? AND trade_date = ?",
                (key, trade_date.isoformat()),
            )
            connection.executemany(
                """INSERT INTO selection_results
                   (strategy_key, trade_date, rank, row_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        key, trade_date.isoformat(), int(record["rank"]),
                        json.dumps(record, ensure_ascii=False), created_at,
                    )
                    for record in records
                ],
            )

    def cached_dates(self, key: str, start_date: date, end_date: date) -> set[date]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT DISTINCT trade_date FROM selection_results
                   WHERE strategy_key = ? AND trade_date BETWEEN ? AND ?""",
                (key, start_date.isoformat(), end_date.isoformat()),
            ).fetchall()
        return {date.fromisoformat(str(row[0])) for row in rows}

    def history(self, key: str, start_date: date, end_date: date) -> pd.DataFrame:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT trade_date, rank, row_json FROM selection_results
                   WHERE strategy_key = ? AND trade_date BETWEEN ? AND ?
                   ORDER BY trade_date, rank""",
                (key, start_date.isoformat(), end_date.isoformat()),
            ).fetchall()
        records = []
        for trade_date_value, rank, row_json in rows:
            row = json.loads(row_json)
            records.append({
                "trade_date": date.fromisoformat(str(trade_date_value)),
                "rank": int(rank),
                "ts_code": str(row["ts_code"]),
            })
        return pd.DataFrame(records, columns=["trade_date", "rank", "ts_code"])
