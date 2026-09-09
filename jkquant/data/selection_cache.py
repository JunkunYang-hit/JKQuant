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
            connection.execute(
                """CREATE TABLE IF NOT EXISTS top20_streaks (
                    strategy_key TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    ts_code TEXT NOT NULL,
                    streak_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (strategy_key, trade_date, ts_code)
                )"""
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

    def get_streaks(self, key: str, trade_date: date) -> dict[str, int] | None:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT ts_code, streak_count FROM top20_streaks
                   WHERE strategy_key = ? AND trade_date = ?""",
                (key, trade_date.isoformat()),
            ).fetchall()
        if not rows:
            return None
        return {str(code): int(count) for code, count in rows}

    def put_streaks(self, key: str, trade_date: date, streaks: dict[str, int]) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM top20_streaks WHERE strategy_key = ? AND trade_date = ?",
                (key, trade_date.isoformat()),
            )
            connection.executemany(
                """INSERT INTO top20_streaks
                   (strategy_key, trade_date, ts_code, streak_count, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (key, trade_date.isoformat(), code, int(count), created_at)
                    for code, count in streaks.items()
                ],
            )
