from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd


class AccountStore:
    """Small persistent account ledger used by the local research UI."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS holdings (
                    ts_code TEXT PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT '',
                    quantity REAL NOT NULL CHECK (quantity > 0),
                    cost_price REAL NOT NULL CHECK (cost_price > 0),
                    entry_date TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def upsert(
        self, ts_code: str, name: str, quantity: float, cost_price: float,
        entry_date: date, notes: str = "",
    ) -> None:
        if quantity <= 0 or cost_price <= 0:
            raise ValueError("持仓数量和成本价必须大于 0")
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO holdings(ts_code, name, quantity, cost_price, entry_date, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ts_code) DO UPDATE SET
                    name=excluded.name, quantity=excluded.quantity,
                    cost_price=excluded.cost_price, entry_date=excluded.entry_date,
                    notes=excluded.notes, updated_at=excluded.updated_at
            """, (
                ts_code, name, float(quantity), float(cost_price), entry_date.isoformat(), notes,
                datetime.now().isoformat(timespec="seconds"),
            ))

    def delete(self, ts_code: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM holdings WHERE ts_code = ?", (ts_code,))

    def get(self, ts_code: str) -> dict | None:
        frame = self.list_holdings()
        rows = frame[frame["ts_code"].eq(ts_code)]
        return None if rows.empty else rows.iloc[0].to_dict()

    def list_holdings(self) -> pd.DataFrame:
        with self._connect() as connection:
            frame = pd.read_sql_query(
                "SELECT ts_code, name, quantity, cost_price, entry_date, notes, updated_at "
                "FROM holdings ORDER BY entry_date, ts_code", connection,
            )
        if not frame.empty:
            frame["entry_date"] = pd.to_datetime(frame["entry_date"]).dt.date
        return frame
