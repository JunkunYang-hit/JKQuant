from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


class AIAnalysisCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS ai_analyses (
                    analysis_date TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    usage_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (analysis_date, provider, model, prompt_version, input_hash)
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def get(self, analysis_date: date, provider: str, model: str,
            prompt_version: str, input_hash: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT response_json, input_json, usage_json, created_at
                   FROM ai_analyses WHERE analysis_date=? AND provider=? AND model=?
                   AND prompt_version=? AND input_hash=?""",
                (analysis_date.isoformat(), provider, model, prompt_version, input_hash),
            ).fetchone()
        if row is None:
            return None
        return {
            "analysis": json.loads(row[0]), "input": json.loads(row[1]),
            "usage": json.loads(row[2]), "created_at": row[3], "cached": True,
        }

    def latest(self, analysis_date: date) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT response_json, input_json, usage_json, created_at, provider, model,
                          prompt_version, input_hash
                   FROM ai_analyses WHERE analysis_date=? ORDER BY created_at DESC LIMIT 1""",
                (analysis_date.isoformat(),),
            ).fetchone()
        if row is None:
            return None
        return {
            "analysis": json.loads(row[0]), "input": json.loads(row[1]),
            "usage": json.loads(row[2]), "created_at": row[3], "provider": row[4],
            "model": row[5], "prompt_version": row[6], "input_hash": row[7],
            "cached": True,
        }

    def put(self, analysis_date: date, provider: str, model: str, prompt_version: str,
            input_hash: str, analysis: dict[str, Any], context: dict[str, Any],
            usage: dict[str, Any]) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat()
        values = (
            analysis_date.isoformat(), provider, model, prompt_version, input_hash,
            json.dumps(analysis, ensure_ascii=False),
            json.dumps(context, ensure_ascii=False), json.dumps(usage, ensure_ascii=False),
            created_at,
        )
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO ai_analyses
                   (analysis_date, provider, model, prompt_version, input_hash,
                    response_json, input_json, usage_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", values,
            )
        return {
            "analysis": analysis, "input": context, "usage": usage,
            "created_at": created_at, "provider": provider, "model": model,
            "prompt_version": prompt_version, "input_hash": input_hash, "cached": False,
        }
