"""
prompt_shield/store.py — SQLite trace log for all brittleness check runs.

PAT-048 (Daniel 5): The writing on the wall is permanent.
All verdicts are inscribed — not ephemeral.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from .models import BrittlenessResult


class BrittlenessStore:
    """Persists all brittleness run results to SQLite."""

    CREATE_RUNS_TABLE = """
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at TEXT NOT NULL,
        prompt_name TEXT NOT NULL,
        prompt_hash TEXT NOT NULL,
        test_input_count INTEGER NOT NULL,
        variant_count INTEGER NOT NULL,
        brittleness_score REAL NOT NULL,
        threshold REAL NOT NULL,
        verdict TEXT NOT NULL,
        confidence_lower REAL,
        confidence_upper REAL,
        certificate_json TEXT NOT NULL,
        run_duration_seconds REAL
    )
    """

    CREATE_BASELINES_TABLE = """
    CREATE TABLE IF NOT EXISTS baselines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt_name TEXT NOT NULL UNIQUE,
        registered_at TEXT NOT NULL,
        approved_score REAL NOT NULL,
        approved_verdict TEXT NOT NULL,
        certificate_id TEXT NOT NULL
    )
    """

    def __init__(self, store_path: str = "./shield.db"):
        self.path = Path(store_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute(self.CREATE_RUNS_TABLE)
            conn.execute(self.CREATE_BASELINES_TABLE)
            conn.commit()

    def log_run(self, result: BrittlenessResult):
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                INSERT INTO runs (
                    run_at, prompt_name, prompt_hash, test_input_count,
                    variant_count, brittleness_score, threshold, verdict,
                    confidence_lower, confidence_upper, certificate_json,
                    run_duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.certificate.issued_at.isoformat(),
                result.certificate.prompt_name,
                result.certificate.prompt_hash,
                result.test_input_count,
                result.certificate.variant_count,
                result.score,
                result.certificate.threshold,
                result.verdict,
                result.certificate.confidence_lower,
                result.certificate.confidence_upper,
                result.certificate.to_json(),
                result.run_duration_seconds
            ))
            conn.commit()

    def register_baseline(
        self,
        prompt_name: str,
        score: float,
        verdict: str,
        certificate_id: str,
    ):
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO baselines
                (prompt_name, registered_at, approved_score, approved_verdict, certificate_id)
                VALUES (?, datetime('now'), ?, ?, ?)
            """, (prompt_name, score, verdict, certificate_id))
            conn.commit()

    def get_baseline(self, prompt_name: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT * FROM baselines WHERE prompt_name = ?", (prompt_name,)
            ).fetchone()
        if row is None:
            return None
        cols = ["id", "prompt_name", "registered_at", "approved_score",
                "approved_verdict", "certificate_id"]
        return dict(zip(cols, row))

    def get_runs(self, prompt_name: str = None, limit: int = 20) -> list[dict]:
        with sqlite3.connect(self.path) as conn:
            if prompt_name:
                rows = conn.execute(
                    "SELECT run_at, prompt_name, brittleness_score, verdict, variant_count "
                    "FROM runs WHERE prompt_name = ? ORDER BY run_at DESC LIMIT ?",
                    (prompt_name, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT run_at, prompt_name, brittleness_score, verdict, variant_count "
                    "FROM runs ORDER BY run_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
        cols = ["run_at", "prompt_name", "brittleness_score", "verdict", "variant_count"]
        return [dict(zip(cols, row)) for row in rows]
