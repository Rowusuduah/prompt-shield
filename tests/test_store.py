"""
Tests for prompt_shield/store.py

Pure SQLite — no ML dependencies.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from prompt_shield.models import (
    BrittleCertificate,
    BrittlenessResult,
    FaultLine,
    LevelBreakdown,
)
from prompt_shield.store import BrittlenessStore


def _make_result(
    prompt_name: str = "my_prompt",
    score: float = 0.05,
    verdict: str = "ROBUST",
    variant_count: int = 8,
    threshold: float = 0.30,
) -> BrittlenessResult:
    cert = BrittleCertificate(
        certificate_id=f"shld_test_{prompt_name[:4]}",
        issued_at=datetime(2026, 3, 27, 12, 0, 0, tzinfo=timezone.utc),
        prompt_hash="sha256:abcd1234",
        prompt_name=prompt_name,
        verdict=verdict,
        brittleness_score=score,
        threshold=threshold,
        confidence_lower=0.02,
        confidence_upper=0.10,
        variant_count=variant_count,
        level_breakdown=[
            LevelBreakdown("lexical", score, variant_count // 2, 0, verdict),
        ],
        fault_lines=[],
    )
    return BrittlenessResult(
        score=score,
        verdict=verdict,
        certificate=cert,
        variant_results=[],
        test_input_count=2,
        run_duration_seconds=0.5,
    )


class TestBrittlenessStoreInit:
    def test_creates_database_file(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        assert (tmp_path / "shield.db").exists()

    def test_creates_runs_table(self, tmp_path):
        import sqlite3
        db = str(tmp_path / "shield.db")
        BrittlenessStore(db)
        with sqlite3.connect(db) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "runs" in tables

    def test_creates_baselines_table(self, tmp_path):
        import sqlite3
        db = str(tmp_path / "shield.db")
        BrittlenessStore(db)
        with sqlite3.connect(db) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "baselines" in tables

    def test_idempotent_init(self, tmp_path):
        db = str(tmp_path / "shield.db")
        BrittlenessStore(db)
        BrittlenessStore(db)  # second init should not raise


class TestLogRun:
    def test_log_run_saves_to_db(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        result = _make_result("test_prompt")
        store.log_run(result)
        runs = store.get_runs()
        assert len(runs) == 1

    def test_log_run_preserves_prompt_name(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        store.log_run(_make_result("customer_support"))
        runs = store.get_runs()
        assert runs[0]["prompt_name"] == "customer_support"

    def test_log_run_preserves_verdict(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        store.log_run(_make_result(verdict="BRITTLE", score=0.60))
        runs = store.get_runs()
        assert runs[0]["verdict"] == "BRITTLE"

    def test_log_run_preserves_score(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        store.log_run(_make_result(score=0.42))
        runs = store.get_runs()
        assert abs(runs[0]["brittleness_score"] - 0.42) < 0.001

    def test_multiple_runs_logged(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        store.log_run(_make_result("p1"))
        store.log_run(_make_result("p2"))
        store.log_run(_make_result("p3"))
        runs = store.get_runs(limit=10)
        assert len(runs) == 3

    def test_log_run_certificate_json_is_valid_json(self, tmp_path):
        import sqlite3
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        store.log_run(_make_result())
        with sqlite3.connect(db) as conn:
            row = conn.execute("SELECT certificate_json FROM runs LIMIT 1").fetchone()
        assert row is not None
        parsed = json.loads(row[0])
        assert "verdict" in parsed


class TestGetRuns:
    def test_get_runs_returns_list(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        runs = store.get_runs()
        assert isinstance(runs, list)

    def test_get_runs_empty_when_no_data(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        assert store.get_runs() == []

    def test_get_runs_filter_by_prompt_name(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        store.log_run(_make_result("support"))
        store.log_run(_make_result("search"))
        runs = store.get_runs(prompt_name="support")
        assert len(runs) == 1
        assert runs[0]["prompt_name"] == "support"

    def test_get_runs_limit_respected(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        for i in range(5):
            store.log_run(_make_result(f"p{i}"))
        runs = store.get_runs(limit=3)
        assert len(runs) == 3


class TestBaseline:
    def test_register_baseline(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        store.register_baseline("my_prompt", 0.10, "ROBUST", "shld_abc123")
        baseline = store.get_baseline("my_prompt")
        assert baseline is not None
        assert baseline["prompt_name"] == "my_prompt"
        assert baseline["approved_verdict"] == "ROBUST"
        assert abs(baseline["approved_score"] - 0.10) < 0.001
        assert baseline["certificate_id"] == "shld_abc123"

    def test_get_baseline_returns_none_when_missing(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        assert store.get_baseline("nonexistent") is None

    def test_register_baseline_upsert(self, tmp_path):
        db = str(tmp_path / "shield.db")
        store = BrittlenessStore(db)
        store.register_baseline("p1", 0.10, "ROBUST", "shld_1")
        store.register_baseline("p1", 0.20, "CONDITIONAL", "shld_2")
        baseline = store.get_baseline("p1")
        assert baseline["approved_score"] == 0.20
        assert baseline["approved_verdict"] == "CONDITIONAL"
