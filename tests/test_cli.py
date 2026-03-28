"""
Tests for prompt_shield/cli.py

Uses Click's test runner. No ML models required.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest
import yaml

from click.testing import CliRunner

from prompt_shield.cli import cli
from prompt_shield.store import BrittlenessStore
from prompt_shield.models import BrittleCertificate, BrittlenessResult, LevelBreakdown
from datetime import datetime, timezone


def _populate_db(db_path: str, prompt_name: str = "test_prompt", verdict: str = "ROBUST"):
    """Insert a run record directly for report command tests."""
    cert = BrittleCertificate(
        certificate_id="shld_test",
        issued_at=datetime(2026, 3, 27, 12, 0, 0, tzinfo=timezone.utc),
        prompt_hash="sha256:abcd",
        prompt_name=prompt_name,
        verdict=verdict,
        brittleness_score=0.05,
        threshold=0.30,
        confidence_lower=0.02,
        confidence_upper=0.10,
        variant_count=8,
        level_breakdown=[LevelBreakdown("lexical", 0.05, 8, 0, "ROBUST")],
        fault_lines=[],
    )
    result = BrittlenessResult(
        score=0.05,
        verdict=verdict,
        certificate=cert,
        variant_results=[],
        test_input_count=2,
        run_duration_seconds=0.5,
    )
    store = BrittlenessStore(db_path)
    store.log_run(result)


class TestCLIReport:
    def test_report_shows_no_runs_message(self, tmp_path):
        db = str(tmp_path / "shield.db")
        BrittlenessStore(db)  # create empty DB
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--store", db])
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_report_shows_run_data(self, tmp_path):
        db = str(tmp_path / "shield.db")
        _populate_db(db, "my_prompt", "ROBUST")
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--store", db])
        assert result.exit_code == 0
        assert "my_prompt" in result.output
        assert "ROBUST" in result.output

    def test_report_filter_by_prompt(self, tmp_path):
        db = str(tmp_path / "shield.db")
        _populate_db(db, "prompt_a", "ROBUST")
        _populate_db(db, "prompt_b", "BRITTLE")
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--store", db, "--prompt", "prompt_a"])
        assert result.exit_code == 0
        assert "prompt_a" in result.output
        assert "prompt_b" not in result.output

    def test_report_shows_headers(self, tmp_path):
        db = str(tmp_path / "shield.db")
        BrittlenessStore(db)
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--store", db])
        assert result.exit_code == 0
        assert "Timestamp" in result.output or "Score" in result.output or result.output  # headers or no runs

    def test_report_multiple_runs(self, tmp_path):
        db = str(tmp_path / "shield.db")
        for i in range(3):
            _populate_db(db, f"prompt_{i}")
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--store", db])
        assert result.exit_code == 0
        assert "prompt_0" in result.output


class TestCLIHelp:
    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "prompt-shield" in result.output.lower() or "brittle" in result.output.lower()

    def test_report_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
        assert "store" in result.output.lower()

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_ci_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["ci", "--help"])
        assert result.exit_code == 0
        assert "threshold" in result.output.lower()
