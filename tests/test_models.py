"""
Tests for prompt_shield/models.py

All model tests are pure Python — no ML dependencies required.
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
    ParaphraseVariant,
    VariantResult,
)


# ── ParaphraseVariant ─────────────────────────────────────────────────────────

class TestParaphraseVariant:
    def test_creation_defaults(self):
        pv = ParaphraseVariant(
            original="What is Python?",
            variant="Could you tell me about Python?",
            level="semantic",
            similarity_score=0.87,
        )
        assert pv.original == "What is Python?"
        assert pv.variant == "Could you tell me about Python?"
        assert pv.level == "semantic"
        assert pv.similarity_score == 0.87
        assert pv.validated is False

    def test_creation_validated(self):
        pv = ParaphraseVariant(
            original="x", variant="y", level="lexical",
            similarity_score=0.80, validated=True
        )
        assert pv.validated is True

    def test_all_three_levels(self):
        for level in ("lexical", "syntactic", "semantic"):
            pv = ParaphraseVariant("a", "b", level, 0.85, True)
            assert pv.level == level


# ── VariantResult ─────────────────────────────────────────────────────────────

class TestVariantResult:
    def _make_variant(self, level="lexical") -> ParaphraseVariant:
        return ParaphraseVariant("input", "variant", level, 0.85, True)

    def test_creation_not_deviant(self):
        vr = VariantResult(
            variant=self._make_variant(),
            output="same output",
            deviation_score=0.05,
            is_deviant=False,
        )
        assert vr.is_deviant is False
        assert vr.deviation_score == 0.05

    def test_creation_deviant(self):
        vr = VariantResult(
            variant=self._make_variant(),
            output="very different output",
            deviation_score=0.60,
            is_deviant=True,
        )
        assert vr.is_deviant is True


# ── FaultLine ─────────────────────────────────────────────────────────────────

class TestFaultLine:
    def test_creation(self):
        fl = FaultLine(
            level="semantic",
            variant="A reworded variant",
            deviation_score=0.55,
            canonical_fragment="canonical output",
            actual_fragment="totally different",
            recommendation="Add diverse few-shot examples.",
        )
        assert fl.level == "semantic"
        assert fl.deviation_score == 0.55
        assert "few-shot" in fl.recommendation


# ── LevelBreakdown ────────────────────────────────────────────────────────────

class TestLevelBreakdown:
    def test_creation(self):
        lb = LevelBreakdown(
            level="lexical",
            score=0.10,
            variant_count=8,
            deviant_count=1,
            verdict="ROBUST",
        )
        assert lb.level == "lexical"
        assert lb.variant_count == 8
        assert lb.deviant_count == 1
        assert lb.verdict == "ROBUST"


# ── BrittleCertificate ────────────────────────────────────────────────────────

def _make_certificate(verdict="ROBUST", score=0.05, fault_lines=None) -> BrittleCertificate:
    return BrittleCertificate(
        certificate_id="shld_abc123",
        issued_at=datetime(2026, 3, 27, 12, 0, 0, tzinfo=timezone.utc),
        prompt_hash="sha256:abcd1234",
        prompt_name="test_prompt",
        verdict=verdict,
        brittleness_score=score,
        threshold=0.30,
        confidence_lower=0.02,
        confidence_upper=0.10,
        variant_count=16,
        level_breakdown=[
            LevelBreakdown("lexical", 0.05, 8, 0, "ROBUST"),
            LevelBreakdown("semantic", 0.05, 8, 0, "ROBUST"),
        ],
        fault_lines=fault_lines or [],
    )


class TestBrittleCertificate:
    def test_to_json_structure(self):
        cert = _make_certificate()
        raw = cert.to_json()
        data = json.loads(raw)

        assert data["certificate_id"] == "shld_abc123"
        assert data["verdict"] == "ROBUST"
        assert data["brittleness_score"] == 0.05
        assert data["threshold"] == 0.30
        assert "confidence_interval" in data
        assert len(data["confidence_interval"]) == 2
        assert "level_breakdown" in data
        assert "fault_lines" in data

    def test_to_json_level_breakdown_keys(self):
        cert = _make_certificate()
        data = json.loads(cert.to_json())
        assert "lexical" in data["level_breakdown"]
        assert "semantic" in data["level_breakdown"]

    def test_to_json_scores_rounded(self):
        cert = _make_certificate(score=0.123456789)
        data = json.loads(cert.to_json())
        assert data["brittleness_score"] == round(0.123456789, 4)

    def test_to_json_empty_fault_lines(self):
        cert = _make_certificate()
        data = json.loads(cert.to_json())
        assert data["fault_lines"] == []

    def test_to_json_with_fault_lines(self):
        fl = FaultLine("semantic", "a variant", 0.55, "canon", "actual", "fix this")
        cert = _make_certificate(verdict="BRITTLE", score=0.55, fault_lines=[fl])
        data = json.loads(cert.to_json())
        assert len(data["fault_lines"]) == 1
        assert data["fault_lines"][0]["level"] == "semantic"
        assert data["fault_lines"][0]["deviation_score"] == 0.55

    def test_to_markdown_robust(self):
        cert = _make_certificate(verdict="ROBUST")
        md = cert.to_markdown()
        assert "ROBUST" in md
        assert "✅" in md
        assert "test_prompt" in md
        assert "BrittlenessScore" in md
        assert "Level Breakdown" in md

    def test_to_markdown_brittle(self):
        cert = _make_certificate(verdict="BRITTLE", score=0.60)
        md = cert.to_markdown()
        assert "BRITTLE" in md
        assert "❌" in md

    def test_to_markdown_conditional(self):
        cert = _make_certificate(verdict="CONDITIONAL", score=0.25)
        md = cert.to_markdown()
        assert "CONDITIONAL" in md
        assert "⚠️" in md

    def test_to_markdown_includes_fault_lines(self):
        fl = FaultLine("lexical", "a variant", 0.40, "canon", "actual", "add synonyms")
        cert = _make_certificate(verdict="BRITTLE", score=0.40, fault_lines=[fl])
        md = cert.to_markdown()
        assert "Fault Lines" in md
        assert "a variant" in md
        assert "add synonyms" in md

    def test_to_markdown_no_fault_lines_section_when_empty(self):
        cert = _make_certificate(verdict="ROBUST")
        md = cert.to_markdown()
        assert "Fault Lines" not in md

    def test_certificate_id_preserved(self):
        cert = _make_certificate()
        data = json.loads(cert.to_json())
        assert data["certificate_id"] == "shld_abc123"

    def test_issued_at_isoformat(self):
        cert = _make_certificate()
        data = json.loads(cert.to_json())
        assert "2026-03-27" in data["issued_at"]


# ── BrittlenessResult ─────────────────────────────────────────────────────────

class TestBrittlenessResult:
    def test_creation(self):
        cert = _make_certificate()
        result = BrittlenessResult(
            score=0.05,
            verdict="ROBUST",
            certificate=cert,
            variant_results=[],
            test_input_count=2,
            run_duration_seconds=0.5,
        )
        assert result.score == 0.05
        assert result.verdict == "ROBUST"
        assert result.test_input_count == 2
        assert result.run_duration_seconds == 0.5

    def test_certificate_reference(self):
        cert = _make_certificate()
        result = BrittlenessResult(
            score=0.0,
            verdict="ROBUST",
            certificate=cert,
            variant_results=[],
            test_input_count=1,
            run_duration_seconds=0.1,
        )
        assert result.certificate is cert
