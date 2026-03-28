"""
Tests for prompt_shield/runner.py

All tests use injected similarity_fn + deviation_fn to avoid ML model loading.
No network calls. No LLM API calls.
"""
from __future__ import annotations

import tempfile
import os
from pathlib import Path

import pytest

from prompt_shield.engine import BrittlenessEngine
from prompt_shield.models import ParaphraseVariant, VariantResult
from prompt_shield.runner import BrittlenessRunner


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_engine_with_variants(variants: list[ParaphraseVariant], similarity_fn=None):
    """Engine whose generate_variants() always returns the given list."""
    engine = BrittlenessEngine(
        levels=["syntactic"],
        similarity_fn=similarity_fn or (lambda a, b: 0.85),
    )
    engine.generate_variants = lambda text: variants
    return engine


def identity_llm(text: str) -> str:
    """LLM that echoes input — always ROBUST."""
    return f"response to: {text}"


def brittle_llm(calls=[0]):
    """LLM that gives completely different outputs each call."""
    def fn(text: str) -> str:
        calls[0] += 1
        return f"output_{calls[0]}_xyz_abcdefghijklmnop"
    return fn


def _variant(original="in", variant_text="variant", level="syntactic") -> ParaphraseVariant:
    return ParaphraseVariant(
        original=original,
        variant=variant_text,
        level=level,
        similarity_score=0.85,
        validated=True,
    )


# ── Initialization ────────────────────────────────────────────────────────────

class TestBrittlenessRunnerInit:
    def test_default_deviation_threshold(self, tmp_path):
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            store_path=str(tmp_path / "shield.db"),
        )
        assert runner.deviation_threshold == 0.15

    def test_custom_deviation_threshold(self, tmp_path):
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            deviation_threshold=0.25,
            store_path=str(tmp_path / "shield.db"),
        )
        assert runner.deviation_threshold == 0.25

    def test_deviation_fn_injection(self, tmp_path):
        fn = lambda a, b: 0.05
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            deviation_fn=fn,
            store_path=str(tmp_path / "shield.db"),
        )
        assert runner._deviation_fn is fn


# ── _compute_deviation ────────────────────────────────────────────────────────

class TestComputeDeviation:
    def test_uses_injected_fn(self, tmp_path):
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            deviation_fn=lambda a, b: 0.42,
            store_path=str(tmp_path / "shield.db"),
        )
        result = runner._compute_deviation("canonical", "variant")
        assert result == 0.42

    def test_deviation_fn_receives_both_texts(self, tmp_path):
        received = []
        def fn(a, b):
            received.append((a, b))
            return 0.1
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            deviation_fn=fn,
            store_path=str(tmp_path / "shield.db"),
        )
        runner._compute_deviation("canonical_text", "variant_text")
        assert received == [("canonical_text", "variant_text")]


# ── _compute_confidence_interval ──────────────────────────────────────────────

class TestComputeConfidenceInterval:
    def _runner(self, tmp_path):
        return BrittlenessRunner(
            llm_function=identity_llm,
            deviation_fn=lambda a, b: 0.05,
            store_path=str(tmp_path / "shield.db"),
        )

    def test_zero_total_returns_zeros(self, tmp_path):
        runner = self._runner(tmp_path)
        lo, hi = runner._compute_confidence_interval(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_zero_deviants(self, tmp_path):
        runner = self._runner(tmp_path)
        lo, hi = runner._compute_confidence_interval(0, 16)
        assert 0.0 <= lo <= hi
        assert hi <= 0.25  # should be close to 0

    def test_all_deviants(self, tmp_path):
        runner = self._runner(tmp_path)
        lo, hi = runner._compute_confidence_interval(16, 16)
        assert lo > 0.75  # should be close to 1
        assert hi <= 1.0

    def test_half_deviants(self, tmp_path):
        runner = self._runner(tmp_path)
        lo, hi = runner._compute_confidence_interval(8, 16)
        assert lo < 0.5 < hi  # CI straddles 0.5

    def test_bounds_within_zero_one(self, tmp_path):
        runner = self._runner(tmp_path)
        for deviants, total in [(0, 5), (3, 5), (5, 5), (1, 100)]:
            lo, hi = runner._compute_confidence_interval(deviants, total)
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0
            assert lo <= hi


# ── _verdict ──────────────────────────────────────────────────────────────────

class TestVerdict:
    def _runner(self, tmp_path):
        return BrittlenessRunner(
            llm_function=identity_llm,
            deviation_fn=lambda a, b: 0.05,
            store_path=str(tmp_path / "shield.db"),
        )

    def test_robust_at_zero(self, tmp_path):
        runner = self._runner(tmp_path)
        assert runner._verdict(0.0, 0.30) == "ROBUST"

    def test_robust_at_threshold(self, tmp_path):
        runner = self._runner(tmp_path)
        assert runner._verdict(0.15, 0.30) == "ROBUST"

    def test_conditional_above_robust(self, tmp_path):
        runner = self._runner(tmp_path)
        assert runner._verdict(0.20, 0.30) == "CONDITIONAL"

    def test_conditional_at_brittle_threshold(self, tmp_path):
        runner = self._runner(tmp_path)
        assert runner._verdict(0.30, 0.30) == "CONDITIONAL"

    def test_brittle_above_threshold(self, tmp_path):
        runner = self._runner(tmp_path)
        assert runner._verdict(0.31, 0.30) == "BRITTLE"

    def test_brittle_at_one(self, tmp_path):
        runner = self._runner(tmp_path)
        assert runner._verdict(1.0, 0.30) == "BRITTLE"


# ── _generate_recommendation ──────────────────────────────────────────────────

class TestGenerateRecommendation:
    def _runner(self, tmp_path):
        return BrittlenessRunner(
            llm_function=identity_llm,
            deviation_fn=lambda a, b: 0.05,
            store_path=str(tmp_path / "shield.db"),
        )

    def _make_vr(self, level):
        v = ParaphraseVariant("orig", "var", level, 0.85, True)
        return VariantResult(variant=v, output="out", deviation_score=0.5, is_deviant=True)

    def test_lexical_recommendation(self, tmp_path):
        runner = self._runner(tmp_path)
        rec = runner._generate_recommendation(self._make_vr("lexical"))
        assert "word-level" in rec.lower() or "synonym" in rec.lower() or "keyword" in rec.lower()

    def test_syntactic_recommendation(self, tmp_path):
        runner = self._runner(tmp_path)
        rec = runner._generate_recommendation(self._make_vr("syntactic"))
        assert "structure" in rec.lower() or "statement" in rec.lower()

    def test_semantic_recommendation(self, tmp_path):
        runner = self._runner(tmp_path)
        rec = runner._generate_recommendation(self._make_vr("semantic"))
        assert "few-shot" in rec.lower() or "semantic" in rec.lower()

    def test_unknown_level_returns_fallback(self, tmp_path):
        runner = self._runner(tmp_path)
        v = ParaphraseVariant("o", "v", "lexical", 0.85, True)
        vr = VariantResult(v, "out", 0.5, True)
        # Monkey-patch level
        vr.variant.level = "unknown_level"  # type: ignore
        rec = runner._generate_recommendation(vr)
        assert isinstance(rec, str) and len(rec) > 0


# ── run() — ROBUST scenario ───────────────────────────────────────────────────

class TestRunRobust:
    def test_robust_verdict_with_low_deviation(self, tmp_path):
        variants = [_variant("in", "variant_a"), _variant("in", "variant_b")]
        engine = make_engine_with_variants(variants)
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=lambda a, b: 0.05,  # below threshold=0.15
        )
        result = runner.run(["What is Python?"], prompt_name="test")
        assert result.verdict == "ROBUST"
        assert result.score == 0.0  # 0 deviants
        assert result.certificate.prompt_name == "test"

    def test_robust_persisted_to_store(self, tmp_path):
        variants = [_variant()]
        engine = make_engine_with_variants(variants)
        db = str(tmp_path / "shield.db")
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=db,
            deviation_fn=lambda a, b: 0.05,
        )
        runner.run(["What is Python?"], prompt_name="my_prompt")
        from prompt_shield.store import BrittlenessStore
        store = BrittlenessStore(db)
        runs = store.get_runs()
        assert len(runs) == 1
        assert runs[0]["prompt_name"] == "my_prompt"
        assert runs[0]["verdict"] == "ROBUST"

    def test_run_returns_correct_test_input_count(self, tmp_path):
        engine = make_engine_with_variants([])  # no variants
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=lambda a, b: 0.05,
        )
        result = runner.run(["input1", "input2", "input3"])
        assert result.test_input_count == 3

    def test_run_duration_is_positive(self, tmp_path):
        engine = make_engine_with_variants([])
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=lambda a, b: 0.05,
        )
        result = runner.run(["input"])
        assert result.run_duration_seconds >= 0


# ── run() — BRITTLE scenario ──────────────────────────────────────────────────

class TestRunBrittle:
    def test_brittle_verdict_with_high_deviation(self, tmp_path):
        # 4 variants, all deviant → score = 1.0 > 0.30 threshold
        variants = [
            _variant("in", f"variant_{i}") for i in range(4)
        ]
        engine = make_engine_with_variants(variants)
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=lambda a, b: 0.90,  # above deviation_threshold=0.15
        )
        result = runner.run(["What is Python?"])
        assert result.verdict == "BRITTLE"
        assert result.score == 1.0

    def test_fault_lines_capped_at_five(self, tmp_path):
        variants = [_variant("in", f"v_{i}") for i in range(8)]
        engine = make_engine_with_variants(variants)
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=lambda a, b: 0.90,
        )
        result = runner.run(["What is Python?"])
        assert len(result.certificate.fault_lines) <= 5

    def test_fault_lines_sorted_by_deviation_desc(self, tmp_path):
        # Vary deviation scores per call
        call_count = [0]
        deviations = [0.90, 0.50, 0.80, 0.60]

        def varying_deviation(a, b):
            idx = call_count[0] % len(deviations)
            call_count[0] += 1
            return deviations[idx]

        variants = [_variant("in", f"v_{i}") for i in range(4)]
        engine = make_engine_with_variants(variants)
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=varying_deviation,
        )
        result = runner.run(["What is Python?"])
        scores = [fl.deviation_score for fl in result.certificate.fault_lines]
        assert scores == sorted(scores, reverse=True)


# ── run() — CONDITIONAL scenario ─────────────────────────────────────────────

class TestRunConditional:
    def test_conditional_verdict(self, tmp_path):
        # 1 of 4 variants deviant → score = 0.25, between 0.15 and 0.30
        call_count = [0]
        def conditional_deviation(a, b):
            call_count[0] += 1
            return 0.90 if call_count[0] == 1 else 0.05

        variants = [_variant("in", f"v_{i}") for i in range(4)]
        engine = make_engine_with_variants(variants)
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=conditional_deviation,
        )
        result = runner.run(["What is Python?"])
        assert result.verdict == "CONDITIONAL"
        assert result.score == 0.25


# ── run() — no valid variants ─────────────────────────────────────────────────

class TestRunNoVariants:
    def test_score_zero_when_no_variants(self, tmp_path):
        engine = make_engine_with_variants([])  # no variants
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=lambda a, b: 0.05,
        )
        result = runner.run(["What is Python?"])
        assert result.score == 0.0
        assert result.verdict == "ROBUST"
        assert result.certificate.variant_count == 0

    def test_unvalidated_variants_skipped(self, tmp_path):
        unvalidated = [
            ParaphraseVariant("in", "v1", "syntactic", 0.85, False),
            ParaphraseVariant("in", "v2", "syntactic", 0.85, False),
        ]
        engine = make_engine_with_variants(unvalidated)
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=lambda a, b: 0.90,  # would be deviant if processed
        )
        result = runner.run(["What is Python?"])
        assert result.score == 0.0  # no validated variants processed


# ── run() — level breakdown ───────────────────────────────────────────────────

class TestRunLevelBreakdown:
    def test_level_breakdown_populated(self, tmp_path):
        variants = [
            ParaphraseVariant("in", "lex_v", "lexical", 0.85, True),
            ParaphraseVariant("in", "syn_v", "syntactic", 0.85, True),
        ]
        engine = make_engine_with_variants(variants)
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=lambda a, b: 0.05,
        )
        result = runner.run(["input"])
        levels_found = {lb.level for lb in result.certificate.level_breakdown}
        assert "lexical" in levels_found
        assert "syntactic" in levels_found

    def test_level_breakdown_counts_correct(self, tmp_path):
        variants = [
            ParaphraseVariant("in", "lex_1", "lexical", 0.85, True),
            ParaphraseVariant("in", "lex_2", "lexical", 0.85, True),
            ParaphraseVariant("in", "sem_1", "semantic", 0.85, True),
        ]
        engine = make_engine_with_variants(variants)
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=lambda a, b: 0.05,
        )
        result = runner.run(["input"])
        by_level = {lb.level: lb for lb in result.certificate.level_breakdown}
        assert by_level["lexical"].variant_count == 2
        assert by_level["semantic"].variant_count == 1


# ── Certificate integrity ─────────────────────────────────────────────────────

class TestCertificateIntegrity:
    def test_certificate_id_starts_with_shld(self, tmp_path):
        engine = make_engine_with_variants([])
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=lambda a, b: 0.05,
        )
        result = runner.run(["input"])
        assert result.certificate.certificate_id.startswith("shld_")

    def test_certificate_ids_unique(self, tmp_path):
        engine = make_engine_with_variants([])
        db = str(tmp_path / "shield.db")
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=db,
            deviation_fn=lambda a, b: 0.05,
        )
        r1 = runner.run(["input"])
        r2 = runner.run(["input"])
        assert r1.certificate.certificate_id != r2.certificate.certificate_id

    def test_prompt_hash_has_sha256_prefix(self, tmp_path):
        engine = make_engine_with_variants([])
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=lambda a, b: 0.05,
        )
        result = runner.run(["input"])
        assert result.certificate.prompt_hash.startswith("sha256:")

    def test_threshold_preserved_in_certificate(self, tmp_path):
        engine = make_engine_with_variants([])
        runner = BrittlenessRunner(
            llm_function=identity_llm,
            engine=engine,
            store_path=str(tmp_path / "shield.db"),
            deviation_fn=lambda a, b: 0.05,
        )
        result = runner.run(["input"], threshold=0.25)
        assert result.certificate.threshold == 0.25
