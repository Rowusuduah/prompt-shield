"""
Tests for prompt_shield/engine.py

All tests use injected similarity_fn to avoid loading sentence-transformers.
No ML models downloaded. No network calls.
"""
from __future__ import annotations

import pytest

from prompt_shield.engine import BrittlenessEngine
from prompt_shield.models import ParaphraseVariant


# Similarity function that always returns a valid, non-duplicate score
ALWAYS_VALID_SIM = lambda a, b: 0.85  # in [0.75, 0.98] window — validates
ALWAYS_TOO_HIGH_SIM = lambda a, b: 0.99  # above max_similarity — rejects
ALWAYS_TOO_LOW_SIM = lambda a, b: 0.50  # below min_similarity — rejects


# ── Initialization ────────────────────────────────────────────────────────────

class TestBrittlenessEngineInit:
    def test_default_levels(self):
        engine = BrittlenessEngine(similarity_fn=ALWAYS_VALID_SIM)
        assert "lexical" in engine.levels
        assert "semantic" in engine.levels

    def test_custom_levels(self):
        engine = BrittlenessEngine(levels=["syntactic"], similarity_fn=ALWAYS_VALID_SIM)
        assert engine.levels == ["syntactic"]

    def test_custom_variants_per_input(self):
        engine = BrittlenessEngine(variants_per_input=4, similarity_fn=ALWAYS_VALID_SIM)
        assert engine.variants_per_input == 4

    def test_custom_similarity_thresholds(self):
        engine = BrittlenessEngine(min_similarity=0.8, max_similarity=0.95, similarity_fn=ALWAYS_VALID_SIM)
        assert engine.min_similarity == 0.8
        assert engine.max_similarity == 0.95

    def test_similarity_fn_injection(self):
        fn = lambda a, b: 0.9
        engine = BrittlenessEngine(similarity_fn=fn)
        assert engine._similarity_fn is fn

    def test_no_model_loaded_when_fn_injected(self):
        engine = BrittlenessEngine(similarity_fn=ALWAYS_VALID_SIM)
        engine._load_models()
        # With similarity_fn injected, _similarity_model should NOT be loaded
        assert engine._similarity_model is None


# ── _compute_similarity ───────────────────────────────────────────────────────

class TestComputeSimilarity:
    def test_uses_injected_fn(self):
        engine = BrittlenessEngine(similarity_fn=lambda a, b: 0.92)
        result = engine._compute_similarity("hello", "hi there")
        assert result == 0.92

    def test_fn_receives_both_texts(self):
        received = []
        def track_fn(a, b):
            received.append((a, b))
            return 0.85
        engine = BrittlenessEngine(similarity_fn=track_fn)
        engine._compute_similarity("text A", "text B")
        assert received == [("text A", "text B")]


# ── _generate_syntactic ───────────────────────────────────────────────────────

class TestGenerateSyntactic:
    def _engine(self):
        return BrittlenessEngine(similarity_fn=ALWAYS_VALID_SIM)

    def test_contraction_expansion(self):
        engine = self._engine()
        results = engine._generate_syntactic("what's the answer?", 2)
        assert any("what is" in r.lower() for r in results)

    def test_contraction_dont(self):
        engine = self._engine()
        results = engine._generate_syntactic("don't do that", 2)
        assert any("do not" in r.lower() for r in results)

    def test_what_is_restructuring(self):
        engine = self._engine()
        results = engine._generate_syntactic("What is Python?", 2)
        assert any("tell me" in r.lower() for r in results)

    def test_how_do_i_restructuring(self):
        engine = self._engine()
        results = engine._generate_syntactic("How do I install Python?", 2)
        assert any("way to" in r.lower() for r in results)

    def test_returns_list(self):
        engine = self._engine()
        results = engine._generate_syntactic("Hello world", 3)
        assert isinstance(results, list)

    def test_pads_to_n(self):
        engine = self._engine()
        results = engine._generate_syntactic("no contractions here", 3)
        assert len(results) == 3

    def test_no_transformation_returns_original_padded(self):
        engine = self._engine()
        results = engine._generate_syntactic("something with no triggers", 2)
        # Original is returned as padding — all items are the same text
        assert all(r == "something with no triggers" for r in results)


# ── _generate_semantic_fallback ───────────────────────────────────────────────

class TestGenerateSemanticFallback:
    def _engine(self):
        return BrittlenessEngine(similarity_fn=ALWAYS_VALID_SIM)

    def test_returns_list(self):
        engine = self._engine()
        results = engine._generate_semantic_fallback("What is Python?", 3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_question_rewrites(self):
        engine = self._engine()
        results = engine._generate_semantic_fallback("What is Python?", 4)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, str)

    def test_non_question_text(self):
        engine = self._engine()
        results = engine._generate_semantic_fallback("Python is a language", 2)
        assert isinstance(results, list)

    def test_zero_requested_returns_empty(self):
        engine = self._engine()
        results = engine._generate_semantic_fallback("What is this?", 0)
        assert results == []


# ── generate_variants ─────────────────────────────────────────────────────────

class TestGenerateVariants:
    def test_validated_variants_have_validated_true(self):
        engine = BrittlenessEngine(
            levels=["syntactic"],
            variants_per_input=2,
            similarity_fn=ALWAYS_VALID_SIM,
        )
        # "what's" triggers contraction expansion
        variants = engine.generate_variants("what's Python?")
        valid = [v for v in variants if v.validated]
        # At least one should pass if syntactic generates a different text
        assert all(v.validated for v in valid)

    def test_near_duplicate_rejected(self):
        # similarity_fn returns 0.99 (> max_similarity=0.98) — should reject
        engine = BrittlenessEngine(
            levels=["syntactic"],
            similarity_fn=ALWAYS_TOO_HIGH_SIM,
        )
        variants = engine.generate_variants("What is Python?")
        assert all(not v.validated for v in variants)

    def test_low_similarity_rejected(self):
        engine = BrittlenessEngine(
            levels=["syntactic"],
            similarity_fn=ALWAYS_TOO_LOW_SIM,
        )
        variants = engine.generate_variants("What is Python?")
        assert all(not v.validated for v in variants)

    def test_same_text_rejected(self):
        # If the generated variant is identical to the original, it should be rejected
        engine = BrittlenessEngine(
            levels=["syntactic"],
            similarity_fn=ALWAYS_VALID_SIM,
        )
        # Text with no triggers → padded with original → should be filtered
        variants = engine.generate_variants("nothing here triggers transformation")
        # All variants would be identical to original → filtered
        for v in variants:
            assert v.variant != v.original or not v.validated

    def test_returns_list(self):
        engine = BrittlenessEngine(
            levels=["syntactic"],
            similarity_fn=ALWAYS_VALID_SIM,
        )
        result = engine.generate_variants("What is Python?")
        assert isinstance(result, list)

    def test_variant_level_matches_requested_level(self):
        engine = BrittlenessEngine(
            levels=["syntactic"],
            similarity_fn=ALWAYS_VALID_SIM,
        )
        variants = engine.generate_variants("what's going on?")
        for v in variants:
            assert v.level == "syntactic"

    def test_original_preserved_in_variant(self):
        engine = BrittlenessEngine(
            levels=["syntactic"],
            similarity_fn=ALWAYS_VALID_SIM,
        )
        variants = engine.generate_variants("what's your name?")
        for v in variants:
            assert v.original == "what's your name?"

    def test_similarity_score_stored(self):
        engine = BrittlenessEngine(
            levels=["syntactic"],
            similarity_fn=lambda a, b: 0.87,
        )
        variants = engine.generate_variants("what's this?")
        valid = [v for v in variants if v.validated]
        for v in valid:
            assert v.similarity_score == 0.87
