"""
Tests for prompt_shield/decorators.py

Tests run without LLM API keys. Uses injected similarity and deviation functions.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from prompt_shield.decorators import BrittlePromptError, brittle_check
from prompt_shield.engine import BrittlenessEngine
from prompt_shield.models import ParaphraseVariant


def _make_brittle_engine_variants():
    """Return one validated syntactic variant."""
    return [ParaphraseVariant("input", "what is the way to know this?", "syntactic", 0.85, True)]


class TestBrittlePromptError:
    def test_creation(self):
        from prompt_shield.models import BrittleCertificate, LevelBreakdown
        from datetime import datetime, timezone
        cert = BrittleCertificate(
            certificate_id="shld_x",
            issued_at=datetime.now(timezone.utc),
            prompt_hash="sha256:abc",
            prompt_name="p",
            verdict="BRITTLE",
            brittleness_score=0.60,
            threshold=0.30,
            confidence_lower=0.4,
            confidence_upper=0.8,
            variant_count=4,
            level_breakdown=[LevelBreakdown("lexical", 0.60, 4, 2, "BRITTLE")],
            fault_lines=[],
        )
        err = BrittlePromptError(0.60, 0.30, "BRITTLE", cert)
        assert err.score == 0.60
        assert err.threshold == 0.30
        assert err.verdict == "BRITTLE"
        assert err.certificate is cert

    def test_str_representation(self):
        err = BrittlePromptError(0.60, 0.30, "BRITTLE", None)
        assert "0.6" in str(err)
        assert "BRITTLE" in str(err)

    def test_is_exception(self):
        err = BrittlePromptError(0.60, 0.30, "BRITTLE", None)
        assert isinstance(err, Exception)


class TestBrittleCheckDecorator:
    def test_passthrough_in_production_mode(self, tmp_path):
        """Decorator is transparent when not in test/CI mode."""
        os.environ.pop("SHIELD_CHECK", None)
        os.environ.pop("PYTEST_CURRENT_TEST", None)

        @brittle_check(
            threshold=0.30,
            test_inputs=["What is Python?"],
            store_path=str(tmp_path / "shield.db"),
        )
        def my_fn(text: str) -> str:
            return f"result: {text}"

        # In production mode (no env vars), decorator passes through
        # We temporarily clear the PYTEST_CURRENT_TEST var
        pytest_var = os.environ.pop("PYTEST_CURRENT_TEST", None)
        try:
            result = my_fn("hello")
            assert result == "result: hello"
        finally:
            if pytest_var:
                os.environ["PYTEST_CURRENT_TEST"] = pytest_var

    def test_shield_config_attached_to_function(self, tmp_path):
        @brittle_check(
            threshold=0.25,
            variants=6,
            levels=["lexical"],
            store_path=str(tmp_path / "shield.db"),
        )
        def my_fn(text: str) -> str:
            return text

        assert my_fn._shield_config["threshold"] == 0.25
        assert my_fn._shield_config["variants"] == 6
        assert my_fn._shield_config["levels"] == ["lexical"]

    def test_function_name_preserved(self, tmp_path):
        @brittle_check(store_path=str(tmp_path / "shield.db"))
        def my_important_function(text: str) -> str:
            return text

        assert my_important_function.__name__ == "my_important_function"

    def test_shield_check_env_triggers_audit_robust(self, tmp_path, monkeypatch):
        """When SHIELD_CHECK=true, decorator runs audit and returns ROBUST result."""
        # Provide a syntactic input that generates no variants (no triggers)
        monkeypatch.setenv("SHIELD_CHECK", "true")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        # Engine that generates no valid variants → score=0.0 → ROBUST
        engine = BrittlenessEngine(
            levels=["syntactic"],
            similarity_fn=lambda a, b: 0.99,  # too high → rejected → no valid variants
        )

        @brittle_check(
            threshold=0.30,
            test_inputs=["What is Python?"],
            store_path=str(tmp_path / "shield.db"),
        )
        def my_fn(text: str) -> str:
            return f"answer: {text}"

        # Manually patch the engine in the wrapper by using the injection params
        # We test via brittle_check with similarity_fn
        @brittle_check(
            threshold=0.30,
            test_inputs=["What is Python?"],
            store_path=str(tmp_path / "shield2.db"),
            similarity_fn=lambda a, b: 0.99,   # rejects all → no variants → ROBUST
            deviation_fn=lambda a, b: 0.05,
        )
        def my_fn2(text: str) -> str:
            return f"answer: {text}"

        result = my_fn2("What is Python?")
        assert result == "answer: What is Python?"
        assert hasattr(my_fn2, "_last_shield_result")
        assert my_fn2._last_shield_result.verdict == "ROBUST"

    def test_brittle_prompt_raises_error(self, tmp_path, monkeypatch):
        """When score exceeds threshold and raise_on_brittle=True, raises BrittlePromptError."""
        monkeypatch.setenv("SHIELD_CHECK", "true")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        @brittle_check(
            threshold=0.30,
            test_inputs=["what's Python?"],  # "what's" → triggers syntactic variant
            store_path=str(tmp_path / "shield.db"),
            similarity_fn=lambda a, b: 0.85,   # validates the variant
            deviation_fn=lambda a, b: 0.90,    # very deviant → BRITTLE
            raise_on_brittle=True,
        )
        def brittle_fn(text: str) -> str:
            return f"answer: {text}"

        with pytest.raises(BrittlePromptError) as exc_info:
            brittle_fn("what's Python?")
        assert exc_info.value.verdict == "BRITTLE"

    def test_no_raise_when_raise_on_brittle_false(self, tmp_path, monkeypatch):
        """When raise_on_brittle=False, BRITTLE verdict doesn't raise."""
        monkeypatch.setenv("SHIELD_CHECK", "true")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        @brittle_check(
            threshold=0.30,
            test_inputs=["what's Python?"],
            store_path=str(tmp_path / "shield.db"),
            similarity_fn=lambda a, b: 0.85,
            deviation_fn=lambda a, b: 0.90,
            raise_on_brittle=False,
        )
        def my_fn(text: str) -> str:
            return f"answer: {text}"

        # Should not raise
        result = my_fn("what's Python?")
        assert result == "answer: what's Python?"

    def test_no_inputs_skips_audit(self, tmp_path, monkeypatch):
        """When no test_inputs provided and no positional args, skip audit."""
        monkeypatch.setenv("SHIELD_CHECK", "true")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        @brittle_check(
            test_inputs=None,
            store_path=str(tmp_path / "shield.db"),
        )
        def my_fn() -> str:
            return "no inputs"

        result = my_fn()
        assert result == "no inputs"
