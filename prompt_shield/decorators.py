"""
prompt_shield/decorators.py — @brittle_check decorator for LLM functions.

Runs a brittleness audit when the test suite executes, blocking deployment
of prompts with BrittlenessScore above the configured threshold.
"""
from __future__ import annotations

import functools
import os
from typing import Callable, Optional

from .engine import BrittlenessEngine
from .runner import BrittlenessRunner


class BrittlePromptError(Exception):
    """Raised when a decorated function's prompt exceeds the brittleness threshold."""

    def __init__(self, score: float, threshold: float, verdict: str, certificate):
        self.score = score
        self.threshold = threshold
        self.verdict = verdict
        self.certificate = certificate
        super().__init__(
            f"BrittlePromptError: score={score:.4f} > threshold={threshold} ({verdict})"
        )


def brittle_check(
    threshold: float = 0.30,
    variants: int = 8,
    levels: list = None,
    test_inputs: list[str] = None,
    raise_on_brittle: bool = True,
    store_path: str = "./shield.db",
    similarity_fn: Optional[Callable] = None,
    deviation_fn: Optional[Callable] = None,
):
    """
    Decorator that runs a brittleness audit when the test suite executes.

    Only activates when:
      - SHIELD_CHECK=true env var is set, OR
      - PYTEST_CURRENT_TEST env var is set (running under pytest)

    Usage:
        @brittle_check(threshold=0.25, variants=10, levels=["lexical", "semantic"])
        def my_llm_function(user_input: str) -> str:
            ...

    The decorator:
    1. Detects when running in test mode (pytest, or SHIELD_CHECK=true env var)
    2. Generates ``variants`` paraphrase variants per test input
    3. Runs the wrapped function on each variant
    4. Computes BrittlenessScore
    5. Raises BrittlePromptError if score > threshold and raise_on_brittle=True
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            in_test_mode = (
                os.environ.get("SHIELD_CHECK", "").lower() == "true"
                or os.environ.get("PYTEST_CURRENT_TEST") is not None
            )

            if not in_test_mode:
                return func(*args, **kwargs)

            inputs = test_inputs or (list(args[:1]) if args else [])
            if not inputs:
                return func(*args, **kwargs)

            engine = BrittlenessEngine(
                variants_per_input=variants,
                levels=levels or ["lexical", "semantic"],
                similarity_fn=similarity_fn,
            )

            def wrapped_func(text: str) -> str:
                new_args = (text,) + args[1:]
                return func(*new_args, **kwargs)

            runner = BrittlenessRunner(
                llm_function=wrapped_func,
                engine=engine,
                store_path=store_path,
                deviation_fn=deviation_fn,
            )
            result = runner.run(
                test_inputs=inputs,
                threshold=threshold,
                prompt_name=func.__name__,
            )

            if raise_on_brittle and result.verdict == "BRITTLE":
                raise BrittlePromptError(
                    score=result.score,
                    threshold=threshold,
                    verdict=result.verdict,
                    certificate=result.certificate,
                )

            wrapper._last_shield_result = result
            return func(*args, **kwargs)

        wrapper._shield_config = {
            "threshold": threshold,
            "variants": variants,
            "levels": levels,
        }
        return wrapper

    return decorator
