"""
prompt_shield/runner.py — Executes the brittleness audit.

PAT-048 (Daniel 5 — TEKEL Audit):
The runner is the independent evaluator. It weighs the prompt on the scales
using the user's own eval function as the measurement instrument —
just as Daniel used the words themselves as the measurement instrument.
"""
from __future__ import annotations

import hashlib
import math
import secrets
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from .engine import BrittlenessEngine
from .models import (
    BrittleCertificate,
    BrittlenessResult,
    BrittlenessVerdict,
    FaultLine,
    LevelBreakdown,
    VariantResult,
)
from .store import BrittlenessStore


class BrittlenessRunner:
    """
    Runs the TEKEL audit (PAT-048) — weighs each prompt on the scales.

    Accepts an optional ``deviation_fn`` to bypass ML model loading in tests:
        runner = BrittlenessRunner(llm_fn, deviation_fn=lambda a, b: 0.05)
    """

    ROBUST_THRESHOLD = 0.15
    BRITTLE_THRESHOLD = 0.30

    def __init__(
        self,
        llm_function: Callable[[str], str],
        engine: BrittlenessEngine = None,
        store_path: str = "./shield.db",
        deviation_threshold: float = 0.15,
        similarity_model: str = "all-MiniLM-L6-v2",
        deviation_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.llm_function = llm_function
        self.engine = engine or BrittlenessEngine()
        self.store = BrittlenessStore(store_path)
        self.deviation_threshold = deviation_threshold
        self._similarity_model = None
        self._similarity_model_name = similarity_model
        self._deviation_fn = deviation_fn  # injection point for tests

    def _load_similarity_model(self):
        if self._deviation_fn is not None:
            return  # test injection bypasses model loading
        if self._similarity_model is None:
            from sentence_transformers import SentenceTransformer
            self._similarity_model = SentenceTransformer(self._similarity_model_name)

    def _compute_deviation(self, canonical: str, variant_output: str) -> float:
        """Compute output deviation (0.0 = identical, 1.0 = completely different)."""
        if self._deviation_fn is not None:
            return self._deviation_fn(canonical, variant_output)

        from sentence_transformers import util
        self._load_similarity_model()
        embs = self._similarity_model.encode([canonical, variant_output])
        similarity = float(util.cos_sim(embs[0], embs[1]))
        return 1.0 - similarity

    def _compute_confidence_interval(self, deviants: int, total: int) -> tuple[float, float]:
        """Wilson score interval for proportion of deviants (95% confidence)."""
        if total == 0:
            return 0.0, 0.0
        n = total
        p = deviants / n
        z = 1.96  # 95% confidence
        center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / (1 + z**2 / n)
        return max(0.0, center - margin), min(1.0, center + margin)

    def _verdict(self, score: float, threshold: float) -> BrittlenessVerdict:
        if score <= self.ROBUST_THRESHOLD:
            return "ROBUST"
        elif score <= threshold:
            return "CONDITIONAL"
        else:
            return "BRITTLE"

    def _generate_recommendation(self, result: VariantResult) -> str:
        """Generate actionable fix recommendation for a fault line."""
        level = result.variant.level
        if level == "lexical":
            return (
                "Prompt is brittle to word-level variation. "
                "Add few-shot examples with diverse vocabulary. "
                "Avoid relying on specific keyword triggers."
            )
        elif level == "syntactic":
            return (
                "Prompt is brittle to sentence structure changes. "
                "Test prompt with questions phrased as statements and vice versa. "
                "Add explicit instruction: 'The user may phrase their request in different ways.'"
            )
        elif level == "semantic":
            return (
                "Prompt is brittle to semantic rephrasing. "
                "The prompt relies on surface-form patterns, not semantic understanding. "
                "Add diverse few-shot examples covering multiple phrasings of the same intent."
            )
        return "Review prompt for surface-form dependency."

    def run(
        self,
        test_inputs: list[str],
        threshold: float = 0.30,
        prompt_name: str = "unnamed_prompt",
    ) -> BrittlenessResult:
        """
        Run the TEKEL audit (PAT-048) — weigh the prompt on the scales.

        Args:
            test_inputs: List of test input strings to generate variants for.
            threshold: BrittlenessScore above which verdict is BRITTLE.
            prompt_name: Name for this prompt (used in certificate).

        Returns:
            BrittlenessResult with score, verdict, and BrittleCertificate.
        """
        start_time = time.time()
        all_variant_results: list[VariantResult] = []
        level_stats: dict[str, dict] = {}

        for test_input in test_inputs:
            # Get canonical output (the reference)
            canonical_output = self.llm_function(test_input)

            # Generate variants
            variants = self.engine.generate_variants(test_input)

            for variant in variants:
                if not variant.validated:
                    continue

                # Run LLM function on variant
                variant_output = self.llm_function(variant.variant)

                # Compute deviation
                deviation = self._compute_deviation(canonical_output, variant_output)
                is_deviant = deviation > self.deviation_threshold

                vr = VariantResult(
                    variant=variant,
                    output=variant_output,
                    deviation_score=deviation,
                    is_deviant=is_deviant
                )
                all_variant_results.append(vr)

                # Track level stats
                level = variant.level
                if level not in level_stats:
                    level_stats[level] = {"total": 0, "deviant": 0}
                level_stats[level]["total"] += 1
                if is_deviant:
                    level_stats[level]["deviant"] += 1

        # Compute overall score
        total = len(all_variant_results)
        deviants = sum(1 for r in all_variant_results if r.is_deviant)
        score = deviants / total if total > 0 else 0.0
        verdict = self._verdict(score, threshold)

        # Level breakdown
        level_breakdowns = []
        for level, stats in level_stats.items():
            lv = stats["deviant"] / stats["total"] if stats["total"] > 0 else 0.0
            level_breakdowns.append(LevelBreakdown(
                level=level,
                score=lv,
                variant_count=stats["total"],
                deviant_count=stats["deviant"],
                verdict=self._verdict(lv, threshold)
            ))

        # Fault lines — top 5 most deviant variants
        fault_lines = []
        sorted_deviant = sorted(
            [r for r in all_variant_results if r.is_deviant],
            key=lambda r: r.deviation_score,
            reverse=True
        )[:5]
        for r in sorted_deviant:
            fault_lines.append(FaultLine(
                level=r.variant.level,
                variant=r.variant.variant,
                deviation_score=r.deviation_score,
                canonical_fragment="",
                actual_fragment=r.output[:100],
                recommendation=self._generate_recommendation(r)
            ))

        # Confidence interval
        ci_lower, ci_upper = self._compute_confidence_interval(deviants, total)

        # Build certificate (PAT-050 — Proverbs 17:3 — the crucible output)
        prompt_hash = hashlib.sha256(str(test_inputs).encode()).hexdigest()[:16]
        certificate = BrittleCertificate(
            certificate_id=f"shld_{secrets.token_hex(8)}",
            issued_at=datetime.now(timezone.utc),
            prompt_hash=f"sha256:{prompt_hash}",
            prompt_name=prompt_name,
            verdict=verdict,
            brittleness_score=score,
            threshold=threshold,
            confidence_lower=ci_lower,
            confidence_upper=ci_upper,
            variant_count=total,
            level_breakdown=level_breakdowns,
            fault_lines=fault_lines
        )

        duration = time.time() - start_time

        result = BrittlenessResult(
            score=score,
            verdict=verdict,
            certificate=certificate,
            variant_results=all_variant_results,
            test_input_count=len(test_inputs),
            run_duration_seconds=duration
        )

        # Persist to store
        self.store.log_run(result)

        return result
