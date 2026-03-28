"""
prompt_shield/models.py — Core data models for prompt-shield.

PAT-048 (Daniel 5:25-28 — TEKEL): The prompt is weighed on the scales.
PAT-049 (Matthew 7:24-27 — Two Builders): Three stress levels = rain/streams/wind.
PAT-050 (Proverbs 17:3 — The Crucible): Certificate is the crucible output — proof of refinement.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal


BrittlenessVerdict = Literal["ROBUST", "CONDITIONAL", "BRITTLE"]
ParaphraseLevel = Literal["lexical", "syntactic", "semantic"]


@dataclass
class ParaphraseVariant:
    """A single semantically-equivalent variant of an original input."""
    original: str
    variant: str
    level: ParaphraseLevel
    similarity_score: float  # cosine similarity to original (0.0-1.0)
    validated: bool = False  # True if similarity_score >= min_similarity threshold


@dataclass
class VariantResult:
    """Output produced by the LLM function for a single variant."""
    variant: ParaphraseVariant
    output: str
    deviation_score: float   # 1 - cosine_similarity(canonical_output, variant_output)
    is_deviant: bool         # True if deviation_score > deviation_threshold


@dataclass
class FaultLine:
    """A specific paraphrase variant that causes brittleness."""
    level: ParaphraseLevel
    variant: str
    deviation_score: float
    canonical_fragment: str  # first 100 chars of canonical output
    actual_fragment: str     # first 100 chars of variant output
    recommendation: str


@dataclass
class LevelBreakdown:
    """Brittleness breakdown by paraphrase level."""
    level: ParaphraseLevel
    score: float
    variant_count: int
    deviant_count: int
    verdict: BrittlenessVerdict


@dataclass
class BrittleCertificate:
    """
    The crucible output — PAT-050 (Proverbs 17:3).
    Structured artifact proving the prompt passed (or failed) the brittleness audit.
    """
    certificate_id: str
    issued_at: datetime
    prompt_hash: str
    prompt_name: str
    verdict: BrittlenessVerdict
    brittleness_score: float
    threshold: float
    confidence_lower: float
    confidence_upper: float
    variant_count: int
    level_breakdown: list[LevelBreakdown]
    fault_lines: list[FaultLine]

    def to_json(self) -> str:
        data = {
            "certificate_id": self.certificate_id,
            "issued_at": self.issued_at.isoformat(),
            "prompt_hash": self.prompt_hash,
            "prompt_name": self.prompt_name,
            "verdict": self.verdict,
            "brittleness_score": round(self.brittleness_score, 4),
            "threshold": self.threshold,
            "confidence_interval": [
                round(self.confidence_lower, 4),
                round(self.confidence_upper, 4)
            ],
            "variant_count": self.variant_count,
            "level_breakdown": {
                b.level: {
                    "score": round(b.score, 4),
                    "variant_count": b.variant_count,
                    "deviant_count": b.deviant_count,
                    "verdict": b.verdict
                }
                for b in self.level_breakdown
            },
            "fault_lines": [
                {
                    "level": f.level,
                    "variant": f.variant,
                    "deviation_score": round(f.deviation_score, 4),
                    "recommendation": f.recommendation
                }
                for f in self.fault_lines
            ]
        }
        return json.dumps(data, indent=2)

    def to_markdown(self) -> str:
        verdict_emoji = {"ROBUST": "✅", "CONDITIONAL": "⚠️", "BRITTLE": "❌"}[self.verdict]
        lines = [
            f"# BrittleCertificate — {self.prompt_name}",
            f"",
            f"**Verdict:** {verdict_emoji} {self.verdict}",
            f"**BrittlenessScore:** {self.brittleness_score:.4f} (threshold: {self.threshold})",
            f"**Confidence Interval:** [{self.confidence_lower:.4f}, {self.confidence_upper:.4f}]",
            f"**Variants Tested:** {self.variant_count}",
            f"**Issued:** {self.issued_at.isoformat()}",
            f"**Certificate ID:** `{self.certificate_id}`",
            f"",
            f"## Level Breakdown",
            f"",
            f"| Level | Score | Deviant/Total | Verdict |",
            f"|-------|-------|---------------|---------|",
        ]
        for b in self.level_breakdown:
            v = {"ROBUST": "✅", "CONDITIONAL": "⚠️", "BRITTLE": "❌"}[b.verdict]
            lines.append(f"| {b.level} | {b.score:.4f} | {b.deviant_count}/{b.variant_count} | {v} {b.verdict} |")

        if self.fault_lines:
            lines.extend(["", "## Fault Lines", ""])
            for i, f in enumerate(self.fault_lines, 1):
                lines.extend([
                    f"### {i}. {f.level.capitalize()} brittleness",
                    f"**Variant:** `{f.variant}`",
                    f"**Deviation:** {f.deviation_score:.4f}",
                    f"**Recommendation:** {f.recommendation}",
                    ""
                ])

        return "\n".join(lines)


@dataclass
class BrittlenessResult:
    """Complete result of a brittleness check run."""
    score: float
    verdict: BrittlenessVerdict
    certificate: BrittleCertificate
    variant_results: list[VariantResult]
    test_input_count: int
    run_duration_seconds: float
