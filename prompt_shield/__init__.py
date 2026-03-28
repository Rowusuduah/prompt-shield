"""
prompt-shield — Catch brittle prompts before production does.

Brittleness testing under paraphrase for LLM prompts.
Three stress levels: lexical (rain), syntactic (streams), semantic (wind).
PAT-049 (Matthew 7:24-27 — Two Builders): Only the prompt built on rock survives all three.

Quick start:
    from prompt_shield import BrittlenessRunner, BrittlenessEngine

    def my_llm(user_input: str) -> str:
        return call_my_llm(user_input)

    runner = BrittlenessRunner(llm_function=my_llm)
    result = runner.run(
        test_inputs=["What is the return policy?"],
        prompt_name="support_prompt",
    )
    print(result.verdict)  # ROBUST / CONDITIONAL / BRITTLE
    print(result.certificate.to_markdown())
"""
from .models import (
    BrittleCertificate,
    BrittlenessResult,
    BrittlenessVerdict,
    FaultLine,
    LevelBreakdown,
    ParaphraseLevel,
    ParaphraseVariant,
    VariantResult,
)
from .engine import BrittlenessEngine
from .runner import BrittlenessRunner
from .store import BrittlenessStore
from .decorators import brittle_check, BrittlePromptError

__version__ = "0.1.0"
__all__ = [
    "BrittlenessEngine",
    "BrittlenessRunner",
    "BrittlenessStore",
    "BrittleCertificate",
    "BrittlenessResult",
    "BrittlenessVerdict",
    "FaultLine",
    "LevelBreakdown",
    "ParaphraseLevel",
    "ParaphraseVariant",
    "VariantResult",
    "brittle_check",
    "BrittlePromptError",
]
