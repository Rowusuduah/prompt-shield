# prompt-shield

**Catch brittle prompts before production does.**

`prompt-shield` runs your LLM function against semantically-equivalent paraphrases of your test inputs. If outputs diverge, your prompt is brittle — and production will find it before you do.

## Install

```bash
pip install prompt-shield
```

## Quick Start

```python
from prompt_shield import BrittlenessRunner

def my_llm(user_input: str) -> str:
    return call_my_llm(user_input)  # your LLM function

runner = BrittlenessRunner(llm_function=my_llm)
result = runner.run(
    test_inputs=["What is the return policy?"],
    prompt_name="support_prompt",
)

print(result.verdict)   # ROBUST / CONDITIONAL / BRITTLE
print(result.score)     # BrittlenessScore (0.0–1.0)
print(result.certificate.to_markdown())
```

## Three Stress Levels

Based on Matthew 7:24-27 (Two Builders) — three storm vectors:

| Level | Vector | Example |
|-------|--------|---------|
| `lexical` | Rain — synonym substitution | "What is" → "What's the meaning of" |
| `syntactic` | Streams — structural transformation | "What is X?" → "Tell me about X" |
| `semantic` | Wind — full meaning reformulation | "How do I cancel?" → "I'd like to end my subscription" |

## CLI

```bash
# Run audit
shield run --config shield.yaml

# CI gate (exit 0 = pass, 1 = brittle)
shield ci --config shield.yaml

# History
shield report --store ./shield.db
```

## Verdicts

| Verdict | BrittlenessScore | Meaning |
|---------|-----------------|---------|
| `ROBUST` | ≤ 0.15 | Prompt handles paraphrase variation |
| `CONDITIONAL` | 0.15–0.30 | Some sensitivity — review fault lines |
| `BRITTLE` | > 0.30 | Prompt relies on surface form — fix before deploying |

## Biblical Pattern

PAT-048 (Daniel 5:25-28 — TEKEL): The prompt is weighed on the scales.
PAT-049 (Matthew 7:24-27 — Two Builders): Three storm levels stress-test every prompt.
PAT-050 (Proverbs 17:3 — The Crucible): The BrittleCertificate is the crucible output.

## License

MIT
