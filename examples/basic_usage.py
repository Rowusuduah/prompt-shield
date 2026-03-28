"""
prompt-shield basic usage example.

Demonstrates how to run a brittleness audit on a prompt using a mock LLM.
In production, replace mock_llm with a real Anthropic/OpenAI call.
"""
from prompt_shield import BrittlenessEngine, BrittlenessRunner


# -------------------------------------------------------------------
# 1. Define your LLM function (replace with real API call in production)
# -------------------------------------------------------------------
def mock_llm(user_input: str) -> str:
    """Mock LLM — always returns a static response regardless of input."""
    return "The return policy allows returns within 30 days with a receipt."


# -------------------------------------------------------------------
# 2. Configure the brittleness engine
# -------------------------------------------------------------------
# - variants_per_input: how many paraphrase variants to generate per test input
# - levels: which stress levels to apply (lexical/syntactic/semantic)
# - similarity_fn: inject a mock similarity function (no ML model needed)
#   In production, omit similarity_fn and the engine uses sentence-transformers.
engine = BrittlenessEngine(
    variants_per_input=6,
    levels=["lexical", "syntactic", "semantic"],
    similarity_fn=lambda a, b: 0.85,  # mock — remove in production
)

# -------------------------------------------------------------------
# 3. Configure the runner
# -------------------------------------------------------------------
# - llm_function: your callable that takes a string and returns a string
# - engine: the BrittlenessEngine configured above
# - store_path: where to persist results (SQLite)
# - deviation_fn: inject a mock deviation function (no ML model needed)
#   In production, omit deviation_fn and the runner uses sentence-transformers.
runner = BrittlenessRunner(
    llm_function=mock_llm,
    engine=engine,
    store_path="./shield.db",
    deviation_fn=lambda a, b: 0.05,  # mock — remove in production
)

# -------------------------------------------------------------------
# 4. Run the brittleness audit
# -------------------------------------------------------------------
result = runner.run(
    test_inputs=[
        "What is the return policy?",
        "How do I get a refund?",
    ],
    threshold=0.30,     # above this score → BRITTLE
    prompt_name="customer_service_prompt",
)

# -------------------------------------------------------------------
# 5. Read the results
# -------------------------------------------------------------------
print(f"BrittlenessScore: {result.score:.4f}")   # 0.0 (robust) to 1.0 (brittle)
print(f"Verdict: {result.verdict}")               # ROBUST | CONDITIONAL | BRITTLE
print(f"Variants tested: {result.certificate.variant_count}")
print(f"Run duration: {result.run_duration_seconds:.3f}s")
print()
print("Level breakdown:")
for lb in result.certificate.level_breakdown:
    print(f"  {lb.level}: score={lb.score:.4f}, {lb.deviant_count}/{lb.variant_count} deviant ({lb.verdict})")

if result.certificate.fault_lines:
    print()
    print("Fault lines (top brittle variants):")
    for i, fl in enumerate(result.certificate.fault_lines, 1):
        print(f"  {i}. [{fl.level}] '{fl.variant}'")
        print(f"     Deviation: {fl.deviation_score:.4f}")
        print(f"     Fix: {fl.recommendation}")

print()
print("Full JSON certificate:")
print(result.certificate.to_json())
