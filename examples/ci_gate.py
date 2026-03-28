"""
prompt-shield CI gate example.

Demonstrates how to use prompt-shield as a deployment gate in CI/CD pipelines.
If any prompt's BrittlenessScore exceeds the threshold, the script exits with
code 1 (blocks deployment).

In production:
1. Replace mock_llm_* with real API calls.
2. Remove similarity_fn and deviation_fn (uses sentence-transformers instead).
3. Run via: python ci_gate.py
   Or via CLI: shield ci --config shield.yaml --threshold 0.30
"""
import sys
from prompt_shield import BrittlenessEngine, BrittlenessRunner


# -------------------------------------------------------------------
# Production LLM functions (replace these with real implementations)
# -------------------------------------------------------------------
def customer_service_handler(user_input: str) -> str:
    """
    Replace with real implementation:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            system="You are a helpful customer service assistant.",
            messages=[{"role": "user", "content": user_input}]
        )
        return response.content[0].text
    """
    return "I can help you with your account. What would you like to know?"


def search_handler(user_input: str) -> str:
    """Search prompt handler — replace with real implementation."""
    return "Here are the search results for your query."


# -------------------------------------------------------------------
# CI gate configuration
# -------------------------------------------------------------------
PROMPTS_TO_TEST = [
    {
        "name": "customer_service_prompt",
        "function": customer_service_handler,
        "test_inputs": [
            "What is my account balance?",
            "How do I reset my password?",
            "I need to dispute a charge",
        ],
        "threshold": 0.25,
        "variants_per_input": 8,
        "levels": ["lexical", "syntactic", "semantic"],
    },
    {
        "name": "search_prompt",
        "function": search_handler,
        "test_inputs": [
            "Find me recent news about AI",
            "Show me the latest articles on machine learning",
        ],
        "threshold": 0.30,
        "variants_per_input": 6,
        "levels": ["lexical", "semantic"],
    },
]


def run_ci_gate() -> bool:
    """
    Run brittleness audit on all configured prompts.
    Returns True if all passed, False if any are BRITTLE.
    """
    any_brittle = False
    print("=" * 60)
    print("prompt-shield CI Gate")
    print("=" * 60)

    for prompt_cfg in PROMPTS_TO_TEST:
        engine = BrittlenessEngine(
            variants_per_input=prompt_cfg["variants_per_input"],
            levels=prompt_cfg["levels"],
            # Remove similarity_fn in production (uses sentence-transformers)
            similarity_fn=lambda a, b: 0.85,
        )
        runner = BrittlenessRunner(
            llm_function=prompt_cfg["function"],
            engine=engine,
            store_path="./ci_shield.db",
            # Remove deviation_fn in production (uses sentence-transformers)
            deviation_fn=lambda a, b: 0.05,
        )

        result = runner.run(
            test_inputs=prompt_cfg["test_inputs"],
            threshold=prompt_cfg["threshold"],
            prompt_name=prompt_cfg["name"],
        )

        verdict_symbols = {"ROBUST": "[PASS]", "CONDITIONAL": "[WARN]", "BRITTLE": "[FAIL]"}
        symbol = verdict_symbols[result.verdict]
        print(f"{symbol} {prompt_cfg['name']}: score={result.score:.4f} ({result.verdict})")

        if result.verdict == "BRITTLE":
            any_brittle = True
            print("  Fault lines:")
            for fl in result.certificate.fault_lines[:3]:
                print(f"    [{fl.level}] '{fl.variant}'")
                print(f"    Deviation: {fl.deviation_score:.4f}")
                print(f"    Fix: {fl.recommendation}")

        # Save certificate to file
        cert_path = f"./shield-cert-{prompt_cfg['name']}.json"
        with open(cert_path, "w") as f:
            f.write(result.certificate.to_json())
        print(f"  Certificate: {cert_path}")

    print("=" * 60)
    if any_brittle:
        print("BRITTLENESS DETECTED — deployment blocked.")
        return False
    else:
        print("All prompts passed brittleness audit — deployment approved.")
        return True


if __name__ == "__main__":
    passed = run_ci_gate()
    sys.exit(0 if passed else 1)
