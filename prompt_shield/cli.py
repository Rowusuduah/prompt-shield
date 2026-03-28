"""
prompt_shield/cli.py — CLI for prompt-shield.

Entry point: shield

Commands:
  shield run     — run brittleness audit and display results
  shield ci      — CI gate (exit 0=pass, 1=brittle)
  shield report  — show brittleness history from SQLite store
"""
from __future__ import annotations

import sys
from pathlib import Path

import click


@click.group()
def cli():
    """prompt-shield — Catch brittle prompts before production does."""
    pass


@cli.command()
@click.option("--config", default="shield.yaml", help="Config file path")
@click.option("--threshold", default=0.30, type=float, help="Brittleness threshold (0.0-1.0)")
@click.option("--output", default=None, help="Output certificate JSON path")
def run(config, threshold, output):
    """Run brittleness audit and display results."""
    from .config import load_config
    from .engine import BrittlenessEngine
    from .runner import BrittlenessRunner

    cfg = load_config(config)

    for prompt_cfg in cfg.get("prompts", []):
        engine = BrittlenessEngine(
            variants_per_input=prompt_cfg.get("variants_per_input", 8),
            levels=prompt_cfg.get("levels", ["lexical", "semantic"]),
        )
        llm_function = _load_function(prompt_cfg["function"])
        runner = BrittlenessRunner(llm_function=llm_function, engine=engine)

        result = runner.run(
            test_inputs=prompt_cfg["test_inputs"],
            threshold=prompt_cfg.get("threshold", threshold),
            prompt_name=prompt_cfg["name"],
        )

        click.echo(f"\n{'='*50}")
        click.echo(f"Prompt: {prompt_cfg['name']}")
        click.echo(f"BrittlenessScore: {result.score:.4f}")
        click.echo(f"Verdict: {result.verdict}")
        click.echo(f"Variants tested: {result.certificate.variant_count}")

        cert_path = output or (cfg.get("output") or {}).get("certificate")
        if cert_path:
            Path(cert_path).write_text(result.certificate.to_json())
            click.echo(f"Certificate written to: {cert_path}")


@cli.command()
@click.option("--config", default="shield.yaml", help="Config file path")
@click.option("--threshold", default=0.30, type=float, help="Brittleness threshold")
def ci(config, threshold):
    """
    CI gate — exits 0 if ROBUST/CONDITIONAL, 1 if BRITTLE.
    The TEKEL test (PAT-048): weigh on the scales. Exit code is the verdict.
    """
    from .config import load_config
    from .engine import BrittlenessEngine
    from .runner import BrittlenessRunner

    cfg = load_config(config)
    any_brittle = False

    for prompt_cfg in cfg.get("prompts", []):
        engine = BrittlenessEngine(
            variants_per_input=prompt_cfg.get("variants_per_input", 8),
            levels=prompt_cfg.get("levels", ["lexical", "semantic"]),
        )
        llm_function = _load_function(prompt_cfg["function"])
        runner = BrittlenessRunner(llm_function=llm_function, engine=engine)

        result = runner.run(
            test_inputs=prompt_cfg["test_inputs"],
            threshold=prompt_cfg.get("threshold", threshold),
            prompt_name=prompt_cfg["name"],
        )

        verdict_symbol = {"ROBUST": "✅", "CONDITIONAL": "⚠️", "BRITTLE": "❌"}[result.verdict]
        click.echo(f"{verdict_symbol} {prompt_cfg['name']}: {result.score:.4f} ({result.verdict})")

        if result.verdict == "BRITTLE":
            any_brittle = True
            click.echo("  Fault lines:")
            for fl in result.certificate.fault_lines[:3]:
                click.echo(f"    [{fl.level}] '{fl.variant}' → deviation {fl.deviation_score:.4f}")
                click.echo(f"    Fix: {fl.recommendation}")

        outputs = cfg.get("output") or {}
        if outputs.get("certificate"):
            Path(outputs["certificate"]).write_text(result.certificate.to_json())

    if any_brittle:
        click.echo("\n❌ BRITTLENESS DETECTED — deployment blocked.")
        sys.exit(1)
    else:
        click.echo("\n✅ All prompts passed the brittleness audit.")
        sys.exit(0)


@cli.command()
@click.option("--store", default="./shield.db", help="Store database path")
@click.option("--prompt", default=None, help="Filter by prompt name")
def report(store, prompt):
    """Show brittleness history from the SQLite store."""
    from .store import BrittlenessStore

    bs = BrittlenessStore(store_path=store)
    rows = bs.get_runs(prompt_name=prompt, limit=20)

    click.echo(f"\n{'Timestamp':<30} {'Prompt':<30} {'Score':<10} Verdict")
    click.echo("-" * 80)
    for row in rows:
        click.echo(
            f"{row['run_at']:<30} {row['prompt_name']:<30} "
            f"{row['brittleness_score']:<10.4f} {row['verdict']}"
        )

    if not rows:
        click.echo("No runs found.")


def _load_function(function_path: str):
    """Load a Python function from a dotted path string (e.g. 'mymodule.my_fn')."""
    module_path, func_name = function_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, func_name)
