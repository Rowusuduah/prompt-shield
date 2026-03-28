"""
prompt_shield/pytest_plugin.py — pytest plugin for prompt-shield.

Registered via pyproject.toml entry_points:
    [project.entry-points."pytest11"]
    shield = "prompt_shield.pytest_plugin"

Adds the --shield-threshold option to pytest.
"""
from __future__ import annotations


def pytest_addoption(parser):
    """Add prompt-shield options to pytest CLI."""
    group = parser.getgroup("prompt-shield", "prompt-shield brittleness testing")
    group.addoption(
        "--shield-threshold",
        action="store",
        default=0.30,
        type=float,
        help="Global brittleness threshold for @brittle_check decorators (default: 0.30)",
    )
    group.addoption(
        "--shield-report",
        action="store_true",
        default=False,
        help="Print a brittleness summary after the test session",
    )


def pytest_configure(config):
    """Register markers."""
    config.addinivalue_line(
        "markers",
        "brittle_check: marks tests that run prompt brittleness checks",
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print brittleness summary if --shield-report is enabled."""
    if not config.getoption("--shield-report", default=False):
        return

    # Collect any _last_shield_result attributes from the session
    results = getattr(terminalreporter.config, "_shield_results", [])
    if not results:
        return

    terminalreporter.write_sep("=", "prompt-shield brittleness summary")
    for r in results:
        verdict_symbol = {"ROBUST": "✅", "CONDITIONAL": "⚠️", "BRITTLE": "❌"}[r.verdict]
        terminalreporter.write_line(
            f"  {verdict_symbol} {r.certificate.prompt_name}: "
            f"score={r.score:.4f} ({r.verdict})"
        )
