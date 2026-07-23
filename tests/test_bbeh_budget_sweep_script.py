"""Regression checks for the reproducible BBEH communication-budget sweep."""

from pathlib import Path


def test_bbeh_budget_sweep_covers_all_budget_visibility_pairs() -> None:
    script = Path(__file__).resolve().parents[1] / "run_bbeh_communication_sweep.sh"
    content = script.read_text(encoding="utf-8")

    assert 'BUDGETS=(50 20 10 5 2 1 0)' in content
    assert 'VISIBILITIES=(hidden visible)' in content
    assert '--limit 80' in content
    assert '--concurrency 10' in content
    assert '--solver-count 3' in content
    assert '--max-turns 100' in content
    assert '--seed 42' in content
