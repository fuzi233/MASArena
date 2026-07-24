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


def test_bounded_v2_role_sweep_covers_four_diagonal_budget_visibility_pairs() -> None:
    script = Path(__file__).resolve().parents[1] / "run_bbeh_bounded_v2_role_sweep.sh"
    content = script.read_text(encoding="utf-8")

    assert 'MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"' in content
    assert 'ROLE_BUDGETS=(' in content
    assert 'VISIBILITIES=(hidden visible)' in content
    assert 'if [[ "$solver_budget" == "0" && "$aggregator_budget" == "0" ]]' in content
    for pair in ('"0 0"', '"1 1"', '"2 2"', '"5 5"'):
        assert pair in content
    assert '"1 0"' not in content
    assert '"2 10"' not in content
    assert '--protocol-version bounded-v2' in content
    assert '--budget-visibility "$visibility"' in content
    assert '--max-turns 100' in content
    assert '--aggregator-max-steps 100' in content


def test_foundation_sweep_covers_all_approved_pairs_with_retry_policy() -> None:
    script = Path(__file__).resolve().parents[1] / "run_bbeh_foundation_sweep.sh"
    content = script.read_text(encoding="utf-8")

    assert 'MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"' in content
    assert 'APPROVED_PAIRS=(' in content
    assert 'VISIBILITIES=(hidden visible)' in content
    for pair in ('"0 0"', '"1 1"', '"2 2"', '"5 5"', '"10 10"'):
        assert pair in content
    assert 'Usage: $0 <solver_budget> <aggregator_budget>' in content
    assert 'Run exactly one approved budget pair' in content
    assert '[[ -z "${OPENAI_API_KEY:-}" && -f .env ]]' in content
    assert 'source .env' in content
    assert 'for visibility in "${VISIBILITIES[@]}"; do' in content
    assert 'for role_budget in' not in content
    assert '--limit 80' in content
    assert '--concurrency 4' in content
    assert '--seed 42' in content
    assert '--max-model-retries 20' in content
    assert '--retry-delay-seconds 2' in content
    assert '--protocol-version bounded-v2' in content
    assert 'LOG_DIR="logs/gpt4o-mini-foundation"' in content
    assert 'mkdir -p "$LOG_DIR"' in content
    assert 'uv run python -u main.py' in content
    assert 'tee "$log_file"' in content


def test_qwen_role_sweep_requires_external_api_configuration() -> None:
    script = Path(__file__).resolve().parents[1] / "run_bbeh_bounded_v2_role_sweep_qwen.sh"
    content = script.read_text(encoding="utf-8")

    assert 'MODEL_NAME="qwen3.5-plus"' in content
    assert 'QWEN_API_KEY' in content
    assert 'QWEN_API_BASE' in content
    assert 'export OPENAI_API_KEY="$QWEN_API_KEY"' in content
    assert 'export OPENAI_API_BASE="$QWEN_API_BASE"' in content
    for pair in ('"0 0"', '"1 0"', '"0 1"', '"1 1"', '"1 2"', '"2 2"'):
        assert pair in content
    assert '"2 5"' not in content
    assert '"2 10"' not in content
    assert '--max-turns 100' in content
    assert '--aggregator-max-steps 100' in content


def test_deepseek_pilot_sweep_reuses_project_api_configuration_for_four_diagonal_cells() -> None:
    script = Path(__file__).resolve().parents[1] / "run_bbeh_bounded_v2_deepseek_pilot.sh"
    content = script.read_text(encoding="utf-8")

    assert 'MODEL_NAME="DeepSeek-V4-Flash"' in content
    assert 'DEEPSEEK_API_KEY' not in content
    assert 'export OPENAI_API_KEY=' not in content
    assert 'OPENAI_API_BASE=' not in content
    assert 'VISIBILITIES=(hidden visible)' in content
    assert 'if [[ "$solver_budget" == "0" && "$aggregator_budget" == "0" ]]' in content
    for pair in ('"0 0"', '"1 1"', '"2 2"', '"5 5"'):
        assert pair in content
    assert '"1 0"' not in content
    assert '--limit 80' in content
    assert '--budget-visibility "$visibility"' in content
    assert '--max-turns 100' in content
    assert '--aggregator-max-steps 100' in content
