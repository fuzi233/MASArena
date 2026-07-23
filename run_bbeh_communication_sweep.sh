#!/usr/bin/env bash
# Run the full BBEH communication-budget sweep sequentially on one fixed subset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUDGETS=(50 20 10 5 2 1 0)
VISIBILITIES=(hidden visible)

for budget in "${BUDGETS[@]}"; do
  for visibility in "${VISIBILITIES[@]}"; do
    echo "Running BBEH: N=3, K=${budget}, visibility=${visibility}, T=100"
    uv run python main.py \
      --benchmark bbeh \
      --agent-system communication_budget \
      --async-run \
      --limit 80 \
      --concurrency 10 \
      --seed 42 \
      --solver-count 3 \
      --communication-budget "$budget" \
      --budget-visibility "$visibility" \
      --max-turns 100 \
      --temperature 1.0
  done
done
