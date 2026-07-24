#!/usr/bin/env bash
# Run the bounded-v2 diagonal role-budget ablation sequentially on the fixed BBEH subset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Keep the intended model explicit while permitting an intentional caller override.
MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
export MODEL_NAME

# Each pair is: per-solver question budget, unique-aggregator question budget.
ROLE_BUDGETS=(
  "0 0"
  "1 1"
  "2 2"
  "5 5"
)
VISIBILITIES=(hidden visible)

for role_budget in "${ROLE_BUDGETS[@]}"; do
  read -r solver_budget aggregator_budget <<< "$role_budget"
  if [[ "$solver_budget" == "0" && "$aggregator_budget" == "0" ]]; then
    current_visibilities=(hidden)
  else
    current_visibilities=("${VISIBILITIES[@]}")
  fi
  for visibility in "${current_visibilities[@]}"; do
    echo "Running BBEH bounded-v2: model=${MODEL_NAME}, N=3, S=${solver_budget}, A=${aggregator_budget}, visibility=${visibility}, T=100"
    uv run python main.py \
      --benchmark bbeh \
      --agent-system communication_budget \
      --async-run \
      --limit 80 \
      --concurrency 10 \
      --seed 42 \
      --solver-count 3 \
      --solver-communication-budget "$solver_budget" \
      --aggregator-communication-budget "$aggregator_budget" \
      --budget-visibility "$visibility" \
      --max-turns 100 \
      --aggregator-max-steps 100 \
      --protocol-version bounded-v2 \
      --temperature 1.0
  done
done
