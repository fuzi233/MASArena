#!/usr/bin/env bash
# Run the six-cell bounded-v2 role-budget ablation with qwen3.5-plus.
# Before running, configure your provider outside the repository, for example:
#   export QWEN_API_KEY='...'
#   export QWEN_API_BASE='https://<your-openai-compatible-endpoint>/v1'

set -euo pipefail

: "${QWEN_API_KEY:?Set QWEN_API_KEY before running this script.}"
: "${QWEN_API_BASE:?Set QWEN_API_BASE to your OpenAI-compatible Qwen endpoint before running this script.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="qwen3.5-plus"
export MODEL_NAME
export OPENAI_API_KEY="$QWEN_API_KEY"
export OPENAI_API_BASE="$QWEN_API_BASE"

# Each pair is: per-solver question budget, unique-aggregator question budget.
ROLE_BUDGETS=(
  "0 0"
  "1 0"
  "0 1"
  "1 1"
  "1 2"
  "2 2"
)

for role_budget in "${ROLE_BUDGETS[@]}"; do
  read -r solver_budget aggregator_budget <<< "$role_budget"
  echo "Running BBEH bounded-v2: model=${MODEL_NAME}, N=3, S=${solver_budget}, A=${aggregator_budget}, T=100"
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
    --budget-visibility visible \
    --max-turns 100 \
    --aggregator-max-steps 100 \
    --protocol-version bounded-v2 \
    --temperature 1.0
done
