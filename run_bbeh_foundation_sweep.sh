#!/usr/bin/env bash
# Run one approved gpt-4o-mini BBEH foundation budget pair sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APPROVED_PAIRS=(
  "0 0"
  "1 1"
  "2 2"
  "5 5"
  "10 10"
)

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <solver_budget> <aggregator_budget>" >&2
  exit 2
fi

solver_budget="$1"
aggregator_budget="$2"
selected_pair="$solver_budget $aggregator_budget"
pair_is_approved=false
for approved_pair in "${APPROVED_PAIRS[@]}"; do
  if [[ "$selected_pair" == "$approved_pair" ]]; then
    pair_is_approved=true
    break
  fi
done

if [[ "$pair_is_approved" != true ]]; then
  echo "Run exactly one approved budget pair: (0,0), (1,1), (2,2), (5,5), or (10,10)." >&2
  exit 2
fi

if [[ -z "${OPENAI_API_KEY:-}" && -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before launching the foundation sweep.}"
MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
export MODEL_NAME
LOG_DIR="logs/gpt4o-mini-foundation"
mkdir -p "$LOG_DIR"

VISIBILITIES=(hidden visible)

for visibility in "${VISIBILITIES[@]}"; do
  echo "Running foundation sweep: model=${MODEL_NAME}, S=${solver_budget}, A=${aggregator_budget}, visibility=${visibility}"
  run_stamp="$(date +%Y%m%d_%H%M%S)"
  log_file="$LOG_DIR/bbeh_s${solver_budget}_a${aggregator_budget}_${visibility}_${run_stamp}_$$.log"
  uv run python -u main.py \
    --benchmark bbeh \
    --data ../manifests/bbeh_probe_v1.jsonl \
    --agent-system communication_budget \
    --async-run \
    --limit 80 \
    --concurrency 4 \
    --seed 42 \
    --solver-count 3 \
    --solver-communication-budget "$solver_budget" \
    --aggregator-communication-budget "$aggregator_budget" \
    --budget-visibility "$visibility" \
    --max-turns 100 \
    --aggregator-max-steps 100 \
    --protocol-version bounded-v2 \
    --temperature 1.0 \
    --max-model-retries 20 \
    --retry-delay-seconds 2 \
    --results-dir results/gpt4o-mini-foundation \
    2>&1 | tee "$log_file"
done
