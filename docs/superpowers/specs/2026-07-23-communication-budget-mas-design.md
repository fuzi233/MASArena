# Communication Budget MAS Design

## Goal

Provide a small, configurable MASArena system for BBEH evaluation in which a
dynamic team of solvers can use bounded communication to resolve uncertainty.
Solvers collaborate before a single aggregator begins final synthesis. Every
solver and the aggregator owns an independent question budget `K`.

The protocol deliberately distinguishes a paid question from a free reply:
questions spend the sender's `K`; replies never spend `K` or a solver step.

## Scope

- Register `communication_budget` as a MASArena `AgentSystem`.
- Register `bbeh` as a MASArena benchmark and evaluate it with the canonical
  BBEH correctness function.
- Expose solver count, `K`, budget visibility, maximum solver steps, and model
  temperature through `MASArena/main.py`.
- Keep `final_answer` as the complete raw final response from the aggregator.

The root-level `commprobe` package remains the historical controlled-experiment
runner and analysis tool.

## Protocol

### Solver phase: synchronous supersteps

All active solvers execute one active step concurrently in each superstep. A
solver sees only its own accumulated working record and all messages delivered
before that superstep began. It cannot observe another solver's same-step
action.

An active step consumes one unit of `--max-turns` and is exactly one of:

- `continue_reasoning`: append a concise `reasoning_note` to the solver's
  working record.
- `ask`: append a concise `reasoning_note`, send one question to any other
  solver, and spend the asker's one `K` unit.
- `submit`: append a concise `reasoning_note`, candidate answer, and mark the
  solver submitted.

After all active-step calls finish, the system records `continue_reasoning`
and `submit` actions and delivers all questions. It then enters a communication
settlement barrier. Each recipient processes its own inbound questions in FIFO
order, ordered by `(superstep, question_sequence)`. A reply is an extra action:
it does not consume `K` or `max-turns`, but it is appended to the sender's and
recipient's records. Different recipients may process their separate FIFO
queues concurrently. No question may be created during settlement.

The next synchronous superstep begins only after every reply from the current
settlement barrier has completed. This makes all message visibility
deterministic even when solver model calls are concurrent.

If a solver reaches `--max-turns` without voluntarily submitting, the system
creates an automatic submission. Submitted solvers no longer take active
steps, but remain available to answer free questions from other solvers and
from the aggregator.

Solver prompts encourage divergent, evidence-based exploration. They should
ask about their least certain assumptions, derivations, counterexamples, or
facts. They should not normally share their proposed final answer during solver
communication; messages should focus on resolving uncertainty instead.

### Aggregator phase: delayed synthesis and bounded review

The aggregator is not invoked during solver supersteps. It starts only after a
`SolverReport` exists for every solver. It receives every report, including
candidate answer, concise reasoning notes, and the full visible message
history.

The aggregator first reasons over those reports. When a material uncertainty
remains, it may ask any solver one focused question. Each aggregator question
spends one unit of the aggregator's own `K`, and it must wait for that solver's
free reply before issuing its next question or synthesizing the final result.
The aggregator has no independent solver-step limit; its number of questions is
bounded solely by `K`.

Its prompt directs it to choose one of three reasoning strategies internally:

1. Select the most credible existing candidate answer.
2. Repair one otherwise strong report using localized evidence.
3. Combine independent reasoning from multiple reports to derive a new answer.

The final response must be exactly one XML element:

```xml
<final_answer>the exact answer required by the task</final_answer>
```

It contains no reasoning, label, Markdown, or other surrounding text.

## Component Boundaries

`CommunicationBudgetMAS` is the MASArena adapter. It validates config,
constructs a per-problem `TeamOrchestrator`, and maps the result to MASArena's
`messages` and `final_answer` contract.

`TeamOrchestrator` owns phase sequencing: solver supersteps, settlement
barriers, report assembly, aggregator review, and final synthesis.
`CommunicationState` owns per-problem state, independent budgets, solver
statuses, working records, inboxes, monotonic event sequencing, and snapshots.
Roles never mutate this state directly.

`PromptComposer` is the sole prompt construction point. It owns distinct
templates for solver active steps, solver replies, aggregator review questions,
and final aggregation. Hidden budget mode must omit budget figures and budget
words from prompts while enforcing the same state-machine rules.

## Data Contracts

### Protocol events

Each event has a monotonic `sequence_id`, `superstep`, `kind`, `sender_id`,
`recipient_id` when applicable, `content`, and `charged_budget`.

Event kinds are:

- `continue_reasoning`
- `solver_question`
- `solver_reply`
- `solver_submission`
- `aggregator_question`
- `aggregator_reply`

### Solver report

```python
{
    "solver_id": str,
    "candidate_answer": str,
    "reasoning_notes": list[str],
    "visible_message_history": list[dict[str, object]],
    "submitted_by": "voluntary" | "step_exhausted",
}
```

### Agent result

`CommunicationBudgetMAS.run_agent()` returns at least:

```python
{
    "final_answer": str,
    "messages": list[dict[str, object]],
    "team_state": dict[str, object],
}
```

`final_answer` always contains the original complete aggregator response. The
BBEH evaluator independently extracts the XML content into `extracted_answer`
for scoring and auditability.

## CLI

```text
--solver-count N                 N >= 1; default 2
--communication-budget K         K >= 0; default 0
--budget-visibility visible|hidden
--max-turns N                    N >= 1; default 4
--temperature T                  default 0.7
```

Visible mode gives an acting questioner its current allowance. Hidden mode
enforces the same allowance without revealing allowance values or terminology.

## Error Handling and Testing

Invalid numeric CLI values are rejected by argparse. Invalid action payloads,
unknown actors, self-questions, and exhausted budgets produce explicit protocol
errors. A malformed
optional model action fails closed into `continue_reasoning` with a recorded
diagnostic rather than corrupting the final synthesis path.

Tests cover independent per-problem state under concurrent benchmark execution,
superstep visibility barriers, FIFO replies, free replies, per-sender K debit,
automatic submissions, delayed aggregator invocation, aggregator wait behavior,
prompt budget visibility, temperature forwarding, BBEH scoring, and the
required `final_answer` field.
