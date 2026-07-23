# Synchronous Communication Budget MAS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace free-form messaging with synchronous solver supersteps, free FIFO replies, and delayed aggregator review.

**Architecture:** `CommunicationState` is the only per-problem protocol state machine. `TeamOrchestrator` executes concurrent solver actions followed by a reply barrier, builds reports, and then runs bounded aggregator review. `PromptComposer` owns all structured prompts; the adapter only forwards configuration.

**Tech Stack:** Python 3.11, asyncio, Pydantic, OpenAI SDK, pytest, MASArena.

---

### Task 1: Model the superstep protocol

**Files:**
- Modify: `mas_arena/agents/communication/config.py`
- Modify: `mas_arena/agents/communication/actions.py`
- Modify: `mas_arena/agents/communication/state.py`
- Test: `tests/test_communication_budget.py`

- [ ] **Step 1: Write failing state tests**

```python
def test_question_debits_only_the_asker_and_reply_is_free() -> None:
    state = CommunicationState.create(solver_count=2, budget=1, max_turns=2)
    question = state.record_solver_question("solver_1", "solver_2", "Check this")
    reply = state.record_solver_reply("solver_2", question["sequence_id"], "It fails")
    assert state.remaining("solver_1") == 0
    assert state.remaining("solver_2") == 1
    assert reply["charged_budget"] == 0

def test_exhausted_solver_has_a_complete_automatic_report() -> None:
    state = CommunicationState.create(solver_count=1, budget=0, max_turns=1)
    state.record_continue("solver_1", "Need more evidence")
    report = state.auto_submit_exhausted("solver_1")
    assert report["submitted_by"] == "step_exhausted"
    assert report["reasoning_notes"] == ["Need more evidence"]
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run --project . --with pytest --with pytest-asyncio python -m pytest tests/test_communication_budget.py -q`

Expected: FAIL because the new state methods do not exist.

- [ ] **Step 3: Implement the contracts**

Add `temperature: float = 0.7` to `TeamConfig`, with inclusive range validation `[0.0, 2.0]`. Define `SolverAction` with literals `continue_reasoning`, `ask`, and `submit`; every action requires non-empty `reasoning_note`, while `ask` additionally requires recipient and question and `submit` requires candidate answer. Store monotonic events containing `sequence_id`, `superstep`, `kind`, sender, recipient, content, and `charged_budget`. Add state methods `record_continue`, `record_solver_question`, `record_solver_reply`, `record_submission`, `auto_submit_exhausted`, `solver_report`, and `all_submitted`.

- [ ] **Step 4: Verify the state tests pass**

Run the Step 2 command. Expected: PASS.

### Task 2: Centralize protocol prompts

**Files:**
- Modify: `mas_arena/agents/communication/prompts.py`
- Test: `tests/test_communication_budget.py`

- [ ] **Step 1: Write failing prompt tests**

```python
def test_solver_step_prompt_encourages_uncertainty_queries() -> None:
    state = CommunicationState.create(solver_count=1, budget=1, max_turns=1)
    prompt = PromptComposer.solver_step_prompt(
        actor_id="solver_1", problem="P", state=state, visibility="visible"
    )
    assert "least certain" in prompt[0]["content"]
    assert "not normally share" in prompt[0]["content"]

def test_final_prompt_lists_all_synthesis_choices() -> None:
    state = CommunicationState.create(solver_count=1, budget=0, max_turns=1)
    prompt = PromptComposer.final_aggregator_prompt(
        problem="P", solver_reports=[state.auto_submit_exhausted("solver_1")], state=state
    )
    assert "Select the most credible" in prompt[0]["content"]
    assert "Repair one otherwise strong" in prompt[0]["content"]
    assert "Combine independent reasoning" in prompt[0]["content"]
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run --project . --with pytest --with pytest-asyncio python -m pytest tests/test_communication_budget.py -q`

Expected: FAIL because these prompt builders do not exist.

- [ ] **Step 3: Implement the prompt builders**

Add `solver_step_prompt`, `solver_reply_prompt`, and `aggregator_question_prompt`. Each accepts only the actor's permitted context. In hidden mode omit the words `budget` and `remaining` and all allowance values. Make final aggregation receive `solver_reports` plus review events, direct its internal choice among selection, single-source repair, and multi-source synthesis, and enforce XML-only final output.

- [ ] **Step 4: Verify the prompt tests pass**

Run the Step 2 command. Expected: PASS.

### Task 3: Implement solver barriers and aggregator review

**Files:**
- Modify: `mas_arena/agents/communication/orchestrator.py`
- Test: `tests/test_communication_budget.py`

- [ ] **Step 1: Write failing orchestration tests**

```python
@pytest.mark.asyncio
async def test_all_solver_actions_finish_before_reply_barrier() -> None:
    client = ScriptedClient([
        '{"action":"ask","reasoning_note":"uncertain","recipient_id":"solver_2","question":"Q?"}',
        '{"action":"continue_reasoning","reasoning_note":"independent"}',
        "R", '{"action":"finalize"}', "<final_answer>A</final_answer>",
    ])
    result = await TeamOrchestrator(
        TeamConfig(solver_count=2, communication_budget=1, max_turns=1), client
    ).solve("P")
    kinds = [event["kind"] for event in result["team_state"]["events"]]
    assert kinds[:3] == ["solver_question", "continue_reasoning", "solver_reply"]

@pytest.mark.asyncio
async def test_aggregator_starts_after_every_solver_submits() -> None:
    client = OrderedClient([
        '{"action":"continue_reasoning","reasoning_note":"one"}',
        '{"action":"continue_reasoning","reasoning_note":"two"}',
        '{"action":"finalize"}', "<final_answer>A</final_answer>",
    ])
    await TeamOrchestrator(TeamConfig(solver_count=2, max_turns=1), client).solve("P")
    assert client.first_aggregator_call > client.last_submission_event
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run --project . --with pytest --with pytest-asyncio python -m pytest tests/test_communication_budget.py -q`

Expected: FAIL because current execution uses initial drafts and free-form turns.

- [ ] **Step 3: Implement supersteps and settlement**

Remove independent-draft and generic communication loops. For every superstep, render all active solver prompts from the same pre-step snapshot and await their actions with `asyncio.gather`. Parse all results before state mutation, record actions in solver-id order, then deliver questions. Reply to each target's FIFO queue serially; process distinct recipient queues concurrently. Forbid asks during reply processing. After every barrier, auto-submit solvers that consumed their final step. Do not invoke the aggregator until `all_submitted` is true.

- [ ] **Step 4: Implement bounded aggregator review**

After reports are complete, ask the model for either an `ask` action or `finalize`. Each aggregator ask must debit only the aggregator, invoke exactly one free solver reply, record both events, and wait before the next review action. A finalization invokes the XML-only final aggregator prompt and returns the raw text as `final_answer`.

- [ ] **Step 5: Verify the orchestration tests pass**

Run the Step 2 command. Expected: PASS.

### Task 4: Forward temperature and run regression checks

**Files:**
- Modify: `mas_arena/agents/communication_budget.py`
- Modify: `main.py`
- Modify: `docs/quick_start/usage.md`
- Test: `tests/test_communication_budget.py`

- [ ] **Step 1: Write failing CLI/client tests**

```python
def test_cli_help_exposes_temperature() -> None:
    result = subprocess.run([sys.executable, "main.py", "--help"], capture_output=True, text=True)
    assert "--temperature" in result.stdout

@pytest.mark.asyncio
async def test_openai_client_forwards_temperature(monkeypatch) -> None:
    client = OpenAITextClient(model_name="x", api_key=None, base_url=None, temperature=0.7)
    captured = {}
    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    monkeypatch.setattr(client.client.chat.completions, "create", create)
    await client.complete_text([{"role": "user", "content": "P"}])
    assert captured["temperature"] == 0.7
```

- [ ] **Step 2: Verify the tests fail, then implement**

Run: `uv run --project . --with pytest --with pytest-asyncio python -m pytest tests/test_communication_budget.py -q`

Add `--temperature` with default `0.7` and CLI range validation; forward it into `TeamConfig` and OpenAI `chat.completions.create`. Update the usage example to explain paid questions, free replies, and synchronous steps.

- [ ] **Step 3: Run complete checks**

```bash
uv run --project . --with pytest --with pytest-asyncio python -m pytest tests/test_communication_budget.py -q
uv run python -c 'import bbeh; from mas_arena.agents.communication_budget import CommunicationBudgetMAS'
uv run python main.py --help > /dev/null
git diff --check
```

Expected: all commands exit with status 0.
