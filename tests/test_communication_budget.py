"""Behavioral tests for the communication-budget MAS integration."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import os

import pytest

from mas_arena.agents import AVAILABLE_AGENT_SYSTEMS
from mas_arena.agents.communication.config import TeamConfig
from mas_arena.agents.communication.actions import AggregatorAction
from mas_arena.agents.communication.prompts import PromptComposer
from mas_arena.agents.communication.state import CommunicationState
from mas_arena.agents.communication_budget import CommunicationBudgetMAS
from mas_arena.agents.step_single_agent import StepSingleAgent
from mas_arena.evaluators import BENCHMARKS
from mas_arena.evaluators.bbeh_evaluator import BBEHEvaluator
from mas_arena.evaluators.math_evaluator import MathEvaluator
from mas_arena.benchmark_runner import BenchmarkRunner


def test_each_member_owns_an_independent_communication_budget() -> None:
    state = CommunicationState.create(solver_count=2, budget=1)

    state.record_solver_question("solver_1", "solver_2", "Check the evidence", "I am uncertain")

    assert state.remaining("solver_1") == 0
    assert state.remaining("solver_2") == 1
    assert state.remaining("aggregator") == 1


def test_zero_budget_disables_all_communication() -> None:
    state = CommunicationState.create(solver_count=2, budget=0)

    assert state.can_send("solver_1") is False
    assert state.can_send("aggregator") is False
    with pytest.raises(ValueError, match="communication budget exhausted"):
        state.record_solver_question("solver_1", "solver_2", "Check the evidence", "I am uncertain")


def test_solver_question_debits_only_asker_and_reply_is_free() -> None:
    state = CommunicationState.create(solver_count=2, budget=1, max_turns=2)

    question = state.record_solver_question(
        "solver_1", "solver_2", "Check this assumption", "I am uncertain about the assumption"
    )
    reply = state.record_solver_reply("solver_2", question["sequence_id"], "The assumption fails")

    assert state.remaining("solver_1") == 0
    assert state.remaining("solver_2") == 1
    assert question["charged_budget"] == 1
    assert reply["charged_budget"] == 0


def test_exhausted_solver_generates_report_with_reasoning_notes() -> None:
    state = CommunicationState.create(solver_count=1, budget=0, max_turns=1)

    state.record_continue("solver_1", "I need more evidence")
    report = state.auto_submit_exhausted("solver_1")

    assert report["submitted_by"] == "step_exhausted"
    assert report["reasoning_notes"] == ["I need more evidence"]
    assert report["candidate_answer"] == ""


def test_rejected_question_does_not_consume_a_solver_step() -> None:
    state = CommunicationState.create(solver_count=2, budget=0, max_turns=1)

    with pytest.raises(ValueError, match="communication budget exhausted"):
        state.record_solver_question("solver_1", "solver_2", "Check this", "I am uncertain")

    assert state.active_steps["solver_1"] == 0
    state.record_continue("solver_1", "I continued privately")
    assert state.active_steps["solver_1"] == 1


def test_visible_action_prompt_reports_current_remaining_budget() -> None:
    state = CommunicationState.create(solver_count=1, budget=2)
    prompt = PromptComposer.solver_step_prompt(
        actor_id="solver_1",
        problem="Question",
        state=state,
        visibility="visible",
    )

    assert "Remaining question allowance: 2" in prompt[0]["content"]


def test_hidden_action_prompt_does_not_leak_budget_information() -> None:
    state = CommunicationState.create(solver_count=1, budget=2)
    prompt = PromptComposer.solver_step_prompt(
        actor_id="solver_1",
        problem="Question",
        state=state,
        visibility="hidden",
    )

    content = "\n".join(message["content"] for message in prompt).lower()
    assert "remaining" not in content
    assert "budget" not in content


def test_solver_step_prompt_requires_focused_nonrepeating_questions() -> None:
    state = CommunicationState.create(solver_count=2, budget=1, max_turns=2)

    prompt = PromptComposer.solver_step_prompt(
        actor_id="solver_1", problem="Question", state=state, visibility="visible"
    )

    assert "least certain" in prompt[0]["content"]
    assert "Do not repeat a question" in prompt[0]["content"]
    assert "Remaining question allowance: 1" in prompt[0]["content"]
    assert "Remaining steps: 2" in prompt[0]["content"]
    assert "start with `{` and end with `}`" in prompt[0]["content"]
    assert '"action":"think"' in prompt[0]["content"]
    assert "```" in prompt[0]["content"]


def test_final_aggregator_prompt_requires_a_machine_readable_decision() -> None:
    state = CommunicationState.create(solver_count=1, budget=0, max_turns=1)
    state.record_continue("solver_1", "Check a boundary")
    report = state.auto_submit_exhausted("solver_1")

    prompt = PromptComposer.final_aggregator_prompt(
        problem="Question", solver_reports=[report], state=state
    )

    content = prompt[0]["content"]
    assert "aggregation_decision" in content
    assert "select_candidate" in content
    assert "rederive_from_trace" in content


def test_json_repair_prompt_uses_valid_action_examples() -> None:
    prompt = PromptComposer.json_repair_prompt(invalid_output="not json", action_kind="solver")

    content = prompt[0]["content"]
    assert '"action":"continue_reasoning|ask|submit"' not in content
    assert '"action":"submit"' in content
    assert "start with `{` and end with `}`" in content


def test_team_config_uses_separate_role_budgets_and_inherits_aggregator_steps() -> None:
    config = TeamConfig.from_mapping({"max_turns": 4})

    assert config.solver_communication_budget == 1
    assert config.resolved_aggregator_communication_budget == 1
    assert config.resolved_aggregator_max_steps == 4


def test_team_config_treats_explicit_none_aggregator_budget_as_the_solver_default() -> None:
    config = TeamConfig.from_mapping(
        {"solver_communication_budget": 1, "aggregator_communication_budget": None, "max_turns": 100}
    )

    assert config.resolved_aggregator_communication_budget == 1
    assert config.resolved_aggregator_max_steps == 100


def test_aggregator_action_uses_the_same_think_ask_submit_contract() -> None:
    assert AggregatorAction(action="think", reasoning_note="I need to compare the reports.").action == "think"
    with pytest.raises(ValueError, match="final_answer"):
        AggregatorAction(action="submit", reasoning_note="The evidence is decisive.")


def test_role_budgets_are_separate_and_duplicate_questions_do_not_charge() -> None:
    state = CommunicationState.create(
        solver_count=2,
        solver_budget=2,
        aggregator_budget=1,
        max_turns=3,
        aggregator_max_steps=3,
    )

    state.record_solver_question("solver_1", "solver_2", "Check premise P.", "Premise P is uncertain.")
    assert state.remaining("solver_1") == 1
    assert state.remaining("aggregator") == 1

    with pytest.raises(ValueError, match="duplicate question"):
        state.record_solver_question("solver_1", "solver_2", "  check   premise p. ", "I still need confirmation.")

    assert state.remaining("solver_1") == 1


def test_role_prompts_use_runtime_recipients_without_fixed_solver_examples() -> None:
    state = CommunicationState.create(
        solver_count=3,
        solver_budget=1,
        aggregator_budget=1,
        max_turns=2,
        aggregator_max_steps=2,
    )

    solver_prompt = PromptComposer.solver_step_prompt(
        actor_id="solver_1", problem="Question", state=state, visibility="visible"
    )[0]["content"]
    aggregator_prompt = PromptComposer.aggregator_step_prompt(
        problem="Question", solver_reports=[], state=state, visibility="visible"
    )[0]["content"]

    assert '"recipient_id":"solver_1"' not in aggregator_prompt
    assert '"recipient_id":"solver_2"' not in solver_prompt
    assert "eligible_recipient_ids" in solver_prompt
    assert "eligible_recipient_ids" in aggregator_prompt


@pytest.mark.asyncio
async def test_aggregator_thinks_asks_and_submits_a_final_answer() -> None:
    class ScriptedClient:
        def __init__(self) -> None:
            self.aggregator_actions = iter(
                [
                    '{"action":"think","reasoning_note":"I need to compare the report to the question."}',
                    '{"action":"ask","reasoning_note":"The derivation has one uncertain step.","recipient_id":"solver_1","question":"Which premise supports A?"}',
                    '{"action":"submit","reasoning_note":"The reply confirms the candidate.","final_answer":"A"}',
                ]
            )

        async def complete_text(self, messages: list[dict[str, str]]) -> str:
            system = messages[0]["content"]
            if "synchronous team step" in system:
                return '{"action":"submit","reasoning_note":"I derived A.","candidate_answer":"A"}'
            if "bounded sequential review step" in system:
                return next(self.aggregator_actions)
            if "Answer the received question" in system:
                return "Premise P supports A."
            raise AssertionError(f"Unexpected prompt: {system}")

    result = await CommunicationBudgetMAS(
        config={
            "solver_count": 1,
            "solver_communication_budget": 1,
            "aggregator_communication_budget": 1,
            "max_turns": 3,
            "model_client": ScriptedClient(),
        }
    ).run_agent({"id": "x", "problem": "Question", "solution": "A"})

    events = result["team_state"]["events"]
    assert [event["kind"] for event in events][-4:] == [
        "aggregator_think",
        "aggregator_question",
        "aggregator_reply",
        "aggregator_submission",
    ]
    assert events[-1]["content"]["reasoning_note"] == "The reply confirms the candidate."
    assert result["final_answer"] == "A"


@pytest.mark.asyncio
async def test_solver_questions_are_settled_after_the_full_superstep() -> None:
    class ScriptedClient:
        async def complete_text(self, messages: list[dict[str, str]]) -> str:
            system = messages[0]["content"]
            if "synchronous team step" in system and "solver_1" in system:
                return '{"action":"ask","reasoning_note":"I doubt a premise","recipient_id":"solver_2","question":"Is premise P valid?"}'
            if "synchronous team step" in system:
                return '{"action":"continue_reasoning","reasoning_note":"I checked an independent path"}'
            if "Answer the received question" in system:
                return "Premise P is invalid."
            if "All solver reports are complete" in system:
                return '{"action":"finalize"}'
            return "<final_answer>A</final_answer>"

    result = await CommunicationBudgetMAS(
        config={"solver_count": 2, "communication_budget": 1, "max_turns": 1, "model_client": ScriptedClient()}
    ).run_agent({"id": "x", "problem": "Question", "solution": "A"})

    kinds = [event["kind"] for event in result["team_state"]["events"]]
    assert kinds[:3] == ["solver_question", "continue_reasoning", "solver_reply"]


@pytest.mark.asyncio
async def test_aggregator_waits_for_free_reply_before_next_review_action() -> None:
    class ScriptedClient:
        def __init__(self) -> None:
            self.aggregator_actions = 0

        async def complete_text(self, messages: list[dict[str, str]]) -> str:
            system = messages[0]["content"]
            if "synchronous team step" in system:
                return '{"action":"submit","reasoning_note":"Solved it","candidate_answer":"A"}'
            if "Answer the received question" in system:
                return "The candidate is supported."
            if "bounded sequential review step" in system:
                self.aggregator_actions += 1
                return (
                    '{"action":"ask","reasoning_note":"I need confirmation.","recipient_id":"solver_1","question":"Confirm?"}'
                    if self.aggregator_actions == 1
                    else '{"action":"submit","reasoning_note":"The reply confirms A.","final_answer":"A"}'
                )
            return "<final_answer>A</final_answer>"

    result = await CommunicationBudgetMAS(
        config={"solver_count": 1, "communication_budget": 1, "max_turns": 1, "aggregator_max_steps": 2, "model_client": ScriptedClient()}
    ).run_agent({"id": "x", "problem": "Question", "solution": "A"})

    kinds = [event["kind"] for event in result["team_state"]["events"]]
    assert kinds[-3:] == ["aggregator_question", "aggregator_reply", "aggregator_submission"]


@pytest.mark.asyncio
async def test_final_aggregator_records_its_decision_and_solver_sources() -> None:
    class ScriptedClient:
        async def complete_text(self, messages: list[dict[str, str]]) -> str:
            system = messages[0]["content"]
            if "synchronous team step" in system:
                return '{"action":"submit","reasoning_note":"Solved","candidate_answer":"A"}'
            if "bounded sequential review step" in system:
                return '{"action":"think","reasoning_note":"I need one more review step."}'
            return (
                '<aggregation_decision>{"mode":"rederive_from_trace","source_solver_ids":["solver_1"],'
                '"selected_candidate_solver_id":null,"rationale":"I checked the submitted derivation."}'
                '</aggregation_decision><final_answer>A</final_answer>'
            )

    result = await CommunicationBudgetMAS(
        config={"solver_count": 1, "communication_budget": 0, "max_turns": 1, "model_client": ScriptedClient()}
    ).run_agent({"id": "x", "problem": "Question", "solution": "A"})

    decision = result["team_state"]["aggregation_decision"]
    assert decision["mode"] == "rederive_from_trace"
    assert decision["source_solver_ids"] == ["solver_1"]
    assert decision["selected_candidate_solver_id"] is None


@pytest.mark.asyncio
async def test_runner_persists_communication_trace_for_each_problem(tmp_path: Path) -> None:
    class TraceAgent:
        name = "communication_budget"

        async def evaluate(self, problem: dict[str, str], **kwargs: object) -> dict[str, object]:
            return {
                "status": "success",
                "score": 1.0,
                "is_correct": True,
                "extracted_answer": "A",
                "execution_time_ms": 1,
                "llm_usage": {"total_tokens": 10},
                "team_state": {
                    "events": [{"kind": "solver_question"}],
                    "aggregation_decision": {"mode": "select_candidate", "source_solver_ids": ["solver_1"]},
                },
            }

    runner = BenchmarkRunner(results_dir=str(tmp_path))
    entry = await runner._process_one_problem(
        0,
        {"id": "item-1", "input": "Question", "target": "A"},
        TraceAgent(),
        {"normalization_keys": {"id": "id", "problem": "input", "solution": "target"}},
        verbose=False,
    )

    assert entry["communication_trace"]["events"] == [{"kind": "solver_question"}]
    assert entry["aggregation_decision"]["mode"] == "select_candidate"


@pytest.mark.asyncio
async def test_invalid_solver_action_is_reprompted_once_before_fallback() -> None:
    class RetryClient:
        def __init__(self) -> None:
            self.retries = 0

        async def complete_text(self, messages: list[dict[str, str]]) -> str:
            system = messages[0]["content"]
            if "Re-output only valid JSON" in system:
                self.retries += 1
                return '{"action":"submit","reasoning_note":"Reformatted action","candidate_answer":"A"}'
            if "synchronous team step" in system:
                return "I choose to submit: A"
            if "bounded sequential review step" in system:
                return '{"action":"submit","reasoning_note":"The candidate is valid.","final_answer":"A"}'
            return "<final_answer>A</final_answer>"

    client = RetryClient()
    result = await CommunicationBudgetMAS(
        config={"solver_count": 1, "communication_budget": 1, "max_turns": 1, "model_client": client}
    ).run_agent({"id": "x", "problem": "Question", "solution": "A"})

    assert client.retries == 1
    assert result["team_state"]["reports"]["solver_1"]["submitted_by"] == "voluntary"


@pytest.mark.asyncio
async def test_invalid_aggregator_action_is_reprompted_once_before_finalization() -> None:
    class RetryClient:
        def __init__(self) -> None:
            self.retries = 0

        async def complete_text(self, messages: list[dict[str, str]]) -> str:
            system = messages[0]["content"]
            if "Re-output only valid JSON" in system:
                self.retries += 1
                return '{"action":"finalize"}'
            if "synchronous team step" in system:
                return '{"action":"submit","reasoning_note":"Solved","candidate_answer":"A"}'
            if "All solver reports are complete" in system:
                return "I will finalize now."
            return "<final_answer>A</final_answer>"

    client = RetryClient()
    await CommunicationBudgetMAS(
        config={"solver_count": 1, "communication_budget": 1, "max_turns": 1, "model_client": client}
    ).run_agent({"id": "x", "problem": "Question", "solution": "A"})

    assert client.retries == 1


@pytest.mark.asyncio
async def test_model_usage_is_preserved_before_the_final_answer() -> None:
    class UsageClient:
        async def complete_text(self, messages: list[dict[str, str]]) -> str:
            system = messages[0]["content"]
            if "synchronous team step" in system:
                return '{"action":"submit","reasoning_note":"Solved","candidate_answer":"A"}'
            if "All solver reports are complete" in system:
                return '{"action":"finalize"}'
            return "<final_answer>A</final_answer>"

        def drain_usage(self) -> list[dict[str, object]]:
            return [{"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "input_token_details": {}, "output_token_details": {}}]

    result = await CommunicationBudgetMAS(
        config={"solver_count": 1, "communication_budget": 0, "max_turns": 1, "model_client": UsageClient()}
    ).run_agent({"id": "x", "problem": "Question", "solution": "A"})

    assert result["messages"][-1]["content"] == "<final_answer>A</final_answer>"
    assert result["messages"][-2]["usage_metadata"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_communication_agent_closes_its_model_client() -> None:
    class ClosableClient:
        def __init__(self) -> None:
            self.closed = False

        async def complete_text(self, messages: list[dict[str, str]]) -> str:
            return ""

        async def aclose(self) -> None:
            self.closed = True

    client = ClosableClient()
    agent = CommunicationBudgetMAS(config={"model_client": client})

    await agent.aclose()

    assert client.closed is True


@pytest.mark.asyncio
async def test_agent_keeps_aggregator_response_in_final_answer() -> None:
    class StubClient:
        async def complete_text(self, messages: list[dict[str, str]]) -> str:
            system = messages[0]["content"]
            if "final aggregator" in system.lower():
                return "aggregator final"
            return "solver draft"

    agent = CommunicationBudgetMAS(
        config={
            "solver_count": 3,
            "communication_budget": 0,
            "model_client": StubClient(),
        }
    )

    result = await agent.run_agent({"id": "x", "problem": "Question", "solution": "answer"})

    assert result["final_answer"] == "aggregator final"
    assert result["team_state"]["solver_count"] == 3
    assert all(event["charged_budget"] == 0 for event in result["team_state"]["events"])


def test_final_aggregator_prompt_requires_a_single_final_answer_tag() -> None:
    state = CommunicationState.create(solver_count=1, budget=0, max_turns=1)
    state.record_continue("solver_1", "draft")
    prompt = PromptComposer.final_aggregator_prompt(
        problem="Question",
        solver_reports=[state.auto_submit_exhausted("solver_1")],
        state=state,
    )

    assert "<final_answer>" in prompt[0]["content"]
    assert "</final_answer>" in prompt[0]["content"]
    assert "only the answer value" in prompt[0]["content"]
    assert "exactly `Yes` or `No`" in prompt[0]["content"]
    assert "Position" in prompt[0]["content"]


def test_bbeh_evaluator_scores_final_answer() -> None:
    evaluator = BBEHEvaluator(name="bbeh", config={"data_path": "unused", "log_path": "test_logs"})

    result = evaluator.evaluate(
        {"id": "case", "problem": "Question", "solution": "A"},
        {"final_answer": "The answer is: A"},
    )

    assert result["score"] == 1.0
    assert result["final_answer"] == "The answer is: A"


def test_bbeh_evaluator_extracts_the_explicit_final_answer_tag() -> None:
    evaluator = BBEHEvaluator(name="bbeh", config={"data_path": "unused", "log_path": "test_logs"})

    result = evaluator.evaluate(
        {"id": "case", "problem": "Question", "solution": "8"},
        {"final_answer": "reasoning that must not be scored <final_answer>8</final_answer>"},
    )

    assert result["final_answer"] == "reasoning that must not be scored <final_answer>8</final_answer>"
    assert result["extracted_answer"] == "8"
    assert result["score"] == 1.0


def test_math_evaluator_scores_the_explicit_final_answer_tag() -> None:
    evaluator = MathEvaluator(name="math", config={"log_path": "test_logs"})

    result = evaluator.evaluate(
        {"id": "case", "problem": "Question", "solution": "The answer is \\boxed{5120}."},
        {"messages": [{"content": "<final_answer>5120</final_answer>"}]},
    )

    assert result["extracted_answer"] == "5120"
    assert result["score"] == 1


def test_new_agent_and_benchmark_are_registered() -> None:
    assert "communication_budget" in AVAILABLE_AGENT_SYSTEMS
    assert "step_single_agent" in AVAILABLE_AGENT_SYSTEMS
    assert "bbeh" in BENCHMARKS
    assert Path(BENCHMARKS["bbeh"]["data_path"]).is_file()
    assert TeamConfig(solver_count=1).solver_count == 1


@pytest.mark.asyncio
async def test_step_single_agent_forces_a_final_submission_after_max_turns() -> None:
    class StubClient:
        async def complete_text(self, messages: list[dict[str, str]]) -> str:
            system = messages[0]["content"]
            if "forced final submission" in system:
                return "<final_answer>42</final_answer>"
            return '{"action":"continue_reasoning","reasoning_note":"I need another step"}'

        def drain_usage(self) -> list[object]:
            return []

    result = await StepSingleAgent(
        config={"max_turns": 2, "model_client": StubClient()}
    ).run_agent({"id": "x", "problem": "Question", "solution": "42"})

    assert result["final_answer"] == "<final_answer>42</final_answer>"
    assert len(result["reasoning_notes"]) == 2


def test_cli_help_exposes_team_configuration() -> None:
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--solver-count" in result.stdout
    assert "--communication-budget" in result.stdout
    assert "--solver-communication-budget" in result.stdout
    assert "--aggregator-communication-budget" in result.stdout
    assert "--aggregator-max-steps" in result.stdout
    assert "--budget-visibility" in result.stdout
    assert "--max-turns" in result.stdout
    assert "--temperature" in result.stdout


def test_team_config_defaults_to_standard_temperature() -> None:
    assert TeamConfig.from_mapping({}).temperature == 1.0
    assert TeamConfig.from_mapping({"temperature": 1.1}).temperature == 1.1
    with pytest.raises(ValueError, match="temperature"):
        TeamConfig(temperature=2.1)


def test_communication_result_label_contains_team_configuration(tmp_path: Path) -> None:
    runner = BenchmarkRunner(results_dir=str(tmp_path))

    label = runner._agent_label(
        "communication_budget",
        {"solver_count": 3, "communication_budget": 4, "budget_visibility": "hidden", "max_turns": 10, "temperature": 0.7},
    )

    assert label == "communication_budget_n3_sk4_ak4_hidden_t10_at10_temp0p7"


def test_benchmark_summary_reports_token_usage_for_all_problems(tmp_path: Path) -> None:
    runner = BenchmarkRunner(results_dir=str(tmp_path))
    runner.experiment_config = {}

    summary = runner._finalize_benchmark(
        [
            {"status": "success", "score": 1, "duration_ms": 1, "llm_usage": {"total_tokens": 10}},
            {"status": "success", "score": 0, "duration_ms": 1, "llm_usage": {"total_tokens": 15}},
            {"status": "error", "score": 0, "duration_ms": 1, "llm_usage": {"total_tokens": 7}},
        ],
        "bbeh",
        "step_single_agent",
        tmp_path / "results.json",
        verbose=False,
    )

    assert summary["total_tokens_all_problems"] == 32
    assert summary["avg_tokens_per_problem"] == pytest.approx(32 / 3)
    assert summary["total_tokens_completed_problems"] == 25
    assert summary["avg_tokens_per_completed_problem"] == pytest.approx(12.5)


def test_bbeh_evaluator_imports_without_pythonpath_override() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-c", "from mas_arena.evaluators.bbeh_evaluator import BBEHEvaluator"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_project_venv_imports_bbeh_without_editable_path_hooks() -> None:
    project_root = Path(__file__).resolve().parents[1]
    python = project_root / ".venv" / "bin" / "python"

    result = subprocess.run(
        [str(python), "-c", "import bbeh; print(bbeh.__file__)"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_agentops_is_not_initialized_without_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    calls: list[dict[str, str]] = []
    monkeypatch.delenv("AGENTOPS_API_KEY", raising=False)
    monkeypatch.setattr(main.agentops, "init", lambda **kwargs: calls.append(kwargs))

    main.initialize_agentops()

    assert calls == []
