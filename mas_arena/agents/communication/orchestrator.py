"""Synchronous solver supersteps followed by bounded aggregator review."""

import asyncio
import json
import re
from typing import Protocol

from .actions import AggregatorAction, LegacyAggregatorAction, SolverAction
from .config import TeamConfig
from .prompts import PromptComposer
from .state import CommunicationState


class TextModelClient(Protocol):
    async def complete_text(self, messages: list[dict[str, str]]) -> str: ...


class TeamOrchestrator:
    """Run one isolated team state for a single benchmark problem."""

    def __init__(self, config: TeamConfig, client: TextModelClient) -> None:
        self.config = config
        self.client = client

    async def solve(self, problem: str) -> dict[str, object]:
        state = CommunicationState.create(
            solver_count=self.config.solver_count,
            solver_budget=self.config.solver_communication_budget,
            aggregator_budget=self.config.resolved_aggregator_communication_budget,
            max_turns=self.config.max_turns,
            aggregator_max_steps=self.config.resolved_aggregator_max_steps,
        )
        await self._run_solver_phase(problem, state)
        reports = [state.solver_report(solver_id) for solver_id in state.solver_ids]
        if self.config.protocol_version == "legacy-v1":
            await self._run_legacy_aggregator_review(problem, reports, state)
            state.record_forced_final("legacy_final_aggregation")
            final_answer, decision = await self._complete_final_aggregation(
                PromptComposer.legacy_final_aggregator_prompt(problem=problem, solver_reports=reports, state=state), state
            )
            state.record_aggregation_decision(decision)
        else:
            final_answer = await self._run_aggregator_review(problem, reports, state)
            if final_answer is None:
                state.record_forced_final("bounded_review_exhausted_or_invalid")
                final_answer, decision = await self._complete_final_aggregation(
                    PromptComposer.final_aggregator_prompt(problem=problem, solver_reports=reports, state=state), state
                )
                state.record_aggregation_decision(decision)
        if not final_answer.strip():
            raise ValueError("aggregator returned an empty final_answer")
        messages = [
            {"agent_id": str(event["sender_id"]), "name": str(event["sender_id"]), "role": "assistant", "content": json.dumps(event, ensure_ascii=False)}
            for event in state.events
        ]
        drain_usage = getattr(self.client, "drain_usage", None)
        if callable(drain_usage):
            messages.extend(
                {
                    "agent_id": f"model_call_{index}",
                    "name": "model_call",
                    "role": "assistant",
                    "content": "",
                    "usage_metadata": usage,
                }
                for index, usage in enumerate(drain_usage(), start=1)
            )
        messages.append({"agent_id": "aggregator", "name": "aggregator", "role": "assistant", "content": final_answer})
        return {"final_answer": final_answer, "messages": messages, "team_state": state.snapshot()}

    async def _complete_final_aggregation(
        self, prompt: list[dict[str, str]], state: CommunicationState
    ) -> tuple[str, dict[str, object]]:
        """Request a forced final answer and retry invalid decision provenance once."""
        final_output = await self.client.complete_text(prompt)
        decision = self._parse_aggregation_decision(final_output, state)
        if decision["mode"] != "unparsed":
            return final_output, decision

        repair_output = await self.client.complete_text(
            PromptComposer.final_aggregation_repair_prompt(invalid_output=final_output, state=state)
        )
        repaired_decision = self._parse_aggregation_decision(repair_output, state)
        repaired = repaired_decision["mode"] != "unparsed"
        state.record_aggregator_final_decision_repair(
            invalid_output=final_output,
            validation_error=str(decision["rationale"]),
            repair_output=repair_output,
            repaired=repaired,
            repair_error=None if repaired else str(repaired_decision["rationale"]),
        )
        return (repair_output, repaired_decision) if repaired else (final_output, decision)

    @staticmethod
    def _parse_aggregation_decision(final_output: str, state: CommunicationState) -> dict[str, object]:
        """Parse declared final-decision metadata without risking answer evaluation."""
        match = re.search(r"<aggregation_decision>\s*(.*?)\s*</aggregation_decision>", final_output, flags=re.DOTALL)
        if match is None:
            return {"mode": "unparsed", "source_solver_ids": [], "selected_candidate_solver_id": None, "rationale": "No aggregation_decision tag returned."}
        try:
            decision = json.loads(match.group(1))
            mode = decision["mode"]
            sources = decision["source_solver_ids"]
            selected = decision.get("selected_candidate_solver_id")
            rationale = decision["rationale"]
            if mode not in {"select_candidate", "single_source_repair", "rederive_from_trace"}:
                raise ValueError("unknown mode")
            if (
                not isinstance(sources, list)
                or not sources
                or any(not isinstance(source, str) or source not in state.solver_ids for source in sources)
                or len(set(sources)) != len(sources)
            ):
                raise ValueError("invalid source_solver_ids")
            if mode in {"select_candidate", "single_source_repair"} and selected not in sources:
                raise ValueError("candidate selection or repair requires selected_candidate_solver_id in source_solver_ids")
            if mode == "single_source_repair" and len(sources) != 1:
                raise ValueError("single_source_repair requires one source")
            if mode == "rederive_from_trace" and selected is not None:
                raise ValueError("rederive_from_trace must not select a candidate solver")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("missing rationale")
            return {
                "mode": mode,
                "source_solver_ids": sources,
                "selected_candidate_solver_id": selected,
                "rationale": rationale,
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return {"mode": "unparsed", "source_solver_ids": [], "selected_candidate_solver_id": None, "rationale": "Invalid aggregation_decision payload.", "raw_decision": match.group(1)}

    async def _run_solver_phase(self, problem: str, state: CommunicationState) -> None:
        while state.active_solver_ids():
            actor_ids = state.active_solver_ids()
            actions = await asyncio.gather(
                *(
                    self._request_json_action(
                        (
                            PromptComposer.legacy_solver_step_prompt
                            if self.config.protocol_version == "legacy-v1"
                            else PromptComposer.solver_step_prompt
                        )(
                            actor_id=solver_id,
                            problem=problem,
                            state=state,
                            visibility=self.config.budget_visibility,
                        ),
                        "solver",
                        state=state,
                        actor_id=solver_id,
                    )
                    for solver_id in actor_ids
                )
            )
            questions: list[dict[str, object]] = []
            for solver_id, action in zip(actor_ids, actions, strict=True):
                try:
                    if action is None:
                        raise ValueError("invalid solver action after retry")
                    assert isinstance(action, SolverAction)
                    if action.action == "think":
                        state.record_continue(solver_id, action.reasoning_note)
                    elif action.action == "ask":
                        questions.append(state.record_solver_question(solver_id, action.recipient_id or "", action.question or "", action.reasoning_note))
                    else:
                        state.record_submission(solver_id, action.reasoning_note, action.candidate_answer or "")
                except (ValueError, json.JSONDecodeError):
                    state.record_continue(solver_id, "Malformed action; continued private reasoning.")
            await self._settle_solver_questions(problem, state, questions)
            for solver_id in state.active_solver_ids():
                if state.active_steps[solver_id] >= state.max_turns:
                    state.auto_submit_exhausted(solver_id)
            if state.active_solver_ids():
                state.advance_superstep()

    async def _settle_solver_questions(self, problem: str, state: CommunicationState, questions: list[dict[str, object]]) -> None:
        by_recipient: dict[str, list[dict[str, object]]] = {}
        for question in sorted(questions, key=lambda event: int(event["sequence_id"])):
            by_recipient.setdefault(str(question["recipient_id"]), []).append(question)

        async def reply_to_queue(recipient_id: str, queue: list[dict[str, object]]) -> None:
            for question in queue:
                reply = await self.client.complete_text(
                    PromptComposer.solver_reply_prompt(actor_id=recipient_id, problem=problem, question=question, state=state)
                )
                state.record_solver_reply(recipient_id, int(question["sequence_id"]), reply)

        await asyncio.gather(*(reply_to_queue(recipient_id, queue) for recipient_id, queue in by_recipient.items()))

    async def _run_aggregator_review(
        self, problem: str, reports: list[dict[str, object]], state: CommunicationState
    ) -> str | None:
        while state.aggregator_active():
            action = await self._request_json_action(
                PromptComposer.aggregator_step_prompt(
                    problem=problem,
                    solver_reports=reports,
                    state=state,
                    visibility=self.config.budget_visibility,
                ),
                "aggregator",
                repair_solver_ids=state.solver_ids,
                state=state,
                actor_id="aggregator",
            )
            if action is None:
                return None
            assert isinstance(action, AggregatorAction)
            if action.action == "think":
                state.record_aggregator_think(action.reasoning_note)
                continue
            if action.action == "submit":
                try:
                    decision = self._decision_from_aggregator_action(action, state)
                except ValueError as exc:
                    repaired_action, repaired_decision = await self._repair_aggregator_provenance(action, state, str(exc))
                    if repaired_action is None or repaired_decision is None:
                        state.record_aggregator_think("Malformed final submission provenance; continued review.")
                        continue
                    action = repaired_action
                    decision = repaired_decision
                state.record_aggregator_submission(action.reasoning_note, action.final_answer or "", decision)
                state.record_aggregation_decision(decision)
                return action.final_answer
            try:
                question = state.record_aggregator_question(
                    action.recipient_id or "", action.question or "", action.reasoning_note
                )
            except ValueError as exc:
                state.record_aggregator_think(f"Rejected clarification request: {exc}.")
                continue
            reply = await self.client.complete_text(
                PromptComposer.solver_reply_prompt(
                    actor_id=action.recipient_id or "",
                    problem=problem,
                    question=question,
                    state=state,
                )
            )
            state.record_aggregator_reply(action.recipient_id or "", int(question["sequence_id"]), reply)
        return None

    async def _repair_aggregator_provenance(
        self, action: AggregatorAction, state: CommunicationState, validation_error: str
    ) -> tuple[AggregatorAction | None, dict[str, object] | None]:
        """Retry one semantically invalid bounded-review submission and retain both attempts."""
        invalid_action = action.model_dump(mode="json")
        repair_output = await self.client.complete_text(
            PromptComposer.json_repair_prompt(
                invalid_output=json.dumps(invalid_action, ensure_ascii=False),
                action_kind="aggregator",
                solver_ids=state.solver_ids,
            )
        )
        try:
            repaired_action = AggregatorAction.model_validate(json.loads(repair_output))
            repaired_decision = self._decision_from_aggregator_action(repaired_action, state)
        except (ValueError, json.JSONDecodeError) as exc:
            state.record_aggregator_provenance_repair(
                invalid_action=invalid_action,
                validation_error=validation_error,
                repair_output=repair_output,
                repaired=False,
                repair_error=str(exc),
            )
            return None, None
        state.record_aggregator_provenance_repair(
            invalid_action=invalid_action,
            validation_error=validation_error,
            repair_output=repair_output,
            repaired=True,
        )
        return repaired_action, repaired_decision

    @staticmethod
    def _decision_from_aggregator_action(action: AggregatorAction, state: CommunicationState) -> dict[str, object]:
        """Validate and preserve provenance declared by a bounded-review submission."""
        sources = action.source_solver_ids or []
        selected = action.selected_candidate_solver_id
        if any(source not in state.solver_ids for source in sources):
            raise ValueError("submission named an unknown solver source")
        if action.decision_mode in {"select_candidate", "single_source_repair"} and selected not in sources:
            raise ValueError("selected solver must occur in source_solver_ids")
        if action.decision_mode == "single_source_repair" and len(sources) != 1:
            raise ValueError("single_source_repair must name exactly one source")
        if action.decision_mode == "select_candidate":
            selected_report = state.solver_report(selected or "")
            if action.final_answer != selected_report["candidate_answer"]:
                raise ValueError("select_candidate final_answer must exactly match the selected candidate")
        return {
            "mode": action.decision_mode,
            "source_solver_ids": sources,
            "selected_candidate_solver_id": selected,
            "rationale": action.reasoning_note,
        }

    async def _run_legacy_aggregator_review(
        self, problem: str, reports: list[dict[str, object]], state: CommunicationState
    ) -> None:
        while state.can_send("aggregator"):
            action = await self._request_json_action(
                PromptComposer.legacy_aggregator_action_prompt(
                    problem=problem,
                    solver_reports=reports,
                    state=state,
                    visibility=self.config.budget_visibility,
                ),
                "legacy_aggregator",
                state=state,
                actor_id="aggregator",
            )
            if action is None or not isinstance(action, LegacyAggregatorAction) or action.action == "finalize":
                return
            try:
                question = state.record_legacy_aggregator_question(action.recipient_id or "", action.question or "")
            except ValueError:
                return
            reply = await self.client.complete_text(
                PromptComposer.solver_reply_prompt(
                    actor_id=action.recipient_id or "",
                    problem=problem,
                    question=question,
                    state=state,
                )
            )
            state.record_aggregator_reply(action.recipient_id or "", int(question["sequence_id"]), reply)

    async def _request_json_action(
        self,
        messages: list[dict[str, str]],
        action_kind: str,
        repair_solver_ids: tuple[str, ...] = (),
        state: CommunicationState | None = None,
        actor_id: str | None = None,
    ) -> SolverAction | AggregatorAction | LegacyAggregatorAction | None:
        raw = await self.client.complete_text(messages)
        action_type = (
            SolverAction if action_kind == "solver" else LegacyAggregatorAction if action_kind == "legacy_aggregator" else AggregatorAction
        )
        try:
            return action_type.model_validate(json.loads(raw))
        except (ValueError, json.JSONDecodeError):
            if state is not None and actor_id is not None:
                state.record_json_repair(actor_id, action_kind)
            retry_raw = await self.client.complete_text(
                PromptComposer.json_repair_prompt(
                    invalid_output=raw, action_kind=action_kind, solver_ids=repair_solver_ids
                )  # type: ignore[arg-type]
            )
            try:
                return action_type.model_validate(json.loads(retry_raw))
            except (ValueError, json.JSONDecodeError):
                return None
