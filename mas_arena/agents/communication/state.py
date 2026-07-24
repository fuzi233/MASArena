"""Per-problem state for the synchronous communication protocol."""

from dataclasses import dataclass, field


@dataclass
class CommunicationState:
    """Own budgets, solver working records, and an ordered protocol event log."""

    budgets: dict[str, int]
    max_turns: int
    aggregator_max_steps: int
    active_steps: dict[str, int]
    statuses: dict[str, str]
    reasoning_notes: dict[str, list[str]]
    aggregator_steps: int = 0
    aggregator_reasoning_notes: list[str] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)
    reports: dict[str, dict[str, object]] = field(default_factory=dict)
    aggregation_decision: dict[str, object] | None = None
    superstep: int = 1
    _next_sequence_id: int = 1

    @classmethod
    def create(
        cls,
        *,
        solver_count: int,
        budget: int | None = None,
        solver_budget: int | None = None,
        aggregator_budget: int | None = None,
        max_turns: int = 4,
        aggregator_max_steps: int | None = None,
    ) -> "CommunicationState":
        if solver_count < 1:
            raise ValueError("solver_count must be at least 1")
        resolved_solver_budget = budget if solver_budget is None and budget is not None else (solver_budget if solver_budget is not None else 0)
        resolved_aggregator_budget = (
            resolved_solver_budget if aggregator_budget is None else aggregator_budget
        )
        if resolved_solver_budget < 0 or resolved_aggregator_budget < 0:
            raise ValueError("communication budgets must be non-negative")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        resolved_aggregator_steps = max_turns if aggregator_max_steps is None else aggregator_max_steps
        if resolved_aggregator_steps < 1:
            raise ValueError("aggregator_max_steps must be at least 1")
        solver_ids = tuple(f"solver_{index}" for index in range(1, solver_count + 1))
        actors = (*solver_ids, "aggregator")
        return cls(
            budgets={**{solver_id: resolved_solver_budget for solver_id in solver_ids}, "aggregator": resolved_aggregator_budget},
            max_turns=max_turns,
            aggregator_max_steps=resolved_aggregator_steps,
            active_steps={solver_id: 0 for solver_id in solver_ids},
            statuses={solver_id: "active" for solver_id in solver_ids},
            reasoning_notes={solver_id: [] for solver_id in solver_ids},
        )

    @property
    def solver_ids(self) -> tuple[str, ...]:
        return tuple(actor for actor in self.budgets if actor.startswith("solver_"))

    @property
    def transcript(self) -> list[dict[str, object]]:
        """Backward-compatible alias for the ordered event log."""
        return self.events

    def remaining(self, actor_id: str) -> int:
        self._require_actor(actor_id)
        return self.budgets[actor_id]

    def can_send(self, actor_id: str) -> bool:
        return self.remaining(actor_id) > 0

    def active_solver_ids(self) -> tuple[str, ...]:
        return tuple(solver_id for solver_id in self.solver_ids if self.statuses[solver_id] == "active")

    def aggregator_active(self) -> bool:
        return self.aggregator_steps < self.aggregator_max_steps

    def actor_step_count(self, actor_id: str) -> int:
        self._require_actor(actor_id)
        return self.aggregator_steps if actor_id == "aggregator" else self.active_steps[actor_id]

    def record_continue(self, solver_id: str, reasoning_note: str) -> dict[str, object]:
        self._consume_solver_step(solver_id, reasoning_note)
        return self._append_event("continue_reasoning", solver_id, None, {"reasoning_note": reasoning_note}, 0)

    def record_solver_question(self, solver_id: str, recipient_id: str, question: str, reasoning_note: str = "") -> dict[str, object]:
        self._validate_solver_step(solver_id, reasoning_note)
        self._require_solver(recipient_id)
        if solver_id == recipient_id:
            raise ValueError("solvers may not ask themselves")
        if not question.strip():
            raise ValueError("question must not be empty")
        if not self.can_send(solver_id):
            raise ValueError("communication budget exhausted")
        self._consume_solver_step(solver_id, reasoning_note)
        self.budgets[solver_id] -= 1
        return self._append_event(
            "solver_question",
            solver_id,
            recipient_id,
            {"question": question, "reasoning_note": reasoning_note},
            1,
        )

    def record_solver_reply(self, solver_id: str, question_sequence_id: int, reply: str) -> dict[str, object]:
        self._require_solver(solver_id)
        if not reply.strip():
            raise ValueError("reply must not be empty")
        question = self._find_pending_question(question_sequence_id, solver_id, "solver_question")
        return self._append_event(
            "solver_reply",
            solver_id,
            str(question["sender_id"]),
            {"reply": reply, "in_reply_to": question_sequence_id},
            0,
        )

    def record_submission(self, solver_id: str, reasoning_note: str, candidate_answer: str) -> dict[str, object]:
        self._consume_solver_step(solver_id, reasoning_note)
        if not candidate_answer.strip():
            raise ValueError("candidate_answer must not be empty")
        self.statuses[solver_id] = "submitted"
        self.reports[solver_id] = self._build_report(solver_id, candidate_answer, "voluntary")
        return self._append_event(
            "solver_submission",
            solver_id,
            None,
            {"reasoning_note": reasoning_note, "candidate_answer": candidate_answer, "submitted_by": "voluntary"},
            0,
        )

    def record_aggregator_think(self, reasoning_note: str) -> dict[str, object]:
        self._consume_aggregator_step(reasoning_note)
        return self._append_event("aggregator_think", "aggregator", None, {"reasoning_note": reasoning_note}, 0)

    def record_aggregator_question(
        self, recipient_id: str, question: str, reasoning_note: str = "Aggregator requested clarification."
    ) -> dict[str, object]:
        self._require_solver(recipient_id)
        if not question.strip():
            raise ValueError("question must not be empty")
        if not self.can_send("aggregator"):
            raise ValueError("communication budget exhausted")
        self._consume_aggregator_step(reasoning_note)
        self.budgets["aggregator"] -= 1
        return self._append_event(
            "aggregator_question", "aggregator", recipient_id, {"question": question, "reasoning_note": reasoning_note}, 1
        )

    def record_legacy_aggregator_question(self, recipient_id: str, question: str) -> dict[str, object]:
        """Record the pre-v2 aggregator question without step or duplicate guards."""
        self._require_solver(recipient_id)
        if not question.strip():
            raise ValueError("question must not be empty")
        if not self.can_send("aggregator"):
            raise ValueError("communication budget exhausted")
        self.budgets["aggregator"] -= 1
        return self._append_event("aggregator_question", "aggregator", recipient_id, {"question": question}, 1)

    def record_aggregator_submission(
        self, reasoning_note: str, final_answer: str, aggregation_decision: dict[str, object]
    ) -> dict[str, object]:
        self._consume_aggregator_step(reasoning_note)
        if not final_answer.strip():
            raise ValueError("final_answer must not be empty")
        return self._append_event(
            "aggregator_submission",
            "aggregator",
            None,
            {
                "reasoning_note": reasoning_note,
                "final_answer": final_answer,
                "aggregation_decision": aggregation_decision,
            },
            0,
        )

    def record_aggregator_provenance_repair(
        self,
        *,
        invalid_action: dict[str, object],
        validation_error: str,
        repair_output: str,
        repaired: bool,
        repair_error: str | None = None,
    ) -> dict[str, object]:
        """Preserve one bounded-review provenance repair attempt without charging a step."""
        content: dict[str, object] = {
            "invalid_action": invalid_action,
            "validation_error": validation_error,
            "repair_output": repair_output,
            "repaired": repaired,
        }
        if repair_error is not None:
            content["repair_error"] = repair_error
        return self._append_event("aggregator_provenance_repair", "aggregator", None, content, 0)

    def record_aggregator_final_decision_repair(
        self,
        *,
        invalid_output: str,
        validation_error: str,
        repair_output: str,
        repaired: bool,
        repair_error: str | None = None,
    ) -> dict[str, object]:
        """Preserve one forced-final decision repair attempt without charging a step."""
        content: dict[str, object] = {
            "invalid_output": invalid_output,
            "validation_error": validation_error,
            "repair_output": repair_output,
            "repaired": repaired,
        }
        if repair_error is not None:
            content["repair_error"] = repair_error
        return self._append_event("aggregator_final_decision_repair", "aggregator", None, content, 0)

    def record_json_repair(self, actor_id: str, action_kind: str) -> dict[str, object]:
        """Record a schema-repair request without changing protocol state."""
        self._require_actor(actor_id)
        return self._append_event("json_repair", actor_id, None, {"action_kind": action_kind}, 0)

    def record_forced_final(self, reason: str) -> dict[str, object]:
        """Record the final fallback call made outside a bounded review step."""
        return self._append_event("forced_final", "aggregator", None, {"reason": reason}, 0)

    def record_aggregator_reply(self, solver_id: str, question_sequence_id: int, reply: str) -> dict[str, object]:
        self._require_solver(solver_id)
        if not reply.strip():
            raise ValueError("reply must not be empty")
        question = self._find_pending_question(question_sequence_id, solver_id, "aggregator_question")
        return self._append_event(
            "aggregator_reply",
            solver_id,
            str(question["sender_id"]),
            {"reply": reply, "in_reply_to": question_sequence_id},
            0,
        )

    def auto_submit_exhausted(self, solver_id: str) -> dict[str, object]:
        self._require_solver(solver_id)
        if self.statuses[solver_id] == "submitted":
            return self.reports[solver_id]
        if self.active_steps[solver_id] < self.max_turns:
            raise ValueError("solver still has active steps available")
        self.statuses[solver_id] = "submitted"
        report = self._build_report(solver_id, "", "step_exhausted")
        self.reports[solver_id] = report
        self._append_event("solver_submission", solver_id, None, {"submitted_by": "step_exhausted"}, 0)
        return report

    def all_submitted(self) -> bool:
        return len(self.reports) == len(self.solver_ids)

    def solver_report(self, solver_id: str) -> dict[str, object]:
        self._require_solver(solver_id)
        return self.reports[solver_id]

    def record_aggregation_decision(self, decision: dict[str, object]) -> None:
        """Store the final aggregator's declared decision for experiment analysis."""
        self.aggregation_decision = dict(decision)

    def advance_superstep(self) -> None:
        self.superstep += 1

    def public_transcript(self, actor_id: str) -> list[dict[str, object]]:
        self._require_actor(actor_id)
        if actor_id == "aggregator":
            return list(self.events)
        return [
            event
            for event in self.events
            if actor_id in {event["sender_id"], event["recipient_id"]}
        ]

    def snapshot(self) -> dict[str, object]:
        return {
            "solver_count": len(self.solver_ids),
            "remaining_budget": dict(self.budgets),
            "communication_events": len(self.events),
            "events": list(self.events),
            "solver_statuses": dict(self.statuses),
            "solver_steps": dict(self.active_steps),
            "aggregator_steps": self.aggregator_steps,
            "aggregator_max_steps": self.aggregator_max_steps,
            "reports": dict(self.reports),
            "aggregation_decision": self.aggregation_decision,
        }

    def _consume_solver_step(self, solver_id: str, reasoning_note: str) -> None:
        self._validate_solver_step(solver_id, reasoning_note)
        self.active_steps[solver_id] += 1
        self.reasoning_notes[solver_id].append(reasoning_note)

    def _validate_solver_step(self, solver_id: str, reasoning_note: str) -> None:
        self._require_solver(solver_id)
        if self.statuses[solver_id] != "active":
            raise ValueError("submitted solvers cannot take active steps")
        if self.active_steps[solver_id] >= self.max_turns:
            raise ValueError("solver step limit exhausted")
        if not reasoning_note.strip():
            raise ValueError("reasoning_note must not be empty")

    def _consume_aggregator_step(self, reasoning_note: str) -> None:
        if not self.aggregator_active():
            raise ValueError("aggregator step limit exhausted")
        if not reasoning_note.strip():
            raise ValueError("reasoning_note must not be empty")
        self.aggregator_steps += 1
        self.aggregator_reasoning_notes.append(reasoning_note)

    def _build_report(self, solver_id: str, candidate_answer: str, submitted_by: str) -> dict[str, object]:
        return {
            "solver_id": solver_id,
            "candidate_answer": candidate_answer,
            "reasoning_notes": list(self.reasoning_notes[solver_id]),
            "visible_message_history": self.public_transcript(solver_id),
            "submitted_by": submitted_by,
        }

    def _append_event(
        self,
        kind: str,
        sender_id: str,
        recipient_id: str | None,
        content: dict[str, object],
        charged_budget: int,
    ) -> dict[str, object]:
        event = {
            "sequence_id": self._next_sequence_id,
            "superstep": self.superstep,
            "kind": kind,
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "content": content,
            "charged_budget": charged_budget,
        }
        self._next_sequence_id += 1
        self.events.append(event)
        return event

    def _find_pending_question(self, sequence_id: int, recipient_id: str, kind: str) -> dict[str, object]:
        for event in self.events:
            if event["sequence_id"] == sequence_id and event["kind"] == kind and event["recipient_id"] == recipient_id:
                if not any(
                    reply["content"].get("in_reply_to") == sequence_id
                    for reply in self.events
                    if reply["kind"] in {"solver_reply", "aggregator_reply"}
                ):
                    return event
        raise ValueError("unknown or already answered question")

    def _require_actor(self, actor_id: str) -> None:
        if actor_id not in self.budgets:
            raise ValueError(f"unknown actor: {actor_id}")

    def _require_solver(self, solver_id: str) -> None:
        self._require_actor(solver_id)
        if solver_id not in self.solver_ids:
            raise ValueError("expected a solver")
