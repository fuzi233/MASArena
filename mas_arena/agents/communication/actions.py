"""Strict structured actions for synchronous solver and aggregator phases."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class SolverAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["think", "ask", "submit"]
    reasoning_note: str
    recipient_id: str | None = None
    question: str | None = None
    candidate_answer: str | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_continue_reasoning(cls, values: Any) -> Any:
        if isinstance(values, dict) and values.get("action") == "continue_reasoning":
            return {**values, "action": "think"}
        return values

    @model_validator(mode="after")
    def validate_shape(self) -> "SolverAction":
        if not self.reasoning_note.strip():
            raise ValueError("reasoning_note must not be empty")
        if self.action == "ask" and (not self.recipient_id or not self.question or not self.question.strip()):
            raise ValueError("ask requires recipient_id and question")
        if self.action == "submit" and (not self.candidate_answer or not self.candidate_answer.strip()):
            raise ValueError("submit requires candidate_answer")
        return self


class AggregatorAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["think", "ask", "submit"]
    reasoning_note: str
    recipient_id: str | None = None
    question: str | None = None
    final_answer: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "AggregatorAction":
        if not self.reasoning_note.strip():
            raise ValueError("reasoning_note must not be empty")
        if self.action == "ask" and (not self.recipient_id or not self.question or not self.question.strip()):
            raise ValueError("ask requires recipient_id and question")
        if self.action == "submit" and (not self.final_answer or not self.final_answer.strip()):
            raise ValueError("submit requires final_answer")
        return self


class LegacyAggregatorAction(BaseModel):
    """Original aggregator action schema retained for legacy-v1 reproduction."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["ask", "finalize"]
    recipient_id: str | None = None
    question: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "LegacyAggregatorAction":
        if self.action == "ask" and (not self.recipient_id or not self.question or not self.question.strip()):
            raise ValueError("ask requires recipient_id and question")
        return self


class SingleStepAction(BaseModel):
    """One private reasoning step for the no-communication baseline."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["continue_reasoning", "submit"]
    reasoning_note: str
    candidate_answer: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "SingleStepAction":
        if not self.reasoning_note.strip():
            raise ValueError("reasoning_note must not be empty")
        if self.action == "submit" and (not self.candidate_answer or not self.candidate_answer.strip()):
            raise ValueError("submit requires candidate_answer")
        return self
