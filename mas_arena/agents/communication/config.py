"""Validated configuration for one communication-budget team."""

from dataclasses import dataclass
from typing import Literal, Mapping, Any


BudgetVisibility = Literal["visible", "hidden"]
ProtocolVersion = Literal["bounded-v2", "legacy-v1"]


@dataclass(frozen=True)
class TeamConfig:
    """Runtime options shared by the team, excluding model-client details."""

    solver_count: int = 2
    communication_budget: int | None = None
    solver_communication_budget: int = 1
    aggregator_communication_budget: int | None = None
    budget_visibility: BudgetVisibility = "visible"
    max_turns: int = 4
    aggregator_max_steps: int | None = None
    protocol_version: ProtocolVersion = "bounded-v2"
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.solver_count < 1:
            raise ValueError("solver_count must be at least 1")
        if self.communication_budget is not None and self.communication_budget < 0:
            raise ValueError("communication_budget must be non-negative")
        if self.solver_communication_budget < 0:
            raise ValueError("solver_communication_budget must be non-negative")
        if self.aggregator_communication_budget is not None and self.aggregator_communication_budget < 0:
            raise ValueError("aggregator_communication_budget must be non-negative")
        if self.budget_visibility not in {"visible", "hidden"}:
            raise ValueError("budget_visibility must be 'visible' or 'hidden'")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.aggregator_max_steps is not None and self.aggregator_max_steps < 1:
            raise ValueError("aggregator_max_steps must be at least 1")
        if self.protocol_version not in {"bounded-v2", "legacy-v1"}:
            raise ValueError("protocol_version must be 'bounded-v2' or 'legacy-v1'")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "TeamConfig":
        legacy_budget = config.get("communication_budget")
        solver_budget = config.get("solver_communication_budget")
        if solver_budget is None:
            solver_budget = legacy_budget if legacy_budget is not None else 1
        aggregator_budget = config.get("aggregator_communication_budget")
        if aggregator_budget is None:
            aggregator_budget = legacy_budget if legacy_budget is not None else solver_budget
        return cls(
            solver_count=int(config.get("solver_count", 2)),
            communication_budget=int(legacy_budget) if legacy_budget is not None else None,
            solver_communication_budget=int(solver_budget),
            aggregator_communication_budget=int(aggregator_budget),
            budget_visibility=config.get("budget_visibility", "visible"),
            max_turns=int(config.get("max_turns", 4)),
            aggregator_max_steps=(
                int(config["aggregator_max_steps"]) if config.get("aggregator_max_steps") is not None else None
            ),
            protocol_version=config.get("protocol_version", "bounded-v2"),
            temperature=float(config.get("temperature", 1.0)),
        )

    @property
    def resolved_aggregator_communication_budget(self) -> int:
        return self.solver_communication_budget if self.aggregator_communication_budget is None else self.aggregator_communication_budget

    @property
    def resolved_aggregator_max_steps(self) -> int:
        return self.max_turns if self.aggregator_max_steps is None else self.aggregator_max_steps
