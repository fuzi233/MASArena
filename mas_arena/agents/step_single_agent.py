"""Private multi-step single-agent baseline without communication."""

import json
import os
from typing import Any, Dict

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

from .communication.actions import SingleStepAction
from .communication.prompts import PromptComposer
from .communication_budget import OpenAITextClient


class StepSingleAgent(AgentSystem):
    """One solver that may reason privately for a bounded number of steps."""

    def __init__(self, name: str = "step_single_agent", config: Dict[str, Any] | None = None):
        super().__init__(name, config)
        self.config = config or {}
        self.max_turns = int(self.config.get("max_turns", 4))
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.temperature = float(self.config.get("temperature", 1.0))
        self.model_client = self.config.get("model_client") or OpenAITextClient(
            model_name=self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini"),
            api_key=self.config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            base_url=self.config.get("api_base") or os.getenv("OPENAI_API_BASE"),
            temperature=self.temperature,
        )

    async def run_agent(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        problem_text = problem["problem"]
        notes: list[str] = []
        submitted_answer: str | None = None
        for _ in range(self.max_turns):
            action = await self._request_action(problem_text, notes)
            if action is None:
                notes.append("Malformed action; continued private reasoning.")
                continue
            notes.append(action.reasoning_note)
            if action.action == "submit":
                submitted_answer = action.candidate_answer
                break
        final_answer = (
            f"<final_answer>{submitted_answer}</final_answer>"
            if submitted_answer is not None
            else await self.model_client.complete_text(PromptComposer.single_final_prompt(problem=problem_text, reasoning_notes=notes))
        )
        messages = [
            {"agent_id": "step_single_agent", "name": "step_single_agent", "role": "assistant", "content": note}
            for note in notes
        ]
        drain_usage = getattr(self.model_client, "drain_usage", None)
        if callable(drain_usage):
            messages.extend({"agent_id": f"model_call_{index}", "name": "model_call", "role": "assistant", "content": "", "usage_metadata": usage} for index, usage in enumerate(drain_usage(), start=1))
        messages.append({"agent_id": "step_single_agent", "name": "step_single_agent", "role": "assistant", "content": final_answer})
        return {"final_answer": final_answer, "messages": messages, "reasoning_notes": notes}

    async def _request_action(self, problem: str, notes: list[str]) -> SingleStepAction | None:
        raw = await self.model_client.complete_text(PromptComposer.single_step_prompt(problem=problem, reasoning_notes=notes))
        try:
            return SingleStepAction.model_validate(json.loads(raw))
        except (ValueError, json.JSONDecodeError):
            repair = await self.model_client.complete_text(PromptComposer.json_repair_prompt(invalid_output=raw, action_kind="solver"))
            try:
                return SingleStepAction.model_validate(json.loads(repair))
            except (ValueError, json.JSONDecodeError):
                return None

    async def aclose(self) -> None:
        close = getattr(self.model_client, "aclose", None)
        if callable(close):
            await close()


AgentSystemRegistry.register("step_single_agent", StepSingleAgent)
