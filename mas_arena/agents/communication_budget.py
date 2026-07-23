"""MASArena adapter for the configurable communication-budget team."""

import os
from typing import Any, Dict

from openai import AsyncOpenAI

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

from .communication.config import TeamConfig
from .communication.orchestrator import TeamOrchestrator


class OpenAITextClient:
    """Small OpenAI adapter kept outside protocol and prompt code."""

    def __init__(self, *, model_name: str, api_key: str | None, base_url: str | None, temperature: float = 1.0) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._usage_records: list[object] = []

    async def complete_text(self, messages: list[dict[str, str]]) -> str:
        response = await self.client.chat.completions.create(
            model=self.model_name, messages=messages, temperature=self.temperature
        )
        if response.usage is not None:
            self._usage_records.append(response.usage)
        return (response.choices[0].message.content or "").strip()

    def drain_usage(self) -> list[object]:
        """Return usage for this problem and reset the per-problem buffer."""
        records = self._usage_records
        self._usage_records = []
        return records

    async def aclose(self) -> None:
        await self.client.close()


class CommunicationBudgetMAS(AgentSystem):
    """A dynamic solver team with one aggregator and independent per-member K."""

    def __init__(self, name: str = "communication_budget", config: Dict[str, Any] | None = None):
        super().__init__(name, config)
        self.config = config or {}
        self.team_config = TeamConfig.from_mapping(self.config)
        self.model_client = self.config.get("model_client") or OpenAITextClient(
            model_name=self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini"),
            api_key=self.config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            base_url=self.config.get("api_base") or os.getenv("OPENAI_API_BASE"),
            temperature=self.team_config.temperature,
        )

    async def run_agent(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(problem.get("problem"), str) or not problem["problem"].strip():
            raise ValueError("problem must contain non-empty 'problem' text")
        return await TeamOrchestrator(self.team_config, self.model_client).solve(problem["problem"])

    async def aclose(self) -> None:
        close = getattr(self.model_client, "aclose", None)
        if callable(close):
            await close()


AgentSystemRegistry.register("communication_budget", CommunicationBudgetMAS)
