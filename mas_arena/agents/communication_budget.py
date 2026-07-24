"""MASArena adapter for the configurable communication-budget team."""

import asyncio
import logging
import os
from typing import Any, Dict

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

from .communication.config import TeamConfig
from .communication.orchestrator import TeamOrchestrator


logger = logging.getLogger(__name__)

MAX_MODEL_RETRIES = 5
RETRY_DELAY_SECONDS = 1.0


class ModelRequestError(RuntimeError):
    """Expose safe HTTP diagnostics after a model request exhausts retries."""

    def __init__(self, error: Exception, attempts: int) -> None:
        status_code = getattr(error, "status_code", None)
        request_id = getattr(error, "request_id", None)
        details = [f"model request failed after {attempts} attempt(s)", f"error_type={type(error).__name__}"]
        if status_code is not None:
            details.append(f"status_code={status_code}")
        if request_id:
            details.append(f"request_id={request_id}")
        details.append(f"message={str(error)[:500]}")
        super().__init__("; ".join(details))


class OpenAITextClient:
    """Small OpenAI adapter kept outside protocol and prompt code."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None,
        base_url: str | None,
        temperature: float = 1.0,
        max_model_retries: int = MAX_MODEL_RETRIES,
        retry_delay_seconds: float = RETRY_DELAY_SECONDS,
    ) -> None:
        if max_model_retries < 0:
            raise ValueError("max_model_retries must be non-negative")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        self.model_name = model_name
        self.temperature = temperature
        self.max_model_retries = max_model_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self._usage_records: list[object] = []

    async def complete_text(self, messages: list[dict[str, str]]) -> str:
        for attempt in range(self.max_model_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=self.temperature
                )
                if response.usage is not None:
                    self._usage_records.append(response.usage)
                return (response.choices[0].message.content or "").strip()
            except (APIConnectionError, APITimeoutError, APIStatusError) as error:
                status_code = getattr(error, "status_code", None)
                retryable = not self._is_quota_exhausted(error) and (
                    isinstance(error, (APIConnectionError, APITimeoutError)) or status_code in {408, 409, 429} or (
                    isinstance(status_code, int) and status_code >= 500
                    )
                )
                if not retryable or attempt == self.max_model_retries:
                    raise ModelRequestError(error, attempt + 1) from error
                logger.warning(
                    "Model request failed (%s); retry %d/%d in %.1fs",
                    type(error).__name__,
                    attempt + 1,
                    self.max_model_retries,
                    self.retry_delay_seconds,
                )
                await asyncio.sleep(self.retry_delay_seconds)

        raise RuntimeError("unreachable")

    @staticmethod
    def _is_quota_exhausted(error: Exception) -> bool:
        details = " ".join(
            str(value)
            for value in (str(error), getattr(error, "body", None))
            if value is not None
        ).casefold()
        return any(
            marker in details
            for marker in (
                "insufficient_quota",
                "insufficient quota",
                "quota is not enough",
                "quota exhausted",
                "balance is insufficient",
                "insufficient balance",
                "insufficient credit",
            )
        )

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
            max_model_retries=int(self.config.get("max_model_retries", MAX_MODEL_RETRIES)),
            retry_delay_seconds=float(self.config.get("retry_delay_seconds", RETRY_DELAY_SECONDS)),
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
