import importlib
from types import SimpleNamespace

import httpx
import pytest
from openai import InternalServerError
from unittest.mock import AsyncMock, Mock

from mas_arena.agents.base import AgentSystem
from mas_arena.agents.communication_budget import OpenAITextClient
from mas_arena.benchmark_runner import BenchmarkRunner


communication_budget_module = importlib.import_module("mas_arena.agents.communication_budget")


@pytest.mark.asyncio
async def test_retryable_http_error_retries_five_times_with_one_second_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingCompletions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **kwargs: object) -> object:
            self.calls += 1
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            response = httpx.Response(500, request=request)
            raise InternalServerError("upstream unavailable", response=response, body={"error": "busy"})

    class FailingClient:
        def __init__(self) -> None:
            self.chat = type("Chat", (), {"completions": FailingCompletions()})()

    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    client = OpenAITextClient(model_name="test-model", api_key="test-key", base_url="https://example.test/v1")
    client.client = FailingClient()
    monkeypatch.setattr(
        communication_budget_module,
        "asyncio",
        SimpleNamespace(sleep=fake_sleep),
        raising=False,
    )

    with pytest.raises(Exception):
        await client.complete_text([{"role": "user", "content": "hello"}])

    assert client.client.chat.completions.calls == 6
    assert delays == [1.0] * 5


@pytest.mark.asyncio
async def test_retry_policy_uses_configured_attempts_and_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingCompletions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **kwargs: object) -> object:
            self.calls += 1
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            response = httpx.Response(429, request=request)
            raise InternalServerError("rate limited", response=response, body={"error": "busy"})

    class FailingClient:
        def __init__(self) -> None:
            self.chat = type("Chat", (), {"completions": FailingCompletions()})()

    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    client = OpenAITextClient(
        model_name="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_model_retries=2,
        retry_delay_seconds=2.0,
    )
    client.client = FailingClient()
    monkeypatch.setattr(communication_budget_module, "asyncio", SimpleNamespace(sleep=fake_sleep), raising=False)

    with pytest.raises(Exception):
        await client.complete_text([{"role": "user", "content": "hello"}])

    assert client.client.chat.completions.calls == 3
    assert delays == [2.0, 2.0]


@pytest.mark.asyncio
async def test_quota_exhausted_429_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingCompletions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **kwargs: object) -> object:
            self.calls += 1
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            response = httpx.Response(429, request=request)
            raise InternalServerError("insufficient quota", response=response, body={"error": {"code": "insufficient_quota"}})

    class FailingClient:
        def __init__(self) -> None:
            self.chat = type("Chat", (), {"completions": FailingCompletions()})()

    async def fail_sleep(delay: float) -> None:
        raise AssertionError(f"quota exhaustion must not sleep, got {delay}")

    client = OpenAITextClient(
        model_name="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_model_retries=20,
        retry_delay_seconds=2.0,
    )
    client.client = FailingClient()
    monkeypatch.setattr(communication_budget_module, "asyncio", SimpleNamespace(sleep=fail_sleep), raising=False)

    with pytest.raises(Exception, match="failed after 1 attempt"):
        await client.complete_text([{"role": "user", "content": "hello"}])

    assert client.client.chat.completions.calls == 1


@pytest.mark.asyncio
async def test_evaluation_error_preserves_duration_and_error_details(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    class FailingAgent(AgentSystem):
        async def run_agent(self, problem: dict[str, object], **kwargs: object) -> dict[str, object]:
            raise RuntimeError("status_code=500; request_id=req_123")

    perf_counter_values = iter([10.0, 10.25])
    base_module = importlib.import_module("mas_arena.agents.base")
    monkeypatch.setattr(base_module.time, "perf_counter", lambda: next(perf_counter_values))
    agent = FailingAgent(
        config={
            "evaluator": "math",
            "responses_dir": str(tmp_path / "responses"),
            "visualizations_dir": str(tmp_path / "visualizations"),
        }
    )

    result = await agent.evaluate({"id": "case-1", "problem": "Question", "solution": "A"})

    assert result["status"] == "error"
    assert result["execution_time_ms"] == pytest.approx(250.0)
    assert result["error_type"] == "RuntimeError"
    assert result["error"] == "status_code=500; request_id=req_123"


@pytest.mark.asyncio
async def test_runner_prints_error_summary_with_actual_duration(tmp_path: object, capsys: pytest.CaptureFixture[str]) -> None:
    agent = Mock()
    agent.name = "test-agent"
    agent.evaluate = AsyncMock(
        return_value={
            "status": "error",
            "score": 0.0,
            "is_correct": False,
            "reasoning": "Evaluation failed with error: status_code=500",
            "error": "status_code=500; request_id=req_123",
            "error_type": "ModelRequestError",
            "execution_time_ms": 1_250.0,
            "llm_usage": {},
        }
    )
    runner = BenchmarkRunner(results_dir=str(tmp_path))

    result = await runner._process_one_problem(
        0,
        {"id": "case-1", "problem": "Question", "solution": "A"},
        agent,
        {"normalization_keys": {"id": "id", "problem": "problem", "solution": "solution"}},
        verbose=True,
    )

    assert result["error_type"] == "ModelRequestError"
    assert result["error"] == "status_code=500; request_id=req_123"
    assert "Result: E (1250ms) — status_code=500; request_id=req_123" in capsys.readouterr().out
