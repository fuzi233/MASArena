"""
MBPP Evaluator
"""

from __future__ import annotations

import traceback
from threading import Thread
from typing import Any, Dict, List, Tuple

from mas_arena.evaluators.base_code_evaluator import BaseCodeEvaluator
from mas_arena.evaluators.utils.sanitize import sanitize
from mas_arena.evaluators.registry import register_benchmark


class TimeoutError(Exception):
    """Raised when the sandboxed execution exceeds the time-limit."""


def run_with_timeout(func, args: tuple[Any, ...] = (), timeout: int = 15):
    """
    Execute `func(*args)` in a daemon thread.
    Raise `TimeoutError` if it runs longer than `timeout` seconds.
    """
    result: list[Any] = []
    exception: list[BaseException] = []

    def target():
        try:
            result.append(func(*args))
        except BaseException as e:
            exception.append(e)

    thread = Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"Execution timed out after {timeout}s")

    if exception:
        raise exception[0]

    return result[0] if result else None


@register_benchmark(
    name="mbpp",
    normalization_keys={
        "id": "task_id",
        "problem": "prompt",
        "solution": "code",
        "test": "test",
        "entry_point": "entry_point",
        "test_imports": "test_imports"
    }
)
class MBPPEvaluator(BaseCodeEvaluator):
    """Evaluator for MBPP code-generation tasks."""

    def __init__(self, name: str = "mbpp", config: Dict[str, Any] | None = None):
        super().__init__(name, config)

    def check_solution(
        self,
        code: str,
        test: str,
        entry_point: str,
        test_imports: List[str] | None = None,
    ) -> Tuple[bool, str]:
        """
        Compile user code, run official MBPP `check()`.

        Returns:
            (passed: bool, message: str)
            `passed` is True iff all assertions succeed within the time-limit.
        """
        try:
            # Remove Markdown, ensure the target function exists
            code_clean = sanitize(code=code, entrypoint=entry_point)

            # Isolated global namespace
            env: Dict[str, Any] = {}
            exec(code_clean, env)

            # Execute additional import statements if provided
            for stmt in test_imports or []:
                exec(stmt, env)

            if entry_point not in env:
                raise ValueError(f"Function `{entry_point}` is missing in submitted code.")

            # Inject and run the official unit tests
            exec(test, env)
            check_fn = env["check"]

            run_with_timeout(check_fn, timeout=15)  # `check()` takes no args
            return True, "All tests passed"

        except TimeoutError as te:
            return False, str(te)
        except AssertionError as ae:
            return False, f"Assertion failed: {ae}"
        except Exception as exc:  # noqa: BLE001
            if self.config.get("verbose"):
                self.logger.error(traceback.format_exc())
            return False, f"Execution error: {exc}"
