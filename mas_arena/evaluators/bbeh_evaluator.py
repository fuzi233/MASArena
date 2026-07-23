"""BBEH evaluator backed by the repository's canonical answer matching logic."""

from pathlib import Path
import re
from typing import Any, Dict

from bbeh.evaluate import evaluate_correctness, extract_answer

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark


_BBEH_MANIFEST = Path(__file__).resolve().parents[3] / "manifests" / "bbeh_probe_v1.jsonl"


@register_benchmark(
    name="bbeh",
    normalization_keys={"id": "id", "problem": "input", "solution": "target"},
    data_path=str(_BBEH_MANIFEST),
)
class BBEHEvaluator(BaseEvaluator):
    """Score raw aggregator output while preserving it in `final_answer`."""

    def extract_answer(self, text: str) -> str:
        # 新协议优先：只接受唯一且非空的最终答案标签，避免把推理文本计入评分。
        matches = re.findall(r"<final_answer>\s*(.*?)\s*</final_answer>", text, flags=re.DOTALL)
        if len(matches) == 1 and matches[0]:
            return matches[0]
        # 对历史运行和模型未遵守协议的输出保持兼容。
        return extract_answer(text)

    def verify_answer(self, prediction: str, reference: str | Dict[str, Any]) -> bool:
        expected = reference if isinstance(reference, str) else str(reference.get("solution", ""))
        return evaluate_correctness(prediction, expected)

    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        final_answer = str(run_result.get("final_answer", ""))
        extracted_answer = self.extract_answer(final_answer)
        is_correct = self.verify_answer(extracted_answer, problem["solution"])
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": float(is_correct),
            "is_correct": is_correct,
            "message": "Correct" if is_correct else "Incorrect",
        }
