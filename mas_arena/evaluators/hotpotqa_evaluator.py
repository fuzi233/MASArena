"""
HotpotQA Evaluator

This module provides a standalone evaluator for HotpotQA (Multi-hop Question Answering) problems.
"""

import re
import string
import time
from collections import Counter
from typing import Dict, Any, Tuple

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark


@register_benchmark(
    name="hotpotqa",
    normalization_keys={
        "id": "id",
        "problem": "question",
        "solution": "answer",
    }
)
class HotpotQAEvaluator(BaseEvaluator):
    """Evaluator for HotpotQA problems"""
    
    def __init__(self, name: str = "hotpotqa", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.run_evaluator = RunEvaluator()
        
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)
        
    def normalize_answer(self, s: str) -> str:
        """
        Normalize answers for evaluation.
        
        Args:
            s: The answer string to normalize
            
        Returns:
            Normalized answer string
        """
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
            
        def white_space_fix(text):
            return " ".join(text.split())
            
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
            
        def lower(text):
            return text.lower()
            
        return white_space_fix(remove_articles(remove_punc(lower(s))))
        
    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """
        Compute the F1 score between prediction and ground truth answers.
        
        Args:
            ground_truth: The ground truth answer
            prediction: The predicted answer
            
        Returns:
            Tuple of (f1_score, prediction)
        """
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0, prediction
            
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1, prediction
        
    def create_run(self, problem: Dict[str, Any], final_answer: str, extracted_answer: str, score: float) -> Run:
        """Create a LangSmith run for evaluation"""
        import uuid
        
        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={
                "question": problem["question"],
                "context": problem["context"]
            },
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": problem["answer"],
                "score": score,
                "passed": score >= 0.3,  # HotpotQA uses 0.3 as threshold
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            trace_id=str(uuid.uuid4()),
        )
        
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a problem given the agent's response.
        
        Args:
            problem: The problem dictionary with "question", "context", and "answer" keys
            run_result: The result from running the agent system
            
        Returns:
            Evaluation results dictionary
        """
        # Extract the final answer
        final_answer = run_result.get("final_answer", "")
        
        # Process context
        paragraphs = [item[1] for item in problem["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        
        # Calculate score
        score, extracted_answer = self.calculate_score(problem["answer"], final_answer)
        
        # # Create LangSmith run
        # run = self.create_run(problem, final_answer, extracted_answer, score)
        # run_evaluation = self.run_evaluator.evaluate_run(run=run)
        
        # Log mismatch if score is too low
        if score < 0.3:
            with open(f"{self.log_path}/mismatches.log", "a") as f:
                f.write(f"\nQuestion: {problem['question']}\n")
                f.write(f"Context: {context_str}\n")
                f.write(f"Expected: {problem['answer']}\n")
                f.write(f"Predicted: {final_answer}\n")
                f.write(f"Score: {score}\n")
        
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
            "context": context_str
        }