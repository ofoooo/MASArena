"""
AIME Evaluator

Standalone evaluator for AIME-style math problems using Math-Verify for robust mathematical expression evaluation.
"""

import re
from typing import Dict, Any, Tuple

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark
from mas_arena.evaluators.utils.math_equal import calculate_score
from mas_arena.evaluators.utils import extract_answer_numeric

@register_benchmark(
    name="aime",
    normalization_keys={
        "problem": "question",
        "solution": "answer",
    }
)
class AIMEEvaluator(BaseEvaluator):
    """
    Evaluator for AIME-style math problems.
    Uses Math-Verify for robust mathematical expression evaluation.
    """
    # SUPPORTS_CONCURRENCY = False
    
    def __init__(self, name: str = "aime", config: Dict[str, Any] = None):
        super().__init__(name, config)
    
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)

    def extract_answer(self, text: str) -> str:
        """
        Extract the answer from model output text (last number or string).
        """
        return extract_answer_numeric(text)

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        return calculate_score(expected_output, prediction)

 

    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        final_answer = run_result.get("final_answer", "")
        score, extracted_answer = self.calculate_score(problem["solution"], final_answer)
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
        }