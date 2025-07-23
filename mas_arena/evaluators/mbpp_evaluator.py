"""
MBPP Evaluator
"""

from __future__ import annotations

import asyncio
import re
import time
import traceback
from threading import Thread
from typing import Any, Dict, List, Tuple

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run
from mas_arena.evaluators.base_code_evaluator import BaseCodeEvaluator
from mas_arena.evaluators.utils.sanitize import sanitize
from mas_arena.evaluators.registry import register_benchmark
from mas_arena.evaluators.utils.sanitize import sanitize, code_extract


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
    
    def _load_data(self):
        self._train_data = []
        self._dev_data = self._load_dateset_from_path(f"data/{self.name}_validate.jsonl")
        self._test_data = self._load_dateset_from_path(f"data/{self.name}_test.jsonl")
        self._test_cases = self._load_dateset_from_path(f"data/{self.name}_public_test.jsonl")

    async def async_evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        evaluate_result = await asyncio.to_thread(self.evaluate, run_result=run_result, problem=problem)
        return evaluate_result
    
    def calculate_score(
        self, test_code: str, prediction: str, entry_point: str
    ) -> Tuple[float, str, str]:
        """
        Return ``(score, code_used_for_test, message)`` where *score* is 1.0 on success, 0.0 otherwise.
        """
        passed, message = self.check_solution(prediction, test_code, entry_point)
        return (1.0 if passed else 0.0), prediction, message
        
    def create_run(
        self,
        problem: Dict[str, Any],
        final_answer: str,
        extracted_answer: str,
        score: float,
        message: str,
    ) -> Run:
        """Package the evaluation result as a ``Run`` object for LangSmith."""
        import uuid

        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"problem": problem["problem"], "task_id": problem["id"]},
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": problem["test"],
                "score": score,
                "message": message,
                "passed": score == 1.0,
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
            trace_id=str(uuid.uuid4()),
        )

    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point – keeps the outer interface unchanged.
        Consumes one *problem* dict and the model *run_result*, returns a detailed evaluation dict.
        """
        final_answer = run_result.get("final_answer", "")
        extracted_answer = final_answer
        if not run_result.get("extracted"):
            extracted_answer = self.extract_code(final_answer)

        score, extracted_answer, message = self.calculate_score(
            problem["test"], extracted_answer, problem["entry_point"]
        )

        run = self.create_run(problem, final_answer, extracted_answer, score, message)
        run_evaluation = self.run_evaluator.evaluate_run(run=run)

        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
            "message": message,
            "run_evaluation": run_evaluation,
        }

    def extract_test_cases_with_entry_point(self, entry_point: str):
        """
        Extract test cases with the given entry point.
        """

        hardcoded_cases = {
            "find_zero": "",
            "decode_cyclic": "",
            "decode_shift": "",
            "by_length": "",
            "add": "",
            "triangle_area": "",
            "correct_bracketing": "",
            "solve": "",
            "sum_squares": "",
            "starts_one_ends": "",
        }
        if entry_point in hardcoded_cases:
            return hardcoded_cases[entry_point]

        for case in self._test_cases:
            if case["entry_point"] == entry_point:
                return case["test"]

        return None