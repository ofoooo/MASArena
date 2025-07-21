"""
MBPP Evaluator
"""

from __future__ import annotations

import asyncio
import random
import traceback
from threading import Thread
from typing import Any, Dict, List, Tuple, Optional

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

    def _load_data(self):

        self._train_data = []
        self._dev_data = self._load_dateset_from_path(f"data/{self.name}_validate.jsonl")
        self._test_data = self._load_dateset_from_path(f"data/{self.name}_test.jsonl")
        self._test_cases = self._load_dateset_from_path(f"data/{self.name}_public_test.jsonl")

    async def async_evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        evaluate_result = await asyncio.to_thread(self.evaluate, run_result=run_result, problem=problem)
        return evaluate_result

    def extract_test_cases_with_entry_point(self, entry_point: str):

        hardcoded_cases = {
            "remove_odd": "",
            "replace_spaces": "",
            "snake_to_camel": "",
            "Split": "",
            "swap_List": "",
            "square_Sum": "",
            "sort_sublists": "",
            "unique_sublists": "",
        }
        if entry_point in hardcoded_cases:
            return hardcoded_cases[entry_point]

        for case in self._test_cases:
            if case["entry_point"] == entry_point:
                return case["test"]

        return None

    def _get_data(self, data: List[dict], indices: Optional[List[int]] = None, sample_size: Optional[int] = None,
                  seed: Optional[int] = None) -> List[dict]:
        if indices is None:
            indices = list(range(len(data)))
        if sample_size is not None:
            if seed is not None:
                random.seed(seed)
            indices = random.sample(indices, k=min(sample_size, len(indices)))
        return_data = [data[idx] for idx in indices]
        return return_data

    def get_train_data(self, indices: Optional[List[int]] = None, sample_size: Optional[int] = None,
                       seed: Optional[int] = None) -> List[dict]:
        if self._train_data is None:
            print(f"Train data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return []

        return self._get_data(data=self._train_data, indices=indices, sample_size=sample_size, seed=seed)

    def get_dev_data(self, indices: Optional[List[int]] = None, sample_size: Optional[int] = None,
                     seed: Optional[int] = None) -> List[dict]:
        if self._dev_data is None:
            print(f"Dev data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return []

        return self._get_data(data=self._dev_data, indices=indices, sample_size=sample_size, seed=seed)

    def get_test_data(self, indices: Optional[List[int]] = None, sample_size: Optional[int] = None,
                      seed: Optional[int] = None) -> List[dict]:
        if self._test_data is None:
            print(f"Test data for evaluator {type(self).__name__} is not loaded or None. Return an empty list.")
            return []

        return self._get_data(data=self._test_data, indices=indices, sample_size=sample_size, seed=seed)
