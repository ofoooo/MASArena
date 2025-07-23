import mas_arena.core_serializer.operators as operator
from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.utils.llm_utils import get_agent_by_name
from . import prompt as prompt_custom

class Workflow:

    def __init__(
            self,
            name: str,
            agent_name: str,
            evaluator: BaseEvaluator
    ):
        self.name = name
        self.agent = get_agent_by_name(agent_name)
        self.evaluator = evaluator
        self.custom = operator.Custom(self.agent)
        self.custom_code_generate = operator.CustomCodeGenerate(self.agent)
        self.test = operator.Test(self.agent)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point,
                                                   instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
        test_result = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point,
                                      evaluator=self.evaluator)
        
        if test_result['result']:
            return test_result['solution']
        else:
            return solution['response']
