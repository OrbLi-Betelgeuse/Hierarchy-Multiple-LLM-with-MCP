"""
QAEvaluation Utilities

Provides LLM-based evaluation for QA answer correctness using an executor as a judge.
"""

from typing import List, Optional
from models.mcp_protocol import Task as MCPTask

class QAEvaluator:
    """LLM-based evaluator for QA answer correctness."""
    def __init__(self, executor):
        self.executor = executor

    async def judge_similarity(self, questions: List[str], answers: List[str], expected_answers: List[str]) -> List[int]:
        """
        Use the LLM executor to judge if each answer matches the expected answer semantically.
        Returns a list of 0/1 scores.
        """
        results = []
        for idx, (q, a, e) in enumerate(zip(questions, answers, expected_answers)):
            prompt = (
                f"Given the following QA pair, judge if the model answer is semantically correct and matches the expected answer.\n"
                f"Question: {q}\n"
                f"Expected Answer: {e}\n"
                f"Model Answer: {a}\n"
                "If the model answer can be considered correct, reply with: Answer is correct!! Otherwise, reply: Answer is incorrect. Only output the sentence."
            )
            task_obj = MCPTask(
                task_id=f"judge_{idx+1}",
                task_type="judge",
                description=prompt,
                parameters={}
            )
            response = await self.executor.execute_task(task_obj)
            # Debug print for judge prompt and response
            print(f"[QAEval] Judge prompt:\n{prompt}\n[QAEval] LLM response: {response}")
            score = 0
            out = ""
            if hasattr(response, "result") and response.result and response.result.get("output"):
                out = response.result["output"].strip()
            elif isinstance(response, str):
                out = response.strip()
            # Accept if 'Answer is correct!!' appears anywhere in the output (case-insensitive)
            if "answer is correct!!" in out.lower():
                score = 1
            results.append(score)
        return results

    async def evaluate_qa(self, questions: List[str], answers: List[str], expected: Optional[List[str]]) -> float:
        if not expected or not answers:
            return 0.0
        scores = await self.judge_similarity(questions, answers, expected)
        return sum(scores) / len(scores) if scores else 0.0
