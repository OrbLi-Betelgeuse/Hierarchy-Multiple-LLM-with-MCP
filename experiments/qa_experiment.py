"""
Multi-round Question Answering Experiment

Evaluates the Manager-Executor collaboration model on multi-turn question answering tasks.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from rich.console import Console
from models.executor import Executor
from models.llm_interface import create_llm_interface
from models.manager import Manager

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class QATask:
    """Represents a QA task."""

    task_id: str
    context: str
    questions: List[str]
    expected_answers: Optional[List[str]] = None


@dataclass
class QAResult:
    """Result of QA experiment."""

    task_id: str
    questions: List[str]
    answers: List[str]
    execution_time: float
    accuracy_score: Optional[float] = None


class QAExperiment:
    """Experiment class for multi-round question answering."""

    def __init__(
        self, manager_config: Dict[str, Any], executor_configs: List[Dict[str, Any]]
    ):
        self.manager_config = manager_config
        self.executor_configs = executor_configs
        self.manager = None
        self.executors = []
        self.results = []

        # Sample QA tasks
        self.sample_tasks = [
            QATask(
                task_id="qa_1",
                context="Artificial Intelligence (AI) is transforming healthcare through improved diagnostics and patient care.",
                questions=[
                    "What is AI doing in healthcare?",
                    "What are the main benefits?",
                    "Are there any challenges?",
                ],
                expected_answers=[
                    "AI is transforming healthcare through improved diagnostics and patient care",
                    "Improved diagnostics and better patient care",
                    "Yes, there are challenges in implementation and validation",
                ],
            ),
            QATask(
                task_id="qa_2",
                context="The Amazon rainforest is the largest tropical rainforest in the world and is home to a vast diversity of species.",
                questions=[
                    "What is the Amazon rainforest?",
                    "Why is it important?",
                    "Name one threat to the Amazon rainforest.",
                ],
                expected_answers=[
                    "The largest tropical rainforest in the world",
                    "It is important for biodiversity and climate regulation",
                    "Deforestation",
                ],
            ),
            QATask(
                task_id="qa_3",
                context="Python is a popular programming language known for its readability and versatility.",
                questions=[
                    "What is Python?",
                    "What is Python known for?",
                ],
                expected_answers=[
                    "A popular programming language",
                    "Readability and versatility",
                ],
            ),
        ]

    async def setup(self):
        """Enhanced setup with executor instance tracking, supports executor-only mode."""
        try:
            # Only set up manager if manager_config is provided
            if self.manager_config is not None:
                manager_llm = create_llm_interface(
                    provider=self.manager_config["provider"],
                    model_name=self.manager_config["model"],
                    **self.manager_config.get("kwargs", {}),
                )
                self.manager = Manager(
                    manager_id=self.manager_config["manager_id"], llm_interface=manager_llm
                )
            else:
                self.manager = None

            # Executors setup with instance tracking
            for config in self.executor_configs:
                executor_llm = create_llm_interface(
                    provider=config["provider"],
                    model_name=config["model"],
                    **config.get("kwargs", {}),
                )
                executor = Executor(
                    executor_id=config["executor_id"],
                    llm_interface=executor_llm,
                    capabilities=config["capabilities"],
                )
                self.executors.append(executor)

                # Register with manager if present
                if self.manager is not None:
                    await self.manager.register_executor(
                        executor_id=config["executor_id"],
                        capabilities=config["capabilities"],
                    )
                    if not hasattr(self.manager, "executor_instances"):
                        self.manager.executor_instances = {}
                    self.manager.executor_instances[config["executor_id"]] = executor

            logger.info(f"Initialized: {('1 manager, ' if self.manager else '0 manager, ')}{len(self.executors)} executors")

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

    async def run_experiment(self) -> List[QAResult]:
        results = []
        for task in self.sample_tasks:
            try:
                result = await self._execute_with_fallback(task)
                self._display_task_result(task, result)
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {task.task_id} - {e}")
                results.append(
                    QAResult(
                        task_id=task.task_id,
                        questions=task.questions,
                        answers=["" for _ in task.questions],
                        execution_time=0.0,
                        accuracy_score=0.0,
                    )
                )
        self.results = results
        return results

    async def _execute_with_fallback(self, task: QATask) -> QAResult:
        """Try to answer each question individually if multi-question prompt fails. Supports executor-only mode and LLM-based scoring."""
        start_time = asyncio.get_event_loop().time()
        answers = []
        scores = []
        per_question_times = []
        try:
            prompt = self._build_prompt(task)
            console.print(f"\n[yellow]DEBUG Prompt Sent:[/yellow]\n{prompt}")
            if self.manager is None:
                # Executor-only mode: send each question to the first executor
                if not self.executors:
                    raise RuntimeError("No executors available")
                executor = self.executors[0]
                from models.executor import Task
                from utils.evaluation import Evaluator
                evaluator = Evaluator()
                for idx, q in enumerate(task.questions):
                    task_obj = Task(
                        task_id=f"exec_{task.task_id}_q{idx+1}",
                        description=f"Context: {task.context}\nQuestion: {q}\nPlease answer by extracting only the relevant factual information from the context above. Do not include any reasoning, speculation, or meta-cognitive statements. Respond with concise facts only.",
                        parameters={},
                        task_type="qa"
                    )
                    result = await executor.execute_task(task_obj)
                    # Record per-question execution time if available
                    exec_time = getattr(result, "execution_time", None)
                    if exec_time is not None:
                        per_question_times.append(exec_time)
                    else:
                        per_question_times.append(0.0)
                    if hasattr(result, "result"):
                        ans = result.result.get("output", "")
                    elif isinstance(result, dict):
                        ans = result.get("output", "")
                    else:
                        ans = str(result) if result else ""
                    # Strip <think>...</think> sections from answer
                    ans = evaluator.strip_think_sections(ans)
                    answers.append(ans if ans else "No answer generated")
                # LLM-based scoring if expected answers are available
                if task.expected_answers:
                    for idx, (gen, exp) in enumerate(zip(answers, task.expected_answers)):
                        score_prompt = (
                            f"Compare the following answer to the expected answer and provide a standardized accuracy score between 0 and 1.\n"
                            f"Answer: {gen}\n"
                            f"Expected: {exp}\n"
                            f"Score (0-1):"
                        )
                        score_task = Task(
                            task_id=f"score_{task.task_id}_q{idx+1}",
                            description=score_prompt,
                            parameters={},
                            task_type="qa"
                        )
                        score_result = await executor.execute_task(score_task)
                        score_str = ""
                        if hasattr(score_result, "result"):
                            score_str = score_result.result.get("output", "")
                        elif isinstance(score_result, dict):
                            score_str = score_result.get("output", "")
                        else:
                            score_str = str(score_result) if score_result else ""
                        # Try to extract a float between 0 and 1
                        try:
                            score = float(next((s for s in score_str.split() if self._is_float(s) and 0 <= float(s) <= 1), "0"))
                        except Exception:
                            score = 0.0
                        scores.append(score)
                else:
                    scores = [0.0 for _ in answers]
            else:
                # Full manager-executor flow (no LLM-based scoring)
                response = await self.manager.execute_task(
                    task_description=prompt,
                    task_type="qa",
                )
                console.print(f"[yellow]DEBUG Raw Response:[/yellow]\n{json.dumps(response, indent=2)}")
                extracted = False
                if response:
                    if response.get("answers"):
                        answers = [Evaluator().strip_think_sections(a) for a in response["answers"]]
                        extracted = True
                    elif response.get("output"):
                        output = response["output"]
                        if isinstance(output, list):
                            answers = [Evaluator().strip_think_sections(a) for a in output]
                            extracted = True
                        elif isinstance(output, str):
                            answers = [Evaluator().strip_think_sections(line.strip()) for line in output.split("\n") if line.strip()]
                            extracted = bool(answers)
                    elif response.get("task_results"):
                        task_results = response["task_results"]
                        if isinstance(task_results, dict):
                            outputs = []
                            for key in sorted(task_results.keys()):
                                out = task_results[key].get("output", "")
                                outputs.append(Evaluator().strip_think_sections(out))
                            if any(o for o in outputs):
                                if len(outputs) == len(task.questions):
                                    answers = outputs
                                else:
                                    answers = [outputs[0]] * len(task.questions)
                                extracted = True
                # If nothing extracted or all answers are empty, try per-question execution
                if not extracted or not any(a.strip() for a in answers):
                    answers = []
                    for q in task.questions:
                        single_prompt = (
                            f"Context: {task.context}\nQuestion: {q}\nPlease answer by extracting only the relevant factual information from the context above. Do not include any reasoning, speculation, or meta-cognitive statements. Respond with concise facts only."
                        )
                        single_response = await self.manager.execute_task(
                            task_description=single_prompt,
                            task_type="qa",
                        )
                        ans = ""
                        if single_response:
                            if isinstance(single_response, dict):
                                if single_response.get("output"):
                                    ans = single_response["output"]
                                elif single_response.get("answers"):
                                    a_list = single_response["answers"]
                                    if isinstance(a_list, list) and a_list:
                                        ans = a_list[0]
                            elif isinstance(single_response, str):
                                ans = single_response
                        # Strip <think>...</think> sections from answer
                        ans = Evaluator().strip_think_sections(ans)
                        answers.append(ans if ans else "No answer generated")
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            answers = await self._simple_generation(task)
            scores = [0.0 for _ in answers]
        # Ensure answers list matches number of questions
        if len(answers) < len(task.questions):
            answers += ["No answer generated" for _ in range(len(task.questions) - len(answers))]
            scores += [0.0 for _ in range(len(task.questions) - len(scores))]
        elif len(answers) > len(task.questions):
            answers = answers[:len(task.questions)]
            scores = scores[:len(task.questions)]
        exec_time = asyncio.get_event_loop().time() - start_time
        # Use average LLM-based score as accuracy_score
        accuracy = sum(scores) / len(scores) if scores else 0.0
        # Store per-question execution times in QAResult (as a new field)
        result_obj = QAResult(
            task_id=task.task_id,
            questions=task.questions,
            answers=answers,
            execution_time=exec_time,
            accuracy_score=accuracy,
        )
        # Attach per-question times for output
        result_obj.per_question_times = per_question_times
        return result_obj

    def _is_float(self, s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    def _build_prompt(self, task: QATask) -> str:
        prompt = f"Context: {task.context}\n"
        for idx, q in enumerate(task.questions, 1):
            prompt += f"Q{idx}: {q}\n"
        prompt += (
            "Please answer each question by extracting only the relevant factual information from the context above. Do not include any reasoning, speculation, or meta-cognitive statements. Respond with concise facts only."
        )
        return prompt

    async def _direct_executor_execution(self, task: QATask) -> List[str]:
        # Fallback: use first executor directly
        if self.executors:
            executor = self.executors[0]
            answers = []
            for q in task.questions:
                response = await executor.execute_task(
                    task_description=f"Context: {task.context}\nQuestion: {q}",
                    task_type="qa",
                )
                if isinstance(response, dict) and response.get("output"):
                    answers.append(response["output"])
                elif isinstance(response, str):
                    answers.append(response)
                else:
                    answers.append(str(response))
            return answers
        return ["No executor available"] * len(task.questions)

    async def _simple_generation(self, task: QATask) -> List[str]:
        # Final fallback: echo context or dummy answers
        return ["No answer generated"] * len(task.questions)

    def _evaluate_qa(self, answers: List[str], expected: Optional[List[str]]) -> float:
        if not expected or not answers:
            return 0.0
        correct = 0
        for a, e in zip(answers, expected):
            if self._normalize_answer(a) == self._normalize_answer(e):
                correct += 1
        return correct / len(expected)

    def _normalize_answer(self, s: str) -> str:
        return s.strip().lower()

    def _display_task_result(self, task: QATask, result: QAResult):
        console.print(f"\n[bold]Task {task.task_id}[/bold]")
        console.print(f"[blue]Context:[/blue] {task.context}")
        for idx, (q, a) in enumerate(zip(task.questions, result.answers), 1):
            console.print(f"[green]Q{idx}:[/green] {q}")
            console.print(f"[yellow]A{idx}:[/yellow] {a}")
        if task.expected_answers:
            for idx, ea in enumerate(task.expected_answers, 1):
                console.print(f"[cyan]Expected A{idx}:[/cyan] {ea}")
        console.print(f"[magenta]Accuracy: {result.accuracy_score:.2f} | Time: {result.execution_time:.2f}s[/magenta]")

    def generate_report(self) -> Dict[str, Any]:
        if not self.results:
            return {"status": "no_results"}
        successful = [r for r in self.results if r.accuracy_score and r.accuracy_score >0.5]
        avg_accuracy = sum(r.accuracy_score or 0 for r in self.results) / len(self.results)
        return {
            "experiment_type": "qa",
            "total_tasks": len(self.results),
            "successful_tasks": len(successful),
            "success_rate": len(successful) / len(self.results) if self.results else 0,
            "average_accuracy": avg_accuracy,
            "average_execution_time": sum(r.execution_time for r in self.results) / len(self.results),
            "detailed_results": [
                {
                    "task_id": r.task_id,
                    "accuracy_score": r.accuracy_score,
                    "execution_time": r.execution_time,
                    "per_question_times": getattr(r, "per_question_times", []),
                    "status": "success" if (r.accuracy_score or 0) >0.5 else "failed",
                }
                for r in self.results
            ],
        }
