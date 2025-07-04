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
from utils.evaluation import Evaluator
from utils.qa_evaluation import QAEvaluator

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
            # QATask(
            #     task_id="qa_1",
            #     context="Artificial Intelligence (AI) is transforming healthcare through improved diagnostics and patient care.",
            #     questions=[
            #         "What is AI doing in healthcare?",
            #         "What are the main benefits?",
            #         "Are there any challenges?",
            #     ],
            #     expected_answers=[
            #         "AI is transforming healthcare through improved diagnostics and patient care",
            #         "Improved diagnostics and better patient care",
            #         "Yes, there are challenges in implementation and validation",
            #     ],
            # ),
            # QATask(
            #     task_id="qa_2",
            #     context="The Amazon rainforest is the largest tropical rainforest in the world and is home to a vast diversity of species.",
            #     questions=[
            #         "What is the Amazon rainforest?",
            #         "Why is it important?",
            #         "Name one threat to the Amazon rainforest.",
            #     ],
            #     expected_answers=[
            #         "The largest tropical rainforest in the world",
            #         "It is important for biodiversity and climate regulation",
            #         "Deforestation",
            #     ],
            # ),
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

    async def run_single_qa_task(self, task: QATask) -> QAResult:
        """Run a single QA task: decompose, assign each question to an executor, collect answers and metrics."""
        start_time = asyncio.get_event_loop().time()
        answers = []
        per_question_times = []
        accuracy = 0.0
        scores = []
        try:
            if not self.executors:
                raise RuntimeError("No executors available")
            for idx, q in enumerate(task.questions):
                assigned_executor = self.executors[idx % len(self.executors)]
                prompt = (
                    "[SYSTEM]\nYou must answer ONLY using information from the provided context. "
                    "Do NOT use any outside knowledge. Use the original wording from the context as much as possible.\n"
                    f"[CONTEXT]\n{task.context}\n"
                    f"[QUESTION]\n{q}\n"
                    "[INSTRUCTIONS] Answer the question using only the information from the context above."
                )
                from models.mcp_protocol import Task as MCPTask
                task_obj = MCPTask(
                    task_id=f"{task.task_id}_q{idx+1}",
                    task_type="qa",
                    description=prompt,
                    parameters={}
                )
                console.print(f"\n[bold magenta]Executor {assigned_executor.executor_id} receives prompt:[/bold magenta]\n[white]{task_obj.description}[/white]")
                response = await assigned_executor.execute_task(task_obj)
                console.print("[bold red]\n========== EXECUTOR RAW RESPONSE ==========[/bold red]")
                console.print(f"[bold yellow]{str(response)}[/bold yellow]")
                console.print("[bold red]==========================================\n[/bold red]")
                exec_time = getattr(response, "execution_time", None)
                if exec_time is not None:
                    per_question_times.append(exec_time)
                else:
                    per_question_times.append(0.0)
                ans = ""
                if hasattr(response, "result") and response.result and response.result.get("output"):
                    ans = response.result["output"]
                elif isinstance(response, dict) and response.get("output"):
                    ans = response["output"]
                elif isinstance(response, str):
                    ans = response
                ans = Evaluator().strip_think_sections(ans)
                answers.append(ans if ans else "No answer generated")
            # Accuracy scoring using LLM judge
            if task.expected_answers:
                evaluator = QAEvaluator(self.executors[0])
                questions = task.questions if task.questions else ["" for _ in answers]
                scores = await evaluator.judge_similarity(questions, answers, task.expected_answers)
                accuracy = sum(scores) / len(scores) if scores else 0.0
            else:
                accuracy = 0.0
                scores = [0.0 for _ in answers]
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            answers = await self._simple_generation(task)
            scores = [0.0 for _ in answers]
            accuracy = 0.0
        if len(answers) < len(task.questions):
            answers += ["No answer generated" for _ in range(len(task.questions) - len(answers))]
            scores += [0.0 for _ in range(len(task.questions) - len(scores))]
        elif len(answers) > len(task.questions):
            answers = answers[:len(task.questions)]
            scores = scores[:len(task.questions)]
        exec_time = asyncio.get_event_loop().time() - start_time
        result_obj = QAResult(
            task_id=task.task_id,
            questions=task.questions,
            answers=answers,
            execution_time=exec_time,
            accuracy_score=accuracy,
        )
        result_obj.per_question_times = per_question_times
        result_obj.scores = scores  # Optionally store per-question scores
        return result_obj

    async def run_experiment(self, tasks: Optional[List[QATask]] = None) -> List[QAResult]:
        """Run the complete QA experiment using run_single_qa_task for each task."""
        if tasks is None:
            tasks = self.sample_tasks
        results = []
        for task in tasks:
            try:
                result = await self.run_single_qa_task(task)
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
        # Fallback: use first executor directly to generate answers if available, using MCPTask
        if self.executors:
            executor = self.executors[0]
            answers = []
            from models.mcp_protocol import Task as MCPTask
            for idx, q in enumerate(task.questions):
                prompt = (
                    "[SYSTEM]\nYou must answer ONLY using information from the provided context. "
                    "Do NOT use any outside knowledge. Use the original wording from the context as much as possible.\n"
                    f"[CONTEXT]\n{task.context}\n"
                    f"[QUESTION]\n{q}\n"
                    "[INSTRUCTIONS] Answer the question using only the information from the context above."
                )
                task_obj = MCPTask(
                    task_id=f"{task.task_id}_q{idx+1}_fallback",
                    task_type="qa",
                    description=prompt,
                    parameters={}
                )
                try:
                    response = await executor.execute_task(task_obj)
                    if hasattr(response, "result") and response.result and response.result.get("output"):
                        answers.append(response.result["output"])
                    elif isinstance(response, dict) and response.get("output"):
                        answers.append(response["output"])
                    elif isinstance(response, str):
                        answers.append(response)
                    else:
                        answers.append(str(response))
                except Exception as e:
                    answers.append(f"No answer generated (error: {e})")
            return answers
        return ["No answer generated"] * len(task.questions)

    async def _llm_judge_similarity(self, answers, expected_answers, executor=None):
        # Use LLM to judge if each answer and expected answer are semantically equivalent
        if executor is None:
            executor = self.executors[0]
        results = []
        from models.mcp_protocol import Task as MCPTask
        for idx, (a, e) in enumerate(zip(answers, expected_answers)):
            prompt = (
                f"Given the following QA pair, judge if the model answer is semantically correct and matches the expected answer.\n"
                f"Question: {self.sample_tasks[0].questions[idx] if self.sample_tasks else ''}\n"
                f"Expected Answer: {e}\n"
                f"Model Answer: {a}\n"
                "If the model answer can be considered correct, reply 1. Otherwise, reply 0. Only output the number."
            )
            task_obj = MCPTask(
                task_id=f"judge_{idx+1}",
                task_type="judge",
                description=prompt,
                parameters={}
            )
            response = await executor.execute_task(task_obj)
            print(response)
            score = 0
            if hasattr(response, "result") and response.result and response.result.get("output"):
                out = response.result["output"].strip()
                if out.startswith("1"):
                    score = 1
            elif isinstance(response, str) and response.strip().startswith("1"):
                score = 1
            results.append(score)
        print(results)
        return results

    async def _evaluate_qa(self, answers: List[str], expected: Optional[List[str]], questions: Optional[List[str]] = None) -> float:
        if not expected or not answers:
            return 0.0
        evaluator = QAEvaluator(self.executors[0])
        if questions is None:
            questions = [""] * len(answers)
        scores = await evaluator.judge_similarity(questions, answers, expected)
        return sum(scores) / len(scores) if scores else 0.0

    def _normalize_answer(self, s: str) -> str:
        import re
        return re.sub(r'[^a-z0-9 ]', '', s.strip().lower())

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
