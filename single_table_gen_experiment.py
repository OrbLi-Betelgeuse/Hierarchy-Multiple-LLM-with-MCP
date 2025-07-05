"""
Single Agent Table Generation Experiment
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from difflib import SequenceMatcher
from rich.console import Console
from rich.table import Table as RichTable
from rich.box import SIMPLE
import tracemalloc

from models.executor import Executor
from models.llm_interface import create_llm_interface

logger = logging.getLogger(__name__)
console = Console()

@dataclass
class TableTask:
    id: str
    input_text: str
    required_columns: List[str]
    format_type: str  # "markdown", "csv", "json"
    expected_table: Optional[str] = None

@dataclass
class TableResult:
    task_id: str
    generated_table: str
    execution_time: float
    structure_score: Optional[float] = None
    content_accuracy: Optional[float] = None
    error_messages: Optional[List[str]] = None

class SingleTableGenerationExperiment:
    def __init__(self, executor_config: Dict[str, Any]):
        self.executor_config = executor_config
        self.executor = None
        self.results = []
        self.sample_tasks = [
            {
                "id": "table_1",
                "input_text": "Laptop ($999), Phone ($699), Tablet ($399)",
                "required_columns": ["Product", "Price"],
                "format_type": "markdown",
                "expected_table": (
                    "| Product | Price |\n"
                    "|---------|-------|\n"
                    "| Laptop  | $999  |\n"
                    "| Phone   | $699  |\n"
                    "| Tablet  | $399  |\n"
                ),
                },
                {
                    "id": "table_2",
                    "input_text": "Apple (Fruit), Banana (Fruit), Carrot (Vegetable)",
                    "required_columns": ["Item", "Type"],
                    "format_type": "markdown",
                    "expected_table": (
                        "| Item   | Type      |\n"
                        "|--------|-----------|\n"
                        "| Apple  | Fruit     |\n"
                        "| Banana | Fruit     |\n"
                        "| Carrot | Vegetable |"
                    )
                },
                {
                    "id": "table_3",
                    "input_text": "Python (1991), Java (1995), JavaScript (1995)",
                    "required_columns": ["Language", "Year"],
                    "format_type": "markdown",
                    "expected_table": (
                        "| Language   | Year |\n"
                        "|------------|------|\n"
                        "| Python     | 1991 |\n"
                        "| Java       | 1995 |\n"
                        "| JavaScript | 1995 |"
                    )
            }
        ]


    async def setup(self):
        try:
            executor_llm = create_llm_interface(
                provider=self.executor_config["provider"],
                model_name=self.executor_config["model"],
                **self.executor_config.get("kwargs", {}),
            )
            self.executor = Executor(
                executor_id=self.executor_config["executor_id"],
                llm_interface=executor_llm,
                capabilities=self.executor_config["capabilities"],
            )
            logger.info(f"Initialized single executor: {self.executor_config['executor_id']}")
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

    async def run_experiment(self, wall_time_holder: dict = None) -> List[TableResult]:
        start_wall = time.time()
        # 资源监控开始
        tracemalloc.start()
        cpu_start = os.times()
        results = []
        for task_data in self.sample_tasks:
            task = TableTask(**task_data)
            try:
                result = await self._execute_task(task)
                self._display_task_result(task, result)
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {task.id} - {e}")
                results.append(
                    TableResult(
                        task_id=task.id,
                        generated_table="",
                        execution_time=0,
                        error_messages=[str(e)],
                    )
                )
        self.results = results
        # 资源监控结束
        cpu_end = os.times()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.resource_utilization = {
            "cpu_user_time": cpu_end.user - cpu_start.user,
            "cpu_system_time": cpu_end.system - cpu_start.system,
            "memory_peak_bytes": peak
        }
        if wall_time_holder is not None:
            wall_time_holder["experiment_wall_time"] = time.time() - start_wall
        return results

    async def _execute_task(self, task: TableTask) -> TableResult:
        from models.executor import Task
        start_time = asyncio.get_event_loop().time()
        generated = ""
        error_messages = []
        try:
            prompt = self._build_simplified_prompt(task)
            console.print(f"\n[yellow]DEBUG Prompt Sent:[/yellow]\n{prompt}")
            task_obj = Task(
                task_id=f"exec_{task.id}",
                description=prompt,
                parameters={},
                task_type="table_generation",
            )
            result = await self.executor.execute_task(task_obj)
            # Debug: 打印所有原始返回
            console.print(f"[bold cyan]Executor result (repr):[/bold cyan] {repr(result)}")
            console.print(f"[bold cyan]Executor result (str):[/bold cyan] {str(result)}")
            if hasattr(result, "result"):
                console.print(f"[bold cyan]Executor result.result:[/bold cyan] {repr(result.result)}")
                if isinstance(result.result, dict) and 'output' in result.result:
                    console.print(f"[bold cyan]Executor result.result['output']:[/bold cyan] {result.result['output']}")
                    raw_output = result.result.get("output", "")
                else:
                    raw_output = str(result.result)
            elif isinstance(result, dict) and result.get("output"):
                console.print(f"[bold cyan]Executor dict output:[/bold cyan] {result['output']}")
                raw_output = result.get("output", "")
            else:
                raw_output = str(result) if result else ""
            generated = self._clean_table_output(raw_output)
        except Exception as e:
            error_messages.append(f"Execution error: {str(e)}")
            logger.error(f"Task execution failed: {str(e)}")
            generated = await self._simple_generation(task)
        # Evaluation
        metrics = self._evaluate_table(
            generated=generated,
            expected=task.expected_table,
            columns=task.required_columns,
            format_type=task.format_type,
        )
        metrics["error_messages"] = error_messages + (
            metrics.get("error_messages") or []
        )
        structure_score = metrics.get("structure_score", 0)
        status = "completed" if structure_score and structure_score > 0.5 else "failed"
        metrics["status"] = status
        return TableResult(
            task_id=task.id,
            generated_table=generated,
            execution_time=asyncio.get_event_loop().time() - start_time,
            structure_score=metrics.get("structure_score"),
            content_accuracy=metrics.get("content_accuracy"),
            error_messages=metrics.get("error_messages"),
        )

    async def _simple_generation(self, task: TableTask) -> str:
        try:
            header = f"| {' | '.join(task.required_columns)} |"
            separator = (
                "|"
                + "|".join(["-" * (len(col) + 2) for col in task.required_columns])
                + "|"
            )
            items = [item.strip() for item in task.input_text.split(",")]
            rows = []
            for item in items:
                if "(" in item and ")" in item:
                    product = item.split("(")[0].strip()
                    price = item.split("(")[1].split(")")[0].strip()
                    rows.append(f"| {product} | {price} |")
            return "\n".join([header, separator] + rows)
        except Exception as e:
            logger.error(f"Simple generation failed: {e}")
            return "Table generation failed"

    def _build_simplified_prompt(self, task: TableTask) -> str:
        return (
            f"Create a {task.format_type} table with columns: {', '.join(task.required_columns)}\n"
            f"Data: {task.input_text}\n"
            f"Format: Use proper {task.format_type} table syntax.\n"
            f"Only output the table itself, do not include any explanations, code block markers, or extra text."
        )

    def _clean_table_output(self, text: str) -> str:
        import re
        text = re.sub(r"```[a-zA-Z]*", "", text)
        text = re.sub(r"```", "", text)
        think_match = re.search(r"<think>[\s\S]*?</think>", text, flags=re.IGNORECASE)
        if think_match:
            after_think = text[think_match.end():].strip()
            if after_think:
                text = after_think
            else:
                text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        else:
            text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        table_match = re.search(r"(\|.+\|\n(\|[- :]+\|\n)?(\|.+\|\n?)+)", text)
        if table_match:
            return table_match.group(1).strip()
        return text.strip()

    def _evaluate_table(self, generated: str, expected: Optional[str], columns: List[str], format_type: str) -> Dict[str, Any]:
        metrics = {
            "structure_score": 0.0,
            "content_accuracy": None,
            "error_messages": [],
        }
        generated = self._clean_table_output(generated)
        headers = []
        try:
            if format_type == "markdown":
                lines = [line.strip() for line in generated.split("\n") if line.strip()]
                if len(lines) >= 1:
                    if "|" in lines[0]:
                        headers = [h.strip() for h in lines[0].split("|") if h.strip()]
                    else:
                        headers = [h.strip() for h in lines[0].split() if h.strip()]
                    valid_columns = len(set(headers) & set(columns))
                    metrics["structure_score"] = (
                        valid_columns / len(columns) if columns else 0.0
                    )
        except Exception as e:
            metrics["error_messages"].append(str(e))
        print(f"DEBUG: headers={headers}, columns={columns}")
        if expected:
            try:
                metrics["content_accuracy"] = SequenceMatcher(
                    None,
                    self._normalize_table(generated),
                    self._normalize_table(expected),
                ).ratio()
            except Exception as e:
                metrics["error_messages"].append(f"Content evaluation error: {e}")
        return metrics

    def _normalize_table(self, table: str) -> str:
        return " ".join(line.strip() for line in table.split("\n") if line.strip())

    def _display_task_result(self, task: TableTask, result: TableResult):
        console.print(f"\n[bold]Task {task.id}[/bold]")
        console.print("[blue]Generated:[/blue]")
        self._visualize_table(result.generated_table, task.format_type)
        if task.expected_table:
            console.print("[green]Expected:[/green]")
            self._visualize_table(task.expected_table, task.format_type)
        console.print(
            f"[cyan]Structure Score: {result.structure_score:.2f}[/cyan] | "
            f"[magenta]Time: {result.execution_time:.2f}s[/magenta]"
        )

    def _visualize_table(self, table: str, format_type: str):
        try:
            if format_type == "markdown":
                lines = [line.strip() for line in table.split("\n") if line.strip()]
                if len(lines) >= 2:
                    rich_table = RichTable(box=SIMPLE, show_header=True)
                    headers = [h.strip() for h in lines[0].split("|") if h.strip()]
                    for header in headers:
                        rich_table.add_column(header)
                    for line in lines[2:]:
                        cells = [c.strip() for c in line.split("|") if c.strip()]
                        if len(cells) == len(headers):
                            rich_table.add_row(*cells)
                    console.print(rich_table)
        except Exception as e:
            console.print(f"[red]Visualization error: {e}[/red]")

    def generate_report(self, experiment_wall_time: float = None) -> Dict[str, Any]:
        if not self.results:
            return {"status": "no_results"}
        successful = [
            r for r in self.results if r.structure_score and r.structure_score > 0.5
        ]
        avg_structure = sum(r.structure_score or 0 for r in self.results) / len(self.results)
        ru = getattr(self, "resource_utilization", {})
        resource_utilization = {
            "cpu_user_time": ru.get("cpu_user_time", 0),
            "cpu_system_time": ru.get("cpu_system_time", 0),
            "memory_peak_bytes": ru.get("memory_peak_bytes", 0),
            "token_total": ru.get("token_total", 0),
            "token_per_sec": ru.get("token_per_sec", 0),
        }
        num_executors = max(1, len(getattr(self, 'executors', [])))
        avg_exec_time = sum(r.execution_time for r in self.results) / num_executors if self.results else 0
        return {
            "experiment_type": "table_generation",
            "total_tasks": len(self.results),
            "successful_tasks": len(successful),
            "success_rate": len(successful) / len(self.results) if self.results else 0,
            "average_structure_score": avg_structure,
            "average_execution_time": avg_exec_time,
            "experiment_wall_time": experiment_wall_time,
            "resource_utilization": resource_utilization,
            "detailed_results": [
                {
                    "task_id": r.task_id,
                    "structure_score": r.structure_score,
                    "execution_time": r.execution_time,
                    "status": "success" if (r.structure_score or 0) > 0.5 else "failed",
                }
                for r in self.results
            ],
        }
