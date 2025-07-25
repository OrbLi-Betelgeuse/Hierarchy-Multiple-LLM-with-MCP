"""
Enhanced Table Generation Experiment with Robust Error Handling
"""

import asyncio
import json
import logging
import os
import time
import tracemalloc
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from difflib import SequenceMatcher
from rich.console import Console
from rich.table import Table as RichTable
from rich.box import SIMPLE

from models.executor import Executor
from models.llm_interface import create_llm_interface
from models.manager import Manager

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


class TableGenerationExperiment:
    def __init__(
        self, manager_config: Dict[str, Any], executor_configs: List[Dict[str, Any]]
    ):
        self.manager_config = manager_config
        self.executor_configs = executor_configs
        self.manager = None
        self.executors = []
        self.results = []

        # Enhanced sample task with simpler description
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
                    manager_id=self.manager_config["manager_id"],
                    llm_interface=manager_llm,
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
                    # Ensure executor instance is accessible to manager
                    if not hasattr(self.manager, "executor_instances"):
                        self.manager.executor_instances = {}
                    self.manager.executor_instances[config["executor_id"]] = executor

            logger.info(
                f"Initialized: {('1 manager, ' if self.manager else '0 manager, ')}{len(self.executors)} executors"
            )

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

    async def run_experiment(self, wall_time_holder: dict = None) -> List[TableResult]:
        start_wall = time.time()
        # 资源监控开始
        tracemalloc.start()
        cpu_start = os.times()
        async def run_one(task_data):
            task = TableTask(**task_data)
            try:
                result = await self._execute_with_fallback(task)
                self._display_task_result(task, result)
                return result
            except Exception as e:
                logger.error(f"Task failed: {task.id} - {e}")
                return TableResult(
                    task_id=task.id,
                    generated_table="",
                    execution_time=0,
                    error_messages=[str(e)],
                )
        results = await asyncio.gather(*(run_one(td) for td in self.sample_tasks))
        self.results = results
        # 资源监控结束
        cpu_end = os.times()
        try:
            cpu_user_time = cpu_end.user - cpu_start.user
        except Exception:
            import time as _time
            cpu_user_time = _time.process_time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.resource_utilization = {
            "cpu_user_time": cpu_user_time,
            "cpu_system_time": cpu_end.system - cpu_start.system if hasattr(cpu_end, 'system') else 0,
            "memory_peak_bytes": peak
        }
        # 汇总token统计
        token_total = 0
        if hasattr(self, "manager") and hasattr(self.manager, "llm_interface"):
            token_total += getattr(self.manager.llm_interface, "token_counter", 0)
        if hasattr(self, "executors"):
            for exe in self.executors:
                if hasattr(exe, "llm_interface"):
                    token_total += getattr(exe.llm_interface, "token_counter", 0)
        duration = wall_time_holder["experiment_wall_time"] if wall_time_holder and "experiment_wall_time" in wall_time_holder else (time.time() - start_wall)
        self.resource_utilization["token_total"] = token_total
        self.resource_utilization["token_per_sec"] = token_total / duration if duration > 0 else 0
        if wall_time_holder is not None:
            wall_time_holder["experiment_wall_time"] = time.time() - start_wall
        return results

    async def _execute_with_fallback(self, task: TableTask) -> TableResult:
        """Execute task with multiple fallback strategies, supporting executor-only mode."""
        start_time = asyncio.get_event_loop().time()
        generated = ""
        error_messages = []
        exec_time = None  # Will be set from LLM/executor if available

        try:
            prompt = self._build_simplified_prompt(task)
            console.print(f"\n[yellow]DEBUG Prompt Sent:[/yellow]\n{prompt}")

            if self.manager is None:
                # Executor-only mode: send the whole task to the first executor
                if not self.executors:
                    raise RuntimeError("No executors available")
                executor = self.executors[0]
                from models.executor import Task

                task_obj = Task(
                    task_id=f"exec_{task.id}",
                    description=prompt,
                    parameters={},
                )
                result = await executor.execute_task(task_obj)
                if hasattr(result, "result"):
                    generated = result.result.get("output", "")
                    exec_time = result.result.get("execution_time")
                elif isinstance(result, dict):
                    generated = result.get("output", "")
                    exec_time = result.get("execution_time")
                else:
                    generated = str(result) if result else ""
            else:
                # Full manager-executor flow
                response = await self.manager.execute_task(
                    task_description=prompt,
                    task_type="general",
                )
                console.print(
                    f"[yellow]DEBUG Raw Response:[/yellow]\n{json.dumps(response, indent=2)}"
                )
                if response:
                    if response.get("output"):
                        generated = response["output"]
                        exec_time = response.get("execution_time")
                    elif response.get("task_results"):
                        task_results = response["task_results"]
                        if task_results:
                            first_result = list(task_results.values())[0]
                            generated = first_result.get("output", "")
                            exec_time = first_result.get("execution_time")
                        else:
                            generated = ""
                    elif response.get("summary"):
                        generated = str(response.get("summary", ""))
                    else:
                        generated = str(response)
                else:
                    error_messages.append("Empty response from manager")
                    logger.warning("Empty response from manager.execute_task()")
                    generated = await self._direct_executor_execution(task)

        except Exception as e:
            error_messages.append(f"Execution error: {str(e)}")
            logger.error(f"Task execution failed: {str(e)}")
            generated = await self._simple_generation(task)

        # 清洗表格输出，确保只返回表格内容
        generated = self._clean_table_output(generated)

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
        # Add status field for evaluator compatibility
        structure_score = metrics.get("structure_score", 0)
        status = "completed" if structure_score and structure_score > 0.5 else "failed"
        metrics["status"] = status

        # Use LLM/executor-reported execution time if available, else fallback to wall time
        if exec_time is not None:
            try:
                exec_time = float(exec_time)
            except Exception:
                exec_time = None
        final_exec_time = exec_time if exec_time is not None else (asyncio.get_event_loop().time() - start_time)

        return TableResult(
            task_id=task.id,
            generated_table=generated,
            execution_time=final_exec_time,
            structure_score=metrics.get("structure_score"),
            content_accuracy=metrics.get("content_accuracy"),
            error_messages=metrics.get("error_messages"),
            # status is not a dataclass field, but will be present in dict for evaluator
        )

    async def _direct_executor_execution(self, task: TableTask) -> str:
        """Direct executor fallback"""
        if not self.executors:
            raise RuntimeError("No executors available")

        try:
            executor = self.executors[0]
            # Create a proper task object that matches the expected format
            from models.executor import Task

            task_obj = Task(
                task_id=f"direct_{task.id}",
                description=self._build_simplified_prompt(task),
                parameters={},
                task_type="table_generation",
            )
            result = await executor.execute_task(task_obj)
            # Handle different response formats
            if hasattr(result, "result"):
                raw_output = result.result.get("output", "")
            elif isinstance(result, dict):
                raw_output = result.get("output", "")
            else:
                raw_output = str(result) if result else ""
            # 只返回清洗后的表格
            return self._clean_table_output(raw_output)
        except Exception as e:
            logger.error(f"Direct executor execution failed: {e}")
            raise

    async def _simple_generation(self, task: TableTask) -> str:
        try:
            # Basic markdown table generation
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
            table = "\n".join([header, separator] + rows)
            # 只返回清洗后的表格
            return self._clean_table_output(table)
        except Exception as e:
            logger.error(f"Simple generation failed: {e}")
            return "Table generation failed"

    def _build_simplified_prompt(self, task: TableTask) -> str:
        """Simplified prompt for better parsing, force LLM to only output table."""
        return (
            f"Create a {task.format_type} table with columns: {', '.join(task.required_columns)}\n"
            f"Data: {task.input_text}\n"
            f"Format: Use proper {task.format_type} table syntax.\n"
            f"Only output the table itself, do not include any explanations, code block markers, or extra text."
        )

    def _clean_table_output(self, text: str) -> str:
        """Remove <think>...</think> and code block markers from LLM output. Extract table if present after <think>."""
        import re

        # Remove code block markers (``` and ```markdown)
        text = re.sub(r"```[a-zA-Z]*", "", text)
        text = re.sub(r"```", "", text)
        # If <think>...</think> present, extract content after it
        think_match = re.search(r"<think>[\s\S]*?</think>", text, flags=re.IGNORECASE)
        if think_match:
            after_think = text[think_match.end():].strip()
            if after_think:
                text = after_think
            else:
                # If nothing after <think>, fallback to removing <think>...</think>
                text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        else:
            text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        # Fallback: if still no table, try to extract the first markdown table
        table_match = re.search(r"(\|.+\|\n(\|[- :]+\|\n)?(\|.+\|\n?)+)", text)
        if table_match:
            return table_match.group(1).strip()
        return text.strip()

    def _evaluate_table(
        self,
        generated: str,
        expected: Optional[str],
        columns: List[str],
        format_type: str,
    ) -> Dict[str, Any]:
        """Enhanced evaluation with pipeline-compatible metrics. Expects markdown table format: | header | header | ... |"""
        metrics = {
            "structure_score": 0.0,
            "content_accuracy": None,
            "error_messages": [],
        }

        # Clean generated output and extract table
        generated = self._clean_table_output(generated)

        # Structure evaluation
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

        # Content evaluation
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
        """Normalization for comparison."""
        return " ".join(line.strip() for line in table.split("\n") if line.strip())

    def _display_task_result(self, task: TableTask, result: TableResult):
        """Rich display of task results."""
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
        """Rich visualization for tables."""
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
        # Collect all per-task execution times
        execution_times = [r.execution_time for r in self.results]
        avg_exec_time = sum(execution_times) / len(execution_times) if execution_times else 0
        return {
            "experiment_type": "table_generation",
            "total_tasks": len(self.results),
            "successful_tasks": len(successful),
            "success_rate": len(successful) / len(self.results) if self.results else 0,
            "average_structure_score": avg_structure,
            "average_execution_time": avg_exec_time,
            "execution_times": execution_times,
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
