"""
single_agent_pipeline.py

Pipeline for running tasks using only a single LLM (Manager) without Executor involvement.
This acts as the baseline for comparison against the Manager-Executor system.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel

from models.llm_interface import create_llm_interface
from models.manager import Manager
from utils.evaluation import Evaluator, PerformanceMonitor
from experiments.summarization_experiment import SummarizationExperiment
from experiments.qa_experiment import QAExperiment
from experiments.table_generation_experiment import TableGenerationExperiment


console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SingleAgentPipeline")


class SingleAgentPipeline:
    def __init__(self):
        self.evaluator = Evaluator()
        self.performance_monitor = PerformanceMonitor()
        self.llm_model = None
        self.output_dir = "results"
        self.executor_llm = None

    async def setup(self):
        console.print(
            Panel("[bold cyan]Initializing Single-Agent (Executor-Only) Pipeline")
        )
        self.executor_llm = create_llm_interface(
            provider="ollama",
            model_name=self.llm_model or "deepseek-r1:7b",
            base_url="http://localhost:11434",
        )

    async def run_task(self, task_name: str):
        start_time = time.time()

        executor_config = {
            "provider": "ollama",
            "model": self.llm_model or "deepseek-r1:7b",
            "executor_id": "solo_executor",
            "capabilities": ["summarization", "qa", "table_generation"],
            "kwargs": {"base_url": "http://localhost:11434"},
        }
        manager_config = None  # No manager

        if task_name == "summarization":
            experiment = SummarizationExperiment(manager_config, [executor_config])
            await experiment.setup()
            results = await experiment.run_experiment()
        elif task_name == "qa":
            experiment = QAExperiment(manager_config, [executor_config])
            await experiment.setup()
            results = await experiment.run_experiment()
        elif task_name == "table_generation":
            experiment = TableGenerationExperiment(manager_config, [executor_config])
            await experiment.setup()
            results = await experiment.run_experiment()
        else:
            raise ValueError(f"Unsupported task: {task_name}")

        # Convert result objects to dicts for evaluation
        results_dicts = []
        for r in results:
            if hasattr(r, "to_dict") and callable(getattr(r, "to_dict")):
                results_dicts.append(r.to_dict())
            else:
                results_dicts.append(vars(r))

        elapsed = time.time() - start_time

        metrics = self.evaluator.calculate_basic_metrics(results_dicts)
        performance = {"time_sec": round(elapsed, 2), "metrics": metrics}

        output_path = Path(f"{self.output_dir}/{task_name}/executor_only_output.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"results": results_dicts, "performance": performance}, f, indent=2
            )

        console.print(Panel(f"[green]Task '{task_name}' completed in {elapsed:.2f}s"))
        return performance


@click.command()
@click.option(
    "--experiment",
    "-e",
    "experiment_type",
    type=click.Choice(["summarization", "qa", "table_generation", "all"]),
    default="all",
    help="Type of experiment to run",
)
@click.option("--manager-model", help="Manager LLM model name")
@click.option("--output-dir", help="Output directory for results")
def main(experiment_type, manager_model, output_dir):
    console.print(Panel.fit("Single-Agent (Executor-Only) Pipeline", style="bold blue"))
    pipeline = SingleAgentPipeline()
    if manager_model:
        pipeline.llm_model = manager_model
    if output_dir:
        pipeline.output_dir = output_dir

    async def run():
        tasks = (
            ["summarization", "qa", "table_generation"]
            if experiment_type == "all"
            else [experiment_type]
        )
        for task in tasks:
            await pipeline.run_task(task)

    asyncio.run(run())


if __name__ == "__main__":
    main()
