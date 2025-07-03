"""
Main Pipeline for Manager-Executor Collaboration System

Orchestrates the complete experimental pipeline for evaluating the hierarchical
Manager-Executor collaboration model on natural language tasks.
"""

import asyncio
import json
import logging
import argparse
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Import our modules
from models.manager import Manager
from models.executor import Executor, SpecializedExecutor
from models.llm_interface import create_llm_interface
from experiments.summarization_experiment import SummarizationExperiment
from experiments.qa_experiment import QAExperiment
from experiments.table_generation_experiment import TableGenerationExperiment
from experiments.rag_enhanced_experiment import RAGEnhancedExperiment
from utils.evaluation import Evaluator, PerformanceMonitor, ExperimentMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiment.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

console = Console()


class ExperimentPipeline:
    """Main pipeline for running Manager-Executor collaboration experiments."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.evaluator = Evaluator()
        self.performance_monitor = PerformanceMonitor()
        self.results = {}

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)

        # Default configuration
        return {
            "manager": {
                "provider": "ollama",
                "model": "deepseek-r1:7b",
                "manager_id": "manager_01",
                "kwargs": {"base_url": "http://localhost:11434"},
            },
            "executors": [
                # {
                #     "provider": "ollama",
                #     "model": "llama2:7b",
                #     "executor_id": "executor_01",
                #     "capabilities": ["summarization", "general"],
                #     "specialized": False,
                #     "kwargs": {"base_url": "http://localhost:11434"},
                # },
                # {
                #     "provider": "ollama",
                #     "model": "llama2:7b",
                #     "executor_id": "executor_02",
                #     "capabilities": ["question_answering", "general"],
                #     "specialized": False,
                #     "kwargs": {"base_url": "http://localhost:11434"},
                # },
                # {
                #     "provider": "ollama",
                #     "model": "llama2:7b",
                #     "executor_id": "executor_03",
                #     "capabilities": ["table_generation", "general"],
                #     "specialized": False,
                #     "kwargs": {"base_url": "http://localhost:11434"},
                # },
                {
                    "provider": "ollama",
                    "model": "deepseekr1:1.5b",
                    "executor_id": "executor_04",
                    "capabilities": ["question_answering", "general"],
                    "specialized": False,
                    "kwargs": {"base_url": "http://localhost:11435"},
                },{
                    "provider": "ollama",
                    "model": "deepseekr1:1.5b",
                    "executor_id": "executor_05",
                    "capabilities": ["question_answering", "general"],
                    "specialized": False,
                    "kwargs": {"base_url": "http://localhost:11436"},
                },
            ],
            "experiments": {
                "summarization": {"enabled": True, "num_tasks": 2},
                "question_answering": {"enabled": True, "num_tasks": 2},
                "table_generation": {"enabled": True, "num_tasks": 2},
                "rag_enhanced": {"enabled": True, "num_tasks": 3},
            },
            "output": {
                "results_dir": "results",
                "export_metrics": True,
                "generate_visualizations": False,
            },
        }

    async def setup_system(self):
        """Setup the Manager-Executor system."""
        console.print(
            Panel.fit("Setting up Manager-Executor System", style="bold blue")
        )

        try:
            # Create manager
            manager_config = self.config["manager"]
            manager_llm = create_llm_interface(
                provider=manager_config["provider"],
                model_name=manager_config["model"],
                **manager_config.get("kwargs", {}),
            )

            self.manager = Manager(
                manager_id=manager_config["manager_id"], llm_interface=manager_llm
            )

            # Create executors
            self.executors = []
            for config in self.config["executors"]:
                executor_llm = create_llm_interface(
                    provider=config["provider"],
                    model_name=config["model"],
                    **config.get("kwargs", {}),
                )

                if config.get("specialized", False):
                    executor = SpecializedExecutor(
                        executor_id=config["executor_id"],
                        llm_interface=executor_llm,
                        task_type=config.get("task_type", "general"),
                        specialized_capabilities=config["capabilities"],
                    )
                else:
                    executor = Executor(
                        executor_id=config["executor_id"],
                        llm_interface=executor_llm,
                        capabilities=config["capabilities"],
                    )

                self.executors.append(executor)

                # Register executor with manager
                await self.manager.register_executor(
                    executor_id=config["executor_id"],
                    capabilities=config["capabilities"],
                )

            console.print(
                f"✅ System setup complete: 1 manager, {len(self.executors)} executors"
            )

        except Exception as e:
            console.print(f"❌ Error setting up system: {e}", style="bold red")
            raise

    async def run_summarization_experiment(self) -> Dict[str, Any]:
        """Run the summarization experiment."""
        console.print(Panel.fit("Running Summarization Experiment", style="bold green"))

        experiment = SummarizationExperiment(
            manager_config=self.config["manager"],
            executor_configs=self.config["executors"],
        )

        await experiment.setup()
        results = await experiment.run_experiment()
        report = experiment.generate_report()

        console.print(f"✅ Summarization experiment completed: {len(results)} tasks")
        return report

    async def run_qa_experiment(self) -> Dict[str, Any]:
        """Run the question answering experiment."""
        console.print(
            Panel.fit("Running Question Answering Experiment", style="bold green")
        )

        experiment = QAExperiment(
            manager_config=self.config["manager"],
            executor_configs=self.config["executors"],
        )

        await experiment.setup()
        results = await experiment.run_experiment()
        report = experiment.generate_report()

        console.print(f"✅ QA experiment completed: {len(results)} tasks")
        return report

    async def run_table_generation_experiment(self) -> Dict[str, Any]:
        """Run the table generation experiment."""
        console.print(
            Panel.fit("Running Table Generation Experiment", style="bold green")
        )

        experiment = TableGenerationExperiment(
            manager_config=self.config["manager"],
            executor_configs=self.config["executors"],
        )

        await experiment.setup()
        results = await experiment.run_experiment()
        report = experiment.generate_report()

        console.print(f"✅ Table generation experiment completed: {len(results)} tasks")
        if "success" in report:
            print(f"Table generation success: {report['success']}")
        elif "overall_status" in report:
            print(f"Table generation status: {report['overall_status']}")
        else:
            print("Table generation result status unknown.")
        return report

    async def run_rag_enhanced_experiment(self) -> Dict[str, Any]:
        """Run the enhanced RAG experiment (Experiment D)."""
        console.print(
            Panel.fit(
                "Running Enhanced RAG Experiment (Experiment D)", style="bold blue"
            )
        )

        # RAGFlow configuration
        ragflow_config = {"base_url": "http://localhost:9380", "kwargs": {}}

        experiment = RAGEnhancedExperiment(
            manager_config=self.config["manager"],
            ragflow_config=ragflow_config,
        )

        await experiment.setup()
        results = await experiment.run_experiment()
        report = experiment.generate_report()

        # Display summary table
        experiment.display_summary_table()

        console.print(
            f"✅ Enhanced RAG experiment completed: {len(results)} comparisons"
        )
        return report

    async def run_single_experiment(self, experiment_type: str) -> Dict[str, Any]:
        """Run a single experiment."""
        await self.setup_system()
        console.print(
            Panel.fit(
                f"Running {experiment_type.title()} Experiment", style="bold magenta"
            )
        )

        self.performance_monitor.start_monitoring()

        try:
            if experiment_type == "summarization":
                result = await self.run_summarization_experiment()
            elif experiment_type == "qa":
                result = await self.run_qa_experiment()
            elif experiment_type == "table":
                result = await self.run_table_generation_experiment()
            elif experiment_type == "rag":
                result = await self.run_rag_enhanced_experiment()
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")

            self.results[experiment_type] = result

            # Generate report
            report = self.generate_comprehensive_report()

            # Display results
            self.display_results(report)

            # Export results
            self.export_results(report)

            console.print(
                Panel.fit(
                    f"{experiment_type.title()} Experiment Completed Successfully! 🎉",
                    style="bold green",
                )
            )

            return result

        except Exception as e:
            console.print(
                f"❌ {experiment_type} experiment failed: {e}", style="bold red"
            )
            logger.error(f"{experiment_type} experiment failed: {e}")
            raise
        finally:
            self.performance_monitor.stop_monitoring()

    async def run_all_experiments(self) -> Dict[str, Any]:
        """Run all enabled experiments."""
        console.print(Panel.fit("Starting All Experiments", style="bold magenta"))

        self.performance_monitor.start_monitoring()

        experiments_config = self.config["experiments"]
        all_results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Summarization experiment
            if experiments_config["summarization"]["enabled"]:
                task = progress.add_task(
                    "Running summarization experiment...", total=None
                )
                try:
                    result = await self.run_summarization_experiment()
                    all_results["summarization"] = result
                    progress.update(task, description="✅ Summarization completed")
                except Exception as e:
                    progress.update(task, description=f"❌ Summarization failed: {e}")
                    logger.error(f"Summarization experiment failed: {e}")

            # Question answering experiment
            if experiments_config["question_answering"]["enabled"]:
                task = progress.add_task("Running QA experiment...", total=None)
                try:
                    result = await self.run_qa_experiment()
                    all_results["question_answering"] = result
                    progress.update(task, description="✅ QA completed")
                except Exception as e:
                    progress.update(task, description=f"❌ QA failed: {e}")
                    logger.error(f"QA experiment failed: {e}")

            # Table generation experiment
            if experiments_config["table_generation"]["enabled"]:
                task = progress.add_task(
                    "Running table generation experiment...", total=None
                )
                try:
                    result = await self.run_table_generation_experiment()
                    all_results["table_generation"] = result
                    progress.update(task, description="✅ Table generation completed")
                except Exception as e:
                    progress.update(
                        task, description=f"❌ Table generation failed: {e}"
                    )
                    logger.error(f"Table generation experiment failed: {e}")

            # RAG enhanced experiment
            if experiments_config.get("rag_enhanced", {}).get("enabled", False):
                task = progress.add_task(
                    "Running enhanced RAG experiment...", total=None
                )
                try:
                    result = await self.run_rag_enhanced_experiment()
                    all_results["rag_enhanced"] = result
                    progress.update(task, description="✅ Enhanced RAG completed")
                except Exception as e:
                    progress.update(task, description=f"❌ Enhanced RAG failed: {e}")
                    logger.error(f"Enhanced RAG experiment failed: {e}")

        self.performance_monitor.stop_monitoring()
        self.results = all_results

        return all_results

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of all experiments."""
        console.print(Panel.fit("Generating Comprehensive Report", style="bold yellow"))

        # Performance monitoring summary
        performance_summary = self.performance_monitor.get_summary()

        # Evaluate each experiment
        evaluation_results = {}
        for experiment_type, results in self.results.items():
            # Use the experiment's own report if it has the expected format
            if isinstance(results, dict) and "success_rate" in results:
                # Convert the experiment report to ExperimentMetrics format
                metrics = ExperimentMetrics(
                    experiment_type=results.get("experiment_type", experiment_type),
                    total_tasks=results.get("total_tasks", 0),
                    successful_tasks=results.get("successful_tasks", 0),
                    success_rate=results.get("success_rate", 0.0),
                    average_execution_time=results.get("average_execution_time", 0.0),
                    total_execution_time=results.get("total_execution_time", 0.0),
                    manager_performance=results.get("manager_performance", {}),
                    executor_performance=results.get("executor_performance", {}),
                    quality_metrics=results.get("quality_metrics", {}),
                    resource_utilization=results.get("resource_utilization", {}),
                )
            else:
                # Fallback to evaluator for other formats
                metrics = self.evaluator.generate_comprehensive_report(results)
            evaluation_results[experiment_type] = metrics

        # Overall system metrics
        total_tasks = sum(
            results.get("total_tasks", 0) for results in self.results.values()
        )
        total_successful = sum(
            results.get("successful_tasks", 0) for results in self.results.values()
        )
        overall_success_rate = (
            total_successful / total_tasks if total_tasks > 0 else 0.0
        )

        comprehensive_report = {
            "experiment_summary": {
                "total_experiments": len(self.results),
                "total_tasks": total_tasks,
                "total_successful_tasks": total_successful,
                "overall_success_rate": overall_success_rate,
            },
            "performance_monitoring": performance_summary,
            "experiment_results": evaluation_results,
            "system_configuration": {
                "manager": self.config["manager"],
                "executors": [executor.executor_id for executor in self.executors],
            },
        }

        return comprehensive_report

    def display_results(self, report: Dict[str, Any]):
        """Display results in a formatted table."""
        console.print(Panel.fit("Experiment Results Summary", style="bold cyan"))

        # Create summary table
        table = Table(title="Experiment Results")
        table.add_column("Experiment Type", style="cyan", no_wrap=True)
        table.add_column("Tasks", justify="right", style="green")
        table.add_column("Success Rate", justify="right", style="green")
        table.add_column("Avg Time (s)", justify="right", style="yellow")
        table.add_column("Status", style="bold")

        for experiment_type, results in report["experiment_results"].items():
            table.add_row(
                experiment_type.replace("_", " ").title(),
                str(results.total_tasks),
                f"{results.success_rate:.2%}",
                f"{results.average_execution_time:.2f}",
                (
                    "✅"
                    if results.success_rate > 0.8
                    else "⚠️" if results.success_rate > 0.5 else "❌"
                ),
            )

        console.print(table)

        # Display overall metrics
        summary = report["experiment_summary"]
        console.print(f"\n📊 Overall Performance:")
        console.print(f"   • Total Tasks: {summary['total_tasks']}")
        console.print(f"   • Success Rate: {summary['overall_success_rate']:.2%}")
        console.print(f"   • Experiments Completed: {summary['total_experiments']}")

    def export_results(self, report: Dict[str, Any]):
        """Export results to files."""
        output_config = self.config["output"]
        results_dir = Path(output_config["results_dir"])
        results_dir.mkdir(exist_ok=True)

        if output_config["export_metrics"]:
            # Export comprehensive report
            report_file = results_dir / "comprehensive_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

            # Export individual experiment metrics
            for experiment_type, metrics in report["experiment_results"].items():
                metrics_file = results_dir / f"{experiment_type}_metrics.json"
                self.evaluator.export_metrics(metrics, str(metrics_file))

            console.print(f"📁 Results exported to {results_dir}")

    async def cleanup(self):
        """Cleanup resources and stop executors."""
        console.print("🧹 Cleaning up resources...")

        try:
            # Stop all executors
            for executor in self.executors:
                await executor.stop()
                console.print(f"✅ Stopped executor: {executor.executor_id}")

        except Exception as e:
            console.print(f"⚠️ Error during cleanup: {e}", style="yellow")

    async def run_pipeline(self):
        """Run the complete experimental pipeline."""
        try:
            console.print(Panel.fit("Starting Experiment Pipeline", style="bold green"))

            # Setup system
            await self.setup_system()

            # Run experiments based on configuration
            if self.config["experiments"]["summarization"]["enabled"]:
                self.results["summarization"] = (
                    await self.run_summarization_experiment()
                )

            if self.config["experiments"]["question_answering"]["enabled"]:
                self.results["question_answering"] = await self.run_qa_experiment()

            if self.config["experiments"]["table_generation"]["enabled"]:
                self.results["table_generation"] = (
                    await self.run_table_generation_experiment()
                )

            # Generate comprehensive report
            report = self.generate_comprehensive_report()

            # Display results
            self.display_results(report)

            # Export results
            self.export_results(report)

            console.print(
                Panel.fit("Pipeline Completed Successfully", style="bold green")
            )

        except Exception as e:
            console.print(f"❌ Pipeline failed: {e}", style="bold red")
            logger.error(f"Pipeline error: {e}", exc_info=True)
            raise
        finally:
            # Always cleanup
            await self.cleanup()


@click.command()
@click.option("--config", "-c", "config_path", help="Path to configuration file")
@click.option(
    "--experiment",
    "-e",
    "experiment_type",
    type=click.Choice(["summarization", "qa", "table", "rag", "all"]),
    default="all",
    help="Type of experiment to run",
)
@click.option("--manager-model", help="Manager LLM model name")
@click.option("--executor-model", help="Executor LLM model name")
@click.option("--output-dir", help="Output directory for results")
def main(config_path, experiment_type, manager_model, executor_model, output_dir):
    """Manager-Executor Collaboration System Pipeline."""

    console.print(
        Panel.fit(
            "Manager-Executor Collaboration System\n"
            "Hierarchical Multi-LLM Task Processing",
            style="bold blue",
        )
    )

    # Create pipeline
    pipeline = ExperimentPipeline(config_path)

    # Override config if command line options provided
    if manager_model:
        pipeline.config["manager"]["model"] = manager_model
    if executor_model:
        for executor_config in pipeline.config["executors"]:
            executor_config["model"] = executor_model
    if output_dir:
        pipeline.config["output"]["results_dir"] = output_dir

    # Run specific experiment or all
    if experiment_type == "all":
        asyncio.run(pipeline.run_pipeline())
    else:
        # Run single experiment
        asyncio.run(pipeline.run_single_experiment(experiment_type))


if __name__ == "__main__":
    main()
