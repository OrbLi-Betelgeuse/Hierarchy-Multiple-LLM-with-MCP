"""
请把这个py文件放在和pipeline.py相同的路径下
这个Task_B_Pipeline是根据pipeline.py修改的，针对于TaskB的变种代码
修改的大部分在run_task_b_pipeline()中
代码整体存在标红的部分，但是不影响测试运行
代码思路为：拿到提出的问题（TODO:需要改成从文件读入，数据集和Summary实验的最好一致）
之后丢给Pudge模型切片成三个不同的小问题，传给子处理模型（sub_executors）
最终重用Pudge来对结果进行整合并输出

使用的评价指标为之前写好的utils/evaluation.py，运行结果大部分保存在results/comprehensive_report.json中（TODO：剩下的动态route results和metrics不知道有什么必要，暂时还没写）

小型实验代码在pudge_experiment.py task_solver_experiment.py 和 test.py中，可具体查看
TODO：需要查看evaluate中的expected result，我好像没有找到正确的比对方法。具体可以查看experiments/summarization_experiment.py的比对方法，让GPT迁移到QA问题上，查看具体的结果。
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
from utils.evaluation import Evaluator, PerformanceMonitor
from pprint import pprint
import time

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
              "kwargs": {"base_url": "http://localhost:11435"}
            },
            "executors": [
                {
                    "provider": "ollama",
                    "model": "deepseek-r1:1.5b",
                    "executor_id": "executor1",
                    "kwargs": {"base_url": "http://localhost:11434"}
                },
                {
                    "provider": "ollama",
                    "model": "deepseek-r1:1.5b",
                    "executor_id": "executor2",
                    "kwargs": {"base_url": "http://localhost:11436"}
                # },
                # {
                #     "provider": "ollama",
                #     "model": "qwen2.5:14b",
                #     "executor_id": "executor2",
                #     "kwargs": {"base_url": "http://localhost:11434"}
                # },
                # {
                #     "provider": "ollama",
                #     "model": "qwen2.5:14b",
                #     "executor_id": "executor3",
                #     "kwargs": {"base_url": "http://localhost:11434"}
                }
            ],
            "experiments": {
                "summarization": {"enabled": False, "num_tasks": 2},
                "question_answering": {"enabled": True, "num_tasks": 2},
                "table_generation": {"enabled": False, "num_tasks": 2},
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

            # Store executor references in manager for direct access
            self.manager.executor_instances = {
                executor.executor_id: executor for executor in self.executors
            }

            console.print(
                f"✅ System setup complete: 1 manager, {len(self.executors)} executors"
            )

        except Exception as e:
            console.print(f"❌ Error setting up system: {e}", style="bold red")
            raise

    async def run_task_b_pipeline(self, input_text: str) -> str:
        """运行基于 TaskB 架构的任务处理流程"""
        self.performance_monitor.start_monitoring()  # ✅ 开始监控
        try:
            # 初始化 pudge 模型为 decomposer
            decomposer_cfg = self.config["decomposer"]
            pudge_llm = create_llm_interface(
                provider=decomposer_cfg["provider"],
                model_name=decomposer_cfg["model"],
                **decomposer_cfg.get("kwargs", {})
            )

            # 初始化三个 executor 模型（20B）
            sub_executors = []
            for executor_cfg in self.config["executors"]:
                llm = create_llm_interface(
                    provider=executor_cfg["provider"],
                    model_name=executor_cfg["model"],
                    **executor_cfg.get("kwargs", {})
                )
                sub_executors.append(Executor(
                    executor_id=executor_cfg["executor_id"],
                    llm_interface=llm,
                    capabilities=[]
                ))

            # 步骤1：用 pudge 分解任务
            prompt = (
                f"请将下面这个问题拆分成三个独立的步骤，只返回一个 JSON 数组，例如："
                f'["步骤1", "步骤2", "步骤3"]。\n\n任务如下：{input_text}'
            )

            response = await pudge_llm.generate(prompt)

            # 提取字符串内容
            if isinstance(response, dict):
                content = response.get("content", "")
            else:
                content = getattr(response, "content", str(response))

            # 打印内容调试（推荐保留）
            print("🧪 Pudge 返回内容如下 ↓↓↓")
            print(content)
            print("⛓️ 尝试将其解析为 JSON...")

            # 判断是否为空或空格
            if not isinstance(content, str) or not content.strip():
                raise ValueError("❌ Pudge 模型返回为空，无法解析为步骤")

            # 尝试解析为 JSON
            try:
                steps = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"❌ JSON 解析失败，模型返回内容如下：\n{content}") from e

            # 打印查看是否返回了内容
            if not content.strip():
                raise ValueError("Pudge 模型返回为空，无法解析为步骤")

            try:
                steps = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"❌ JSON 解析失败，模型返回内容为：{content}")
                raise e

            # 步骤2：三个模型并行处理各自步骤
            sub_results = await asyncio.gather(*[
                executor.run_step(steps[i]) for i, executor in enumerate(sub_executors)
            ])

            # 步骤3：将结果交还给 pudge 进行整合
            merge_prompt = f"请根据以下三个结果整合成一个完整答案：{json.dumps(sub_results, ensure_ascii=False)}"
            print(f"请根据以下三个结果整合成一个完整答案：{json.dumps(sub_results, ensure_ascii=False)}")
            final_response = await pudge_llm.generate(merge_prompt)

            # 取返回内容
            if isinstance(final_response, dict):
                final_response_text = final_response.get("content", "")
            else:
                final_response_text = getattr(final_response, "content", str(final_response))

            # 打印结果
            print(final_response_text)

            # ✅ 记录总耗时
            execution_time = time.time() - self.performance_monitor.start_time

            # ✅ 提取模型最终结果（兼容不同格式）
            final_output = getattr(final_response, "content", None)
            if not final_output:
                final_output = final_response if isinstance(final_response, str) else str(final_response)

            # ✅ 构造 evaluation 能识别的结果结构
            self.results["taskb"] = {
                "experiment_type": "taskb",
                "total_tasks": 1,
                "successful_tasks": 1,
                "execution_times": [execution_time],
                "detailed_results": [
                    {
                        "status": "completed",
                        "execution_time": execution_time,
                        "output": final_output
                    }
                ],
                "manager_metrics": {
                    "model": self.config.get("decomposer", {}).get("model", "unknown"),
                    "task": "task decomposition and final synthesis"
                },
                "executor_metrics": {
                    executor.executor_id: {
                        "model": executor.llm_interface.model_name if hasattr(executor.llm_interface,
                                                                              'model_name') else "unknown",
                        "task": f"subtask-{i + 1}"
                    }
                    for i, executor in enumerate(sub_executors)
                }
            }

            # ✅ 生成综合报告
            report = self.generate_comprehensive_report()

            # ✅ 展示结果
            self.display_results(report)

            # ✅ 导出结果
            self.export_results(report)

            # ✅ 返回结果（用于 CLI 显示等用途）
            return final_output

        except Exception as e:
            logger.error(f"TaskB 失败: {e}", exc_info=True)
            raise

        finally:
            self.performance_monitor.stop_monitoring()  # ✅ 停止监控

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

    # async def run_all_experiments(self) -> Dict[str, Any]:
    #     """Run all enabled experiments."""
    #     console.print(Panel.fit("Starting All Experiments", style="bold magenta"))

    #     self.performance_monitor.start_monitoring()

    #     all_results = {}

    #     with Progress(
    #         SpinnerColumn(),
    #         TextColumn("[progress.description]{task.description}"),
    #         console=console,
    #     ) as progress:

    #         # Summarization experiment
    #         if experiments_config["summarization"]["enabled"]:
    #             task = progress.add_task(
    #                 "Running summarization experiment...", total=None
    #             )
    #             try:
    #                 result = await self.run_summarization_experiment()
    #                 all_results["summarization"] = result
    #                 progress.update(task, description="✅ Summarization completed")
    #             except Exception as e:
    #                 progress.update(task, description=f"❌ Summarization failed: {e}")
    #                 logger.error(f"Summarization experiment failed: {e}")

    #         # Question answering experiment
    #         if experiments_config["question_answering"]["enabled"]:
    #             task = progress.add_task("Running QA experiment...", total=None)
    #             try:
    #                 result = await self.run_qa_experiment()
    #                 all_results["question_answering"] = result
    #                 progress.update(task, description="✅ QA completed")
    #             except Exception as e:
    #                 progress.update(task, description=f"❌ QA failed: {e}")
    #                 logger.error(f"QA experiment failed: {e}")

    #         # Table generation experiment
    #         if experiments_config["table_generation"]["enabled"]:
    #             task = progress.add_task(
    #                 "Running table generation experiment...", total=None
    #             )
    #             try:
    #                 result = await self.run_table_generation_experiment()
    #                 all_results["table_generation"] = result
    #                 progress.update(task, description="✅ Table generation completed")
    #             except Exception as e:
    #                 progress.update(
    #                     task, description=f"❌ Table generation failed: {e}"
    #                 )
    #                 logger.error(f"Table generation experiment failed: {e}")

    #     self.performance_monitor.stop_monitoring()
    #     self.results = all_results

    #     return all_results

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of all experiments."""
        console.print(Panel.fit("Generating Comprehensive Report", style="bold yellow"))

        # Performance monitoring summary
        performance_summary = self.performance_monitor.get_summary()

        # Evaluate each experiment
        # Evaluate each experiment
        evaluation_results = {}
        for experiment_type, results in self.results.items():
            metrics = self.evaluator.generate_comprehensive_report(results)
            evaluation_results[experiment_type] = metrics

            # 如果是 TaskB，则手动转换为 dict（否则 json.dump 会报错）
            if experiment_type == "taskb":
                evaluation_results[experiment_type] = metrics.__dict__

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
                "manager": self.config.get("manager", {"model": "N/A"}),
                "executors": [executor.executor_id for executor in getattr(self, "executors", [])],
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
                str(results["total_tasks"]),
                f"{results['success_rate']:.2%}",
                f"{results['average_execution_time']:.2f}",
                (
                    "✅"
                    if results.get("success_rate", 0.0) > 0.8
                    else "⚠️" if results["success_rate"] > 0.5 else "❌"
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
    type=click.Choice(["summarization", "qa", "table", "all", "taskb"]),
    default="taskb",
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
    elif experiment_type == "taskb":
        # input_text = input("请输入问题描述：")
        # TODO: 需要改成从文件读入，数据集和Summary实验的最好一致
        input_text="请帮助我分析中国人口老龄化带来的社会问题，并提出应对策略。"
        asyncio.run(pipeline.run_task_b_pipeline(input_text))
    else:
        # Run single experiment
        asyncio.run(pipeline.run_single_experiment(experiment_type))


if __name__ == "__main__":
    main()
