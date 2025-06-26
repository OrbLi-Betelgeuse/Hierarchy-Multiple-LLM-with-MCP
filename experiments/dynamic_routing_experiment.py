"""
Dynamic Routing Experiment

Tests different routing strategies and compares their performance.
Implements Experiment A from the enhanced experiment design.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import logging
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

from models.mcp_protocol import MCPProtocol, Task, TaskStatus
from models.dynamic_routing import DynamicRouter, RoutingStrategy, RoutingDecision
from models.llm_interface import create_llm_interface
from models.executor import Executor
from utils.evaluation import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class RoutingExperimentConfig:
    """Configuration for routing experiment."""

    experiment_name: str
    routing_strategies: List[RoutingStrategy]
    num_executors: int
    tasks_per_strategy: int
    task_types: List[str]
    executor_capabilities: Dict[str, List[str]]
    load_variation: bool = True
    performance_variation: bool = True
    duration_minutes: int = 30


@dataclass
class RoutingExperimentResult:
    """Result of a routing experiment."""

    strategy: RoutingStrategy
    total_tasks: int
    successful_tasks: int
    avg_execution_time: float
    avg_response_time: float
    load_balance_score: float
    routing_accuracy: float
    total_coordination_overhead: float
    detailed_results: List[Dict[str, Any]]


class DynamicRoutingExperiment:
    """Experiment class for testing dynamic routing strategies."""

    def __init__(self, config: RoutingExperimentConfig):
        self.config = config
        self.mcp_protocol = MCPProtocol()
        self.router = DynamicRouter(self.mcp_protocol)
        self.executors: Dict[str, Executor] = {}
        self.performance_monitor = PerformanceMonitor()
        self.results: Dict[RoutingStrategy, RoutingExperimentResult] = {}

    async def setup(self):
        """Setup the experiment environment."""
        logger.info("Setting up Dynamic Routing Experiment...")

        # Create executors with different capabilities
        for i in range(self.config.num_executors):
            executor_id = f"executor_{i+1:02d}"
            capabilities = self.config.executor_capabilities.get(
                executor_id, ["general"]
            )

            # Create LLM interface (using mock for testing)
            llm_interface = create_llm_interface("mock", "mock_model")

            # Create executor
            executor = Executor(executor_id, llm_interface, capabilities)
            self.executors[executor_id] = executor

            # Initialize metrics
            await self.router.update_executor_metrics(
                executor_id,
                {
                    "current_load": 0,
                    "avg_response_time": random.uniform(1.0, 5.0),
                    "success_rate": random.uniform(0.7, 0.95),
                    "capability_scores": self._generate_capability_scores(capabilities),
                },
            )

            logger.info(
                f"Created executor {executor_id} with capabilities: {capabilities}"
            )

        logger.info(f"Experiment setup complete with {len(self.executors)} executors")

    def _generate_capability_scores(self, capabilities: List[str]) -> Dict[str, float]:
        """Generate capability scores for an executor."""
        scores = {}
        for capability in capabilities:
            # Add some variation to make executors different
            base_score = 0.8
            variation = random.uniform(-0.2, 0.2)
            scores[capability] = max(0.1, min(1.0, base_score + variation))
        return scores

    async def generate_test_tasks(self) -> List[Task]:
        """Generate test tasks for the experiment."""
        tasks = []

        for i in range(
            self.config.tasks_per_strategy * len(self.config.routing_strategies)
        ):
            task_type = random.choice(self.config.task_types)
            task_id = f"task_{i+1:04d}"

            task = Task(
                task_id=task_id,
                task_type=task_type,
                description=f"Test task {i+1} of type {task_type}",
                parameters={
                    "complexity": random.uniform(1.0, 10.0),
                    "priority": random.randint(1, 5),
                    "estimated_duration": random.uniform(10, 60),
                },
                priority=random.randint(1, 5),
            )
            tasks.append(task)

        return tasks

    async def simulate_executor_performance(
        self, executor_id: str, task: Task
    ) -> Dict[str, Any]:
        """Simulate executor performance for a task."""
        # Get current metrics
        metrics = self.router.executor_metrics[executor_id]

        # Simulate execution time based on task complexity and executor performance
        base_time = task.parameters.get("estimated_duration", 30)
        performance_factor = 1.0 / (1.0 + metrics.avg_response_time)
        complexity_factor = task.parameters.get("complexity", 5.0) / 10.0

        execution_time = base_time * (1 + complexity_factor) * performance_factor

        # Add some randomness
        execution_time *= random.uniform(0.8, 1.2)

        # Simulate success based on success rate
        success = random.random() < metrics.success_rate

        # Simulate tokens used
        tokens_used = int(execution_time * 10 + random.uniform(-50, 50))

        return {
            "execution_time": execution_time,
            "success": success,
            "tokens_used": max(0, tokens_used),
            "quality_score": (
                random.uniform(0.6, 0.95) if success else random.uniform(0.1, 0.4)
            ),
        }

    async def run_routing_strategy_test(
        self, strategy: RoutingStrategy, tasks: List[Task]
    ) -> RoutingExperimentResult:
        """Run test for a specific routing strategy."""
        logger.info(f"Testing routing strategy: {strategy.value}")

        strategy_tasks = tasks[: self.config.tasks_per_strategy]
        detailed_results = []

        start_time = time.time()

        for task in strategy_tasks:
            task_start = time.time()

            try:
                # Get routing decision
                routing_decision = await self.router.select_executor(task, strategy)

                # Simulate task execution
                execution_result = await self.simulate_executor_performance(
                    routing_decision.selected_executor, task
                )

                # Update executor metrics
                executor_id = routing_decision.selected_executor
                current_metrics = self.router.executor_metrics[executor_id]

                # Update load
                current_metrics.current_load += 1

                # Update performance metrics (simplified)
                if execution_result["success"]:
                    current_metrics.success_rate = (
                        0.9 * current_metrics.success_rate + 0.1
                    )
                else:
                    current_metrics.success_rate = 0.9 * current_metrics.success_rate

                # Record routing decision
                await self.router.record_routing_decision(routing_decision)

                # Record detailed result
                result = {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "selected_executor": routing_decision.selected_executor,
                    "routing_strategy": strategy.value,
                    "confidence_score": routing_decision.confidence_score,
                    "execution_time": execution_result["execution_time"],
                    "success": execution_result["success"],
                    "tokens_used": execution_result["tokens_used"],
                    "quality_score": execution_result["quality_score"],
                    "routing_time": time.time() - task_start,
                }
                detailed_results.append(result)

                # Simulate task completion and reduce load
                await asyncio.sleep(0.1)  # Simulate processing time
                current_metrics.current_load = max(0, current_metrics.current_load - 1)

            except Exception as e:
                logger.error(f"Error in task execution: {e}")
                detailed_results.append(
                    {"task_id": task.task_id, "error": str(e), "success": False}
                )

        total_time = time.time() - start_time

        # Calculate metrics
        successful_tasks = sum(1 for r in detailed_results if r.get("success", False))
        execution_times = [
            r["execution_time"] for r in detailed_results if "execution_time" in r
        ]
        routing_times = [
            r["routing_time"] for r in detailed_results if "routing_time" in r
        ]

        # Calculate load balance score
        final_loads = [
            metrics.current_load for metrics in self.router.executor_metrics.values()
        ]
        load_balance_score = (
            1.0 / (1.0 + statistics.stdev(final_loads)) if len(final_loads) > 1 else 1.0
        )

        # Calculate routing accuracy (how often the best executor was chosen)
        routing_accuracy = self._calculate_routing_accuracy(detailed_results)

        return RoutingExperimentResult(
            strategy=strategy,
            total_tasks=len(strategy_tasks),
            successful_tasks=successful_tasks,
            avg_execution_time=(
                statistics.mean(execution_times) if execution_times else 0
            ),
            avg_response_time=statistics.mean(routing_times) if routing_times else 0,
            load_balance_score=load_balance_score,
            routing_accuracy=routing_accuracy,
            total_coordination_overhead=total_time,
            detailed_results=detailed_results,
        )

    def _calculate_routing_accuracy(self, results: List[Dict[str, Any]]) -> float:
        """Calculate routing accuracy based on task success and performance."""
        if not results:
            return 0.0

        # Simple accuracy: percentage of successful tasks
        successful = sum(1 for r in results if r.get("success", False))
        return successful / len(results)

    async def run_experiment(self) -> Dict[RoutingStrategy, RoutingExperimentResult]:
        """Run the complete routing experiment."""
        logger.info("Starting Dynamic Routing Experiment...")

        # Setup
        await self.setup()

        # Generate test tasks
        tasks = await self.generate_test_tasks()
        logger.info(f"Generated {len(tasks)} test tasks")

        # Test each routing strategy
        for strategy in self.config.routing_strategies:
            logger.info(f"Testing strategy: {strategy.value}")

            # Reset executor metrics for fair comparison
            for executor_id in self.executors:
                await self.router.update_executor_metrics(
                    executor_id,
                    {
                        "current_load": 0,
                        "avg_response_time": random.uniform(1.0, 5.0),
                        "success_rate": random.uniform(0.7, 0.95),
                        "capability_scores": self._generate_capability_scores(
                            self.config.executor_capabilities.get(
                                executor_id, ["general"]
                            )
                        ),
                    },
                )

            # Run strategy test
            result = await self.run_routing_strategy_test(strategy, tasks)
            self.results[strategy] = result

            logger.info(
                f"Strategy {strategy.value} completed: "
                f"{result.successful_tasks}/{result.total_tasks} successful, "
                f"avg time: {result.avg_execution_time:.2f}s"
            )

        return self.results

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        if not self.results:
            return {"error": "No results available"}

        # Calculate overall metrics
        strategy_comparison = []
        for strategy, result in self.results.items():
            strategy_comparison.append(
                {
                    "strategy": strategy.value,
                    "success_rate": result.successful_tasks / result.total_tasks,
                    "avg_execution_time": result.avg_execution_time,
                    "avg_response_time": result.avg_response_time,
                    "load_balance_score": result.load_balance_score,
                    "routing_accuracy": result.routing_accuracy,
                    "total_tasks": result.total_tasks,
                    "successful_tasks": result.successful_tasks,
                }
            )

        # Find best performing strategy
        best_strategy = max(strategy_comparison, key=lambda x: x["success_rate"])

        # Calculate improvement over baseline (round-robin)
        baseline = next(
            (s for s in strategy_comparison if s["strategy"] == "round_robin"), None
        )
        improvements = {}
        if baseline:
            for strategy in strategy_comparison:
                if strategy["strategy"] != "round_robin":
                    success_improvement = (
                        strategy["success_rate"] - baseline["success_rate"]
                    ) / baseline["success_rate"]
                    time_improvement = (
                        baseline["avg_execution_time"] - strategy["avg_execution_time"]
                    ) / baseline["avg_execution_time"]
                    improvements[strategy["strategy"]] = {
                        "success_rate_improvement": success_improvement,
                        "execution_time_improvement": time_improvement,
                    }

        return {
            "experiment_name": self.config.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "strategy_comparison": strategy_comparison,
            "best_strategy": best_strategy,
            "improvements_over_baseline": improvements,
            "routing_analytics": self.router.get_routing_analytics(),
            "detailed_results": {
                strategy.value: result.detailed_results
                for strategy, result in self.results.items()
            },
        }


async def run_dynamic_routing_experiment():
    """Main function to run the dynamic routing experiment."""

    # Experiment configuration
    config = RoutingExperimentConfig(
        experiment_name="Dynamic Routing Strategy Comparison",
        routing_strategies=[
            RoutingStrategy.ROUND_ROBIN,
            RoutingStrategy.LOAD_BALANCED,
            RoutingStrategy.PERFORMANCE_BASED,
            RoutingStrategy.CAPABILITY_MATCH,
            RoutingStrategy.ADAPTIVE,
        ],
        num_executors=5,
        tasks_per_strategy=20,
        task_types=[
            "summarization",
            "question_answering",
            "table_generation",
            "general",
        ],
        executor_capabilities={
            "executor_01": ["summarization", "general"],
            "executor_02": ["question_answering", "general"],
            "executor_03": ["table_generation", "general"],
            "executor_04": ["summarization", "question_answering"],
            "executor_05": ["table_generation", "question_answering"],
        },
        load_variation=True,
        performance_variation=True,
        duration_minutes=30,
    )

    # Create and run experiment
    experiment = DynamicRoutingExperiment(config)
    results = await experiment.run_experiment()

    # Generate report
    report = experiment.generate_report()

    # Save results
    with open("results/dynamic_routing_results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("DYNAMIC ROUTING EXPERIMENT RESULTS")
    print("=" * 60)

    for strategy, result in results.items():
        success_rate = result.successful_tasks / result.total_tasks
        print(f"\n{strategy.value.upper()}:")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Avg Execution Time: {result.avg_execution_time:.2f}s")
        print(f"  Load Balance Score: {result.load_balance_score:.3f}")
        print(f"  Routing Accuracy: {result.routing_accuracy:.2%}")

    print(f"\nDetailed results saved to: results/dynamic_routing_results.json")

    return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run experiment
    asyncio.run(run_dynamic_routing_experiment())
