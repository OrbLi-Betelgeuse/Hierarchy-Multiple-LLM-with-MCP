"""
Evaluation Utilities

Provides metrics and evaluation functions for the Manager-Executor collaboration system.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetrics:
    """Comprehensive metrics for experiment evaluation."""

    experiment_type: str
    total_tasks: int
    successful_tasks: int
    success_rate: float
    average_execution_time: float
    total_execution_time: float
    manager_performance: Dict[str, Any]
    executor_performance: Dict[str, Any]
    quality_metrics: Dict[str, float]
    resource_utilization: Dict[str, Any]


class Evaluator:
    """Main evaluator class for experiment results."""

    def __init__(self):
        self.metrics_history = []

    def calculate_basic_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic performance metrics."""
        if not results:
            return {}

        total_tasks = len(results)
        successful_tasks = len([r for r in results if r.get("status") == "completed"])
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0

        execution_times = [r.get("execution_time", 0) for r in results]
        avg_execution_time = (
            sum(execution_times) / len(execution_times) if execution_times else 0.0
        )
        total_execution_time = sum(execution_times)

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "total_execution_time": total_execution_time,
            "min_execution_time": min(execution_times) if execution_times else 0.0,
            "max_execution_time": max(execution_times) if execution_times else 0.0,
        }

    def calculate_quality_metrics(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate quality-related metrics."""
        quality_scores = [
            r.get("quality_score", 0)
            for r in results
            if r.get("quality_score") is not None
        ]

        if not quality_scores:
            return {}

        return {
            "average_quality_score": sum(quality_scores) / len(quality_scores),
            "min_quality_score": min(quality_scores),
            "max_quality_score": max(quality_scores),
            "quality_variance": self._calculate_variance(quality_scores),
        }

    def calculate_efficiency_metrics(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate efficiency metrics."""
        if not results:
            return {}

        # Calculate throughput (tasks per second)
        total_time = sum(r.get("execution_time", 0) for r in results)
        throughput = len(results) / total_time if total_time > 0 else 0.0

        # Calculate resource efficiency
        total_tokens = sum(r.get("tokens_used", 0) for r in results)
        avg_tokens_per_task = total_tokens / len(results) if results else 0.0

        return {
            "throughput": throughput,
            "total_tokens_used": total_tokens,
            "average_tokens_per_task": avg_tokens_per_task,
            "efficiency_score": (
                throughput / avg_tokens_per_task if avg_tokens_per_task > 0 else 0.0
            ),
        }

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return squared_diff_sum / (len(values) - 1)

    def generate_comprehensive_report(
        self, experiment_results: Dict[str, Any]
    ) -> ExperimentMetrics:
        """Generate a comprehensive evaluation report."""
        results = experiment_results.get("detailed_results", [])

        basic_metrics = self.calculate_basic_metrics(results)
        quality_metrics = self.calculate_quality_metrics(results)
        efficiency_metrics = self.calculate_efficiency_metrics(results)

        # Extract manager and executor performance
        manager_performance = experiment_results.get("manager_metrics", {})
        executor_performance = experiment_results.get("executor_metrics", {})

        # Calculate resource utilization
        resource_utilization = {
            "cpu_usage": "estimated",
            "memory_usage": "estimated",
            "network_usage": "estimated",
            "storage_usage": "estimated",
        }

        metrics = ExperimentMetrics(
            experiment_type=experiment_results.get("experiment_type", "unknown"),
            total_tasks=basic_metrics.get("total_tasks", 0),
            successful_tasks=basic_metrics.get("successful_tasks", 0),
            success_rate=basic_metrics.get("success_rate", 0.0),
            average_execution_time=basic_metrics.get("average_execution_time", 0.0),
            total_execution_time=basic_metrics.get("total_execution_time", 0.0),
            manager_performance=manager_performance,
            executor_performance=executor_performance,
            quality_metrics=quality_metrics,
            resource_utilization=resource_utilization,
        )

        self.metrics_history.append(metrics)
        return metrics

    def compare_experiments(
        self, experiment1: ExperimentMetrics, experiment2: ExperimentMetrics
    ) -> Dict[str, Any]:
        """Compare two experiments and calculate improvements."""
        comparison = {
            "success_rate_improvement": experiment2.success_rate
            - experiment1.success_rate,
            "execution_time_improvement": experiment1.average_execution_time
            - experiment2.average_execution_time,
            "quality_improvement": experiment2.quality_metrics.get(
                "average_quality_score", 0
            )
            - experiment1.quality_metrics.get("average_quality_score", 0),
            "efficiency_improvement": experiment2.quality_metrics.get(
                "efficiency_score", 0
            )
            - experiment1.quality_metrics.get("efficiency_score", 0),
        }

        # Calculate percentage improvements
        if experiment1.success_rate > 0:
            comparison["success_rate_improvement_pct"] = (
                comparison["success_rate_improvement"] / experiment1.success_rate * 100
            )

        if experiment1.average_execution_time > 0:
            comparison["execution_time_improvement_pct"] = (
                comparison["execution_time_improvement"]
                / experiment1.average_execution_time
                * 100
            )

        return comparison

    def export_metrics(self, metrics: ExperimentMetrics, filename: str):
        """Export metrics to a JSON file."""
        try:
            with open(filename, "w") as f:
                json.dump(metrics.__dict__, f, indent=2, default=str)
            logger.info(f"Metrics exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

    def generate_visualization_data(self, metrics: ExperimentMetrics) -> Dict[str, Any]:
        """Generate data suitable for visualization."""
        return {
            "labels": ["Success Rate", "Execution Time", "Quality Score"],
            "values": [
                metrics.success_rate,
                metrics.average_execution_time,
                metrics.quality_metrics.get("average_quality_score", 0),
            ],
            "colors": ["#28a745", "#ffc107", "#17a2b8"],
        }


class PerformanceMonitor:
    """Monitor system performance during experiments."""

    def __init__(self):
        self.start_time = None
        self.metrics = {}

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        logger.info("Performance monitoring started")

    def record_metric(self, metric_name: str, value: Any):
        """Record a performance metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append({"timestamp": time.time(), "value": value})

    def get_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        if not self.start_time:
            return {"error": "Monitoring not started"}

        total_time = time.time() - self.start_time
        summary = {
            "monitoring_duration": total_time,
            "metrics_recorded": len(self.metrics),
            "metric_summaries": {},
        }

        for metric_name, values in self.metrics.items():
            if values:
                numeric_values = [
                    v["value"] for v in values if isinstance(v["value"], (int, float))
                ]
                if numeric_values:
                    summary["metric_summaries"][metric_name] = {
                        "count": len(numeric_values),
                        "average": sum(numeric_values) / len(numeric_values),
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                    }

        return summary

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.start_time = None
        logger.info("Performance monitoring stopped")
