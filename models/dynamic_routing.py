"""
Dynamic Task Routing System

Implements intelligent task routing and load balancing using MCP protocol.
Provides real-time executor performance monitoring and adaptive task assignment.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from models.mcp_protocol import MCPProtocol, Task, TaskStatus, MCPMessage, MessageType

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Task routing strategies."""

    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    PERFORMANCE_BASED = "performance_based"
    CAPABILITY_MATCH = "capability_match"
    ADAPTIVE = "adaptive"


@dataclass
class ExecutorMetrics:
    """Real-time executor performance metrics."""

    executor_id: str
    current_load: int
    avg_response_time: float
    success_rate: float
    capability_scores: Dict[str, float]
    last_heartbeat: float
    is_available: bool


@dataclass
class RoutingDecision:
    """Result of task routing decision."""

    selected_executor: str
    confidence_score: float
    routing_strategy: RoutingStrategy
    reasoning: str
    alternative_executors: List[str]


class DynamicRouter:
    """Intelligent task router using MCP protocol."""

    def __init__(self, mcp_protocol: MCPProtocol):
        self.mcp_protocol = mcp_protocol
        self.executor_metrics: Dict[str, ExecutorMetrics] = {}
        self.routing_history: List[RoutingDecision] = []
        self.performance_window = 100  # Number of tasks to consider for performance

    async def update_executor_metrics(self, executor_id: str, metrics: Dict[str, Any]):
        """Update executor performance metrics."""
        if executor_id not in self.executor_metrics:
            self.executor_metrics[executor_id] = ExecutorMetrics(
                executor_id=executor_id,
                current_load=0,
                avg_response_time=0.0,
                success_rate=1.0,
                capability_scores={},
                last_heartbeat=asyncio.get_event_loop().time(),
                is_available=True,
            )

        # Update metrics
        current = self.executor_metrics[executor_id]
        current.current_load = metrics.get("current_load", current.current_load)
        current.avg_response_time = metrics.get(
            "avg_response_time", current.avg_response_time
        )
        current.success_rate = metrics.get("success_rate", current.success_rate)
        current.last_heartbeat = asyncio.get_event_loop().time()

        # Update capability scores
        if "capability_scores" in metrics:
            current.capability_scores.update(metrics["capability_scores"])

        logger.info(f"Updated metrics for executor {executor_id}")

    async def select_executor(
        self, task: Task, strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE
    ) -> RoutingDecision:
        """Select the best executor for a given task."""

        available_executors = self._get_available_executors(task)
        if not available_executors:
            raise ValueError(f"No available executors for task {task.task_id}")

        if strategy == RoutingStrategy.ADAPTIVE:
            return await self._adaptive_routing(task, available_executors)
        elif strategy == RoutingStrategy.PERFORMANCE_BASED:
            return await self._performance_based_routing(task, available_executors)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            return await self._load_balanced_routing(task, available_executors)
        elif strategy == RoutingStrategy.CAPABILITY_MATCH:
            return await self._capability_based_routing(task, available_executors)
        else:
            return await self._round_robin_routing(task, available_executors)

    async def _adaptive_routing(
        self, task: Task, available_executors: List[str]
    ) -> RoutingDecision:
        """Adaptive routing that considers multiple factors."""

        scores = {}
        for executor_id in available_executors:
            metrics = self.executor_metrics[executor_id]

            # Calculate composite score
            load_score = 1.0 / (1.0 + metrics.current_load)  # Lower load = higher score
            performance_score = metrics.success_rate * (
                1.0 / (1.0 + metrics.avg_response_time)
            )
            capability_score = self._calculate_capability_match(task, metrics)

            # Weighted combination
            composite_score = (
                0.3 * load_score + 0.4 * performance_score + 0.3 * capability_score
            )

            scores[executor_id] = composite_score

        # Select best executor
        best_executor = max(scores.keys(), key=lambda x: scores[x])

        return RoutingDecision(
            selected_executor=best_executor,
            confidence_score=scores[best_executor],
            routing_strategy=RoutingStrategy.ADAPTIVE,
            reasoning=f"Selected {best_executor} based on load ({self.executor_metrics[best_executor].current_load}), "
            f"performance ({self.executor_metrics[best_executor].success_rate:.2f}), "
            f"and capability match ({self._calculate_capability_match(task, self.executor_metrics[best_executor]):.2f})",
            alternative_executors=[
                e for e in available_executors if e != best_executor
            ],
        )

    def _calculate_capability_match(
        self, task: Task, metrics: ExecutorMetrics
    ) -> float:
        """Calculate how well an executor's capabilities match the task requirements."""
        if not metrics.capability_scores:
            return 0.5  # Default score if no capability data

        # Simple matching based on task type
        task_type = task.task_type.lower()
        if task_type in metrics.capability_scores:
            return metrics.capability_scores[task_type]

        return 0.5

    async def _performance_based_routing(
        self, task: Task, available_executors: List[str]
    ) -> RoutingDecision:
        """Route based on historical performance."""
        best_executor = max(
            available_executors, key=lambda x: self.executor_metrics[x].success_rate
        )

        return RoutingDecision(
            selected_executor=best_executor,
            confidence_score=self.executor_metrics[best_executor].success_rate,
            routing_strategy=RoutingStrategy.PERFORMANCE_BASED,
            reasoning=f"Selected {best_executor} with highest success rate: {self.executor_metrics[best_executor].success_rate:.2f}",
            alternative_executors=[
                e for e in available_executors if e != best_executor
            ],
        )

    async def _load_balanced_routing(
        self, task: Task, available_executors: List[str]
    ) -> RoutingDecision:
        """Route based on current load."""
        best_executor = min(
            available_executors, key=lambda x: self.executor_metrics[x].current_load
        )

        return RoutingDecision(
            selected_executor=best_executor,
            confidence_score=1.0
            / (1.0 + self.executor_metrics[best_executor].current_load),
            routing_strategy=RoutingStrategy.LOAD_BALANCED,
            reasoning=f"Selected {best_executor} with lowest load: {self.executor_metrics[best_executor].current_load}",
            alternative_executors=[
                e for e in available_executors if e != best_executor
            ],
        )

    async def _capability_based_routing(
        self, task: Task, available_executors: List[str]
    ) -> RoutingDecision:
        """Route based on capability matching."""
        best_executor = max(
            available_executors,
            key=lambda x: self._calculate_capability_match(
                task, self.executor_metrics[x]
            ),
        )

        capability_score = self._calculate_capability_match(
            task, self.executor_metrics[best_executor]
        )

        return RoutingDecision(
            selected_executor=best_executor,
            confidence_score=capability_score,
            routing_strategy=RoutingStrategy.CAPABILITY_MATCH,
            reasoning=f"Selected {best_executor} with best capability match: {capability_score:.2f}",
            alternative_executors=[
                e for e in available_executors if e != best_executor
            ],
        )

    async def _round_robin_routing(
        self, task: Task, available_executors: List[str]
    ) -> RoutingDecision:
        """Simple round-robin routing."""
        # Use task_id hash to ensure consistent assignment for same task
        task_hash = hash(task.task_id)
        selected_index = task_hash % len(available_executors)
        selected_executor = available_executors[selected_index]

        return RoutingDecision(
            selected_executor=selected_executor,
            confidence_score=1.0 / len(available_executors),
            routing_strategy=RoutingStrategy.ROUND_ROBIN,
            reasoning=f"Round-robin assignment: {selected_executor}",
            alternative_executors=[
                e for e in available_executors if e != selected_executor
            ],
        )

    def _get_available_executors(self, task: Task) -> List[str]:
        """Get available executors that can handle the task."""
        available = []
        current_time = asyncio.get_event_loop().time()

        for executor_id, metrics in self.executor_metrics.items():
            # Check if executor is available and responsive
            if (
                metrics.is_available and current_time - metrics.last_heartbeat < 30.0
            ):  # 30 second timeout
                available.append(executor_id)

        return available

    async def record_routing_decision(self, decision: RoutingDecision):
        """Record routing decision for analysis."""
        self.routing_history.append(decision)

        # Keep only recent history
        if len(self.routing_history) > self.performance_window:
            self.routing_history.pop(0)

    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get analytics about routing decisions."""
        if not self.routing_history:
            return {"message": "No routing history available"}

        strategy_counts = {}
        avg_confidence = 0.0

        for decision in self.routing_history:
            strategy = decision.routing_strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            avg_confidence += decision.confidence_score

        avg_confidence /= len(self.routing_history)

        return {
            "total_decisions": len(self.routing_history),
            "strategy_distribution": strategy_counts,
            "average_confidence": avg_confidence,
            "recent_decisions": [
                {
                    "executor": d.selected_executor,
                    "strategy": d.routing_strategy.value,
                    "confidence": d.confidence_score,
                }
                for d in self.routing_history[-10:]  # Last 10 decisions
            ],
        }
