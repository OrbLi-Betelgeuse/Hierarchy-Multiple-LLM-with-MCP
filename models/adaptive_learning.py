"""
Adaptive Learning System

Implements continuous learning and optimization using MCP protocol.
Adapts task decomposition strategies and execution patterns based on performance feedback.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
from models.mcp_protocol import MCPProtocol, Task, TaskStatus, MCPMessage

logger = logging.getLogger(__name__)


class LearningMetric(Enum):
    """Types of learning metrics."""

    EXECUTION_TIME = "execution_time"
    SUCCESS_RATE = "success_rate"
    TOKEN_EFFICIENCY = "token_efficiency"
    QUALITY_SCORE = "quality_score"
    COORDINATION_OVERHEAD = "coordination_overhead"


@dataclass
class PerformanceRecord:
    """Record of task performance for learning."""

    task_id: str
    task_type: str
    executor_id: str
    execution_time: float
    success: bool
    tokens_used: Optional[int] = None
    quality_score: Optional[float] = None
    coordination_overhead: float = 0.0
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())


@dataclass
class LearningPattern:
    """Identified pattern for task execution."""

    pattern_id: str
    task_type: str
    complexity_range: Tuple[float, float]
    optimal_executor_type: str
    expected_performance: Dict[str, float]
    confidence: float
    sample_size: int


@dataclass
class AdaptationDecision:
    """Decision made by the adaptive learning system."""

    decision_id: str
    decision_type: str
    reasoning: str
    expected_improvement: float
    confidence: float
    applied: bool = False


class AdaptiveLearner:
    """Adaptive learning system for continuous optimization."""

    def __init__(self, mcp_protocol: MCPProtocol):
        self.mcp_protocol = mcp_protocol
        self.performance_history: List[PerformanceRecord] = []
        self.learning_patterns: Dict[str, LearningPattern] = {}
        self.adaptation_history: List[AdaptationDecision] = []
        self.executor_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.task_complexity_estimates: Dict[str, float] = {}

        # Learning parameters
        self.min_samples_for_pattern = 10
        self.learning_window = 1000  # Number of recent records to consider
        self.adaptation_threshold = 0.1  # Minimum improvement threshold

    async def record_performance(self, record: PerformanceRecord):
        """Record performance data for learning."""
        self.performance_history.append(record)

        # Keep only recent history
        if len(self.performance_history) > self.learning_window:
            self.performance_history.pop(0)

        # Update executor performance metrics
        self.executor_performance[record.executor_id][record.task_type].append(
            {
                "execution_time": record.execution_time,
                "success": record.success,
                "tokens_used": record.tokens_used,
                "quality_score": record.quality_score,
            }
        )

        logger.info(f"Recorded performance for task {record.task_id}")

        # Trigger learning if enough data
        if len(self.performance_history) >= self.min_samples_for_pattern:
            await self._trigger_learning()

    async def _trigger_learning(self):
        """Trigger learning process when enough data is available."""
        try:
            # Analyze recent performance patterns
            patterns = await self._analyze_performance_patterns()

            # Update learning patterns
            for pattern in patterns:
                self.learning_patterns[pattern.pattern_id] = pattern

            # Generate adaptation decisions
            adaptations = await self._generate_adaptations()

            # Apply promising adaptations
            for adaptation in adaptations:
                if adaptation.expected_improvement > self.adaptation_threshold:
                    await self._apply_adaptation(adaptation)
                    self.adaptation_history.append(adaptation)

            logger.info(
                f"Learning cycle completed: {len(patterns)} patterns, {len(adaptations)} adaptations"
            )

        except Exception as e:
            logger.error(f"Error in learning process: {e}")

    async def _analyze_performance_patterns(self) -> List[LearningPattern]:
        """Analyze performance data to identify patterns."""
        patterns = []

        # Group by task type
        task_type_groups = defaultdict(list)
        for record in self.performance_history:
            task_type_groups[record.task_type].append(record)

        for task_type, records in task_type_groups.items():
            if len(records) < self.min_samples_for_pattern:
                continue

            # Analyze executor performance for this task type
            executor_performance = defaultdict(list)
            for record in records:
                executor_performance[record.executor_id].append(record)

            # Find best performing executor
            best_executor = None
            best_avg_time = float("inf")

            for executor_id, executor_records in executor_performance.items():
                if len(executor_records) < 3:  # Need minimum samples
                    continue

                avg_time = np.mean([r.execution_time for r in executor_records])
                success_rate = np.mean([r.success for r in executor_records])

                # Weighted score considering both time and success rate
                weighted_score = (
                    avg_time / success_rate if success_rate > 0 else float("inf")
                )

                if weighted_score < best_avg_time:
                    best_avg_time = weighted_score
                    best_executor = executor_id

            if best_executor:
                # Calculate complexity range
                complexity_scores = [
                    self.task_complexity_estimates.get(r.task_id, 1.0) for r in records
                ]
                min_complexity = min(complexity_scores)
                max_complexity = max(complexity_scores)

                # Calculate expected performance
                best_records = executor_performance[best_executor]
                expected_performance = {
                    "execution_time": np.mean([r.execution_time for r in best_records]),
                    "success_rate": np.mean([r.success for r in best_records]),
                    "tokens_used": np.mean([r.tokens_used or 0 for r in best_records]),
                    "quality_score": np.mean(
                        [r.quality_score or 0.5 for r in best_records]
                    ),
                }

                # Calculate confidence based on sample size
                confidence = min(
                    1.0, len(best_records) / 50.0
                )  # Max confidence at 50 samples

                pattern = LearningPattern(
                    pattern_id=f"{task_type}_{best_executor}",
                    task_type=task_type,
                    complexity_range=(min_complexity, max_complexity),
                    optimal_executor_type=best_executor,
                    expected_performance=expected_performance,
                    confidence=confidence,
                    sample_size=len(best_records),
                )

                patterns.append(pattern)

        return patterns

    async def _generate_adaptations(self) -> List[AdaptationDecision]:
        """Generate adaptation decisions based on learned patterns."""
        adaptations = []

        # Analyze current vs optimal performance
        for pattern in self.learning_patterns.values():
            if pattern.confidence < 0.5:  # Only consider confident patterns
                continue

            # Check if current assignment strategy differs from optimal
            current_performance = await self._get_current_performance(pattern.task_type)

            if current_performance:
                expected_improvement = self._calculate_improvement_potential(
                    current_performance, pattern.expected_performance
                )

                if expected_improvement > self.adaptation_threshold:
                    adaptation = AdaptationDecision(
                        decision_id=f"adapt_{pattern.pattern_id}_{asyncio.get_event_loop().time()}",
                        decision_type="executor_preference",
                        reasoning=f"Pattern shows {pattern.optimal_executor_type} performs {expected_improvement:.1%} better for {pattern.task_type}",
                        expected_improvement=expected_improvement,
                        confidence=pattern.confidence,
                    )
                    adaptations.append(adaptation)

        return adaptations

    async def _get_current_performance(
        self, task_type: str
    ) -> Optional[Dict[str, float]]:
        """Get current performance for a task type."""
        recent_records = [
            r
            for r in self.performance_history[-100:]  # Last 100 records
            if r.task_type == task_type
        ]

        if not recent_records:
            return None

        return {
            "execution_time": np.mean([r.execution_time for r in recent_records]),
            "success_rate": np.mean([r.success for r in recent_records]),
            "tokens_used": np.mean([r.tokens_used or 0 for r in recent_records]),
            "quality_score": np.mean([r.quality_score or 0.5 for r in recent_records]),
        }

    def _calculate_improvement_potential(
        self, current: Dict[str, float], optimal: Dict[str, float]
    ) -> float:
        """Calculate potential improvement from current to optimal performance."""
        improvements = []

        # Execution time improvement (lower is better)
        if current["execution_time"] > optimal["execution_time"]:
            time_improvement = (
                current["execution_time"] - optimal["execution_time"]
            ) / current["execution_time"]
            improvements.append(time_improvement)

        # Success rate improvement (higher is better)
        if optimal["success_rate"] > current["success_rate"]:
            success_improvement = optimal["success_rate"] - current["success_rate"]
            improvements.append(success_improvement)

        # Quality score improvement (higher is better)
        if optimal["quality_score"] > current["quality_score"]:
            quality_improvement = optimal["quality_score"] - current["quality_score"]
            improvements.append(quality_improvement)

        return np.mean(improvements) if improvements else 0.0

    async def _apply_adaptation(self, adaptation: AdaptationDecision):
        """Apply an adaptation decision."""
        try:
            # Send adaptation message through MCP
            message = self.mcp_protocol.create_coordination_message(
                sender="adaptive_learner",
                receiver="manager",
                coordination_type="adaptation_decision",
                data={
                    "decision_id": adaptation.decision_id,
                    "decision_type": adaptation.decision_type,
                    "reasoning": adaptation.reasoning,
                    "expected_improvement": adaptation.expected_improvement,
                    "confidence": adaptation.confidence,
                },
            )

            await self.mcp_protocol.send_message(message)
            adaptation.applied = True

            logger.info(f"Applied adaptation: {adaptation.decision_id}")

        except Exception as e:
            logger.error(f"Error applying adaptation: {e}")

    async def get_optimal_executor(
        self, task_type: str, complexity: float
    ) -> Optional[str]:
        """Get optimal executor for a task based on learned patterns."""
        best_pattern = None
        best_score = -1

        for pattern in self.learning_patterns.values():
            if (
                pattern.task_type == task_type
                and pattern.confidence > 0.5
                and pattern.complexity_range[0]
                <= complexity
                <= pattern.complexity_range[1]
            ):

                # Score based on confidence and performance
                score = (
                    pattern.confidence * pattern.expected_performance["success_rate"]
                )

                if score > best_score:
                    best_score = score
                    best_pattern = pattern

        return best_pattern.optimal_executor_type if best_pattern else None

    async def estimate_task_complexity(self, task_description: str) -> float:
        """Estimate task complexity for learning purposes."""
        # Simple heuristic based on description length and keywords
        complexity = 1.0

        # Length factor
        complexity += len(task_description) / 1000.0

        # Keyword factors
        complex_keywords = [
            "analyze",
            "synthesize",
            "evaluate",
            "compare",
            "research",
            "investigate",
        ]
        simple_keywords = ["summarize", "extract", "list", "identify", "find"]

        for keyword in complex_keywords:
            if keyword in task_description.lower():
                complexity += 0.5

        for keyword in simple_keywords:
            if keyword in task_description.lower():
                complexity -= 0.2

        return max(1.0, min(10.0, complexity))  # Clamp between 1-10

    def get_learning_analytics(self) -> Dict[str, Any]:
        """Get analytics about the learning system."""
        if not self.performance_history:
            return {"message": "No performance data available"}

        # Calculate overall metrics
        total_tasks = len(self.performance_history)
        success_rate = np.mean([r.success for r in self.performance_history])
        avg_execution_time = np.mean(
            [r.execution_time for r in self.performance_history]
        )

        # Pattern analysis
        pattern_count = len(self.learning_patterns)
        avg_pattern_confidence = np.mean(
            [p.confidence for p in self.learning_patterns.values()]
        )

        # Adaptation analysis
        applied_adaptations = [a for a in self.adaptation_history if a.applied]
        adaptation_success_rate = (
            len(applied_adaptations) / len(self.adaptation_history)
            if self.adaptation_history
            else 0
        )

        return {
            "total_tasks_analyzed": total_tasks,
            "overall_success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "learning_patterns": pattern_count,
            "average_pattern_confidence": avg_pattern_confidence,
            "adaptations_applied": len(applied_adaptations),
            "adaptation_success_rate": adaptation_success_rate,
            "recent_performance_trend": self._calculate_performance_trend(),
        }

    def _calculate_performance_trend(self) -> str:
        """Calculate recent performance trend."""
        if len(self.performance_history) < 20:
            return "insufficient_data"

        # Compare recent vs older performance
        recent = self.performance_history[-10:]
        older = self.performance_history[-20:-10]

        recent_success = np.mean([r.success for r in recent])
        older_success = np.mean([r.success for r in older])

        if recent_success > older_success + 0.1:
            return "improving"
        elif recent_success < older_success - 0.1:
            return "declining"
        else:
            return "stable"
