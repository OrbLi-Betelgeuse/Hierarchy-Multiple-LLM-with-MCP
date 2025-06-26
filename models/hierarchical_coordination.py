"""
Hierarchical Coordination System

Extends MCP protocol for multi-level task decomposition and coordination.
Supports nested Manager-Executor hierarchies for complex task handling.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from models.mcp_protocol import MCPProtocol, Task, TaskStatus, MCPMessage, MessageType

logger = logging.getLogger(__name__)


class HierarchyLevel(Enum):
    """Hierarchy levels in the system."""

    ROOT = "root"
    MIDDLE = "middle"
    LEAF = "leaf"


class CoordinationType(Enum):
    """Types of coordination messages."""

    TASK_DECOMPOSITION_REQUEST = "task_decomposition_request"
    SUBTASK_ASSIGNMENT = "subtask_assignment"
    PROGRESS_UPDATE = "progress_update"
    RESULT_AGGREGATION = "result_aggregation"
    CONFLICT_RESOLUTION = "conflict_resolution"
    RESOURCE_NEGOTIATION = "resource_negotiation"


@dataclass
class HierarchyNode:
    """Represents a node in the hierarchical structure."""

    node_id: str
    level: HierarchyLevel
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    current_load: int = 0
    max_capacity: int = 10
    is_active: bool = True


@dataclass
class DecompositionRequest:
    """Request for task decomposition."""

    request_id: str
    task_description: str
    complexity_estimate: float
    required_capabilities: List[str]
    deadline: Optional[float] = None
    priority: int = 1


@dataclass
class DecompositionResult:
    """Result of task decomposition."""

    request_id: str
    subtasks: List[Task]
    dependencies: Dict[str, List[str]]
    execution_order: List[str]
    estimated_duration: float
    confidence_score: float


class HierarchicalCoordinator:
    """Manages hierarchical coordination using MCP protocol."""

    def __init__(self, mcp_protocol: MCPProtocol):
        self.mcp_protocol = mcp_protocol
        self.hierarchy: Dict[str, HierarchyNode] = {}
        self.decomposition_requests: Dict[str, DecompositionRequest] = {}
        self.coordination_history: List[Dict[str, Any]] = []

    async def register_node(
        self,
        node_id: str,
        level: HierarchyLevel,
        capabilities: List[str],
        parent_id: Optional[str] = None,
    ):
        """Register a new node in the hierarchy."""
        node = HierarchyNode(
            node_id=node_id, level=level, parent_id=parent_id, capabilities=capabilities
        )

        self.hierarchy[node_id] = node

        # Update parent's children list
        if parent_id and parent_id in self.hierarchy:
            self.hierarchy[parent_id].children.append(node_id)

        logger.info(f"Registered node {node_id} at level {level.value}")

    async def request_decomposition(
        self,
        task_description: str,
        complexity_estimate: float,
        required_capabilities: List[str],
        deadline: Optional[float] = None,
    ) -> str:
        """Request task decomposition from appropriate level."""

        request_id = f"decomp_{asyncio.get_event_loop().time()}"

        # Determine appropriate level for decomposition
        target_level = self._determine_decomposition_level(complexity_estimate)
        target_nodes = self._get_nodes_at_level(target_level)

        if not target_nodes:
            raise ValueError(f"No nodes available at level {target_level.value}")

        # Create decomposition request
        request = DecompositionRequest(
            request_id=request_id,
            task_description=task_description,
            complexity_estimate=complexity_estimate,
            required_capabilities=required_capabilities,
            deadline=deadline,
        )

        self.decomposition_requests[request_id] = request

        # Send decomposition request to best node
        best_node = self._select_best_node_for_decomposition(target_nodes, request)

        message = self.mcp_protocol.create_coordination_message(
            sender="coordinator",
            receiver=best_node,
            coordination_type=CoordinationType.TASK_DECOMPOSITION_REQUEST.value,
            data={
                "request_id": request_id,
                "task_description": task_description,
                "complexity_estimate": complexity_estimate,
                "required_capabilities": required_capabilities,
                "deadline": deadline,
            },
        )

        await self.mcp_protocol.send_message(message)
        logger.info(f"Sent decomposition request {request_id} to {best_node}")

        return request_id

    def _determine_decomposition_level(self, complexity: float) -> HierarchyLevel:
        """Determine appropriate hierarchy level for task decomposition."""
        if complexity < 3.0:
            return HierarchyLevel.LEAF
        elif complexity < 7.0:
            return HierarchyLevel.MIDDLE
        else:
            return HierarchyLevel.ROOT

    def _get_nodes_at_level(self, level: HierarchyLevel) -> List[str]:
        """Get all nodes at a specific level."""
        return [
            node_id
            for node_id, node in self.hierarchy.items()
            if node.level == level and node.is_active
        ]

    def _select_best_node_for_decomposition(
        self, available_nodes: List[str], request: DecompositionRequest
    ) -> str:
        """Select the best node for task decomposition."""
        best_node = None
        best_score = -1

        for node_id in available_nodes:
            node = self.hierarchy[node_id]

            # Calculate score based on load and capability match
            load_score = 1.0 / (1.0 + node.current_load / node.max_capacity)
            capability_score = self._calculate_capability_match(
                request.required_capabilities, node.capabilities
            )

            composite_score = 0.6 * load_score + 0.4 * capability_score

            if composite_score > best_score:
                best_score = composite_score
                best_node = node_id

        return best_node

    def _calculate_capability_match(
        self, required: List[str], available: List[str]
    ) -> float:
        """Calculate capability match score."""
        if not required:
            return 1.0

        matches = sum(1 for cap in required if cap in available)
        return matches / len(required)

    async def handle_decomposition_response(
        self,
        request_id: str,
        subtasks: List[Dict[str, Any]],
        dependencies: Dict[str, List[str]],
        execution_order: List[str],
        estimated_duration: float,
        confidence_score: float,
    ) -> DecompositionResult:
        """Handle response to decomposition request."""

        if request_id not in self.decomposition_requests:
            raise ValueError(f"Unknown decomposition request: {request_id}")

        # Convert subtask data to Task objects
        task_objects = []
        for subtask_data in subtasks:
            task = Task(
                task_id=subtask_data["task_id"],
                task_type=subtask_data["task_type"],
                description=subtask_data["description"],
                parameters=subtask_data.get("parameters", {}),
                priority=subtask_data.get("priority", 1),
                dependencies=subtask_data.get("dependencies", []),
            )
            task_objects.append(task)

        result = DecompositionResult(
            request_id=request_id,
            subtasks=task_objects,
            dependencies=dependencies,
            execution_order=execution_order,
            estimated_duration=estimated_duration,
            confidence_score=confidence_score,
        )

        # Record coordination
        self.coordination_history.append(
            {
                "timestamp": asyncio.get_event_loop().time(),
                "type": "decomposition_response",
                "request_id": request_id,
                "subtask_count": len(task_objects),
                "confidence_score": confidence_score,
            }
        )

        logger.info(
            f"Received decomposition response for {request_id}: {len(task_objects)} subtasks"
        )

        return result

    async def assign_subtasks_hierarchically(
        self, decomposition: DecompositionResult
    ) -> Dict[str, str]:
        """Assign subtasks using hierarchical approach."""

        assignments = {}

        for subtask in decomposition.subtasks:
            # Find appropriate executor at leaf level
            leaf_nodes = self._get_nodes_at_level(HierarchyLevel.LEAF)
            best_executor = self._select_best_executor_for_subtask(subtask, leaf_nodes)

            if best_executor:
                assignments[subtask.task_id] = best_executor

                # Update node load
                self.hierarchy[best_executor].current_load += 1

                # Send assignment message
                message = self.mcp_protocol.create_task_assignment_message(
                    sender="coordinator", receiver=best_executor, task=subtask
                )
                await self.mcp_protocol.send_message(message)

        return assignments

    def _select_best_executor_for_subtask(
        self, subtask: Task, available_executors: List[str]
    ) -> Optional[str]:
        """Select best executor for a specific subtask."""
        best_executor = None
        best_score = -1

        for executor_id in available_executors:
            node = self.hierarchy[executor_id]

            # Skip if overloaded
            if node.current_load >= node.max_capacity:
                continue

            # Calculate score
            load_score = 1.0 / (1.0 + node.current_load / node.max_capacity)
            capability_score = self._calculate_capability_match(
                [subtask.task_type], node.capabilities
            )

            composite_score = 0.7 * load_score + 0.3 * capability_score

            if composite_score > best_score:
                best_score = composite_score
                best_executor = executor_id

        return best_executor

    async def handle_progress_update(
        self, task_id: str, executor_id: str, progress: float, status: TaskStatus
    ):
        """Handle progress update from executor."""

        # Update node load if task completed
        if status == TaskStatus.COMPLETED and executor_id in self.hierarchy:
            self.hierarchy[executor_id].current_load = max(
                0, self.hierarchy[executor_id].current_load - 1
            )

        # Record coordination
        self.coordination_history.append(
            {
                "timestamp": asyncio.get_event_loop().time(),
                "type": "progress_update",
                "task_id": task_id,
                "executor_id": executor_id,
                "progress": progress,
                "status": status.value,
            }
        )

        logger.info(
            f"Progress update: {task_id} by {executor_id} - {progress:.1%} complete"
        )

    async def aggregate_results(
        self, task_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple subtasks."""

        aggregated = {
            "total_subtasks": len(task_results),
            "completed_subtasks": 0,
            "failed_subtasks": 0,
            "results": {},
            "summary": "",
            "metadata": {
                "aggregation_timestamp": asyncio.get_event_loop().time(),
                "coordination_overhead": len(self.coordination_history),
            },
        }

        for task_id, result in task_results.items():
            if result.get("status") == "completed":
                aggregated["completed_subtasks"] += 1
                aggregated["results"][task_id] = result
            else:
                aggregated["failed_subtasks"] += 1

        # Generate summary
        success_rate = aggregated["completed_subtasks"] / aggregated["total_subtasks"]
        aggregated["summary"] = f"Successfully completed {success_rate:.1%} of subtasks"

        return aggregated

    def get_hierarchy_analytics(self) -> Dict[str, Any]:
        """Get analytics about the hierarchical system."""
        level_counts = {}
        total_load = 0
        active_nodes = 0

        for node in self.hierarchy.values():
            level = node.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
            total_load += node.current_load
            if node.is_active:
                active_nodes += 1

        return {
            "total_nodes": len(self.hierarchy),
            "active_nodes": active_nodes,
            "level_distribution": level_counts,
            "total_load": total_load,
            "average_load": total_load / len(self.hierarchy) if self.hierarchy else 0,
            "coordination_events": len(self.coordination_history),
            "decomposition_requests": len(self.decomposition_requests),
        }
