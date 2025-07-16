"""
Manager Module

Implements the Manager role in the Manager-Executor collaboration model.
Responsible for task decomposition, planning, and coordination.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from models.llm_interface import LLMInterface, LLMResponse
from models.mcp_protocol import MCPManager, Task, TaskStatus, MCPMessage

logger = logging.getLogger(__name__)


@dataclass
class TaskDecomposition:
    """Result of task decomposition."""

    subtasks: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    execution_order: List[str]
    estimated_complexity: Dict[str, float]


class Manager:
    """Manager class responsible for task decomposition and coordination."""

    def __init__(self, manager_id: str, llm_interface: LLMInterface):
        self.manager_id = manager_id
        self.llm_interface = llm_interface
        self.mcp_manager = MCPManager(manager_id)
        self.task_history: List[Dict[str, Any]] = []
        self.executor_registry: Dict[str, Dict[str, Any]] = {}
        self.executor_instances: Dict[str, Any] = {}

        # Manager-specific prompts
        self.decomposition_prompt_template = """
You are a Manager AI responsible for decomposing complex tasks into manageable subtasks.

TASK: {task_description}

AVAILABLE EXECUTORS: {executors}

Please decompose this task into subtasks that can be executed by the available executors. 
Consider the following:
1. Each subtask should be specific and actionable
2. Dependencies between subtasks should be identified
3. Complexity estimation for each subtask
4. Optimal assignment to available executors

Return your response as a JSON object with the following structure:
{{
    "subtasks": [
        {{
            "task_id": "unique_task_id",
            "description": "clear description of what needs to be done",
            "executor_requirements": ["capability1", "capability2"],
            "estimated_complexity": 1-5 scale,
            "dependencies": ["task_id1", "task_id2"],
            "priority": 1-5 scale
        }}
    ],
    "execution_order": ["task_id1", "task_id2", ...],
    "overall_strategy": "brief description of the overall approach"
}}
"""

        self.planning_prompt_template = """
You are a Manager AI responsible for creating execution plans for complex tasks.

TASK: {task_description}
DECOMPOSITION: {decomposition_result}

Based on the task decomposition above, create a detailed execution plan that includes:
1. Task assignment strategy
2. Resource allocation
3. Timeline estimation
4. Quality control measures
5. Contingency plans

Return your response as a JSON object with the following structure:
{{
    "execution_plan": {{
        "phases": [
            {{
                "phase_id": "phase_1",
                "tasks": ["task_id1", "task_id2"],
                "estimated_duration": "time_estimate",
                "dependencies": ["phase_id"],
                "quality_gates": ["check1", "check2"]
            }}
        ],
        "resource_allocation": {{
            "executor_assignments": {{
                "executor_id": ["task_id1", "task_id2"]
            }},
            "estimated_costs": {{
                "computational": "estimate",
                "time": "estimate"
            }}
        }},
        "risk_mitigation": [
            {{
                "risk": "description",
                "mitigation": "strategy"
            }}
        ]
    }}
}}
"""

    async def register_executor(
        self,
        executor_id: str,
        capabilities: List[str],
        performance_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Register an executor with its capabilities."""
        self.executor_registry[executor_id] = {
            "capabilities": capabilities,
            "performance_metrics": performance_metrics or {},
            "current_load": 0,
            "availability": True,
        }
        logger.info(
            f"Executor {executor_id} registered with capabilities: {capabilities}"
        )

    async def decompose_task(
        self, task_description: str, task_type: str = "general"
    ) -> TaskDecomposition:
        """Decompose a complex task into subtasks using LLM."""
        try:
            # Get available executors
            available_executors = [
                executor_id
                for executor_id, info in self.executor_registry.items()
                if info["availability"]
            ]

            if not available_executors:
                raise ValueError("No available executors for task decomposition")

            # Create decomposition prompt
            prompt = self.decomposition_prompt_template.format(
                task_description=task_description,
                executors=json.dumps(available_executors, indent=2),
            )

            # Generate decomposition using LLM
            response = await self.llm_interface.generate_with_system_prompt(
                system_prompt="You are an expert task decomposition specialist. Provide clear, actionable subtasks.",
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=2000,
            )

            # Parse the response
            try:
                decomposition_data = json.loads(response.content)
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse decomposition response: {response.content}"
                )
                # Fallback to simple decomposition
                decomposition_data = self._create_fallback_decomposition(
                    task_description, available_executors
                )

            # Create TaskDecomposition object
            decomposition = TaskDecomposition(
                subtasks=decomposition_data.get("subtasks", []),
                dependencies={},
                execution_order=decomposition_data.get("execution_order", []),
                estimated_complexity={},
            )

            # Extract dependencies and complexity
            for subtask in decomposition.subtasks:
                task_id = subtask.get("task_id", "")
                decomposition.dependencies[task_id] = subtask.get("dependencies", [])
                decomposition.estimated_complexity[task_id] = subtask.get(
                    "estimated_complexity", 1.0
                )

            logger.info(f"Task decomposed into {len(decomposition.subtasks)} subtasks")
            return decomposition

        except Exception as e:
            logger.error(f"Error in task decomposition: {e}")
            raise

    def _create_fallback_decomposition(
        self, task_description: str, available_executors: List[str]
    ) -> Dict[str, Any]:
        """Create a simple fallback decomposition when LLM parsing fails."""
        # Create a simple single-task decomposition
        subtasks = [
            {
                "task_id": "main_task",
                "description": task_description,
                "executor_requirements": ["general"],
                "estimated_complexity": 1.0,
                "dependencies": [],
                "priority": 1,
                "task_type": "general",
            }
        ]

        return {
            "subtasks": subtasks,
            "execution_order": ["main_task"],
            "overall_strategy": "Direct execution of the main task",
        }

    async def create_execution_plan(
        self, task_description: str, decomposition: TaskDecomposition
    ) -> Dict[str, Any]:
        """Create a detailed execution plan based on task decomposition."""
        try:
            prompt = self.planning_prompt_template.format(
                task_description=task_description,
                decomposition_result=json.dumps(decomposition.subtasks, indent=2),
            )

            response = await self.llm_interface.generate_with_system_prompt(
                system_prompt="You are an expert project manager. Create detailed, actionable execution plans.",
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=1500,
            )

            try:
                plan_data = json.loads(response.content)
                return plan_data
            except json.JSONDecodeError:
                logger.error(f"Failed to parse execution plan: {response.content}")
                return self._create_fallback_plan(decomposition)

        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            return self._create_fallback_plan(decomposition)

    def _create_fallback_plan(self, decomposition: TaskDecomposition) -> Dict[str, Any]:
        """Create a simple fallback execution plan."""
        return {
            "execution_plan": {
                "phases": [
                    {
                        "phase_id": "phase_1",
                        "tasks": [task["task_id"] for task in decomposition.subtasks],
                        "estimated_duration": "variable",
                        "dependencies": [],
                        "quality_gates": ["completion_check"],
                    }
                ],
                "resource_allocation": {
                    "executor_assignments": {},
                    "estimated_costs": {"computational": "medium", "time": "variable"},
                },
                "risk_mitigation": [
                    {
                        "risk": "executor failure",
                        "mitigation": "retry with different executor",
                    }
                ],
            }
        }

    async def assign_tasks_to_executors(
        self, decomposition: TaskDecomposition
    ) -> List[Task]:
        """Assign subtasks to available executors."""
        tasks = []

        for subtask in decomposition.subtasks:
            task_id = subtask["task_id"]
            requirements = subtask.get("executor_requirements", ["general"])

            # Find suitable executor - be more flexible in matching
            suitable_executors = []
            for executor_id, info in self.executor_registry.items():
                if info["availability"]:
                    # Check if executor has any of the required capabilities
                    executor_capabilities = info["capabilities"]
                    if any(req in executor_capabilities for req in requirements):
                        suitable_executors.append(executor_id)
                    # Also accept "general" capability as fallback
                    elif "general" in executor_capabilities:
                        suitable_executors.append(executor_id)

            if not suitable_executors:
                logger.warning(f"No suitable executor found for task {task_id}")
                continue

            # Select executor with lowest load
            selected_executor = min(
                suitable_executors,
                key=lambda x: self.executor_registry[x]["current_load"],
            )

            # Create Task object
            task = Task(
                task_id=task_id,
                task_type=subtask.get("task_type", "execution"),
                description=subtask["description"],
                parameters={
                    "executor": selected_executor,
                    "requirements": requirements,
                    "complexity": subtask.get("estimated_complexity", 1.0),
                    "priority": subtask.get("priority", 1),
                },
                priority=subtask.get("priority", 1),
                dependencies=subtask.get("dependencies", []),
                assigned_executor=selected_executor,
            )

            tasks.append(task)
            self.mcp_manager.protocol.register_task(task)

            # Update executor load
            self.executor_registry[selected_executor]["current_load"] += 1

        return tasks

    async def coordinate_execution(self, tasks: List[Task]) -> Dict[str, Any]:
        """Coordinate the execution of assigned tasks using direct executor calls."""
        try:
            # Execute tasks directly with executors
            execution_results = {}

            for task in tasks:
                logger.info(
                    f"Executing task {task.task_id} with executor {task.assigned_executor}"
                )

                # Get the executor instance directly (强制用str类型查找)
                executor_id_str = str(task.assigned_executor)
                executor = getattr(self, "executor_instances", {}).get(executor_id_str)

                if executor:
                    try:
                        # Execute the task directly
                        result = await executor.execute_task(task)
                        summary = result.result.get("output", "") if hasattr(result, "result") else str(result)
                        logger.info(f"[DEBUG] Task {task.task_id} summary from executor: {summary}")
                        execution_results[task.task_id] = {
                            "output": summary,
                            "status": result.status.value,
                            "executor_id": task.assigned_executor,
                            "execution_time": result.execution_time,
                        }
                        logger.info(f"Task {task.task_id} completed successfully")
                    except Exception as e:
                        logger.error(f"Error executing task {task.task_id}: {e}")
                        execution_results[task.task_id] = {
                            "output": f"Task {task.task_id} failed",
                            "status": "failed",
                            "error": str(e),
                            "executor_id": task.assigned_executor,
                        }
                else:
                    print(f"[DEBUG] assigned_executor_id: {executor_id_str}")
                    print(f"[DEBUG] manager.executor_instances keys: {list(self.executor_instances.keys())}")
                    print(f"[ERROR] No executor instance found for task {task.task_id} (executor_id={executor_id_str})")
                    execution_results[task.task_id] = {
                        "output": f"Task {task.task_id} failed - no executor",
                        "status": "failed",
                        "error": "No executor available",
                    }

            # Aggregate results
            final_result = await self._aggregate_results(execution_results, tasks)

            # Log execution summary
            self.task_history.append(
                {
                    "task_count": len(tasks),
                    "execution_time": "calculated",
                    "success_rate": len(execution_results) / len(tasks),
                    "results": final_result,
                }
            )

            return final_result

        except Exception as e:
            logger.error(f"Error in execution coordination: {e}")
            raise

    async def _aggregate_results(
        self, execution_results: Dict[str, Any], tasks: List[Task]
    ) -> Dict[str, Any]:
        """Aggregate results from all completed tasks."""
        completed_tasks = len(execution_results)
        total_tasks = len(tasks)

        # Avoid division by zero
        success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0

        aggregated = {
            "overall_status": "completed" if completed_tasks > 0 else "failed",
            "task_results": execution_results,
            "summary": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "success_rate": success_rate,
            },
            "metadata": {
                "execution_time": "calculated",
                "resource_utilization": "calculated",
            },
        }

        return aggregated

    async def execute_task(
        self, task_description: str, task_type: str = "general"
    ) -> Dict[str, Any]:
        """Main method to execute a complete task using the Manager-Executor system."""
        try:
            logger.info(
                f"Manager {self.manager_id} starting task execution: {task_description}"
            )

            # Step 1: Decompose task
            decomposition = await self.decompose_task(task_description, task_type)

            # Step 2: Create execution plan
            execution_plan = await self.create_execution_plan(
                task_description, decomposition
            )

            # Step 3: Assign tasks to executors
            tasks = await self.assign_tasks_to_executors(decomposition)

            # Step 4: Coordinate execution
            results = await self.coordinate_execution(tasks)

            logger.info(f"Task execution completed successfully")
            return results

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the manager."""
        return {
            "manager_id": self.manager_id,
            "tasks_processed": len(self.task_history),
            "executors_registered": len(self.executor_registry),
            "average_success_rate": (
                sum(task["success_rate"] for task in self.task_history)
                / len(self.task_history)
                if self.task_history
                else 0
            ),
            "current_executor_loads": {
                executor_id: info["current_load"]
                for executor_id, info in self.executor_registry.items()
            },
        }

    def add_executor_instance(self, executor):
        """注册executor实例，便于直接调用其execute_task方法"""
        eid = str(getattr(executor, "executor_id", executor))
        self.executor_instances[eid] = executor
        logger.info(f"Executor instance registered: {eid}")

    def print_executor_instances(self):
        """打印所有已注册的executor实例id"""
        logger.info(f"Registered executor instances: {list(self.executor_instances.keys())}")
        print(f"[Manager] Registered executor instances: {list(self.executor_instances.keys())}")
