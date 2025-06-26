"""
Model Context Protocol (MCP) Implementation

Provides structured communication protocol for Manager-Executor collaboration.
Enables standardized message passing and task coordination between LLMs.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of MCP messages."""

    TASK_DECOMPOSITION = "task_decomposition"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_EXECUTION = "task_execution"
    TASK_RESULT = "task_result"
    COORDINATION = "coordination"
    ERROR = "error"
    COMPLETION = "completion"


class TaskStatus(Enum):
    """Status of task execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a task in the MCP protocol."""

    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any]
    priority: int = 1
    dependencies: List[str] = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_executor: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class MCPMessage:
    """Standardized MCP message format."""

    message_id: str
    message_type: MessageType
    sender: str
    receiver: str
    timestamp: str
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class MCPProtocol:
    """Implementation of the Model Context Protocol for LLM collaboration."""

    def __init__(self):
        self.message_queue: List[MCPMessage] = []
        self.task_registry: Dict[str, Task] = {}
        self.executor_registry: Dict[str, Dict[str, Any]] = {}
        self.message_handlers: Dict[MessageType, callable] = {}

    def register_message_handler(self, message_type: MessageType, handler: callable):
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler

    def create_task_decomposition_message(
        self,
        sender: str,
        receiver: str,
        task_description: str,
        decomposition_strategy: str,
    ) -> MCPMessage:
        """Create a task decomposition message."""
        return MCPMessage(
            message_id=f"decomp_{asyncio.get_event_loop().time()}",
            message_type=MessageType.TASK_DECOMPOSITION,
            sender=sender,
            receiver=receiver,
            timestamp=str(asyncio.get_event_loop().time()),
            content={
                "task_description": task_description,
                "decomposition_strategy": decomposition_strategy,
                "context": {},
            },
        )

    def create_task_assignment_message(
        self, sender: str, receiver: str, task: Task
    ) -> MCPMessage:
        """Create a task assignment message."""
        return MCPMessage(
            message_id=f"assign_{task.task_id}",
            message_type=MessageType.TASK_ASSIGNMENT,
            sender=sender,
            receiver=receiver,
            timestamp=str(asyncio.get_event_loop().time()),
            content={
                "task": asdict(task),
                "execution_instructions": {},
                "deadline": None,
            },
        )

    def create_task_execution_message(
        self, sender: str, receiver: str, task_id: str, execution_data: Dict[str, Any]
    ) -> MCPMessage:
        """Create a task execution message."""
        return MCPMessage(
            message_id=f"exec_{task_id}",
            message_type=MessageType.TASK_EXECUTION,
            sender=sender,
            receiver=receiver,
            timestamp=str(asyncio.get_event_loop().time()),
            content={
                "task_id": task_id,
                "execution_data": execution_data,
                "progress": 0.0,
            },
        )

    def create_task_result_message(
        self,
        sender: str,
        receiver: str,
        task_id: str,
        result: Dict[str, Any],
        status: TaskStatus = TaskStatus.COMPLETED,
    ) -> MCPMessage:
        """Create a task result message."""
        return MCPMessage(
            message_id=f"result_{task_id}",
            message_type=MessageType.TASK_RESULT,
            sender=sender,
            receiver=receiver,
            timestamp=str(asyncio.get_event_loop().time()),
            content={
                "task_id": task_id,
                "result": result,
                "status": status.value,
                "execution_time": None,
                "tokens_used": None,
            },
        )

    def create_coordination_message(
        self, sender: str, receiver: str, coordination_type: str, data: Dict[str, Any]
    ) -> MCPMessage:
        """Create a coordination message."""
        return MCPMessage(
            message_id=f"coord_{asyncio.get_event_loop().time()}",
            message_type=MessageType.COORDINATION,
            sender=sender,
            receiver=receiver,
            timestamp=str(asyncio.get_event_loop().time()),
            content={"coordination_type": coordination_type, "data": data},
        )

    def parse_message(self, message_data: Union[str, Dict[str, Any]]) -> MCPMessage:
        """Parse message data into MCPMessage object."""
        if isinstance(message_data, str):
            message_data = json.loads(message_data)

        # Handle enum conversion properly
        message_type_str = message_data["message_type"]
        try:
            message_type = MessageType(message_type_str)
        except ValueError:
            # Fallback to coordination if unknown type
            message_type = MessageType.COORDINATION
            logger.warning(
                f"Unknown message type: {message_type_str}, defaulting to coordination"
            )

        return MCPMessage(
            message_id=message_data["message_id"],
            message_type=message_type,
            sender=message_data["sender"],
            receiver=message_data["receiver"],
            timestamp=message_data["timestamp"],
            content=message_data["content"],
            metadata=message_data.get("metadata"),
        )

    def serialize_message(self, message: MCPMessage) -> str:
        """Serialize MCPMessage to JSON string."""
        # Convert enum to string for serialization
        message_dict = asdict(message)
        message_dict["message_type"] = message.message_type.value
        return json.dumps(message_dict, default=str)

    async def send_message(self, message: MCPMessage) -> bool:
        """Send a message through the MCP protocol."""
        try:
            self.message_queue.append(message)

            # Handle message if handler is registered
            if message.message_type in self.message_handlers:
                await self.message_handlers[message.message_type](message)

            logger.info(
                f"Message sent: {message.message_id} from {message.sender} to {message.receiver}"
            )
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    async def receive_message(self, receiver: str) -> Optional[MCPMessage]:
        """Receive messages for a specific receiver."""
        for message in self.message_queue:
            if message.receiver == receiver:
                self.message_queue.remove(message)
                return message
        return None

    def register_task(self, task: Task):
        """Register a task in the task registry."""
        self.task_registry[task.task_id] = task
        logger.info(f"Task registered: {task.task_id}")

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """Update task status and result."""
        if task_id in self.task_registry:
            task = self.task_registry[task_id]
            task.status = status
            if result:
                task.result = result
            if error:
                task.error = error
            if status == TaskStatus.COMPLETED:
                task.completed_at = str(asyncio.get_event_loop().time())
            logger.info(f"Task {task_id} status updated to {status.value}")

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.task_registry.get(task_id)

    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        return [
            task
            for task in self.task_registry.values()
            if task.status == TaskStatus.PENDING
        ]

    def register_executor(
        self, executor_id: str, capabilities: List[str], current_load: int = 0
    ):
        """Register an executor with its capabilities."""
        self.executor_registry[executor_id] = {
            "capabilities": capabilities,
            "current_load": current_load,
            "registered_at": str(asyncio.get_event_loop().time()),
        }
        logger.info(
            f"Executor registered: {executor_id} with capabilities: {capabilities}"
        )

    def get_available_executors(self, required_capabilities: List[str]) -> List[str]:
        """Get available executors with required capabilities."""
        available = []
        for executor_id, info in self.executor_registry.items():
            if all(cap in info["capabilities"] for cap in required_capabilities):
                available.append(executor_id)
        return available

    def update_executor_load(self, executor_id: str, load_change: int):
        """Update executor load."""
        if executor_id in self.executor_registry:
            self.executor_registry[executor_id]["current_load"] += load_change
            logger.info(
                f"Executor {executor_id} load updated: {self.executor_registry[executor_id]['current_load']}"
            )


class MCPManager:
    """Manager-side MCP protocol handler."""

    def __init__(self, manager_id: str):
        self.manager_id = manager_id
        self.protocol = MCPProtocol()
        self.setup_handlers()

    def setup_handlers(self):
        """Setup message handlers for the manager."""
        self.protocol.register_message_handler(
            MessageType.TASK_RESULT, self.handle_task_result
        )
        self.protocol.register_message_handler(
            MessageType.COORDINATION, self.handle_coordination
        )

    async def handle_task_result(self, message: MCPMessage):
        """Handle task result messages."""
        task_id = message.content["task_id"]
        result = message.content["result"]
        status = TaskStatus(message.content["status"])

        self.protocol.update_task_status(task_id, status, result)
        logger.info(f"Manager {self.manager_id} received task result for {task_id}")

    async def handle_coordination(self, message: MCPMessage):
        """Handle coordination messages."""
        coord_type = message.content["coordination_type"]
        data = message.content["data"]
        logger.info(f"Manager {self.manager_id} received coordination: {coord_type}")

    async def decompose_task(
        self, task_description: str, available_executors: List[str]
    ) -> List[Task]:
        """Decompose a task into subtasks."""
        # This would typically involve LLM-based task decomposition
        # For now, we'll create a simple decomposition
        tasks = []
        for i, executor in enumerate(available_executors):
            task = Task(
                task_id=f"subtask_{i}_{asyncio.get_event_loop().time()}",
                task_type="execution",
                description=f"Subtask {i} of: {task_description}",
                parameters={"executor": executor, "subtask_index": i},
                assigned_executor=executor,
            )
            tasks.append(task)
            self.protocol.register_task(task)

        return tasks

    async def assign_tasks(self, tasks: List[Task]):
        """Assign tasks to executors."""
        for task in tasks:
            message = self.protocol.create_task_assignment_message(
                self.manager_id, task.assigned_executor, task
            )
            await self.protocol.send_message(message)


class MCPExecutor:
    """Executor-side MCP protocol handler."""

    def __init__(self, executor_id: str, capabilities: List[str]):
        self.executor_id = executor_id
        self.capabilities = capabilities
        self.protocol = MCPProtocol()
        self.setup_handlers()
        self.protocol.register_executor(executor_id, capabilities)

    def setup_handlers(self):
        """Setup message handlers for the executor."""
        self.protocol.register_message_handler(
            MessageType.TASK_ASSIGNMENT, self.handle_task_assignment
        )
        self.protocol.register_message_handler(
            MessageType.COORDINATION, self.handle_coordination
        )

    async def handle_task_assignment(self, message: MCPMessage):
        """Handle task assignment messages."""
        task_data = message.content["task"]
        task = Task(**task_data)

        # Update executor load
        self.protocol.update_executor_load(self.executor_id, 1)

        # Execute the task (this would be implemented by the specific executor)
        logger.info(
            f"Executor {self.executor_id} received task assignment: {task.task_id}"
        )

        # For now, we'll just acknowledge the task
        result_message = self.protocol.create_task_result_message(
            self.executor_id,
            message.sender,
            task.task_id,
            {"status": "acknowledged"},
            TaskStatus.IN_PROGRESS,
        )
        await self.protocol.send_message(result_message)

    async def handle_coordination(self, message: MCPMessage):
        """Handle coordination messages."""
        coord_type = message.content["coordination_type"]
        data = message.content["data"]
        logger.info(f"Executor {self.executor_id} received coordination: {coord_type}")

    async def report_capabilities(self, manager_id: str):
        """Report capabilities to manager."""
        message = self.protocol.create_coordination_message(
            self.executor_id,
            manager_id,
            "capability_report",
            {"capabilities": self.capabilities, "current_load": 0},
        )
        await self.protocol.send_message(message)
