"""
Executor Module

Implements the Executor role in the Manager-Executor collaboration model.
Responsible for executing specific subtasks assigned by the Manager.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from models.llm_interface import LLMInterface, LLMResponse
from models.mcp_protocol import MCPExecutor, Task, TaskStatus, MCPMessage

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of task execution."""

    task_id: str
    status: TaskStatus
    result: Dict[str, Any]
    execution_time: float
    tokens_used: Optional[int] = None
    error: Optional[str] = None


class Executor:
    """Executor class responsible for executing specific subtasks."""

    def __init__(
        self, executor_id: str, llm_interface: LLMInterface, capabilities: List[str]
    ):
        self.executor_id = executor_id
        self.llm_interface = llm_interface
        self.capabilities = capabilities
        self.mcp_executor = MCPExecutor(executor_id, capabilities)
        self.task_history: List[ExecutionResult] = []
        self.current_tasks: Dict[str, Task] = {}

        # Task-specific prompt templates
        self.task_prompts = {
            "summarization": """
You are an expert text summarizer. Create a concise, accurate summary of the provided text.

TEXT TO SUMMARIZE:
{input_text}

TASK REQUIREMENTS:
- Length: {length_requirement}
- Focus: {focus_areas}
- Style: {style_requirement}

Please provide a well-structured summary that captures the key points while maintaining clarity and coherence.
""",
            "question_answering": """
You are an expert question answering system. Provide accurate, helpful answers to the given question.

QUESTION: {question}

CONTEXT: {context}

Please provide a comprehensive answer that directly addresses the question using the provided context.
""",
            "table_generation": """
You are an expert data analyst. Extract structured information from the provided text and format it as a table.

TEXT: {input_text}

TABLE REQUIREMENTS:
- Columns: {columns}
- Format: {format_requirement}
- Data types: {data_types}

Please extract the relevant information and present it in a well-formatted table structure.
""",
            "general": """
You are a task executor. Please complete the following task to the best of your ability.

TASK: {task_description}

INPUT: {input_data}

Please provide a clear, actionable response that addresses the task requirements.
""",
        }

    async def register_with_manager(self, manager_id: str):
        """Register this executor with a manager."""
        await self.mcp_executor.report_capabilities(manager_id)
        logger.info(f"Executor {self.executor_id} registered with manager {manager_id}")

    async def execute_task(self, task: Task) -> ExecutionResult:
        """Execute a specific task."""
        try:
            start_time = asyncio.get_event_loop().time()

            logger.info(f"Executor {self.executor_id} starting task: {task.task_id}")

            # Determine task type and get appropriate prompt
            task_type = self._determine_task_type(task)
            prompt_template = self.task_prompts.get(
                task_type, self.task_prompts["general"]
            )

            # Prepare input data
            input_data = self._prepare_input_data(task)

            # Generate prompt
            if task.description and task.description.strip():
                prompt = task.description
            else:
                prompt = prompt_template.format(**input_data)

            # Execute using LLM
            response = await self.llm_interface.generate_with_system_prompt(
                system_prompt=f"You are an expert {task_type} executor. Focus on accuracy and efficiency.",
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=1000,
            )

            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time

            # Create execution result
            result = ExecutionResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result={
                    "output": response.content,
                    "task_type": task_type,
                    "executor_id": self.executor_id,
                },
                execution_time=execution_time,
                tokens_used=response.tokens_used,
            )

            # Update task status in MCP
            self.mcp_executor.protocol.update_task_status(
                task.task_id, TaskStatus.COMPLETED, result.result
            )

            # Add to history
            self.task_history.append(result)

            # Remove from current tasks
            if task.task_id in self.current_tasks:
                del self.current_tasks[task.task_id]

            logger.info(
                f"Task {task.task_id} completed successfully in {execution_time:.2f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")

            # Create error result
            error_result = ExecutionResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                result={},
                execution_time=0.0,
                error=str(e),
            )

            # Update task status in MCP
            self.mcp_executor.protocol.update_task_status(
                task.task_id, TaskStatus.FAILED, error=str(e)
            )

            # Add to history
            self.task_history.append(error_result)

            # Remove from current tasks
            if task.task_id in self.current_tasks:
                del self.current_tasks[task.task_id]

            return error_result

    def _determine_task_type(self, task: Task) -> str:
        """Determine the type of task based on description and parameters."""
        description = task.description.lower()
        parameters = task.parameters or {}

        if any(
            keyword in description
            for keyword in ["summarize", "summary", "summarization"]
        ):
            return "summarization"
        elif any(
            keyword in description for keyword in ["question", "answer", "qa", "query"]
        ):
            return "question_answering"
        elif any(
            keyword in description
            for keyword in ["table", "extract", "structure", "format"]
        ):
            return "table_generation"
        elif "task_type" in parameters:
            return parameters["task_type"]
        else:
            return "general"

    def _prepare_input_data(self, task: Task) -> Dict[str, Any]:
        """Prepare input data for task execution."""
        parameters = task.parameters or {}

        # Base input data
        input_data = {
            "task_description": task.description,
            "input_data": parameters.get("input", ""),
            "input_text": parameters.get("input", ""),
            "question": parameters.get("question", ""),
            "context": parameters.get("context", ""),
            "length_requirement": parameters.get("length", "medium"),
            "focus_areas": parameters.get("focus", "all"),
            "style_requirement": parameters.get("style", "neutral"),
            "columns": parameters.get("columns", []),
            "format_requirement": parameters.get("format", "markdown"),
            "data_types": parameters.get("data_types", {}),
        }

        return input_data

    async def handle_task_assignment(self, message: MCPMessage):
        """Handle task assignment from manager."""
        try:
            task_data = message.content["task"]
            task = Task(**task_data)

            # Add to current tasks
            self.current_tasks[task.task_id] = task

            # Update executor load
            self.mcp_executor.protocol.update_executor_load(self.executor_id, 1)

            logger.info(
                f"Executor {self.executor_id} received task assignment: {task.task_id}"
            )

            # Execute the task
            result = await self.execute_task(task)

            # Send result back to manager
            result_message = self.mcp_executor.protocol.create_task_result_message(
                self.executor_id,
                message.sender,
                task.task_id,
                result.result,
                result.status,
            )
            await self.mcp_executor.protocol.send_message(result_message)

            # Update executor load
            self.mcp_executor.protocol.update_executor_load(self.executor_id, -1)

        except Exception as e:
            logger.error(f"Error handling task assignment: {e}")

            # Send error message
            error_message = self.mcp_executor.protocol.create_task_result_message(
                self.executor_id,
                message.sender,
                task.task_id,
                {"error": str(e)},
                TaskStatus.FAILED,
            )
            await self.mcp_executor.protocol.send_message(error_message)

    async def process_messages(self):
        """Process incoming messages from the MCP protocol."""
        while True:
            try:
                # Check for messages
                message = await self.mcp_executor.protocol.receive_message(
                    self.executor_id
                )

                if message:
                    if message.message_type.value == "task_assignment":
                        await self.handle_task_assignment(message)
                    elif message.message_type.value == "coordination":
                        await self.mcp_executor.handle_coordination(message)

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error processing messages: {e}")
                await asyncio.sleep(1)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the executor."""
        if not self.task_history:
            return {
                "executor_id": self.executor_id,
                "tasks_completed": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "total_tokens_used": 0,
            }

        completed_tasks = [
            task for task in self.task_history if task.status == TaskStatus.COMPLETED
        ]
        failed_tasks = [
            task for task in self.task_history if task.status == TaskStatus.FAILED
        ]

        total_tasks = len(self.task_history)
        success_rate = len(completed_tasks) / total_tasks if total_tasks > 0 else 0.0

        avg_execution_time = (
            sum(task.execution_time for task in completed_tasks) / len(completed_tasks)
            if completed_tasks
            else 0.0
        )

        total_tokens = sum(task.tokens_used or 0 for task in completed_tasks)

        return {
            "executor_id": self.executor_id,
            "tasks_completed": len(completed_tasks),
            "tasks_failed": len(failed_tasks),
            "total_tasks": total_tasks,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "total_tokens_used": total_tokens,
            "current_load": len(self.current_tasks),
            "capabilities": self.capabilities,
        }

    async def start(self):
        """Start the executor message processing loop."""
        logger.info(f"Executor {self.executor_id} starting message processing")
        await self.process_messages()

    async def stop(self):
        """Stop the executor."""
        logger.info(f"Executor {self.executor_id} stopping")
        # Clean up any remaining tasks
        for task_id in list(self.current_tasks.keys()):
            task = self.current_tasks[task_id]
            self.mcp_executor.protocol.update_task_status(task_id, TaskStatus.CANCELLED)
            del self.current_tasks[task_id]


class SpecializedExecutor(Executor):
    """Specialized executor for specific task types."""

    def __init__(
        self,
        executor_id: str,
        llm_interface: LLMInterface,
        task_type: str,
        specialized_capabilities: List[str],
    ):
        super().__init__(executor_id, llm_interface, specialized_capabilities)
        self.task_type = task_type
        self.specialized_capabilities = specialized_capabilities

    def _determine_task_type(self, task: Task) -> str:
        """Override to always return the specialized task type."""
        return self.task_type

    async def execute_task(self, task: Task) -> ExecutionResult:
        """Execute task with specialized processing."""
        # Add specialized preprocessing if needed
        if self.task_type == "summarization":
            task = await self._preprocess_summarization_task(task)
        elif self.task_type == "question_answering":
            task = await self._preprocess_qa_task(task)
        elif self.task_type == "table_generation":
            task = await self._preprocess_table_task(task)

        return await super().execute_task(task)

    async def _preprocess_summarization_task(self, task: Task) -> Task:
        """Preprocess summarization tasks."""
        # Add summarization-specific preprocessing
        parameters = task.parameters or {}
        if "input" in parameters:
            # Clean and prepare text for summarization
            text = parameters["input"]
            # Remove extra whitespace, normalize formatting, etc.
            cleaned_text = " ".join(text.split())
            parameters["input"] = cleaned_text
            task.parameters = parameters

        return task

    async def _preprocess_qa_task(self, task: Task) -> Task:
        """Preprocess question answering tasks."""
        # Add QA-specific preprocessing
        parameters = task.parameters or {}
        if "question" in parameters:
            # Enhance question with context if available
            question = parameters["question"]
            context = parameters.get("context", "")
            if context:
                enhanced_question = f"Context: {context}\nQuestion: {question}"
                parameters["question"] = enhanced_question
                task.parameters = parameters

        return task

    async def _preprocess_table_task(self, task: Task) -> Task:
        """Preprocess table generation tasks."""
        # Add table generation-specific preprocessing
        parameters = task.parameters or {}
        if "input" in parameters:
            # Extract potential table data
            text = parameters["input"]
            # Identify potential table structures, headers, etc.
            # This could involve more sophisticated text analysis
            parameters["extracted_data"] = self._extract_table_data(text)
            task.parameters = parameters

        return task

    def _extract_table_data(self, text: str) -> Dict[str, Any]:
        """Extract potential table data from text."""
        # Simple extraction - in practice, this could be more sophisticated
        lines = text.split("\n")
        potential_headers = []
        potential_rows = []

        for line in lines:
            if "|" in line or "\t" in line:
                # Potential table row
                if not potential_headers:
                    potential_headers = [
                        col.strip() for col in line.split("|") if col.strip()
                    ]
                else:
                    potential_rows.append(
                        [col.strip() for col in line.split("|") if col.strip()]
                    )

        return {
            "headers": potential_headers,
            "rows": potential_rows,
            "table_format": "detected",
        }
