"""
RAG-Enabled Executor Module

Extends the base executor with RAGFlow capabilities for retrieval-augmented generation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from models.executor import Executor, ExecutionResult
from models.llm_interface import LLMInterface
from models.ragflow_interface import RAGFlowInterface, RAGFlowResponse
from models.mcp_protocol import Task, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class RAGExecutionResult(ExecutionResult):
    """Result of RAG-enabled task execution."""

    retrieved_documents: List[Dict[str, Any]] = None
    confidence_score: float = 0.0
    rag_metadata: Optional[Dict[str, Any]] = None


class RAGExecutor(Executor):
    """RAG-enabled executor that combines LLM and RAGFlow capabilities."""

    def __init__(
        self,
        executor_id: str,
        llm_interface: LLMInterface,
        ragflow_interface: RAGFlowInterface,
        capabilities: List[str],
        knowledge_base_id: Optional[str] = None,
    ):
        super().__init__(executor_id, llm_interface, capabilities)
        self.ragflow_interface = ragflow_interface
        self.knowledge_base_id = knowledge_base_id
        self.rag_enabled = True

        # Add RAG-specific task prompts
        self.rag_task_prompts = {
            "rag_question_answering": """
You are an expert RAG-enabled question answering system. Use the retrieved documents to provide accurate, helpful answers.

QUESTION: {question}

RETRIEVED DOCUMENTS:
{retrieved_documents}

Please provide a comprehensive answer that:
1. Directly addresses the question using the retrieved documents
2. Cites specific information from the documents when possible
3. Acknowledges if the retrieved documents don't fully answer the question
4. Maintains accuracy and avoids speculation beyond the provided context
""",
            "rag_summarization": """
You are an expert RAG-enabled summarization system. Create a concise, accurate summary based on the retrieved documents.

DOCUMENTS TO SUMMARIZE:
{retrieved_documents}

TASK REQUIREMENTS:
- Length: {length_requirement}
- Focus: {focus_areas}
- Style: {style_requirement}

Please provide a well-structured summary that captures the key points from the retrieved documents while maintaining clarity and coherence.
""",
            "rag_research": """
You are an expert RAG-enabled research assistant. Analyze the retrieved documents and provide insights.

RESEARCH QUERY: {query}

RETRIEVED DOCUMENTS:
{retrieved_documents}

Please provide:
1. Key findings from the documents
2. Analysis of the information
3. Connections between different sources
4. Recommendations or conclusions based on the evidence
""",
        }

    async def setup_knowledge_base(self, name: str, description: str = "") -> str:
        """Setup a knowledge base for this executor."""
        try:
            # Check if RAGFlow is accessible
            if not await self.ragflow_interface.check_connection():
                logger.warning("RAGFlow not accessible, falling back to LLM-only mode")
                self.rag_enabled = False
                return ""

            # Create knowledge base
            kb_info = await self.ragflow_interface.create_knowledge_base(
                name, description
            )
            if kb_info:
                self.knowledge_base_id = kb_info.get("id", "")
                logger.info(f"Created knowledge base: {self.knowledge_base_id}")
                return self.knowledge_base_id
            else:
                logger.error("Failed to create knowledge base")
                self.rag_enabled = False
                return ""

        except Exception as e:
            logger.error(f"Error setting up knowledge base: {e}")
            self.rag_enabled = False
            return ""

    async def upload_documents_to_kb(self, documents: List[Dict[str, Any]]) -> bool:
        """Upload documents to the executor's knowledge base."""
        if not self.knowledge_base_id or not self.rag_enabled:
            logger.warning("No knowledge base available for document upload")
            return False

        try:
            return await self.ragflow_interface.upload_documents(
                self.knowledge_base_id, documents
            )
        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            return False

    async def execute_rag_task(self, task: Task) -> RAGExecutionResult:
        """Execute a task using RAG capabilities."""
        try:
            start_time = asyncio.get_event_loop().time()

            logger.info(
                f"RAG Executor {self.executor_id} starting RAG task: {task.task_id}"
            )

            if not self.rag_enabled or not self.knowledge_base_id:
                logger.warning("RAG not enabled, falling back to standard execution")
                return await super().execute_task(task)

            # Determine task type
            task_type = self._determine_rag_task_type(task)

            # Get query from task
            query = self._extract_query_from_task(task)

            # Retrieve relevant documents using RAGFlow
            rag_response = await self.ragflow_interface.query_knowledge_base(
                self.knowledge_base_id, query
            )

            if not rag_response.retrieved_documents:
                logger.warning("No relevant documents found, using LLM-only response")
                # Fall back to standard LLM execution
                standard_result = await super().execute_task(task)
                return RAGExecutionResult(
                    task_id=standard_result.task_id,
                    status=standard_result.status,
                    result=standard_result.result,
                    execution_time=standard_result.execution_time,
                    tokens_used=standard_result.tokens_used,
                    error=standard_result.error,
                    retrieved_documents=[],
                    confidence_score=0.0,
                )

            # Generate response using retrieved documents
            response = await self._generate_rag_response(
                task, task_type, query, rag_response
            )

            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time

            # Create RAG execution result
            result = RAGExecutionResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result={
                    "output": response.content,
                    "task_type": task_type,
                    "executor_id": self.executor_id,
                    "rag_enabled": True,
                },
                execution_time=execution_time,
                tokens_used=(
                    response.metadata.get("tokens_used") if response.metadata else None
                ),
                retrieved_documents=rag_response.retrieved_documents,
                confidence_score=rag_response.confidence_score,
                rag_metadata=response.metadata,
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
                f"RAG task {task.task_id} completed successfully in {execution_time:.2f}s "
                f"(confidence: {rag_response.confidence_score:.2f})"
            )
            return result

        except Exception as e:
            logger.error(f"Error executing RAG task {task.task_id}: {e}")

            # Create error result
            error_result = RAGExecutionResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                result={},
                execution_time=0.0,
                error=str(e),
                retrieved_documents=[],
                confidence_score=0.0,
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

    def _determine_rag_task_type(self, task: Task) -> str:
        """Determine the type of RAG task based on description and parameters."""
        description = task.description.lower()
        parameters = task.parameters or {}

        if any(
            keyword in description for keyword in ["question", "answer", "query", "ask"]
        ):
            return "rag_question_answering"
        elif any(
            keyword in description
            for keyword in ["summarize", "summary", "summarization"]
        ):
            return "rag_summarization"
        elif any(
            keyword in description for keyword in ["research", "analyze", "investigate"]
        ):
            return "rag_research"
        else:
            return "rag_question_answering"  # Default

    def _extract_query_from_task(self, task: Task) -> str:
        """Extract the query from the task description or parameters."""
        parameters = task.parameters or {}

        # Try to get query from parameters first
        if "query" in parameters:
            return parameters["query"]
        elif "question" in parameters:
            return parameters["question"]
        elif "input_text" in parameters:
            return parameters["input_text"]
        else:
            # Use task description as query
            return task.description

    async def _generate_rag_response(
        self, task: Task, task_type: str, query: str, rag_response: RAGFlowResponse
    ) -> RAGFlowResponse:
        """Generate a response using RAG capabilities."""
        try:
            # Get the appropriate prompt template
            prompt_template = self.rag_task_prompts.get(
                task_type, self.rag_task_prompts["rag_question_answering"]
            )

            # Format retrieved documents for the prompt
            retrieved_docs_text = "\n\n".join(
                [
                    f"Document {i+1}:\n{doc.get('content', '')}"
                    for i, doc in enumerate(rag_response.retrieved_documents)
                ]
            )

            # Prepare input data for prompt formatting
            input_data = {
                "question": query,
                "query": query,
                "retrieved_documents": retrieved_docs_text,
                "length_requirement": task.parameters.get(
                    "length_requirement", "medium"
                ),
                "focus_areas": ", ".join(task.parameters.get("focus_areas", [])),
                "style_requirement": task.parameters.get(
                    "style_requirement", "academic"
                ),
            }

            # Generate prompt
            prompt = prompt_template.format(**input_data)

            # Use LLM to generate response
            llm_response = await self.llm_interface.generate_with_system_prompt(
                system_prompt=f"You are an expert {task_type} executor with access to retrieved documents. Use the documents to provide accurate, helpful responses.",
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=1000,
            )

            # Return RAG response with LLM-generated content
            return RAGFlowResponse(
                content=llm_response.content,
                retrieved_documents=rag_response.retrieved_documents,
                confidence_score=rag_response.confidence_score,
                metadata={
                    "llm_tokens_used": llm_response.tokens_used,
                    "llm_latency": llm_response.latency,
                    **(rag_response.metadata or {}),
                },
            )

        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return RAGFlowResponse(
                content="Error generating response",
                retrieved_documents=rag_response.retrieved_documents,
                confidence_score=0.0,
            )

    def get_rag_performance_metrics(self) -> Dict[str, Any]:
        """Get RAG-specific performance metrics."""
        base_metrics = self.get_performance_metrics()

        # Calculate RAG-specific metrics
        rag_tasks = [
            result
            for result in self.task_history
            if isinstance(result, RAGExecutionResult)
        ]
        avg_confidence = 0.0
        total_retrieved_docs = 0

        if rag_tasks:
            avg_confidence = sum(task.confidence_score for task in rag_tasks) / len(
                rag_tasks
            )
            total_retrieved_docs = sum(
                len(task.retrieved_documents or []) for task in rag_tasks
            )

        return {
            **base_metrics,
            "rag_enabled": self.rag_enabled,
            "knowledge_base_id": self.knowledge_base_id,
            "rag_tasks_completed": len(rag_tasks),
            "average_confidence_score": avg_confidence,
            "total_retrieved_documents": total_retrieved_docs,
            "rag_success_rate": (
                len([t for t in rag_tasks if t.status == TaskStatus.COMPLETED])
                / len(rag_tasks)
                if rag_tasks
                else 0.0
            ),
        }
