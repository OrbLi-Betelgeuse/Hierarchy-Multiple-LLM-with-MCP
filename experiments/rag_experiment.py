"""
RAG-Enabled Experiment

Demonstrates the use of RAGFlow for enhanced question answering and research tasks
with retrieval-augmented generation capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from models.manager import Manager
from models.rag_executor import RAGExecutor
from models.llm_interface import create_llm_interface
from models.ragflow_interface import create_ragflow_interface

logger = logging.getLogger(__name__)


@dataclass
class RAGTask:
    """Represents a RAG-enabled task."""

    task_id: str
    query: str
    task_type: str  # "question_answering", "research", "summarization"
    context: Optional[str] = None
    expected_answer: Optional[str] = None


@dataclass
class RAGResult:
    """Result of RAG experiment."""

    task_id: str
    query: str
    response: str
    retrieved_documents: List[Dict[str, Any]]
    confidence_score: float
    execution_time: float
    rag_enabled: bool
    quality_score: Optional[float] = None


class RAGExperiment:
    """Experiment class for RAG-enabled tasks."""

    def __init__(self, manager_config: Dict[str, Any], ragflow_config: Dict[str, Any]):
        self.manager_config = manager_config
        self.ragflow_config = ragflow_config
        self.manager = None
        self.rag_executor = None
        self.results = []

        # Sample RAG tasks
        self.sample_tasks = [
            RAGTask(
                task_id="rag_task_1",
                query="What are the main applications of artificial intelligence in healthcare?",
                task_type="question_answering",
                expected_answer="AI applications in healthcare include diagnostic imaging, drug discovery, patient monitoring, and predictive analytics.",
            ),
            RAGTask(
                task_id="rag_task_2",
                query="How does climate change affect renewable energy adoption?",
                task_type="research",
                expected_answer="Climate change drives renewable energy adoption through policy changes, economic incentives, and technological innovation.",
            ),
            RAGTask(
                task_id="rag_task_3",
                query="Summarize the key challenges in implementing AI systems in healthcare.",
                task_type="summarization",
                expected_answer="Key challenges include data privacy, regulatory approval, ethical considerations, and validation requirements.",
            ),
        ]

        # Sample documents for knowledge base
        self.sample_documents = [
            {
                "content": """
                Artificial Intelligence (AI) has emerged as a transformative force in healthcare, 
                offering unprecedented opportunities to improve patient outcomes, enhance diagnostic 
                accuracy, and streamline healthcare delivery. The integration of AI in healthcare 
                spans multiple domains, from diagnostic imaging to drug discovery, from patient 
                monitoring to administrative efficiency.
                
                In diagnostic imaging, AI systems have achieved performance levels comparable to 
                or exceeding those of human radiologists in detecting various conditions, including 
                breast cancer, lung nodules, and retinal diseases. These systems can process vast 
                amounts of imaging data rapidly, identifying subtle patterns and anomalies that 
                might be missed in routine clinical practice.
                
                Beyond imaging, AI is revolutionizing drug discovery and development. Traditional 
                drug discovery processes are notoriously time-consuming and expensive, often taking 
                over a decade and costing billions of dollars. AI-powered approaches are accelerating 
                this process by predicting molecular interactions, identifying potential drug targets, 
                and optimizing drug candidates through virtual screening and molecular modeling.
                
                Patient monitoring and predictive analytics represent another critical application 
                area. AI systems can continuously monitor patient vital signs, detect early warning 
                signs of deterioration, and predict adverse events before they occur. This proactive 
                approach to patient care has the potential to significantly reduce mortality rates 
                and improve overall healthcare outcomes.
                
                However, the implementation of AI in healthcare is not without challenges. Data 
                privacy and security concerns remain paramount, as healthcare data is among the 
                most sensitive personal information. Ensuring compliance with regulations such as 
                HIPAA while maintaining the utility of AI systems requires careful consideration 
                of data handling practices and security measures.
                
                Another significant challenge is the need for robust validation and regulatory 
                approval processes. AI systems must demonstrate not only technical accuracy but 
                also clinical utility and safety. The regulatory landscape for AI in healthcare 
                is still evolving, requiring ongoing collaboration between developers, healthcare 
                providers, and regulatory bodies.
                
                Ethical considerations also play a crucial role in the adoption of AI in healthcare. 
                Issues of algorithmic bias, transparency in decision-making, and the potential 
                for automation to replace human judgment must be carefully addressed. Ensuring 
                that AI systems are fair, interpretable, and accountable is essential for building 
                trust among healthcare providers and patients.
                """,
                "metadata": {
                    "source": "AI Healthcare Review",
                    "topic": "healthcare_ai",
                },
            },
            {
                "content": """
                Climate change represents one of the most pressing challenges of our time, 
                requiring urgent and comprehensive action across all sectors of society. The 
                scientific consensus is clear: human activities, particularly the burning of 
                fossil fuels, are driving unprecedented changes in Earth's climate system, 
                with far-reaching consequences for ecosystems, economies, and human well-being.
                
                Addressing climate change requires a fundamental transformation of our energy 
                systems, moving away from fossil fuels toward renewable energy sources. 
                Renewable energy technologies, including solar, wind, hydroelectric, and 
                geothermal power, offer clean, sustainable alternatives to fossil fuels.
                
                Solar energy has experienced remarkable growth in recent years, with costs 
                falling dramatically and efficiency improving steadily. Photovoltaic panels 
                can now generate electricity at costs competitive with or lower than fossil 
                fuel sources in many regions. Solar energy is particularly well-suited for 
                distributed generation, allowing homes and businesses to produce their own 
                electricity.
                
                Wind energy has also become a major contributor to global electricity 
                generation. Modern wind turbines are highly efficient and can operate in 
                a wide range of conditions. Offshore wind farms, in particular, offer 
                significant potential for large-scale electricity generation with minimal 
                land use requirements.
                
                The transition to renewable energy is not only necessary for addressing 
                climate change but also offers significant economic opportunities. The 
                renewable energy sector has become a major driver of job creation and 
                economic growth, with employment in renewable energy industries growing 
                rapidly worldwide.
                
                However, the transition to renewable energy faces several challenges. 
                Renewable energy sources are often intermittent, requiring advances in 
                energy storage technology and grid management to ensure reliable electricity 
                supply. The existing energy infrastructure, built around centralized fossil 
                fuel generation, must be adapted to accommodate distributed renewable energy 
                systems.
                
                Policy support is crucial for accelerating the transition to renewable energy. 
                Governments around the world have implemented various policies to support 
                renewable energy development, including feed-in tariffs, renewable portfolio 
                standards, and carbon pricing mechanisms. These policies have been instrumental 
                in driving the growth of renewable energy markets.
                
                International cooperation is essential for addressing climate change and 
                promoting renewable energy development. The Paris Agreement, adopted in 2015, 
                provides a framework for global action on climate change, with countries 
                committing to reduce greenhouse gas emissions and support renewable energy 
                development.
                """,
                "metadata": {
                    "source": "Climate Change Report",
                    "topic": "climate_energy",
                },
            },
        ]

    async def setup(self):
        """Setup the RAG experiment with manager and RAG executor."""
        try:
            # Create manager
            manager_llm = create_llm_interface(
                provider=self.manager_config["provider"],
                model_name=self.manager_config["model"],
                **self.manager_config.get("kwargs", {}),
            )

            self.manager = Manager(
                manager_id=self.manager_config["manager_id"], llm_interface=manager_llm
            )

            # Create RAGFlow interface
            ragflow_interface = create_ragflow_interface(
                base_url=self.ragflow_config.get("base_url", "http://localhost:9380"),
                **self.ragflow_config.get("kwargs", {}),
            )

            # Create RAG executor
            executor_llm = create_llm_interface(
                provider=self.manager_config["provider"],
                model_name=self.manager_config["model"],
                **self.manager_config.get("kwargs", {}),
            )

            self.rag_executor = RAGExecutor(
                executor_id="rag_executor_01",
                llm_interface=executor_llm,
                ragflow_interface=ragflow_interface,
                capabilities=[
                    "rag_question_answering",
                    "rag_research",
                    "rag_summarization",
                    "general",
                ],
            )

            # Register executor with manager
            await self.manager.register_executor(
                executor_id="rag_executor_01",
                capabilities=[
                    "rag_question_answering",
                    "rag_research",
                    "rag_summarization",
                    "general",
                ],
            )

            # Setup knowledge base
            kb_id = await self.rag_executor.setup_knowledge_base(
                name="experiment_kb", description="Knowledge base for RAG experiment"
            )

            if kb_id:
                # Upload sample documents
                await self.rag_executor.upload_documents_to_kb(self.sample_documents)
                logger.info("Knowledge base setup complete with sample documents")
            else:
                logger.warning("Knowledge base setup failed, will use fallback mode")

            logger.info("RAG experiment setup complete")

        except Exception as e:
            logger.error(f"Error setting up RAG experiment: {e}")
            raise

    async def run_single_task(self, task: RAGTask) -> RAGResult:
        """Run a single RAG task."""
        try:
            from models.mcp_protocol import Task

            # Create task for executor
            executor_task = Task(
                task_id=task.task_id,
                task_type=f"rag_{task.task_type}",
                description=task.query,
                parameters={
                    "query": task.query,
                    "context": task.context or "",
                    "task_type": task.task_type,
                },
                priority=1,
                dependencies=[],
                assigned_executor="rag_executor_01",
            )

            # Execute task
            result = await self.rag_executor.execute_rag_task(executor_task)

            # Calculate quality score if expected answer is available
            quality_score = None
            if task.expected_answer:
                quality_score = self._calculate_quality_score(
                    result.result.get("output", ""), task.expected_answer
                )

            return RAGResult(
                task_id=task.task_id,
                query=task.query,
                response=result.result.get("output", ""),
                retrieved_documents=result.retrieved_documents or [],
                confidence_score=result.confidence_score,
                execution_time=result.execution_time,
                rag_enabled=result.result.get("rag_enabled", False),
                quality_score=quality_score,
            )

        except Exception as e:
            logger.error(f"Error running RAG task {task.task_id}: {e}")
            raise

    async def run_experiment(
        self, tasks: Optional[List[RAGTask]] = None
    ) -> List[RAGResult]:
        """Run the complete RAG experiment."""
        if tasks is None:
            tasks = self.sample_tasks

        results = []
        for task in tasks:
            try:
                result = await self.run_single_task(task)
                results.append(result)
                logger.info(f"Completed RAG task {task.task_id}")
            except Exception as e:
                logger.error(f"Failed to complete RAG task {task.task_id}: {e}")

        self.results = results
        return results

    def _calculate_quality_score(
        self, generated_response: str, expected_answer: str
    ) -> float:
        """Calculate a simple quality score based on content overlap."""
        generated_words = set(generated_response.lower().split())
        expected_words = set(expected_answer.lower().split())

        if not expected_words:
            return 0.0

        overlap = len(generated_words.intersection(expected_words))
        return overlap / len(expected_words)

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of the RAG experiment results."""
        if not self.results:
            return {"error": "No results available"}

        # Calculate aggregate metrics
        total_tasks = len(self.results)
        successful_tasks = len([r for r in self.results if r.quality_score is not None])

        avg_execution_time = sum(r.execution_time for r in self.results) / total_tasks
        avg_confidence = sum(r.confidence_score for r in self.results) / total_tasks
        avg_quality_score = (
            sum(r.quality_score for r in self.results if r.quality_score is not None)
            / successful_tasks
            if successful_tasks > 0
            else 0.0
        )

        rag_enabled_tasks = len([r for r in self.results if r.rag_enabled])
        total_retrieved_docs = sum(len(r.retrieved_documents) for r in self.results)

        return {
            "experiment_type": "rag_experiment",
            "summary": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "rag_enabled_tasks": rag_enabled_tasks,
                "success_rate": (
                    successful_tasks / total_tasks if total_tasks > 0 else 0.0
                ),
                "rag_enabled_rate": (
                    rag_enabled_tasks / total_tasks if total_tasks > 0 else 0.0
                ),
            },
            "performance_metrics": {
                "average_execution_time": avg_execution_time,
                "average_confidence_score": avg_confidence,
                "average_quality_score": avg_quality_score,
                "total_retrieved_documents": total_retrieved_docs,
                "average_retrieved_documents_per_task": (
                    total_retrieved_docs / total_tasks if total_tasks > 0 else 0.0
                ),
            },
            "task_results": [
                {
                    "task_id": result.task_id,
                    "query": result.query,
                    "response_length": len(result.response),
                    "retrieved_documents_count": len(result.retrieved_documents),
                    "confidence_score": result.confidence_score,
                    "execution_time": result.execution_time,
                    "rag_enabled": result.rag_enabled,
                    "quality_score": result.quality_score,
                }
                for result in self.results
            ],
            "rag_performance": (
                self.rag_executor.get_rag_performance_metrics()
                if self.rag_executor
                else {}
            ),
        }
