"""
Long Document Summarization Experiment

Evaluates the Manager-Executor collaboration model on long document summarization tasks.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from models.manager import Manager
from models.executor import Executor, SpecializedExecutor
from models.llm_interface import create_llm_interface

logger = logging.getLogger(__name__)


@dataclass
class SummarizationTask:
    """Represents a summarization task."""

    task_id: str
    document: str
    target_length: str  # "short", "medium", "long"
    focus_areas: List[str]
    style: str  # "academic", "casual", "professional"
    expected_summary: Optional[str] = None


@dataclass
class SummarizationResult:
    """Result of summarization experiment."""

    task_id: str
    original_length: int
    summary_length: int
    compression_ratio: float
    execution_time: float
    manager_metrics: Dict[str, Any]
    executor_metrics: Dict[str, Any]
    quality_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None


class SummarizationExperiment:
    """Experiment class for long document summarization."""

    def __init__(
        self, manager_config: Dict[str, Any], executor_configs: List[Dict[str, Any]]
    ):
        self.manager_config = manager_config
        self.executor_configs = executor_configs
        self.manager = None
        self.executors = []
        self.results = []

        # Sample documents for testing
        self.sample_documents = [
            {
                "id": "doc_1",
                "title": "Artificial Intelligence in Healthcare",
                "content": """
                Artificial Intelligence (AI) has emerged as a transformative force in healthcare, 
                offering unprecedented opportunities to improve patient outcomes, enhance diagnostic 
                accuracy, and streamline healthcare delivery. This comprehensive analysis explores 
                the current state of AI applications in healthcare, examining both the remarkable 
                advances and the significant challenges that lie ahead.

                The integration of AI in healthcare spans multiple domains, from diagnostic imaging 
                to drug discovery, from patient monitoring to administrative efficiency. Machine 
                learning algorithms, particularly deep learning models, have demonstrated remarkable 
                capabilities in analyzing medical images, detecting patterns that might elude human 
                observers, and providing quantitative assessments of disease progression.

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

                The future of AI in healthcare holds immense promise, but realizing this potential 
                requires addressing these challenges systematically. Collaboration between technology 
                developers, healthcare professionals, policymakers, and patients will be essential 
                for creating AI systems that truly enhance healthcare delivery while maintaining 
                the highest standards of safety, efficacy, and ethical practice.

                As we move forward, it is crucial to maintain a balanced perspective that recognizes 
                both the transformative potential of AI and the importance of human expertise and 
                judgment in healthcare. The most successful implementations of AI in healthcare 
                will be those that augment rather than replace human capabilities, creating 
                collaborative systems that leverage the strengths of both artificial and human 
                intelligence.
                """,
                "expected_summary": "AI is transforming healthcare through improved diagnostics, drug discovery, and patient monitoring, but faces challenges in privacy, validation, and ethical implementation.",
            },
            {
                "id": "doc_2",
                "title": "Climate Change and Renewable Energy",
                "content": """
                Climate change represents one of the most pressing challenges of our time, 
                requiring urgent and comprehensive action across all sectors of society. The 
                scientific consensus is clear: human activities, particularly the burning of 
                fossil fuels, are driving unprecedented changes in Earth's climate system, 
                with far-reaching consequences for ecosystems, economies, and human well-being.

                The evidence of climate change is overwhelming and continues to mount. Global 
                temperatures have risen by approximately 1.1°C since pre-industrial times, 
                with the rate of warming accelerating in recent decades. This warming is 
                causing widespread and rapid changes in the natural world, including melting 
                glaciers and ice sheets, rising sea levels, and shifts in weather patterns.

                The impacts of climate change are already being felt around the world. Extreme 
                weather events, including hurricanes, droughts, floods, and heatwaves, are 
                becoming more frequent and intense. These events cause significant damage to 
                infrastructure, agriculture, and human settlements, resulting in economic 
                losses and human suffering.

                Rising sea levels threaten coastal communities and island nations, while 
                ocean acidification endangers marine ecosystems and the livelihoods that 
                depend on them. Changes in precipitation patterns affect water availability 
                and agricultural productivity, with implications for food security and 
                economic stability.

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

                Hydroelectric power remains the largest renewable energy source globally, 
                providing reliable, dispatchable electricity generation. While large-scale 
                hydroelectric projects can have significant environmental and social impacts, 
                small-scale hydroelectric systems offer opportunities for sustainable energy 
                generation with minimal environmental disruption.

                Geothermal energy, which harnesses heat from Earth's interior, provides 
                a constant, reliable source of energy that can be used for both electricity 
                generation and direct heating applications. While currently limited to 
                specific geographic regions, advances in technology may expand the potential 
                for geothermal energy development.

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

                Technological innovation continues to drive down costs and improve the 
                performance of renewable energy technologies. Research and development 
                efforts focus on improving efficiency, reducing costs, and developing 
                new applications for renewable energy technologies.

                The transition to renewable energy also requires changes in consumer behavior 
                and business practices. Energy efficiency measures, such as improved building 
                insulation and energy-efficient appliances, can significantly reduce energy 
                demand and complement renewable energy deployment.

                International cooperation is essential for addressing climate change and 
                promoting renewable energy development. The Paris Agreement, adopted in 2015, 
                provides a framework for global action on climate change, with countries 
                committing to reduce greenhouse gas emissions and support renewable energy 
                development.

                The urgency of addressing climate change requires immediate and sustained 
                action. While the challenges are significant, the benefits of transitioning 
                to renewable energy are clear: reduced greenhouse gas emissions, improved 
                air quality, enhanced energy security, and economic opportunities. The 
                transition to renewable energy represents not just a challenge but an 
                opportunity to create a more sustainable, equitable, and prosperous future.
                """,
                "expected_summary": "Climate change requires urgent action through renewable energy transition, with solar, wind, and other technologies offering economic and environmental benefits despite challenges in storage and infrastructure.",
            },
        ]

    async def setup(self):
        """Setup the experiment with manager and executors. Supports executor-only mode."""
        try:
            # Only set up manager if manager_config is provided
            if self.manager_config is not None:
                manager_llm = create_llm_interface(
                    provider=self.manager_config["provider"],
                    model_name=self.manager_config["model"],
                    **self.manager_config.get("kwargs", {}),
                )
                self.manager = Manager(
                    manager_id=self.manager_config["manager_id"], llm_interface=manager_llm
                )
            else:
                self.manager = None

            # Create executors
            for config in self.executor_configs:
                executor_llm = create_llm_interface(
                    provider=config["provider"],
                    model_name=config["model"],
                    **config.get("kwargs", {}),
                )

                if config.get("specialized", False):
                    executor = SpecializedExecutor(
                        executor_id=config["executor_id"],
                        llm_interface=executor_llm,
                        task_type="summarization",
                        specialized_capabilities=config["capabilities"],
                    )
                else:
                    executor = Executor(
                        executor_id=config["executor_id"],
                        llm_interface=executor_llm,
                        capabilities=config["capabilities"],
                    )

                self.executors.append(executor)

                # Register executor with manager if present
                if self.manager is not None:
                    await self.manager.register_executor(
                        executor_id=config["executor_id"],
                        capabilities=config["capabilities"],
                    )

            logger.info(
                f"Experiment setup complete: {('1 manager, ' if self.manager else '0 manager, ')}{len(self.executors)} executors"
            )

        except Exception as e:
            logger.error(f"Error setting up experiment: {e}")
            raise

    async def run_single_task(self, task: SummarizationTask) -> SummarizationResult:
        """Run a single summarization task, supporting executor-only mode."""
        try:
            start_time = asyncio.get_event_loop().time()

            # Create task description for manager/executor
            task_description = f"""
            Summarize the following document:
            
            Title: {task.task_id}
            Target Length: {task.target_length}
            Focus Areas: {', '.join(task.focus_areas)}
            Style: {task.style}
            
            Document:
            {task.document}
            """

            if self.manager is None:
                # Executor-only mode: send the whole task to the first executor
                if not self.executors:
                    raise RuntimeError("No executors available")
                executor = self.executors[0]
                from models.executor import Task
                task_obj = Task(
                    task_id=f"exec_{task.task_id}",
                    description=task_description,
                    parameters={},
                    task_type="summarization"
                )
                result = await executor.execute_task(task_obj)
                # Assume result is a dict or has .result
                if hasattr(result, "result"):
                    summary = result.result.get("output", "")
                elif isinstance(result, dict):
                    summary = result.get("output", "")
                else:
                    summary = str(result) if result else ""
                manager_metrics = {}
                executor_metrics = {getattr(executor, 'executor_id', 'executor'): executor.get_performance_metrics()}
            else:
                # Execute task using manager-executor system
                result = await self.manager.execute_task(task_description, "summarization")
                summary = ""
                if result.get("task_results"):
                    for task_result in result["task_results"].values():
                        if "output" in task_result:
                            summary += task_result["output"] + "\n"
                manager_metrics = self.manager.get_performance_metrics()
                executor_metrics = {}
                for executor in self.executors:
                    executor_metrics[executor.executor_id] = (
                        executor.get_performance_metrics()
                    )

            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time

            # Calculate metrics
            original_length = len(task.document.split())
            summary_length = len(summary.split())
            compression_ratio = (
                summary_length / original_length if original_length > 0 else 0
            )

            # Calculate quality score if expected summary is available
            quality_score = None
            rouge_scores = None
            if task.expected_summary:
                quality_score = self._calculate_quality_score(
                    summary, task.expected_summary
                )
                rouge_scores = self._calculate_rouge_scores(
                    summary, task.expected_summary
                )

            return SummarizationResult(
                task_id=task.task_id,
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=compression_ratio,
                execution_time=execution_time,
                manager_metrics=manager_metrics,
                executor_metrics=executor_metrics,
                quality_score=quality_score,
                rouge_scores=rouge_scores,
            )

        except Exception as e:
            logger.error(f"Error running task {task.task_id}: {e}")
            raise

    async def run_experiment(
        self, tasks: Optional[List[SummarizationTask]] = None
    ) -> List[SummarizationResult]:
        """Run the complete summarization experiment."""
        if tasks is None:
            # Create tasks from sample documents
            tasks = []
            for doc in self.sample_documents:
                task = SummarizationTask(
                    task_id=doc["id"],
                    document=doc["content"],
                    target_length="medium",
                    focus_areas=["key_points", "main_arguments"],
                    style="academic",
                    expected_summary=doc.get("expected_summary"),
                )
                tasks.append(task)

        results = []
        for task in tasks:
            try:
                result = await self.run_single_task(task)
                results.append(result)
                logger.info(f"Completed task {task.task_id}")
            except Exception as e:
                logger.error(f"Failed to complete task {task.task_id}: {e}")

        self.results = results
        return results

    def _calculate_quality_score(
        self, generated_summary: str, expected_summary: str
    ) -> float:
        """Calculate a simple quality score based on content overlap."""
        # Simple implementation - in practice, this could use more sophisticated metrics
        generated_words = set(generated_summary.lower().split())
        expected_words = set(expected_summary.lower().split())

        if not expected_words:
            return 0.0

        overlap = len(generated_words.intersection(expected_words))
        return overlap / len(expected_words)

    def _calculate_rouge_scores(
        self, generated_summary: str, expected_summary: str
    ) -> Dict[str, float]:
        """Calculate ROUGE scores for summarization quality."""
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            scores = scorer.score(expected_summary, generated_summary)

            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        except ImportError:
            logger.warning("rouge-score not available, skipping ROUGE calculation")
            return {}

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of the experiment results."""
        if not self.results:
            return {"error": "No results available"}

        # Calculate aggregate metrics
        total_tasks = len(self.results)
        successful_tasks = len([r for r in self.results if r.quality_score is not None])

        avg_execution_time = sum(r.execution_time for r in self.results) / total_tasks
        avg_compression_ratio = (
            sum(r.compression_ratio for r in self.results) / total_tasks
        )

        avg_quality_score = 0.0
        avg_rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        if successful_tasks > 0:
            avg_quality_score = (
                sum(
                    r.quality_score for r in self.results if r.quality_score is not None
                )
                / successful_tasks
            )

            # Calculate average ROUGE scores
            rouge_counts = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
            for result in self.results:
                if result.rouge_scores:
                    for metric in avg_rouge_scores:
                        if metric in result.rouge_scores:
                            avg_rouge_scores[metric] += result.rouge_scores[metric]
                            rouge_counts[metric] += 1

            for metric in avg_rouge_scores:
                if rouge_counts[metric] > 0:
                    avg_rouge_scores[metric] /= rouge_counts[metric]

        return {
            "experiment_type": "summarization",
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks,
            "average_execution_time": avg_execution_time,
            "average_compression_ratio": avg_compression_ratio,
            "average_quality_score": avg_quality_score,
            "average_rouge_scores": avg_rouge_scores,
            "detailed_results": [
                {
                    "task_id": r.task_id,
                    "execution_time": r.execution_time,
                    "compression_ratio": r.compression_ratio,
                    "quality_score": r.quality_score,
                    "rouge_scores": r.rouge_scores,
                }
                for r in self.results
            ],
        }
