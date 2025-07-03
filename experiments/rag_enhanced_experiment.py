"""
Enhanced RAG Experiment (Experiment D)

Comprehensive comparison between RAG-enhanced and standard LLM approaches
for question answering, summarization, and research tasks.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from difflib import SequenceMatcher

from models.manager import Manager
from models.executor import Executor
from models.rag_executor import RAGExecutor
from models.llm_interface import create_llm_interface
from models.ragflow_interface import create_ragflow_interface

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class RAGComparisonTask:
    """Task for comparing RAG vs non-RAG performance."""

    task_id: str
    query: str
    task_type: str  # "question_answering", "research", "summarization"
    context: Optional[str] = None
    expected_answer: Optional[str] = None
    difficulty_level: str = "medium"  # "easy", "medium", "hard"


@dataclass
class RAGComparisonResult:
    """Result comparing RAG vs non-RAG performance."""

    task_id: str
    query: str
    task_type: str
    rag_response: str
    rag_retrieved_documents: List[Dict[str, Any]]
    rag_confidence_score: float
    rag_execution_time: float
    non_rag_response: str
    non_rag_execution_time: float
    difficulty_level: str
    rag_quality_score: Optional[float] = None
    non_rag_quality_score: Optional[float] = None
    quality_improvement: Optional[float] = None
    time_overhead: Optional[float] = None
    rag_advantage: Optional[bool] = None


class RAGEnhancedExperiment:
    """Enhanced RAG experiment with comprehensive comparison."""

    def __init__(self, manager_config: Dict[str, Any], ragflow_config: Dict[str, Any]):
        self.manager_config = manager_config
        self.ragflow_config = ragflow_config
        self.manager = None
        self.rag_executor = None
        self.standard_executor = None
        self.results = []

        # Enhanced sample tasks with varying difficulty
        self.sample_tasks = [
            RAGComparisonTask(
                task_id="rag_comp_1",
                query="What are the main applications of artificial intelligence in healthcare?",
                task_type="question_answering",
                difficulty_level="easy",
                expected_answer="AI applications in healthcare include diagnostic imaging, drug discovery, patient monitoring, and predictive analytics.",
            ),
            RAGComparisonTask(
                task_id="rag_comp_2",
                query="How does climate change affect renewable energy adoption?",
                task_type="research",
                difficulty_level="medium",
                expected_answer="Climate change drives renewable energy adoption through policy changes, economic incentives, and technological innovation.",
            ),
            RAGComparisonTask(
                task_id="rag_comp_3",
                query="Summarize the key challenges in implementing AI systems in healthcare.",
                task_type="summarization",
                difficulty_level="medium",
                expected_answer="Key challenges include data privacy, regulatory approval, ethical considerations, and validation requirements.",
            ),
            RAGComparisonTask(
                task_id="rag_comp_4",
                query="What are the latest developments in quantum computing for machine learning?",
                task_type="research",
                difficulty_level="hard",
                expected_answer="Recent developments include quantum neural networks, quantum feature maps, and hybrid quantum-classical algorithms.",
            ),
            RAGComparisonTask(
                task_id="rag_comp_5",
                query="Explain the impact of blockchain technology on supply chain management.",
                task_type="question_answering",
                difficulty_level="medium",
                expected_answer="Blockchain improves supply chain transparency, traceability, and security through decentralized record-keeping.",
            ),
        ]

        # Comprehensive knowledge base documents
        self.knowledge_base_documents = [
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
                    "date": "2024",
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
                    "date": "2024",
                },
            },
            {
                "content": """
                Quantum computing represents a paradigm shift in computational capabilities, 
                leveraging the principles of quantum mechanics to perform calculations that 
                would be infeasible for classical computers. While still in its early stages, 
                quantum computing has shown promising applications in various fields, including 
                machine learning, cryptography, and optimization.
                
                In the context of machine learning, quantum computing offers several potential 
                advantages. Quantum neural networks can theoretically process information in 
                ways that classical neural networks cannot, potentially leading to more efficient 
                training and better generalization. Quantum feature maps can transform classical 
                data into quantum states, enabling the use of quantum algorithms for classical 
                machine learning problems.
                
                Hybrid quantum-classical algorithms represent a practical approach to leveraging 
                quantum computing for machine learning. These algorithms combine quantum and 
                classical components, using quantum computers for specific subroutines while 
                keeping the overall algorithm structure classical. This approach allows for 
                the use of current quantum hardware while providing a path toward fully 
                quantum machine learning systems.
                
                Quantum machine learning applications include quantum support vector machines, 
                quantum principal component analysis, and quantum clustering algorithms. These 
                algorithms can potentially provide speedups for certain types of problems, 
                particularly those involving large datasets or complex optimization tasks.
                
                However, quantum machine learning faces several challenges. Current quantum 
                hardware is limited in terms of qubit count, coherence time, and error rates. 
                These limitations restrict the size and complexity of problems that can be 
                solved on quantum computers. Additionally, the development of quantum machine 
                learning algorithms requires expertise in both quantum computing and machine 
                learning, creating a barrier to entry for researchers and practitioners.
                
                Despite these challenges, significant progress has been made in quantum 
                machine learning. Research institutions and technology companies are actively 
                developing quantum machine learning frameworks and algorithms. Cloud-based 
                quantum computing platforms have made quantum hardware more accessible to 
                researchers and developers.
                
                The future of quantum machine learning depends on advances in quantum hardware, 
                algorithm development, and the identification of practical applications where 
                quantum computing provides clear advantages over classical approaches.
                """,
                "metadata": {
                    "source": "Quantum Computing Review",
                    "topic": "quantum_ml",
                    "date": "2024",
                },
            },
            {
                "content": """
                Blockchain technology has emerged as a revolutionary innovation with the 
                potential to transform various industries, particularly supply chain management. 
                At its core, blockchain is a decentralized, distributed ledger technology 
                that enables secure, transparent, and tamper-proof record-keeping.
                
                In supply chain management, blockchain offers several key advantages. 
                Transparency is one of the most significant benefits, as all participants 
                in the supply chain can access the same information in real-time. This 
                transparency helps reduce fraud, improve compliance, and build trust among 
                supply chain partners.
                
                Traceability is another critical advantage of blockchain in supply chains. 
                Each product can be tracked from its origin through every step of the supply 
                chain, providing a complete audit trail. This capability is particularly 
                valuable for industries such as food and pharmaceuticals, where product 
                safety and authenticity are paramount.
                
                Security is enhanced through blockchain's cryptographic features. Once 
                information is recorded on the blockchain, it cannot be altered without 
                consensus from the network participants. This immutability ensures the 
                integrity of supply chain data and reduces the risk of fraud or manipulation.
                
                Smart contracts, which are self-executing contracts with the terms directly 
                written into code, can automate various supply chain processes. These contracts 
                can automatically trigger payments, verify compliance, and enforce agreements 
                when predefined conditions are met.
                
                Despite these advantages, blockchain implementation in supply chains faces 
                several challenges. Scalability remains a significant concern, as current 
                blockchain networks may not be able to handle the high transaction volumes 
                required by large supply chains. Energy consumption is another issue, 
                particularly for proof-of-work consensus mechanisms.
                
                Interoperability between different blockchain platforms and existing 
                supply chain systems is also a challenge. Standardization efforts are 
                ongoing to address this issue and enable seamless integration across 
                different platforms and systems.
                
                Cost considerations are important when implementing blockchain solutions. 
                While blockchain can reduce costs in the long term through improved 
                efficiency and reduced fraud, the initial implementation costs can be 
                significant. Organizations must carefully evaluate the return on investment 
                and develop a clear implementation strategy.
                
                The future of blockchain in supply chain management depends on addressing 
                these challenges and demonstrating clear value propositions for different 
                industries and use cases.
                """,
                "metadata": {
                    "source": "Blockchain Supply Chain Analysis",
                    "topic": "blockchain_supply_chain",
                    "date": "2024",
                },
            },
        ]

    async def setup(self):
        """Setup the enhanced RAG experiment."""
        try:
            console.print(
                Panel.fit("Setting up Enhanced RAG Experiment", style="bold blue")
            )

            # Create manager
            manager_llm = create_llm_interface(
                provider=self.manager_config["provider"],
                model_name=self.manager_config["model"],
                **self.manager_config.get("kwargs", {}),
            )

            self.manager = Manager(
                manager_id=self.manager_config["manager_id"], llm_interface=manager_llm
            )

            # Create standard executor
            standard_llm = create_llm_interface(
                provider=self.manager_config["provider"],
                model_name=self.manager_config["model"],
                **self.manager_config.get("kwargs", {}),
            )

            self.standard_executor = Executor(
                executor_id="standard_executor_01",
                llm_interface=standard_llm,
                capabilities=[
                    "question_answering",
                    "research",
                    "summarization",
                    "general",
                ],
            )

            # Create RAGFlow interface
            ragflow_interface = create_ragflow_interface(
                base_url=self.ragflow_config.get("base_url", "http://localhost:9380"),
                **self.ragflow_config.get("kwargs", {}),
            )

            # Create RAG executor
            rag_llm = create_llm_interface(
                provider=self.manager_config["provider"],
                model_name=self.manager_config["model"],
                **self.manager_config.get("kwargs", {}),
            )

            self.rag_executor = RAGExecutor(
                executor_id="rag_executor_01",
                llm_interface=rag_llm,
                ragflow_interface=ragflow_interface,
                capabilities=[
                    "rag_question_answering",
                    "rag_research",
                    "rag_summarization",
                    "general",
                ],
            )

            # Register executors with manager
            await self.manager.register_executor(
                executor_id="standard_executor_01",
                capabilities=[
                    "question_answering",
                    "research",
                    "summarization",
                    "general",
                ],
            )

            await self.manager.register_executor(
                executor_id="rag_executor_01",
                capabilities=[
                    "rag_question_answering",
                    "rag_research",
                    "rag_summarization",
                    "general",
                ],
            )

            # Setup knowledge base for RAG
            console.print("🔧 Setting up knowledge base...")
            kb_id = await self.rag_executor.setup_knowledge_base(
                name="enhanced_rag_experiment_kb",
                description="Knowledge base for enhanced RAG experiment",
            )

            if kb_id:
                # Upload documents
                upload_success = await self.rag_executor.upload_documents_to_kb(
                    self.knowledge_base_documents
                )
                if upload_success:
                    console.print("✅ Knowledge base setup complete with documents")
                else:
                    console.print("⚠️ Document upload failed, will use fallback mode")
            else:
                console.print("⚠️ Knowledge base setup failed, will use fallback mode")

            console.print("✅ Enhanced RAG experiment setup complete")

        except Exception as e:
            console.print(f"❌ Error setting up RAG experiment: {e}", style="bold red")
            logger.error(f"Error setting up RAG experiment: {e}")
            raise

    async def run_single_comparison(
        self, task: RAGComparisonTask
    ) -> RAGComparisonResult:
        """Run a single task comparing RAG vs non-RAG performance."""
        try:
            from models.mcp_protocol import Task

            console.print(f"\n🔄 Running comparison for task: {task.task_id}")

            # Create task objects
            standard_task = Task(
                task_id=f"standard_{task.task_id}",
                task_type=task.task_type,
                description=task.query,
                parameters={"query": task.query, "context": task.context or ""},
                priority=1,
                dependencies=[],
                assigned_executor="standard_executor_01",
            )

            rag_task = Task(
                task_id=f"rag_{task.task_id}",
                task_type=f"rag_{task.task_type}",
                description=task.query,
                parameters={"query": task.query, "context": task.context or ""},
                priority=1,
                dependencies=[],
                assigned_executor="rag_executor_01",
            )

            # Execute standard task
            console.print("📝 Executing standard (non-RAG) task...")
            standard_start = asyncio.get_event_loop().time()
            standard_result = await self.standard_executor.execute_task(standard_task)
            standard_time = asyncio.get_event_loop().time() - standard_start

            # Execute RAG task
            console.print("🔍 Executing RAG-enhanced task...")
            rag_start = asyncio.get_event_loop().time()
            rag_result = await self.rag_executor.execute_rag_task(rag_task)
            rag_time = asyncio.get_event_loop().time() - rag_start

            # --- Robust fallback handling ---
            def safe_get(obj, attr, default):
                return getattr(obj, attr, default)

            # For rag_result
            rag_output = safe_get(
                rag_result.result,
                "get",
                lambda k, d=None: (
                    rag_result.result.get(k, d)
                    if isinstance(rag_result.result, dict)
                    else d
                ),
            )("output", "")
            rag_retrieved_documents = (
                safe_get(rag_result, "retrieved_documents", []) or []
            )
            rag_confidence_score = safe_get(rag_result, "confidence_score", 0.0)

            # For standard_result
            standard_output = safe_get(
                standard_result.result,
                "get",
                lambda k, d=None: (
                    standard_result.result.get(k, d)
                    if isinstance(standard_result.result, dict)
                    else d
                ),
            )("output", "")

            # Calculate quality scores
            standard_quality = None
            rag_quality = None
            if task.expected_answer:
                standard_quality = self._calculate_quality_score(
                    standard_output, task.expected_answer
                )
                rag_quality = self._calculate_quality_score(
                    rag_output, task.expected_answer
                )

            # Calculate comparison metrics
            quality_improvement = None
            time_overhead = None
            rag_advantage = None

            if standard_quality is not None and rag_quality is not None:
                quality_improvement = rag_quality - standard_quality
                time_overhead = rag_time - standard_time
                rag_advantage = quality_improvement > 0

            # Create comparison result
            result = RAGComparisonResult(
                task_id=task.task_id,
                query=task.query,
                task_type=task.task_type,
                rag_response=rag_output,
                rag_retrieved_documents=rag_retrieved_documents,
                rag_confidence_score=rag_confidence_score,
                rag_execution_time=rag_time,
                non_rag_response=standard_output,
                non_rag_execution_time=standard_time,
                difficulty_level=task.difficulty_level,
                rag_quality_score=rag_quality,
                non_rag_quality_score=standard_quality,
                quality_improvement=quality_improvement,
                time_overhead=time_overhead,
                rag_advantage=rag_advantage,
            )

            # Display results
            self._display_comparison_result(result)

            return result

        except Exception as e:
            console.print(
                f"❌ Error in comparison task {task.task_id}: {e}", style="bold red"
            )
            logger.error(f"Error in comparison task {task.task_id}: {e}")
            raise

    async def run_experiment(
        self, tasks: Optional[List[RAGComparisonTask]] = None
    ) -> List[RAGComparisonResult]:
        """Run the complete enhanced RAG experiment."""
        if tasks is None:
            tasks = self.sample_tasks

        console.print(Panel.fit("Running Enhanced RAG Experiment", style="bold green"))

        results = []
        for i, task in enumerate(tasks, 1):
            try:
                console.print(
                    f"\n📊 Task {i}/{len(tasks)}: {task.task_type} ({task.difficulty_level})"
                )
                result = await self.run_single_comparison(task)
                results.append(result)
            except Exception as e:
                console.print(
                    f"❌ Failed to complete task {task.task_id}: {e}", style="bold red"
                )
                logger.error(f"Failed to complete task {task.task_id}: {e}")

        self.results = results
        return results

    def _calculate_quality_score(
        self, generated_response: str, expected_answer: str
    ) -> float:
        """Calculate quality score based on content similarity."""
        if not generated_response or not expected_answer:
            return 0.0

        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(
            None, generated_response.lower(), expected_answer.lower()
        ).ratio()

        # Additional scoring based on keyword overlap
        generated_words = set(generated_response.lower().split())
        expected_words = set(expected_answer.lower().split())

        if not expected_words:
            return similarity

        keyword_overlap = len(generated_words.intersection(expected_words)) / len(
            expected_words
        )

        # Combine similarity and keyword overlap
        return (similarity + keyword_overlap) / 2

    def _display_comparison_result(self, result: RAGComparisonResult):
        """Display a single comparison result."""
        console.print(f"\n📋 Task: {result.task_id}")
        console.print(f"🔍 Query: {result.query}")

        # Quality comparison
        if result.quality_improvement is not None:
            improvement_text = f"{result.quality_improvement:+.3f}"
            improvement_style = "green" if result.quality_improvement > 0 else "red"
            console.print(
                f"📈 Quality Improvement: [{improvement_style}]{improvement_text}[/{improvement_style}]"
            )

        # Time comparison
        if result.time_overhead is not None:
            overhead_text = f"{result.time_overhead:+.2f}s"
            overhead_style = "red" if result.time_overhead > 0 else "green"
            console.print(
                f"⏱️ Time Overhead: [{overhead_style}]{overhead_text}[/{overhead_style}]"
            )

        # RAG advantage
        if result.rag_advantage is not None:
            advantage_text = (
                "✅ RAG Advantage" if result.rag_advantage else "❌ No RAG Advantage"
            )
            advantage_style = "green" if result.rag_advantage else "red"
            console.print(
                f"🎯 Result: [{advantage_style}]{advantage_text}[/{advantage_style}]"
            )

        # Retrieved documents
        if result.rag_retrieved_documents:
            console.print(
                f"📚 Retrieved Documents: {len(result.rag_retrieved_documents)}"
            )
            console.print(f"🎯 RAG Confidence: {result.rag_confidence_score:.3f}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of the enhanced RAG experiment."""
        if not self.results:
            return {"error": "No results available"}

        # Calculate aggregate metrics
        total_tasks = len(self.results)
        successful_comparisons = len(
            [r for r in self.results if r.quality_improvement is not None]
        )

        # RAG performance metrics
        rag_quality_scores = [
            r.rag_quality_score for r in self.results if r.rag_quality_score is not None
        ]
        non_rag_quality_scores = [
            r.non_rag_quality_score
            for r in self.results
            if r.non_rag_quality_score is not None
        ]

        avg_rag_quality = (
            sum(rag_quality_scores) / len(rag_quality_scores)
            if rag_quality_scores
            else 0.0
        )
        avg_non_rag_quality = (
            sum(non_rag_quality_scores) / len(non_rag_quality_scores)
            if non_rag_quality_scores
            else 0.0
        )

        # Time metrics
        avg_rag_time = sum(r.rag_execution_time for r in self.results) / total_tasks
        avg_non_rag_time = (
            sum(r.non_rag_execution_time for r in self.results) / total_tasks
        )

        # RAG advantage analysis
        rag_advantages = [r for r in self.results if r.rag_advantage is True]
        rag_advantage_rate = (
            len(rag_advantages) / successful_comparisons
            if successful_comparisons > 0
            else 0.0
        )

        # Quality improvements
        quality_improvements = [
            r.quality_improvement
            for r in self.results
            if r.quality_improvement is not None
        ]
        avg_quality_improvement = (
            sum(quality_improvements) / len(quality_improvements)
            if quality_improvements
            else 0.0
        )

        # Time overheads
        time_overheads = [
            r.time_overhead for r in self.results if r.time_overhead is not None
        ]
        avg_time_overhead = (
            sum(time_overheads) / len(time_overheads) if time_overheads else 0.0
        )

        # Task type analysis
        task_type_results = {}
        for task_type in set(r.task_type for r in self.results):
            type_results = [r for r in self.results if r.task_type == task_type]
            type_advantages = [r for r in type_results if r.rag_advantage is True]
            task_type_results[task_type] = {
                "total_tasks": len(type_results),
                "rag_advantages": len(type_advantages),
                "advantage_rate": (
                    len(type_advantages) / len(type_results) if type_results else 0.0
                ),
                "avg_quality_improvement": (
                    sum(
                        r.quality_improvement
                        for r in type_results
                        if r.quality_improvement is not None
                    )
                    / len(type_results)
                    if type_results
                    else 0.0
                ),
            }

        # Difficulty level analysis
        difficulty_results = {}
        for difficulty in set(r.difficulty_level for r in self.results):
            diff_results = [r for r in self.results if r.difficulty_level == difficulty]
            diff_advantages = [r for r in diff_results if r.rag_advantage is True]
            difficulty_results[difficulty] = {
                "total_tasks": len(diff_results),
                "rag_advantages": len(diff_advantages),
                "advantage_rate": (
                    len(diff_advantages) / len(diff_results) if diff_results else 0.0
                ),
                "avg_quality_improvement": (
                    sum(
                        r.quality_improvement
                        for r in diff_results
                        if r.quality_improvement is not None
                    )
                    / len(diff_results)
                    if diff_results
                    else 0.0
                ),
            }

        return {
            "experiment_type": "rag_enhanced_experiment",
            "summary": {
                "total_tasks": total_tasks,
                "successful_comparisons": successful_comparisons,
                "rag_advantage_rate": rag_advantage_rate,
                "overall_quality_improvement": avg_quality_improvement,
                "overall_time_overhead": avg_time_overhead,
            },
            "performance_metrics": {
                "average_rag_quality": avg_rag_quality,
                "average_non_rag_quality": avg_non_rag_quality,
                "average_rag_execution_time": avg_rag_time,
                "average_non_rag_execution_time": avg_non_rag_time,
                "quality_improvement_percentage": (
                    (avg_quality_improvement / avg_non_rag_quality * 100)
                    if avg_non_rag_quality > 0
                    else 0.0
                ),
                "time_overhead_percentage": (
                    (avg_time_overhead / avg_non_rag_time * 100)
                    if avg_non_rag_time > 0
                    else 0.0
                ),
            },
            "task_type_analysis": task_type_results,
            "difficulty_analysis": difficulty_results,
            "detailed_results": [
                {
                    "task_id": r.task_id,
                    "task_type": r.task_type,
                    "difficulty_level": r.difficulty_level,
                    "rag_quality_score": r.rag_quality_score,
                    "non_rag_quality_score": r.non_rag_quality_score,
                    "quality_improvement": r.quality_improvement,
                    "rag_execution_time": r.rag_execution_time,
                    "non_rag_execution_time": r.non_rag_execution_time,
                    "time_overhead": r.time_overhead,
                    "rag_advantage": r.rag_advantage,
                    "retrieved_documents_count": len(r.rag_retrieved_documents),
                    "rag_confidence_score": r.rag_confidence_score,
                }
                for r in self.results
            ],
        }

    def display_summary_table(self):
        """Display a summary table of the experiment results."""
        if not self.results:
            console.print("❌ No results to display")
            return

        report = self.generate_report()

        # Create summary table
        table = Table(title="Enhanced RAG Experiment Results")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")
        table.add_column("Description", style="yellow")

        table.add_row(
            "Total Tasks",
            str(report["summary"]["total_tasks"]),
            "Number of tasks compared",
        )
        table.add_row(
            "RAG Advantage Rate",
            f"{report['summary']['rag_advantage_rate']:.1%}",
            "Percentage of tasks where RAG performed better",
        )
        table.add_row(
            "Quality Improvement",
            f"{report['summary']['overall_quality_improvement']:+.3f}",
            "Average improvement in quality scores",
        )
        table.add_row(
            "Time Overhead",
            f"{report['summary']['overall_time_overhead']:+.2f}s",
            "Average additional time for RAG",
        )
        table.add_row(
            "RAG Quality",
            f"{report['performance_metrics']['average_rag_quality']:.3f}",
            "Average quality score with RAG",
        )
        table.add_row(
            "Non-RAG Quality",
            f"{report['performance_metrics']['average_non_rag_quality']:.3f}",
            "Average quality score without RAG",
        )

        console.print(table)

        # Display task type analysis
        if report["task_type_analysis"]:
            console.print("\n📊 Task Type Analysis:")
            type_table = Table()
            type_table.add_column("Task Type", style="cyan")
            type_table.add_column("Tasks", justify="right")
            type_table.add_column("RAG Advantages", justify="right")
            type_table.add_column("Advantage Rate", justify="right")
            type_table.add_column("Avg Improvement", justify="right")

            for task_type, data in report["task_type_analysis"].items():
                type_table.add_row(
                    task_type.replace("_", " ").title(),
                    str(data["total_tasks"]),
                    str(data["rag_advantages"]),
                    f"{data['advantage_rate']:.1%}",
                    f"{data['avg_quality_improvement']:+.3f}",
                )

            console.print(type_table)
