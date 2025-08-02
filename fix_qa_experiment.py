#!/usr/bin/env python3
"""
Fix QA Experiment - Run a proper QA experiment with multiple tasks
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Any
from models.executor import Executor
from models.llm_interface import create_llm_interface
from models.manager import Manager
from models.mcp_protocol import Task as MCPTask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedQAExperiment:
    """Fixed QA experiment with proper task execution"""

    def __init__(
        self, manager_config: Dict[str, Any], executor_configs: List[Dict[str, Any]]
    ):
        self.manager_config = manager_config
        self.executor_configs = executor_configs
        self.manager = None
        self.executors = []

        # Multiple QA tasks for proper testing
        self.qa_tasks = [
            {
                "task_id": "qa_1",
                "context": "Artificial Intelligence (AI) is transforming healthcare through improved diagnostics and patient care.",
                "questions": [
                    "What is AI doing in healthcare?",
                    "What are the main benefits?",
                ],
                "expected_answers": [
                    "AI is transforming healthcare through improved diagnostics and patient care",
                    "Improved diagnostics and better patient care",
                ],
            },
            {
                "task_id": "qa_2",
                "context": "Python is a popular programming language known for its readability and versatility.",
                "questions": [
                    "What is Python?",
                    "What is Python known for?",
                ],
                "expected_answers": [
                    "A popular programming language",
                    "Readability and versatility",
                ],
            },
            {
                "task_id": "qa_3",
                "context": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                "questions": [
                    "What is machine learning?",
                    "How does it relate to AI?",
                ],
                "expected_answers": [
                    "A subset of artificial intelligence that enables computers to learn without being explicitly programmed",
                    "It is a subset of artificial intelligence",
                ],
            },
        ]

    async def setup(self):
        """Setup manager and executors"""
        try:
            # Setup manager
            if self.manager_config:
                manager_llm = create_llm_interface(
                    provider=self.manager_config["provider"],
                    model_name=self.manager_config["model"],
                    **self.manager_config.get("kwargs", {}),
                )
                self.manager = Manager(
                    manager_id=self.manager_config["manager_id"],
                    llm_interface=manager_llm,
                )

            # Setup executors
            for config in self.executor_configs:
                executor_llm = create_llm_interface(
                    provider=config["provider"],
                    model_name=config["model"],
                    **config.get("kwargs", {}),
                )
                executor = Executor(
                    executor_id=config["executor_id"],
                    llm_interface=executor_llm,
                    capabilities=config["capabilities"],
                )
                self.executors.append(executor)

                # Register with manager
                if self.manager:
                    await self.manager.register_executor(
                        executor_id=config["executor_id"],
                        capabilities=config["capabilities"],
                    )

            logger.info(f"Setup complete: {len(self.executors)} executors")

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

    async def run_qa_task(self, task_data: Dict) -> Dict:
        """Run a single QA task"""
        task_id = task_data["task_id"]
        context = task_data["context"]
        questions = task_data["questions"]
        expected_answers = task_data["expected_answers"]

        results = {
            "task_id": task_id,
            "questions": questions,
            "answers": [],
            "expected_answers": expected_answers,
            "success": False,
            "execution_time": 0,
            "accuracy": 0.0,
        }

        try:
            start_time = time.time()

            # Execute each question
            for i, question in enumerate(questions):
                if not self.executors:
                    raise RuntimeError("No executors available")

                # Select executor (round-robin)
                executor = self.executors[i % len(self.executors)]

                # Create prompt
                prompt = f"""Context: {context}

Question: {question}

Please answer the question based on the context provided. Be concise and accurate."""

                # Create MCP task
                mcp_task = MCPTask(
                    task_id=f"{task_id}_q{i+1}",
                    task_type="question_answering",
                    description=prompt,
                    parameters={},
                )

                # Execute task
                logger.info(
                    f"Executing question {i+1} with executor {executor.executor_id}"
                )
                response = await executor.execute_task(mcp_task)

                # Extract answer from response
                if hasattr(response, "result") and response.result:
                    answer = str(response.result)
                elif hasattr(response, "content") and response.content:
                    answer = str(response.content)
                else:
                    answer = str(response)

                results["answers"].append(answer)
                logger.info(f"Answer {i+1}: {answer}")

            # Calculate execution time
            results["execution_time"] = time.time() - start_time

            # Calculate accuracy (simple string matching)
            correct_answers = 0
            for i, (answer, expected) in enumerate(
                zip(results["answers"], expected_answers)
            ):
                # Simple similarity check
                answer_lower = answer.lower().strip()
                expected_lower = expected.lower().strip()

                # Check if expected answer is contained in the response
                if expected_lower in answer_lower or answer_lower in expected_lower:
                    correct_answers += 1
                    logger.info(f"Question {i+1} correct")
                else:
                    logger.info(
                        f"Question {i+1} incorrect. Expected: {expected}, Got: {answer}"
                    )

            results["accuracy"] = correct_answers / len(questions) if questions else 0
            results["success"] = results["accuracy"] > 0

            logger.info(f"Task {task_id} completed: {results['accuracy']:.2%} accuracy")

        except Exception as e:
            logger.error(f"Error in task {task_id}: {e}")
            results["success"] = False

        return results

    async def run_experiment(self) -> Dict:
        """Run the complete QA experiment"""
        logger.info("Starting QA experiment...")

        # Setup
        await self.setup()

        # Run all tasks
        all_results = []
        total_tasks = len(self.qa_tasks)
        successful_tasks = 0
        total_execution_time = 0

        for task_data in self.qa_tasks:
            result = await self.run_qa_task(task_data)
            all_results.append(result)

            if result["success"]:
                successful_tasks += 1

            total_execution_time += result["execution_time"]

        # Calculate overall metrics
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        avg_execution_time = (
            total_execution_time / total_tasks if total_tasks > 0 else 0
        )
        avg_accuracy = (
            sum(r["accuracy"] for r in all_results) / len(all_results)
            if all_results
            else 0
        )

        # Generate report
        report = {
            "experiment_type": "qa",
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "total_execution_time": total_execution_time,
            "average_accuracy": avg_accuracy,
            "task_results": all_results,
            "manager_performance": {},
            "executor_performance": {},
            "quality_metrics": {
                "overall_accuracy": avg_accuracy,
                "task_success_rate": success_rate,
            },
            "resource_utilization": {
                "cpu_user_time": 0.0,
                "cpu_system_time": 0.0,
                "memory_peak_bytes": 0,
                "token_total": 0,
                "token_per_sec": 0.0,
            },
        }

        logger.info(
            f"Experiment completed: {success_rate:.1%} success rate, {avg_accuracy:.1%} accuracy"
        )

        return report


async def main():
    """Main function to run the fixed QA experiment"""

    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    # Run experiment
    experiment = FixedQAExperiment(
        manager_config=config["manager"], executor_configs=config["executors"]
    )

    results = await experiment.run_experiment()

    # Save results
    with open("results/qa_metrics_fixed.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ QA experiment completed!")
    print(f"   Success Rate: {results['success_rate']:.1%}")
    print(f"   Average Accuracy: {results['average_accuracy']:.1%}")
    print(f"   Average Time: {results['average_execution_time']:.4f}s")
    print(f"   Results saved to: results/qa_metrics_fixed.json")


if __name__ == "__main__":
    asyncio.run(main())
