#!/usr/bin/env python3
"""
Simplified Pipeline for Manager-Executor Collaboration System

This version bypasses the complex MCP protocol and directly calls executors
to avoid hanging issues.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
from rich.console import Console
from rich.panel import Panel

from models.llm_interface import create_llm_interface
from models.manager import Manager
from models.executor import Executor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class SimplePipeline:
    """Simplified pipeline that directly calls executors."""

    def __init__(self):
        self.manager = None
        self.executors = []
        self.results = []

    async def setup_system(self):
        """Setup the Manager-Executor system."""
        console.print(Panel.fit("Setting up Simplified System", style="bold blue"))

        try:
            # Create manager
            manager_llm = create_llm_interface(
                provider="ollama",
                model_name="llama2:7b",
                base_url="http://localhost:11434",
            )

            self.manager = Manager(
                manager_id="simple_manager", llm_interface=manager_llm
            )

            # Create executors
            executor_llm = create_llm_interface(
                provider="ollama",
                model_name="llama2:7b",
                base_url="http://localhost:11434",
            )

            executor = Executor(
                executor_id="simple_executor",
                llm_interface=executor_llm,
                capabilities=["summarization", "general"],
            )

            self.executors.append(executor)

            # Register executor with manager
            await self.manager.register_executor(
                executor_id="simple_executor", capabilities=["summarization", "general"]
            )

            console.print("✅ Simplified system setup complete")

        except Exception as e:
            console.print(f"❌ Error setting up system: {e}", style="bold red")
            raise

    async def run_simple_summarization(self):
        """Run a simple summarization task."""
        console.print(Panel.fit("Running Simple Summarization", style="bold green"))

        try:
            # Simple test document
            test_document = """
            Artificial Intelligence (AI) has emerged as a transformative force in healthcare, 
            offering unprecedented opportunities to improve patient outcomes, enhance diagnostic 
            accuracy, and streamline healthcare delivery. This comprehensive analysis explores 
            the current state of AI applications in healthcare, examining both the remarkable 
            advances and the significant challenges that lie ahead.
            """

            # Create a simple task
            from models.mcp_protocol import Task

            task = Task(
                task_id="simple_task_1",
                task_type="summarization",
                description="Summarize the provided text about AI in healthcare",
                parameters={"input_text": test_document},
                priority=1,
                dependencies=[],
                assigned_executor="simple_executor",
            )

            # Execute task directly with executor
            console.print("🎯 Executing task directly with executor...")
            result = await self.executors[0].execute_task(task)

            console.print(f"✅ Task completed in {result.execution_time:.2f}s")
            console.print(f"📝 Summary: {result.result['output'][:200]}...")

            return result

        except Exception as e:
            console.print(f"❌ Error in summarization: {e}", style="bold red")
            raise

    async def run_pipeline(self):
        """Run the simplified pipeline."""
        try:
            console.print(Panel.fit("Starting Simplified Pipeline", style="bold green"))

            # Setup system
            await self.setup_system()

            # Run simple summarization
            result = await self.run_simple_summarization()

            console.print(
                Panel.fit("Pipeline Completed Successfully", style="bold green")
            )
            return result

        except Exception as e:
            console.print(f"❌ Pipeline failed: {e}", style="bold red")
            logger.error(f"Pipeline error: {e}", exc_info=True)
            raise


async def main():
    """Main function."""
    pipeline = SimplePipeline()
    await pipeline.run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
