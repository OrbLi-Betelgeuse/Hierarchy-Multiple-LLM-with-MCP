#!/usr/bin/env python3
"""
Test script for RAGFlow integration.
"""

import asyncio
import logging
from models.llm_interface import create_llm_interface
from models.ragflow_interface import create_ragflow_interface
from models.rag_executor import RAGExecutor
from models.mcp_protocol import Task

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_ragflow_integration():
    """Test RAGFlow integration."""
    print("🚀 Testing RAGFlow Integration...")

    try:
        # Create LLM interface
        print("📝 Creating LLM interface...")
        llm_interface = create_llm_interface(
            provider="ollama",
            model_name="llama2:7b",
            base_url="http://localhost:11434",
        )

        # Create RAGFlow interface
        print("🔍 Creating RAGFlow interface...")
        ragflow_interface = create_ragflow_interface(base_url="http://localhost:9380")

        # Test RAGFlow connection
        print("🔗 Testing RAGFlow connection...")
        is_connected = await ragflow_interface.check_connection()

        if not is_connected:
            print("⚠️ RAGFlow not accessible, testing fallback mode...")
            # Test with RAG disabled
            rag_executor = RAGExecutor(
                executor_id="test_rag_executor",
                llm_interface=llm_interface,
                ragflow_interface=ragflow_interface,
                capabilities=["rag_question_answering", "general"],
            )

            # Test simple task execution
            task = Task(
                task_id="test_task",
                task_type="question_answering",
                description="What is artificial intelligence?",
                parameters={"query": "What is artificial intelligence?"},
                priority=1,
                dependencies=[],
                assigned_executor="test_rag_executor",
            )

            result = await rag_executor.execute_rag_task(task)
            print(
                f"✅ Fallback execution completed: {result.result['output'][:100]}..."
            )

        else:
            print("✅ RAGFlow connection successful!")

            # Create RAG executor
            rag_executor = RAGExecutor(
                executor_id="test_rag_executor",
                llm_interface=llm_interface,
                ragflow_interface=ragflow_interface,
                capabilities=["rag_question_answering", "general"],
            )

            # Setup knowledge base
            print("📚 Setting up knowledge base...")
            kb_id = await rag_executor.setup_knowledge_base(
                name="test_kb", description="Test knowledge base for AI questions"
            )

            if kb_id:
                # Upload sample documents
                sample_docs = [
                    {
                        "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence.",
                        "metadata": {"source": "AI textbook", "page": 1},
                    },
                    {
                        "content": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
                        "metadata": {"source": "ML guide", "page": 5},
                    },
                ]

                print("📄 Uploading sample documents...")
                upload_success = await rag_executor.upload_documents_to_kb(sample_docs)

                if upload_success:
                    # Test RAG task execution
                    task = Task(
                        task_id="test_rag_task",
                        task_type="rag_question_answering",
                        description="What is artificial intelligence?",
                        parameters={"query": "What is artificial intelligence?"},
                        priority=1,
                        dependencies=[],
                        assigned_executor="test_rag_executor",
                    )

                    print("🎯 Executing RAG task...")
                    result = await rag_executor.execute_rag_task(task)

                    print(f"✅ RAG task completed!")
                    print(f"📝 Response: {result.result['output'][:200]}...")
                    print(f"🎯 Confidence: {result.confidence_score:.2f}")
                    print(f"📚 Retrieved docs: {len(result.retrieved_documents)}")

                    # Get RAG performance metrics
                    metrics = rag_executor.get_rag_performance_metrics()
                    print(f"📊 RAG Metrics: {metrics}")

                else:
                    print("❌ Failed to upload documents")
            else:
                print("❌ Failed to setup knowledge base")

        print("🎉 RAGFlow integration test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_ragflow_integration())
