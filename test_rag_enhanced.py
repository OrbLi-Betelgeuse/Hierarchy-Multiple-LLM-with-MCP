#!/usr/bin/env python3
"""
Test script for Enhanced RAG Experiment (Experiment D)
"""

import asyncio
import json
from experiments.rag_enhanced_experiment import RAGEnhancedExperiment


async def test_rag_enhanced():
    """Test the enhanced RAG experiment."""

    # Configuration
    manager_config = {
        "provider": "ollama",
        "model": "qwen2.5:7b",
        "manager_id": "manager_01",
        "kwargs": {"base_url": "http://localhost:11434"},
    }

    ragflow_config = {"base_url": "http://localhost:9380", "kwargs": {}}

    print("🚀 Starting Enhanced RAG Experiment (Experiment D)")
    print("=" * 60)

    # Create and setup experiment
    experiment = RAGEnhancedExperiment(manager_config, ragflow_config)

    try:
        await experiment.setup()
        print("✅ Setup completed successfully")

        # Run experiment
        print("\n🔄 Running experiment...")
        results = await experiment.run_experiment()

        print(f"\n✅ Experiment completed with {len(results)} comparisons")

        # Generate and display report
        report = experiment.generate_report()

        print("\n📊 Experiment Report:")
        print(f"   Total Tasks: {report['summary']['total_tasks']}")
        print(f"   RAG Advantage Rate: {report['summary']['rag_advantage_rate']:.1%}")
        print(
            f"   Overall Quality Improvement: {report['summary']['overall_quality_improvement']:+.3f}"
        )
        print(
            f"   Overall Time Overhead: {report['summary']['overall_time_overhead']:+.2f}s"
        )

        print(f"\n📈 Performance Metrics:")
        print(
            f"   Average RAG Quality: {report['performance_metrics']['average_rag_quality']:.3f}"
        )
        print(
            f"   Average Non-RAG Quality: {report['performance_metrics']['average_non_rag_quality']:.3f}"
        )
        print(
            f"   Quality Improvement %: {report['performance_metrics']['quality_improvement_percentage']:+.1f}%"
        )
        print(
            f"   Time Overhead %: {report['performance_metrics']['time_overhead_percentage']:+.1f}%"
        )

        # Display detailed results
        print(f"\n📋 Detailed Results:")
        for result in results:
            print(f"   Task {result.task_id}:")
            print(f"     Type: {result.task_type}")
            print(f"     RAG Quality: {result.rag_quality_score:.3f}")
            print(f"     Non-RAG Quality: {result.non_rag_quality_score:.3f}")
            print(f"     Improvement: {result.quality_improvement:+.3f}")
            print(f"     RAG Advantage: {'✅' if result.rag_advantage else '❌'}")
            print(f"     Retrieved Docs: {len(result.rag_retrieved_documents)}")
            print()

        # Display summary table
        print("\n📊 Summary Table:")
        experiment.display_summary_table()

        return report

    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(test_rag_enhanced())
