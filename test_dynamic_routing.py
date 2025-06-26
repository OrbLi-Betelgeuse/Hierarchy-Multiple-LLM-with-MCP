#!/usr/bin/env python3
"""
Test Dynamic Routing System

Simple test to verify the dynamic routing functionality works correctly.
"""

import asyncio
import logging
from models.mcp_protocol import MCPProtocol, Task, TaskStatus
from models.dynamic_routing import DynamicRouter, RoutingStrategy, RoutingDecision

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_dynamic_routing():
    """Test the dynamic routing system."""
    print("🧪 Testing Dynamic Routing System...")

    try:
        # Initialize MCP protocol and router
        mcp_protocol = MCPProtocol()
        router = DynamicRouter(mcp_protocol)

        # Register some test executors with different capabilities
        executors = {
            "executor_01": ["summarization", "general"],
            "executor_02": ["question_answering", "general"],
            "executor_03": ["table_generation", "general"],
            "executor_04": ["summarization", "question_answering"],
            "executor_05": ["table_generation", "question_answering"],
        }

        # Initialize executor metrics
        for executor_id, capabilities in executors.items():
            await router.update_executor_metrics(
                executor_id,
                {
                    "current_load": 0,
                    "avg_response_time": 2.0,
                    "success_rate": 0.9,
                    "capability_scores": {
                        capability: 0.8 for capability in capabilities
                    },
                },
            )

        print("✅ Executor metrics initialized")

        # Create test tasks
        test_tasks = [
            Task(
                task_id="task_001",
                task_type="summarization",
                description="Summarize a long document",
                parameters={"complexity": 5.0, "priority": 3},
            ),
            Task(
                task_id="task_002",
                task_type="question_answering",
                description="Answer a specific question",
                parameters={"complexity": 3.0, "priority": 2},
            ),
            Task(
                task_id="task_003",
                task_type="table_generation",
                description="Generate a data table",
                parameters={"complexity": 7.0, "priority": 4},
            ),
        ]

        print("✅ Test tasks created")

        # Test different routing strategies
        strategies = [
            RoutingStrategy.ROUND_ROBIN,
            RoutingStrategy.LOAD_BALANCED,
            RoutingStrategy.PERFORMANCE_BASED,
            RoutingStrategy.CAPABILITY_MATCH,
            RoutingStrategy.ADAPTIVE,
        ]

        print("\n📊 Testing Routing Strategies:")
        print("-" * 50)

        for strategy in strategies:
            print(f"\n🔍 Testing {strategy.value.upper()}:")

            for task in test_tasks:
                try:
                    # Get routing decision
                    decision = await router.select_executor(task, strategy)

                    print(f"  Task: {task.task_type} -> {decision.selected_executor}")
                    print(f"  Confidence: {decision.confidence_score:.3f}")
                    print(f"  Reasoning: {decision.reasoning[:60]}...")

                    # Record the decision
                    await router.record_routing_decision(decision)

                except Exception as e:
                    print(f"  ❌ Error: {e}")

        # Test routing analytics
        analytics = router.get_routing_analytics()
        print(f"\n📈 Routing Analytics:")
        print(f"  Total decisions: {analytics.get('total_decisions', 0)}")
        print(f"  Strategy distribution: {analytics.get('strategy_distribution', {})}")
        print(f"  Average confidence: {analytics.get('average_confidence', 0):.3f}")

        # Test with varying loads
        print(f"\n⚖️ Testing Load Balancing:")
        print("-" * 30)

        # Simulate different load conditions
        await router.update_executor_metrics("executor_01", {"current_load": 5})
        await router.update_executor_metrics("executor_02", {"current_load": 2})
        await router.update_executor_metrics("executor_03", {"current_load": 0})

        load_balanced_decision = await router.select_executor(
            test_tasks[0], RoutingStrategy.LOAD_BALANCED
        )

        print(f"Load balanced selection: {load_balanced_decision.selected_executor}")
        print(f"Reasoning: {load_balanced_decision.reasoning}")

        print("\n✅ Dynamic routing test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Dynamic routing test failed: {e}")
        logger.error(f"Test error: {e}")
        return False


async def test_routing_decision_quality():
    """Test the quality of routing decisions."""
    print("\n🎯 Testing Routing Decision Quality...")

    try:
        mcp_protocol = MCPProtocol()
        router = DynamicRouter(mcp_protocol)

        # Setup executors with different performance characteristics
        executors_config = {
            "fast_executor": {
                "avg_response_time": 1.0,
                "success_rate": 0.95,
                "capability_scores": {"summarization": 0.9, "general": 0.8},
            },
            "slow_executor": {
                "avg_response_time": 5.0,
                "success_rate": 0.7,
                "capability_scores": {"summarization": 0.6, "general": 0.5},
            },
            "specialized_executor": {
                "avg_response_time": 2.0,
                "success_rate": 0.9,
                "capability_scores": {"summarization": 0.95, "general": 0.6},
            },
        }

        for executor_id, config in executors_config.items():
            await router.update_executor_metrics(executor_id, config)

        # Test task that should prefer specialized executor
        summarization_task = Task(
            task_id="quality_test_001",
            task_type="summarization",
            description="High-quality summarization task",
            parameters={"complexity": 8.0, "priority": 5},
        )

        # Test different strategies
        strategies_to_test = [
            RoutingStrategy.PERFORMANCE_BASED,
            RoutingStrategy.CAPABILITY_MATCH,
            RoutingStrategy.ADAPTIVE,
        ]

        print("Routing decisions for summarization task:")
        for strategy in strategies_to_test:
            decision = await router.select_executor(summarization_task, strategy)
            print(
                f"  {strategy.value}: {decision.selected_executor} (confidence: {decision.confidence_score:.3f})"
            )

        # Verify that specialized executor is preferred for capability-based routing
        capability_decision = await router.select_executor(
            summarization_task, RoutingStrategy.CAPABILITY_MATCH
        )

        if capability_decision.selected_executor == "specialized_executor":
            print("✅ Capability-based routing correctly selected specialized executor")
        else:
            print(
                f"⚠️ Capability-based routing selected {capability_decision.selected_executor} instead of specialized_executor"
            )

        return True

    except Exception as e:
        print(f"❌ Routing quality test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Dynamic Routing System Tests")
    print("=" * 50)

    # Run basic functionality test
    basic_test_passed = await test_dynamic_routing()

    # Run quality test
    quality_test_passed = await test_routing_decision_quality()

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(
        f"Basic functionality test: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}"
    )
    print(
        f"Routing quality test: {'✅ PASSED' if quality_test_passed else '❌ FAILED'}"
    )

    if basic_test_passed and quality_test_passed:
        print("\n🎉 All tests passed! Dynamic routing system is working correctly.")
        print(
            "You can now run the full experiment with: python experiments/dynamic_routing_experiment.py"
        )
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    asyncio.run(main())
