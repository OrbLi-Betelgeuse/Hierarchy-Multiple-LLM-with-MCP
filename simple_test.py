#!/usr/bin/env python3
"""
Simple Test for Manager-Executor Collaboration System

Basic functionality test without LLM calls to avoid hanging.
"""

import asyncio
import logging
from models.mcp_protocol import MCPProtocol, MCPManager, MCPExecutor
from models.llm_interface import create_llm_interface
from models.manager import Manager
from models.executor import Executor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_mcp_protocol():
    """Test MCP protocol functionality."""
    print("🧪 Testing MCP Protocol...")

    try:
        protocol = MCPProtocol()

        # Test message creation
        message = protocol.create_task_decomposition_message(
            "manager_01", "executor_01", "Test task", "simple"
        )
        print("✅ MCP message created successfully")

        # Test message serialization
        serialized = protocol.serialize_message(message)
        print("✅ MCP message serialized successfully")

        # Test message parsing
        parsed = protocol.parse_message(serialized)
        print("✅ MCP message parsed successfully")

        return True
    except Exception as e:
        print(f"❌ MCP protocol test failed: {e}")
        return False


async def test_manager_executor_setup():
    """Test Manager and Executor setup."""
    print("\n🧪 Testing Manager-Executor Setup...")

    try:
        # Create MCP Manager and Executor
        manager = MCPManager("test_manager")
        executor = MCPExecutor("test_executor", ["general"])

        print("✅ MCP Manager and Executor created successfully")

        # Test task decomposition
        available_executors = ["executor_01", "executor_02"]
        tasks = await manager.decompose_task(
            "Test task description", available_executors
        )
        print(f"✅ Task decomposition created {len(tasks)} subtasks")

        return True
    except Exception as e:
        print(f"❌ Manager-Executor setup test failed: {e}")
        return False


async def test_llm_interface_creation():
    """Test LLM interface creation without making calls."""
    print("\n🧪 Testing LLM Interface Creation...")

    try:
        # Test Ollama interface creation
        llm = create_llm_interface("ollama", "llama2:7b")
        print("✅ LLM interface created successfully")

        # Test OpenAI interface creation (if API key available)
        try:
            llm_openai = create_llm_interface("openai", "gpt-3.5-turbo")
            print("✅ OpenAI interface created successfully")
        except Exception as e:
            print(f"⚠️ OpenAI interface creation failed (expected): {e}")

        return True
    except Exception as e:
        print(f"❌ LLM interface creation test failed: {e}")
        return False


async def test_basic_workflow():
    """Test basic workflow without LLM calls."""
    print("\n🧪 Testing Basic Workflow...")

    try:
        # Create manager and executor
        manager = MCPManager("test_manager")
        executor = MCPExecutor("test_executor", ["summarization", "general"])

        # Register executor
        manager.protocol.register_executor(
            "test_executor", ["summarization", "general"]
        )

        # Create a simple task
        task = manager.protocol.task_registry.get("test_task")
        if not task:
            # Create a test task
            task = manager.protocol.task_registry["test_task"] = {
                "task_id": "test_task",
                "description": "Test task",
                "status": "pending",
            }

        print("✅ Basic workflow setup completed")
        return True
    except Exception as e:
        print(f"❌ Basic workflow test failed: {e}")
        return False


async def test_basic_functionality():
    """Test basic Manager-Executor functionality."""
    print("🚀 Starting basic functionality test...")

    try:
        # Create LLM interfaces
        print("📝 Creating LLM interfaces...")
        manager_llm = create_llm_interface(
            provider="ollama",
            model_name="llama2:7b",
            base_url="http://localhost:11434",
        )
        executor_llm = create_llm_interface(
            provider="ollama",
            model_name="llama2:7b",
            base_url="http://localhost:11434",
        )

        # Create manager and executor
        print("👨‍💼 Creating manager...")
        manager = Manager(manager_id="test_manager", llm_interface=manager_llm)

        print("👷 Creating executor...")
        executor = Executor(
            executor_id="test_executor",
            llm_interface=executor_llm,
            capabilities=["summarization", "general"],
        )

        # Register executor
        print("📋 Registering executor...")
        await manager.register_executor(
            executor_id="test_executor", capabilities=["summarization", "general"]
        )

        # Start executor
        print("▶️ Starting executor...")
        await executor.start()

        # Test simple task execution
        print("🎯 Testing task execution...")
        test_task_description = "Summarize this text: Artificial intelligence is transforming healthcare through improved diagnostics and patient care."

        print("⏳ Executing task (this may take a while for first run)...")
        result = await manager.execute_task(test_task_description, "summarization")

        print("✅ Task execution completed!")
        print(f"📊 Result: {result}")

        # Stop executor
        print("⏹️ Stopping executor...")
        await executor.stop()

        print("🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}", exc_info=True)
        raise


async def main():
    """Run all tests."""
    print("🚀 Starting Simple Manager-Executor Collaboration System Tests\n")

    tests = [
        ("MCP Protocol", test_mcp_protocol),
        ("Manager-Executor Setup", test_manager_executor_setup),
        ("LLM Interface Creation", test_llm_interface_creation),
        ("Basic Workflow", test_basic_workflow),
        ("Basic Functionality", test_basic_functionality),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! System is ready for experiments.")
    elif passed >= len(results) - 1:
        print("✅ Most tests passed! System should work for experiments.")
    else:
        print("⚠️ Some tests failed. Please check the setup.")

    print("\n💡 To run full experiments with LLM calls:")
    print("   python pipeline.py --experiment summarization")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
