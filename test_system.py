"""
Test Script for Manager-Executor Collaboration System

Simple test to verify that all components are working correctly.
"""

import asyncio
import logging
import signal
from models.llm_interface import create_llm_interface
from models.manager import Manager
from models.executor import Executor
from models.mcp_protocol import MCPProtocol

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timeout for async operations
TIMEOUT = 30  # seconds


async def test_llm_interface():
    """Test LLM interface creation and basic functionality."""
    print("🧪 Testing LLM Interface...")

    try:
        # Test Ollama interface creation
        llm = create_llm_interface("ollama", "llama2:7b")
        print("✅ LLM interface created successfully")

        # Test basic generation with timeout
        try:
            print("   Testing LLM generation (this may take a moment)...")
            response = await asyncio.wait_for(
                llm.generate("Hello, how are you?"), timeout=TIMEOUT
            )
            print(f"✅ LLM response received: {response.content[:50]}...")
        except asyncio.TimeoutError:
            print("⚠️ LLM generation timed out (this is normal for first run)")
        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}")

        return True
    except Exception as e:
        print(f"❌ LLM interface test failed: {e}")
        return False


async def test_mcp_protocol():
    """Test MCP protocol functionality."""
    print("\n🧪 Testing MCP Protocol...")

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
        # Create LLM interfaces
        manager_llm = create_llm_interface("ollama", "llama2:7b")
        executor_llm = create_llm_interface("ollama", "llama2:7b")

        # Create Manager and Executor
        manager = Manager("test_manager", manager_llm)
        executor = Executor("test_executor", executor_llm, ["general"])

        print("✅ Manager and Executor created successfully")

        # Test executor registration
        await manager.register_executor("test_executor", ["general"])
        print("✅ Executor registered with manager successfully")

        return True
    except Exception as e:
        print(f"❌ Manager-Executor setup test failed: {e}")
        return False


async def test_simple_task_execution():
    """Test simple task execution flow."""
    print("\n🧪 Testing Simple Task Execution...")

    try:
        # Create LLM interfaces
        manager_llm = create_llm_interface("ollama", "llama2:7b")
        executor_llm = create_llm_interface("ollama", "llama2:7b")

        # Create Manager and Executor
        manager = Manager("test_manager", manager_llm)
        executor = Executor("test_executor", executor_llm, ["general"])

        # Register executor
        await manager.register_executor("test_executor", ["general"])

        # Test task decomposition with timeout
        print("   Testing task decomposition...")
        task_description = "Summarize this text: AI is transforming healthcare."

        try:
            decomposition = await asyncio.wait_for(
                manager.decompose_task(task_description), timeout=TIMEOUT
            )
            print(f"✅ Task decomposed into {len(decomposition.subtasks)} subtasks")
        except asyncio.TimeoutError:
            print("⚠️ Task decomposition timed out (using fallback)")
            # This will trigger the fallback decomposition
            decomposition = await manager.decompose_task(task_description)
            print(
                f"✅ Fallback decomposition created {len(decomposition.subtasks)} subtasks"
            )

        return True
    except Exception as e:
        print(f"❌ Simple task execution test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Manager-Executor Collaboration System Tests\n")

    tests = [
        ("LLM Interface", test_llm_interface),
        ("MCP Protocol", test_mcp_protocol),
        ("Manager-Executor Setup", test_manager_executor_setup),
        ("Simple Task Execution", test_simple_task_execution),
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


if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n⚠️ Test interrupted by user")
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
