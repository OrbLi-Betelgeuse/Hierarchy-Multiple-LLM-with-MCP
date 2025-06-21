#!/usr/bin/env python3
"""
Debug test to isolate hanging issues.
"""

import asyncio
import logging
from models.llm_interface import create_llm_interface

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_ollama_connection():
    """Test basic Ollama connection."""
    print("🔍 Testing Ollama connection...")

    try:
        # Test 1: Create LLM interface
        print("📝 Creating LLM interface...")
        llm = create_llm_interface(
            provider="ollama",
            model_name="llama2:7b",
            base_url="http://localhost:11434",
        )
        print("✅ LLM interface created")

        # Test 2: Simple generation
        print("🤖 Testing simple generation...")
        response = await llm.generate("Hello, how are you?")
        print(f"✅ Generation completed: {response.content[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}", exc_info=True)
        return False


async def main():
    """Main test function."""
    print("🚀 Starting debug test...")

    success = await test_ollama_connection()

    if success:
        print("🎉 Debug test completed successfully!")
    else:
        print("💥 Debug test failed!")


if __name__ == "__main__":
    asyncio.run(main())
