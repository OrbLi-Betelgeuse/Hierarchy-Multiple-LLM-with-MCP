#!/usr/bin/env python3
"""
Quick test to check Ollama connectivity.
"""

import requests
import json
import time


def test_ollama_basic():
    """Test basic Ollama connectivity."""
    print("🔍 Testing Ollama basic connectivity...")

    try:
        # Test 1: Check if Ollama is responding
        print("📡 Checking Ollama API...")
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        print(f"✅ Ollama API responded: {response.status_code}")

        # Test 2: Check available models
        models = response.json()
        print(f"📋 Available models: {[m['name'] for m in models.get('models', [])]}")

        # Test 3: Try a simple generation request
        print("🤖 Testing simple generation...")
        payload = {
            "model": "llama2:7b",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": False,
        }

        start_time = time.time()
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=60,  # 1 minute timeout
        )
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generation successful in {end_time - start_time:.2f}s")
            print(f"📝 Response: {result['message']['content'][:100]}...")
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"📄 Response: {response.text}")

        return True

    except requests.exceptions.Timeout:
        print("⏰ Request timed out - Ollama may be overloaded or model not loaded")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_ollama_basic()

    if success:
        print("🎉 Quick test completed successfully!")
    else:
        print("💥 Quick test failed!")
        print("\n💡 Suggestions:")
        print("1. Check if Ollama is running: ollama serve")
        print("2. Check if model is downloaded: ollama list")
        print("3. Try pulling the model: ollama pull llama2:7b")
        print("4. Check system resources (CPU, memory)")
