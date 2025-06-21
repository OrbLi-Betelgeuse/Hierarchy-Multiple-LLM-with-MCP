#!/usr/bin/env python3
"""
Test with smaller model to avoid resource issues.
"""

import requests
import json
import time


def test_small_model():
    """Test with a smaller model."""
    print("🔍 Testing with smaller model...")

    try:
        # Test 1: Check if Ollama is responding
        print("📡 Checking Ollama API...")
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        print(f"✅ Ollama API responded: {response.status_code}")

        # Test 2: Check available models
        models = response.json()
        available_models = [m["name"] for m in models.get("models", [])]
        print(f"📋 Available models: {available_models}")

        # Test 3: Try with a smaller model if available
        test_models = ["llama2:7b", "llama2:latest", "llama2:70b"]
        working_model = None

        for model in test_models:
            if model in available_models:
                print(f"🤖 Testing model: {model}")
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "stream": False,
                }

                try:
                    start_time = time.time()
                    response = requests.post(
                        "http://localhost:11434/api/chat",
                        json=payload,
                        timeout=30,  # 30 seconds timeout
                    )
                    end_time = time.time()

                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ {model} working in {end_time - start_time:.2f}s")
                        print(f"📝 Response: {result['message']['content'][:100]}...")
                        working_model = model
                        break
                    else:
                        print(f"❌ {model} failed: {response.status_code}")

                except requests.exceptions.Timeout:
                    print(f"⏰ {model} timed out")
                    continue
                except Exception as e:
                    print(f"❌ {model} error: {e}")
                    continue

        if working_model:
            print(f"🎉 Found working model: {working_model}")
            return working_model
        else:
            print("💥 No working models found")
            return None

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    working_model = test_small_model()

    if working_model:
        print(f"\n✅ Use this model in your experiments: {working_model}")
        print("💡 Update your config to use this model instead of llama2:7b")
    else:
        print("\n💥 No working models found!")
        print("\n💡 Suggestions:")
        print("1. Restart Ollama: pkill -f 'ollama serve' && ollama serve")
        print("2. Check system resources (CPU, memory)")
        print("3. Try a smaller model: ollama pull llama2:3b")
        print("4. Check if Docker container is working properly")
