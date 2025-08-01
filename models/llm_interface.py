"""
LLM Interface Module

Provides unified interfaces for different LLM providers including Ollama, OpenAI, and Anthropic.
Supports both Manager and Executor LLM interactions.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import requests
import ollama
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response format for LLM interactions."""

    content: str
    model: str
    tokens_used: Optional[int] = None
    latency: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.token_counter = 0  # 累计本实例的token数

    def add_tokens(self, n):
        self.token_counter += n

    def get_token_count(self):
        return self.token_counter

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from the LLM."""
        pass

    @abstractmethod
    async def generate_with_system_prompt(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> LLMResponse:
        """Generate response with system and user prompts."""
        pass


class MockInterface(LLMInterface):
    """Mock interface for testing purposes."""

    def __init__(self, model_name: str = "mock", **kwargs):
        super().__init__(model_name, **kwargs)

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response."""
        await asyncio.sleep(0.1)  # Simulate processing time

        resp = LLMResponse(
            content=f"Mock response to: {prompt[:50]}...",
            model=self.model_name,
            tokens_used=len(prompt.split()) + 10,
            latency=0.1,
            metadata={"mock": True},
        )
        self.add_tokens(resp.tokens_used or 0)
        return resp

    async def generate_with_system_prompt(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> LLMResponse:
        """Generate mock response with system prompt."""
        await asyncio.sleep(0.1)  # Simulate processing time

        resp = LLMResponse(
            content=f"Mock response (system: {system_prompt[:30]}...): {user_prompt[:50]}...",
            model=self.model_name,
            tokens_used=len(system_prompt.split()) + len(user_prompt.split()) + 10,
            latency=0.1,
            metadata={"mock": True},
        )
        self.add_tokens(resp.tokens_used or 0)
        return resp


class OllamaInterface(LLMInterface):
    """Interface for Ollama LLM models."""

    def __init__(
        self, model_name: str, base_url: str = "http://localhost:11434", **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Ollama."""
        try:
            start_time = asyncio.get_event_loop().time()

            # Add timeout to prevent hanging
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat,
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=120,  # 2 minutes timeout
            )

            end_time = asyncio.get_event_loop().time()
            latency = end_time - start_time

            resp = LLMResponse(
                content=response["message"]["content"],
                model=self.model_name,
                tokens_used=response.get("prompt_eval_count", 0)
                + response.get("eval_count", 0),
                latency=latency,
                metadata={
                    "prompt_eval_count": response.get("prompt_eval_count"),
                    "eval_count": response.get("eval_count"),
                    "eval_duration": response.get("eval_duration"),
                },
            )
            self.add_tokens(resp.tokens_used or 0)
            return resp
        except asyncio.TimeoutError:
            logger.error(f"Ollama request timed out after 120 seconds")
            raise Exception("LLM request timed out")
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            raise

    async def generate_with_system_prompt(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> LLMResponse:
        """Generate response with system and user prompts using Ollama."""
        try:
            start_time = asyncio.get_event_loop().time()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Add timeout to prevent hanging
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat, model=self.model_name, messages=messages
                ),
                timeout=120,  # 2 minutes timeout
            )

            end_time = asyncio.get_event_loop().time()
            latency = end_time - start_time

            resp = LLMResponse(
                content=response["message"]["content"],
                model=self.model_name,
                tokens_used=response.get("prompt_eval_count", 0)
                + response.get("eval_count", 0),
                latency=latency,
                metadata={
                    "prompt_eval_count": response.get("prompt_eval_count"),
                    "eval_count": response.get("eval_count"),
                    "eval_duration": response.get("eval_duration"),
                },
            )
            self.add_tokens(resp.tokens_used or 0)
            return resp
        except asyncio.TimeoutError:
            logger.error(f"Ollama request timed out after 120 seconds")
            raise Exception("LLM request timed out")
        except Exception as e:
            logger.error(
                f"Error generating response with system prompt using Ollama: {e}"
            )
            raise


class OpenAIInterface(LLMInterface):
    """Interface for OpenAI LLM models."""

    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        import openai

        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI."""
        try:
            start_time = asyncio.get_event_loop().time()

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            end_time = asyncio.get_event_loop().time()
            latency = end_time - start_time

            resp = LLMResponse(
                content=response.choices[0].message.content,
                model=self.model_name,
                tokens_used=response.usage.total_tokens if response.usage else None,
                latency=latency,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                },
            )
            self.add_tokens(resp.tokens_used or 0)
            return resp
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            raise

    async def generate_with_system_prompt(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> LLMResponse:
        """Generate response with system and user prompts using OpenAI."""
        try:
            start_time = asyncio.get_event_loop().time()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.client.chat.completions.create(
                model=self.model_name, messages=messages, **kwargs
            )

            end_time = asyncio.get_event_loop().time()
            latency = end_time - start_time

            resp = LLMResponse(
                content=response.choices[0].message.content,
                model=self.model_name,
                tokens_used=response.usage.total_tokens if response.usage else None,
                latency=latency,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                },
            )
            self.add_tokens(resp.tokens_used or 0)
            return resp
        except Exception as e:
            logger.error(
                f"Error generating response with system prompt using OpenAI: {e}"
            )
            raise


def create_llm_interface(
    provider: str, model_name: str = "mock", **kwargs
) -> LLMInterface:
    """Factory function to create LLM interface based on provider."""
    if provider.lower() == "mock":
        return MockInterface(model_name, **kwargs)
    elif provider.lower() == "ollama":
        return OllamaInterface(model_name, **kwargs)
    elif provider.lower() == "openai":
        return OpenAIInterface(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
