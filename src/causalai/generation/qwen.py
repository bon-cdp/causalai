"""
Alibaba Qwen provider via DashScope.

Provides integration with Qwen models through Alibaba's Model Studio API.
"""

import os
import time
from typing import AsyncIterator

import dashscope
from dashscope import Generation

from causalai.generation.providers import (
    LLMProvider,
    GenerationConfig,
    GenerationResult,
)


class QwenProvider(LLMProvider):
    """Qwen LLM provider using DashScope API.

    Supports Qwen models including qwen-plus, qwen-turbo, qwen-max, etc.

    Example:
        >>> provider = QwenProvider(api_key="sk-xxx")
        >>> result = provider.generate(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     config=GenerationConfig(model="qwen-plus"),
        ... )
        >>> print(result.content)
    """

    # Singapore region endpoint
    BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"

    def __init__(self, api_key: str | None = None):
        """Initialize the Qwen provider.

        Args:
            api_key: DashScope API key. If not provided, uses DASHSCOPE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set DASHSCOPE_API_KEY env var or pass api_key parameter."
            )

        # Configure dashscope
        dashscope.base_http_api_url = self.BASE_URL

    def generate(
        self,
        messages: list[dict],
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate a response synchronously.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            config: Generation configuration

        Returns:
            GenerationResult with the response
        """
        start_time = time.perf_counter()

        # Add system prompt if provided
        if config.system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": config.system_prompt}] + messages

        response = Generation.call(
            api_key=self.api_key,
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            result_format="message",
            stop=config.stop_sequences if config.stop_sequences else None,
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Handle response
        if response.status_code != 200:
            raise RuntimeError(
                f"Qwen API error: {response.code} - {response.message}"
            )

        output = response.output
        content = output.choices[0].message.content if output.choices else ""
        finish_reason = output.choices[0].finish_reason if output.choices else "error"

        # Get token usage
        usage = response.usage or {}
        token_count = usage.get("total_tokens", 0)

        return GenerationResult(
            content=content,
            model=config.model,
            finish_reason=finish_reason,
            token_count=token_count,
            latency_ms=latency_ms,
            raw_response=response,
        )

    async def agenerate(
        self,
        messages: list[dict],
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate a response asynchronously.

        Note: DashScope doesn't have native async support, so this wraps
        the sync call. For true async, consider using aiohttp directly.
        """
        import asyncio

        # Run sync call in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(messages, config),
        )

    def stream(
        self,
        messages: list[dict],
        config: GenerationConfig,
    ):
        """Stream a response token by token.

        Yields:
            String chunks as they arrive
        """
        if config.system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": config.system_prompt}] + messages

        responses = Generation.call(
            api_key=self.api_key,
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            result_format="message",
            stream=True,
            incremental_output=True,
        )

        for response in responses:
            if response.status_code == 200:
                if response.output.choices:
                    content = response.output.choices[0].message.content
                    if content:
                        yield content
            else:
                raise RuntimeError(
                    f"Qwen API error: {response.code} - {response.message}"
                )


def create_qwen_provider(api_key: str | None = None) -> QwenProvider:
    """Factory function to create a Qwen provider.

    Args:
        api_key: Optional API key. Uses env var if not provided.

    Returns:
        Configured QwenProvider instance
    """
    return QwenProvider(api_key=api_key)
