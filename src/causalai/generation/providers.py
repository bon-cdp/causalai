"""
LLM provider abstractions for CausalAI.

Provides a unified interface for different LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator
import time


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""

    model: str = "qwen-plus"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    system_prompt: str | None = None
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    content: str
    model: str
    finish_reason: str = "stop"
    token_count: int = 0
    latency_ms: int = 0
    raw_response: dict | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate a response synchronously."""
        pass

    @abstractmethod
    async def agenerate(
        self,
        messages: list[dict],
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate a response asynchronously."""
        pass
