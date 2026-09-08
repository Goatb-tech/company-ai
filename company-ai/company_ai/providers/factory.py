from __future__ import annotations

from .base import BaseProvider
from .gemini_provider import GeminiProvider
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider


class ProviderFactory:
    _providers: dict[str, BaseProvider] = {
        "gemini": GeminiProvider(),
        "groq": GroqProvider(),
        "ollama": OllamaProvider(),
    }

    @classmethod
    def get(cls, provider: str) -> BaseProvider:
        normalized = provider.lower()
        if normalized not in cls._providers:
            raise ValueError(f"Unsupported provider: {provider}")
        return cls._providers[normalized]
