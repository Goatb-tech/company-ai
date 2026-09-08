from __future__ import annotations

from abc import ABC, abstractmethod

from company_ai.types import LLMResponse, ModelEndpoint


class BaseProvider(ABC):
    @abstractmethod
    def generate(
        self, endpoint: ModelEndpoint, messages: list[dict[str, str]]
    ) -> LLMResponse:
        raise NotImplementedError
