from __future__ import annotations

import os

import httpx

from company_ai.types import LLMResponse, ModelEndpoint, Usage

from .base import BaseProvider


class OllamaProvider(BaseProvider):
    def __init__(self) -> None:
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    def generate(
        self, endpoint: ModelEndpoint, messages: list[dict[str, str]]
    ) -> LLMResponse:
        with httpx.Client(base_url=self.base_url, timeout=120.0) as client:
            response = client.post(
                "/api/chat",
                json={
                    "model": endpoint.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": endpoint.temperature,
                        "num_predict": endpoint.max_output_tokens,
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()

        return LLMResponse(
            text=payload.get("message", {}).get("content", ""),
            usage=Usage(
                prompt_tokens=payload.get("prompt_eval_count", 0) or 0,
                completion_tokens=payload.get("eval_count", 0) or 0,
                total_tokens=(payload.get("prompt_eval_count", 0) or 0)
                + (payload.get("eval_count", 0) or 0),
            ),
            raw=payload,
        )
