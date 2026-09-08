from __future__ import annotations

import os

from groq import Groq

from company_ai.types import LLMResponse, ModelEndpoint, Usage

from .base import BaseProvider


class GroqProvider(BaseProvider):
    def generate(
        self, endpoint: ModelEndpoint, messages: list[dict[str, str]]
    ) -> LLMResponse:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=endpoint.model,
            messages=messages,
            temperature=endpoint.temperature,
            max_tokens=endpoint.max_output_tokens,
        )
        usage = response.usage
        text = response.choices[0].message.content or ""
        return LLMResponse(
            text=text,
            usage=Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            ),
            raw=response,
        )
