from __future__ import annotations

import os

from google import genai
from google.genai import types

from company_ai.types import LLMResponse, ModelEndpoint, Usage

from .base import BaseProvider


class GeminiProvider(BaseProvider):
    def generate(
        self, endpoint: ModelEndpoint, messages: list[dict[str, str]]
    ) -> LLMResponse:
        system_chunks: list[str] = []
        contents: list[dict] = []
        role_map = {"user": "user", "assistant": "model"}

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                system_chunks.append(content)
                continue
            if role not in role_map:
                role = "user"
            contents.append({"role": role_map[role], "parts": [{"text": content}]})

        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        response = client.models.generate_content(
            model=endpoint.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=endpoint.temperature,
                max_output_tokens=endpoint.max_output_tokens,
                system_instruction="\n\n".join(system_chunks)
                if system_chunks
                else None,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        client.close()
        return LLMResponse(
            text=getattr(response, "text", "") or "",
            usage=Usage(
                prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                total_tokens=getattr(usage, "total_token_count", 0) or 0,
            ),
            raw=response,
        )
