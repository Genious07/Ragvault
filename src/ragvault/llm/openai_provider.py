from __future__ import annotations

from typing import Iterator

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI()
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install 'ragvault[openai]'"
            )

    def complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=model or self.DEFAULT_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    def stream(self, system: str, user: str, model: str, max_tokens: int) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=model or self.DEFAULT_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
