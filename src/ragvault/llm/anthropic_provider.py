from __future__ import annotations

from typing import Iterator

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self) -> None:
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install 'ragvault[anthropic]'"
            )

    def complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=model or self.DEFAULT_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def stream(self, system: str, user: str, model: str, max_tokens: int) -> Iterator[str]:
        with self._client.messages.stream(
            model=model or self.DEFAULT_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as s:
            yield from s.text_stream
