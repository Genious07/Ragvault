from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class LLMProvider(ABC):
    DEFAULT_MODEL: str = ""

    @abstractmethod
    def complete(self, system: str, user: str, model: str, max_tokens: int) -> str: ...

    @abstractmethod
    def stream(self, system: str, user: str, model: str, max_tokens: int) -> Iterator[str]: ...
