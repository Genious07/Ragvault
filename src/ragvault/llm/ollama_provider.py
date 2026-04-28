from __future__ import annotations

import json
from typing import Iterator

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Local Ollama provider — pure stdlib HTTP, no extra SDK required."""

    DEFAULT_MODEL = "llama3"

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def _post(self, endpoint: str, payload: dict, stream: bool = False):
        import urllib.request

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req)

    def complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        resp = self._post("/api/generate", {
            "model": model or self.DEFAULT_MODEL,
            "prompt": f"{system}\n\n{user}",
            "stream": False,
            "options": {"num_predict": max_tokens},
        })
        return json.loads(resp.read())["response"]

    def stream(self, system: str, user: str, model: str, max_tokens: int) -> Iterator[str]:
        resp = self._post("/api/generate", {
            "model": model or self.DEFAULT_MODEL,
            "prompt": f"{system}\n\n{user}",
            "stream": True,
            "options": {"num_predict": max_tokens},
        })
        for line in resp:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("response"):
                yield chunk["response"]
            if chunk.get("done"):
                break
