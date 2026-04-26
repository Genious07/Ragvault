from __future__ import annotations

import re
import numpy as np


class SemanticChunker:
    """
    Splits text into semantically coherent chunks by detecting topic shifts.

    Strategy: embed every sentence, then split whenever cosine similarity
    between adjacent sentences drops below `threshold`.
    """

    def __init__(self, embedder=None, threshold: float = 0.75, min_chunk_size: int = 50):
        self.embedder = embedder
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size

    def _split_sentences(self, text: str) -> list[str]:
        # split on sentence boundaries while keeping the delimiter attached
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    def chunk(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        embeddings = self.embedder.embed(sentences)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)

        chunks: list[str] = []
        current: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i], embeddings[i - 1]))
            if sim < self.threshold and len(" ".join(current)) >= self.min_chunk_size:
                chunks.append(" ".join(current))
                current = []
            current.append(sentences[i])

        if current:
            chunks.append(" ".join(current))

        return chunks

    def chunk_documents(self, texts: list[str]) -> list[str]:
        """Chunk multiple documents and return a flat list of chunks."""
        all_chunks: list[str] = []
        for text in texts:
            all_chunks.extend(self.chunk(text))
        return all_chunks
