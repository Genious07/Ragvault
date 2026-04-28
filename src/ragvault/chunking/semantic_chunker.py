from __future__ import annotations

import re

import numpy as np


class SemanticChunker:
    """
    Splits text into semantically coherent chunks by detecting topic shifts.

    Strategy: embed every sentence, then split whenever cosine similarity
    between adjacent sentences drops below `threshold`.

    Pass return_embeddings=True to get pooled chunk embeddings for free,
    avoiding a second embedding pass at index time.
    """

    def __init__(self, embedder=None, threshold: float = 0.75, min_chunk_size: int = 50):
        self.embedder = embedder
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size

    def _split_sentences(self, text: str) -> list[str]:
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    def chunk(
        self,
        text: str,
        return_embeddings: bool = False,
    ) -> list[str] | tuple[list[str], np.ndarray]:
        """
        Chunk text into semantically coherent segments.

        Args:
            text: Input text.
            return_embeddings: If True, return (chunks, embeddings) where embeddings
                are computed by mean-pooling sentence embeddings within each chunk.
                Saves a full re-embed pass at index time.

        Returns:
            list[str] when return_embeddings=False,
            tuple[list[str], np.ndarray] when return_embeddings=True.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return ([], np.empty((0, 1))) if return_embeddings else []
        if len(sentences) == 1:
            result = [text.strip()]
            if return_embeddings:
                embs = self.embedder.embed(result)
                return result, embs
            return result

        embeddings = self.embedder.embed(sentences)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        unit_embs = embeddings / np.maximum(norms, 1e-9)

        chunks: list[str] = []
        chunk_emb_list: list[np.ndarray] = []
        current: list[str] = [sentences[0]]
        current_idx: list[int] = [0]

        for i in range(1, len(sentences)):
            sim = float(np.dot(unit_embs[i], unit_embs[i - 1]))
            if sim < self.threshold and len(" ".join(current)) >= self.min_chunk_size:
                chunks.append(" ".join(current))
                if return_embeddings:
                    chunk_emb_list.append(unit_embs[current_idx].mean(axis=0))
                current = []
                current_idx = []
            current.append(sentences[i])
            current_idx.append(i)

        if current:
            chunks.append(" ".join(current))
            if return_embeddings:
                chunk_emb_list.append(unit_embs[current_idx].mean(axis=0))

        if return_embeddings:
            dim = embeddings.shape[1]
            chunk_embs = np.stack(chunk_emb_list) if chunk_emb_list else np.empty((0, dim))
            return chunks, chunk_embs

        return chunks

    def chunk_documents(self, texts: list[str]) -> list[str]:
        """Chunk multiple documents and return a flat list of chunks."""
        result: list[str] = []
        for text in texts:
            result.extend(self.chunk(text))
        return result
