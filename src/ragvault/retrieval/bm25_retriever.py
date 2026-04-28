from __future__ import annotations

import numpy as np


class BM25Retriever:
    """Sparse BM25 keyword retrieval via rank_bm25."""

    def __init__(self, chunks: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        self.bm25 = BM25Okapi([c.lower().split() for c in self._chunks])

    def get_scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(query.lower().split())

    def add(self, new_chunks: list[str]) -> None:
        """Append chunks and rebuild index once (single rebuild regardless of batch size)."""
        from rank_bm25 import BM25Okapi

        self._chunks = self._chunks + new_chunks
        self.bm25 = BM25Okapi([c.lower().split() for c in self._chunks])

    @classmethod
    def from_state(cls, chunks: list[str], bm25_model) -> "BM25Retriever":
        """Restore from a pre-built BM25 model (used by RagVault.load)."""
        obj = cls.__new__(cls)
        obj._chunks = list(chunks)
        obj.bm25 = bm25_model
        return obj
