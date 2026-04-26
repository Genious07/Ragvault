from __future__ import annotations

import numpy as np


class BM25Retriever:
    """Sparse BM25 keyword retrieval via rank_bm25."""

    def __init__(self, chunks: list[str]):
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def get_scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(query.lower().split())

    def add(self, new_chunks: list[str]) -> None:
        """Rebuild index after adding new chunks (BM25Okapi is not incremental)."""
        from rank_bm25 import BM25Okapi

        self._chunks = self._chunks + new_chunks
        tokenized = [c.lower().split() for c in self._chunks]
        self.bm25 = BM25Okapi(tokenized)
