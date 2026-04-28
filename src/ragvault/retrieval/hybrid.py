from __future__ import annotations

import numpy as np

from .bm25_retriever import BM25Retriever
from .vector_store import FaissVectorStore


class HybridRetriever:
    """
    Dense (FAISS) + sparse (BM25) hybrid retrieval fused with Reciprocal Rank Fusion.

    RRF score for doc d:  sum_over_systems( 1 / (k + rank(d)) )  k=60 standard constant.

    mode='fast'     — dense-only, skips BM25 entirely
    mode='balanced' or 'quality' — full hybrid RRF

    allowed_indices — when set, only these chunk indices are eligible for retrieval.
                      Used for metadata filtering and soft-delete.
    """

    def __init__(
        self,
        chunks: list[str],
        embeddings: np.ndarray,
        rrf_k: int = 60,
        index_type: str = "flat",
    ) -> None:
        self.chunks = list(chunks)
        self.rrf_k = rrf_k
        self.vector_store = FaissVectorStore(index_type=index_type)
        self.vector_store.build(embeddings)
        self.bm25 = BM25Retriever(chunks)

    def add_documents(self, new_chunks: list[str], new_embeddings: np.ndarray) -> None:
        """Append chunks to both indexes in a single operation."""
        self.chunks.extend(new_chunks)
        self.vector_store.add(new_embeddings)
        self.bm25.add(new_chunks)

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_n: int = 20,
        allowed_indices: set[int] | None = None,
        mode: str = "balanced",
    ) -> list[int]:
        """Return up to top_n chunk indices sorted by descending relevance."""
        n = len(self.chunks)
        if n == 0:
            return []

        top_n = min(top_n, n)
        search_k = min(n, top_n * 4)
        rrf: dict[int, float] = {}

        # Dense path — always active
        _, vec_indices = self.vector_store.search(query_embedding, search_k)
        for rank, idx in enumerate(vec_indices):
            if idx < 0:
                continue
            i = int(idx)
            if allowed_indices is not None and i not in allowed_indices:
                continue
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (self.rrf_k + rank)

        # Sparse path — skipped in fast mode
        if mode != "fast":
            bm25_scores = self.bm25.get_scores(query)
            for rank, idx in enumerate(np.argsort(-bm25_scores)):
                i = int(idx)
                if allowed_indices is not None and i not in allowed_indices:
                    continue
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (self.rrf_k + rank)

        return sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_n]

    @classmethod
    def from_components(
        cls,
        chunks: list[str],
        vector_store: FaissVectorStore,
        bm25: BM25Retriever,
        rrf_k: int = 60,
    ) -> "HybridRetriever":
        """Reconstruct from pre-built components (used by RagVault.load)."""
        obj = cls.__new__(cls)
        obj.chunks = list(chunks)
        obj.rrf_k = rrf_k
        obj.vector_store = vector_store
        obj.bm25 = bm25
        return obj
