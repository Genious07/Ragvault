from __future__ import annotations

import numpy as np

from .bm25_retriever import BM25Retriever
from .vector_store import FaissVectorStore


class HybridRetriever:
    """
    Combines dense (FAISS) and sparse (BM25) retrieval using Reciprocal Rank Fusion.

    RRF score for doc d:  sum_over_systems( 1 / (k + rank(d)) )
    where k=60 is the standard smoothing constant.
    No score normalisation needed — only ranks matter.
    """

    def __init__(self, chunks: list[str], embeddings: np.ndarray, rrf_k: int = 60):
        self.chunks = list(chunks)
        self.rrf_k = rrf_k
        self.vector_store = FaissVectorStore()
        self.vector_store.build(embeddings)
        self.bm25 = BM25Retriever(chunks)

    def add_documents(self, new_chunks: list[str], new_embeddings: np.ndarray) -> None:
        """Incrementally add more documents to both indexes."""
        self.chunks.extend(new_chunks)
        self.vector_store.add(new_embeddings)
        self.bm25.add(new_chunks)

    def retrieve(self, query: str, query_embedding: np.ndarray, top_n: int = 20) -> list[str]:
        n = len(self.chunks)
        if n == 0:
            return []

        top_n = min(top_n, n)
        search_k = min(n, top_n * 4)

        bm25_scores = self.bm25.get_scores(query)
        bm25_ranks = np.argsort(-bm25_scores)

        _, vec_indices = self.vector_store.search(query_embedding, search_k)

        rrf: dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranks):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, idx in enumerate(vec_indices):
            if idx < 0:
                continue
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (self.rrf_k + rank)

        top_indices = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_n]
        return [self.chunks[i] for i in top_indices]
