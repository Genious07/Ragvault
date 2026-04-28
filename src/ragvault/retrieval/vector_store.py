from __future__ import annotations

import numpy as np


class FaissVectorStore:
    """
    FAISS index wrapper.

    index_type='flat'  — IndexFlatIP, exact search, works at any scale
    index_type='hnsw'  — IndexHNSWFlat with inner-product metric, O(log n)
                         approximate search; faster queries at 1M+ chunks
    """

    def __init__(self, index_type: str = "flat") -> None:
        self.index = None
        self.dimension: int | None = None
        self.index_type = index_type

    def build(self, embeddings: np.ndarray) -> None:
        import faiss

        self.dimension = embeddings.shape[1]
        if self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype("float32"))

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Call build() before search()")
        scores, indices = self.index.search(
            query_embedding.astype("float32").reshape(1, -1), top_k
        )
        return scores[0], indices[0]

    def add(self, embeddings: np.ndarray) -> None:
        if self.index is None:
            self.build(embeddings)
        else:
            self.index.add(embeddings.astype("float32"))

    def save(self, path: str) -> None:
        import faiss
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, index_type: str = "flat") -> "FaissVectorStore":
        import faiss
        store = cls(index_type=index_type)
        store.index = faiss.read_index(path)
        store.dimension = store.index.d
        return store

    @property
    def ntotal(self) -> int:
        return self.index.ntotal if self.index is not None else 0
