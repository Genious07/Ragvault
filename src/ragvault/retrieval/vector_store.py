from __future__ import annotations

import numpy as np


class FaissVectorStore:
    """
    In-memory FAISS flat index.

    Uses inner-product search — works as cosine similarity when embeddings
    are L2-normalised (which BGEEmbedder does by default).
    """

    def __init__(self):
        self.index = None
        self.dimension: int | None = None

    def build(self, embeddings: np.ndarray) -> None:
        import faiss

        self.dimension = embeddings.shape[1]
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
        """Add more embeddings to an existing index."""
        if self.index is None:
            self.build(embeddings)
        else:
            self.index.add(embeddings.astype("float32"))
