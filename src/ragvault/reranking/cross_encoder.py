from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


class CrossEncoderReranker:
    """
    Cross-encoder reranker using BGE reranker models via FlagEmbedding.

    Cross-encoders score (query, doc) pairs jointly — much more accurate
    than bi-encoder dot products but slower (runs inference per pair).
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-large", use_fp16: bool = True) -> None:
        from FlagEmbedding import FlagReranker

        self.model_name = model_name
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(self, query: str, chunks: list[str], top_n: int = 5) -> list[str]:
        """Score all (query, chunk) pairs and return top_n by descending score."""
        if not chunks:
            return []
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.reranker.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = scores.tolist()
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_n]]

    def rerank_with_scores(self, query: str, chunks: list[str]) -> list[tuple[str, float]]:
        """Return all chunks with their scores, sorted descending."""
        if not chunks:
            return []
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.reranker.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = scores.tolist()
        return sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    def rerank_tagged(
        self,
        query: str,
        tagged_chunks: list[tuple[str, T]],
        top_n: int = 5,
    ) -> list[tuple[str, T, float]]:
        """
        Rerank (chunk, tag) pairs, preserving the tag through reranking.

        The tag can be any value (an index, a metadata dict, etc.) and passes
        through unchanged. Returns list of (chunk, tag, score), top_n.
        """
        if not tagged_chunks:
            return []
        chunks = [c for c, _ in tagged_chunks]
        tags = [t for _, t in tagged_chunks]
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.reranker.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = scores.tolist()
        ranked = sorted(zip(chunks, tags, scores), key=lambda x: x[2], reverse=True)
        return ranked[:top_n]
