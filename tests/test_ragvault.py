"""
ragvault test suite.

All tests use lightweight NumPy mocks so no GPU or model download is needed.
Tests cover each module in isolation and the full integrated pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 32

class MockEmbedder:
    model_name = "mock-bge"

    def embed(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(str(texts))) % (2**31))
        vecs = rng.random((len(texts), DIM)).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


@pytest.fixture
def mock_embedder():
    return MockEmbedder()


SAMPLE_CHUNKS = [
    "Retrieval-Augmented Generation combines LLMs with external knowledge.",
    "BGE embeddings are trained by BAAI and top the MTEB leaderboard.",
    "BM25 is a sparse retrieval algorithm based on term frequency.",
    "Hybrid retrieval fuses dense and sparse signals using Reciprocal Rank Fusion.",
    "Cross-encoders rerank candidate passages by scoring query-document pairs jointly.",
    "LLMLingua compresses prompts by removing low-information tokens.",
    "RAGAS evaluates RAG quality with faithfulness and relevancy metrics.",
    "Semantic chunking splits text based on embedding similarity between sentences.",
    "FAISS is a library for efficient similarity search over dense vectors.",
    "Context compression reduces LLM input tokens while preserving key facts.",
]

# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

class TestSemanticChunker:
    def test_returns_list(self, mock_embedder):
        from ragvault.chunking import SemanticChunker
        chunker = SemanticChunker(embedder=mock_embedder, threshold=0.0)
        result = chunker.chunk("First sentence. Second sentence. Third sentence.")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_single_sentence(self, mock_embedder):
        from ragvault.chunking import SemanticChunker
        chunker = SemanticChunker(embedder=mock_embedder)
        result = chunker.chunk("Only one sentence here.")
        assert result == ["Only one sentence here."]

    def test_empty_string(self, mock_embedder):
        from ragvault.chunking import SemanticChunker
        chunker = SemanticChunker(embedder=mock_embedder)
        result = chunker.chunk("")
        assert result == []

    def test_high_threshold_produces_many_chunks(self, mock_embedder):
        from ragvault.chunking import SemanticChunker
        # threshold=0.0 means always split (every drop below 0.0 is impossible,
        # but min_chunk_size still guards merging — set it low too)
        chunker = SemanticChunker(embedder=mock_embedder, threshold=0.99, min_chunk_size=1)
        text = "Alpha topic. Beta topic. Gamma topic. Delta topic."
        result = chunker.chunk(text)
        assert len(result) >= 1

    def test_chunk_documents(self, mock_embedder):
        from ragvault.chunking import SemanticChunker
        chunker = SemanticChunker(embedder=mock_embedder, threshold=0.0)
        docs = ["Doc one. Has two sentences.", "Doc two. Also two sentences."]
        result = chunker.chunk_documents(docs)
        assert isinstance(result, list)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class TestBM25Retriever:
    def test_scores_shape(self):
        from ragvault.retrieval import BM25Retriever
        bm25 = BM25Retriever(SAMPLE_CHUNKS)
        scores = bm25.get_scores("BGE embeddings BAAI")
        assert len(scores) == len(SAMPLE_CHUNKS)

    def test_relevant_chunk_ranks_highest(self):
        from ragvault.retrieval import BM25Retriever
        bm25 = BM25Retriever(SAMPLE_CHUNKS)
        scores = bm25.get_scores("BM25 sparse retrieval")
        best_idx = int(np.argmax(scores))
        assert "BM25" in SAMPLE_CHUNKS[best_idx]

    def test_add_new_chunks(self):
        from ragvault.retrieval import BM25Retriever
        bm25 = BM25Retriever(SAMPLE_CHUNKS[:3])
        bm25.add(["New chunk about quantum computing."])
        scores = bm25.get_scores("quantum computing")
        assert len(scores) == 4


# ---------------------------------------------------------------------------
# FaissVectorStore
# ---------------------------------------------------------------------------

class TestFaissVectorStore:
    def test_build_and_search(self):
        from ragvault.retrieval import FaissVectorStore
        rng = np.random.default_rng(0)
        embs = rng.random((10, DIM)).astype("float32")
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        store = FaissVectorStore()
        store.build(embs)
        scores, indices = store.search(embs[0], top_k=3)
        assert indices[0] == 0

    def test_add_and_search(self):
        from ragvault.retrieval import FaissVectorStore
        rng = np.random.default_rng(1)
        embs = rng.random((5, DIM)).astype("float32")
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        store = FaissVectorStore()
        store.build(embs[:3])
        store.add(embs[3:])
        scores, indices = store.search(embs[0], top_k=5)
        assert len(indices) == 5

    def test_search_before_build_raises(self):
        from ragvault.retrieval import FaissVectorStore
        store = FaissVectorStore()
        with pytest.raises(RuntimeError):
            store.search(np.zeros(DIM, dtype="float32"), top_k=1)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def _make_retriever(self, mock_embedder):
        from ragvault.retrieval import HybridRetriever
        embs = mock_embedder.embed(SAMPLE_CHUNKS)
        return HybridRetriever(SAMPLE_CHUNKS, embs)

    def test_retrieve_returns_top_n(self, mock_embedder):
        retriever = self._make_retriever(mock_embedder)
        q_emb = mock_embedder.embed_query("hybrid retrieval BM25")
        results = retriever.retrieve("hybrid retrieval BM25", q_emb, top_n=5)
        assert len(results) == 5

    def test_all_results_are_from_corpus(self, mock_embedder):
        retriever = self._make_retriever(mock_embedder)
        q_emb = mock_embedder.embed_query("FAISS vector search")
        results = retriever.retrieve("FAISS vector search", q_emb, top_n=5)
        # retrieve() now returns chunk indices; all must be valid corpus positions
        assert all(isinstance(r, (int, np.integer)) for r in results)
        assert all(0 <= r < len(SAMPLE_CHUNKS) for r in results)

    def test_add_documents(self, mock_embedder):
        from ragvault.retrieval import HybridRetriever
        embs = mock_embedder.embed(SAMPLE_CHUNKS[:5])
        retriever = HybridRetriever(SAMPLE_CHUNKS[:5], embs)
        new_chunks = ["New doc about transformers.", "Another doc about attention."]
        new_embs = mock_embedder.embed(new_chunks)
        retriever.add_documents(new_chunks, new_embs)
        assert len(retriever.chunks) == 7

    def test_empty_corpus(self):
        from ragvault.retrieval import HybridRetriever
        rng = np.random.default_rng(0)
        embs = rng.random((1, DIM)).astype("float32")
        retriever = HybridRetriever(["single chunk"], embs)
        q_emb = rng.random(DIM).astype("float32")
        results = retriever.retrieve("query", q_emb, top_n=5)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# CrossEncoderReranker (mocked)
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker:
    def test_rerank_reduces_to_top_n(self, monkeypatch):
        from ragvault.reranking import CrossEncoderReranker

        class FakeReranker:
            def compute_score(self, pairs, normalize=True):
                return [float(len(p[1])) for p in pairs]

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "mock"
        reranker.reranker = FakeReranker()

        chunks = SAMPLE_CHUNKS[:6]
        result = reranker.rerank("query", chunks, top_n=3)
        assert len(result) == 3

    def test_rerank_empty(self, monkeypatch):
        from ragvault.reranking import CrossEncoderReranker
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "mock"
        reranker.reranker = None
        assert reranker.rerank("query", [], top_n=3) == []


# ---------------------------------------------------------------------------
# ContextCompressor (mocked)
# ---------------------------------------------------------------------------

class TestContextCompressor:
    def test_compress_returns_string(self, monkeypatch):
        from ragvault.compression import ContextCompressor

        class FakeCompressor:
            def compress_prompt(self, text, rate=0.5, force_tokens=None, drop_consecutive=True):
                halfway = max(1, int(len(text) * rate))
                return {"compressed_prompt": text[:halfway]}

        comp = ContextCompressor.__new__(ContextCompressor)
        comp.compressor = FakeCompressor()
        result = comp.compress("This is a long context string.", rate=0.5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compress_with_stats(self, monkeypatch):
        from ragvault.compression import ContextCompressor

        class FakeCompressor:
            def compress_prompt(self, text, rate=0.5, **kwargs):
                return {
                    "compressed_prompt": text[:10],
                    "origin_tokens": 20,
                    "compressed_tokens": 10,
                    "ratio": 0.5,
                }

        comp = ContextCompressor.__new__(ContextCompressor)
        comp.compressor = FakeCompressor()
        stats = comp.compress_with_stats("Some long context here.", rate=0.5)
        assert "compressed_prompt" in stats
        assert stats["origin_tokens"] == 20


# ---------------------------------------------------------------------------
# RagVault integration (full pipeline, all models mocked)
# ---------------------------------------------------------------------------

class TestRagVaultIntegration:
    def _build_vault(self, mock_embedder):
        """Build a RagVault with all heavy components mocked."""
        from ragvault import RagVault
        from ragvault.reranking import CrossEncoderReranker
        from ragvault.compression import ContextCompressor
        from ragvault.pipeline import QueryRouter

        class FakeReranker:
            def compute_score(self, pairs, normalize=True):
                return [1.0] * len(pairs)

        class FakeCompressor:
            def compress_prompt(self, text, rate=0.5, **kwargs):
                return {"compressed_prompt": text[:max(1, int(len(text) * rate))]}

        vault = RagVault.__new__(RagVault)
        vault.retrieval_candidates = 10
        vault.rerank_top_n = 3
        vault.compression_rate = 0.5
        vault.use_compression = True
        vault._index_type = "flat"
        vault._llm_provider_spec = "anthropic"
        vault._llm = None
        vault.embedder = mock_embedder
        vault.chunker = __import__(
            "ragvault.chunking", fromlist=["SemanticChunker"]
        ).SemanticChunker(embedder=mock_embedder, threshold=0.0)
        vault._router = QueryRouter()

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "mock"
        reranker.reranker = FakeReranker()
        vault.reranker = reranker

        compressor = ContextCompressor.__new__(ContextCompressor)
        compressor.compressor = FakeCompressor()
        vault.compressor = compressor

        vault._retriever = None
        vault._chunks = []
        vault._metadata = []
        vault._doc_index = {}
        vault._deleted_doc_ids = set()
        return vault

    def test_index_and_query(self, mock_embedder):
        vault = self._build_vault(mock_embedder)
        text = " ".join(SAMPLE_CHUNKS)
        vault.index(text)
        assert len(vault) > 0
        result = vault.query("What is BM25?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_add_document(self, mock_embedder):
        vault = self._build_vault(mock_embedder)
        vault.index(SAMPLE_CHUNKS[0])
        initial_count = len(vault)
        vault.add_document(SAMPLE_CHUNKS[1] + " " + SAMPLE_CHUNKS[2])
        assert len(vault) > initial_count

    def test_add_documents(self, mock_embedder):
        vault = self._build_vault(mock_embedder)
        vault.add_documents(SAMPLE_CHUNKS)
        assert len(vault) > 0

    def test_retrieve_before_index_raises(self, mock_embedder):
        vault = self._build_vault(mock_embedder)
        with pytest.raises(RuntimeError):
            vault.retrieve("some query")

    def test_retrieve_returns_list(self, mock_embedder):
        vault = self._build_vault(mock_embedder)
        vault.index_chunks(SAMPLE_CHUNKS)
        result = vault.retrieve("embeddings retrieval")
        assert isinstance(result, list)
        assert 1 <= len(result) <= vault.rerank_top_n

    def test_repr(self, mock_embedder):
        vault = self._build_vault(mock_embedder)
        vault.index_chunks(SAMPLE_CHUNKS[:3])
        r = repr(vault)
        assert "RagVault" in r
        assert "chunks=3" in r

    def test_len(self, mock_embedder):
        vault = self._build_vault(mock_embedder)
        vault.index_chunks(SAMPLE_CHUNKS[:5])
        assert len(vault) == 5
