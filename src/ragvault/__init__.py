from __future__ import annotations

from .chunking import SemanticChunker
from .embeddings import BGEEmbedder
from .retrieval import HybridRetriever
from .reranking import CrossEncoderReranker
from .compression import ContextCompressor
from .evaluation import RAGASEvaluator

__version__ = "0.1.0"
__all__ = [
    "RagVault",
    "SemanticChunker",
    "BGEEmbedder",
    "HybridRetriever",
    "CrossEncoderReranker",
    "ContextCompressor",
    "RAGASEvaluator",
]


class RagVault:
    """
    End-to-end RAG pipeline combining:
      1. Semantic chunking  — context-aware splits on topic shifts
      2. BGE embeddings     — state-of-the-art dense vectors (BAAI)
      3. Hybrid retrieval   — FAISS vector search + BM25, fused with RRF
      4. Cross-encoder reranking — precise (query, doc) joint scoring
      5. Context compression — LLMLingua-2 token reduction

    Quick start::

        vault = RagVault()
        vault.index("your long document text...")
        context = vault.query("your question")   # compressed context string
        answer  = vault.ask("your question")     # full answer via Claude
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-large",
        chunk_threshold: float = 0.75,
        retrieval_candidates: int = 20,
        rerank_top_n: int = 5,
        compression_rate: float = 0.5,
        use_compression: bool = True,
    ):
        self.retrieval_candidates = retrieval_candidates
        self.rerank_top_n = rerank_top_n
        self.compression_rate = compression_rate
        self.use_compression = use_compression

        self.embedder = BGEEmbedder(model_name=embedding_model)
        self.chunker = SemanticChunker(embedder=self.embedder, threshold=chunk_threshold)
        self.reranker = CrossEncoderReranker(model_name=reranker_model)
        self.compressor = ContextCompressor() if use_compression else None

        self._retriever: HybridRetriever | None = None
        self._chunks: list[str] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, text: str) -> None:
        """Chunk and index a single document, replacing any existing index."""
        self._chunks = self.chunker.chunk(text)
        embeddings = self.embedder.embed(self._chunks)
        self._retriever = HybridRetriever(self._chunks, embeddings)

    def index_chunks(self, chunks: list[str]) -> None:
        """Index pre-split chunks directly, replacing any existing index."""
        self._chunks = list(chunks)
        embeddings = self.embedder.embed(chunks)
        self._retriever = HybridRetriever(chunks, embeddings)

    def add_document(self, text: str) -> None:
        """Add a document to the existing index (incremental)."""
        new_chunks = self.chunker.chunk(text)
        if not new_chunks:
            return
        new_embeddings = self.embedder.embed(new_chunks)
        if self._retriever is None:
            self._chunks = new_chunks
            self._retriever = HybridRetriever(new_chunks, new_embeddings)
        else:
            self._chunks.extend(new_chunks)
            self._retriever.add_documents(new_chunks, new_embeddings)

    def add_documents(self, texts: list[str]) -> None:
        """Add multiple documents to the existing index."""
        for text in texts:
            self.add_document(text)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[str]:
        """
        Full retrieval pipeline: hybrid retrieval → cross-encoder reranking.
        Returns a list of the most relevant chunks (not compressed).
        """
        if self._retriever is None:
            raise RuntimeError("No documents indexed. Call .index() or .add_document() first.")

        query_embedding = self.embedder.embed_query(query)
        candidates = self._retriever.retrieve(
            query, query_embedding, top_n=self.retrieval_candidates
        )
        return self.reranker.rerank(query, candidates, top_n=self.rerank_top_n)

    def query(self, query: str) -> str:
        """
        Retrieve context and optionally compress it.
        Returns a single string ready to be inserted into an LLM prompt.
        """
        chunks = self.retrieve(query)
        context = "\n\n".join(chunks)
        if self.use_compression and self.compressor and context:
            context = self.compressor.compress(context, rate=self.compression_rate)
        return context

    # ------------------------------------------------------------------
    # LLM answer generation (Claude via Anthropic API)
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ) -> str:
        """
        Retrieve context then call Claude to generate an answer.

        Requires ANTHROPIC_API_KEY to be set in the environment.

        Args:
            question:      The user's question.
            model:         Claude model ID.
            max_tokens:    Maximum tokens in the generated answer.
            system_prompt: Override default system prompt.

        Returns:
            The generated answer string.
        """
        import anthropic

        context = self.query(question)

        default_system = (
            "You are a precise question-answering assistant. "
            "Answer the user's question using ONLY the provided context. "
            "If the answer is not in the context, say 'I don't have enough information to answer that.' "
            "Be concise and factual."
        )

        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt or default_system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
        metrics: list | None = None,
    ):
        """
        Evaluate RAG quality using RAGAS.

        Args:
            questions:     Questions asked.
            answers:       LLM-generated answers.
            contexts:      Retrieved chunks used per question (list of lists).
            ground_truths: Reference / correct answers.
            metrics:       RAGAS metric instances (defaults to 4 standard ones).
        """
        evaluator = RAGASEvaluator()
        return evaluator.evaluate(questions, answers, contexts, ground_truths, metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def chunks(self) -> list[str]:
        return list(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return (
            f"RagVault("
            f"docs={len(self._chunks)}, "
            f"compression={'on' if self.use_compression else 'off'}, "
            f"embedder={self.embedder.model_name!r}"
            f")"
        )
