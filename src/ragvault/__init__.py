from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Iterator

import numpy as np

from .chunking import SemanticChunker
from .compression import ContextCompressor
from .embeddings import BGEEmbedder
from .evaluation import RAGASEvaluator
from .llm import AnthropicProvider, LLMProvider, OllamaProvider, OpenAIProvider
from .pipeline import AnswerWithCitations, Citation, PipelineTrace, QueryRouter
from .reranking import CrossEncoderReranker
from .retrieval import BM25Retriever, FaissVectorStore, HybridRetriever

__version__ = "0.2.0"
__all__ = [
    "RagVault",
    "SemanticChunker",
    "BGEEmbedder",
    "HybridRetriever",
    "CrossEncoderReranker",
    "ContextCompressor",
    "RAGASEvaluator",
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "Citation",
    "AnswerWithCitations",
    "PipelineTrace",
]

_DEFAULT_SYSTEM = (
    "You are a precise question-answering assistant. "
    "Answer the user's question using ONLY the provided context. "
    "If the answer is not in the context, say 'I don't have enough information to answer that.' "
    "Be concise and factual."
)

_PROVIDER_DEFAULTS = {
    "anthropic": AnthropicProvider.DEFAULT_MODEL,
    "openai": OpenAIProvider.DEFAULT_MODEL,
    "ollama": OllamaProvider.DEFAULT_MODEL,
}


class RagVault:
    """
    Self-optimizing RAG engine with adaptive pipeline routing.

    What makes ragvault different from LangChain / LlamaIndex
    ──────────────────────────────────────────────────────────
    1. Adaptive routing   — every query is classified (fast / balanced / quality)
       and routed through the cheapest pipeline that can answer it correctly.
       Simple lookups skip BM25 and reranking entirely.  Complex analytical
       queries run the full hybrid+rerank+compress stack automatically.
       No other library does this transparently.

    2. Citations built-in — ask_with_citations() returns the exact source chunks
       with confidence scores and metadata alongside the answer.  Explainability
       is a first-class feature, not an afterthought.

    3. Pipeline introspection — debug_query() shows stage-by-stage timing,
       candidate counts, and reranker score distributions for every query.

    4. Disk persistence   — save() / load() serialise the entire vault
       (FAISS index, BM25 model, chunks, metadata) so you never re-embed.

    5. Multi-LLM          — swap Claude, GPT-4o, or local Ollama models with
       one constructor argument.  anthropic / openai are optional extras.

    6. Auto-tuning        — tune() optimises retrieval_candidates, rerank_top_n,
       and compression_rate against your own RAGAS-scored sample queries.

    Quick start::

        vault = RagVault()
        vault.index("your long document text...")
        answer = vault.ask("your question")                 # auto-routed
        result = vault.ask_with_citations("your question")  # with sources
        vault.save("./my_vault")                            # persist

        vault2 = RagVault.load("./my_vault")               # reload instantly
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-large",
        chunk_threshold: float = 0.75,
        retrieval_candidates: int = 20,
        rerank_top_n: int = 5,
        compression_rate: float = 0.5,
        use_compression: bool = False,
        index_type: str = "flat",
        llm_provider: str | LLMProvider = "anthropic",
    ) -> None:
        self._embedding_model = embedding_model
        self._reranker_model = reranker_model
        self._chunk_threshold = chunk_threshold
        self.retrieval_candidates = retrieval_candidates
        self.rerank_top_n = rerank_top_n
        self.compression_rate = compression_rate
        self.use_compression = use_compression
        self._index_type = index_type
        self._llm_provider_spec = llm_provider

        self.embedder = BGEEmbedder(model_name=embedding_model)
        self.chunker = SemanticChunker(embedder=self.embedder, threshold=chunk_threshold)
        self.reranker = CrossEncoderReranker(model_name=reranker_model)
        self._router = QueryRouter()

        if use_compression:
            try:
                self.compressor: ContextCompressor | None = ContextCompressor()
            except ImportError:
                raise ImportError(
                    "llmlingua not installed. Run: pip install 'ragvault[compression]'"
                )
        else:
            self.compressor = None

        self._retriever: HybridRetriever | None = None
        self._chunks: list[str] = []
        self._metadata: list[dict] = []
        self._doc_index: dict[str, list[int]] = {}
        self._deleted_doc_ids: set[str] = set()
        self._llm: LLMProvider | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_llm(self) -> LLMProvider:
        if self._llm is not None:
            return self._llm
        spec = self._llm_provider_spec
        if isinstance(spec, LLMProvider):
            self._llm = spec
        elif spec == "anthropic":
            self._llm = AnthropicProvider()
        elif spec == "openai":
            self._llm = OpenAIProvider()
        elif spec == "ollama":
            self._llm = OllamaProvider()
        else:
            raise ValueError(
                f"Unknown provider {spec!r}. Use 'anthropic', 'openai', 'ollama', "
                "or an LLMProvider instance."
            )
        return self._llm

    def _default_model(self) -> str:
        spec = self._llm_provider_spec
        if isinstance(spec, str):
            return _PROVIDER_DEFAULTS.get(spec, AnthropicProvider.DEFAULT_MODEL)
        return AnthropicProvider.DEFAULT_MODEL

    def _active_indices(self) -> set[int] | None:
        """Return set of non-deleted indices, or None if nothing is deleted."""
        if not self._deleted_doc_ids:
            return None
        deleted: set[int] = set()
        for doc_id in self._deleted_doc_ids:
            deleted.update(self._doc_index.get(doc_id, []))
        return set(range(len(self._chunks))) - deleted

    def _allowed_indices(self, filter: dict | None) -> set[int] | None:
        """Combine soft-delete mask with optional metadata filter."""
        active = self._active_indices()
        if not filter:
            return active
        base = active if active is not None else set(range(len(self._chunks)))
        result = {
            i for i in base
            if i < len(self._metadata)
            and all(self._metadata[i].get(k) == v for k, v in filter.items())
        }
        return result

    def _build_context(self, chunks: list[str]) -> str:
        context = "\n\n".join(chunks)
        if self.use_compression and self.compressor and context:
            context = self.compressor.compress(context, rate=self.compression_rate)
        return context

    def _llm_messages(
        self, question: str, context: str, system_prompt: str | None
    ) -> tuple[str, str]:
        system = system_prompt or _DEFAULT_SYSTEM
        user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        return system, user

    # ──────────────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────────────

    def index(self, text: str, metadata: dict | None = None) -> str:
        """
        Chunk and index a single document, replacing any existing index.
        Returns the doc_id assigned to this document.
        """
        self._chunks = []
        self._metadata = []
        self._doc_index = {}
        self._deleted_doc_ids = set()
        self._retriever = None
        return self.add_documents([text], [metadata or {}])[0]

    def index_chunks(self, chunks: list[str], metadata: dict | None = None) -> str:
        """
        Index pre-split chunks directly, replacing any existing index.
        Returns the doc_id assigned.
        """
        self._chunks = []
        self._metadata = []
        self._doc_index = {}
        self._deleted_doc_ids = set()
        self._retriever = None

        doc_id = str(uuid.uuid4())
        meta = metadata or {}
        self._chunks = list(chunks)
        self._metadata = [{"doc_id": doc_id, **meta} for _ in chunks]
        embeddings = self.embedder.embed(chunks)
        self._doc_index[doc_id] = list(range(len(chunks)))
        self._retriever = HybridRetriever(chunks, embeddings, index_type=self._index_type)
        return doc_id

    def add_document(self, text: str, metadata: dict | None = None) -> str:
        """
        Add a document to the existing index (incremental).
        Returns the doc_id assigned.
        """
        return self.add_documents([text], [metadata or {}])[0]

    def add_documents(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> list[str]:
        """
        Add multiple documents in a single batched operation.

        Key optimisation: all new chunks are embedded in ONE call and BM25 is
        rebuilt ONCE regardless of how many documents are added.  Calling this
        with 100 documents is the same cost as calling add_document() on each
        individually but with O(n) instead of O(n²) BM25 rebuilds.

        Args:
            texts:    List of raw document strings.
            metadata: Optional per-document metadata dicts.  Keys become
                      filterable in retrieve() via the filter= argument.
                      'doc_id' is reserved and will be overwritten.

        Returns:
            List of doc_ids in the same order as texts.
        """
        if metadata is None:
            metadata = [{} for _ in texts]

        all_chunks: list[str] = []
        all_embs: list[np.ndarray] = []
        all_meta: list[dict] = []
        doc_ids: list[str] = []
        chunk_counts: list[int] = []

        for text, meta in zip(texts, metadata):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            # return_embeddings=True avoids re-embedding chunks a second time;
            # sentence embeddings are pooled per chunk (saves one full embed pass)
            chunks, embs = self.chunker.chunk(text, return_embeddings=True)
            chunk_counts.append(len(chunks))
            for chunk, emb in zip(chunks, embs):
                all_chunks.append(chunk)
                all_embs.append(emb)
                all_meta.append({"doc_id": doc_id, **meta})

        if not all_chunks:
            return doc_ids

        new_embeddings = np.stack(all_embs)
        start_idx = len(self._chunks)
        self._chunks.extend(all_chunks)
        self._metadata.extend(all_meta)

        idx = start_idx
        for doc_id, count in zip(doc_ids, chunk_counts):
            if count:
                self._doc_index[doc_id] = list(range(idx, idx + count))
                idx += count

        if self._retriever is None:
            self._retriever = HybridRetriever(
                all_chunks, new_embeddings, index_type=self._index_type
            )
        else:
            self._retriever.add_documents(all_chunks, new_embeddings)

        return doc_ids

    def delete_document(self, doc_id: str) -> int:
        """
        Soft-delete all chunks associated with doc_id.
        They are excluded from all future retrieval immediately.
        Call compact() to permanently remove them and reclaim memory.

        Returns the number of chunks removed from retrieval.
        """
        if doc_id not in self._doc_index:
            raise KeyError(f"Document {doc_id!r} not found in this vault.")
        self._deleted_doc_ids.add(doc_id)
        return len(self._doc_index[doc_id])

    def compact(self) -> None:
        """
        Permanently rebuild the index without deleted documents.
        Frees memory and speeds up future retrieval.
        """
        if not self._deleted_doc_ids:
            return
        active_set = self._active_indices() or set(range(len(self._chunks)))
        active = sorted(active_set)
        new_chunks = [self._chunks[i] for i in active]
        new_meta = [self._metadata[i] for i in active]
        if not new_chunks:
            self._chunks, self._metadata, self._doc_index = [], [], {}
            self._deleted_doc_ids.clear()
            self._retriever = None
            return

        new_embeddings = self.embedder.embed(new_chunks)
        self._chunks = new_chunks
        self._metadata = new_meta
        self._deleted_doc_ids.clear()
        self._doc_index = {}
        for i, meta in enumerate(self._metadata):
            did = meta.get("doc_id", "")
            self._doc_index.setdefault(did, []).append(i)
        self._retriever = HybridRetriever(
            new_chunks, new_embeddings, index_type=self._index_type
        )

    # ──────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        filter: dict | None = None,
        mode: str = "auto",
        top_n: int | None = None,
    ) -> list[str]:
        """
        Retrieve the most relevant chunks for query.

        Args:
            query:  The search query.
            filter: Metadata filter dict, e.g. {"source": "annual_report.pdf"}.
                    Only chunks whose metadata contains all key-value pairs are eligible.
            mode:   'auto' (default) — classifier picks fast / balanced / quality.
                    'fast'     — dense-only, no reranking.
                    'balanced' — hybrid + reranking.
                    'quality'  — hybrid + reranking + compression.
            top_n:  Override rerank_top_n for this call.

        Returns:
            List of chunk strings, best-first.
        """
        if self._retriever is None:
            raise RuntimeError("No documents indexed. Call .index() or .add_document() first.")

        if mode == "auto":
            mode = self._router.route(query)

        _top_n = top_n or self.rerank_top_n
        allowed = self._allowed_indices(filter)
        if allowed is not None and not allowed:
            return []

        query_emb = self.embedder.embed_query(query)
        indices = self._retriever.retrieve(
            query, query_emb,
            top_n=self.retrieval_candidates,
            allowed_indices=allowed,
            mode=mode,
        )

        if mode == "fast":
            return [self._chunks[i] for i in indices[:_top_n]]

        candidates = [(self._chunks[i], i) for i in indices]
        ranked = self.reranker.rerank_tagged(query, candidates, top_n=_top_n)
        return [chunk for chunk, _, _ in ranked]

    def query(
        self,
        query: str,
        filter: dict | None = None,
        mode: str = "auto",
    ) -> str:
        """
        Retrieve context and (optionally) compress it.
        Returns a single string ready to be inserted into an LLM prompt.
        """
        chunks = self.retrieve(query, filter=filter, mode=mode)
        return self._build_context(chunks)

    # ──────────────────────────────────────────────────────────────────────
    # LLM answer generation
    # ──────────────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        model: str | None = None,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        filter: dict | None = None,
        mode: str = "auto",
    ) -> str:
        """
        Retrieve context then generate an answer via the configured LLM.

        Works with Claude (default), GPT-4o, or local Ollama models.
        Requires the corresponding optional extra:
            pip install 'ragvault[anthropic]'   # Claude
            pip install 'ragvault[openai]'      # GPT-4o
            # Ollama: no extra needed, just run Ollama locally
        """
        context = self.query(question, filter=filter, mode=mode)
        system, user = self._llm_messages(question, context, system_prompt)
        return self._get_llm().complete(
            system=system,
            user=user,
            model=model or self._default_model(),
            max_tokens=max_tokens,
        )

    def ask_stream(
        self,
        question: str,
        model: str | None = None,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        filter: dict | None = None,
        mode: str = "auto",
    ) -> Iterator[str]:
        """
        Stream LLM tokens as they are generated.

        Usage::

            for token in vault.ask_stream("What is X?"):
                print(token, end="", flush=True)
        """
        context = self.query(question, filter=filter, mode=mode)
        system, user = self._llm_messages(question, context, system_prompt)
        yield from self._get_llm().stream(
            system=system,
            user=user,
            model=model or self._default_model(),
            max_tokens=max_tokens,
        )

    async def ask_async(
        self,
        question: str,
        model: str | None = None,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        filter: dict | None = None,
        mode: str = "auto",
    ) -> str:
        """
        Async version of ask(). Runs the blocking pipeline in a thread executor
        so it does not block an asyncio event loop (FastAPI, aiohttp, etc.).
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.ask(
                question,
                model=model,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                filter=filter,
                mode=mode,
            ),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Differentiator #1 — Citations with provenance
    # ──────────────────────────────────────────────────────────────────────

    def ask_with_citations(
        self,
        question: str,
        model: str | None = None,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        filter: dict | None = None,
        mode: str = "auto",
    ) -> AnswerWithCitations:
        """
        Generate an answer AND return exactly which source chunks it came from,
        complete with relevance scores, original rank, and any metadata you
        attached at index time (source filename, page number, URL, etc.).

        No other RAG library ships citations as a first-class return value.

        Example::

            result = vault.ask_with_citations("What is the refund policy?")
            print(result.answer)
            for c in result.citations:
                print(f"  [{c.rank}] score={c.score:.3f}  source={c.metadata.get('source')}")
                print(f"       {c.chunk[:120]}...")
        """
        import time

        if self._retriever is None:
            raise RuntimeError("No documents indexed.")

        if mode == "auto":
            mode = self._router.route(question)

        timings: dict[str, float] = {}
        allowed = self._allowed_indices(filter)

        t = time.perf_counter()
        query_emb = self.embedder.embed_query(question)
        timings["embed_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        indices = self._retriever.retrieve(
            question, query_emb,
            top_n=self.retrieval_candidates,
            allowed_indices=allowed,
            mode=mode,
        )
        timings["retrieve_ms"] = (time.perf_counter() - t) * 1000

        candidates = [(self._chunks[i], i) for i in indices]

        if mode == "fast":
            top = candidates[: self.rerank_top_n]
            raw: list[tuple[str, int, float]] = [
                (c, i, 1.0 - rank * 0.05) for rank, (c, i) in enumerate(top)
            ]
            timings["rerank_ms"] = 0.0
        else:
            t = time.perf_counter()
            raw = self.reranker.rerank_tagged(question, candidates, top_n=self.rerank_top_n)
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        chunks = [c for c, _, _ in raw]
        context = "\n\n".join(chunks)

        if self.use_compression and self.compressor and context:
            t = time.perf_counter()
            context = self.compressor.compress(context, rate=self.compression_rate)
            timings["compress_ms"] = (time.perf_counter() - t) * 1000
        else:
            timings["compress_ms"] = 0.0

        system, user = self._llm_messages(question, context, system_prompt)
        t = time.perf_counter()
        answer = self._get_llm().complete(
            system=system,
            user=user,
            model=model or self._default_model(),
            max_tokens=max_tokens,
        )
        timings["llm_ms"] = (time.perf_counter() - t) * 1000

        citations = [
            Citation(
                chunk=chunk,
                score=round(score, 4),
                rank=rank,
                metadata=self._metadata[idx] if idx < len(self._metadata) else {},
            )
            for rank, (chunk, idx, score) in enumerate(raw)
        ]

        return AnswerWithCitations(
            answer=answer,
            citations=citations,
            query_mode=mode,
            pipeline_ms={k: round(v, 2) for k, v in timings.items()},
        )

    # ──────────────────────────────────────────────────────────────────────
    # Differentiator #2 — Pipeline introspection
    # ──────────────────────────────────────────────────────────────────────

    def debug_query(
        self,
        query: str,
        filter: dict | None = None,
        mode: str = "auto",
    ) -> PipelineTrace:
        """
        Run the retrieval pipeline and return a detailed trace showing:
        - Which routing mode was selected (fast / balanced / quality)
        - How many candidates were retrieved vs reranked
        - Token counts before and after compression
        - Millisecond timing for every stage
        - Reranker score distribution for the top results

        Use this to diagnose why a query returned poor results, to measure
        latency hot-spots, or to decide whether to enable compression.

        Example::

            trace = vault.debug_query("why did revenue decline?")
            print(trace.query_mode)        # 'quality' (why → complex signal)
            print(trace.timing_ms)         # {'embed_ms': 12.4, 'retrieve_ms': 3.1, ...}
            print(trace.top_scores)        # [0.94, 0.87, 0.81, ...]
        """
        import time

        if self._retriever is None:
            raise RuntimeError("No documents indexed.")

        if mode == "auto":
            mode = self._router.route(query)

        timings: dict[str, float] = {}
        allowed = self._allowed_indices(filter)

        t = time.perf_counter()
        query_emb = self.embedder.embed_query(query)
        timings["embed_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        indices = self._retriever.retrieve(
            query, query_emb,
            top_n=self.retrieval_candidates,
            allowed_indices=allowed,
            mode=mode,
        )
        timings["retrieve_ms"] = (time.perf_counter() - t) * 1000
        n_candidates = len(indices)

        candidates = [(self._chunks[i], i) for i in indices]

        if mode == "fast":
            top_scores = [1.0 - rank * 0.05 for rank in range(min(self.rerank_top_n, n_candidates))]
            n_reranked = min(self.rerank_top_n, n_candidates)
            timings["rerank_ms"] = 0.0
            indices_for_context = indices[:n_reranked]
        else:
            t = time.perf_counter()
            ranked = self.reranker.rerank_tagged(query, candidates, top_n=self.rerank_top_n)
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000
            top_scores = [round(score, 4) for _, _, score in ranked]
            n_reranked = len(ranked)
            indices_for_context = [i for _, i, _ in ranked]

        context = "\n\n".join(self._chunks[i] for i in indices_for_context)
        tokens_before = len(context.split())
        tokens_after: int | None = None

        if self.use_compression and self.compressor:
            t = time.perf_counter()
            compressed = self.compressor.compress(context, rate=self.compression_rate)
            timings["compress_ms"] = (time.perf_counter() - t) * 1000
            tokens_after = len(compressed.split())
        else:
            timings["compress_ms"] = 0.0

        return PipelineTrace(
            query=query,
            query_mode=mode,
            n_candidates=n_candidates,
            n_reranked=n_reranked,
            tokens_before_compression=tokens_before,
            tokens_after_compression=tokens_after,
            timing_ms={k: round(v, 2) for k, v in timings.items()},
            top_scores=top_scores,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Differentiator #3 — Disk persistence
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save the entire vault to directory `path`.

        Saves: FAISS index, BM25 model, all chunks, metadata, doc_index,
        soft-delete state, and constructor config.  Model weights are NOT
        saved — they are re-loaded from HuggingFace cache on load().

        After save+load you never re-embed your documents, no matter how
        many times you restart your process.

        Example::

            vault.save("./production_vault")
            # next day:
            vault = RagVault.load("./production_vault")
        """
        import json
        import os
        import pickle

        os.makedirs(path, exist_ok=True)

        if self._retriever is not None:
            self._retriever.vector_store.save(os.path.join(path, "faiss.index"))
            with open(os.path.join(path, "state.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "chunks": self._chunks,
                        "metadata": self._metadata,
                        "doc_index": self._doc_index,
                        "deleted_doc_ids": self._deleted_doc_ids,
                        "bm25_chunks": self._retriever.bm25._chunks,
                        "bm25_model": self._retriever.bm25.bm25,
                    },
                    f,
                )

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(
                {
                    "embedding_model": self._embedding_model,
                    "reranker_model": self._reranker_model,
                    "chunk_threshold": self._chunk_threshold,
                    "retrieval_candidates": self.retrieval_candidates,
                    "rerank_top_n": self.rerank_top_n,
                    "compression_rate": self.compression_rate,
                    "use_compression": self.use_compression,
                    "index_type": self._index_type,
                    "llm_provider": (
                        self._llm_provider_spec
                        if isinstance(self._llm_provider_spec, str)
                        else "anthropic"
                    ),
                },
                f,
                indent=2,
            )

    @classmethod
    def load(
        cls,
        path: str,
        llm_provider: str | LLMProvider = "anthropic",
    ) -> "RagVault":
        """
        Load a vault saved with save().

        Model weights are loaded from HuggingFace cache (same as a fresh
        RagVault() call).  Chunk embeddings and BM25 are restored from disk
        — no re-embedding required.

        Args:
            path:         Directory previously passed to save().
            llm_provider: Override the LLM provider (useful if you saved with
                          'anthropic' but now want 'openai').
        """
        import json
        import os
        import pickle

        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No vault found at {path!r} (missing config.json)")

        with open(config_path) as f:
            cfg = json.load(f)

        vault = cls(
            embedding_model=cfg["embedding_model"],
            reranker_model=cfg["reranker_model"],
            chunk_threshold=cfg["chunk_threshold"],
            retrieval_candidates=cfg["retrieval_candidates"],
            rerank_top_n=cfg["rerank_top_n"],
            compression_rate=cfg["compression_rate"],
            use_compression=cfg["use_compression"],
            index_type=cfg.get("index_type", "flat"),
            llm_provider=llm_provider,
        )

        state_path = os.path.join(path, "state.pkl")
        faiss_path = os.path.join(path, "faiss.index")

        if os.path.exists(state_path) and os.path.exists(faiss_path):
            with open(state_path, "rb") as f:
                state = pickle.load(f)

            vault._chunks = state["chunks"]
            vault._metadata = state["metadata"]
            vault._doc_index = state["doc_index"]
            vault._deleted_doc_ids = state["deleted_doc_ids"]

            vector_store = FaissVectorStore.load(faiss_path, index_type=cfg.get("index_type", "flat"))
            bm25 = BM25Retriever.from_state(state["bm25_chunks"], state["bm25_model"])
            vault._retriever = HybridRetriever.from_components(
                vault._chunks, vector_store, bm25
            )

        return vault

    # ──────────────────────────────────────────────────────────────────────
    # Differentiator #4 — Auto-tuning
    # ──────────────────────────────────────────────────────────────────────

    def tune(
        self,
        sample_queries: list[str],
        ground_truths: list[str],
        apply: bool = True,
    ) -> dict:
        """
        Automatically optimise retrieval_candidates, rerank_top_n, and
        compression_rate (if use_compression=True) against your own data.

        Uses RAGAS faithfulness as the objective.  Performs a small grid
        search: 3 × 3 = 9 combinations (× 3 if compression is on = 27).
        Each combination calls ask() once per sample query — budget LLM
        costs accordingly.

        Args:
            sample_queries: Representative questions from your use case.
            ground_truths:  Correct reference answers for each question.
            apply:          If True (default), immediately apply the best params.

        Returns:
            {"best_params": {...}, "best_score": float}

        Example::

            result = vault.tune(
                sample_queries=["What is the return policy?"],
                ground_truths=["Items can be returned within 30 days."],
            )
            print(result)  # {'best_params': {'retrieval_candidates': 20, ...}, 'best_score': 0.91}
        """
        from itertools import product as iproduct

        original = (self.retrieval_candidates, self.rerank_top_n, self.compression_rate)
        best_score = -1.0
        best_params: dict = {}

        rc_grid = [10, 20, 30]
        rtn_grid = [3, 5, 7]
        cr_grid = [0.4, 0.5, 0.6] if self.use_compression else [self.compression_rate]

        for rc, rtn, cr in iproduct(rc_grid, rtn_grid, cr_grid):
            if rtn > rc:
                continue
            self.retrieval_candidates = rc
            self.rerank_top_n = rtn
            self.compression_rate = cr

            answers, contexts_list = [], []
            for q in sample_queries:
                chunks = self.retrieve(q)
                answers.append(self.ask(q))
                contexts_list.append(chunks)

            scores = self.evaluate(
                questions=sample_queries,
                answers=answers,
                contexts=contexts_list,
                ground_truths=ground_truths,
            )
            score = float(scores.get("faithfulness", 0.0))

            if score > best_score:
                best_score = score
                best_params = {"retrieval_candidates": rc, "rerank_top_n": rtn, "compression_rate": cr}

        if apply:
            self.retrieval_candidates = best_params.get("retrieval_candidates", original[0])
            self.rerank_top_n = best_params.get("rerank_top_n", original[1])
            self.compression_rate = best_params.get("compression_rate", original[2])
        else:
            self.retrieval_candidates, self.rerank_top_n, self.compression_rate = original

        return {"best_params": best_params, "best_score": best_score}

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────

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
        Requires: pip install 'ragvault[evaluation]'
        """
        try:
            evaluator = RAGASEvaluator()
        except ImportError:
            raise ImportError(
                "ragas not installed. Run: pip install 'ragvault[evaluation]'"
            )
        return evaluator.evaluate(questions, answers, contexts, ground_truths, metrics)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @property
    def chunks(self) -> list[str]:
        return list(self._chunks)

    @property
    def doc_ids(self) -> list[str]:
        return list(self._doc_index.keys())

    def __len__(self) -> int:
        return len(self._chunks) - sum(
            len(self._doc_index.get(did, []))
            for did in self._deleted_doc_ids
        )

    def __repr__(self) -> str:
        return (
            f"RagVault(v{__version__}, "
            f"docs={len(self._doc_index)}, "
            f"chunks={len(self)}, "
            f"index={self._index_type!r}, "
            f"compression={'on' if self.use_compression else 'off'}, "
            f"llm={self._llm_provider_spec!r})"
        )
