# Changelog

All notable changes to ragvault are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.2.0] — 2026-04-28

### ⚠️ Breaking Changes

These changes will break code written for v0.1.0.

- **`HybridRetriever.retrieve()` now returns `list[int]` (chunk indices) instead of `list[str]` (chunk text).**
  This enables metadata filtering and soft-delete without coupling the retriever to vault state.
  If you called `retriever.retrieve()` directly, map the returned indices back to text:
  ```python
  # Before (0.1.0)
  results: list[str] = retriever.retrieve(query, query_emb, top_n=5)

  # After (0.2.0)
  indices: list[int] = retriever.retrieve(query, query_emb, top_n=5)
  results: list[str] = [retriever.chunks[i] for i in indices]
  ```

- **`RagVault.index()` now returns a `str` (doc_id) instead of `None`.**
  ```python
  # Before (0.1.0)
  vault.index(text)           # returns None

  # After (0.2.0)
  doc_id = vault.index(text)  # returns str UUID
  ```

- **`RagVault.add_document()` and `add_documents()` now return doc_id(s).**
  ```python
  doc_id  = vault.add_document(text)
  doc_ids = vault.add_documents([text1, text2])
  ```

- **`use_compression` now defaults to `False`.**
  In v0.1.0 compression was on by default, which silently required `llmlingua`.
  Set `use_compression=True` explicitly if you want it.

- **`anthropic` is no longer a required dependency.**
  Install the extra you need:
  ```bash
  pip install 'ragvault[anthropic]'   # Claude (was implicit in 0.1.0)
  pip install 'ragvault[openai]'      # GPT-4o
  # Ollama needs no extra
  ```

---

### Added

**Adaptive pipeline routing**
- `QueryRouter` classifies every query as `fast`, `balanced`, or `quality` before touching the index.
- Simple lookups skip BM25 and reranking. Analytical queries run the full stack. Automatic, zero config.
- Override per-call with `mode="fast"` / `mode="balanced"` / `mode="quality"` on `retrieve()`, `query()`, `ask()`, and `ask_with_citations()`.

**Citations**
- `vault.ask_with_citations(question)` → `AnswerWithCitations`
  Returns the answer and a ranked list of `Citation(chunk, score, rank, metadata)` objects.
  Every answer is now explainable by source.

**Pipeline introspection**
- `vault.debug_query(query)` → `PipelineTrace`
  Returns per-stage timing (`embed_ms`, `retrieve_ms`, `rerank_ms`, `compress_ms`), candidate counts, and reranker score distributions. Use this to diagnose poor results or find latency hot-spots.

**Disk persistence**
- `vault.save("./my_vault")` — serialises FAISS index, BM25 model, all chunks, and metadata to disk.
- `RagVault.load("./my_vault")` — restores the full vault without re-embedding anything.

**Document management**
- `add_document(text, metadata={"source": "report.pdf"})` — attach arbitrary metadata to any document.
- `retrieve(query, filter={"source": "report.pdf"})` — scope retrieval to matching documents.
- `delete_document(doc_id)` — soft-delete all chunks for a document; excluded from retrieval immediately.
- `compact()` — permanently rebuild the index without deleted chunks, freeing memory.

**Multi-LLM support**
- `RagVault(llm_provider="anthropic")` — Claude (default)
- `RagVault(llm_provider="openai")` — GPT-4o
- `RagVault(llm_provider="ollama")` — any local Ollama model, no extra SDK required
- `RagVault(llm_provider=MyProvider())` — any custom `LLMProvider` subclass

**Streaming**
- `vault.ask_stream(question)` — generator that yields tokens as they arrive.
  ```python
  for token in vault.ask_stream("What is X?"):
      print(token, end="", flush=True)
  ```

**Async**
- `await vault.ask_async(question)` — runs the blocking pipeline in a thread executor, compatible with FastAPI and other asyncio frameworks.

**HNSW index for large-scale retrieval**
- `RagVault(index_type="hnsw")` — uses `IndexHNSWFlat` with inner-product metric.
  O(log n) approximate search vs O(n) brute-force. Recommended for 500k+ chunks.

**Auto-tuning**
- `vault.tune(sample_queries, ground_truths)` — grid-searches over `retrieval_candidates`, `rerank_top_n`, and `compression_rate` using RAGAS faithfulness as the objective, then applies the best params.
  Note: makes multiple LLM API calls (up to 9 × len(sample_queries)). Budget accordingly.

**New public types**
- `ragvault.Citation` — `(chunk, score, rank, metadata)`
- `ragvault.AnswerWithCitations` — `(answer, citations, query_mode, pipeline_ms)`
- `ragvault.PipelineTrace` — full per-stage trace
- `ragvault.llm.LLMProvider` — ABC for custom LLM providers
- `ragvault.llm.AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`

---

### Changed

- **`add_documents()` now batches all chunk embeddings in one call and rebuilds BM25 once**, regardless of how many documents are added. In v0.1.0 it called `add_document()` in a loop, causing O(n²) BM25 rebuilds.
- **`SemanticChunker.chunk()` gained a `return_embeddings` parameter.** When `True`, returns `(chunks, embeddings)` by mean-pooling sentence-level embeddings — avoids a second full embed pass at index time.
- **`CrossEncoderReranker` gained `rerank_tagged(query, tagged_chunks, top_n)`** — reranks `(chunk, tag)` pairs and preserves the tag through reranking. Used internally to track chunk indices through the pipeline.
- **`FaissVectorStore` gained `save(path)` / `load(path)` / `index_type` / `ntotal`.**
- **`BM25Retriever` gained `from_state(chunks, bm25_model)`** classmethod for deserialization.
- **`HybridRetriever` gained `from_components()`** classmethod and `allowed_indices` / `mode` params on `retrieve()`.
- **Optional dependencies split** — `anthropic`, `openai`, `compression`, `evaluation` are now separate extras. Core install only requires the retrieval stack.

---

### Fixed

- BM25 index was rebuilt once per document when calling `add_documents([...])` — now rebuilt once total.
- Sentence embeddings computed during chunking were discarded, forcing a second full embed pass at index time — now reused via `return_embeddings=True`.
- `anthropic` SDK was a hard dependency even for users who never called `vault.ask()`.
- Context compression was enabled by default, silently requiring `llmlingua` on every install.

---

## [0.1.0] — 2026-04-26

Initial release.

- Semantic chunking with cosine similarity boundary detection
- BGE-large-en-v1.5 dense embeddings via FlagEmbedding
- FAISS IndexFlatIP vector store
- BM25Okapi sparse retrieval via rank-bm25
- Reciprocal Rank Fusion hybrid retrieval (k=60)
- BGE-reranker-large cross-encoder reranking
- LLMLingua-2 context compression
- RAGAS evaluation (faithfulness, answer relevancy, context precision, context recall)
- `vault.ask()` via Claude (Anthropic SDK)
- 26 unit tests, all mocked (no GPU required)

[0.2.0]: https://github.com/Genious07/Ragvault/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Genious07/Ragvault/releases/tag/v0.1.0
