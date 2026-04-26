# ragvault

**Production-grade RAG in one pip install.**

[![PyPI version](https://img.shields.io/pypi/v/ragvault.svg)](https://pypi.org/project/ragvault/)
[![Python](https://img.shields.io/pypi/pyversions/ragvault.svg)](https://pypi.org/project/ragvault/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Genious07%2FRagvault-blue?logo=github)](https://github.com/Genious07/Ragvault)

```bash
pip install ragvault
```

Most RAG implementations use basic chunking, a single retrieval method, and skip reranking. ragvault stacks five state-of-the-art techniques into a single unified pipeline — from raw text to a compressed, Claude-powered answer.

---

## Live Links

- **PyPI:** https://pypi.org/project/ragvault/0.1.0/
- **GitHub:** https://github.com/Genious07/Ragvault

---

## Pipeline at a glance

```
Raw Text
    │
    ▼  ① Semantic Chunking
    │     Split on topic shifts (cosine similarity), not fixed token counts.
    │     Every chunk covers one coherent idea.
    │
    ▼  ② BGE Embeddings  (BAAI/bge-large-en-v1.5)
    │     State-of-the-art dense vectors, #1 on MTEB leaderboard.
    │     Separate encode paths for queries vs passages.
    │
    ▼  ③ Hybrid Retrieval  (FAISS + BM25 → RRF)
    │     Dense search finds semantically similar chunks.
    │     BM25 catches exact keyword matches.
    │     Reciprocal Rank Fusion merges both lists without score normalisation.
    │
    ▼  ④ Cross-Encoder Reranking  (BAAI/bge-reranker-large)
    │     Scores each (query, chunk) pair jointly.
    │     Far more accurate than bi-encoder dot products.
    │     Runs only over ~20 candidates — fast enough for production.
    │
    ▼  ⑤ Context Compression  (LLMLingua-2)
    │     Removes redundant tokens from retrieved context.
    │     Typically 50% token reduction with minimal quality loss.
    │     Cuts LLM API cost and latency in half.
    │
    ▼  ⑥ LLM Answer  (Claude via Anthropic API)
    │     Grounded answer generation using only the retrieved context.
    │     Hallucination-resistant by design.
    │
    ▼  ⑦ RAGAS Evaluation
          Faithfulness · Answer Relevancy · Context Precision · Context Recall
```

---

## Quick start

```python
from ragvault import RagVault

vault = RagVault()
vault.index(open("my_doc.txt").read())

# Compressed context string — plug into any LLM
context = vault.query("What is hybrid retrieval?")

# Or let ragvault call Claude and return the answer directly
answer = vault.ask("What is hybrid retrieval?")
print(answer)
```

Set your API key before calling `.ask()`:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Installation

```bash
pip install ragvault
```

**What gets installed:**

| Package | Purpose |
|---|---|
| `FlagEmbedding` | BGE dense embeddings + cross-encoder reranker |
| `faiss-cpu` | Fast approximate nearest-neighbour vector search |
| `rank-bm25` | Sparse BM25 keyword retrieval |
| `llmlingua` | LLMLingua-2 context compression |
| `ragas` | RAG evaluation framework |
| `anthropic` | Claude API for answer generation |
| `torch` | Model inference backend |

> For GPU inference replace `faiss-cpu` with `faiss-gpu` after install.

---

## Feature walkthrough

### Semantic Chunking

Traditional fixed-size chunking cuts sentences mid-thought, polluting chunks with unrelated content and degrading retrieval accuracy. ragvault embeds every sentence using BGE and measures cosine similarity between adjacent sentences. A new chunk begins wherever similarity drops below a configurable threshold, ensuring each chunk covers a single coherent topic.

```python
from ragvault import SemanticChunker, BGEEmbedder

embedder = BGEEmbedder()
chunker  = SemanticChunker(embedder=embedder, threshold=0.75)

chunks = chunker.chunk(long_document)
# or chunk multiple docs at once:
chunks = chunker.chunk_documents([doc1, doc2, doc3])
```

**Why it matters:** A 1000-token chunk that mixes three topics will retrieve for all three — adding noise. A semantic chunk that covers exactly one topic retrieves precisely.

---

### BGE Embeddings

BGE (BAAI General Embeddings) from the Beijing Academy of AI consistently tops the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). ragvault uses `bge-large-en-v1.5` with separate encoding paths — queries get a task instruction prefix, passages do not. This alignment is critical: without it, query and document embeddings live in slightly different semantic spaces.

```python
from ragvault import BGEEmbedder

embedder = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5")

doc_vectors   = embedder.embed(["passage one", "passage two"])   # (N, 1024)
query_vector  = embedder.embed_query("what is semantic chunking?")  # (1024,)
```

Swap to `BAAI/bge-m3` for multilingual support across 100+ languages.

---

### Hybrid Retrieval

Dense retrieval alone misses exact keyword matches. BM25 alone misses semantic paraphrases. ragvault runs both and fuses the results using **Reciprocal Rank Fusion (RRF)**:

```
RRF score(doc) = Σ  1 / (k + rank_in_system)   where k = 60
```

k=60 is the standard smoothing constant — it prevents the top-ranked document in one system from dominating when the other system ranks it low. No score normalisation is needed; only ranks matter. This makes fusion robust to score distribution differences between dense and sparse systems.

```python
from ragvault import HybridRetriever
import numpy as np

retriever = HybridRetriever(chunks, embeddings, rrf_k=60)
results   = retriever.retrieve("my query", query_embedding, top_n=20)

# Incremental indexing — no need to rebuild from scratch
retriever.add_documents(new_chunks, new_embeddings)
```

---

### Cross-Encoder Reranking

Bi-encoders (like BGE) encode query and document independently, then compare vectors. Cross-encoders process query and document **together as a single input**, allowing the model to attend to token-level interactions between them. This produces much more accurate relevance scores.

The trade-off: cross-encoders are too slow to run over a full corpus (O(n) inference calls) but fast enough to rescore a short candidate list of 20 documents.

```python
from ragvault import CrossEncoderReranker

reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-large")

top5 = reranker.rerank(query, candidates, top_n=5)

# Or get all chunks with scores for custom filtering
scored = reranker.rerank_with_scores(query, candidates)
# → [("most relevant chunk", 0.97), ("second best", 0.84), ...]
```

---

### Context Compression

After reranking, the top-5 chunks are concatenated and sent to an LLM. But these chunks often contain filler sentences, repeated context, and low-information tokens that increase cost without improving answers.

LLMLingua-2 uses a BERT-based classifier trained to label each token as essential or droppable. At 50% compression, roughly half the tokens are removed while preserving key facts.

```python
from ragvault import ContextCompressor

compressor = ContextCompressor()

compressed = compressor.compress(context, rate=0.5)

# With token statistics
stats = compressor.compress_with_stats(context, rate=0.5)
print(f"Compressed {stats['origin_tokens']} → {stats['compressed_tokens']} tokens")
print(f"Ratio: {stats['ratio']:.1%}")
```

**Impact:** At 50% compression with 5 chunks of ~200 tokens each = ~500 tokens saved per query. At $0.003 per 1K tokens, that's meaningful at scale.

---

### LLM Answer Generation

`vault.ask()` runs the complete retrieval pipeline, compresses the context, and calls Claude to generate a grounded answer.

```python
vault = RagVault()
vault.index(document)

answer = vault.ask(
    "What is semantic chunking?",
    model="claude-sonnet-4-6",      # or claude-opus-4-7 for hardest questions
    max_tokens=512,
    system_prompt=None,             # uses a sensible default grounding prompt
)
```

The default system prompt instructs Claude to answer **only from the provided context** and say "I don't have enough information" when the context doesn't support an answer. This minimises hallucinations.

---

### RAGAS Evaluation

RAGAS measures four dimensions that cover the full RAG failure surface:

| Metric | What fails without it | Score range |
|---|---|---|
| **Faithfulness** | LLM makes up facts not in the context | 0 → 1 |
| **Answer Relevancy** | LLM answers a different question | 0 → 1 |
| **Context Precision** | Retriever returns noisy / off-topic chunks | 0 → 1 |
| **Context Recall** | Retriever misses chunks needed for the answer | 0 → 1 |

```python
results = vault.evaluate(
    questions    = ["What is RRF?", "How does compression work?"],
    answers      = ["RRF fuses ranked lists ...", "LLMLingua removes ..."],
    contexts     = [["chunk about RRF ..."], ["chunk about LLMLingua ..."]],
    ground_truths= ["RRF scores docs as 1/(k+rank) ...", "LLMLingua-2 uses BERT ..."],
)

print(results)
# {'faithfulness': 0.96, 'answer_relevancy': 0.91,
#  'context_precision': 0.88, 'context_recall': 0.93}
```

---

## Multi-document indexing

ragvault supports incremental indexing — add documents one at a time without rebuilding from scratch.

```python
vault = RagVault()

# Start with one document
vault.index(primary_doc)
print(f"{len(vault)} chunks")

# Add more without re-indexing
vault.add_document(second_doc)
vault.add_documents([third_doc, fourth_doc, fifth_doc])

print(f"{len(vault)} chunks total")

# Query across the entire corpus
answer = vault.ask("Compare the approaches in document 1 and document 3.")
```

---

## Configuration reference

```python
vault = RagVault(
    # Embedding model — swap for bge-m3 for multilingual
    embedding_model    = "BAAI/bge-large-en-v1.5",

    # Reranker model — bge-reranker-v2-m3 for multilingual
    reranker_model     = "BAAI/bge-reranker-large",

    # Semantic chunking sensitivity
    # Lower = fewer, larger chunks (more context per chunk)
    # Higher = more, smaller chunks (more precise retrieval)
    chunk_threshold    = 0.75,

    # How many candidates to pass to the cross-encoder
    # Higher = more recall, slower reranking
    retrieval_candidates = 20,

    # How many chunks to keep after reranking
    rerank_top_n       = 5,

    # LLMLingua compression ratio (0.5 = keep 50% of tokens)
    compression_rate   = 0.5,

    # Set False to skip compression (faster, higher token cost)
    use_compression    = True,
)
```

---

## Running the demo

```bash
git clone https://github.com/Genious07/Ragvault.git
cd Ragvault
pip install -e .

export ANTHROPIC_API_KEY=sk-ant-...

python example.py
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 26 tests run with lightweight NumPy mocks — no GPU, no model download, no API key needed.

```
26 passed in 2.46s
```

---

## Why each choice was made

| Decision | Reason |
|---|---|
| BGE over OpenAI embeddings | No API cost, runs locally, competitive quality on MTEB |
| FAISS over Chroma/Qdrant | Zero infrastructure, in-memory, production-fast |
| RRF over score interpolation | No normalisation needed, robust to score distribution mismatch |
| BGE reranker over Cohere Rerank | Free, local, same quality tier |
| LLMLingua-2 over naive truncation | Preserves key facts; truncation cuts from the end blindly |
| RAGAS over human eval | Scalable, automated, covers retrieval + generation jointly |

---

## License

MIT — use freely in commercial and personal projects.

---

*Built with Claude Code · Published on [PyPI](https://pypi.org/project/ragvault/0.1.0/) · Source on [GitHub](https://github.com/Genious07/Ragvault)*
