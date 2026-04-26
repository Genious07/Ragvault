# ragvault

**The best-in-class RAG library.** Five production-grade techniques in one clean package.

```
pip install ragvault
```

---

## What's inside

| Stage | Technique | Model / Library |
|---|---|---|
| Chunking | Semantic (similarity-based splits) | BGE embeddings |
| Embeddings | BGE dense vectors | `BAAI/bge-large-en-v1.5` |
| Retrieval | Hybrid = FAISS + BM25 fused via RRF | `faiss-cpu` + `rank-bm25` |
| Reranking | Cross-encoder joint scoring | `BAAI/bge-reranker-large` |
| Compression | LLM token reduction | LLMLingua-2 |
| Evaluation | RAG quality metrics | RAGAS |

---

## Quick start

```python
from ragvault import RagVault

vault = RagVault()

# Index a document
vault.index(open("my_doc.txt").read())

# Get compressed context ready for any LLM
context = vault.query("What is hybrid retrieval?")

# Or let ragvault call Claude and return the answer directly
answer = vault.ask("What is hybrid retrieval?")
print(answer)
```

---

## Installation

```bash
pip install ragvault
```

**Dependencies installed automatically:**

```
FlagEmbedding   # BGE embeddings + reranker
faiss-cpu       # vector index
rank-bm25       # sparse retrieval
llmlingua       # context compression
ragas           # evaluation
anthropic       # Claude integration
torch           # model inference
```

> For GPU acceleration replace `faiss-cpu` with `faiss-gpu` after install.

---

## Pipeline explained

### 1. Semantic Chunking

Unlike fixed-size chunking, ragvault embeds every sentence and splits only when
cosine similarity between adjacent sentences drops below a threshold. Each chunk
covers a single coherent topic.

```python
from ragvault import SemanticChunker, BGEEmbedder

embedder = BGEEmbedder()
chunker  = SemanticChunker(embedder=embedder, threshold=0.75)
chunks   = chunker.chunk(long_text)
```

### 2. BGE Embeddings

State-of-the-art dense embeddings from BAAI, consistently top-ranked on the
MTEB leaderboard. Separate encoding paths for queries vs passages apply the
correct instruction prefix automatically.

```python
from ragvault import BGEEmbedder

embedder = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5")
doc_vecs  = embedder.embed(["passage one", "passage two"])
query_vec = embedder.embed_query("what is rag?")
```

### 3. Hybrid Retrieval (FAISS + BM25 + RRF)

Dense vector search captures semantic similarity; BM25 captures exact keyword
matches. Reciprocal Rank Fusion merges both ranked lists:

```
score(doc) = Σ  1 / (60 + rank_in_system)
```

No score normalisation needed — only ranks matter.

```python
from ragvault import HybridRetriever
import numpy as np

retriever = HybridRetriever(chunks, embeddings)
results   = retriever.retrieve("my query", query_embedding, top_n=20)
```

### 4. Cross-Encoder Reranking

The hybrid retriever produces ~20 candidates fast. The cross-encoder then
scores each `(query, chunk)` pair jointly, giving much more precise relevance
scores. Only the top-N are kept.

```python
from ragvault import CrossEncoderReranker

reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-large")
top5     = reranker.rerank(query, candidates, top_n=5)
```

### 5. Context Compression

LLMLingua-2 removes redundant tokens from the retrieved context, typically
achieving 50% compression with minimal quality loss. Reduces LLM API costs
and latency.

```python
from ragvault import ContextCompressor

compressor = ContextCompressor()
compressed = compressor.compress(context, rate=0.5)

# With token stats
stats = compressor.compress_with_stats(context, rate=0.5)
print(stats["origin_tokens"], "→", stats["compressed_tokens"])
```

### 6. LLM Answer Generation

`vault.ask()` wraps the full retrieval pipeline and calls Claude to generate
an answer. Requires `ANTHROPIC_API_KEY` in your environment.

```python
vault = RagVault()
vault.index(document)

answer = vault.ask(
    "What is semantic chunking?",
    model="claude-sonnet-4-6",
    max_tokens=512,
)
```

### 7. RAGAS Evaluation

Evaluate your pipeline across four key dimensions:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer contain only information from the context? |
| **Answer Relevancy** | Does the answer actually address the question? |
| **Context Precision** | Are the retrieved chunks useful (not noisy)? |
| **Context Recall** | Were all necessary chunks retrieved? |

```python
results = vault.evaluate(
    questions=["What is RRF?"],
    answers=["RRF combines ranked lists by scoring 1/(k+rank)."],
    contexts=[["RRF merges ranked lists without score normalisation."]],
    ground_truths=["RRF fuses retrieval systems using rank-based scores."],
)
print(results)
```

---

## Multi-document indexing

```python
vault = RagVault()

# Index the first document
vault.index(doc1)

# Add more documents incrementally (no re-indexing from scratch)
vault.add_document(doc2)
vault.add_documents([doc3, doc4, doc5])

print(f"{len(vault)} chunks indexed")
```

---

## Configuration

```python
vault = RagVault(
    embedding_model="BAAI/bge-large-en-v1.5",   # swap for bge-m3 for multilingual
    reranker_model="BAAI/bge-reranker-large",
    chunk_threshold=0.75,      # lower = fewer, larger chunks
    retrieval_candidates=20,   # candidates passed to cross-encoder
    rerank_top_n=5,            # chunks kept after reranking
    compression_rate=0.5,      # 0.5 = keep 50% of tokens
    use_compression=True,      # set False to skip LLMLingua
)
```

---

## Running the demo

```bash
git clone <repo>
cd ragvault
pip install -e .

# Optional: set your API key for Claude + RAGAS
export ANTHROPIC_API_KEY=sk-ant-...

python example.py
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 26 tests run with lightweight NumPy mocks — no GPU or model download needed.

```
26 passed in 2.46s
```

---

## Architecture

```
Document(s)
    │
    ▼
SemanticChunker          ← topic-shift detection via cosine similarity
    │  chunks
    ▼
BGEEmbedder              ← BAAI/bge-large-en-v1.5, normalised L2 vectors
    │  embeddings
    ▼
HybridRetriever
  ├── FaissVectorStore   ← dense inner-product search
  ├── BM25Retriever      ← sparse keyword scoring
  └── RRF Fusion         ← score = Σ 1/(60 + rank)
    │  top-20 candidates
    ▼
CrossEncoderReranker     ← BAAI/bge-reranker-large, joint (q, doc) scoring
    │  top-5 chunks
    ▼
ContextCompressor        ← LLMLingua-2, ~50% token reduction
    │  compressed context
    ▼
LLM (Claude)             ← anthropic SDK, grounded answer generation
    │  answer
    ▼
RAGASEvaluator           ← faithfulness · relevancy · precision · recall
```

---

## License

MIT
