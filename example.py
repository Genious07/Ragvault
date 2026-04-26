"""
ragvault — end-to-end demo

Pipeline:
  document → semantic chunks → BGE embeddings
  → hybrid retrieval (FAISS + BM25 + RRF)
  → cross-encoder reranking
  → context compression (LLMLingua-2)
  → Claude answer generation
  → RAGAS evaluation

Run:
    pip install -e .
    python example.py
"""
import os
from ragvault import RagVault, RAGASEvaluator

# ---------------------------------------------------------------------------
# Sample document
# ---------------------------------------------------------------------------
DOCUMENT = """
Retrieval-Augmented Generation (RAG) is an AI architecture that augments large language
models with a dynamic external knowledge base. Instead of relying solely on parametric
knowledge encoded during pre-training, RAG retrieves relevant documents at inference time
and provides them as context to the model. This dramatically reduces hallucinations and
allows the model to answer questions about up-to-date or proprietary information.

Semantic chunking is a document splitting strategy that preserves meaning across chunk
boundaries. Traditional fixed-size chunking often cuts sentences mid-thought, degrading
retrieval quality. Semantic chunking embeds each sentence and measures cosine similarity
between adjacent sentences. When similarity drops below a threshold, a new chunk begins,
ensuring each chunk covers a single coherent topic.

BGE (BAAI General Embeddings) are dense embedding models developed by the Beijing Academy
of Artificial Intelligence. The bge-large-en-v1.5 model consistently ranks at the top of
the MTEB (Massive Text Embedding Benchmark) leaderboard. BGE models use separate encoding
paths for queries and passages, applying a task-specific instruction prefix to query
embeddings to improve retrieval alignment.

Hybrid retrieval combines dense vector search with sparse keyword matching. Dense retrieval
via FAISS captures semantic similarity even when exact terms differ, while BM25 excels at
precise keyword matching. Reciprocal Rank Fusion (RRF) merges the two ranked lists without
requiring score normalisation — each document receives a score of 1/(k + rank), where
k=60 is the smoothing constant. The final ranking is determined by summing RRF scores
across both systems.

Cross-encoder reranking is a second-stage precision filter. Unlike bi-encoders that produce
independent embeddings for query and document, a cross-encoder processes the query and
document together as a single input. This joint encoding captures fine-grained interactions
between query and document tokens, yielding much more accurate relevance scores. The
trade-off is latency: cross-encoders are too slow to run over an entire corpus but fast
enough to rescore a short candidate list of 20-50 documents.

Context compression with LLMLingua removes redundant or low-information tokens from the
retrieved context before it is sent to the LLM. LLMLingua-2 uses a BERT-based model
trained to classify each token as essential or droppable. At a 50% compression rate,
roughly half the tokens are removed while preserving the key facts, reducing LLM API
costs and latency without significantly degrading answer quality.

RAGAS (Retrieval-Augmented Generation Assessment) is an evaluation framework specifically
designed for RAG pipelines. It measures four key dimensions: faithfulness (does the answer
contain only information from the context?), answer relevancy (does the answer address
the question?), context precision (are the retrieved chunks actually useful?), and context
recall (were all necessary chunks retrieved?). RAGAS uses an LLM-as-judge approach,
making it reference-free for most metrics.
"""

QA_PAIRS = [
    {
        "question": "What is Reciprocal Rank Fusion and why is k=60 used?",
        "ground_truth": (
            "RRF merges ranked lists by scoring each document as 1/(k + rank). "
            "k=60 is the standard smoothing constant that prevents very high ranks "
            "from dominating the final score."
        ),
    },
    {
        "question": "How does semantic chunking decide where to split?",
        "ground_truth": (
            "It embeds each sentence and measures cosine similarity between adjacent "
            "sentences. A split is made where similarity drops below a threshold."
        ),
    },
    {
        "question": "What are the four RAGAS metrics?",
        "ground_truth": (
            "Faithfulness, answer relevancy, context precision, and context recall."
        ),
    },
]


def main():
    use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))

    print("=" * 60)
    print("ragvault — end-to-end RAG demo")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Build and index
    # -----------------------------------------------------------------------
    print("\n[1/4] Initialising RagVault (loading models)...")
    vault = RagVault(
        chunk_threshold=0.72,
        retrieval_candidates=15,
        rerank_top_n=4,
        compression_rate=0.55,
        use_compression=True,
    )

    print("[1/4] Indexing document...")
    vault.index(DOCUMENT)
    print(f"      → {len(vault)} semantic chunks created")

    # -----------------------------------------------------------------------
    # 2. Query loop
    # -----------------------------------------------------------------------
    print("\n[2/4] Running queries through full pipeline...\n")

    answers = []
    retrieved_contexts = []

    for qa in QA_PAIRS:
        question = qa["question"]
        print(f"  Q: {question}")

        chunks = vault.retrieve(question)
        retrieved_contexts.append(chunks)

        context = "\n\n".join(chunks)
        if vault.use_compression and vault.compressor:
            context = vault.compressor.compress(context, rate=vault.compression_rate)

        if use_llm:
            answer = vault.ask(question)
        else:
            answer = f"[LLM disabled] Context: {context[:200]}..."

        answers.append(answer)
        print(f"  A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print()

    # -----------------------------------------------------------------------
    # 3. RAGAS evaluation
    # -----------------------------------------------------------------------
    if use_llm:
        print("[3/4] Running RAGAS evaluation...")
        questions = [qa["question"] for qa in QA_PAIRS]
        ground_truths = [qa["ground_truth"] for qa in QA_PAIRS]

        try:
            results = vault.evaluate(
                questions=questions,
                answers=answers,
                contexts=retrieved_contexts,
                ground_truths=ground_truths,
            )
            print("\n  RAGAS Results:")
            print(f"  {results}")
        except Exception as e:
            print(f"  RAGAS evaluation requires LLM API access: {e}")
    else:
        print("[3/4] RAGAS evaluation skipped (set ANTHROPIC_API_KEY to enable).")

    # -----------------------------------------------------------------------
    # 4. Multi-document indexing demo
    # -----------------------------------------------------------------------
    print("\n[4/4] Multi-document incremental indexing demo...")
    extra_docs = [
        "Vector databases like Pinecone, Weaviate, and Qdrant store and index embeddings at scale.",
        "Prompt engineering techniques like chain-of-thought improve LLM reasoning capabilities.",
    ]
    before = len(vault)
    vault.add_documents(extra_docs)
    after = len(vault)
    print(f"      → Added {after - before} new chunks (total: {after})")

    print("\nDone.")
    print("=" * 60)


if __name__ == "__main__":
    main()
