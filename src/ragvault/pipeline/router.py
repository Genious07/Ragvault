from __future__ import annotations

_COMPLEX = {
    "compare", "contrast", "difference", "differences", "between",
    "why", "explain", "analyze", "analyse", "summarize", "summarise",
    "how", "steps", "process", "mechanism", "relationship",
    "pros", "cons", "advantages", "disadvantages", "tradeoff", "tradeoffs",
    "evaluate", "assess", "critique", "impact", "implications",
}

_SIMPLE = {
    "what", "who", "when", "where", "which", "define", "definition",
    "name", "list", "give",
}


class QueryRouter:
    """
    Routes queries to the cheapest pipeline that can answer them correctly.

    fast     — dense-only retrieval, no reranking   (<50 ms extra overhead)
    balanced — hybrid retrieval + reranking, no compression
    quality  — full pipeline: hybrid + reranking + compression
    """

    def route(self, query: str) -> str:
        tokens = set(query.lower().split())
        word_count = len(query.split())
        has_complex = bool(tokens & _COMPLEX)
        has_simple = bool(tokens & _SIMPLE)

        if has_complex or word_count > 15:
            return "quality"
        if has_simple and word_count <= 10:
            return "fast"
        return "balanced"
