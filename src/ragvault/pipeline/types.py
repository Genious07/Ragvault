from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Citation:
    chunk: str
    score: float
    rank: int
    metadata: dict = field(default_factory=dict)


@dataclass
class AnswerWithCitations:
    answer: str
    citations: list[Citation]
    query_mode: str
    pipeline_ms: dict[str, float]


@dataclass
class PipelineTrace:
    query: str
    query_mode: str
    n_candidates: int
    n_reranked: int
    tokens_before_compression: int | None
    tokens_after_compression: int | None
    timing_ms: dict[str, float]
    top_scores: list[float]
