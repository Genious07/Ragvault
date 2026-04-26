from __future__ import annotations

import numpy as np


class BGEEmbedder:
    """
    Dense embeddings using BAAI BGE models via FlagEmbedding.

    Uses separate encode paths for documents vs queries so the model
    applies the correct instruction prefix automatically.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", use_fp16: bool = True):
        from FlagEmbedding import FlagModel

        self.model_name = model_name
        self.model = FlagModel(
            model_name,
            use_fp16=use_fp16,
            query_instruction_for_retrieval=(
                "Represent this sentence for searching relevant passages:"
            ),
        )

    @property
    def dimension(self) -> int:
        return self.model.encode(["probe"], normalize_embeddings=True).shape[1]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of passage/document strings."""
        return self.model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string (applies query instruction prefix)."""
        return self.model.encode_queries([query], normalize_embeddings=True)[0]
