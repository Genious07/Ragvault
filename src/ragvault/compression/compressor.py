from __future__ import annotations


class ContextCompressor:
    """
    Compresses retrieved context using LLMLingua-2 to reduce LLM input tokens.

    Removes redundant / low-information tokens while preserving key facts,
    typically achieving 50% compression with minimal quality loss.
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2: bool = True,
        device_map: str = "cpu",
    ):
        from llmlingua import PromptCompressor

        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=use_llmlingua2,
            device_map=device_map,
        )

    def compress(
        self,
        context: str,
        rate: float = 0.5,
        force_tokens: list[str] | None = None,
        drop_consecutive: bool = True,
    ) -> str:
        """
        Compress context to approximately `rate` of original token count.

        Args:
            context: The retrieved context string to compress.
            rate: Target compression ratio (0.5 = keep ~50% of tokens).
            force_tokens: Tokens that must never be removed.
            drop_consecutive: Remove runs of repeated whitespace/newlines.
        """
        result = self.compressor.compress_prompt(
            context,
            rate=rate,
            force_tokens=force_tokens or [],
            drop_consecutive=drop_consecutive,
        )
        return result["compressed_prompt"]

    def compress_with_stats(self, context: str, rate: float = 0.5) -> dict:
        """Return compressed context plus token count statistics."""
        result = self.compressor.compress_prompt(context, rate=rate)
        return {
            "compressed_prompt": result["compressed_prompt"],
            "origin_tokens": result.get("origin_tokens", None),
            "compressed_tokens": result.get("compressed_tokens", None),
            "ratio": result.get("ratio", None),
        }
