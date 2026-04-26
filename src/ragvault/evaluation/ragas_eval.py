from __future__ import annotations

from typing import Any


class RAGASEvaluator:
    """
    Evaluates RAG pipeline quality using RAGAS metrics.

    Supports RAGAS >= 0.2.x schema:
      - user_input     (was: question)
      - response       (was: answer)
      - retrieved_contexts  (was: contexts, now list[str] per sample)
      - reference      (was: ground_truth)

    Metrics reported by default:
      faithfulness, answer_relevancy, context_precision, context_recall
    """

    def _get_default_metrics(self) -> list:
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )
        return [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
        metrics: list | None = None,
        llm=None,
        embeddings=None,
    ) -> Any:
        """
        Run RAGAS evaluation.

        Args:
            questions:     List of user questions.
            answers:       List of LLM-generated answers.
            contexts:      List of retrieved context lists (one list per question).
            ground_truths: List of reference / ground-truth answers.
            metrics:       Override default metrics list.
            llm:           Optional custom LangChain LLM for RAGAS judge calls.
            embeddings:    Optional custom embeddings for RAGAS.
        """
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

        samples = [
            SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=ctx,
                reference=gt,
            )
            for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths)
        ]
        dataset = EvaluationDataset(samples=samples)
        chosen_metrics = metrics or self._get_default_metrics()

        kwargs: dict[str, Any] = {"dataset": dataset, "metrics": chosen_metrics}
        if llm is not None:
            kwargs["llm"] = llm
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        return evaluate(**kwargs)

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
        metrics: list | None = None,
    ) -> Any:
        """Convenience wrapper for evaluating a single QA sample."""
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth],
            metrics=metrics,
        )
