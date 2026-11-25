import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
    "600",
)
os.environ.setdefault(
    "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE",
    "1200",
)
os.environ.setdefault(
    "DEEPEVAL_RETRY_MAX_ATTEMPTS",
    "1",
)

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from rag_evaluation_system.llm.groq_judge import ollama_judge


def normalize_context(context) -> list[str]:
    if isinstance(context, str):
        return [context] if context.strip() else []

    if isinstance(context, list):
        return [
            str(item)
            for item in context
            if str(item).strip()
        ]

    return [str(context)] if context else []


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate raw RAG results and save scored JSONL output."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw RAG results JSONL.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save evaluated JSONL results.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as file:
        results = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    test_cases = [
        LLMTestCase(
            input=result["question"],
            actual_output=result["actual_answer"],
            expected_output=result["expected_answer"],
            retrieval_context=normalize_context(
                result.get("context")
            ),
        )
        for result in results
    ]

    metrics = [
        AnswerRelevancyMetric(
            threshold=0.7,
            model=ollama_judge,
            include_reason=False,
        ),
        FaithfulnessMetric(
            threshold=0.7,
            model=ollama_judge,
            include_reason=False,
        ),
        ContextualRelevancyMetric(
            threshold=0.7,
            model=ollama_judge,
            include_reason=False,
        ),
        GEval(
            name="Correctness",
            criteria=(
                "Determine whether the actual output correctly and "
                "directly answers the input and is factually consistent "
                "with the expected output. Do not penalize additional "
                "correct and relevant details. Penalize contradictions, "
                "factual errors, or answers that fail to address the input."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,
            model=ollama_judge,
        ),
    ]

    evaluation_results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        cache_config=CacheConfig(
            use_cache=True,
            write_cache=True,
        ),
        async_config=AsyncConfig(run_async=False),
        error_config=ErrorConfig(ignore_errors=False),
    )

    if len(evaluation_results.test_results) != len(results):
        raise RuntimeError(
            f"Expected {len(results)} results but received "
            f"{len(evaluation_results.test_results)}."
        )

    evaluated_results = []

    for original, test_result in zip(
        results,
        evaluation_results.test_results,
    ):
        scores = {
            "answer_relevancy": None,
            "faithfulness": None,
            "context_relevancy": None,
            "correctness": None,
        }

        for metric in test_result.metrics_data or []:
            metric_name = metric.name.strip().lower()

            if metric_name == "answer relevancy":
                scores["answer_relevancy"] = metric.score

            elif metric_name == "faithfulness":
                scores["faithfulness"] = metric.score

            elif metric_name == "contextual relevancy":
                scores["context_relevancy"] = metric.score

            elif metric_name.startswith("correctness"):
                scores["correctness"] = metric.score

        valid_scores = [
            score
            for score in scores.values()
            if score is not None
        ]

        success = (
            len(valid_scores) == 4
            and all(score >= 0.7 for score in valid_scores)
        )

        evaluated_results.append(
            {
                **original,
                **scores,
                "success": success,
            }
        )

    with open(output_path, "w", encoding="utf-8") as file:
        for result in evaluated_results:
            file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    print(
        f"Done. Saved {len(evaluated_results)} evaluated results "
        f"to: {output_path}"
    )


if __name__ == "__main__":
    main()