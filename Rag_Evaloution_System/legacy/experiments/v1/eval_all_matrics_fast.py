import json
import os
import sys
from pathlib import Path

# DeepEval timeout configurations
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1200")
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "1")

# Add project root to sys.path so 'Lllm' can be imported from nested folders
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from Lllm.groq_judge import ollama_judge


def normalize_context(context) -> list[str]:
    if isinstance(context, str):
        return [context] if context.strip() else []
    if isinstance(context, list):
        return [str(item) for item in context if str(item).strip()]
    return [str(context)] if context else []


# Look for results.jsonl in v1 folder, fall back to root
input_path = Path(__file__).parent / "results.jsonl"
if not input_path.exists():
    input_path = ROOT_DIR / "results.jsonl"

output_path = Path(__file__).parent / "evaluated_results1.jsonl"

with open(input_path, "r", encoding="utf-8") as file:
    results = [json.loads(line) for line in file if line.strip()]

test_cases = [
    LLMTestCase(
        input=res["question"],
        actual_output=res["actual_answer"],
        expected_output=res["expected_answer"],
        retrieval_context=normalize_context(res.get("context")),
    )
    for res in results
]

metrics = [
    AnswerRelevancyMetric(threshold=0.7, model=ollama_judge, include_reason=False),
    FaithfulnessMetric(threshold=0.7, model=ollama_judge, include_reason=False),
    ContextualRelevancyMetric(threshold=0.7, model=ollama_judge, include_reason=False),
    GEval(
        name="Correctness",
        criteria=(
            "Determine whether the actual output correctly and directly answers "
            "the input and is factually consistent with the expected output. "
            "Do not penalize additional correct, relevant details. Penalize "
            "contradictions, factual errors, or answers that fail to address the input."
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

# Run sequentially to stay safely below Groq's 8,000 TPM free tier ceiling
evaluation_results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    cache_config=CacheConfig(use_cache=True, write_cache=True),
    async_config=AsyncConfig(run_async=False),
    error_config=ErrorConfig(ignore_errors=False),
)

if len(evaluation_results.test_results) != len(results):
    raise RuntimeError(
        f"Incomplete evaluation: expected {len(results)} results, "
        f"received {len(evaluation_results.test_results)}."
    )

evaluated_results = []
for original_result, test_result in zip(results, evaluation_results.test_results):
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

    evaluated_results.append({**original_result, **scores})

with open(output_path, "w", encoding="utf-8") as file:
    for result in evaluated_results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"\nDone. Saved {len(evaluated_results)} results to {output_path.name}")