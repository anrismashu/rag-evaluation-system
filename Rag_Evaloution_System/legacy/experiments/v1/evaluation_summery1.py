import json
from pathlib import Path
from statistics import mean

input_path = Path(__file__).parent / "evaluated_results1.jsonl"

with open(input_path, "r", encoding="utf-8") as file:
    results = [json.loads(line) for line in file if line.strip()]

metrics = [
    "answer_relevancy",
    "faithfulness",
    "context_relevancy",
    "correctness",
]

threshold = 0.7
print("\n======================== RAG V1 (k=1) Evaluation Summary ========================\n")

for metric in metrics:
    scores = [result[metric] for result in results if result.get(metric) is not None]

    if not scores:
        continue

    average_score = mean(scores)
    passed = sum(score >= threshold for score in scores)
    failed = len(scores) - passed
    pass_rate = (passed / len(scores)) * 100

    print(metric.replace("_", " ").title())
    print(f"  Average Score : {average_score:.2f}")
    print(f"  Pass Rate     : {pass_rate:.1f}%")
    print(f"  Passed        : {passed}")
    print(f"  Failed        : {failed}")
    print("-" * 64)