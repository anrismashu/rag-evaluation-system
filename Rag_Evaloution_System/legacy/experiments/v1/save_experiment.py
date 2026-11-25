import json
from pathlib import Path
from statistics import mean

current_dir = Path(__file__).parent
ROOT_DIR = Path(__file__).resolve().parents[2]

input_path = current_dir / "evaluated_results1.jsonl"
# Stores all run comparisons in the central project root
output_path = ROOT_DIR / "experiments.jsonl"

with open(input_path, "r", encoding="utf-8") as file:
    results = [json.loads(line) for line in file if line.strip()]

experiment = {
    "experiment": "experiment_v1_k1",
    "retriever_k": 1,
    "number_of_test_cases": len(results),
    "metrics": {
        "answer_relevancy": mean(r["answer_relevancy"] for r in results if r.get("answer_relevancy") is not None),
        "faithfulness": mean(r["faithfulness"] for r in results if r.get("faithfulness") is not None),
        "context_relevancy": mean(r["context_relevancy"] for r in results if r.get("context_relevancy") is not None),
        "correctness": mean(r["correctness"] for r in results if r.get("correctness") is not None),
    },
}

with open(output_path, "a", encoding="utf-8") as file:
    file.write(json.dumps(experiment, ensure_ascii=False) + "\n")

print(f"Experiment logged to {output_path.name} as 'experiment_v1_k1'")