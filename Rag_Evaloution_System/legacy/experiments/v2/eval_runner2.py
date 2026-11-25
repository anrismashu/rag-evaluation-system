import json
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag1.agent import ask_question

# Point directly to the root eval_dataset.jsonl
dataset_path = ROOT_DIR / "eval_dataset.jsonl"
output_path = Path(__file__).parent / "results_v2.jsonl"

with open(dataset_path, "r", encoding="utf-8") as file:
    test_cases = [json.loads(line) for line in file if line.strip()]

results = []

for test_case in test_cases:
    question = test_case["question"]
    expected_answer = test_case["expected_answer"]

    print(f"\nQuestion: {question}")
    rag_result = ask_question(question)

    result = {
        "question": question,
        "expected_answer": expected_answer,
        "actual_answer": rag_result["answer"],
        "context": rag_result["context"],
    }
    results.append(result)
    print(f"Answer: {rag_result['answer']}")

with open(output_path, "w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"\nDone. Results saved to {output_path.name}")