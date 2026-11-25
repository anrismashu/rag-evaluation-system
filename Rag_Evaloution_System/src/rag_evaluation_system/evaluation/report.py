import json
from statistics import mean
from pathlib import Path

# 1. Replaced the hardcoded C:\ drive path with a dynamic Path object.
# This ensures it automatically looks for the v2 results in the same folder as the script.
input_path = Path(__file__).parent / "evaluated_results3.jsonl"

with open(input_path, "r", encoding="utf-8") as file:
    results = [json.loads(line) for line in file if line.strip()]
    
metrics = [
    "answer_relevancy",
    "faithfulness",  
    "context_relevancy",
    "correctness"   
]

threshold = 0.7
print(f"\n====================== V2 RAG Evaluation Summary ======================\n")
print(f"Analyzing {len(results)} test cases from {input_path.name}\n")

for metric in metrics:
    scores = [result[metric] for result in results if result.get(metric) is not None]
    
    if not scores:
        print(f"{metric.replace('_', ' ').title()}: No data found.")
        print("-" * 64)
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