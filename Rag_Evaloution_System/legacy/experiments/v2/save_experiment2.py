import json
from statistics import mean 
from pathlib import Path

# 1. Replaced the hardcoded C:\ drive path with a dynamic Path object targeting v2
current_dir = Path(__file__).parent
input_path = current_dir / "evaluated_results2.jsonl"
output_path = current_dir / "experiments.jsonl"

with open(input_path, "r", encoding="utf-8") as file:
    results = [json.loads(line) for line in file if line.strip()]
    
experiment = {
    # 2. Updated the experiment name to reflect v2
    "experiment": "experiment_v2_k1",
    "retriever_k": 1,
    "number_of_test_cases": len(results),
    "metrics": {
        "answer_relevancy": mean(r["answer_relevancy"] for r in results if r.get("answer_relevancy") is not None),
        "faithfulness": mean(r["faithfulness"] for r in results if r.get("faithfulness") is not None),  
        "context_relevancy": mean(r["context_relevancy"] for r in results if r.get("context_relevancy") is not None),
        "correctness": mean(r["correctness"] for r in results if r.get("correctness") is not None)  
    }
}

# 3. Save to the experiments.jsonl file inside the v2 folder
with open(output_path, "a", encoding="utf-8") as file:
    file.write(json.dumps(experiment) + "\n")
    
print(f"Experiment saved as a new line in {output_path.name}")