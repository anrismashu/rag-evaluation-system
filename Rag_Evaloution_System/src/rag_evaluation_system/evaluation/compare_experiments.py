import json
from pathlib import Path

# 1. Safely locate your master experiments log
log_path = Path("experiments.jsonl")

if not log_path.exists():
    print(f"Error: {log_path.name} not found. Please run an evaluation first.")
    exit()

# 2. Load all saved runs from the JSONL file
with open(log_path, "r", encoding="utf-8") as file:
    runs = [json.loads(line) for line in file if line.strip()]

if len(runs) < 2:
    print("Error: You need at least 2 runs in experiments.jsonl to perform an A/B comparison.")
    exit()

# 3. Assign the first run as Baseline and the last run as the Experiment
baseline = runs[0]
experiment = runs[-1]

# Extract dynamic headers from the files themselves
base_label = f"{baseline.get('experiment', 'Baseline')} (k={baseline.get('retriever_k', '?')})"
exp_label = f"{experiment.get('experiment', 'Experiment')} (k={experiment.get('retriever_k', '?')})"

print("\n" + "=" * 80)
print(f"{'EXPERIMENT A/B DELTA REPORT':^80}")
print("=" * 80 + "\n")

# Adjust column widths for clean formatting
print(f"{'Metric':<24} {base_label:<25} {exp_label:<20} {'Delta':<10}")
print("-" * 80)

# Calculate the Delta for each metric safely
for metric, base_score in baseline.get("metrics", {}).items():
    exp_score = experiment.get("metrics", {}).get(metric)

    if exp_score is None or base_score is None:
        delta_str = "N/A"
        exp_score_str = "N/A"
    else:
        delta = exp_score - base_score
        # Display the change with a + or - sign
        delta_str = f"{delta:+.3f}"
        exp_score_str = f"{exp_score:<20.3f}"

    base_score_str = f"{base_score:<25.3f}" if base_score is not None else "N/A"

    print(
        f"{metric.replace('_', ' ').title():<24} "
        f"{base_score_str} "
        f"{exp_score_str} "
        f"{delta_str}"
    )

print("-" * 80 + "\n")