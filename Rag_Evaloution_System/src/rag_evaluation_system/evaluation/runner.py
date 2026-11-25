import argparse
import json
import time
from pathlib import Path

from rag_evaluation_system.rag.agent import ask_question

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 8


def load_completed_results(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []

    with open(output_path, "r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    parser = argparse.ArgumentParser(
        description="Run the RAG agent against an evaluation dataset."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the JSONL evaluation dataset.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save raw RAG answers as JSONL.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, "r", encoding="utf-8") as file:
        test_cases = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    completed_results = load_completed_results(output_path)

    completed_questions = {
        result["question"]
        for result in completed_results
        if result.get("actual_answer")
    }

    print(
        f"Found {len(completed_questions)} completed test cases. "
        "Previously completed questions will be skipped."
    )

    with open(output_path, "a", encoding="utf-8") as file:
        for index, test_case in enumerate(test_cases, start=1):
            question = test_case["question"]

            if question in completed_questions:
                print(f"[{index}/{len(test_cases)}] Skipping completed question.")
                continue

            expected_answer = test_case["expected_answer"]

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"\n[{index}/{len(test_cases)}] Question: {question}")
                    print(f"Attempt {attempt}/{MAX_RETRIES}")

                    started_at = time.perf_counter()
                    rag_result = ask_question(question)
                    duration = round(
                        time.perf_counter() - started_at,
                        2,
                    )

                    result = {
                        "name": test_case.get(
                            "name",
                            f"test_case_{index}",
                        ),
                        "question": question,
                        "expected_answer": expected_answer,
                        "actual_answer": rag_result["answer"],
                        "context": rag_result["context"],
                        "run_duration": duration,
                    }

                    file.write(
                        json.dumps(result, ensure_ascii=False) + "\n"
                    )
                    file.flush()

                    completed_questions.add(question)

                    print(f"Answer: {rag_result['answer']}")
                    print(f"Latency: {duration}s")
                    break

                except Exception as error:
                    if attempt == MAX_RETRIES:
                        print(
                            "\nRunner stopped after retries. "
                            "Completed results were saved safely."
                        )
                        raise error

                    wait_time = RETRY_DELAY_SECONDS * attempt

                    print(
                        f"\nTemporary request error: {error}"
                    )
                    print(
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

    print(
        f"\nDone. Saved {len(completed_questions)} "
        f"raw results to: {output_path}"
    )


if __name__ == "__main__":
    main()