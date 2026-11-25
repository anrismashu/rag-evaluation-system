import json
from deepeval.metrics import (
    AnswerRelevancyMetric, 
    FaithfulnessMetric, 
    ContextualRelevancyMetric, 
    GEval
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from Lllm.groq_judge import ollama_judge as gemini_judge

# 1. Initialize all 4 metrics using your Gemini judge
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=gemini_judge)
faithfulness = FaithfulnessMetric(threshold=0.7, model=gemini_judge)
context_relevancy = ContextualRelevancyMetric(threshold=0.7, model=gemini_judge)
correctness = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    model=gemini_judge
)

evaluated_results = []

print("Starting Full Batch Evaluation...\n")

with open("results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue          
        data = json.loads(line)       
        # 3. Create a comprehensive test case
        test_case = LLMTestCase(
            input=data["question"],
            actual_output=data["actual_answer"],
            expected_output=data["expected_answer"],
            retrieval_context=[data["context"]]
        )      
        print(f"Question: {data['question']}")     
        # 4. Measure the test case against all 4 metrics[cite: 1, 3, 9]
        answer_relevancy.measure(test_case=test_case)
        faithfulness.measure(test_case=test_case)
        context_relevancy.measure(test_case=test_case)
        correctness.measure(test_case=test_case)    
        # 5. Print the scores matching your screenshot format
        print("Answer Relevancy: ", answer_relevancy.score)
        print("faithfulness: ", faithfulness.score)
        print("context_relvancy: ", context_relevancy.score)
        print("correctness: ", correctness.score)
        print("-" * 50)    
        # 6. Unpack the original data using ** and append the new scores
        evaluated_result = {
            **data,
            "answer_relevancy": answer_relevancy.score,
            "faithfulness": faithfulness.score,
            "context_relevancy": context_relevancy.score,
            "correctness": correctness.score
        }   
        evaluated_results.append(evaluated_result)
# 7. Save to the final output file
with open("evaluated_results1.jsonl", "w", encoding="utf-8") as file:
    for result in evaluated_results:
        file.write(json.dumps(result) + "\n")
print("\nDone! Results saved to evaluated_results1.jsonl")