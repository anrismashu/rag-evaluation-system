from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from test_single_eval_answer_relevany import gemini_judge  # Importing your custom Gemini wrapper

# 1. Create the test case
test_case = LLMTestCase(
    input="where do penguins live?",
    actual_output="Penguins live in homes and building inside shoping mall",
    retrieval_context=[
        "Penguins are flightless seabirds that are specially adapted for life in the Southern Hemisphere. "
        "Although people often associate them only with Antarctica, penguins live in a wide range of environments, "
        "including the icy coasts of Antarctica, the cool shores of South America and southern Africa, "
        "and even the Galapagos Islands near the equator."
    ]
)

# 2. Initialize the metric with your Gemini Judge
metric = ContextualRelevancyMetric(
    threshold=0.7,
    model=gemini_judge
)

# 3. Measure the score
metric.measure(test_case=test_case)

# 4. Print the results (Remembering the () for is_successful)
print("Score:", metric.score)
print("Reason:", metric.reason)
print("Passed:", metric.is_successful())