from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams
from test_single_eval_answer_relevany import gemini_judge  # Importing your custom Gemini wrapper

test_case = LLMTestCase(
    input="where do penguins live?",
    actual_output="Penguins live in homes and building inside shoping mall",
   # expected_output="Penguins live in the Southern Hemisphere, including Antarctica, South America, southern Africa, and the Galapagos Islands.",
    retrieval_context=["Penguins are flightless seabirds that are specially adapted for life in the Southern Hemisphere. Although people often associate them only with Antarctica, penguins live in a wide range of environments, including the icy coasts of Antarctica, the cool shores of South America and southern Africa, and even the Galapagos Islands near the equator."]
)

metric = FaithfulnessMetric(
    threshold=0.7,
    model=gemini_judge
)

metric.measure(test_case=test_case)

print("Score:", metric.score)
print("Reason:", metric.reason)
print("Passed:", metric.is_successful)