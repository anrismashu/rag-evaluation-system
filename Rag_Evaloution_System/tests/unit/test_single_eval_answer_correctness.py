from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from test_single_eval_answer_relevany import gemini_judge  # Importing your custom Gemini wrapper

test_case = LLMTestCase(
    input="where do penguins live?",
    actual_output="Penguins live in homes and building inside shoping mall",
    expected_output="Penguins live in the Southern Hemisphere, including Antarctica, South America, southern Africa, and the Galapagos Islands."
)

metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output",
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT
    ],
    model=gemini_judge  # <-- Add your Gemini judge here
)

metric.measure(test_case=test_case)

#print("Score:", metric.score)
#print("Reason:", metric.reason)