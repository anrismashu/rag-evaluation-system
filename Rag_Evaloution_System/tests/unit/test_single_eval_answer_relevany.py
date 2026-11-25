import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

load_dotenv()

# 1. Create the Custom Wrapper for Gemini
class GeminiEvaluator(DeepEvalBaseLLM):
    def __init__(self, model_name="gemini-3.1-flash-lite"):
        # Initialize your LangChain Google GenAI model
        self.model = ChatGoogleGenerativeAI(model=model_name)
        self.model_name = model_name
    def load_model(self):
        return self.model
    def _extract_text(self, content) -> str:
        """Helper to ensure the output is always a string for DeepEval."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # If LangChain returns a list of parts, combine them into one string
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            return "".join(text_parts)
        return str(content)
    def generate(self, prompt: str, **kwargs) -> str:
        response = self.model.invoke(prompt)
        return self._extract_text(response.content)
    async def a_generate(self, prompt: str, **kwargs) -> str:
        response = await self.model.ainvoke(prompt)
        return self._extract_text(response.content)
    def get_model_name(self):
        return self.model_name
# 2. Instantiate the custom Gemini model
gemini_judge = GeminiEvaluator()
# 3. Create your test case
test_case = LLMTestCase(
    input="what do penguins eat",
    actual_output="Penguins primarily feed on marine animals such as krill and tiny fish"
)
# 4. Pass the custom Gemini model to the metric
metric = AnswerRelevancyMetric(threshold=0.7, model=gemini_judge)
metric.measure(test_case=test_case)

#print("Score:", metric.score)
#print("Reason:", metric.reason)
#print("Passed:", metric.is_successful)