from typing import Optional, Type

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from pydantic import BaseModel


class GroqEvaluator(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        self.model = ChatGroq(
            model=model_name,
            temperature=0,
            max_tokens=1024,
            timeout=120,
            max_retries=3,
        )

    def load_model(self):
        return self.model

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ):
        if schema is not None:
            # Added method="json_mode" to stop the Groq tool hallucination
            return self.model.with_structured_output(schema, method="json_mode").invoke(prompt)

        return self.model.invoke(prompt).content

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ):
        if schema is not None:
            # Added method="json_mode" to stop the Groq tool hallucination
            return await self.model.with_structured_output(schema, method="json_mode").ainvoke(prompt)

        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return f"groq/{self.model_name}"


# Kept under the old name so eval_all_matrics_fast.py needs no import changes.
ollama_judge = GroqEvaluator()