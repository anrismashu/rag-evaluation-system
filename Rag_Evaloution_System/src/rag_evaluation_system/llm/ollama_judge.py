from typing import Optional, Type

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from pydantic import BaseModel


class LocalOllamaEvaluator(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name

        self.model = ChatOllama(
            model=model_name,
            temperature=0,
            num_ctx=8192,
            num_predict=1024,
            keep_alive="30m",
        )

    def load_model(self):
        return self.model

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ):
        if schema is not None:
            return self.model.with_structured_output(schema).invoke(prompt)

        return self.model.invoke(prompt).content

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ):
        if schema is not None:
            return await self.model.with_structured_output(schema).ainvoke(prompt)

        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return f"ollama/{self.model_name}"


ollama_judge = LocalOllamaEvaluator("llama3.1")