import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

def get_llm(provider="gemini"):
    """
    Factory function to easily switch between LLM providers for the RAG agent.
    """
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", 
            temperature=0
        )
        
    elif provider == "groq":
        return ChatGroq(
            # Using the 120B model as it handles tool-calling and Pydantic schemas exceptionally well
            model="openai/gpt-oss-120b", 
            temperature=0
        )
        
    else:
        raise ValueError(f"Unknown provider requested: {provider}")