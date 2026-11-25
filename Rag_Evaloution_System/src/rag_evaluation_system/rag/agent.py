import sys
import time

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from rag_evaluation_system.llm.models import get_llm
from rag_evaluation_system.rag import tools as tl
from rag_evaluation_system.rag.objects import Answer
from rag_evaluation_system.rag.prompts import system_agent_prompt
from rag_evaluation_system.rag.tools import collection_info

load_dotenv()

tools = [collection_info]

# Change to "groq" when needed.
llm = get_llm(provider="gemini")

agent = create_agent(
    llm,
    tools,
    system_prompt=system_agent_prompt,
    response_format=Answer,
)


def ask_question(question: str) -> dict:
    """Answer a user question using the PDF RAG agent."""

    tl.last_retrived_context = []

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    structured_response = result["structured_response"]

    return {
        "answer": structured_response.answer,
        "context": structured_response.context,
    }


if __name__ == "__main__":
    print("🐧 Welcome to the RAG Chatbot! Type 'exit' or 'quit' to stop.")
    print("-" * 65)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            print("Thinking...")
            response = ask_question(user_input)

            print("\nAgent: ", end="")

            for char in response["answer"]:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.015)

            print("\n")

        except Exception as error:
            print(f"\n[Error]: {error}")