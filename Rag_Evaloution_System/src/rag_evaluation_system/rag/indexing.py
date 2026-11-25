from dotenv import load_dotenv
from rag_evaluation_system.rag.file_reader import extract_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)
vector_store = InMemoryVectorStore(embeddings)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data" / "source_documents" / "Penguins_ACL.pdf"

chunks = extract_paragraphs(str(PDF_PATH))
vector_store.add_texts(texts=chunks)


