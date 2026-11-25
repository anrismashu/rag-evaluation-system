from langchain_core.tools import tool
from rag_evaluation_system.rag.indexing import vector_store
last_retrived_context = []
 
@tool
def collection_info(query:str):
    """Search the pdf knowledge base and return relevent information based on the query """
    global last_retrived_context
    
    docs = vector_store.similarity_search(query=query,k=2)
    
    last_retrived_context = [doc.page_content for doc in docs ]
    
    return "\n\n".join(last_retrived_context)