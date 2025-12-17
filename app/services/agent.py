from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

from app.core.config import settings
from .vector_db import get_retriever
from .rag_chain import format_docs


@tool("retrieve_context", response_format="content_and_artifact")
def retrieve_context(query: str, k: int = 3):
    """Retrieve relevant internal documents for a natural language query.

    Args:
        query: The user's question.
        k: Number of documents to retrieve.
    Returns:
        A tuple of (formatted_context_string, raw_documents_list).
    """
    docs = get_retriever(k=k).invoke(query)
    return format_docs(docs), docs

def build_agent(model_name: str):
    llm = ChatOpenAI(model=model_name, temperature=0.1)
    tools = [retrieve_context]

    system_prompt = (
        "You may call tools to retrieve internal context. "
        "Useful for questions about policies and internal documents."
    )

    return create_agent(llm, tools, system_prompt=system_prompt)
