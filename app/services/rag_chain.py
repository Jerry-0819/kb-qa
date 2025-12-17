from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path

from app.core.config import settings
from .vector_db import get_retriever


SYSTEM_TEMPLATE = """
You are a helpful internal assistant.
<context>
{context}
</context>
"""
USER_TEMPLATE = "Question: {question}"

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("user", USER_TEMPLATE),
])

def format_docs(docs):
    parts = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page")
        page_str = f"(page {page})" if page else ""
        parts.append(f"Source: {Path(src).name} {page_str}\n{d.page_content}")
    return "\n\n".join(parts)

def build_rag_chain(model_name: str, k: int):
    llm = ChatOpenAI(model=model_name, temperature=0.2)
    retriever = get_retriever(k=k)

    def retrieve(q: str):
        return retriever.invoke(q)

    def build_prompt(inputs):
        ctx = format_docs(inputs["docs"])
        return {"context": ctx, "question": inputs["question"]}

    chain = (
        {
            "question": RunnableLambda(lambda x: x),
            "docs": RunnableLambda(retrieve)
        }
        | RunnableLambda(build_prompt)
        | prompt
        | llm
    )

    return chain
