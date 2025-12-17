from langchain_core.messages import AIMessage
from app.services.rag_chain import build_rag_chain
from app.services.agent import build_agent
from app.services.vector_db import similarity_with_scores
from pathlib import Path


def format_matches(query: str, k: int):
    hits = similarity_with_scores(query, k)
    matches = []

    for doc, score in hits:
        meta = doc.metadata or {}
        matches.append({
            "meta": {
                "file": Path(meta.get("source", "unknown")).name,
                "page": meta.get("page", "N/A")
            },
            "score": float(score),
            "snippet": doc.page_content[:500]
        })
    return matches


def run_rag(query: str, mode: str, k: int, model: str):
    if mode == "chain":
        chain = build_rag_chain(model, k)
        answer: AIMessage = chain.invoke(query)
        return answer.content, format_matches(query, k)

    agent = build_agent(model)
    result = None
    for event in agent.stream({"messages": [{"role": "user", "content": query}]},stream_mode="values"):
        result = event

    final_msg = result["messages"][-1]
    return final_msg.content, format_matches(query, k)
