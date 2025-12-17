from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.core.config import settings
from app.core.logger import logger

_VECTORSTORE = None

def load_vectorstore():
    global _VECTORSTORE

    if _VECTORSTORE is not None:
        return _VECTORSTORE

    logger.info(f"Loading FAISS index → {settings.index_path}")

    embeddings = OpenAIEmbeddings(model=settings.embeddings_model)
    _VECTORSTORE = FAISS.load_local(
        str(settings.index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    return _VECTORSTORE


def get_retriever(k: int = 4):
    vs = load_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})


def similarity_with_scores(query: str, k: int):
    vs = load_vectorstore()
    return vs.similarity_search_with_score(query, k=k)
