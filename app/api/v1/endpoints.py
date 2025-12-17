from fastapi import APIRouter
from app.schemas.base import ChatPayload
from app.core.config import settings
from app.services.rag_service import run_rag

router = APIRouter()

@router.post("/chat")
async def chat_api(p: ChatPayload):
    answer, matches = run_rag(
        query=p.query,
        mode=p.mode,
        k=p.k,
        model=settings.chat_model
    )

    return {
        "mode": p.mode,
        "model": settings.chat_model,
        "answer": answer,
        "matches": matches
    }
