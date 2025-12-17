from fastapi import FastAPI
from app.core.initialization import lifespan
from app.api.v1.endpoints import router as api_v1_router
from app.core.config import settings

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
    )

    app.include_router(api_v1_router, prefix="/api/v1", tags=["RAG"])
    return app

app = create_app()
