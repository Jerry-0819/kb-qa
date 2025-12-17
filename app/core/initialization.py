from fastapi import FastAPI
from contextlib import asynccontextmanager

from .logger import logger
from app.services.vector_db import load_vectorstore

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting…")

    try:
        load_vectorstore()
        logger.info("Vector store loaded on startup")
    except Exception as e:
        logger.exception(f"Error loading vector store: {e}")

    yield

    logger.info("Application shutting down…")
