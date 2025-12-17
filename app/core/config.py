from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

class Settings(BaseSettings):
    app_name: str = "KB RAG Chatbot"

    openai_api_key: str
    embeddings_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4.1-mini"

    data_raw: Path = ROOT / "data" / "raw"
    index_dir: Path = ROOT / "data" / "index"
    index_path: Path = ROOT / "data" / "index"

    langsmith_api_key: str | None = None
    langsmith_tracing: bool = False

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  
    )

settings = Settings()

