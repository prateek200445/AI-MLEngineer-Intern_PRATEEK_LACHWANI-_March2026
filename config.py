from dotenv import load_dotenv
from functools import lru_cache
from typing import List, Optional
import os

from pydantic import BaseModel, validator

load_dotenv()


class Settings(BaseModel):
    APP_NAME: str = "RAG Search API"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # =========================
    # QDRANT (AZURE VM)
    # =========================
    QDRANT_URL: str = "http://20.187.144.184:6333"   # ✅ FIXED PUBLIC URL
    QDRANT_HOST: str = "20.187.144.184"              # (fallback / optional)
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "pdf_db"
    QDRANT_API_KEY: Optional[str] = None
    PREFER_GRPC: bool = False                        # ✅ MUST be False for public IP

    # =========================
    # EMBEDDINGS
    # =========================
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"    # ✅ Stable & lighter
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_NORMALIZE: bool = True

    # =========================
    # INGESTION
    # =========================
    PDF_FOLDER: str = "data_pdfs"

    # =========================
    # LLM
    # =========================
    GOOGLE_API_KEY: Optional[str] = None
    GENERATIVE_MODEL_NAME: str = "gemini-2.5-flash"
    LLM_PROVIDER: str = "google"

    LOG_LEVEL: str = "info"

    @validator("CORS_ORIGINS", pre=True)
    def _assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]


@lru_cache()
def get_settings() -> Settings:
    """
    Environment variables override defaults,
    but defaults already point to Azure Qdrant.
    """
    keys = [
        "APP_NAME",
        "DEBUG",
        "HOST",
        "PORT",
        "CORS_ORIGINS",
        "QDRANT_URL",
        "QDRANT_HOST",
        "QDRANT_PORT",
        "QDRANT_COLLECTION",
        "QDRANT_API_KEY",
        "PREFER_GRPC",
        "EMBEDDING_MODEL",
        "EMBEDDING_DEVICE",
        "EMBEDDING_NORMALIZE",
        "PDF_FOLDER",
        "GOOGLE_API_KEY",
        "GENERATIVE_MODEL_NAME",
        "LOG_LEVEL",
    ]

    env_values = {}
    for k in keys:
        val = os.getenv(k)
        if val:
            env_values[k] = val

    return Settings(**env_values)


# Singleton
settings = get_settings()
