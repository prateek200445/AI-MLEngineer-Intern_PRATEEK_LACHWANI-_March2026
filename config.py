from dotenv import load_dotenv
from functools import lru_cache
from typing import List, Optional
import os

from pydantic import BaseModel, validator

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))


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
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_HOST: str = "localhost"                   # (fallback / optional)
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "policy_db"
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
    PDF_FOLDER: str = "policy_docs"

    # =========================
    # LLM
    # =========================
    GOOGLE_API_KEY: Optional[str] = None
    GENERATIVE_MODEL_NAME: str = "gemini-2.5-flash"
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL_NAME: str = "google/gemma-3-27b-it:free"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    # Default provider set to OpenRouter
    LLM_PROVIDER: str = "openrouter"
    # Local Ollama settings
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_CLI: str = "ollama"
    OLLAMA_TIMEOUT: int = 60

    # =========================
    # CONFIDENCE / NO-GUESS POLICY
    # =========================
    ELIGIBILITY_RAG_CONFIDENCE_THRESHOLD: float = 0.45
    POLICY_RAG_CONFIDENCE_THRESHOLD: float = 0.60
    ELIGIBILITY_MIN_TOKEN_OVERLAP: int = 1
    POLICY_MIN_TOKEN_OVERLAP: int = 2

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
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL_NAME",
        "OPENROUTER_BASE_URL",
        "GENERATIVE_MODEL_NAME",
        "OLLAMA_MODEL",
        "OLLAMA_CLI",
        "OLLAMA_TIMEOUT",
        "ELIGIBILITY_RAG_CONFIDENCE_THRESHOLD",
        "POLICY_RAG_CONFIDENCE_THRESHOLD",
        "ELIGIBILITY_MIN_TOKEN_OVERLAP",
        "POLICY_MIN_TOKEN_OVERLAP",
        "LOG_LEVEL",
    ]

    env_values = {}
    for k in keys:
        val = os.getenv(k)
        if val:
            env_values[k] = val

    # Backward compatibility for existing lowercase env var name.
    if "OPENROUTER_API_KEY" not in env_values:
        legacy_openrouter = os.getenv("open_router_api_key")
        if legacy_openrouter:
            env_values["OPENROUTER_API_KEY"] = legacy_openrouter

    return Settings(**env_values)


# Singleton
settings = get_settings()
