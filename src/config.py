# src/config.py
from pydantic_settings import BaseSettings
import os

class AppSettings(BaseSettings):
    OPENAI_API_KEY: str
    MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    # Use /tmp on Vercel (ephemeral but writable); falls back to local for dev
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "/tmp/vector_store")
    REPO_PATH: str = os.getenv("REPO_PATH", "/tmp/repo")

    class Config:
        env_file = ".env"

settings = AppSettings()  # Raises error if required vars are missing!
