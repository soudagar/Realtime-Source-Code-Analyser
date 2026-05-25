# src/config.py
from pydantic_settings import BaseSettings
import os

class AppSettings(BaseSettings):
    GROK_API_KEY: str
    MODEL_NAME: str = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Use /tmp on Vercel (ephemeral but writable); falls back to local for dev
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "/tmp/vector_store")
    REPO_PATH: str = os.getenv("REPO_PATH", "/tmp/repo")

    class Config:
        env_file = ".env"

settings = AppSettings()  # Raises error if required vars are missing!
