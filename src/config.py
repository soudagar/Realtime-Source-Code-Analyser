# src/config.py
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    OPENAI_API_KEY: str
    MODEL_NAME: str = "llama3.2:latest"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_STORE_PATH: str = "./vector_store"
    REPO_PATH: str = "repo"

    class Config:
        env_file = ".env"

settings = AppSettings()  # Raises error if required vars are missing!
