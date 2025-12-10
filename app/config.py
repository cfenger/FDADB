from functools import lru_cache
from pathlib import Path
from typing import List, Set

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    openai_api_key: str = Field(..., description="OpenAI API key")
    qa_model: str = Field("gpt-5.1", alias="FDA_QA_MODEL")

    app_host: str = Field("0.0.0.0", alias="APP_HOST")
    app_port: int = Field(8000, alias="APP_PORT")
    allowed_origins: str = Field("*", alias="ALLOWED_ORIGINS")

    max_file_size_mb: int = Field(10, alias="MAX_FILE_SIZE_MB")
    allowed_file_types: str = Field(".txt,.md,.pdf,.docx", alias="ALLOWED_FILE_TYPES")
    max_files_per_request: int = Field(10, alias="MAX_FILES_PER_REQUEST")

    metadata_path: Path = Field(
        Path("data/metadata.json"), description="Path to persisted document metadata"
    )
    uploads_dir: Path = Field(Path("data/uploads"), description="Local scratch area")
    vector_store_name: str = Field(
        "FDA-RAG-Vector-Store", description="Name for the shared vector store"
    )
    file_index_timeout_seconds: int = Field(
        120, alias="FILE_INDEX_TIMEOUT_SECONDS", description="How long to wait for OpenAI indexing before returning"
    )

    @property
    def allowed_extensions(self) -> Set[str]:
        return {ext.strip().lower() for ext in self.allowed_file_types.split(",") if ext}

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]
