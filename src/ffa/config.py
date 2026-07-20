"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for all application layers."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr | None = None
    openai_model: str = ""
    openai_classifier_model: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    database_url: str = "postgresql+psycopg://ffa_app:pass@postgres:5432/ffa"
    database_url_readonly: str = "postgresql+psycopg://ffa_ro:pass@postgres:5432/ffa"

    sec_user_agent: str = ""
    sec_max_rps: float = Field(default=8, gt=0, le=10)
    sec_cache_dir: Path = Path("/data/sec_cache")

    ticker_universe: str = "AAPL,MSFT,NVDA,GOOGL,AMZN"

    chunk_max_tokens: int = Field(default=700, gt=0)
    chunk_overlap: float = Field(default=0.15, ge=0, lt=1)
    retrieval_top_k: int = Field(default=20, gt=0)
    rerank_top_n: int = Field(default=5, gt=0)
    embedding_dim: int = Field(default=1536, gt=0)

    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    airflow_core_executor: str = Field(
        default="LocalExecutor",
        validation_alias="AIRFLOW__CORE__EXECUTOR",
    )
    airflow_core_load_examples: bool = Field(
        default=False,
        validation_alias="AIRFLOW__CORE__LOAD_EXAMPLES",
    )

    @property
    def ticker_symbols(self) -> tuple[str, ...]:
        """Return configured ticker symbols in normalized input order."""
        return tuple(symbol.strip() for symbol in self.ticker_universe.split(",") if symbol.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
