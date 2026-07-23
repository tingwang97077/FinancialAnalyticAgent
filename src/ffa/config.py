"""Application settings loaded from environment variables."""

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
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
    retrieval_strategy: Literal[
        "vector_rerank",
        "hybrid_rerank",
        "vector",
        "hybrid",
    ] = "vector_rerank"
    embedding_dim: int = Field(default=1536, gt=0)

    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = ""
    langfuse_base_url: str = ""

    price_mini_input: Decimal | None = Field(default=None, ge=0)
    price_mini_cached_input: Decimal | None = Field(default=None, ge=0)
    price_mini_output: Decimal | None = Field(default=None, ge=0)
    price_nano_input: Decimal | None = Field(default=None, ge=0)
    price_nano_cached_input: Decimal | None = Field(default=None, ge=0)
    price_nano_output: Decimal | None = Field(default=None, ge=0)
    price_embedding: Decimal | None = Field(default=None, ge=0)

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

    @property
    def langfuse_endpoint(self) -> str | None:
        """Return LANGFUSE_HOST, falling back to LANGFUSE_BASE_URL."""
        return self.langfuse_host.strip() or self.langfuse_base_url.strip() or None

    @field_validator(
        "price_mini_input",
        "price_mini_cached_input",
        "price_mini_output",
        "price_nano_input",
        "price_nano_cached_input",
        "price_nano_output",
        "price_embedding",
        mode="before",
    )
    @classmethod
    def _empty_price_is_unconfigured(cls, value: object) -> object:
        """Allow blank example values without inventing a default price."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
