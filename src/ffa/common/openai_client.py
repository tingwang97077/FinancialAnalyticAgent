"""Shared OpenAI API client wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI

from ffa.config import Settings, get_settings


class EmbeddingProvider(Protocol):
    """Protocol implemented by embedding backends."""

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        dimensions: int,
    ) -> list[list[float]]:
        """Embed texts in input order."""
        ...


class OpenAIClient:
    """Small application-facing wrapper around the OpenAI SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        embedding_model: str,
        client: OpenAI | None = None,
    ) -> None:
        """Initialize the wrapper with explicit credentials and model selection."""
        normalized_api_key = api_key.strip()
        if not normalized_api_key:
            raise ValueError("OPENAI_API_KEY must be configured before creating embeddings.")
        normalized_model = embedding_model.strip()
        if not normalized_model:
            raise ValueError("OPENAI_EMBEDDING_MODEL must not be empty.")
        self._embedding_model = normalized_model
        self._client = client or OpenAI(api_key=normalized_api_key)

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        client: OpenAI | None = None,
    ) -> OpenAIClient:
        """Create the wrapper from central application settings."""
        resolved_settings = settings or get_settings()
        if resolved_settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY must be configured before creating embeddings.")
        return cls(
            api_key=resolved_settings.openai_api_key.get_secret_value(),
            embedding_model=resolved_settings.openai_embedding_model,
            client=client,
        )

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        dimensions: int,
    ) -> list[list[float]]:
        """Create embedding vectors and preserve the input order."""
        if not texts:
            return []
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero.")
        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=list(texts),
            dimensions=dimensions,
        )
        ordered_data = sorted(response.data, key=lambda item: item.index)
        if len(ordered_data) != len(texts):
            raise RuntimeError("OpenAI returned an unexpected number of embeddings.")
        return [list(item.embedding) for item in ordered_data]
