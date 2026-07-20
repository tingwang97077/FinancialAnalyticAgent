"""Batch text chunks into OpenAI embedding vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ffa.common.openai_client import EmbeddingProvider, OpenAIClient
from ffa.config import Settings, get_settings
from ffa.ingestion.unstructured.chunk import ChunkRow


class EmbeddedChunk(ChunkRow):
    """Chunk row enriched with its dense embedding."""

    embedding: list[float]


def embed_chunks(
    chunks: Sequence[Mapping[str, object]],
    *,
    client: EmbeddingProvider | None = None,
    settings: Settings | None = None,
    batch_size: int = 100,
) -> list[EmbeddedChunk]:
    """Embed chunks in bounded batches while preserving their input order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if not chunks:
        return []

    resolved_settings = settings or get_settings()
    resolved_client = client or OpenAIClient.from_settings(resolved_settings)
    prepared = [_coerce_chunk(chunk) for chunk in chunks]
    embedded: list[EmbeddedChunk] = []

    for offset in range(0, len(prepared), batch_size):
        batch = prepared[offset : offset + batch_size]
        vectors = resolved_client.embed_texts(
            [chunk["text"] for chunk in batch],
            dimensions=resolved_settings.embedding_dim,
        )
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding provider returned an unexpected vector count.")
        for chunk, vector in zip(batch, vectors, strict=True):
            if len(vector) != resolved_settings.embedding_dim:
                raise RuntimeError(
                    "Embedding provider returned a vector with an unexpected dimension."
                )
            embedded.append(EmbeddedChunk(**chunk, embedding=list(vector)))
    return embedded


def _coerce_chunk(chunk: Mapping[str, object]) -> ChunkRow:
    required_fields = set(ChunkRow.__required_keys__)
    if not required_fields.issubset(chunk):
        missing = sorted(required_fields.difference(chunk))
        raise ValueError(f"Chunk metadata is incomplete: {', '.join(missing)}.")
    return dict(chunk)  # type: ignore[return-value]
