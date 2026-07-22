"""pgvector cosine retrieval over filing chunk embeddings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection

from ffa.common.db import create_rw_engine
from ffa.common.openai_client import EmbeddingProvider, OpenAIClient
from ffa.config import Settings, get_settings
from ffa.retrieval.base import (
    CHUNK_SELECT_COLUMNS,
    Chunk,
    build_filter_sql,
    connection_scope,
    row_to_chunk,
    validate_search_request,
)

_SET_EF_SEARCH_SQL = text("SELECT set_config('hnsw.ef_search', :ef_search, true)")


@dataclass(frozen=True, slots=True)
class PreparedVectorQuery:
    """Validated query text and its reusable embedding."""

    text: str
    embedding: tuple[float, ...]


class VectorSearchIndex:
    """Search filing chunks with the HNSW cosine-distance index."""

    def __init__(
        self,
        *,
        engine: Engine | Connection | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        settings: Settings | None = None,
        ef_search: int = 100,
    ) -> None:
        """Initialize the vector index and its reusable query embedding provider."""
        if isinstance(ef_search, bool) or not isinstance(ef_search, int) or ef_search <= 0:
            raise ValueError("HNSW ef_search must be a positive integer.")
        resolved_settings = settings or get_settings()
        self._owns_engine = engine is None
        self._bind = engine or create_rw_engine(resolved_settings.database_url)
        self._embedding_provider = embedding_provider or OpenAIClient.from_settings(
            resolved_settings
        )
        self._embedding_dim = resolved_settings.embedding_dim
        self._ef_search = ef_search

    def search(
        self,
        query: str,
        *,
        filters: dict[str, object],
        k: int,
    ) -> list[Chunk]:
        """Return chunks ordered by pgvector cosine distance."""
        prepared_query = self.prepare_query(query)
        return self.search_prepared(prepared_query, filters=filters, k=k)

    def prepare_query(self, query: str) -> PreparedVectorQuery:
        """Embed one query for reuse across searches with different filters."""
        normalized_query = validate_search_request(query, 1)
        return PreparedVectorQuery(
            text=normalized_query,
            embedding=tuple(self._embed_query(normalized_query)),
        )

    def search_prepared(
        self,
        prepared_query: PreparedVectorQuery,
        *,
        filters: dict[str, object],
        k: int,
    ) -> list[Chunk]:
        """Search with a previously validated query embedding."""
        validate_search_request(prepared_query.text, k)
        query_embedding = prepared_query.embedding
        if len(query_embedding) != self._embedding_dim or not all(
            math.isfinite(value) for value in query_embedding
        ):
            raise ValueError("Prepared query embedding has an invalid vector shape or value.")
        filter_sql, filter_parameters = build_filter_sql(filters)
        statement = text(
            f"""
            SELECT
                {CHUNK_SELECT_COLUMNS},
                1 - (dc.embedding <=> CAST(:query_embedding AS vector)) AS score
            FROM doc_chunks AS dc
            WHERE TRUE
                {filter_sql}
            ORDER BY dc.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :limit
            """
        )
        parameters = {
            "query_embedding": _vector_literal(query_embedding),
            "limit": k,
            **filter_parameters,
        }
        with connection_scope(self._bind) as connection:
            connection.execute(_SET_EF_SEARCH_SQL, {"ef_search": str(self._ef_search)})
            rows = connection.execute(statement, parameters).mappings().all()
        return [row_to_chunk(row) for row in rows]

    def close(self) -> None:
        """Dispose the internally owned engine."""
        if self._owns_engine and isinstance(self._bind, Engine):
            self._bind.dispose()

    def _embed_query(self, query: str) -> list[float]:
        vectors = self._embedding_provider.embed_texts(
            [query],
            dimensions=self._embedding_dim,
        )
        if len(vectors) != 1 or len(vectors[0]) != self._embedding_dim:
            raise RuntimeError("Embedding provider returned an unexpected query vector shape.")
        vector = [float(value) for value in vectors[0]]
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError("Embedding provider returned non-finite query vector values.")
        return vector


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in vector) + "]"
