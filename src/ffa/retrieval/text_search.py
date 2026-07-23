"""PostgreSQL full-text retrieval over generated tsvector documents."""

from __future__ import annotations

from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection

from ffa.common.db import create_rw_engine
from ffa.config import Settings, get_settings
from ffa.retrieval.base import (
    CHUNK_SELECT_COLUMNS,
    Chunk,
    build_filter_sql,
    connection_scope,
    row_to_chunk,
    validate_search_request,
)


class TextSearchIndex:
    """Search filing chunks with PostgreSQL GIN-backed full-text search."""

    def __init__(
        self,
        *,
        engine: Engine | Connection | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize a persistent search index with an optional injected bind."""
        self._owns_engine = engine is None
        self._bind = engine or create_rw_engine((settings or get_settings()).database_url)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, object],
        k: int,
    ) -> list[Chunk]:
        """Return chunks ranked by cover-density text rank."""
        normalized_query = validate_search_request(query, k)
        filter_sql, filter_parameters = build_filter_sql(filters)
        statement = text(
            f"""
            WITH search_terms AS (
                SELECT unnest(
                    tsvector_to_array(to_tsvector('english', :query))
                ) AS term
            ),
            search_query AS (
                SELECT CASE
                    WHEN count(*) = 0 THEN NULL
                    ELSE to_tsquery(
                        'english',
                        string_agg(quote_literal(term), ' | ' ORDER BY term)
                    )
                END AS value
                FROM search_terms
            )
            SELECT
                {CHUNK_SELECT_COLUMNS},
                ts_rank_cd(dc.text_tsv, sq.value) AS score
            FROM doc_chunks AS dc
            CROSS JOIN search_query AS sq
            WHERE sq.value IS NOT NULL
                AND dc.text_tsv @@ sq.value
                {filter_sql}
            ORDER BY score DESC, dc.id ASC
            LIMIT :limit
            """
        )
        parameters = {
            "query": normalized_query,
            "limit": k,
            **filter_parameters,
        }
        with connection_scope(self._bind) as connection:
            rows = connection.execute(statement, parameters).mappings().all()
        return [row_to_chunk(row) for row in rows]

    def close(self) -> None:
        """Dispose the internally owned engine."""
        if self._owns_engine and isinstance(self._bind, Engine):
            self._bind.dispose()
