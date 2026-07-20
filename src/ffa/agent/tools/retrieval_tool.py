"""Narrative retrieval tool composed from hybrid search and reranking."""

from __future__ import annotations

import logging
from functools import lru_cache

from ffa.agent.schemas import Intent, Understanding
from ffa.common.db import create_rw_engine
from ffa.config import Settings, get_settings
from ffa.retrieval.base import Chunk, SearchIndex
from ffa.retrieval.hybrid import HybridSearchIndex
from ffa.retrieval.rerank import CrossEncoderReranker, Reranker
from ffa.retrieval.text_search import TextSearchIndex
from ffa.retrieval.vector_search import VectorSearchIndex

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Apply metadata-filtered hybrid search followed by a pluggable reranker."""

    def __init__(
        self,
        *,
        index: SearchIndex,
        reranker: Reranker,
        default_k: int,
        default_top_n: int,
    ) -> None:
        """Initialize reusable retrieval components and request defaults."""
        self._index = index
        self._reranker = reranker
        self._default_k = _positive_integer(default_k, name="k")
        self._default_top_n = _positive_integer(default_top_n, name="top_n")

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RetrievalPipeline:
        """Build the step-6 hybrid stack with matching query embeddings."""
        resolved_settings = settings or get_settings()
        engine = create_rw_engine(resolved_settings.database_url)
        text_index = TextSearchIndex(engine=engine, settings=resolved_settings)
        vector_index = VectorSearchIndex(engine=engine, settings=resolved_settings)
        return cls(
            index=HybridSearchIndex(text_index, vector_index),
            reranker=CrossEncoderReranker(),
            default_k=resolved_settings.retrieval_top_k,
            default_top_n=resolved_settings.rerank_top_n,
        )

    def retrieve(
        self,
        understanding: Understanding,
        *,
        k: int | None = None,
        top_n: int | None = None,
    ) -> list[Chunk]:
        """Search with entity filters, then rerank the candidate chunks."""
        if understanding.intent not in {Intent.NARRATIVE, Intent.HYBRID}:
            raise ValueError("The retrieval tool only accepts narrative or hybrid understanding.")
        result_count = self._default_k if k is None else _positive_integer(k, name="k")
        rerank_count = (
            self._default_top_n if top_n is None else _positive_integer(top_n, name="top_n")
        )
        filters = _metadata_filters(understanding)
        chunks = self._index.search(
            understanding.rewritten_query,
            filters=filters,
            k=result_count,
        )
        if not chunks and understanding.intent is Intent.HYBRID:
            relaxed_filters = _without_numeric_period_filters(filters)
            if relaxed_filters != filters:
                logger.info(
                    "No hybrid narrative evidence matched numeric period filters; "
                    "retrying with company and section filters."
                )
                chunks = self._index.search(
                    understanding.rewritten_query,
                    filters=relaxed_filters,
                    k=result_count,
                )
        return self._reranker.rerank(
            understanding.rewritten_query,
            chunks,
            rerank_count,
        )


def retrieval_tool(
    understanding: Understanding,
    *,
    k: int | None = None,
    top_n: int | None = None,
) -> list[Chunk]:
    """Return reranked narrative evidence using configured defaults when omitted."""
    return _get_default_retrieval_pipeline().retrieve(
        understanding,
        k=k,
        top_n=top_n,
    )


@lru_cache(maxsize=1)
def _get_default_retrieval_pipeline() -> RetrievalPipeline:
    """Load the cross-encoder and retrieval clients at most once per process."""
    return RetrievalPipeline.from_settings(get_settings())


def _metadata_filters(understanding: Understanding) -> dict[str, object]:
    """Translate all extracted entities into SQL metadata predicates."""
    entities = understanding.entities
    filters: dict[str, object] = {}
    if entities.tickers:
        filters["ticker"] = list(entities.tickers)
    if entities.fiscal_years:
        filters["fiscal_year"] = list(entities.fiscal_years)
    if entities.fiscal_periods:
        filters["fiscal_period"] = list(entities.fiscal_periods)
    if entities.sections:
        filters["section"] = list(entities.sections)
    return filters


def _without_numeric_period_filters(filters: dict[str, object]) -> dict[str, object]:
    """Relax only SQL-oriented periods when a hybrid narrative leg is empty."""
    return {
        name: value
        for name, value in filters.items()
        if name not in {"fiscal_year", "fiscal_period"}
    }


def _positive_integer(value: int, *, name: str) -> int:
    """Validate positive retrieval result counts."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value
