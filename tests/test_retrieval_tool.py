"""Tests for metadata-filtered hybrid retrieval and reranking orchestration."""

from __future__ import annotations

import pytest

import ffa.agent.tools.retrieval_tool as retrieval_tool_module
from ffa.agent.schemas import Entities, Intent, Understanding
from ffa.agent.tools.retrieval_tool import RetrievalPipeline, retrieval_tool
from ffa.config import Settings
from ffa.retrieval.base import Chunk
from ffa.retrieval.hybrid import HybridSearchIndex
from ffa.retrieval.rerank import CrossEncoderReranker, NoOpReranker
from ffa.retrieval.vector_search import PreparedVectorQuery, VectorSearchIndex


class FakeHybridIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def search(
        self,
        query: str,
        *,
        filters: dict[str, object],
        k: int,
    ) -> list[Chunk]:
        self.calls.append((query, filters, k))
        return self.chunks[:k]


class SequencedHybridIndex(FakeHybridIndex):
    def __init__(self, responses: list[list[Chunk]]) -> None:
        super().__init__([])
        self.responses = responses

    def search(
        self,
        query: str,
        *,
        filters: dict[str, object],
        k: int,
    ) -> list[Chunk]:
        self.calls.append((query, filters, k))
        return self.responses[len(self.calls) - 1][:k]


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Chunk], int]] = []

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        self.calls.append((query, chunks, top_n))
        return list(reversed(chunks))[:top_n]


class CountingPreparedVectorIndex(VectorSearchIndex):
    """Minimal vector leg that counts preparation separately from SQL passes."""

    def __init__(self, responses: list[list[Chunk]] | None = None) -> None:
        self.prepare_calls: list[str] = []
        self.search_calls: list[tuple[PreparedVectorQuery, dict[str, object], int]] = []
        self.responses = responses or []

    def prepare_query(self, query: str) -> PreparedVectorQuery:
        self.prepare_calls.append(query)
        return PreparedVectorQuery(text=query, embedding=(1.0,))

    def search_prepared(
        self,
        prepared_query: PreparedVectorQuery,
        *,
        filters: dict[str, object],
        k: int,
    ) -> list[Chunk]:
        self.search_calls.append((prepared_query, filters, k))
        call_index = len(self.search_calls) - 1
        if call_index >= len(self.responses):
            return []
        return self.responses[call_index][:k]


@pytest.mark.parametrize(
    ("strategy", "index_type", "reranker_type"),
    [
        ("vector_rerank", VectorSearchIndex, CrossEncoderReranker),
        ("hybrid_rerank", HybridSearchIndex, CrossEncoderReranker),
        ("vector", VectorSearchIndex, NoOpReranker),
        ("hybrid", HybridSearchIndex, NoOpReranker),
    ],
)
def test_from_settings_selects_configured_retrieval_strategy(
    strategy: str,
    index_type: type[object],
    reranker_type: type[object],
) -> None:
    settings = Settings(
        _env_file=None,
        database_url="sqlite+pysqlite://",
        openai_api_key="test-key",
        embedding_dim=3,
        retrieval_strategy=strategy,
    )

    pipeline = RetrievalPipeline.from_settings(settings)

    assert isinstance(pipeline._index, index_type)
    assert isinstance(pipeline._reranker, reranker_type)


def test_default_retrieval_strategy_is_vector_rerank() -> None:
    settings = Settings(_env_file=None)

    assert settings.retrieval_strategy == "vector_rerank"


def test_unknown_retrieval_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="retrieval_strategy"):
        Settings(_env_file=None, retrieval_strategy="bm25")


def test_retrieval_pipeline_searches_then_reranks_with_entity_filters() -> None:
    chunks = [_chunk(1), _chunk(2)]
    index = FakeHybridIndex(chunks)
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(
        index=index,
        reranker=reranker,
        default_k=20,
        default_top_n=5,
    )
    understanding = Understanding(
        intent=Intent.NARRATIVE,
        entities=Entities(
            tickers=["AAPL", "MSFT"],
            ciks=[320193, 789019],
            fiscal_years=[2023, 2024],
            fiscal_periods=["Q3", "FY"],
            sections=["MD&A", "Risk Factors"],
        ),
        rewritten_query="Risk factors disclosed by Apple in fiscal 2025",
    )

    results = pipeline.retrieve(understanding, k=8, top_n=2)

    filters = {
        "ticker": ["AAPL", "MSFT"],
        "fiscal_year": [2023, 2024],
        "fiscal_period": ["Q3", "FY"],
        "section": ["MD&A", "Risk Factors"],
    }
    assert index.calls == [(understanding.rewritten_query, filters, 8)]
    assert reranker.calls == [(understanding.rewritten_query, chunks, 2)]
    assert [chunk["id"] for chunk in results] == [2, 1]


def test_retrieval_pipeline_uses_configured_defaults() -> None:
    chunks = [_chunk(1)]
    index = FakeHybridIndex(chunks)
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(
        index=index,
        reranker=reranker,
        default_k=7,
        default_top_n=3,
    )
    understanding = Understanding(
        intent=Intent.NARRATIVE,
        entities=Entities(tickers=["AAPL"], ciks=[320193]),
        rewritten_query="Apple risk factors",
    )

    pipeline.retrieve(understanding)

    assert index.calls[0][2] == 7
    assert reranker.calls[0][2] == 3


def test_hybrid_retrieval_relaxes_only_periods_after_empty_filtered_search() -> None:
    chunk = _chunk(1)
    index = SequencedHybridIndex([[], [chunk]])
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(
        index=index,
        reranker=reranker,
        default_k=20,
        default_top_n=5,
    )
    understanding = Understanding(
        intent=Intent.HYBRID,
        entities=Entities(
            tickers=["AAPL"],
            ciks=[320193],
            metrics=["net_income"],
            fiscal_years=[2023, 2024],
            sections=["MD&A"],
        ),
        rewritten_query="Apple net income change and management explanation",
    )

    results = pipeline.retrieve(understanding)

    assert index.calls == [
        (
            understanding.rewritten_query,
            {
                "ticker": ["AAPL"],
                "fiscal_year": [2023, 2024],
                "section": ["MD&A"],
            },
            20,
        ),
        (
            understanding.rewritten_query,
            {"ticker": ["AAPL"], "section": ["MD&A"]},
            20,
        ),
    ]
    assert reranker.calls == [(understanding.rewritten_query, [chunk], 5)]
    assert results == [chunk]


def test_hybrid_fallback_prepares_query_embedding_only_once() -> None:
    chunk = _chunk(1)
    text_index = SequencedHybridIndex([[], [chunk]])
    vector_index = CountingPreparedVectorIndex()
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(
        index=HybridSearchIndex(text_index, vector_index),
        reranker=reranker,
        default_k=20,
        default_top_n=5,
    )
    understanding = Understanding(
        intent=Intent.HYBRID,
        entities=Entities(
            tickers=["AAPL"],
            ciks=[320193],
            metrics=["net_income"],
            fiscal_years=[2024],
            sections=["Risk Factors"],
        ),
        rewritten_query="Apple net income and supply chain risks",
    )

    results = pipeline.retrieve(understanding)

    expected_filtered = {
        "ticker": ["AAPL"],
        "fiscal_year": [2024],
        "section": ["Risk Factors"],
    }
    expected_relaxed = {
        "ticker": ["AAPL"],
        "section": ["Risk Factors"],
    }
    assert vector_index.prepare_calls == [understanding.rewritten_query]
    assert [call[1] for call in vector_index.search_calls] == [
        expected_filtered,
        expected_relaxed,
    ]
    assert [chunk["id"] for chunk in results] == [1]


def test_vector_fallback_prepares_query_embedding_only_once() -> None:
    chunk = _chunk(1)
    vector_index = CountingPreparedVectorIndex([[], [chunk]])
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(
        index=vector_index,
        reranker=reranker,
        default_k=20,
        default_top_n=5,
    )
    understanding = Understanding(
        intent=Intent.HYBRID,
        entities=Entities(
            tickers=["AAPL"],
            ciks=[320193],
            metrics=["net_income"],
            fiscal_years=[2024],
            sections=["Risk Factors"],
        ),
        rewritten_query="Apple net income and supply chain risks",
    )

    results = pipeline.retrieve(understanding)

    assert vector_index.prepare_calls == [understanding.rewritten_query]
    assert [call[1] for call in vector_index.search_calls] == [
        {
            "ticker": ["AAPL"],
            "fiscal_year": [2024],
            "section": ["Risk Factors"],
        },
        {
            "ticker": ["AAPL"],
            "section": ["Risk Factors"],
        },
    ]
    assert [chunk["id"] for chunk in results] == [1]


def test_public_retrieval_tool_delegates_to_cached_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = FakeHybridIndex([_chunk(1)])
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(
        index=index,
        reranker=reranker,
        default_k=20,
        default_top_n=5,
    )
    understanding = Understanding(
        intent=Intent.NARRATIVE,
        entities=Entities(tickers=["AAPL"], ciks=[320193]),
        rewritten_query="Apple risk factors",
    )
    monkeypatch.setattr(
        retrieval_tool_module,
        "_get_default_retrieval_pipeline",
        lambda: pipeline,
    )

    results = retrieval_tool(understanding, k=9, top_n=1)

    assert [chunk["id"] for chunk in results] == [1]
    assert index.calls[0][2] == 9
    assert reranker.calls[0][2] == 1


def _chunk(chunk_id: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        accession_no="0000320193-25-000079",
        cik=320193,
        ticker="AAPL",
        fiscal_year=2025,
        fiscal_period="FY",
        section="Risk Factors",
        chunk_index=chunk_id,
        text=f"Risk disclosure {chunk_id}.",
        token_count=4,
        source_url="https://www.sec.gov/filing.htm",
        score=0.5,
    )
