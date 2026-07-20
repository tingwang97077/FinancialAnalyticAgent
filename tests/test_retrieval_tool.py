"""Tests for metadata-filtered hybrid retrieval and reranking orchestration."""

from __future__ import annotations

import pytest

import ffa.agent.tools.retrieval_tool as retrieval_tool_module
from ffa.agent.schemas import Entities, Intent, Understanding
from ffa.agent.tools.retrieval_tool import RetrievalPipeline, retrieval_tool
from ffa.retrieval.base import Chunk


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
