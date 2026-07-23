"""Tests for retrieval ground-truth quality and fair approach comparison."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from ffa.config import Settings
from ffa.evaluation.build_groundtruth import (
    GeneratedQuestion,
    GroundTruthBuildStats,
    GroundTruthPair,
    RetrievalGroundTruth,
    SourceChunk,
    assess_question_quality,
    balanced_sample,
)
from ffa.evaluation.eval_retrieval import (
    CachingEmbeddingProvider,
    CachingReranker,
    EvaluationTextCappedHybridIndex,
    ResolvedPair,
    RetrievalRunResult,
    compute_retrieval_metrics,
    evaluate_index,
    evaluate_reranker,
    persist_retrieval_runs,
    reciprocal_rank_fusion,
    select_best_approach,
    summarize_text_match_counts,
)
from ffa.retrieval.base import Chunk


class FakeEmbeddingProvider:
    """Record batches and return stable two-dimensional vectors."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        dimensions: int,
    ) -> list[list[float]]:
        self.calls.append((list(texts), dimensions))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FakeIndex:
    """Return configured chunks while recording common search arguments."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def search(
        self,
        query: str,
        *,
        filters: dict[str, object],
        k: int,
    ) -> list[Chunk]:
        self.calls.append((query, filters, k))
        return self._chunks[:k]


class FakeReranker:
    """Assign deterministic scores while recording only actually scored chunks."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[int], int]] = []
        self.preload_calls = 0

    def preload(self) -> None:
        self.preload_calls += 1

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        self.calls.append((query, [chunk["id"] for chunk in chunks], top_n))
        scored: list[Chunk] = []
        for chunk in chunks:
            ranked_chunk = Chunk(**chunk)
            ranked_chunk["score"] = float(chunk["id"])
            ranked_chunk["rerank_score"] = float(chunk["id"])
            scored.append(ranked_chunk)
        return sorted(scored, key=lambda chunk: -chunk["score"])[:top_n]


class FakePersistResult:
    """Minimal SQLAlchemy result returning one generated ID."""

    def __init__(self, row_id: int) -> None:
        self._row_id = row_id

    def one(self) -> tuple[int]:
        return (self._row_id,)


class FakePersistConnection:
    """Capture persisted JSON payloads."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def execute(self, _: object, parameters: dict[str, str]) -> FakePersistResult:
        self.calls.append(parameters)
        return FakePersistResult(len(self.calls))


class FakeBeginContext(AbstractContextManager[FakePersistConnection]):
    """Expose a fake transactional connection."""

    def __init__(self, connection: FakePersistConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakePersistConnection:
        return self._connection

    def __exit__(self, *args: object) -> None:
        del args


class FakePersistEngine:
    """Provide one captured transaction for eval_runs persistence."""

    def __init__(self) -> None:
        self.connection = FakePersistConnection()

    def begin(self) -> FakeBeginContext:
        return FakeBeginContext(self.connection)


def test_balanced_sampling_is_seeded_and_covers_all_strata() -> None:
    chunks = [
        _source(
            chunk_id=index,
            ticker=ticker,
            section=section,
            token_count=token_count,
        )
        for index, (ticker, section, token_count) in enumerate(
            (
                (ticker, section, token_count)
                for ticker in ("AAPL", "JPM")
                for section in ("MD&A", "Risk Factors", "Notes")
                for token_count in (300, 600, 700)
            ),
            start=1,
        )
    ]

    first = balanced_sample(chunks, count=len(chunks), seed=42)
    second = balanced_sample(chunks, count=len(chunks), seed=42)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert {chunk.ticker for chunk in first} == {"AAPL", "JPM"}
    assert {chunk.section for chunk in first} == {"MD&A", "Risk Factors", "Notes"}
    assert {chunk.size_bucket for chunk in first} == {"short", "medium", "long"}


def test_quality_filter_accepts_paraphrase_and_rejects_copy_or_generic_question() -> None:
    chunk = _source(
        chunk_id=1,
        ticker="AAPL",
        section="Risk Factors",
        token_count=500,
        text=(
            "Reliance on a small group of component manufacturers could interrupt production "
            "when capacity is unavailable and delay product deliveries."
        ),
    )
    paraphrase = GeneratedQuestion(
        chunk_id=1,
        question=(
            "How could Apple face shipment disruption when specialized vendors cannot provide "
            "enough parts?"
        ),
        is_specific=True,
    )
    copied = GeneratedQuestion(
        chunk_id=1,
        question=(
            "How could Apple reliance on a small group of component manufacturers interrupt "
            "production?"
        ),
        is_specific=True,
    )
    generic = GeneratedQuestion(
        chunk_id=1,
        question="What important business risks does Apple discuss with its investors?",
        is_specific=True,
    )

    assert assess_question_quality(chunk, paraphrase)[0] is True
    assert assess_question_quality(chunk, copied)[1] == "copied_phrase"
    assert assess_question_quality(chunk, generic)[1] == "too_generic"


def test_embedding_cache_primes_once_and_reuses_identical_query_vectors() -> None:
    raw_provider = FakeEmbeddingProvider()
    provider = CachingEmbeddingProvider(raw_provider)

    provider.prime(["question one", "question two", "question one"], dimensions=2)
    vectors = provider.embed_texts(["question two", "question one"], dimensions=2)

    assert raw_provider.calls == [(["question one", "question two"], 2)]
    assert vectors == [[1.0, 1.0], [0.0, 1.0]]


def test_all_indexes_are_evaluated_with_empty_filters_and_same_k() -> None:
    pair = _resolved_pair("pair-1", expected_chunk_id=1)
    index = FakeIndex([_chunk(1), _chunk(2)])

    rankings, chunks, _ = evaluate_index(index, [pair], k=10)

    assert index.calls == [(pair.pair.question, {}, 10)]
    assert rankings == {"pair-1": [1, 2]}
    assert [chunk["id"] for chunk in chunks["pair-1"]] == [1, 2]


def test_text_capped_hybrid_limits_only_text_candidates_before_rrf() -> None:
    text_index = FakeIndex([_chunk(index) for index in range(1, 61)])
    vector_index = FakeIndex([_chunk(index) for index in range(40, 80)])
    index = EvaluationTextCappedHybridIndex(
        text_index,
        vector_index,
        text_cap=50,
    )

    results = index.search("natural language question", filters={}, k=10)

    assert text_index.calls == [("natural language question", {}, 50)]
    assert vector_index.calls == [("natural language question", {}, 30)]
    assert len(results) == 10
    assert all("rrf_score" in chunk for chunk in results)


def test_reciprocal_rank_fusion_rewards_overlap_without_comparing_raw_scores() -> None:
    text_results = [_chunk(1), _chunk(2)]
    vector_results = [_chunk(3), _chunk(2)]
    text_results[0]["score"] = 99.0
    vector_results[0]["score"] = 0.01

    results = reciprocal_rank_fusion(text_results, vector_results, k=3)

    assert [chunk["id"] for chunk in results] == [2, 1, 3]
    assert results[0]["text_score"] == 0.5
    assert results[0]["vector_score"] == 0.5


def test_rerank_score_cache_scores_shared_candidates_only_once() -> None:
    pair = _resolved_pair("pair-1", expected_chunk_id=1)
    delegate = FakeReranker()
    reranker = CachingReranker(delegate)

    first, _ = evaluate_reranker(
        reranker,
        [pair],
        {"pair-1": [_chunk(1), _chunk(2)]},
        top_n=2,
    )
    second, _ = evaluate_reranker(
        reranker,
        [pair],
        {"pair-1": [_chunk(2), _chunk(3)]},
        top_n=2,
    )

    assert first == {"pair-1": [2, 1]}
    assert second == {"pair-1": [3, 2]}
    assert delegate.calls == [
        (pair.pair.question, [1, 2], 2),
        (pair.pair.question, [3], 1),
    ]


def test_text_match_stats_report_raw_result_distribution() -> None:
    stats = summarize_text_match_counts([0, 2000, 2500, 2500])

    assert stats.mean_matches == 1750.0
    assert stats.min_matches == 0
    assert stats.max_matches == 2500
    assert stats.zero_match_questions == 1


def test_metrics_match_single_relevant_chunk_ranks() -> None:
    pairs = [
        _resolved_pair("pair-1", expected_chunk_id=1),
        _resolved_pair("pair-2", expected_chunk_id=2),
    ]
    rankings = {
        "pair-1": [9, 1],
        "pair-2": [2, 8],
    }

    metrics = compute_retrieval_metrics(pairs, rankings, cutoffs=(1, 2))

    assert metrics == {
        "hit_rate@1": 0.5,
        "mrr@1": 0.5,
        "recall@1": 0.5,
        "ndcg@1": 0.5,
        "hit_rate@2": 1.0,
        "mrr@2": 0.75,
        "recall@2": 1.0,
        "ndcg@2": 0.815465,
    }


def test_best_approach_uses_ndcg_then_mrr_at_primary_cutoff() -> None:
    results = [
        _run_result("text_search", ndcg=0.4, mrr=0.5),
        _run_result("vector_search", ndcg=0.5, mrr=0.4),
        _run_result("hybrid_rrf", ndcg=0.5, mrr=0.6),
        _run_result("hybrid_rrf_rerank", ndcg=0.7, mrr=0.7),
        _run_result("vector_rerank", ndcg=0.8, mrr=0.8),
    ]

    assert select_best_approach(results, primary_cutoff=10) == "vector_rerank"


def test_each_approach_is_persisted_as_a_retrieval_eval_run() -> None:
    engine = FakePersistEngine()
    dataset = _dataset()
    results = [
        _run_result("text_search", ndcg=0.4, mrr=0.5),
        _run_result("vector_search", ndcg=0.5, mrr=0.4),
        _run_result("hybrid_rrf", ndcg=0.6, mrr=0.6),
        _run_result("hybrid_rrf_rerank", ndcg=0.7, mrr=0.7),
        _run_result("vector_rerank", ndcg=0.8, mrr=0.8),
    ]

    run_ids = persist_retrieval_runs(
        engine,  # type: ignore[arg-type]
        dataset=dataset,
        dataset_path=Path("evaluation/retrieval_groundtruth.json"),
        cutoffs=(5, 10),
        max_k=10,
        settings=Settings(
            _env_file=None,
            openai_embedding_model="configured-embedding",
        ),
        results=results,
    )

    assert run_ids == {
        "text_search": 1,
        "vector_search": 2,
        "hybrid_rrf": 3,
        "hybrid_rrf_rerank": 4,
        "vector_rerank": 5,
    }
    assert len(engine.connection.calls) == 5
    configs = [json_load(call["config"]) for call in engine.connection.calls]
    assert [config["approach"] for config in configs] == list(run_ids)
    assert all(config["filters"] == {} for config in configs)
    assert all(config["max_k"] == 10 for config in configs)


def _source(
    *,
    chunk_id: int,
    ticker: str,
    section: str,
    token_count: int,
    text: str = "Distinct narrative filing disclosure.",
) -> SourceChunk:
    return SourceChunk(
        id=chunk_id,
        accession_no=f"accession-{chunk_id}",
        ticker=ticker,
        company_name="Apple Inc." if ticker == "AAPL" else "JPMorgan Chase & Co.",
        section=section,
        chunk_index=chunk_id,
        token_count=token_count,
        text=text,
        source_url="https://www.sec.gov/filing.htm",
    )


def _pair(pair_id: str, expected_chunk_id: int) -> GroundTruthPair:
    return GroundTruthPair(
        pair_id=pair_id,
        question=f"How does Apple explain the distinct issue number {expected_chunk_id}?",
        expected_chunk_id=expected_chunk_id,
        expected_accession_no=f"accession-{expected_chunk_id}",
        expected_section="MD&A",
        expected_chunk_index=expected_chunk_id,
        ticker="AAPL",
        source="synthetic",
        lexical_overlap=0.2,
    )


def _resolved_pair(pair_id: str, *, expected_chunk_id: int) -> ResolvedPair:
    return ResolvedPair(
        pair=_pair(pair_id, expected_chunk_id),
        expected_chunk_id=expected_chunk_id,
    )


def _chunk(chunk_id: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        accession_no=f"accession-{chunk_id}",
        cik=320193,
        ticker="AAPL",
        fiscal_year=2025,
        fiscal_period="FY",
        section="MD&A",
        chunk_index=chunk_id,
        text="Distinct narrative filing disclosure.",
        token_count=5,
        source_url="https://www.sec.gov/filing.htm",
        score=1.0 / chunk_id,
    )


def _run_result(approach: str, *, ndcg: float, mrr: float) -> RetrievalRunResult:
    return RetrievalRunResult(
        approach=approach,
        metrics={
            "hit_rate@10": ndcg,
            "mrr@10": mrr,
            "recall@10": ndcg,
            "ndcg@10": ndcg,
        },
        elapsed_seconds=1.0,
    )


def _dataset() -> RetrievalGroundTruth:
    pair = _pair("pair-1", 1)
    return RetrievalGroundTruth(
        version=1,
        seed=42,
        generated_at=datetime.now(UTC),
        generation_model="configured-classifier",
        corpus_fingerprint="fingerprint",
        eligible_sections=("MD&A", "Risk Factors", "Notes"),
        filter_policy={},
        stats=GroundTruthBuildStats(
            candidates_sampled=1,
            generated_questions=1,
            quality_passed=1,
            retained_synthetic=1,
            rejected_synthetic=0,
            manual_questions=0,
            rejection_reasons={},
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            cost_usd=Decimal("0"),
            elapsed_seconds=0.1,
        ),
        pairs=[pair],
    )


def json_load(value: str) -> dict[str, Any]:
    import json

    return json.loads(value)
