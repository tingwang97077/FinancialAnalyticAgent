"""Compare retrieval approaches on one frozen, single-relevance ground truth."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine, text

from ffa.common.db import create_rw_engine
from ffa.common.openai_client import EmbeddingProvider, OpenAIClient
from ffa.config import Settings, get_settings
from ffa.evaluation.build_groundtruth import (
    DEFAULT_OUTPUT_PATH,
    GroundTruthPair,
    RetrievalGroundTruth,
    SourceChunk,
    corpus_fingerprint,
    load_groundtruth,
    load_source_chunks,
)
from ffa.monitoring.tracing import RequestTracer
from ffa.retrieval.base import Chunk, SearchIndex, validate_search_request
from ffa.retrieval.hybrid import HybridSearchIndex
from ffa.retrieval.rerank import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
    Reranker,
)
from ffa.retrieval.text_search import TextSearchIndex
from ffa.retrieval.vector_search import VectorSearchIndex

DEFAULT_CUTOFFS = (5, 10)
PRODUCTION_APPROACH = "hybrid_rrf_rerank"
VECTOR_RERANK_APPROACH = "vector_rerank"
APPROACH_ORDER = (
    "text_search",
    "vector_search",
    "hybrid_rrf",
    PRODUCTION_APPROACH,
    VECTOR_RERANK_APPROACH,
)
EXPLORATORY_TEXT_CAPS = (10, 50, 100)
RRF_K = 60
VECTOR_CANDIDATE_MULTIPLIER = 3


class RetrievalRunResult(BaseModel):
    """Metrics and persisted identity for one retrieval configuration."""

    model_config = ConfigDict(frozen=True)

    approach: str
    metrics: dict[str, float]
    elapsed_seconds: float
    eval_run_id: int | None = None


class TextMatchStats(BaseModel):
    """Distribution of raw full-text matches before SQL LIMIT and ranking."""

    model_config = ConfigDict(frozen=True)

    mean_matches: float
    min_matches: int
    max_matches: int
    zero_match_questions: int


class RetrievalEvaluationSummary(BaseModel):
    """Complete comparison report for one frozen benchmark."""

    model_config = ConfigDict(frozen=True)

    dataset_path: str
    corpus_fingerprint: str
    question_count: int
    cutoffs: tuple[int, ...]
    max_k: int
    filters: dict[str, object]
    embedding_model: str
    embedding_input_tokens: int
    embedding_cost_usd: Decimal
    embedding_seconds: float
    reranker_model: str
    reranker_load_seconds: float
    text_match_stats: TextMatchStats
    total_elapsed_seconds: float
    best_approach: str
    production_approach: str
    production_confirmed: bool
    results: list[RetrievalRunResult]
    exploratory_results: list[RetrievalRunResult]


@dataclass(frozen=True, slots=True)
class ResolvedPair:
    """Ground-truth pair resolved to the current database row ID."""

    pair: GroundTruthPair
    expected_chunk_id: int


class PreloadableReranker(Reranker, Protocol):
    """Reranker that can load model weights before timed inference."""

    def preload(self) -> None:
        """Load model state once."""
        ...


class CachingEmbeddingProvider:
    """Cache query embeddings so all configurations use identical vectors."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        """Wrap one embedding backend with a text-and-dimension cache."""
        self._provider = provider
        self._cache: dict[tuple[str, int], list[float]] = {}

    def prime(self, texts: Sequence[str], *, dimensions: int) -> None:
        """Embed all unique missing questions in one batch before evaluation."""
        self.embed_texts(list(dict.fromkeys(texts)), dimensions=dimensions)

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        dimensions: int,
    ) -> list[list[float]]:
        """Return cached vectors and batch only cache misses."""
        missing = [
            value for value in dict.fromkeys(texts) if (value, dimensions) not in self._cache
        ]
        if missing:
            vectors = self._provider.embed_texts(missing, dimensions=dimensions)
            if len(vectors) != len(missing):
                raise RuntimeError("Embedding provider returned an unexpected vector count.")
            for value, vector in zip(missing, vectors, strict=True):
                self._cache[(value, dimensions)] = list(vector)
        return [list(self._cache[(value, dimensions)]) for value in texts]


class CachingReranker:
    """Reuse deterministic cross-encoder scores across evaluation configurations."""

    def __init__(self, reranker: PreloadableReranker) -> None:
        """Wrap one preloadable reranker with a query/chunk score cache."""
        self._reranker = reranker
        self._scores: dict[tuple[str, int], float] = {}

    def preload(self) -> None:
        """Load the wrapped model once."""
        self._reranker.preload()

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """Score unseen query/chunk pairs and rank the requested candidate set."""
        normalized_query = validate_search_request(query, top_n)
        missing = [chunk for chunk in chunks if (normalized_query, chunk["id"]) not in self._scores]
        if missing:
            scored_missing = self._reranker.rerank(
                normalized_query,
                missing,
                len(missing),
            )
            for chunk in scored_missing:
                self._scores[(normalized_query, chunk["id"])] = float(chunk["score"])

        ranked: list[tuple[int, Chunk]] = []
        for original_rank, chunk in enumerate(chunks):
            ranked_chunk = Chunk(**chunk)
            score = self._scores[(normalized_query, chunk["id"])]
            ranked_chunk["rerank_score"] = score
            ranked_chunk["score"] = score
            ranked.append((original_rank, ranked_chunk))
        ranked.sort(key=lambda item: (-item[1]["score"], item[0]))
        return [chunk for _, chunk in ranked[:top_n]]


class EvaluationTextCappedHybridIndex:
    """Evaluation-only RRF with an independently bounded full-text leg."""

    def __init__(
        self,
        text_index: SearchIndex,
        vector_index: SearchIndex,
        *,
        text_cap: int,
        rrf_k: int = RRF_K,
        vector_candidate_multiplier: int = VECTOR_CANDIDATE_MULTIPLIER,
    ) -> None:
        """Configure a text cap without altering the production hybrid index."""
        if text_cap <= 0:
            raise ValueError("Text candidate cap must be positive.")
        self._text_index = text_index
        self._vector_index = vector_index
        self._text_cap = text_cap
        self._rrf_k = rrf_k
        self._vector_candidate_multiplier = vector_candidate_multiplier

    def search(
        self,
        query: str,
        *,
        filters: dict[str, object],
        k: int,
    ) -> list[Chunk]:
        """Fuse a capped text list with the production-sized vector list."""
        normalized_query = validate_search_request(query, k)
        text_results = self._text_index.search(
            normalized_query,
            filters=filters,
            k=self._text_cap,
        )
        vector_results = self._vector_index.search(
            normalized_query,
            filters=filters,
            k=k * self._vector_candidate_multiplier,
        )
        return reciprocal_rank_fusion(
            text_results,
            vector_results,
            k=k,
            rrf_k=self._rrf_k,
        )


def evaluate_retrieval(
    *,
    dataset_path: Path = DEFAULT_OUTPUT_PATH,
    engine: Engine | None = None,
    settings: Settings | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    reranker: PreloadableReranker | None = None,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    persist: bool = True,
) -> RetrievalEvaluationSummary:
    """Evaluate five primary and several text-cap retrieval configurations."""
    started_at = perf_counter()
    normalized_cutoffs = _validate_cutoffs(cutoffs)
    max_k = max(normalized_cutoffs)
    resolved_settings = settings or get_settings()
    owns_engine = engine is None
    resolved_engine = engine or create_rw_engine(resolved_settings.database_url)
    try:
        dataset = load_groundtruth(dataset_path)
        corpus = load_source_chunks(resolved_engine)
        current_fingerprint = corpus_fingerprint(corpus)
        if current_fingerprint != dataset.corpus_fingerprint:
            raise RuntimeError(
                "Ground-truth corpus fingerprint does not match the current database. "
                "Rebuild the dataset before comparing retrieval approaches."
            )
        pairs = resolve_current_pairs(dataset, corpus)
        if not pairs:
            raise RuntimeError("Retrieval ground truth contains no questions.")

        raw_embedding_provider = embedding_provider or OpenAIClient.from_settings(resolved_settings)
        cached_embeddings = CachingEmbeddingProvider(raw_embedding_provider)
        text_index = TextSearchIndex(engine=resolved_engine)
        vector_index = VectorSearchIndex(
            engine=resolved_engine,
            embedding_provider=cached_embeddings,
            settings=resolved_settings,
        )
        hybrid_index = HybridSearchIndex(text_index, vector_index)
        resolved_reranker = CachingReranker(reranker or CrossEncoderReranker())
        tracer = RequestTracer(settings=resolved_settings)

        with tracer.trace(
            trace_id=uuid4().hex,
            question="evaluate retrieval benchmark",
            session_id=None,
        ) as evaluation_trace:
            embedding_started = perf_counter()
            cached_embeddings.prime(
                [item.pair.question for item in pairs],
                dimensions=resolved_settings.embedding_dim,
            )
            embedding_seconds = perf_counter() - embedding_started

            reranker_started = perf_counter()
            resolved_reranker.preload()
            reranker_load_seconds = perf_counter() - reranker_started

            text_match_stats = measure_text_match_stats(resolved_engine, pairs)
            text_rankings, _, text_seconds = evaluate_index(
                text_index,
                pairs,
                k=max_k,
            )
            vector_rankings, vector_chunks, vector_seconds = evaluate_index(
                vector_index,
                pairs,
                k=max_k,
            )
            hybrid_rankings, hybrid_chunks, hybrid_seconds = evaluate_index(
                hybrid_index,
                pairs,
                k=max_k,
            )
            hybrid_rerank_rankings, hybrid_rerank_seconds = evaluate_reranker(
                resolved_reranker,
                pairs,
                hybrid_chunks,
                top_n=max_k,
            )
            vector_rerank_rankings, vector_rerank_seconds = evaluate_reranker(
                resolved_reranker,
                pairs,
                vector_chunks,
                top_n=max_k,
            )

            exploratory_rankings: dict[str, dict[str, list[int]]] = {}
            exploratory_elapsed: dict[str, float] = {}
            for text_cap in EXPLORATORY_TEXT_CAPS:
                approach = _text_cap_approach(text_cap)
                capped_index = EvaluationTextCappedHybridIndex(
                    text_index,
                    vector_index,
                    text_cap=text_cap,
                )
                rankings, chunks, elapsed_seconds = evaluate_index(
                    capped_index,
                    pairs,
                    k=max_k,
                )
                exploratory_rankings[approach] = rankings
                exploratory_elapsed[approach] = elapsed_seconds

                rerank_approach = f"{approach}_rerank"
                reranked, rerank_seconds = evaluate_reranker(
                    resolved_reranker,
                    pairs,
                    chunks,
                    top_n=max_k,
                )
                exploratory_rankings[rerank_approach] = reranked
                exploratory_elapsed[rerank_approach] = elapsed_seconds + rerank_seconds

        rankings_by_approach = {
            "text_search": (text_rankings, text_seconds),
            "vector_search": (vector_rankings, vector_seconds),
            "hybrid_rrf": (hybrid_rankings, hybrid_seconds),
            PRODUCTION_APPROACH: (
                hybrid_rerank_rankings,
                hybrid_seconds + hybrid_rerank_seconds,
            ),
            VECTOR_RERANK_APPROACH: (
                vector_rerank_rankings,
                vector_seconds + vector_rerank_seconds,
            ),
        }
        results = [
            RetrievalRunResult(
                approach=approach,
                metrics=compute_retrieval_metrics(
                    pairs,
                    rankings_by_approach[approach][0],
                    cutoffs=normalized_cutoffs,
                ),
                elapsed_seconds=round(rankings_by_approach[approach][1], 3),
            )
            for approach in APPROACH_ORDER
        ]
        exploratory_results = [
            RetrievalRunResult(
                approach=approach,
                metrics=compute_retrieval_metrics(
                    pairs,
                    exploratory_rankings[approach],
                    cutoffs=normalized_cutoffs,
                ),
                elapsed_seconds=round(exploratory_elapsed[approach], 3),
            )
            for text_cap in EXPLORATORY_TEXT_CAPS
            for approach in (
                _text_cap_approach(text_cap),
                f"{_text_cap_approach(text_cap)}_rerank",
            )
        ]
        best_approach = select_best_approach(results, primary_cutoff=max_k)
        run_ids = (
            persist_retrieval_runs(
                resolved_engine,
                dataset=dataset,
                dataset_path=dataset_path,
                cutoffs=normalized_cutoffs,
                max_k=max_k,
                settings=resolved_settings,
                results=[*results, *exploratory_results],
            )
            if persist
            else {}
        )
        persisted_results = [
            result.model_copy(update={"eval_run_id": run_ids.get(result.approach)})
            for result in results
        ]
        persisted_exploratory_results = [
            result.model_copy(update={"eval_run_id": run_ids.get(result.approach)})
            for result in exploratory_results
        ]
        total_elapsed_seconds = perf_counter() - started_at
        return RetrievalEvaluationSummary(
            dataset_path=str(dataset_path),
            corpus_fingerprint=current_fingerprint,
            question_count=len(pairs),
            cutoffs=normalized_cutoffs,
            max_k=max_k,
            filters={},
            embedding_model=resolved_settings.openai_embedding_model,
            embedding_input_tokens=evaluation_trace.metrics.input_tokens,
            embedding_cost_usd=evaluation_trace.metrics.cost_usd,
            embedding_seconds=round(embedding_seconds, 3),
            reranker_model=DEFAULT_CROSS_ENCODER_MODEL,
            reranker_load_seconds=round(reranker_load_seconds, 3),
            text_match_stats=text_match_stats,
            total_elapsed_seconds=round(total_elapsed_seconds, 3),
            best_approach=best_approach,
            production_approach=PRODUCTION_APPROACH,
            production_confirmed=best_approach == PRODUCTION_APPROACH,
            results=persisted_results,
            exploratory_results=persisted_exploratory_results,
        )
    finally:
        if owns_engine:
            resolved_engine.dispose()


def evaluate_index(
    index: SearchIndex,
    pairs: Sequence[ResolvedPair],
    *,
    k: int,
) -> tuple[dict[str, list[int]], dict[str, list[Chunk]], float]:
    """Run one interchangeable index with no metadata filters."""
    started_at = perf_counter()
    chunks_by_pair: dict[str, list[Chunk]] = {}
    for item in pairs:
        chunks_by_pair[item.pair.pair_id] = index.search(
            item.pair.question,
            filters={},
            k=k,
        )
    rankings = {
        pair_id: [chunk["id"] for chunk in chunks] for pair_id, chunks in chunks_by_pair.items()
    }
    return rankings, chunks_by_pair, perf_counter() - started_at


def evaluate_reranker(
    reranker: Reranker,
    pairs: Sequence[ResolvedPair],
    chunks_by_pair: Mapping[str, list[Chunk]],
    *,
    top_n: int,
) -> tuple[dict[str, list[int]], float]:
    """Rerank one candidate configuration and return stable ID rankings."""
    started_at = perf_counter()
    rankings: dict[str, list[int]] = {}
    for item in pairs:
        reranked = reranker.rerank(
            item.pair.question,
            chunks_by_pair[item.pair.pair_id],
            top_n,
        )
        rankings[item.pair.pair_id] = [chunk["id"] for chunk in reranked]
    return rankings, perf_counter() - started_at


def reciprocal_rank_fusion(
    text_results: Sequence[Chunk],
    vector_results: Sequence[Chunk],
    *,
    k: int,
    rrf_k: int = RRF_K,
) -> list[Chunk]:
    """Fuse two pre-ranked lists for evaluation without raw-score comparison."""
    fused: dict[int, Chunk] = {}
    best_rank: dict[int, int] = {}
    for leg_name, results in (("text", text_results), ("vector", vector_results)):
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["id"]
            if chunk_id not in fused:
                fused[chunk_id] = Chunk(**chunk)
                fused[chunk_id]["rrf_score"] = 0.0
                best_rank[chunk_id] = rank
            best_rank[chunk_id] = min(best_rank[chunk_id], rank)
            fused_chunk = fused[chunk_id]
            fused_chunk["rrf_score"] += 1 / (rrf_k + rank)
            if leg_name == "text":
                fused_chunk["text_score"] = chunk["score"]
            else:
                fused_chunk["vector_score"] = chunk["score"]

    for chunk in fused.values():
        chunk["score"] = chunk["rrf_score"]
    return sorted(
        fused.values(),
        key=lambda chunk: (-chunk["score"], best_rank[chunk["id"]], chunk["id"]),
    )[:k]


def measure_text_match_stats(
    engine: Engine,
    pairs: Sequence[ResolvedPair],
) -> TextMatchStats:
    """Count raw OR-query matches before ranking and limiting."""
    statement = text(
        """
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
        SELECT count(*)
        FROM doc_chunks AS dc
        CROSS JOIN search_query AS sq
        WHERE sq.value IS NOT NULL
          AND dc.text_tsv @@ sq.value
        """
    )
    with engine.connect() as connection:
        counts = [
            int(
                connection.execute(
                    statement,
                    {"query": item.pair.question},
                ).scalar_one()
            )
            for item in pairs
        ]
    return summarize_text_match_counts(counts)


def summarize_text_match_counts(counts: Sequence[int]) -> TextMatchStats:
    """Summarize non-negative raw full-text result counts."""
    if not counts:
        raise ValueError("At least one text match count is required.")
    if any(count < 0 for count in counts):
        raise ValueError("Text match counts must be non-negative.")
    return TextMatchStats(
        mean_matches=round(statistics.fmean(counts), 3),
        min_matches=min(counts),
        max_matches=max(counts),
        zero_match_questions=sum(count == 0 for count in counts),
    )


def compute_retrieval_metrics(
    pairs: Sequence[ResolvedPair],
    rankings: Mapping[str, Sequence[int]],
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[str, float]:
    """Compute hit rate, MRR, recall, and binary-relevance NDCG at each k.

    Each benchmark question has exactly one relevant chunk, so recall@k equals
    hit_rate@k by construction. Both are reported for standard comparability.
    """
    normalized_cutoffs = _validate_cutoffs(cutoffs)
    if not pairs:
        raise ValueError("At least one resolved pair is required.")
    totals: dict[str, float] = {}
    for cutoff in normalized_cutoffs:
        hits = 0.0
        reciprocal_ranks = 0.0
        ndcg = 0.0
        for item in pairs:
            ranked_ids = list(rankings.get(item.pair.pair_id, ()))[:cutoff]
            try:
                rank = ranked_ids.index(item.expected_chunk_id) + 1
            except ValueError:
                continue
            hits += 1
            reciprocal_ranks += 1 / rank
            ndcg += 1 / math.log2(rank + 1)
        denominator = len(pairs)
        totals[f"hit_rate@{cutoff}"] = hits / denominator
        totals[f"mrr@{cutoff}"] = reciprocal_ranks / denominator
        totals[f"recall@{cutoff}"] = hits / denominator
        totals[f"ndcg@{cutoff}"] = ndcg / denominator
    return {name: round(value, 6) for name, value in totals.items()}


def resolve_current_pairs(
    dataset: RetrievalGroundTruth,
    corpus: Sequence[SourceChunk],
) -> list[ResolvedPair]:
    """Resolve stable pair identities against current database chunk IDs."""
    current_by_identity = {chunk.identity: chunk for chunk in corpus}

    resolved: list[ResolvedPair] = []
    for pair in dataset.pairs:
        identity = (
            pair.expected_accession_no,
            pair.expected_section,
            pair.expected_chunk_index,
        )
        chunk = current_by_identity.get(identity)
        if chunk is None:
            raise RuntimeError(f"Expected ground-truth chunk is missing: {identity}.")
        if chunk.ticker != pair.ticker:
            raise RuntimeError(f"Ground-truth ticker mismatch for {identity}.")
        resolved.append(
            ResolvedPair(
                pair=pair,
                expected_chunk_id=chunk.id,
            )
        )
    return resolved


def select_best_approach(
    results: Sequence[RetrievalRunResult],
    *,
    primary_cutoff: int,
) -> str:
    """Choose the best ranking by NDCG, then MRR, then hit rate."""
    if not results:
        raise ValueError("At least one retrieval result is required.")
    return max(
        results,
        key=lambda result: (
            result.metrics[f"ndcg@{primary_cutoff}"],
            result.metrics[f"mrr@{primary_cutoff}"],
            result.metrics[f"hit_rate@{primary_cutoff}"],
            -APPROACH_ORDER.index(result.approach),
        ),
    ).approach


def persist_retrieval_runs(
    engine: Engine,
    *,
    dataset: RetrievalGroundTruth,
    dataset_path: Path,
    cutoffs: tuple[int, ...],
    max_k: int,
    settings: Settings,
    results: Sequence[RetrievalRunResult],
) -> dict[str, int]:
    """Persist one eval_runs row per compared configuration atomically."""
    statement = text(
        """
        INSERT INTO eval_runs (run_type, config, metrics)
        VALUES ('retrieval', CAST(:config AS JSONB), CAST(:metrics AS JSONB))
        RETURNING id
        """
    )
    run_ids: dict[str, int] = {}
    with engine.begin() as connection:
        for result in results:
            config = {
                "dataset_path": str(dataset_path),
                "dataset_version": dataset.version,
                "dataset_seed": dataset.seed,
                "corpus_fingerprint": dataset.corpus_fingerprint,
                "question_count": len(dataset.pairs),
                "approach": result.approach,
                "cutoffs": list(cutoffs),
                "max_k": max_k,
                "filters": {},
                "embedding_model": settings.openai_embedding_model,
                "reranker_model": (
                    DEFAULT_CROSS_ENCODER_MODEL if result.approach.endswith("_rerank") else None
                ),
                "exploratory": result.approach not in APPROACH_ORDER,
            }
            text_cap = _text_cap_from_approach(result.approach)
            if text_cap is not None:
                config["text_candidate_cap"] = text_cap
                config["vector_candidate_limit"] = max_k * VECTOR_CANDIDATE_MULTIPLIER
            metrics = {
                **result.metrics,
                "elapsed_seconds": result.elapsed_seconds,
            }
            row = connection.execute(
                statement,
                {
                    "config": json.dumps(config, sort_keys=True),
                    "metrics": json.dumps(metrics, sort_keys=True),
                },
            ).one()
            run_ids[result.approach] = int(row[0])
    return run_ids


def _text_cap_approach(text_cap: int) -> str:
    return f"hybrid_rrf_text_cap_{text_cap}"


def _text_cap_from_approach(approach: str) -> int | None:
    prefix = "hybrid_rrf_text_cap_"
    if not approach.startswith(prefix):
        return None
    raw_cap = approach.removeprefix(prefix).removesuffix("_rerank")
    return int(raw_cap)


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(cutoffs)))
    if not normalized:
        raise ValueError("At least one retrieval cutoff is required.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in normalized
    ):
        raise ValueError("Retrieval cutoffs must be positive integers.")
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-persist", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the retrieval benchmark and print its machine-readable summary."""
    arguments = _parse_args()
    summary = evaluate_retrieval(
        dataset_path=arguments.dataset,
        persist=not arguments.no_persist,
    )
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
