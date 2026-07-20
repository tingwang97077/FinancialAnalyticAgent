"""Reciprocal Rank Fusion over text and vector retrieval legs."""

from __future__ import annotations

from ffa.retrieval.base import Chunk, SearchIndex, validate_search_request


class HybridSearchIndex:
    """Fuse incomparable backend scores using only their result ranks."""

    def __init__(
        self,
        text_index: SearchIndex,
        vector_index: SearchIndex,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
    ) -> None:
        """Initialize independently replaceable retrieval legs and RRF controls."""
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
            raise ValueError("RRF k must be a positive integer.")
        if (
            isinstance(candidate_multiplier, bool)
            or not isinstance(candidate_multiplier, int)
            or candidate_multiplier <= 0
        ):
            raise ValueError("Candidate multiplier must be a positive integer.")
        self._text_index = text_index
        self._vector_index = vector_index
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier

    def search(
        self,
        query: str,
        *,
        filters: dict[str, object],
        k: int,
    ) -> list[Chunk]:
        """Return rank-fused chunks without comparing raw backend scores."""
        normalized_query = validate_search_request(query, k)
        candidate_k = k * self._candidate_multiplier
        text_results = self._text_index.search(
            normalized_query,
            filters=filters,
            k=candidate_k,
        )
        vector_results = self._vector_index.search(
            normalized_query,
            filters=filters,
            k=candidate_k,
        )

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
                fused_chunk["rrf_score"] += 1 / (self._rrf_k + rank)
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
