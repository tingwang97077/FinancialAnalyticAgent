"""Pluggable post-retrieval rerankers."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from typing import Protocol, cast

from ffa.retrieval.base import Chunk, validate_search_request

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Reranker(Protocol):
    """Interchangeable post-retrieval reranker contract."""

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """Return the most relevant chunks in reranked order."""
        ...


class CrossEncoderModel(Protocol):
    """Subset of the sentence-transformers CrossEncoder API used here."""

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        show_progress_bar: bool,
    ) -> Sequence[float]:
        """Score query-document pairs."""
        ...


class CrossEncoderReranker:
    """Lazily load one local cross-encoder and reuse it across requests."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        *,
        model_factory: Callable[[str], CrossEncoderModel] | None = None,
    ) -> None:
        """Configure model identity and an optional injectable model factory."""
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            raise ValueError("Cross-encoder model name must not be empty.")
        self._model_name = normalized_model_name
        self._model_factory = model_factory or _default_model_factory
        self._model: CrossEncoderModel | None = None
        self._model_lock = threading.Lock()

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """Score query/chunk pairs and return the top_n copies."""
        normalized_query = validate_search_request(query, top_n)
        if not chunks:
            return []
        model = self._get_model()
        scores = list(
            model.predict(
                [(normalized_query, chunk["text"]) for chunk in chunks],
                show_progress_bar=False,
            )
        )
        if len(scores) != len(chunks):
            raise RuntimeError("Cross-encoder returned an unexpected score count.")

        reranked: list[tuple[int, Chunk]] = []
        for original_rank, (chunk, raw_score) in enumerate(zip(chunks, scores, strict=True)):
            score = float(raw_score)
            if not math.isfinite(score):
                raise RuntimeError("Cross-encoder returned a non-finite score.")
            ranked_chunk = Chunk(**chunk)
            ranked_chunk["rerank_score"] = score
            ranked_chunk["score"] = score
            reranked.append((original_rank, ranked_chunk))
        reranked.sort(key=lambda item: (-item[1]["score"], item[0]))
        return [chunk for _, chunk in reranked[:top_n]]

    def preload(self) -> None:
        """Load model weights without running or changing the reranking algorithm."""
        self._get_model()

    def _get_model(self) -> CrossEncoderModel:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._model_factory(self._model_name)
        return self._model


class NoOpReranker:
    """Explicitly disable reranking while preserving the shared interface."""

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """Return retrieval order unchanged up to top_n."""
        validate_search_request(query, top_n)
        return [Chunk(**chunk) for chunk in chunks[:top_n]]


def _default_model_factory(model_name: str) -> CrossEncoderModel:
    try:
        import pyarrow
    except ImportError:
        pass
    else:
        if not hasattr(pyarrow, "PyExtensionType"):
            pyarrow.PyExtensionType = pyarrow.ExtensionType

    from sentence_transformers import CrossEncoder

    return cast(CrossEncoderModel, CrossEncoder(model_name))
