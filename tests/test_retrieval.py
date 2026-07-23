"""Tests for text, vector, hybrid, and reranking retrieval components."""

from __future__ import annotations

import inspect
from collections.abc import Sequence

import pytest

from ffa.config import Settings
from ffa.retrieval.base import Chunk, SearchIndex, build_filter_sql
from ffa.retrieval.hybrid import HybridSearchIndex
from ffa.retrieval.rerank import CrossEncoderReranker, NoOpReranker
from ffa.retrieval.text_search import TextSearchIndex
from ffa.retrieval.vector_search import VectorSearchIndex


class FakeResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> FakeResult:
        return self

    def all(self) -> list[dict[str, object]]:
        return self._rows


class FakeConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(
        self,
        statement: object,
        parameters: dict[str, object],
    ) -> FakeResult:
        sql = str(statement)
        self.calls.append((sql, parameters))
        return FakeResult([] if "set_config" in sql else self._rows)


class FakeEngine:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.connection = FakeConnection(rows)

    def connect(self) -> FakeConnection:
        return self.connection


class FakeEmbeddingProvider:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls: list[tuple[list[str], int]] = []

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        dimensions: int,
    ) -> list[list[float]]:
        self.calls.append((list(texts), dimensions))
        return [self._vector]


class StubIndex:
    def __init__(self, results: list[Chunk]) -> None:
        self._results = results
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def search(
        self,
        query: str,
        *,
        filters: dict[str, object],
        k: int,
    ) -> list[Chunk]:
        self.calls.append((query, filters, k))
        return self._results[:k]


class FakeCrossEncoder:
    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        show_progress_bar: bool,
    ) -> list[float]:
        assert show_progress_bar is False
        return [0.9 if "services" in text.lower() else 0.1 for _, text in sentences]


def test_search_implementations_share_exact_public_signature() -> None:
    expected = inspect.signature(SearchIndex.search)
    expected_shape = [
        (parameter.name, parameter.kind) for parameter in expected.parameters.values()
    ]

    for implementation in (
        TextSearchIndex.search,
        VectorSearchIndex.search,
        HybridSearchIndex.search,
    ):
        actual_shape = [
            (parameter.name, parameter.kind)
            for parameter in inspect.signature(implementation).parameters.values()
        ]
        assert actual_shape == expected_shape


def test_text_search_uses_gin_expression_and_all_filters_in_sql() -> None:
    engine = FakeEngine([_sql_row(1, score=0.75)])
    index = TextSearchIndex(engine=engine)  # type: ignore[arg-type]

    results = index.search(
        "services revenue growth",
        filters={
            "ticker": "aapl",
            "fiscal_year": 2025,
            "fiscal_period": "fy",
            "section": "MD&A",
        },
        k=5,
    )

    sql, parameters = engine.connection.calls[0]
    assert "tsvector_to_array(to_tsvector('english', :query))" in sql
    assert "string_agg(quote_literal(term), ' | ' ORDER BY term)" in sql
    assert "to_tsquery(" in sql
    assert "plainto_tsquery" not in sql
    assert "websearch_to_tsquery" not in sql
    assert "sq.value IS NOT NULL" in sql
    assert "dc.text_tsv @@ sq.value" in sql
    assert "ts_rank_cd(dc.text_tsv, sq.value)" in sql
    assert "dc.ticker = :filter_ticker" in sql
    assert "dc.fiscal_year = :filter_fiscal_year" in sql
    assert "dc.fiscal_period = :filter_fiscal_period" in sql
    assert "dc.section = :filter_section" in sql
    assert parameters == {
        "query": "services revenue growth",
        "limit": 5,
        "filter_ticker": "AAPL",
        "filter_fiscal_year": 2025,
        "filter_fiscal_period": "FY",
        "filter_section": "MD&A",
    }
    assert results[0]["id"] == 1
    assert results[0]["score"] == pytest.approx(0.75)


def test_vector_search_uses_cosine_hnsw_operator_filters_and_query_model() -> None:
    engine = FakeEngine([_sql_row(2, score=0.92)])
    provider = FakeEmbeddingProvider([1.0, 0.0, 0.0])
    index = VectorSearchIndex(
        engine=engine,  # type: ignore[arg-type]
        embedding_provider=provider,
        settings=Settings(_env_file=None, embedding_dim=3),
        ef_search=80,
    )

    results = index.search(
        "services revenue growth",
        filters={
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "fiscal_period": "FY",
            "section": "MD&A",
        },
        k=4,
    )

    setting_sql, setting_parameters = engine.connection.calls[0]
    search_sql, search_parameters = engine.connection.calls[1]
    assert "set_config('hnsw.ef_search', :ef_search, true)" in setting_sql
    assert setting_parameters == {"ef_search": "80"}
    assert search_sql.count("<=> CAST(:query_embedding AS vector)") == 2
    assert "dc.ticker = :filter_ticker" in search_sql
    assert "dc.fiscal_year = :filter_fiscal_year" in search_sql
    assert "dc.fiscal_period = :filter_fiscal_period" in search_sql
    assert "dc.section = :filter_section" in search_sql
    assert search_parameters["query_embedding"] == "[1,0,0]"
    assert provider.calls == [(["services revenue growth"], 3)]
    assert results[0]["score"] == pytest.approx(0.92)


def test_vector_search_reuses_one_prepared_embedding_across_filter_passes() -> None:
    engine = FakeEngine([_sql_row(2, score=0.92)])
    provider = FakeEmbeddingProvider([1.0, 0.0, 0.0])
    index = VectorSearchIndex(
        engine=engine,  # type: ignore[arg-type]
        embedding_provider=provider,
        settings=Settings(_env_file=None, embedding_dim=3),
    )

    prepared_query = index.prepare_query("services revenue growth")
    first = index.search_prepared(
        prepared_query,
        filters={"ticker": ["AAPL"], "fiscal_year": [2024]},
        k=4,
    )
    second = index.search_prepared(
        prepared_query,
        filters={"ticker": ["AAPL"]},
        k=4,
    )

    assert provider.calls == [(["services revenue growth"], 3)]
    assert first == second
    search_calls = [call for call in engine.connection.calls if "set_config" not in call[0]]
    assert len(search_calls) == 2
    assert search_calls[0][1]["query_embedding"] == search_calls[1][1]["query_embedding"]


def test_filter_builder_rejects_unknown_or_invalid_filters() -> None:
    with pytest.raises(ValueError, match="Unsupported retrieval filters"):
        build_filter_sql({"accession_no": "unsafe"})
    with pytest.raises(ValueError, match="fiscal_year"):
        build_filter_sql({"fiscal_year": "2025"})
    with pytest.raises(ValueError, match="fiscal_period"):
        build_filter_sql({"fiscal_period": "H1"})
    with pytest.raises(ValueError, match="must not be empty"):
        build_filter_sql({"section": []})


def test_text_search_parameterizes_multiple_years_and_sections() -> None:
    engine = FakeEngine(
        [
            _sql_row(1, score=0.8, fiscal_year=2023, section="MD&A"),
            _sql_row(2, score=0.7, fiscal_year=2024, section="Notes"),
        ]
    )
    index = TextSearchIndex(engine=engine)  # type: ignore[arg-type]

    results = index.search(
        "net income change",
        filters={
            "ticker": ["AAPL"],
            "fiscal_year": [2023, 2024],
            "section": ["MD&A", "Notes"],
        },
        k=5,
    )

    sql, parameters = engine.connection.calls[0]
    assert "dc.ticker IN (:filter_ticker_0)" in sql
    assert "dc.fiscal_year IN (:filter_fiscal_year_0, :filter_fiscal_year_1)" in sql
    assert "dc.section IN (:filter_section_0, :filter_section_1)" in sql
    assert parameters == {
        "query": "net income change",
        "limit": 5,
        "filter_ticker_0": "AAPL",
        "filter_fiscal_year_0": 2023,
        "filter_fiscal_year_1": 2024,
        "filter_section_0": "MD&A",
        "filter_section_1": "Notes",
    }
    assert {chunk["fiscal_year"] for chunk in results} == {2023, 2024}
    assert {chunk["section"] for chunk in results} == {"MD&A", "Notes"}


def test_hybrid_search_fuses_ranks_not_raw_scores_and_forwards_filters() -> None:
    text_index = StubIndex([_chunk(1, 1000.0), _chunk(2, 0.01)])
    vector_index = StubIndex([_chunk(2, -500.0), _chunk(3, 9999.0)])
    index = HybridSearchIndex(
        text_index,
        vector_index,
        rrf_k=10,
        candidate_multiplier=3,
    )
    filters = {"ticker": "AAPL", "fiscal_year": 2025}

    results = index.search("services revenue growth", filters=filters, k=3)

    assert [chunk["id"] for chunk in results] == [2, 1, 3]
    assert results[0]["rrf_score"] == pytest.approx(1 / 12 + 1 / 11)
    assert results[0]["text_score"] == pytest.approx(0.01)
    assert results[0]["vector_score"] == pytest.approx(-500.0)
    assert text_index.calls == [("services revenue growth", filters, 9)]
    assert vector_index.calls == [("services revenue growth", filters, 9)]


def test_cross_encoder_loads_once_and_reranks_copies() -> None:
    factory_calls: list[str] = []

    def factory(model_name: str) -> FakeCrossEncoder:
        factory_calls.append(model_name)
        return FakeCrossEncoder()

    reranker = CrossEncoderReranker("test-cross-encoder", model_factory=factory)
    chunks = [
        _chunk(1, 0.8, text="Tariffs could affect product costs."),
        _chunk(2, 0.7, text="Services revenue growth reflected cloud demand."),
    ]

    first = reranker.rerank("services revenue growth", chunks, 2)
    second = reranker.rerank("services revenue growth", chunks, 1)

    assert [chunk["id"] for chunk in first] == [2, 1]
    assert first[0]["rerank_score"] == pytest.approx(0.9)
    assert second[0]["id"] == 2
    assert factory_calls == ["test-cross-encoder"]
    assert "rerank_score" not in chunks[0]


def test_cross_encoder_preload_warms_the_same_model_used_for_reranking() -> None:
    factory_calls: list[str] = []

    def factory(model_name: str) -> FakeCrossEncoder:
        factory_calls.append(model_name)
        return FakeCrossEncoder()

    reranker = CrossEncoderReranker("test-cross-encoder", model_factory=factory)

    reranker.preload()
    result = reranker.rerank("services", [_chunk(1, 0.5, text="Services grew.")], 1)

    assert factory_calls == ["test-cross-encoder"]
    assert [chunk["id"] for chunk in result] == [1]


def test_noop_reranker_preserves_retrieval_order() -> None:
    chunks = [_chunk(1, 0.8), _chunk(2, 0.7)]

    results = NoOpReranker().rerank("query", chunks, 1)

    assert [chunk["id"] for chunk in results] == [1]


def _sql_row(
    chunk_id: int,
    *,
    score: float,
    fiscal_year: int = 2025,
    section: str = "MD&A",
) -> dict[str, object]:
    return {
        "id": chunk_id,
        "accession_no": "0000320193-25-000079",
        "cik": 320193,
        "ticker": "AAPL",
        "fiscal_year": fiscal_year,
        "fiscal_period": "FY",
        "section": section,
        "chunk_index": chunk_id,
        "text": "Services revenue growth reflected cloud demand.",
        "token_count": 7,
        "source_url": "https://www.sec.gov/filing.htm",
        "score": score,
    }


def _chunk(chunk_id: int, score: float, *, text: str = "Narrative filing text.") -> Chunk:
    row = _sql_row(chunk_id, score=score)
    row["text"] = text
    return Chunk(**row)  # type: ignore[typeddict-item]
