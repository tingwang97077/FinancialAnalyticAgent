"""Shared retrieval contracts and safe metadata-filter helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import NotRequired, Protocol, TypedDict, cast

from sqlalchemy import Engine
from sqlalchemy.engine import Connection

from ffa.common.entities import normalize_ticker

type DatabaseBind = Engine | Connection

CHUNK_SELECT_COLUMNS = """
    dc.id,
    dc.accession_no,
    dc.cik,
    dc.ticker,
    dc.fiscal_year,
    dc.fiscal_period,
    dc.section,
    dc.chunk_index,
    dc.text,
    dc.token_count,
    dc.source_url
"""

_ALLOWED_FILTERS = frozenset({"ticker", "fiscal_year", "fiscal_period", "section"})
_FISCAL_PERIODS = frozenset({"FY", "Q1", "Q2", "Q3", "Q4"})


class Chunk(TypedDict):
    """Retrieved filing chunk with comparable or backend-specific score metadata."""

    id: int
    accession_no: str
    cik: int
    ticker: str
    fiscal_year: int | None
    fiscal_period: str | None
    section: str
    chunk_index: int
    text: str
    token_count: int
    source_url: str
    score: float
    text_score: NotRequired[float]
    vector_score: NotRequired[float]
    rrf_score: NotRequired[float]
    rerank_score: NotRequired[float]


class SearchIndex(Protocol):
    """Interchangeable retrieval backend contract."""

    def search(
        self,
        query: str,
        *,
        filters: dict[str, object],
        k: int,
    ) -> list[Chunk]:
        """Return up to k chunks matching query and SQL metadata filters."""
        ...


def validate_search_request(query: str, k: int) -> str:
    """Validate common search arguments and return normalized query text."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Search query must not be empty.")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("Search result count k must be a positive integer.")
    return normalized_query


def build_filter_sql(
    filters: dict[str, object],
    *,
    table_alias: str = "dc",
) -> tuple[str, dict[str, object]]:
    """Build parameterized SQL predicates for the supported metadata filters."""
    unknown_filters = set(filters).difference(_ALLOWED_FILTERS)
    if unknown_filters:
        names = ", ".join(sorted(unknown_filters))
        raise ValueError(f"Unsupported retrieval filters: {names}.")

    clauses: list[str] = []
    parameters: dict[str, object] = {}
    if "ticker" in filters:
        raw_tickers, is_collection = _filter_values(filters["ticker"], name="ticker")
        tickers: list[object] = []
        for ticker in raw_tickers:
            if not isinstance(ticker, str):
                raise ValueError("ticker filter must contain only strings.")
            tickers.append(normalize_ticker(ticker))
        _append_filter_clause(
            clauses,
            parameters,
            column=f"{table_alias}.ticker",
            parameter_name="filter_ticker",
            values=tickers,
            is_collection=is_collection,
        )

    if "fiscal_year" in filters:
        raw_years, is_collection = _filter_values(
            filters["fiscal_year"],
            name="fiscal_year",
        )
        fiscal_years: list[object] = []
        include_null = False
        for fiscal_year in raw_years:
            if fiscal_year is None:
                include_null = True
                continue
            if (
                isinstance(fiscal_year, bool)
                or not isinstance(fiscal_year, int)
                or fiscal_year <= 0
            ):
                raise ValueError("fiscal_year filter must contain only positive integers or null.")
            fiscal_years.append(fiscal_year)
        _append_filter_clause(
            clauses,
            parameters,
            column=f"{table_alias}.fiscal_year",
            parameter_name="filter_fiscal_year",
            values=fiscal_years,
            is_collection=is_collection,
            include_null=include_null,
        )

    if "fiscal_period" in filters:
        raw_periods, is_collection = _filter_values(
            filters["fiscal_period"],
            name="fiscal_period",
        )
        fiscal_periods: list[object] = []
        include_null = False
        for fiscal_period in raw_periods:
            if fiscal_period is None:
                include_null = True
                continue
            if not isinstance(fiscal_period, str):
                raise ValueError("fiscal_period filter must contain only strings or null.")
            normalized_period = fiscal_period.strip().upper()
            if normalized_period not in _FISCAL_PERIODS:
                raise ValueError("fiscal_period filter values must be FY or Q1 through Q4.")
            fiscal_periods.append(normalized_period)
        _append_filter_clause(
            clauses,
            parameters,
            column=f"{table_alias}.fiscal_period",
            parameter_name="filter_fiscal_period",
            values=fiscal_periods,
            is_collection=is_collection,
            include_null=include_null,
        )

    if "section" in filters:
        raw_sections, is_collection = _filter_values(filters["section"], name="section")
        sections: list[object] = []
        for section in raw_sections:
            if not isinstance(section, str) or not section.strip():
                raise ValueError("section filter must contain only non-empty strings.")
            sections.append(section.strip())
        _append_filter_clause(
            clauses,
            parameters,
            column=f"{table_alias}.section",
            parameter_name="filter_section",
            values=sections,
            is_collection=is_collection,
        )

    suffix = "" if not clauses else " AND " + " AND ".join(clauses)
    return suffix, parameters


def _filter_values(value: object, *, name: str) -> tuple[list[object], bool]:
    """Return one scalar or a non-empty ordered collection of filter values."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
        if not values:
            raise ValueError(f"{name} filter list must not be empty.")
        return values, True
    return [value], False


def _append_filter_clause(
    clauses: list[str],
    parameters: dict[str, object],
    *,
    column: str,
    parameter_name: str,
    values: list[object],
    is_collection: bool,
    include_null: bool = False,
) -> None:
    """Append an equality or parameterized IN predicate with optional null matching."""
    unique_values = list(dict.fromkeys(values))
    predicates: list[str] = []
    if unique_values:
        if is_collection:
            placeholders: list[str] = []
            for index, value in enumerate(unique_values):
                indexed_name = f"{parameter_name}_{index}"
                placeholders.append(f":{indexed_name}")
                parameters[indexed_name] = value
            predicates.append(f"{column} IN ({', '.join(placeholders)})")
        else:
            parameters[parameter_name] = unique_values[0]
            predicates.append(f"{column} = :{parameter_name}")
    if include_null:
        predicates.append(f"{column} IS NULL")
    if not predicates:
        raise ValueError(f"{parameter_name.removeprefix('filter_')} filter has no values.")
    clause = " OR ".join(predicates)
    clauses.append(f"({clause})" if len(predicates) > 1 else clause)


def row_to_chunk(row: Mapping[str, object]) -> Chunk:
    """Convert one SQLAlchemy row mapping into the common chunk contract."""
    return Chunk(
        id=int(row["id"]),
        accession_no=str(row["accession_no"]),
        cik=int(row["cik"]),
        ticker=str(row["ticker"]),
        fiscal_year=None if row["fiscal_year"] is None else int(row["fiscal_year"]),
        fiscal_period=(None if row["fiscal_period"] is None else str(row["fiscal_period"])),
        section=str(row["section"]),
        chunk_index=int(row["chunk_index"]),
        text=str(row["text"]),
        token_count=int(row["token_count"]),
        source_url=str(row["source_url"]),
        score=float(row["score"]),
    )


@contextmanager
def connection_scope(bind: DatabaseBind) -> Iterator[Connection]:
    """Use an injected connection or open a short-lived engine connection."""
    connect = getattr(bind, "connect", None)
    if callable(connect):
        with connect() as connection:
            yield cast(Connection, connection)
        return
    yield cast(Connection, bind)
