"""Idempotent PostgreSQL loader for embedded filing chunks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection

from ffa.common.db import create_rw_engine
from ffa.config import Settings, get_settings
from ffa.ingestion.unstructured.embed import EmbeddedChunk

UNSTRUCTURED_SOURCE = "SEC_EDGAR_UNSTRUCTURED"

_UPSERT_COMPANY_SQL = text(
    """
    INSERT INTO companies (cik, ticker, name, updated_at)
    VALUES (:cik, :ticker, :company_name, now())
    ON CONFLICT (cik) DO UPDATE SET
        ticker = EXCLUDED.ticker,
        name = EXCLUDED.name,
        updated_at = now()
    """
)

_UPSERT_FILING_SQL = text(
    """
    INSERT INTO filings (
        accession_no,
        cik,
        form_type,
        filing_date,
        period_of_report,
        primary_doc_url
    ) VALUES (
        :accession_no,
        :cik,
        :form_type,
        :filing_date,
        :period_of_report,
        :source_url
    )
    ON CONFLICT (accession_no) DO UPDATE SET
        cik = EXCLUDED.cik,
        form_type = EXCLUDED.form_type,
        filing_date = EXCLUDED.filing_date,
        period_of_report = EXCLUDED.period_of_report,
        primary_doc_url = EXCLUDED.primary_doc_url
    """
)

_UPSERT_CHUNK_SQL = text(
    """
    INSERT INTO doc_chunks (
        accession_no,
        cik,
        ticker,
        fiscal_year,
        fiscal_period,
        section,
        chunk_index,
        text,
        token_count,
        embedding,
        source_url
    ) VALUES (
        :accession_no,
        :cik,
        :ticker,
        :fiscal_year,
        :fiscal_period,
        :section,
        :chunk_index,
        :text,
        :token_count,
        CAST(:embedding AS vector),
        :source_url
    )
    ON CONFLICT (accession_no, section, chunk_index) DO UPDATE SET
        cik = EXCLUDED.cik,
        ticker = EXCLUDED.ticker,
        fiscal_year = EXCLUDED.fiscal_year,
        fiscal_period = EXCLUDED.fiscal_period,
        text = EXCLUDED.text,
        token_count = EXCLUDED.token_count,
        embedding = EXCLUDED.embedding,
        source_url = EXCLUDED.source_url
    """
)

_UPSERT_STATE_SQL = text(
    """
    INSERT INTO ingestion_state (
        source,
        cik,
        last_accession,
        last_filing_date,
        updated_at
    ) VALUES (
        :source,
        :cik,
        :last_accession,
        :last_filing_date,
        now()
    )
    ON CONFLICT (source, cik) DO UPDATE SET
        last_accession = EXCLUDED.last_accession,
        last_filing_date = EXCLUDED.last_filing_date,
        updated_at = now()
    WHERE ingestion_state.last_filing_date IS NULL
       OR (EXCLUDED.last_filing_date, EXCLUDED.last_accession) >=
          (ingestion_state.last_filing_date, COALESCE(ingestion_state.last_accession, ''))
    """
)


def load_chunks(
    chunks: Sequence[Mapping[str, object]],
    *,
    engine: Engine | None = None,
    settings: Settings | None = None,
) -> int:
    """Upsert embedded chunks and advance ingestion cursors atomically."""
    if not chunks:
        return 0
    resolved_settings = settings or get_settings()
    prepared = [_coerce_chunk(chunk, resolved_settings.embedding_dim) for chunk in chunks]
    owned_engine = engine is None
    resolved_engine = engine or create_rw_engine(resolved_settings.database_url)
    try:
        with resolved_engine.begin() as connection:
            return _load_transaction(connection, prepared)
    finally:
        if owned_engine:
            resolved_engine.dispose()


def _load_transaction(connection: Connection, chunks: list[EmbeddedChunk]) -> int:
    chunks_by_cik: dict[int, list[EmbeddedChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_cik[chunk["cik"]].append(chunk)

    loaded_count = 0
    for cik, company_chunks in chunks_by_cik.items():
        filings = _group_filings(company_chunks)
        latest_company_chunk = max(
            company_chunks,
            key=lambda chunk: (chunk["filing_date"], chunk["accession_no"]),
        )
        connection.execute(_UPSERT_COMPANY_SQL, _params(latest_company_chunk))

        for filing_chunks in sorted(
            filings.values(),
            key=lambda rows: (rows[0]["filing_date"], rows[0]["accession_no"]),
        ):
            representative = filing_chunks[0]
            connection.execute(_UPSERT_FILING_SQL, _params(representative))
            for chunk in sorted(
                filing_chunks,
                key=lambda row: (row["section"], row["chunk_index"]),
            ):
                connection.execute(_UPSERT_CHUNK_SQL, _params(chunk))
                loaded_count += 1

        latest_filing = max(
            (rows[0] for rows in filings.values()),
            key=lambda chunk: (chunk["filing_date"], chunk["accession_no"]),
        )
        connection.execute(
            _UPSERT_STATE_SQL,
            {
                "source": UNSTRUCTURED_SOURCE,
                "cik": cik,
                "last_accession": latest_filing["accession_no"],
                "last_filing_date": latest_filing["filing_date"],
            },
        )
    return loaded_count


def _group_filings(chunks: list[EmbeddedChunk]) -> dict[str, list[EmbeddedChunk]]:
    filings: dict[str, list[EmbeddedChunk]] = defaultdict(list)
    identities: dict[str, tuple[object, ...]] = {}
    for chunk in chunks:
        accession_no = chunk["accession_no"]
        identity = (
            chunk["cik"],
            chunk["ticker"],
            chunk["company_name"],
            chunk["form_type"],
            chunk["filing_date"],
            chunk["period_of_report"],
            chunk["source_url"],
        )
        existing = identities.setdefault(accession_no, identity)
        if existing != identity:
            raise ValueError("Chunks for one accession must share filing metadata.")
        filings[accession_no].append(chunk)
    return filings


def _coerce_chunk(chunk: Mapping[str, object], embedding_dim: int) -> EmbeddedChunk:
    required_strings = (
        "ticker",
        "company_name",
        "accession_no",
        "form_type",
        "section",
        "text",
        "source_url",
    )
    values: dict[str, Any] = dict(chunk)
    for field in required_strings:
        value = values.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string.")
        values[field] = value.strip()

    cik = values.get("cik")
    chunk_index = values.get("chunk_index")
    token_count = values.get("token_count")
    if isinstance(cik, bool) or not isinstance(cik, int) or cik <= 0:
        raise ValueError("cik must be a positive integer.")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("chunk_index must be a non-negative integer.")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise ValueError("token_count must be a positive integer.")

    fiscal_year = values.get("fiscal_year")
    fiscal_period = values.get("fiscal_period")
    if fiscal_year is not None and (
        isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int) or fiscal_year <= 0
    ):
        raise ValueError("fiscal_year must be a positive integer when present.")
    if fiscal_period is not None and fiscal_period not in {"FY", "Q1", "Q2", "Q3", "Q4"}:
        raise ValueError("fiscal_period must be FY or Q1 through Q4 when present.")

    raw_embedding = values.get("embedding")
    if not isinstance(raw_embedding, Sequence) or isinstance(raw_embedding, (str, bytes)):
        raise ValueError("embedding must be a numeric sequence.")
    embedding: list[float] = []
    for value in raw_embedding:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("embedding values must be numeric.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("embedding values must be finite.")
        embedding.append(numeric_value)
    if len(embedding) != embedding_dim:
        raise ValueError(f"embedding must contain exactly {embedding_dim} values.")

    return EmbeddedChunk(
        ticker=str(values["ticker"]),
        cik=cik,
        company_name=str(values["company_name"]),
        accession_no=str(values["accession_no"]),
        form_type=str(values["form_type"]),
        filing_date=_coerce_date(values.get("filing_date"), "filing_date"),
        period_of_report=_optional_date(values.get("period_of_report"), "period_of_report"),
        fiscal_year=fiscal_year,
        fiscal_period=None if fiscal_period is None else str(fiscal_period),
        section=str(values["section"]),
        chunk_index=chunk_index,
        text=str(values["text"]),
        token_count=token_count,
        source_url=str(values["source_url"]),
        embedding=embedding,
    )


def _params(chunk: EmbeddedChunk) -> dict[str, Any]:
    params = dict(chunk)
    params["embedding"] = (
        "[" + ",".join(format(value, ".17g") for value in chunk["embedding"]) + "]"
    )
    return params


def _coerce_date(value: object, field: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field} must be a valid ISO date.") from exc
    raise ValueError(f"{field} must be a date or ISO date string.")


def _optional_date(value: object, field: str) -> date | None:
    if value is None or value == "":
        return None
    return _coerce_date(value, field)
