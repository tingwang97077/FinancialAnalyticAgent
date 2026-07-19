"""Idempotent PostgreSQL loader for structured financial facts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection

from ffa.common.db import create_rw_engine
from ffa.config import Settings, get_settings
from ffa.ingestion.structured.normalize import FinancialFactRow

STRUCTURED_SOURCE = "SEC_EDGAR_STRUCTURED"

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
        :period_end,
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

_LATEST_FACT_DATE_SQL = text(
    """
    SELECT max(filing_date)
    FROM financial_facts
    WHERE cik = :cik
      AND metric = :metric
      AND fiscal_year = :fiscal_year
      AND fiscal_period = :fiscal_period
    """
)

_DELETE_SUPERSEDED_FACTS_SQL = text(
    """
    DELETE FROM financial_facts
    WHERE cik = :cik
      AND metric = :metric
      AND fiscal_year = :fiscal_year
      AND fiscal_period = :fiscal_period
      AND (
          filing_date < :filing_date
          OR (filing_date = :filing_date AND taxonomy_tag <> :taxonomy_tag)
      )
    """
)

_UPSERT_FACT_SQL = text(
    """
    INSERT INTO financial_facts (
        cik,
        ticker,
        metric,
        taxonomy_tag,
        unit,
        fiscal_year,
        fiscal_period,
        period_start,
        period_end,
        value,
        form_type,
        filing_date,
        accession_no,
        source_url
    ) VALUES (
        :cik,
        :ticker,
        :metric,
        :taxonomy_tag,
        :unit,
        :fiscal_year,
        :fiscal_period,
        :period_start,
        :period_end,
        :value,
        :form_type,
        :filing_date,
        :accession_no,
        :source_url
    )
    ON CONFLICT (
        cik,
        metric,
        fiscal_year,
        fiscal_period,
        taxonomy_tag,
        filing_date
    ) DO UPDATE SET
        ticker = EXCLUDED.ticker,
        unit = EXCLUDED.unit,
        period_start = EXCLUDED.period_start,
        period_end = EXCLUDED.period_end,
        value = EXCLUDED.value,
        form_type = EXCLUDED.form_type,
        accession_no = EXCLUDED.accession_no,
        source_url = EXCLUDED.source_url
    """
)

_SELECT_STATE_SQL = text(
    """
    SELECT last_filing_date
    FROM ingestion_state
    WHERE source = :source AND cik = :cik
    FOR UPDATE
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
       OR EXCLUDED.last_filing_date >= ingestion_state.last_filing_date
    """
)


def load_facts(
    rows: Sequence[Mapping[str, Any]],
    *,
    engine: Engine | None = None,
    settings: Settings | None = None,
) -> int:
    """Load canonical facts and advance structured ingestion state atomically.

    Existing state excludes filings already processed. For each canonical fiscal
    key, a newer filing supersedes older database rows before the selected row is
    inserted with ``ON CONFLICT DO UPDATE``.

    Args:
        rows: Canonical rows produced by ``normalize_facts``.
        engine: Optional injected read-write SQLAlchemy engine.
        settings: Optional settings used to construct the engine.

    Returns:
        Number of fact rows inserted or updated during this call.
    """
    prepared_rows = _prepare_rows(rows)
    if not prepared_rows:
        return 0

    owned_engine = engine is None
    resolved_engine = engine or create_rw_engine((settings or get_settings()).database_url)
    try:
        with resolved_engine.begin() as connection:
            return _load_transaction(connection, prepared_rows)
    finally:
        if owned_engine:
            resolved_engine.dispose()


def _load_transaction(connection: Connection, rows: list[FinancialFactRow]) -> int:
    rows_by_cik: dict[int, list[FinancialFactRow]] = defaultdict(list)
    for row in rows:
        rows_by_cik[row["cik"]].append(row)

    loaded_count = 0
    for cik, company_rows in rows_by_cik.items():
        company_rows.sort(key=lambda row: (row["filing_date"], row["accession_no"]))
        connection.execute(_UPSERT_COMPANY_SQL, _params(company_rows[-1]))

        last_filing_date = connection.execute(
            _SELECT_STATE_SQL,
            {"source": STRUCTURED_SOURCE, "cik": cik},
        ).scalar_one_or_none()
        incremental_rows = [
            row
            for row in company_rows
            if last_filing_date is None or row["filing_date"] > last_filing_date
        ]

        for row in incremental_rows:
            params = _params(row)
            newest_stored_date = connection.execute(
                _LATEST_FACT_DATE_SQL,
                params,
            ).scalar_one_or_none()
            if newest_stored_date is not None and newest_stored_date > row["filing_date"]:
                continue

            connection.execute(_UPSERT_FILING_SQL, params)
            connection.execute(_DELETE_SUPERSEDED_FACTS_SQL, params)
            connection.execute(_UPSERT_FACT_SQL, params)
            loaded_count += 1

        latest_row = max(
            company_rows,
            key=lambda row: (row["filing_date"], row["accession_no"]),
        )
        connection.execute(
            _UPSERT_STATE_SQL,
            {
                "source": STRUCTURED_SOURCE,
                "cik": cik,
                "last_accession": latest_row["accession_no"],
                "last_filing_date": latest_row["filing_date"],
            },
        )

    return loaded_count


def _prepare_rows(rows: Sequence[Mapping[str, Any]]) -> list[FinancialFactRow]:
    deduplicated: dict[tuple[int, str, int, str], FinancialFactRow] = {}
    for raw_row in rows:
        row = _coerce_row(raw_row)
        key = (row["cik"], row["metric"], row["fiscal_year"], row["fiscal_period"])
        current = deduplicated.get(key)
        if current is None or (row["filing_date"], row["accession_no"]) > (
            current["filing_date"],
            current["accession_no"],
        ):
            deduplicated[key] = row
    return list(deduplicated.values())


def _coerce_row(row: Mapping[str, Any]) -> FinancialFactRow:
    fiscal_period = _required_string(row, "fiscal_period")
    if fiscal_period not in {"FY", "Q1", "Q2", "Q3", "Q4"}:
        raise ValueError("fiscal_period must be FY or Q1 through Q4.")
    cik = row.get("cik")
    fiscal_year = row.get("fiscal_year")
    value = row.get("value")
    if isinstance(cik, bool) or not isinstance(cik, int) or cik <= 0:
        raise ValueError("cik must be a positive integer.")
    if isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int) or fiscal_year <= 0:
        raise ValueError("fiscal_year must be a positive integer.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be numeric.")

    period_start_value = row.get("period_start")
    period_start = (
        None if period_start_value is None else _coerce_date(period_start_value, "period_start")
    )
    return FinancialFactRow(
        cik=cik,
        ticker=_required_string(row, "ticker"),
        company_name=_required_string(row, "company_name"),
        metric=_required_string(row, "metric"),
        taxonomy_tag=_required_string(row, "taxonomy_tag"),
        unit=_required_string(row, "unit"),
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,  # type: ignore[typeddict-item]
        period_start=period_start,
        period_end=_coerce_date(row.get("period_end"), "period_end"),
        value=value,
        form_type=_required_string(row, "form_type"),
        filing_date=_coerce_date(row.get("filing_date"), "filing_date"),
        accession_no=_required_string(row, "accession_no"),
        source_url=_required_string(row, "source_url"),
    )


def _params(row: FinancialFactRow) -> dict[str, Any]:
    return dict(row)


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


def _coerce_date(value: object, field: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field} must be a valid ISO date.") from exc
    raise ValueError(f"{field} must be a date or ISO date string.")
