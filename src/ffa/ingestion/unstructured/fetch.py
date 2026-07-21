"""Discover and fetch SEC filing documents for unstructured ingestion."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Literal, TypedDict

from sqlalchemy import Engine, text

from ffa.common.db import create_rw_engine
from ffa.common.entities import Company, normalize_ticker
from ffa.common.sec_client import SEC_DATA_BASE_URL, SecEdgarClient
from ffa.config import Settings, get_settings

logger = logging.getLogger(__name__)

UNSTRUCTURED_SOURCE = "SEC_EDGAR_UNSTRUCTURED"

_SUPPORTED_FORMS = {
    "10-K": "10-K",
    "10-K/A": "10-K",
    "10-Q": "10-Q",
    "10-Q/A": "10-Q",
}
_ACCESSION_PATTERN = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_NO_TEN_K_REASON = "No 10-K or 10-K/A filing exists in SEC submissions."

_SELECT_STATE_SQL = text(
    """
    SELECT last_filing_date, last_accession
    FROM ingestion_state
    WHERE source = :source AND cik = :cik
    """
)

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
        :primary_doc_url
    )
    ON CONFLICT (accession_no) DO UPDATE SET
        cik = EXCLUDED.cik,
        form_type = EXCLUDED.form_type,
        filing_date = EXCLUDED.filing_date,
        period_of_report = EXCLUDED.period_of_report,
        primary_doc_url = EXCLUDED.primary_doc_url
    """
)

_SELECT_FISCAL_CONTEXT_SQL = text(
    """
    SELECT fiscal_year, fiscal_period, count(*) AS fact_count
    FROM financial_facts
    WHERE accession_no = :accession_no
    GROUP BY fiscal_year, fiscal_period
    ORDER BY fact_count DESC, fiscal_year DESC, fiscal_period DESC
    LIMIT 1
    """
)


class FilingMetadata(TypedDict):
    """Serializable SEC filing identity and fiscal metadata."""

    ticker: str
    cik: int
    company_name: str
    accession_no: str
    form_type: str
    filing_date: date
    period_of_report: date | None
    primary_document: str
    primary_doc_url: str
    fiscal_year: int | None
    fiscal_period: str | None


class FetchedFilingDocument(TypedDict):
    """Filing metadata paired with its raw primary-document HTML."""

    filing: FilingMetadata
    html: str


class FilingDiscoverySummary(TypedDict):
    """Per-company status included in an unstructured discovery report."""

    ticker: str
    cik: int
    status: Literal["OK", "SKIPPED"]
    filing_count: int
    reason: str | None


class FilingDiscoveryResult(FilingDiscoverySummary):
    """Per-company discovery result with filings eligible for ingestion."""

    filings: list[FilingMetadata]


class FilingDiscoveryReport(TypedDict):
    """Aggregate report that preserves successful and skipped companies."""

    filings: list[FilingMetadata]
    ok: list[FilingDiscoverySummary]
    skipped: list[FilingDiscoverySummary]


def list_filings(
    company: Mapping[str, object],
    *,
    client: SecEdgarClient | None = None,
    engine: Engine | None = None,
    settings: Settings | None = None,
    use_cache: bool = True,
) -> list[FilingMetadata]:
    """List eligible filings while preserving the original public API."""
    return discover_filings(
        company,
        client=client,
        engine=engine,
        settings=settings,
        use_cache=use_cache,
    )["filings"]


def discover_filings(
    company: Mapping[str, object],
    *,
    client: SecEdgarClient | None = None,
    engine: Engine | None = None,
    settings: Settings | None = None,
    use_cache: bool = True,
) -> FilingDiscoveryResult:
    """Discover incremental filings and classify companies with no 10-K as skipped.

    The current submissions document and every relevant historical submissions
    file are considered. Fiscal periods are intentionally left unset here; they
    are later joined from canonical XBRL facts by accession number. A company
    whose SEC submissions contain no 10-K or 10-K/A is a valid skip rather than
    an ingestion failure.

    Args:
        company: Mapping containing ``ticker``, ``cik``, and ``name``.
        client: Optional injected SEC client.
        engine: Optional injected read-write SQLAlchemy engine.
        settings: Optional settings used to construct owned resources.
        use_cache: Whether SEC submission responses may use the disk cache.

    Returns:
        Discovery status and new supported filings sorted from oldest to newest.
    """
    normalized_company = _validate_company(company)
    resolved_settings = settings or get_settings()
    owned_engine = engine is None
    resolved_engine = engine or create_rw_engine(resolved_settings.database_url)
    owned_client = client is None
    resolved_client = client or SecEdgarClient.from_settings(resolved_settings)

    try:
        state = _read_state(resolved_engine, normalized_company["cik"])
        root_payload = resolved_client.fetch_submissions(
            normalized_company["cik"],
            use_cache=use_cache,
        )
        _validate_submission_identity(root_payload, normalized_company)

        filings = parse_submission_filings(root_payload, normalized_company)
        has_ten_k = _contains_ten_k_form(root_payload)
        for history_file in _history_files(root_payload, state):
            history_url = f"{SEC_DATA_BASE_URL}/submissions/{history_file}"
            history_payload = resolved_client.get_json(history_url, use_cache=use_cache)
            filings.extend(parse_submission_filings(history_payload, normalized_company))
            has_ten_k = has_ten_k or _contains_ten_k_form(history_payload)

        if not has_ten_k:
            logger.warning(
                "Skipping SEC unstructured ingestion for ticker=%s cik=%s: %s",
                normalized_company["ticker"],
                normalized_company["cik"],
                _NO_TEN_K_REASON,
            )
            return FilingDiscoveryResult(
                ticker=normalized_company["ticker"],
                cik=normalized_company["cik"],
                status="SKIPPED",
                filing_count=0,
                reason=_NO_TEN_K_REASON,
                filings=[],
            )

        deduplicated = {filing["accession_no"]: filing for filing in filings}
        incremental_filings = sorted(
            (filing for filing in deduplicated.values() if _is_after_state(filing, state)),
            key=lambda filing: (filing["filing_date"], filing["accession_no"]),
        )
        return FilingDiscoveryResult(
            ticker=normalized_company["ticker"],
            cik=normalized_company["cik"],
            status="OK",
            filing_count=len(incremental_filings),
            reason=None,
            filings=incremental_filings,
        )
    finally:
        if owned_client:
            resolved_client.close()
        if owned_engine:
            resolved_engine.dispose()


def summarize_filing_discoveries(
    discoveries: Sequence[FilingDiscoveryResult],
) -> FilingDiscoveryReport:
    """Flatten eligible filings and report every successful or skipped ticker."""
    filings: list[FilingMetadata] = []
    ok: list[FilingDiscoverySummary] = []
    skipped: list[FilingDiscoverySummary] = []
    for discovery in discoveries:
        summary = FilingDiscoverySummary(
            ticker=discovery["ticker"],
            cik=discovery["cik"],
            status=discovery["status"],
            filing_count=discovery["filing_count"],
            reason=discovery["reason"],
        )
        if discovery["status"] == "OK":
            ok.append(summary)
            filings.extend(discovery["filings"])
        else:
            skipped.append(summary)

    logger.info(
        "SEC unstructured discovery report: OK=%s; SKIPPED=%s",
        [summary["ticker"] for summary in ok],
        [
            {
                "ticker": summary["ticker"],
                "cik": summary["cik"],
                "reason": summary["reason"],
            }
            for summary in skipped
        ],
    )
    return FilingDiscoveryReport(filings=filings, ok=ok, skipped=skipped)


def parse_submission_filings(
    payload: Mapping[str, Any],
    company: Mapping[str, object],
) -> list[FilingMetadata]:
    """Parse supported filings from a current or historical submissions payload."""
    normalized_company = _validate_company(company)
    records = _submission_columns(payload)
    accession_numbers = _required_column(records, "accessionNumber")
    forms = _required_column(records, "form")
    filing_dates = _required_column(records, "filingDate")
    report_dates = _required_column(records, "reportDate")
    primary_documents = _required_column(records, "primaryDocument")

    lengths = {
        len(accession_numbers),
        len(forms),
        len(filing_dates),
        len(report_dates),
        len(primary_documents),
    }
    if len(lengths) != 1:
        raise ValueError("SEC submissions columns must have equal lengths.")

    filings: list[FilingMetadata] = []
    for accession, raw_form, raw_filing_date, raw_report_date, primary_document in zip(
        accession_numbers,
        forms,
        filing_dates,
        report_dates,
        primary_documents,
        strict=True,
    ):
        if raw_form not in _SUPPORTED_FORMS:
            continue
        if not isinstance(accession, str) or not _ACCESSION_PATTERN.fullmatch(accession):
            raise ValueError("SEC filing accession number is malformed.")
        if not isinstance(primary_document, str) or not primary_document.strip():
            logger.warning(
                "Skipping SEC filing %s because its primary document is missing.",
                accession,
            )
            continue

        filing_date = _parse_date(raw_filing_date, field="filingDate")
        period_of_report = _optional_date(raw_report_date, field="reportDate")
        primary_doc_url = _primary_document_url(
            normalized_company["cik"], accession, primary_document.strip()
        )
        filings.append(
            FilingMetadata(
                ticker=normalized_company["ticker"],
                cik=normalized_company["cik"],
                company_name=normalized_company["name"],
                accession_no=accession,
                form_type=_SUPPORTED_FORMS[str(raw_form)],
                filing_date=filing_date,
                period_of_report=period_of_report,
                primary_document=primary_document.strip(),
                primary_doc_url=primary_doc_url,
                fiscal_year=None,
                fiscal_period=None,
            )
        )
    return filings


def fetch_documents(
    filing: Mapping[str, object],
    *,
    client: SecEdgarClient | None = None,
    engine: Engine | None = None,
    settings: Settings | None = None,
    use_cache: bool = True,
) -> FetchedFilingDocument:
    """Fetch one primary filing document and upsert its relational metadata.

    Fiscal metadata comes from structured facts with the same accession number.
    It is never inferred from calendar dates.
    """
    normalized_filing = _validate_filing(filing)
    resolved_settings = settings or get_settings()
    owned_engine = engine is None
    resolved_engine = engine or create_rw_engine(resolved_settings.database_url)
    owned_client = client is None
    resolved_client = client or SecEdgarClient.from_settings(resolved_settings)

    try:
        html = resolved_client.fetch_filing_document(
            normalized_filing["primary_doc_url"],
            use_cache=use_cache,
        )
        if not html.strip():
            raise ValueError("SEC filing document must not be empty.")

        params = dict(normalized_filing)
        with resolved_engine.begin() as connection:
            connection.execute(_UPSERT_COMPANY_SQL, params)
            connection.execute(_UPSERT_FILING_SQL, params)
            fiscal_context = (
                connection.execute(
                    _SELECT_FISCAL_CONTEXT_SQL,
                    {"accession_no": normalized_filing["accession_no"]},
                )
                .mappings()
                .one_or_none()
            )

        if fiscal_context is not None:
            normalized_filing["fiscal_year"] = int(fiscal_context["fiscal_year"])
            normalized_filing["fiscal_period"] = str(fiscal_context["fiscal_period"])
        return {"filing": normalized_filing, "html": html}
    finally:
        if owned_client:
            resolved_client.close()
        if owned_engine:
            resolved_engine.dispose()


def _submission_columns(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    filings = payload.get("filings")
    if isinstance(filings, Mapping):
        recent = filings.get("recent")
        if isinstance(recent, Mapping):
            return recent
    return payload


def _required_column(records: Mapping[str, Any], name: str) -> Sequence[object]:
    value = records.get(name)
    if not isinstance(value, list):
        raise ValueError(f"SEC submissions payload must contain a {name} list.")
    return value


def _contains_ten_k_form(payload: Mapping[str, Any]) -> bool:
    forms = _required_column(_submission_columns(payload), "form")
    return any(form in {"10-K", "10-K/A"} for form in forms)


def _history_files(
    payload: Mapping[str, Any],
    state: tuple[date | None, str | None],
) -> list[str]:
    filings = payload.get("filings")
    if not isinstance(filings, Mapping):
        return []
    files = filings.get("files")
    if not isinstance(files, list):
        return []

    last_filing_date, _ = state
    names: list[str] = []
    for record in files:
        if not isinstance(record, Mapping):
            raise ValueError("SEC submissions history entries must be objects.")
        name = record.get("name")
        filing_to = _optional_date(record.get("filingTo"), field="filingTo")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("SEC submissions history file name must not be empty.")
        if last_filing_date is None or filing_to is None or filing_to >= last_filing_date:
            names.append(name.strip())
    return names


def _read_state(engine: Engine, cik: int) -> tuple[date | None, str | None]:
    with engine.connect() as connection:
        row = connection.execute(
            _SELECT_STATE_SQL,
            {"source": UNSTRUCTURED_SOURCE, "cik": cik},
        ).one_or_none()
    if row is None:
        return None, None
    return row.last_filing_date, row.last_accession


def _is_after_state(
    filing: FilingMetadata,
    state: tuple[date | None, str | None],
) -> bool:
    last_filing_date, last_accession = state
    if last_filing_date is None:
        return True
    return (filing["filing_date"], filing["accession_no"]) > (
        last_filing_date,
        last_accession or "",
    )


def _validate_submission_identity(payload: Mapping[str, Any], company: Company) -> None:
    raw_cik = payload.get("cik")
    if isinstance(raw_cik, bool) or not isinstance(raw_cik, (int, str)):
        raise ValueError("SEC submissions response must contain a numeric CIK.")
    try:
        payload_cik = int(raw_cik)
    except ValueError as exc:
        raise ValueError("SEC submissions response CIK must contain only digits.") from exc
    if payload_cik != company["cik"]:
        raise ValueError("SEC submissions response CIK does not match the requested company.")


def _validate_company(company: Mapping[str, object]) -> Company:
    ticker = company.get("ticker")
    cik = company.get("cik")
    name = company.get("name")
    if not isinstance(ticker, str) or not isinstance(name, str):
        raise ValueError("Company ticker and name must be strings.")
    if isinstance(cik, bool) or not isinstance(cik, int) or cik <= 0:
        raise ValueError("Company CIK must be a positive integer.")
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("Company name must not be empty.")
    return Company(ticker=normalize_ticker(ticker), cik=cik, name=normalized_name)


def _validate_filing(filing: Mapping[str, object]) -> FilingMetadata:
    company = _validate_company(
        {
            "ticker": filing.get("ticker"),
            "cik": filing.get("cik"),
            "name": filing.get("company_name"),
        }
    )
    accession_no = filing.get("accession_no")
    form_type = filing.get("form_type")
    primary_document = filing.get("primary_document")
    primary_doc_url = filing.get("primary_doc_url")
    if not isinstance(accession_no, str) or not _ACCESSION_PATTERN.fullmatch(accession_no):
        raise ValueError("SEC filing accession number is malformed.")
    if form_type not in {"10-K", "10-Q"}:
        raise ValueError("SEC filing form_type must be 10-K or 10-Q.")
    if not isinstance(primary_document, str) or not primary_document.strip():
        raise ValueError("SEC filing primary document must not be empty.")
    if not isinstance(primary_doc_url, str) or not primary_doc_url.strip():
        raise ValueError("SEC filing primary document URL must not be empty.")

    fiscal_year = filing.get("fiscal_year")
    fiscal_period = filing.get("fiscal_period")
    if fiscal_year is not None and (
        isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int) or fiscal_year <= 0
    ):
        raise ValueError("fiscal_year must be a positive integer when present.")
    if fiscal_period is not None and fiscal_period not in {"FY", "Q1", "Q2", "Q3", "Q4"}:
        raise ValueError("fiscal_period must be FY or Q1 through Q4 when present.")

    return FilingMetadata(
        ticker=company["ticker"],
        cik=company["cik"],
        company_name=company["name"],
        accession_no=accession_no,
        form_type=str(form_type),
        filing_date=_parse_date(filing.get("filing_date"), field="filing_date"),
        period_of_report=_optional_date(filing.get("period_of_report"), field="period_of_report"),
        primary_document=primary_document.strip(),
        primary_doc_url=primary_doc_url.strip(),
        fiscal_year=fiscal_year,
        fiscal_period=None if fiscal_period is None else str(fiscal_period),
    )


def _primary_document_url(cik: int, accession_no: str, primary_document: str) -> str:
    compact_accession = accession_no.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{compact_accession}/{primary_document}"


def _parse_date(value: object, *, field: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field} must be a valid ISO date.") from exc
    raise ValueError(f"{field} must be a date or ISO date string.")


def _optional_date(value: object, *, field: str) -> date | None:
    if value is None or value == "":
        return None
    return _parse_date(value, field=field)
