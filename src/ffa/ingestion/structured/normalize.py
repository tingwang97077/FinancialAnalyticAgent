"""Normalize SEC company facts into canonical financial fact rows."""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, TypedDict

from ffa.common.entities import normalize_ticker

logger = logging.getLogger(__name__)

# The tuple order is semantic preference order. Tags are added only after their
# accounting meaning has been reviewed; unknown tags are never inferred from names.
CANONICAL_METRIC_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "net_income": (
        "NetIncomeLoss",
        "ProfitLoss",
    ),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "cash_and_equivalents": ("CashAndCashEquivalentsAtCarryingValue",),
}

type FiscalPeriod = Literal["FY", "Q1", "Q2", "Q3", "Q4"]
type PeriodKind = Literal["duration", "instant"]
type NumericValue = int | float

_ALLOWED_PERIODS = frozenset({"FY", "Q1", "Q2", "Q3", "Q4"})
_FORM_TYPES = {
    "10-K": "10-K",
    "10-K/A": "10-K",
    "10-Q": "10-Q",
    "10-Q/A": "10-Q",
}
_METRIC_PERIOD_KIND: dict[str, PeriodKind] = {
    "revenue": "duration",
    "net_income": "duration",
    "total_assets": "instant",
    "total_liabilities": "instant",
    "cash_and_equivalents": "instant",
}
_METRIC_UNITS: dict[str, frozenset[str]] = {
    metric: frozenset({"USD"}) for metric in CANONICAL_METRIC_TAGS
}
_ACCESSION_PATTERN = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_PERIOD_ORDER = {"FY": 0, "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


class FinancialFactRow(TypedDict):
    """Canonical row accepted by the structured fact loader."""

    cik: int
    ticker: str
    company_name: str
    metric: str
    taxonomy_tag: str
    unit: str
    fiscal_year: int
    fiscal_period: FiscalPeriod
    period_start: date | None
    period_end: date
    value: NumericValue
    form_type: str
    filing_date: date
    accession_no: str
    source_url: str


class NormalizationError(ValueError):
    """Raised when a companyfacts envelope cannot be normalized safely."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    row: FinancialFactRow
    tag_priority: int
    period_kind: PeriodKind


def normalize_facts(raw: Mapping[str, Any]) -> list[FinancialFactRow]:
    """Normalize mapped XBRL facts and retain only primary fiscal-period values.

    Selection occurs in two explicit phases. First, each filing retains the fact
    ending at its latest reported date; quarterly duration metrics choose the
    shortest duration while annual metrics choose the longest duration. Second,
    facts sharing a canonical fiscal key retain the most recent filing date.

    Args:
        raw: Envelope returned by ``structured.fetch.fetch_companyfacts``.

    Returns:
        Canonical, deduplicated rows sorted deterministically.

    Raises:
        NormalizationError: If the envelope or company identity is malformed.
    """
    company, payload = _unwrap_envelope(raw)
    cik = _positive_int(company.get("cik"), field="company.cik")
    ticker_value = company.get("ticker")
    company_name_value = company.get("name")
    if not isinstance(ticker_value, str) or not isinstance(company_name_value, str):
        raise NormalizationError("Company ticker and name must be strings.")
    ticker = normalize_ticker(ticker_value)
    company_name = company_name_value.strip()
    if not company_name:
        raise NormalizationError("Company name must not be empty.")

    payload_cik = _positive_int(payload.get("cik"), field="companyfacts.cik")
    if payload_cik != cik:
        raise NormalizationError("Company identity and companyfacts CIK do not match.")

    facts = payload.get("facts")
    if not isinstance(facts, Mapping):
        raise NormalizationError("SEC companyfacts payload must contain a facts object.")

    candidates: list[_Candidate] = []
    tag_index = _tag_index()
    for taxonomy_name, taxonomy_facts in facts.items():
        if not isinstance(taxonomy_name, str) or not isinstance(taxonomy_facts, Mapping):
            logger.warning("Skipping malformed XBRL taxonomy object.")
            continue
        for taxonomy_tag, concept in taxonomy_facts.items():
            if not isinstance(taxonomy_tag, str):
                logger.warning("Skipping non-string XBRL taxonomy tag.")
                continue
            mapped_tag = tag_index.get(taxonomy_tag)
            if mapped_tag is None:
                logger.info(
                    "Skipping unmapped XBRL tag.",
                    extra={
                        "taxonomy_namespace": taxonomy_name,
                        "taxonomy_tag": taxonomy_tag,
                    },
                )
                continue
            metric, tag_priority = mapped_tag
            candidates.extend(
                _concept_candidates(
                    concept=concept,
                    cik=cik,
                    ticker=ticker,
                    company_name=company_name,
                    metric=metric,
                    taxonomy_tag=taxonomy_tag,
                    tag_priority=tag_priority,
                )
            )

    primary_contexts = _select_primary_filing_contexts(candidates)
    rows = _select_latest_restatements(primary_contexts)
    return sorted(
        rows,
        key=lambda row: (
            row["fiscal_year"],
            _PERIOD_ORDER[row["fiscal_period"]],
            row["metric"],
            row["ticker"],
        ),
    )


def _tag_index() -> dict[str, tuple[str, int]]:
    index: dict[str, tuple[str, int]] = {}
    for metric, tags in CANONICAL_METRIC_TAGS.items():
        for priority, tag in enumerate(tags):
            if tag in index:
                raise RuntimeError(f"XBRL tag {tag!r} maps to more than one canonical metric.")
            index[tag] = (metric, priority)
    return index


def _concept_candidates(
    *,
    concept: object,
    cik: int,
    ticker: str,
    company_name: str,
    metric: str,
    taxonomy_tag: str,
    tag_priority: int,
) -> list[_Candidate]:
    if not isinstance(concept, Mapping):
        logger.warning(
            "Skipping malformed mapped XBRL concept.", extra={"taxonomy_tag": taxonomy_tag}
        )
        return []
    units = concept.get("units")
    if not isinstance(units, Mapping):
        logger.warning(
            "Skipping mapped XBRL concept without units.",
            extra={"taxonomy_tag": taxonomy_tag},
        )
        return []

    candidates: list[_Candidate] = []
    for unit, records in units.items():
        if not isinstance(unit, str) or unit not in _METRIC_UNITS[metric]:
            continue
        if not isinstance(records, list):
            logger.warning(
                "Skipping mapped XBRL unit with malformed records.",
                extra={"taxonomy_tag": taxonomy_tag, "unit": unit},
            )
            continue
        for record in records:
            try:
                candidate = _candidate_from_record(
                    record=record,
                    cik=cik,
                    ticker=ticker,
                    company_name=company_name,
                    metric=metric,
                    taxonomy_tag=taxonomy_tag,
                    tag_priority=tag_priority,
                    unit=unit,
                )
            except NormalizationError as exc:
                logger.warning(
                    "Skipping invalid mapped XBRL fact: %s",
                    exc,
                    extra={"taxonomy_tag": taxonomy_tag, "unit": unit},
                )
                continue
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _candidate_from_record(
    *,
    record: object,
    cik: int,
    ticker: str,
    company_name: str,
    metric: str,
    taxonomy_tag: str,
    tag_priority: int,
    unit: str,
) -> _Candidate | None:
    if not isinstance(record, Mapping):
        raise NormalizationError("XBRL fact record must be an object.")

    raw_form = record.get("form")
    raw_period = record.get("fp")
    if raw_form not in _FORM_TYPES or raw_period not in _ALLOWED_PERIODS:
        return None
    form_type = _FORM_TYPES[str(raw_form)]
    fiscal_period = str(raw_period)
    if (form_type == "10-K" and fiscal_period != "FY") or (
        form_type == "10-Q" and fiscal_period == "FY"
    ):
        return None

    fiscal_year = _positive_int(record.get("fy"), field="fy")
    filing_date = _iso_date(record.get("filed"), field="filed")
    period_end = _iso_date(record.get("end"), field="end")
    period_start = _optional_iso_date(record.get("start"), field="start")
    period_kind = _METRIC_PERIOD_KIND[metric]
    if period_kind == "duration" and period_start is None:
        raise NormalizationError("Duration fact must declare a start date.")
    if period_kind == "instant" and period_start is not None:
        raise NormalizationError("Instant fact must not declare a start date.")
    if period_start is not None and period_start > period_end:
        raise NormalizationError("Fact start date must not be after its end date.")

    value = record.get("val")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalizationError("Fact value must be numeric.")
    if isinstance(value, float) and not math.isfinite(value):
        raise NormalizationError("Fact value must be finite.")

    accession_no = record.get("accn")
    if not isinstance(accession_no, str) or not _ACCESSION_PATTERN.fullmatch(accession_no):
        raise NormalizationError("Fact accession number is malformed.")

    source_url = _filing_index_url(cik, accession_no)
    row = FinancialFactRow(
        cik=cik,
        ticker=ticker,
        company_name=company_name,
        metric=metric,
        taxonomy_tag=taxonomy_tag,
        unit=unit,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,  # type: ignore[typeddict-item]
        period_start=period_start,
        period_end=period_end,
        value=value,
        form_type=form_type,
        filing_date=filing_date,
        accession_no=accession_no,
        source_url=source_url,
    )
    return _Candidate(row=row, tag_priority=tag_priority, period_kind=period_kind)


def _select_primary_filing_contexts(candidates: list[_Candidate]) -> list[_Candidate]:
    groups: dict[tuple[str, str, int, FiscalPeriod], list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        row = candidate.row
        key = (
            row["accession_no"],
            row["metric"],
            row["fiscal_year"],
            row["fiscal_period"],
        )
        groups[key].append(candidate)

    selected: list[_Candidate] = []
    for group in groups.values():
        latest_end = max(candidate.row["period_end"] for candidate in group)
        current = [candidate for candidate in group if candidate.row["period_end"] == latest_end]
        period_kind = current[0].period_kind
        fiscal_period = current[0].row["fiscal_period"]
        if period_kind == "duration":
            starts = [
                candidate.row["period_start"]
                for candidate in current
                if candidate.row["period_start"] is not None
            ]
            target_start = min(starts) if fiscal_period == "FY" else max(starts)
            current = [
                candidate for candidate in current if candidate.row["period_start"] == target_start
            ]
        selected.append(
            min(
                current,
                key=lambda candidate: (
                    candidate.tag_priority,
                    candidate.row["taxonomy_tag"],
                    str(candidate.row["value"]),
                ),
            )
        )
    return selected


def _select_latest_restatements(candidates: list[_Candidate]) -> list[FinancialFactRow]:
    groups: dict[tuple[int, str, int, FiscalPeriod], list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        row = candidate.row
        key = (row["cik"], row["metric"], row["fiscal_year"], row["fiscal_period"])
        groups[key].append(candidate)

    selected: list[FinancialFactRow] = []
    for group in groups.values():
        latest = max(
            group,
            key=lambda candidate: (
                candidate.row["filing_date"],
                candidate.row["accession_no"],
            ),
        )
        if len(group) > 1:
            logger.info(
                "Selected most recently filed XBRL restatement.",
                extra={
                    "cik": latest.row["cik"],
                    "metric": latest.row["metric"],
                    "fiscal_year": latest.row["fiscal_year"],
                    "fiscal_period": latest.row["fiscal_period"],
                    "filing_date": latest.row["filing_date"].isoformat(),
                },
            )
        selected.append(latest.row)
    return selected


def _unwrap_envelope(
    raw: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    company = raw.get("company")
    payload = raw.get("companyfacts")
    if not isinstance(company, Mapping) or not isinstance(payload, Mapping):
        raise NormalizationError(
            "Structured fetch envelope must contain company and companyfacts objects."
        )
    return company, payload


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise NormalizationError(f"{field} must be a positive integer.")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise NormalizationError(f"{field} must be a positive integer.") from exc
    if parsed <= 0:
        raise NormalizationError(f"{field} must be a positive integer.")
    return parsed


def _iso_date(value: object, *, field: str) -> date:
    if not isinstance(value, str):
        raise NormalizationError(f"{field} must be an ISO date string.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise NormalizationError(f"{field} must be a valid ISO date.") from exc


def _optional_iso_date(value: object, *, field: str) -> date | None:
    return None if value is None else _iso_date(value, field=field)


def _filing_index_url(cik: int, accession_no: str) -> str:
    accession_compact = accession_no.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{accession_compact}/{accession_no}-index.html"
    )
