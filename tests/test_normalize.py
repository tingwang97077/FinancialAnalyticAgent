"""Tests for structured SEC companyfacts normalization."""

import logging
from collections.abc import Mapping
from datetime import date
from typing import Any

from ffa.ingestion.structured.normalize import CANONICAL_METRIC_TAGS, normalize_facts

CIK = 320193
COMPANY = {"ticker": "AAPL", "cik": CIK, "name": "Apple Inc."}


def _envelope(facts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "company": COMPANY,
        "companyfacts": {
            "cik": CIK,
            "entityName": "Apple Inc.",
            "facts": {"us-gaap": dict(facts)},
        },
    }


def _concept(*records: dict[str, Any]) -> dict[str, Any]:
    return {"units": {"USD": list(records)}}


def _record(
    *,
    value: int,
    accession_no: str,
    fiscal_year: int,
    fiscal_period: str,
    start: str | None,
    end: str,
    filed: str,
    form: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "val": value,
        "accn": accession_no,
        "fy": fiscal_year,
        "fp": fiscal_period,
        "end": end,
        "filed": filed,
        "form": form,
    }
    if start is not None:
        record["start"] = start
    return record


def test_mapping_is_explicit_and_covers_required_metrics() -> None:
    assert {
        "revenue",
        "net_income",
        "total_assets",
        "total_liabilities",
        "cash_and_equivalents",
    } <= CANONICAL_METRIC_TAGS.keys()
    assert CANONICAL_METRIC_TAGS["revenue"][:3] == (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    )


def test_normalize_accepts_numeric_string_companyfacts_cik() -> None:
    facts = {
        "NetIncomeLoss": _concept(
            _record(
                value=112_010_000_000,
                accession_no="0000320193-25-000079",
                fiscal_year=2025,
                fiscal_period="FY",
                start="2024-09-29",
                end="2025-09-27",
                filed="2025-10-31",
                form="10-K",
            )
        )
    }
    envelope = _envelope(facts)
    envelope["companyfacts"]["cik"] = str(CIK)

    rows = normalize_facts(envelope)

    assert len(rows) == 1
    assert rows[0]["cik"] == CIK
    assert isinstance(rows[0]["cik"], int)


def test_maps_preferred_revenue_tag_preserves_origin_and_logs_unknown(
    caplog: Any,
) -> None:
    common_record = _record(
        value=100,
        accession_no="0000320193-25-000079",
        fiscal_year=2025,
        fiscal_period="FY",
        start="2024-09-29",
        end="2025-09-27",
        filed="2025-10-31",
        form="10-K",
    )
    facts = {
        "Revenues": _concept({**common_record, "val": 99}),
        "RevenueFromContractWithCustomerExcludingAssessedTax": _concept(common_record),
        "UnreviewedCompanySpecificMetric": _concept(common_record),
    }

    with caplog.at_level(logging.INFO):
        rows = normalize_facts(_envelope(facts))

    assert len(rows) == 1
    assert rows[0]["metric"] == "revenue"
    assert rows[0]["taxonomy_tag"] == ("RevenueFromContractWithCustomerExcludingAssessedTax")
    assert rows[0]["value"] == 100
    assert rows[0]["unit"] == "USD"
    assert any(
        getattr(record, "taxonomy_tag", None) == "UnreviewedCompanySpecificMetric"
        for record in caplog.records
    )


def test_apple_non_calendar_quarter_uses_declared_fiscal_period_and_quarter_value() -> None:
    accession_no = "0000320193-25-000057"
    facts = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": _concept(
            # Comparative period from the same filing must not become the current fact.
            _record(
                value=90_753_000_000,
                accession_no=accession_no,
                fiscal_year=2025,
                fiscal_period="Q2",
                start="2023-12-31",
                end="2024-03-30",
                filed="2025-05-02",
                form="10-Q",
            ),
            # Six-month YTD value.
            _record(
                value=219_659_000_000,
                accession_no=accession_no,
                fiscal_year=2025,
                fiscal_period="Q2",
                start="2024-09-29",
                end="2025-03-29",
                filed="2025-05-02",
                form="10-Q",
            ),
            # Standalone fiscal Q2 value.
            _record(
                value=95_359_000_000,
                accession_no=accession_no,
                fiscal_year=2025,
                fiscal_period="Q2",
                start="2024-12-29",
                end="2025-03-29",
                filed="2025-05-02",
                form="10-Q",
            ),
        )
    }

    rows = normalize_facts(_envelope(facts))

    assert len(rows) == 1
    row = rows[0]
    assert row["fiscal_year"] == 2025
    assert row["fiscal_period"] == "Q2"
    assert row["period_start"] == date(2024, 12, 29)
    assert row["period_end"] == date(2025, 3, 29)
    assert row["value"] == 95_359_000_000


def test_annual_duration_uses_full_year_not_shorter_period() -> None:
    accession_no = "0000320193-25-000079"
    facts = {
        "NetIncomeLoss": _concept(
            _record(
                value=28_000_000_000,
                accession_no=accession_no,
                fiscal_year=2025,
                fiscal_period="FY",
                start="2025-06-29",
                end="2025-09-27",
                filed="2025-10-31",
                form="10-K",
            ),
            _record(
                value=112_010_000_000,
                accession_no=accession_no,
                fiscal_year=2025,
                fiscal_period="FY",
                start="2024-09-29",
                end="2025-09-27",
                filed="2025-10-31",
                form="10-K",
            ),
        )
    }

    rows = normalize_facts(_envelope(facts))

    assert rows[0]["period_start"] == date(2024, 9, 29)
    assert rows[0]["value"] == 112_010_000_000


def test_restatement_keeps_value_from_latest_filing_date() -> None:
    facts = {
        "NetIncomeLoss": _concept(
            _record(
                value=90_000_000_000,
                accession_no="0000320193-24-000100",
                fiscal_year=2024,
                fiscal_period="FY",
                start="2023-10-01",
                end="2024-09-28",
                filed="2024-10-31",
                form="10-K",
            ),
            _record(
                value=91_000_000_000,
                accession_no="0000320193-24-000101",
                fiscal_year=2024,
                fiscal_period="FY",
                start="2023-10-01",
                end="2024-09-28",
                filed="2024-11-15",
                form="10-K/A",
            ),
        )
    }

    rows = normalize_facts(_envelope(facts))

    assert len(rows) == 1
    assert rows[0]["value"] == 91_000_000_000
    assert rows[0]["filing_date"] == date(2024, 11, 15)
    assert rows[0]["accession_no"] == "0000320193-24-000101"
