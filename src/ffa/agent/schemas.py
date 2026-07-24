"""Typed contracts shared by the agent layer."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

_FISCAL_PERIODS = frozenset({"FY", "Q1", "Q2", "Q3", "Q4"})
_CANONICAL_SECTIONS = {
    "md&a": "MD&A",
    "risk factors": "Risk Factors",
    "notes": "Notes",
}
CANONICAL_METRICS = frozenset(
    {
        "revenue",
        "net_income",
        "total_assets",
        "total_liabilities",
        "cash_and_equivalents",
    }
)
METRIC_SYNONYMS_VERSION = 1
METRIC_SYNONYMS = MappingProxyType(
    {
        "revenue": "revenue",
        "revenues": "revenue",
        "total_revenue": "revenue",
        "net_revenue": "revenue",
        "net_sales": "revenue",
        "net_income": "net_income",
        "net_income_loss": "net_income",
        "net_earnings": "net_income",
        "total_assets": "total_assets",
        "assets": "total_assets",
        "total_liabilities": "total_liabilities",
        "liabilities": "total_liabilities",
        "cash_and_equivalents": "cash_and_equivalents",
        "cash_and_cash_equivalents": "cash_and_equivalents",
        "cash_and_cash_equivalent": "cash_and_equivalents",
        "cash_equivalents": "cash_and_equivalents",
    }
)


class _AgentModel(BaseModel):
    """Strict base model for data crossing agent boundaries."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        str_strip_whitespace=True,
    )


class Intent(StrEnum):
    """Supported query-routing intents."""

    NUMERIC = "numeric"
    NARRATIVE = "narrative"
    HYBRID = "hybrid"
    OUT_OF_SCOPE = "out_of_scope"


class Entities(_AgentModel):
    """Financial entities and periods extracted from a user question."""

    tickers: list[str] = Field(default_factory=list)
    ciks: list[Annotated[int, Field(gt=0)]] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    fiscal_years: list[Annotated[int, Field(gt=0)]] = Field(default_factory=list)
    fiscal_periods: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)

    @field_validator("fiscal_periods")
    @classmethod
    def validate_fiscal_periods(cls, periods: list[str]) -> list[str]:
        """Normalize fiscal periods and reject unsupported calendar labels."""
        normalized = [period.strip().upper() for period in periods]
        invalid = set(normalized).difference(_FISCAL_PERIODS)
        if invalid:
            raise ValueError("Fiscal periods must be FY or Q1 through Q4.")
        return _deduplicate(normalized)

    @field_validator("sections")
    @classmethod
    def validate_sections(cls, sections: list[str]) -> list[str]:
        """Normalize corpus section names and reject unsupported sections."""
        try:
            normalized = [_CANONICAL_SECTIONS[section.strip().casefold()] for section in sections]
        except KeyError as exc:
            raise ValueError("Sections must be MD&A, Risk Factors, or Notes.") from exc
        return _deduplicate(normalized)

    @field_validator("tickers")
    @classmethod
    def deduplicate_tickers(cls, values: list[str]) -> list[str]:
        """Remove empty and duplicate extracted values while preserving order."""
        return _deduplicate(value.strip() for value in values if value.strip())

    @field_validator("metrics")
    @classmethod
    def normalize_metrics(cls, values: list[str]) -> list[str]:
        """Map known synonyms and discard metrics outside the canonical schema."""
        return normalize_metric_names(values)

    @field_validator("ciks", "fiscal_years")
    @classmethod
    def deduplicate_integers(cls, values: list[int]) -> list[int]:
        """Remove duplicate numeric entities while preserving order."""
        return list(dict.fromkeys(values))


class Understanding(_AgentModel):
    """Structured interpretation used to choose downstream tools."""

    intent: Intent
    entities: Entities
    rewritten_query: str = Field(min_length=1)


class Citation(_AgentModel):
    """Source attribution for a generated answer."""

    source_url: str = Field(min_length=1)
    section: str | None = None
    accession_no: str | None = None


class NumberFact(_AgentModel):
    """Canonical numeric fact returned by the SQL tool."""

    metric: str = Field(min_length=1)
    fiscal_year: Annotated[int, Field(gt=0)]
    fiscal_period: str
    value: float
    unit: str = Field(min_length=1)

    @field_validator("fiscal_period")
    @classmethod
    def validate_fiscal_period(cls, period: str) -> str:
        """Normalize and validate a single fiscal period."""
        normalized = period.strip().upper()
        if normalized not in _FISCAL_PERIODS:
            raise ValueError("Fiscal period must be FY or Q1 through Q4.")
        return normalized


class Answer(_AgentModel):
    """Grounded response returned by the generation layer."""

    text: str = Field(min_length=1)
    numbers: list[NumberFact] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True


def _deduplicate[T](values: Iterable[T]) -> list[T]:
    """Return unique hashable values in their original order."""
    return list(dict.fromkeys(values))


def normalize_metric_names(values: Iterable[str]) -> list[str]:
    """Return only versioned canonical metric names in input order."""
    normalized: list[str] = []
    for value in values:
        key = "_".join(value.strip().casefold().replace("-", " ").split())
        if canonical := METRIC_SYNONYMS.get(key):
            normalized.append(canonical)
    return _deduplicate(normalized)
