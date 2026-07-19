"""Ticker and SEC CIK entity resolution."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypedDict

from ffa.common.sec_client import SecEdgarClient
from ffa.config import Settings, get_settings

_TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]*$")


class Company(TypedDict):
    """Resolved SEC company identity."""

    ticker: str
    cik: int
    name: str


class EntityResolutionError(ValueError):
    """Raised when SEC entity data cannot resolve a requested ticker."""


class UnknownTickerError(EntityResolutionError):
    """Raised when a ticker does not exist in the SEC company dataset."""


class _CompanyTickerClient(Protocol):
    def fetch_company_tickers(self) -> dict[str, Any]: ...


def normalize_ticker(ticker: str) -> str:
    """Normalize user ticker notation to the SEC dash convention.

    Args:
        ticker: User-provided ticker, such as ``BRK.B`` or ``aapl``.

    Returns:
        An uppercase ticker using dashes for share classes.

    Raises:
        EntityResolutionError: If the ticker is empty or malformed.
    """
    normalized = ticker.strip().upper().replace(".", "-")
    if not normalized or not _TICKER_PATTERN.fullmatch(normalized):
        raise EntityResolutionError(f"Invalid ticker symbol: {ticker!r}.")
    return normalized


def build_company_index(payload: Mapping[str, Any]) -> dict[str, Company]:
    """Build a normalized ticker index from the SEC company ticker payload."""
    index: dict[str, Company] = {}
    for record in payload.values():
        if not isinstance(record, Mapping):
            raise EntityResolutionError("SEC company ticker records must be JSON objects.")
        try:
            ticker_value = record["ticker"]
            cik_value = record["cik_str"]
            name_value = record["title"]
        except KeyError as exc:
            raise EntityResolutionError(
                f"SEC company ticker record is missing {exc.args[0]!r}."
            ) from exc

        if not isinstance(ticker_value, str) or not isinstance(name_value, str):
            raise EntityResolutionError("SEC company ticker and title values must be strings.")
        if isinstance(cik_value, bool) or not isinstance(cik_value, (int, str)):
            raise EntityResolutionError("SEC company CIK values must be integers or digit strings.")
        try:
            cik = int(cik_value)
        except ValueError as exc:
            raise EntityResolutionError("SEC company CIK values must contain only digits.") from exc
        if cik <= 0:
            raise EntityResolutionError("SEC company CIK values must be positive.")

        ticker = normalize_ticker(ticker_value)
        company = Company(ticker=ticker, cik=cik, name=name_value.strip())
        existing = index.get(ticker)
        if existing is not None and existing["cik"] != cik:
            raise EntityResolutionError(
                f"SEC company dataset contains conflicting CIKs for {ticker}."
            )
        index[ticker] = company
    return index


def resolve_ticker(ticker: str, company_index: Mapping[str, Company]) -> Company:
    """Resolve one ticker against a previously built SEC company index."""
    normalized = normalize_ticker(ticker)
    try:
        return company_index[normalized]
    except KeyError as exc:
        raise UnknownTickerError(f"Ticker {normalized!r} was not found in SEC EDGAR.") from exc


def resolve_universe(
    tickers: str | Sequence[str] | None = None,
    *,
    client: _CompanyTickerClient | None = None,
    settings: Settings | None = None,
) -> list[Company]:
    """Resolve the configured ticker universe to SEC CIKs.

    Args:
        tickers: Comma-separated tickers or a sequence. Uses ``TICKER_UNIVERSE`` when omitted.
        client: Optional injected SEC client.
        settings: Optional injected application settings.

    Returns:
        Resolved companies in requested order, with duplicates removed.
    """
    resolved_settings = settings
    if tickers is None or client is None:
        resolved_settings = settings or get_settings()

    requested_tickers = _parse_tickers(
        resolved_settings.ticker_symbols if tickers is None else tickers
    )

    if client is not None:
        return _resolve_with_client(requested_tickers, client)

    if resolved_settings is None:
        raise AssertionError("Settings must be available when the SEC client is not injected.")
    with SecEdgarClient.from_settings(resolved_settings) as owned_client:
        return _resolve_with_client(requested_tickers, owned_client)


def _resolve_with_client(tickers: Sequence[str], client: _CompanyTickerClient) -> list[Company]:
    company_index = build_company_index(client.fetch_company_tickers())
    return [resolve_ticker(ticker, company_index) for ticker in tickers]


def _parse_tickers(tickers: str | Sequence[str]) -> tuple[str, ...]:
    raw_tickers = tickers.split(",") if isinstance(tickers, str) else tickers
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in raw_tickers:
        if not isinstance(ticker, str):
            raise EntityResolutionError("Ticker universe entries must be strings.")
        symbol = normalize_ticker(ticker)
        if symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)
    if not normalized:
        raise EntityResolutionError("Ticker universe must contain at least one symbol.")
    return tuple(normalized)
