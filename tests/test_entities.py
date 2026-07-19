"""Tests for ticker and CIK resolution."""

from typing import Any

import pytest

from ffa.common.entities import (
    EntityResolutionError,
    UnknownTickerError,
    build_company_index,
    normalize_ticker,
    resolve_ticker,
    resolve_universe,
)


class FakeSecClient:
    """Minimal SEC client test double."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def fetch_company_tickers(self) -> dict[str, Any]:
        """Return the configured ticker payload."""
        self.calls += 1
        return self.payload


@pytest.fixture
def company_payload() -> dict[str, Any]:
    """Return representative SEC company ticker data."""
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway Inc."},
    }


def test_resolve_universe_preserves_order_normalizes_and_deduplicates(
    company_payload: dict[str, Any],
) -> None:
    client = FakeSecClient(company_payload)

    companies = resolve_universe("brk.b, AAPL,brk-b", client=client)

    assert companies == [
        {"ticker": "BRK-B", "cik": 1067983, "name": "Berkshire Hathaway Inc."},
        {"ticker": "AAPL", "cik": 320193, "name": "Apple Inc."},
    ]
    assert client.calls == 1


def test_unknown_ticker_raises_domain_error(company_payload: dict[str, Any]) -> None:
    company_index = build_company_index(company_payload)

    with pytest.raises(UnknownTickerError, match="MSFT"):
        resolve_ticker("MSFT", company_index)


def test_malformed_company_record_is_rejected() -> None:
    with pytest.raises(EntityResolutionError, match="missing 'title'"):
        build_company_index({"0": {"cik_str": 320193, "ticker": "AAPL"}})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(" aapl ", "AAPL"), ("brk.b", "BRK-B")],
)
def test_normalize_ticker(raw: str, expected: str) -> None:
    assert normalize_ticker(raw) == expected
