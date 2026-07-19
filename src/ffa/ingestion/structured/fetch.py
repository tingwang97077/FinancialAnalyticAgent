"""Fetch SEC company facts for structured ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from ffa.common.entities import Company, normalize_ticker
from ffa.common.sec_client import JsonObject, SecEdgarClient
from ffa.config import Settings


class FetchedCompanyFacts(TypedDict):
    """Serializable envelope containing company identity and SEC facts."""

    company: Company
    companyfacts: JsonObject


def fetch_companyfacts(
    company: Mapping[str, object],
    *,
    client: SecEdgarClient | None = None,
    settings: Settings | None = None,
    use_cache: bool = True,
) -> FetchedCompanyFacts:
    """Fetch company facts while preserving the ticker resolved by the universe task.

    Args:
        company: Mapping containing ``ticker``, ``cik``, and ``name``.
        client: Optional injected SEC client.
        settings: Optional settings used when constructing the SEC client.
        use_cache: Whether the SEC client may serve and persist its disk cache entry.

    Returns:
        A serializable company-and-companyfacts envelope.

    Raises:
        ValueError: If company identity is malformed or disagrees with the SEC response.
    """
    normalized_company = _validate_company(company)

    if client is not None:
        payload = client.fetch_companyfacts(normalized_company["cik"], use_cache=use_cache)
    else:
        with SecEdgarClient.from_settings(settings) as owned_client:
            payload = owned_client.fetch_companyfacts(
                normalized_company["cik"],
                use_cache=use_cache,
            )

    payload_cik = payload.get("cik")
    if isinstance(payload_cik, bool) or not isinstance(payload_cik, (int, str)):
        raise ValueError("SEC companyfacts response must contain a numeric CIK.")
    try:
        parsed_payload_cik = int(payload_cik)
    except ValueError as exc:
        raise ValueError("SEC companyfacts response CIK must contain only digits.") from exc
    if parsed_payload_cik != normalized_company["cik"]:
        raise ValueError("SEC companyfacts response CIK does not match the requested company.")

    return {"company": normalized_company, "companyfacts": payload}


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
