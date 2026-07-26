"""Command-line orchestration for complete SEC ingestion."""

from __future__ import annotations

import json
import logging
from typing import TypedDict

from ffa.common.db import create_rw_engine
from ffa.common.entities import Company, resolve_universe
from ffa.common.openai_client import OpenAIClient
from ffa.common.sec_client import SecEdgarClient
from ffa.config import Settings, get_settings
from ffa.ingestion.structured.fetch import fetch_companyfacts
from ffa.ingestion.structured.load import load_facts
from ffa.ingestion.structured.normalize import normalize_facts
from ffa.ingestion.unstructured.chunk import chunk_text
from ffa.ingestion.unstructured.clean import clean_text
from ffa.ingestion.unstructured.embed import embed_chunks
from ffa.ingestion.unstructured.fetch import (
    FilingDiscoverySummary,
    discover_filings,
    fetch_documents,
    summarize_filing_discoveries,
)
from ffa.ingestion.unstructured.load import load_chunks

logger = logging.getLogger(__name__)


class IngestionFailure(TypedDict):
    """One isolated company or filing failure in a complete run."""

    stage: str
    ticker: str
    accession_no: str | None
    error_type: str
    reason: str


class IngestionSummary(TypedDict):
    """Serializable totals emitted by the command-line runner."""

    ticker_count: int
    structured_rows_loaded: int
    filings_discovered: int
    chunks_loaded: int
    skipped: list[FilingDiscoverySummary]
    failures: list[IngestionFailure]


def run_ingestion(settings: Settings | None = None) -> IngestionSummary:
    """Run both incremental ingestion branches for the configured ticker universe."""
    resolved_settings = settings or get_settings()
    engine = create_rw_engine(resolved_settings.database_url)
    embedding_client = OpenAIClient.from_settings(resolved_settings)
    failures: list[IngestionFailure] = []
    structured_rows_loaded = 0
    chunks_loaded = 0

    try:
        with SecEdgarClient.from_settings(resolved_settings) as sec_client:
            companies = resolve_universe(
                client=sec_client,
                settings=resolved_settings,
            )
            for company in companies:
                try:
                    raw = fetch_companyfacts(
                        company,
                        client=sec_client,
                        settings=resolved_settings,
                        use_cache=False,
                    )
                    rows = normalize_facts(raw)
                    loaded = load_facts(
                        rows,
                        engine=engine,
                        settings=resolved_settings,
                    )
                    structured_rows_loaded += loaded
                    logger.info(
                        "Structured ingestion completed for ticker=%s rows=%s",
                        company["ticker"],
                        loaded,
                    )
                except Exception as exc:
                    _record_failure(
                        failures,
                        stage="structured",
                        company=company,
                        accession_no=None,
                        exc=exc,
                    )

            discoveries = []
            for company in companies:
                try:
                    discoveries.append(
                        discover_filings(
                            company,
                            client=sec_client,
                            engine=engine,
                            settings=resolved_settings,
                            use_cache=False,
                        )
                    )
                except Exception as exc:
                    _record_failure(
                        failures,
                        stage="unstructured_discovery",
                        company=company,
                        accession_no=None,
                        exc=exc,
                    )

            discovery_report = summarize_filing_discoveries(discoveries)
            for filing in discovery_report["filings"]:
                try:
                    document = fetch_documents(
                        filing,
                        client=sec_client,
                        engine=engine,
                        settings=resolved_settings,
                    )
                    sections = clean_text(document)
                    chunks = chunk_text(sections, settings=resolved_settings)
                    embedded = embed_chunks(
                        chunks,
                        client=embedding_client,
                        settings=resolved_settings,
                    )
                    loaded = load_chunks(
                        embedded,
                        engine=engine,
                        settings=resolved_settings,
                    )
                    chunks_loaded += loaded
                    logger.info(
                        "Unstructured ingestion completed for ticker=%s accession=%s chunks=%s",
                        filing["ticker"],
                        filing["accession_no"],
                        loaded,
                    )
                except Exception as exc:
                    _record_failure(
                        failures,
                        stage="unstructured_filing",
                        company=Company(
                            ticker=filing["ticker"],
                            cik=filing["cik"],
                            name=filing["company_name"],
                        ),
                        accession_no=filing["accession_no"],
                        exc=exc,
                    )
    finally:
        engine.dispose()

    return IngestionSummary(
        ticker_count=len(companies),
        structured_rows_loaded=structured_rows_loaded,
        filings_discovered=len(discovery_report["filings"]),
        chunks_loaded=chunks_loaded,
        skipped=discovery_report["skipped"],
        failures=failures,
    )


def _record_failure(
    failures: list[IngestionFailure],
    *,
    stage: str,
    company: Company,
    accession_no: str | None,
    exc: Exception,
) -> None:
    """Log one failure without preventing other configured tickers from running."""
    logger.exception(
        "SEC ingestion failed at stage=%s ticker=%s accession=%s",
        stage,
        company["ticker"],
        accession_no,
    )
    failures.append(
        IngestionFailure(
            stage=stage,
            ticker=company["ticker"],
            accession_no=accession_no,
            error_type=type(exc).__name__,
            reason=str(exc),
        )
    )


def main() -> None:
    """Run ingestion, print a machine-readable summary, and fail on partial errors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    summary = run_ingestion()
    print(json.dumps(summary, indent=2))
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
