"""Tests for the complete command-line ingestion orchestrator."""

from __future__ import annotations

from typing import Any

import ffa.ingestion.run as ingestion_run
from ffa.config import Settings


class FakeEngine:
    """Record disposal of the shared ingestion engine."""

    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        """Mark the engine as disposed."""
        self.disposed = True


class FakeSecClient:
    """Minimal context-managed SEC client."""

    def __enter__(self) -> FakeSecClient:
        """Return the fake client."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the fake context."""
        return None


def test_run_ingestion_orchestrates_both_branches(monkeypatch: Any) -> None:
    """The CLI runner reuses every existing stage and returns complete totals."""
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        sec_user_agent="TestAgent test@example.com",
        ticker_universe="AAPL",
        database_url="postgresql+psycopg://app:pass@db/ffa",
    )
    company = {"ticker": "AAPL", "cik": 320193, "name": "Apple Inc."}
    filing = {
        "ticker": "AAPL",
        "cik": 320193,
        "company_name": "Apple Inc.",
        "accession_no": "0000320193-25-000079",
    }
    engine = FakeEngine()
    sec_client = FakeSecClient()
    embedding_client = object()
    calls: list[str] = []

    monkeypatch.setattr(ingestion_run, "create_rw_engine", lambda url: engine)
    monkeypatch.setattr(
        ingestion_run.OpenAIClient,
        "from_settings",
        lambda resolved_settings: embedding_client,
    )
    monkeypatch.setattr(
        ingestion_run.SecEdgarClient,
        "from_settings",
        lambda resolved_settings: sec_client,
    )
    monkeypatch.setattr(
        ingestion_run,
        "resolve_universe",
        lambda **kwargs: [company],
    )
    monkeypatch.setattr(
        ingestion_run,
        "fetch_companyfacts",
        lambda *args, **kwargs: calls.append("structured_fetch") or {"raw": True},
    )
    monkeypatch.setattr(
        ingestion_run,
        "normalize_facts",
        lambda raw: calls.append("normalize") or [{"row": True}],
    )
    monkeypatch.setattr(
        ingestion_run,
        "load_facts",
        lambda *args, **kwargs: calls.append("structured_load") or 3,
    )
    monkeypatch.setattr(
        ingestion_run,
        "discover_filings",
        lambda *args, **kwargs: (
            calls.append("discover")
            or {
                "ticker": "AAPL",
                "cik": 320193,
                "status": "OK",
                "filing_count": 1,
                "reason": None,
                "filings": [filing],
            }
        ),
    )
    monkeypatch.setattr(
        ingestion_run,
        "summarize_filing_discoveries",
        lambda discoveries: {
            "filings": [filing],
            "ok": [],
            "skipped": [],
        },
    )
    monkeypatch.setattr(
        ingestion_run,
        "fetch_documents",
        lambda *args, **kwargs: calls.append("document_fetch") or {"document": True},
    )
    monkeypatch.setattr(
        ingestion_run,
        "clean_text",
        lambda document: calls.append("clean") or [{"section": True}],
    )
    monkeypatch.setattr(
        ingestion_run,
        "chunk_text",
        lambda *args, **kwargs: calls.append("chunk") or [{"chunk": True}],
    )
    monkeypatch.setattr(
        ingestion_run,
        "embed_chunks",
        lambda *args, **kwargs: calls.append("embed") or [{"embedded": True}],
    )
    monkeypatch.setattr(
        ingestion_run,
        "load_chunks",
        lambda *args, **kwargs: calls.append("chunk_load") or 2,
    )

    summary = ingestion_run.run_ingestion(settings)

    assert summary == {
        "ticker_count": 1,
        "structured_rows_loaded": 3,
        "filings_discovered": 1,
        "chunks_loaded": 2,
        "skipped": [],
        "failures": [],
    }
    assert calls == [
        "structured_fetch",
        "normalize",
        "structured_load",
        "discover",
        "document_fetch",
        "clean",
        "chunk",
        "embed",
        "chunk_load",
    ]
    assert engine.disposed is True
