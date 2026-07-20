"""Tests for unstructured SEC filing ingestion."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
from bs4 import XMLParsedAsHTMLWarning

from ffa.config import Settings
from ffa.ingestion.unstructured.chunk import chunk_text
from ffa.ingestion.unstructured.clean import clean_text
from ffa.ingestion.unstructured.embed import embed_chunks
from ffa.ingestion.unstructured.fetch import list_filings, parse_submission_filings
from ffa.ingestion.unstructured.load import load_chunks

COMPANY = {"ticker": "AAPL", "cik": 320193, "name": "Apple Inc."}


class WordEncoder:
    """Deterministic word tokenizer used to test chunk boundaries."""

    def __init__(self) -> None:
        self._tokens: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for word in text.split():
            if word not in self._tokens:
                token_id = len(self._tokens) + 1
                self._tokens[word] = token_id
                self._words[token_id] = word
            token_ids.append(self._tokens[word])
        return token_ids

    def decode(self, tokens: Sequence[int]) -> str:
        return " ".join(self._words[token] for token in tokens)


class CharacterEncoder:
    """Treat every character as one token to expose subword-cut regressions."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


class FakeEmbeddingProvider:
    """Record embedding batches and return small deterministic vectors."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        dimensions: int,
    ) -> list[list[float]]:
        self.calls.append((list(texts), dimensions))
        return [[float(index)] * dimensions for index, _ in enumerate(texts)]


class FakeResult:
    """Minimal SQLAlchemy result substitute for ingestion-state tests."""

    def __init__(self, row: object) -> None:
        self._row = row

    def one_or_none(self) -> object:
        return self._row


class FakeConnection:
    """Context-managed connection returning one ingestion cursor."""

    def __init__(self, row: object) -> None:
        self._row = row

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, statement: object, params: object) -> FakeResult:
        return FakeResult(self._row)


class FakeEngine:
    """Minimal engine substitute for filing-discovery tests."""

    def __init__(self, row: object) -> None:
        self._row = row

    def connect(self) -> FakeConnection:
        return FakeConnection(self._row)


class FakeWriteConnection:
    """Record statements executed by the chunk loader."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __enter__(self) -> FakeWriteConnection:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, statement: object, params: dict[str, object]) -> FakeResult:
        self.calls.append((str(statement), params))
        return FakeResult(None)


class FakeWriteEngine:
    """Minimal transactional engine substitute for loader tests."""

    def __init__(self) -> None:
        self.connection = FakeWriteConnection()

    def begin(self) -> FakeWriteConnection:
        return self.connection


class FakeSecClient:
    """SEC client substitute serving current and historical submissions."""

    def __init__(self, root: dict[str, Any], history: dict[str, dict[str, Any]]) -> None:
        self._root = root
        self._history = history
        self.history_urls: list[str] = []

    def fetch_submissions(self, cik: int, *, use_cache: bool = True) -> dict[str, Any]:
        return self._root

    def get_json(self, url: str, *, use_cache: bool = True) -> dict[str, Any]:
        self.history_urls.append(url)
        return self._history[url.rsplit("/", 1)[-1]]


def test_submission_parser_filters_forms_and_canonicalizes_amendments() -> None:
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-25-000001",
                    "0000320193-25-000002",
                    "0000320193-25-000003",
                ],
                "form": ["10-Q", "10-K/A", "8-K"],
                "filingDate": ["2025-05-02", "2025-11-01", "2025-11-02"],
                "reportDate": ["2025-03-29", "2025-09-27", "2025-10-31"],
                "primaryDocument": ["aapl-20250329.htm", "aapl-20250927.htm", "event.htm"],
            }
        }
    }

    filings = parse_submission_filings(payload, COMPANY)

    assert [filing["form_type"] for filing in filings] == ["10-Q", "10-K"]
    assert filings[0]["period_of_report"] == date(2025, 3, 29)
    assert filings[0]["fiscal_year"] is None
    assert filings[0]["fiscal_period"] is None
    assert filings[0]["primary_doc_url"] == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/aapl-20250329.htm"
    )


def test_list_filings_uses_state_and_skips_irrelevant_history_files() -> None:
    root = {
        "cik": "0000320193",
        "filings": {
            "recent": _submission_columns(
                [
                    ("0000320193-25-000010", "10-Q", "2025-05-02"),
                    ("0000320193-25-000012", "10-Q", "2025-05-03"),
                ]
            ),
            "files": [
                {"name": "old.json", "filingTo": "2024-12-31"},
                {"name": "boundary.json", "filingTo": "2025-05-02"},
            ],
        },
    }
    client = FakeSecClient(
        root,
        {
            "boundary.json": _submission_columns(
                [
                    ("0000320193-25-000009", "10-K", "2025-05-01"),
                    ("0000320193-25-000011", "10-K/A", "2025-05-02"),
                ]
            )
        },
    )
    engine = FakeEngine(
        SimpleNamespace(
            last_filing_date=date(2025, 5, 2),
            last_accession="0000320193-25-000010",
        )
    )

    filings = list_filings(
        COMPANY,
        client=client,  # type: ignore[arg-type]
        engine=engine,  # type: ignore[arg-type]
        settings=Settings(_env_file=None),
    )

    assert [filing["accession_no"] for filing in filings] == [
        "0000320193-25-000011",
        "0000320193-25-000012",
    ]
    assert filings[0]["form_type"] == "10-K"
    assert len(client.history_urls) == 1
    assert client.history_urls[0].endswith("/boundary.json")


def test_list_filings_skips_missing_primary_document_and_keeps_other_filings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = {
        "cik": "0000320193",
        "filings": {
            "recent": _submission_columns([("0000320193-25-000020", "10-Q", "2025-08-01")]),
            "files": [{"name": "history.json", "filingTo": "2025-07-31"}],
        },
    }
    history = _submission_columns(
        [
            ("0000320193-24-000001", "10-K", "2024-11-01"),
            ("0000320193-25-000010", "10-Q", "2025-05-02"),
        ]
    )
    history["primaryDocument"][0] = ""
    client = FakeSecClient(root, {"history.json": history})
    engine = FakeEngine(None)

    with caplog.at_level(logging.WARNING):
        filings = list_filings(
            COMPANY,
            client=client,  # type: ignore[arg-type]
            engine=engine,  # type: ignore[arg-type]
            settings=Settings(_env_file=None),
        )

    assert [filing["accession_no"] for filing in filings] == [
        "0000320193-25-000010",
        "0000320193-25-000020",
    ]
    assert "Skipping SEC filing 0000320193-24-000001" in caplog.text


def test_clean_10k_keeps_largest_canonical_sections_and_visible_inline_xbrl() -> None:
    document = _document(
        form_type="10-K",
        html="""
        <html><body>
          <script>remove script content</script>
          <div style="display: none">remove hidden content</div>
          <h2>Item 1A. Risk Factors</h2><p>Table of contents entry</p>
          <h2>Item 7. Management's Discussion and Analysis</h2><p>Contents</p>
          <h2>Item 8. Financial Statements and Supplementary Data</h2><p>Contents</p>
          <h2><span>Item</span> <span>1A.</span> <span>Risk Factors</span></h2>
          <p>Demand volatility and supply constraints could adversely affect operations.</p>
          <p>Foreign exchange movements create additional uncertainty.</p>
          <h2>Item 1B. Unresolved Staff Comments</h2>
          <h2>Item 7. Management's Discussion and Analysis</h2>
          <p>Net sales changed because product mix and services demand evolved.</p>
          <p><ix:nonNumeric>Management monitors liquidity and capital resources.</ix:nonNumeric></p>
          <h2>Item 7A. Quantitative and Qualitative Disclosures</h2>
          <h2>Item 8. Financial Statements and Supplementary Data</h2>
          <p>Notes to Consolidated Financial Statements</p>
          <p>Revenue is recognized when control transfers to the customer.</p>
          <h2>Item 9. Changes in and Disagreements With Accountants</h2>
        </body></html>
        """,
    )

    sections = clean_text(document)
    by_name = {section["section"]: section for section in sections}

    assert set(by_name) == {"MD&A", "Notes", "Risk Factors"}
    assert "Demand volatility" in by_name["Risk Factors"]["text"]
    assert "Table of contents entry" not in by_name["Risk Factors"]["text"]
    assert "Management monitors liquidity" in by_name["MD&A"]["text"]
    assert "remove script content" not in " ".join(section["text"] for section in sections)
    assert "remove hidden content" not in " ".join(section["text"] for section in sections)
    assert by_name["Notes"]["fiscal_year"] == 2025
    assert by_name["Notes"]["source_url"].endswith("aapl-20250927.htm")


def test_clean_10q_uses_part_to_disambiguate_item_numbers() -> None:
    document = _document(
        form_type="10-Q",
        html="""
        <html><body>
          <h1>Part I. Financial Information</h1>
          <h2>Item 1. Financial Statements</h2>
          <p>Notes describe the interim accounting policies.</p>
          <h2>Item 2. Management's Discussion and Analysis</h2>
          <p>Management discusses operating results and liquidity.</p>
          <h1>Part II. Other Information</h1>
          <h2>Item 1A. Risk Factors</h2>
          <p>Market and operational risks could affect future results.</p>
          <h2>Item 2. Unregistered Sales of Equity Securities</h2>
        </body></html>
        """,
    )

    sections = {section["section"]: section["text"] for section in clean_text(document)}

    assert "interim accounting policies" in sections["Notes"]
    assert "operating results" in sections["MD&A"]
    assert "operational risks" in sections["Risk Factors"]
    assert "Unregistered Sales" not in sections["MD&A"]


def test_clean_xhtml_removes_repeated_pagination_and_data_tables() -> None:
    document = _document(
        form_type="10-K",
        html="""<?xml version="1.0" encoding="UTF-8"?>
        <html xmlns="http://www.w3.org/1999/xhtml"
              xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
          <body>
            <h2>Item 7. Management's Discussion and Analysis</h2>
            <p>The company invested $12.5 billion while maintaining operating discipline.</p>
            <div>Apple Inc. | 2025 Form 10-K | 21</div>
            <p>Management continued to monitor demand and liquidity.</p>
            <div>Apple Inc. | 2025 Form 10-K | 22</div>
            <table>
              <tr><th>Segment</th><th>2025</th><th>2024</th><th>Change</th></tr>
              <tr><td>Americas</td><td>178,353</td><td>167,045</td><td>7%</td></tr>
              <tr><td>Europe</td><td>111,032</td><td>101,328</td><td>10%</td></tr>
              <tr><td>Total</td><td>416,161</td><td>391,035</td><td>6%</td></tr>
            </table>
            <div>Apple Inc. | 2025 Form 10-K | 23</div>
            <p>Revenue growth reflected demand for products and services.</p>
            <h2>Item 7A. Market Risk</h2>
          </body>
        </html>
        """,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sections = clean_text(document)
    chunks = chunk_text(sections, max_tokens=100, overlap=0.15, encoder=WordEncoder())
    combined_text = "\n".join(chunk["text"] for chunk in chunks)

    assert "$12.5 billion" in combined_text
    assert "Apple Inc. | 2025 Form 10-K" not in combined_text
    assert "416,161" not in combined_text
    assert "Revenue growth reflected demand" in combined_text
    assert not any(issubclass(item.category, XMLParsedAsHTMLWarning) for item in caught)


def test_chunk_text_respects_section_boundaries_limit_overlap_and_metadata() -> None:
    sections = clean_text(
        _document(
            form_type="10-K",
            html="""
            <html><body>
              <h2>Item 7. MD&amp;A</h2>
              <p>one two three four five six seven</p>
              <h2>Item 8. Notes</h2>
              <p>alpha beta gamma</p>
              <h2>Item 9. Other</h2>
            </body></html>
            """,
        )
    )

    chunks = chunk_text(
        sections,
        max_tokens=4,
        overlap=0.25,
        encoder=WordEncoder(),
    )
    mda_chunks = [chunk for chunk in chunks if chunk["section"] == "MD&A"]
    notes_chunks = [chunk for chunk in chunks if chunk["section"] == "Notes"]

    assert [chunk["chunk_index"] for chunk in mda_chunks] == [0, 1, 2]
    assert all(chunk["token_count"] <= 4 for chunk in chunks)
    assert set(mda_chunks[0]["text"].split()) & set(mda_chunks[1]["text"].split()) == {"three"}
    assert notes_chunks[0]["chunk_index"] == 0
    assert all(chunk["ticker"] == "AAPL" for chunk in chunks)
    assert all(chunk["fiscal_period"] == "FY" for chunk in chunks)


def test_chunk_text_never_splits_words_and_prefers_sentence_endings() -> None:
    text = (
        "Alpha teams improve extraordinaryword carefully. "
        "Beta groups monitor liquidity consistently. "
        "Gamma leaders review operations quarterly."
    )
    chunks = chunk_text(
        [_section(text)],
        max_tokens=52,
        overlap=0.2,
        encoder=CharacterEncoder(),
    )
    source_words = set(text.split())

    assert len(chunks) >= 3
    assert chunks[0]["text"].endswith(".")
    assert all(chunk["text"].split()[0] in source_words for chunk in chunks)
    assert all(chunk["text"].split()[-1] in source_words for chunk in chunks)
    assert all(word in source_words for chunk in chunks for word in chunk["text"].split())
    assert set(chunks[0]["text"].split()) & set(chunks[1]["text"].split())


def test_chunk_text_keeps_one_over_budget_word_intact() -> None:
    chunks = chunk_text(
        [_section("alpha extraordinaryword omega.")],
        max_tokens=10,
        overlap=0.2,
        encoder=CharacterEncoder(),
    )

    assert [chunk["text"] for chunk in chunks] == ["alpha", "extraordinaryword", "omega."]


def test_embed_chunks_batches_and_preserves_metadata() -> None:
    chunks = [
        {
            "ticker": "AAPL",
            "cik": 320193,
            "company_name": "Apple Inc.",
            "accession_no": "0000320193-25-000079",
            "form_type": "10-K",
            "filing_date": date(2025, 10, 31),
            "period_of_report": date(2025, 9, 27),
            "fiscal_year": 2025,
            "fiscal_period": "FY",
            "section": "MD&A",
            "chunk_index": index,
            "text": f"chunk {index}",
            "token_count": 2,
            "source_url": "https://www.sec.gov/filing.htm",
        }
        for index in range(5)
    ]
    provider = FakeEmbeddingProvider()
    settings = Settings(_env_file=None, embedding_dim=3)

    embedded = embed_chunks(chunks, client=provider, settings=settings, batch_size=2)

    assert [len(texts) for texts, _ in provider.calls] == [2, 2, 1]
    assert all(dimensions == 3 for _, dimensions in provider.calls)
    assert len(embedded) == 5
    assert all(len(chunk["embedding"]) == 3 for chunk in embedded)
    assert embedded[3]["chunk_index"] == 3
    assert embedded[3]["section"] == "MD&A"


def test_embed_chunks_rejects_wrong_vector_dimension() -> None:
    class WrongDimensionProvider:
        def embed_texts(
            self,
            texts: Sequence[str],
            *,
            dimensions: int,
        ) -> list[list[float]]:
            return [[0.0] for _ in texts]

    settings = Settings(_env_file=None, embedding_dim=3)

    with pytest.raises(RuntimeError, match="unexpected dimension"):
        embed_chunks(
            [
                {
                    "ticker": "AAPL",
                    "cik": 320193,
                    "company_name": "Apple Inc.",
                    "accession_no": "0000320193-25-000079",
                    "form_type": "10-K",
                    "filing_date": date(2025, 10, 31),
                    "period_of_report": date(2025, 9, 27),
                    "fiscal_year": 2025,
                    "fiscal_period": "FY",
                    "section": "Notes",
                    "chunk_index": 0,
                    "text": "accounting policy",
                    "token_count": 2,
                    "source_url": "https://www.sec.gov/filing.htm",
                }
            ],
            client=WrongDimensionProvider(),
            settings=settings,
        )


def test_load_chunks_executes_idempotent_upserts_and_updates_exact_source() -> None:
    engine = FakeWriteEngine()
    settings = Settings(_env_file=None, embedding_dim=3)
    chunks = [
        {
            "ticker": "AAPL",
            "cik": 320193,
            "company_name": "Apple Inc.",
            "accession_no": "0000320193-25-000079",
            "form_type": "10-K",
            "filing_date": date(2025, 10, 31),
            "period_of_report": date(2025, 9, 27),
            "fiscal_year": 2025,
            "fiscal_period": "FY",
            "section": "MD&A",
            "chunk_index": index,
            "text": f"chunk {index}",
            "token_count": 2,
            "source_url": "https://www.sec.gov/filing.htm",
            "embedding": [0.1, 0.2, 0.3],
        }
        for index in range(2)
    ]

    first_count = load_chunks(
        chunks,
        engine=engine,  # type: ignore[arg-type]
        settings=settings,
    )
    second_count = load_chunks(
        chunks,
        engine=engine,  # type: ignore[arg-type]
        settings=settings,
    )

    chunk_calls = [call for call in engine.connection.calls if "INSERT INTO doc_chunks" in call[0]]
    state_calls = [
        call for call in engine.connection.calls if "INSERT INTO ingestion_state" in call[0]
    ]
    assert first_count == 2
    assert second_count == 2
    assert len(chunk_calls) == 4
    assert all(
        "ON CONFLICT (accession_no, section, chunk_index) DO UPDATE" in statement
        for statement, _ in chunk_calls
    )
    assert all(
        params["embedding"] == "[0.10000000000000001,0.20000000000000001,0.29999999999999999]"
        for _, params in chunk_calls
    )
    assert len(state_calls) == 2
    assert all(params["source"] == "SEC_EDGAR_UNSTRUCTURED" for _, params in state_calls)


def _document(*, form_type: str, html: str) -> dict[str, object]:
    return {
        "filing": {
            "ticker": "AAPL",
            "cik": 320193,
            "company_name": "Apple Inc.",
            "accession_no": "0000320193-25-000079",
            "form_type": form_type,
            "filing_date": date(2025, 10, 31),
            "period_of_report": date(2025, 9, 27),
            "primary_document": "aapl-20250927.htm",
            "primary_doc_url": (
                "https://www.sec.gov/Archives/edgar/data/320193/"
                "000032019325000079/aapl-20250927.htm"
            ),
            "fiscal_year": 2025,
            "fiscal_period": "FY",
        },
        "html": html,
    }


def _section(text: str) -> dict[str, object]:
    return {
        "ticker": "AAPL",
        "cik": 320193,
        "company_name": "Apple Inc.",
        "accession_no": "0000320193-25-000079",
        "form_type": "10-K",
        "filing_date": date(2025, 10, 31),
        "period_of_report": date(2025, 9, 27),
        "fiscal_year": 2025,
        "fiscal_period": "FY",
        "section": "MD&A",
        "text": text,
        "source_url": "https://www.sec.gov/filing.htm",
    }


def _submission_columns(
    records: list[tuple[str, str, str]],
) -> dict[str, list[str]]:
    return {
        "accessionNumber": [record[0] for record in records],
        "form": [record[1] for record in records],
        "filingDate": [record[2] for record in records],
        "reportDate": ["2025-03-29" for _ in records],
        "primaryDocument": ["filing.htm" for _ in records],
    }
