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
from ffa.ingestion.unstructured.fetch import (
    discover_filings,
    list_filings,
    parse_submission_filings,
    summarize_filing_discoveries,
)
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


class StatefulChunkConnection(FakeWriteConnection):
    """Track the final chunk keys produced by loader statements."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: dict[tuple[str, str, int], dict[str, object]] = {}

    def execute(self, statement: object, params: dict[str, object]) -> FakeResult:
        result = super().execute(statement, params)
        sql = str(statement)
        accession = str(params.get("accession_no", ""))
        if "DELETE FROM doc_chunks" in sql:
            self.rows = {key: value for key, value in self.rows.items() if key[0] != accession}
        elif "INSERT INTO doc_chunks" in sql:
            key = (accession, str(params["section"]), int(params["chunk_index"]))
            self.rows[key] = dict(params)
        return result


class StatefulChunkEngine:
    """Transactional engine substitute retaining chunk rows across loads."""

    def __init__(self) -> None:
        self.connection = StatefulChunkConnection()

    def begin(self) -> StatefulChunkConnection:
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


def test_discovery_skips_company_without_10k_and_reports_other_company(
    caplog: pytest.LogCaptureFixture,
) -> None:
    apple_client = FakeSecClient(
        {
            "cik": "0000320193",
            "filings": {
                "recent": _submission_columns([("0000320193-25-000079", "10-K", "2025-10-31")])
            },
        },
        {},
    )
    xom_company = {"ticker": "XOM", "cik": 2115436, "name": "ExxonMobil Holdings Corp"}
    xom_client = FakeSecClient(
        {
            "cik": "0002115436",
            "filings": {
                "recent": _submission_columns([("0002115436-26-000001", "8-K", "2026-07-01")])
            },
        },
        {},
    )
    settings = Settings(_env_file=None)

    with caplog.at_level(logging.INFO):
        apple = discover_filings(
            COMPANY,
            client=apple_client,  # type: ignore[arg-type]
            engine=FakeEngine(None),  # type: ignore[arg-type]
            settings=settings,
        )
        xom = discover_filings(
            xom_company,
            client=xom_client,  # type: ignore[arg-type]
            engine=FakeEngine(None),  # type: ignore[arg-type]
            settings=settings,
        )
        report = summarize_filing_discoveries([apple, xom])

    assert [summary["ticker"] for summary in report["ok"]] == ["AAPL"]
    assert [filing["ticker"] for filing in report["filings"]] == ["AAPL"]
    assert report["skipped"] == [
        {
            "ticker": "XOM",
            "cik": 2115436,
            "status": "SKIPPED",
            "filing_count": 0,
            "reason": "No 10-K or 10-K/A filing exists in SEC submissions.",
        }
    ]
    assert "ticker=XOM cik=2115436" in caplog.text
    assert "SKIPPED=[{'ticker': 'XOM', 'cik': 2115436" in caplog.text


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


def test_clean_repairs_repeated_item_headers_and_prioritizes_unique_anchor() -> None:
    toc_noise = "A misleading table-of-contents description that must not win. " * 20
    document = _document(
        form_type="10-K",
        html=f"""
        <html><body>
          <h2>Item 1A. Risk Factors</h2>
          <p>{toc_noise}</p>
          <h2>Item 1B. Unresolved Staff Comments</h2>
          <h2 id="item_1a_risk_factors">Item 1A. Risk Factors</h2>
          <p>The first page explains semiconductor supply risk.</p>
          <hr style="page-break-after: always"/><div>PART I</div><div>Item 1A</div>
          <p>The second page explains regulatory risk.</p>
          <hr style="page-break-after: always"/><div>PART I</div><div>Item 1A</div>
          <p>The third page explains cybersecurity risk.</p>
          <hr style="page-break-after: always"/><div>PART I</div><div>Item 1A</div>
          <p>The fourth page explains competition risk.</p>
          <h2>Item 1B. Unresolved Staff Comments</h2>
        </body></html>
        """,
    )

    risk_text = {section["section"]: section["text"] for section in clean_text(document)}[
        "Risk Factors"
    ]

    assert "first page" in risk_text
    assert "second page" in risk_text
    assert "third page" in risk_text
    assert "fourth page" in risk_text
    assert "misleading table-of-contents" not in risk_text
    assert "PART I" not in risk_text
    assert "Item 1A" not in risk_text


def test_clean_keeps_risk_section_open_across_repeated_part_headers() -> None:
    document = _document(
        form_type="10-K",
        html="""
        <html><body>
          <h2>Item 1A. Risk Factors</h2>
          <p>Page one discusses credit risk and liquidity pressure.</p>
          <hr style="page-break-after: always"/><div>Part I</div>
          <p>Page two discusses operational resilience.</p>
          <hr style="page-break-after: always"/><div>Part I</div>
          <p>Page three discusses regulatory investigations.</p>
          <hr style="page-break-after: always"/><div>Part I</div>
          <p>Page four discusses employee retention.</p>
          <h2>Item 1B. Unresolved Staff Comments</h2>
        </body></html>
        """,
    )

    risk_text = {section["section"]: section["text"] for section in clean_text(document)}[
        "Risk Factors"
    ]

    assert all(f"Page {word}" in risk_text for word in ("one", "two", "three", "four"))
    assert "Part I" not in risk_text


def test_clean_repairs_jpm_style_incorporated_sections_and_ignores_toc() -> None:
    document = _document(
        form_type="10-K",
        html="""
        <html><body>
          <h2>Item 7. Management's Discussion and Analysis</h2>
          <p>Management's discussion and analysis appears on pages 46-160.</p>
          <h2>Item 7A. Market Risk</h2>
          <h2>Item 8. Financial Statements and Supplementary Data</h2>
          <p>The Consolidated Financial Statements and Notes appear on pages 165-314.</p>
          <h2>Item 9. Changes in Accountants</h2>
          <h2>Item 15. Exhibits and Financial Statement Schedules</h2>

          <div style="font-weight:700">Management's discussion and analysis:</div>
          <p>Table of contents entry for management discussion.</p>
          <div style="font-weight:700">Consolidated Financial Statements</div>
          <div style="font-weight:700">Notes to consolidated financial statements:</div>
          <p>Table of contents entry for the notes.</p>
          <div style="font-weight:700">Supplementary Information:</div>

          <div style="font-weight:700">Management's discussion and analysis</div>
          <p>Management explains consumer banking revenue and credit costs.</p>
          <div style="font-weight:700">Management's discussion and analysis</div>
          <p>Management explains capital, liquidity, and market risk.</p>
          <div style="font-weight:700">
            Management's report on internal control over financial reporting
          </div>
          <p>This internal-control report must not contaminate MD&amp;A.</p>

          <div style="font-weight:700">Notes to consolidated financial statements</div>
          <p>Note 1 describes the basis of presentation and consolidation.</p>
          <div style="font-weight:700">Notes to consolidated financial statements</div>
          <div>(Continued)</div>
          <p>Note 2 describes fair value measurements.</p>
          <div style="font-weight:700">Supplementary Information: Distribution of assets</div>
          <p>Supplementary schedules must not contaminate Notes.</p>
        </body></html>
        """,
    )

    sections = {section["section"]: section["text"] for section in clean_text(document)}

    assert "consumer banking revenue" in sections["MD&A"]
    assert "capital, liquidity" in sections["MD&A"]
    assert "Table of contents entry" not in sections["MD&A"]
    assert "internal-control report" not in sections["MD&A"]
    assert "basis of presentation" in sections["Notes"]
    assert "fair value measurements" in sections["Notes"]
    assert "Table of contents entry" not in sections["Notes"]
    assert "Supplementary schedules" not in sections["Notes"]
    assert "(Continued)" not in sections["Notes"]


def test_clean_repairs_nvda_item_8_reference_through_item_15_and_stops_at_schedule() -> None:
    document = _document(
        form_type="10-K",
        html="""
        <html><body>
          <h2>Item 8. Financial Statements and Supplementary Data</h2>
          <p>The information required by this Item is set forth in our Consolidated
             Financial Statements and Notes included in this Annual Report on Form 10-K.</p>
          <h2>Item 9. Changes in Accountants</h2>
          <h2>Item 15. Exhibits and Financial Statement Schedules</h2>
          <div style="font-weight:700">Notes to the Consolidated Financial Statements</div>
          <p>Note 1 describes NVIDIA's organization and accounting policies.</p>
          <div>Table of Contents</div><div>NVIDIA Corporation and Subsidiaries</div>
          <div style="font-weight:700">Notes to the Consolidated Financial Statements</div>
          <div>(Continued)</div>
          <p>Note 2 describes business combinations and fair value.</p>
          <div>Table of Contents</div><div>NVIDIA Corporation and Subsidiaries</div>
          <div style="font-weight:700">Notes to the Consolidated Financial Statements</div>
          <div>(Continued)</div>
          <p>Note 17 describes leases and contractual obligations.</p>
          <div>Table of Contents</div><div>NVIDIA Corporation and Subsidiaries</div>
          <div style="font-weight:700">Notes to the Consolidated Financial Statements</div>
          <div>(Continued)</div>
          <div style="font-weight:700">Schedule II - Valuation and Qualifying Accounts</div>
          <p>The valuation schedule must not be captured.</p>
          <div style="font-weight:700">Exhibit Index</div>
          <h2>Item 16. Form 10-K Summary</h2>
          <h2>Signatures</h2>
        </body></html>
        """,
    )

    notes = {section["section"]: section["text"] for section in clean_text(document)}["Notes"]

    assert "organization and accounting policies" in notes
    assert "business combinations" in notes
    assert "leases and contractual obligations" in notes
    assert "Table of Contents" not in notes
    assert "NVIDIA Corporation and Subsidiaries" not in notes
    assert "(Continued)" not in notes
    assert "Schedule II" not in notes
    assert "valuation schedule" not in notes
    assert "Exhibit Index" not in notes
    assert "Item 16" not in notes
    assert "Signatures" not in notes


def test_clean_does_not_promote_alternative_title_inside_plain_prose(
    caplog: pytest.LogCaptureFixture,
) -> None:
    document = _document(
        form_type="10-K",
        html="""
        <html><body>
          <h2>Item 8. Financial Statements and Supplementary Data</h2>
          <p>The Notes appear on pages 100-180.</p>
          <h2>Item 9. Changes in Accountants</h2>
          <p>Readers often call this material Notes to Consolidated Financial Statements.</p>
          <p>Notes to Consolidated Financial Statements</p>
          <p>This plain paragraph is not a validated section heading.</p>
          <div style="font-weight:700">Supplementary Information</div>
        </body></html>
        """,
    )

    with caplog.at_level(logging.WARNING):
        notes = {section["section"]: section["text"] for section in clean_text(document)}["Notes"]

    assert "plain paragraph" not in notes
    assert "appear on pages" in notes
    assert "section=Notes" in caplog.text
    assert "ticker=AAPL cik=320193 accession=0000320193-25-000079" in caplog.text
    assert "heading_candidates=" in caplog.text
    assert "no allow-listed standalone heading candidate" in caplog.text


def test_clean_fails_closed_when_alternative_heading_has_no_reliable_end(
    caplog: pytest.LogCaptureFixture,
) -> None:
    document = _document(
        form_type="10-K",
        html="""
        <html><body>
          <h2>Item 7. Management's Discussion and Analysis</h2>
          <p>Management's discussion and analysis appears on pages 46-160.</p>
          <h2>Item 7A. Market Risk</h2>
          <div style="font-weight:700">Management's discussion and analysis</div>
          <p>Unbounded narrative that must not replace the safe historical result.</p>
        </body></html>
        """,
    )

    with caplog.at_level(logging.WARNING):
        mda = {section["section"]: section["text"] for section in clean_text(document)}["MD&A"]

    assert "appears on pages" in mda
    assert "Unbounded narrative" not in mda
    assert "without a reliable closing boundary" in caplog.text


def test_clean_recovers_sections_before_a_compact_late_item_index() -> None:
    mda_title = (
        "Management's Discussion and Analysis of Financial Condition and Results of Operations"
    )
    mda_paragraphs = "".join(
        f"<p>Management analysis paragraph {index} discusses sales, margins, and liquidity.</p>"
        for index in range(35)
    )
    risk_paragraphs = "".join(
        f"<p>Risk paragraph {index} explains operational, market, and strategic uncertainty.</p>"
        for index in range(35)
    )
    notes_paragraphs = "".join(
        f"<p>Accounting policy paragraph {index} explains consolidation and recognition.</p>"
        for index in range(35)
    )
    document = _document(
        form_type="10-K",
        html=f"""
        <html><body>
          <h2>{mda_title}</h2>
          {mda_paragraphs}
          <div>Example Corp 2025 Annual Report 7</div>
          <h2>Risk Factors</h2>
          {risk_paragraphs}
          <h2>Properties</h2>
          <p>Property descriptions must not contaminate risk factors.</p>
          <p>Notes to Consolidated Financial Statements</p>
          <p>Summary of Significant Accounting Policies</p>
          {notes_paragraphs}
          <p>Report of Independent Registered Public Accounting Firm</p>
          <p>The audit report must not contaminate the notes.</p>

          <h2>Part I</h2>
          <p>Item 1 Business</p>
          <p>Item 1A Risk Factors</p>
          <p>Item 1B Unresolved Staff Comments</p>
          <p>Item 1C Cybersecurity</p>
          <p>Item 2 Properties</p>
          <p>Item 3 Legal Proceedings</p>
          <h2>Part II</h2>
          <p>Item 5 Market for Common Equity</p>
          <p>Item 7 Management's Discussion and Analysis</p>
          <p>Item 7A Market Risk</p>
          <p>Item 8 Financial Statements and Supplementary Data</p>
          <p>Item 9 Changes in Accountants</p>
          <h2>Part IV</h2>
          <p>Item 15 Exhibits and Financial Statement Schedules</p>
          <p>Item 16 Form 10-K Summary</p>
        </body></html>
        """,
    )

    sections = {section["section"]: section["text"] for section in clean_text(document)}

    assert "Management analysis paragraph 34" in sections["MD&A"]
    assert "Risk paragraph 34" in sections["Risk Factors"]
    assert "Accounting policy paragraph 34" in sections["Notes"]
    assert "Property descriptions" not in sections["Risk Factors"]
    assert "audit report must not contaminate" not in sections["Notes"]
    assert "Item 15 Exhibits" not in "\n".join(sections.values())
    assert "2025 Annual Report 7" not in "\n".join(sections.values())


def test_clean_selects_substantive_accounting_notes_instead_of_toc_occurrence() -> None:
    accounting_body = "".join(
        f"<p>Revenue recognition narrative {index} describes a substantive accounting policy.</p>"
        for index in range(45)
    )
    document = _document(
        form_type="10-K",
        html=f"""
        <html><body>
          <h2>Notes to the Consolidated Financial Statements</h2>
          <h3>Note 1 - Basis of Presentation</h3>
          <p>Table of contents entry only.</p>
          <h2>Report of Independent Registered Public Accounting Firm</h2>

          <h2>Notes to the Consolidated Financial Statements</h2>
          <h3>Note 1 - Basis of Presentation</h3>
          {accounting_body}
          <h2>Report of Independent Registered Public Accounting Firm</h2>
          <p>Audit opinion outside the accounting notes.</p>

          <h2>Item 8. Financial Statements and Supplementary Data</h2>
          <p>The statements are included earlier in this report.</p>
          <h2>Item 9. Changes in Accountants</h2>
        </body></html>
        """,
    )

    notes = {section["section"]: section["text"] for section in clean_text(document)}["Notes"]

    assert "Revenue recognition narrative 44" in notes
    assert "Table of contents entry only" not in notes
    assert "Audit opinion outside" not in notes


def test_clean_accepts_unstyled_notes_title_only_with_nearby_note_one() -> None:
    accounting_body = "".join(
        f"<p>Oil and gas accounting narrative {index} contains substantive policy detail.</p>"
        for index in range(45)
    )
    document = _document(
        form_type="10-K",
        html=f"""
        <html><body>
          <h2>Item 8. Financial Statements and Supplementary Data</h2>
          <p>The financial table of contents identifies the statements.</p>
          <h2>Item 9. Changes in Accountants</h2>
          <p>Notes to the Consolidated Financial Statements</p>
          <p>Financial Table of Contents</p>
          <p>Millions of dollars, except per-share amounts</p>
          <h3>Note 1</h3>
          {accounting_body}
          <p>Notes to the Consolidated Financial Statements</p>
          <h3>Note 1</h3>
          <p>Continuation text remains in the same accounting body.</p>
          <h2>Part IV</h2>
          <h2>Item 15. Exhibits and Financial Statement Schedules</h2>
        </body></html>
        """,
    )

    notes = {section["section"]: section["text"] for section in clean_text(document)}["Notes"]

    assert "Oil and gas accounting narrative 44" in notes
    assert "Continuation text" in notes
    assert "Part IV" not in notes
    assert "Financial Table of Contents" not in notes
    assert "Millions of dollars" not in notes


def test_clean_accepts_eof_for_unambiguous_post_item_15_accounting_notes() -> None:
    accounting_body = "".join(
        f"<p>Segment accounting narrative {index} describes the business and its policies.</p>"
        for index in range(45)
    )
    document = _document(
        form_type="10-K",
        html=f"""
        <html><body>
          <h2>Item 8. Financial Statements and Supplementary Data</h2>
          <p>See Index to Financial Statements and Supplemental Data.</p>
          <h2>Item 9. Changes in Accountants</h2>
          <h2>Item 15. Exhibits and Financial Statement Schedules</h2>
          <h2>Notes to Consolidated Financial Statements</h2>
          <p>Index entry followed by non-accounting front matter.</p>
          <p>More financial-statement front matter.</p>
          <p>Still no first accounting note near this title.</p>
          <p>Additional front matter.</p>
          <p>Additional front matter.</p>
          <p>Additional front matter.</p>
          <p>Additional front matter.</p>
          <p>Additional front matter.</p>
          <p>Additional front matter.</p>
          <p>Additional front matter.</p>
          <p>Additional front matter.</p>
          <p>Additional front matter.</p>
          <h2>Notes to Consolidated Financial Statements</h2>
          <h3>1 Description of the Business and Segment Information</h3>
          {accounting_body}
        </body></html>
        """,
    )

    notes = {section["section"]: section["text"] for section in clean_text(document)}["Notes"]

    assert "Segment accounting narrative 44" in notes
    assert "Index entry followed" not in notes


def test_clean_keeps_historical_notes_when_accounting_candidates_are_ambiguous(
    caplog: pytest.LogCaptureFixture,
) -> None:
    first_body = "".join(
        f"<p>First candidate policy {index} has equally substantive accounting detail.</p>"
        for index in range(45)
    )
    second_body = "".join(
        f"<p>Second candidate policy {index} has equally substantive accounting detail.</p>"
        for index in range(45)
    )
    document = _document(
        form_type="10-K",
        html=f"""
        <html><body>
          <h2>Item 8. Financial Statements and Supplementary Data</h2>
          <p>Safe historical Notes result.</p>
          <h2>Item 9. Changes in Accountants</h2>
          <h2>Notes to Consolidated Financial Statements</h2>
          <h3>Note 1 - First Candidate</h3>
          {first_body}
          <h2>Report of Independent Registered Public Accounting Firm</h2>
          <h2>Notes to Consolidated Financial Statements</h2>
          <h3>Note 1 - Second Candidate</h3>
          {second_body}
          <h2>Report of Independent Registered Public Accounting Firm</h2>
        </body></html>
        """,
    )

    with caplog.at_level(logging.WARNING):
        notes = {section["section"]: section["text"] for section in clean_text(document)}["Notes"]

    assert notes == "Financial Statements and Supplementary Data\nSafe historical Notes result."
    assert "multiple accounting Notes bodies remained ambiguous" in caplog.text


def test_clean_logs_unfetched_companion_document_and_fails_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    document = _document(
        form_type="10-K",
        html="""
        <html><body>
          <h2>Part I</h2>
          <h2>Item 1A. Risk Factors</h2>
          <p>Information can be found in the 2025 Annual Report to Shareholders and is
             incorporated into this item by reference.</p>
          <h2>Part II</h2>
          <h2>Item 7. Management's Discussion and Analysis</h2>
          <p>Information can be found in the 2025 Annual Report to Shareholders and is
             incorporated into this item by reference.</p>
          <h2>Item 8. Financial Statements and Supplementary Data</h2>
          <p>Information can be found in the 2025 Annual Report to Shareholders and is
             incorporated into this item by reference.</p>
          <h2>Item 9. Changes in Accountants</h2>
        </body></html>
        """,
    )

    with caplog.at_level(logging.WARNING):
        sections = {section["section"]: section["text"] for section in clean_text(document)}

    assert set(sections) == {"MD&A", "Notes", "Risk Factors"}
    assert all("Annual Report to Shareholders" in value for value in sections.values())
    assert "content incorporated in companion document not fetched" in caplog.text
    assert "ticker=AAPL cik=320193 accession=0000320193-25-000079" in caplog.text


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
    delete_calls = [call for call in engine.connection.calls if "DELETE FROM doc_chunks" in call[0]]
    assert first_count == 2
    assert second_count == 2
    assert len(chunk_calls) == 4
    assert len(delete_calls) == 2
    assert all(params == {"accession_no": "0000320193-25-000079"} for _, params in delete_calls)
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


def test_load_chunks_replaces_three_chunks_with_two_without_orphans() -> None:
    engine = StatefulChunkEngine()
    settings = Settings(_env_file=None, embedding_dim=3)

    assert (
        load_chunks(
            _embedded_chunks(3),
            engine=engine,  # type: ignore[arg-type]
            settings=settings,
        )
        == 3
    )
    assert (
        load_chunks(
            _embedded_chunks(2),
            engine=engine,  # type: ignore[arg-type]
            settings=settings,
        )
        == 2
    )

    assert sorted(engine.connection.rows) == [
        ("0000320193-25-000079", "MD&A", 0),
        ("0000320193-25-000079", "MD&A", 1),
    ]


def test_load_chunks_replaces_two_chunks_with_five_without_gaps() -> None:
    engine = StatefulChunkEngine()
    settings = Settings(_env_file=None, embedding_dim=3)

    load_chunks(
        _embedded_chunks(2),
        engine=engine,  # type: ignore[arg-type]
        settings=settings,
    )
    assert (
        load_chunks(
            _embedded_chunks(5),
            engine=engine,  # type: ignore[arg-type]
            settings=settings,
        )
        == 5
    )

    assert sorted(key[2] for key in engine.connection.rows) == [0, 1, 2, 3, 4]
    assert len(engine.connection.rows) == 5


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


def _embedded_chunks(count: int) -> list[dict[str, object]]:
    return [
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
            "text": f"replacement chunk {index}",
            "token_count": 3,
            "source_url": "https://www.sec.gov/filing.htm",
            "embedding": [0.1, 0.2, 0.3],
        }
        for index in range(count)
    ]


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
