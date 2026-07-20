"""Clean SEC filing HTML and extract narrative sections."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from datetime import date
from typing import TypedDict

from bs4 import BeautifulSoup

from ffa.ingestion.unstructured.fetch import FilingMetadata

_PART_PATTERN = re.compile(r"^PART\s+(I{1,3}|IV)\b", re.IGNORECASE)
_ITEM_PATTERN = re.compile(
    r"^(?:PART\s+(?P<inline_part>I{1,3}|IV)\s+)?"
    r"ITEM\s+(?P<item>\d+[A-Z]?)\s*[.\-–—:]?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")
_PAGE_SUFFIX_PATTERN = re.compile(
    r"(?P<prefix>(?:\|\s*|\bpage\s+))\d+(?:\s+of\s+\d+)?\s*$",
    re.IGNORECASE,
)
_REPEATED_BOILERPLATE_MIN_COUNT = 3
_BLOCK_TAGS = (
    "address",
    "article",
    "blockquote",
    "br",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "p",
    "section",
    "tr",
)

_TEN_K_SECTIONS = {
    "1A": "Risk Factors",
    "7": "MD&A",
    "8": "Notes",
}
_TEN_Q_SECTIONS = {
    ("I", "1"): "Notes",
    ("I", "2"): "MD&A",
    ("II", "1A"): "Risk Factors",
}
_TEN_Q_SECTION_FALLBACK = {
    "1": "Notes",
    "2": "MD&A",
    "1A": "Risk Factors",
}


class SectionText(TypedDict):
    """Clean filing section with metadata required by downstream stages."""

    ticker: str
    cik: int
    company_name: str
    accession_no: str
    form_type: str
    filing_date: date
    period_of_report: date | None
    fiscal_year: int | None
    fiscal_period: str | None
    section: str
    text: str
    source_url: str


def clean_text(document: Mapping[str, object]) -> list[SectionText]:
    """Convert filing HTML into the largest MD&A, risk, and notes sections.

    Repeated headings are common in filing tables of contents. All candidates are
    collected and the longest body for each canonical section is retained.
    """
    filing, html = _unwrap_document(document)
    soup = _parse_filing_markup(html)
    _remove_non_content(soup)
    _remove_data_tables(soup)
    lines = _remove_repeated_boilerplate(_normalized_lines(_block_aware_text(soup)))

    candidates: dict[str, list[str]] = {}
    current_part: str | None = None
    current_section: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        if current_section is not None and current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                candidates.setdefault(current_section, []).append(section_text)
        current_lines = []

    for line in lines:
        part_match = _PART_PATTERN.match(line)
        if part_match is not None:
            flush()
            current_section = None
            current_part = part_match.group(1).upper()
            continue

        item_match = _ITEM_PATTERN.match(line)
        if item_match is not None:
            flush()
            inline_part = item_match.group("inline_part")
            if inline_part is not None:
                current_part = inline_part.upper()
            item = item_match.group("item").upper()
            current_section = _canonical_section(filing["form_type"], current_part, item)
            current_lines = []
            title = _normalize_line(item_match.group("title"))
            if current_section is not None and title:
                current_lines.append(title)
            continue

        if current_section is not None:
            current_lines.append(line)
    flush()

    if not candidates:
        fallback_text = "\n".join(lines).strip()
        if not fallback_text:
            raise ValueError("SEC filing HTML contains no usable text.")
        candidates["Full Filing"] = [fallback_text]

    sections: list[SectionText] = []
    for section_name, variants in candidates.items():
        selected_text = max(variants, key=lambda value: (len(value), value))
        sections.append(_section_row(filing, section_name, selected_text))
    return sorted(sections, key=lambda section: section["section"])


def _unwrap_document(
    document: Mapping[str, object],
) -> tuple[FilingMetadata, str]:
    raw_filing = document.get("filing")
    html = document.get("html")
    if not isinstance(raw_filing, Mapping) or not isinstance(html, str):
        raise ValueError("Filing document must contain filing metadata and HTML text.")
    required_fields = {
        "ticker",
        "cik",
        "company_name",
        "accession_no",
        "form_type",
        "filing_date",
        "period_of_report",
        "primary_document",
        "primary_doc_url",
        "fiscal_year",
        "fiscal_period",
    }
    if not required_fields.issubset(raw_filing):
        raise ValueError("Filing metadata is incomplete.")
    return dict(raw_filing), html  # type: ignore[return-value]


def _remove_non_content(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(["script", "style", "noscript", "svg"]):
        tag.decompose()
    for tag in soup.find_all():
        if tag.name is None or tag.attrs is None:
            continue
        name = (tag.name or "").lower()
        prefix = str(getattr(tag, "prefix", "") or "").lower()
        qualified_name = f"{prefix}:{name}" if prefix else name
        style = str(tag.get("style", "")).replace(" ", "").lower()
        if (
            qualified_name in {"ix:header", "ix:hidden", "xbrli:context", "xbrli:unit"}
            or tag.has_attr("hidden")
            or "display:none" in style
            or "visibility:hidden" in style
        ):
            tag.decompose()


def _parse_filing_markup(html: str) -> BeautifulSoup:
    prefix = html.lstrip()[:4096].lower()
    is_xhtml = (
        prefix.startswith("<?xml")
        or "xmlns:ix=" in prefix
        or 'xmlns="http://www.w3.org/1999/xhtml"' in prefix
        or "<ix:" in prefix
    )
    return BeautifulSoup(html, features="xml" if is_xhtml else "lxml")


def _remove_data_tables(soup: BeautifulSoup) -> None:
    for table in reversed(soup.find_all("table")):
        if table.name is None or table.attrs is None:
            continue
        rows = table.find_all("tr")
        cell_groups = [row.find_all(["td", "th"], recursive=False) for row in rows]
        cells = [cell for group in cell_groups for cell in group]
        cell_texts = [_normalize_line(cell.get_text(" ")) for cell in cells]
        nonempty_texts = [text for text in cell_texts if text]
        multi_column_rows = sum(len(group) >= 3 for group in cell_groups)
        numeric_cells = sum(_looks_numeric_cell(text) for text in nonempty_texts)
        is_data_grid = (
            len(rows) >= 3
            and len(nonempty_texts) >= 9
            and multi_column_rows >= 2
            and numeric_cells >= max(3, len(nonempty_texts) // 5)
            and max(map(len, nonempty_texts), default=0) < 500
        )
        if is_data_grid:
            table.decompose()


def _looks_numeric_cell(text: str) -> bool:
    if len(text) > 80 or not any(character.isdigit() for character in text):
        return False
    compact = text.replace(",", "").replace(" ", "")
    return bool(re.fullmatch(r"[$€£]?[()+\-−—]?\d+(?:\.\d+)?%?[)]?", compact))


def _normalized_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _normalize_line(raw_line)
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    return lines


def _remove_repeated_boilerplate(lines: list[str]) -> list[str]:
    signatures = Counter(
        signature for line in lines if (signature := _boilerplate_signature(line)) is not None
    )
    repeated = {
        signature
        for signature, count in signatures.items()
        if count >= _REPEATED_BOILERPLATE_MIN_COUNT
    }
    return [
        line
        for line in lines
        if (signature := _boilerplate_signature(line)) is None or signature not in repeated
    ]


def _boilerplate_signature(line: str) -> str | None:
    normalized = line.casefold()
    page_match = _PAGE_SUFFIX_PATTERN.search(normalized)
    contains_form_label = "form 10-k" in normalized or "form 10-q" in normalized
    if page_match is None and not (contains_form_label and "|" in normalized):
        return None
    if len(normalized) > 180:
        return None
    return _PAGE_SUFFIX_PATTERN.sub(r"\g<prefix>#", normalized)


def _block_aware_text(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(_BLOCK_TAGS):
        if (tag.name or "").lower() == "br":
            tag.replace_with("\n")
            continue
        tag.insert_before("\n")
        tag.insert_after("\n")
    return soup.get_text(" ")


def _normalize_line(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value).strip()


def _canonical_section(form_type: str, part: str | None, item: str) -> str | None:
    if form_type == "10-K":
        return _TEN_K_SECTIONS.get(item)
    if form_type == "10-Q":
        return _TEN_Q_SECTIONS.get((part or "", item)) or _TEN_Q_SECTION_FALLBACK.get(item)
    return None


def _section_row(filing: FilingMetadata, section: str, text: str) -> SectionText:
    return SectionText(
        ticker=filing["ticker"],
        cik=filing["cik"],
        company_name=filing["company_name"],
        accession_no=filing["accession_no"],
        form_type=filing["form_type"],
        filing_date=filing["filing_date"],
        period_of_report=filing["period_of_report"],
        fiscal_year=filing["fiscal_year"],
        fiscal_period=filing["fiscal_period"],
        section=section,
        text=text,
        source_url=filing["primary_doc_url"],
    )
