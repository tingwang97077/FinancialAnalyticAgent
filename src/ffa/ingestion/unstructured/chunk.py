"""Create section-aware token chunks from cleaned filing text."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Protocol, TypedDict

import tiktoken

from ffa.config import Settings, get_settings


class TokenEncoder(Protocol):
    """Tokenizer operations required by the chunker."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token identifiers."""
        ...


class ChunkRow(TypedDict):
    """Text chunk and relational metadata ready for embedding."""

    ticker: str
    cik: int
    company_name: str
    accession_no: str
    form_type: str
    filing_date: object
    period_of_report: object | None
    fiscal_year: int | None
    fiscal_period: str | None
    section: str
    chunk_index: int
    text: str
    token_count: int
    source_url: str


def chunk_text(
    sections: Sequence[Mapping[str, object]],
    *,
    max_tokens: int | None = None,
    overlap: float | None = None,
    settings: Settings | None = None,
    encoder: TokenEncoder | None = None,
) -> list[ChunkRow]:
    """Split sections into overlapping windows at word or sentence boundaries.

    ``max_tokens`` is a target and a hard limit unless one individual word itself
    exceeds it. Sentence or line endings are preferred once at least 60 percent
    of the target has been filled. Overlap always begins at a complete word.
    """
    resolved_settings = settings
    if max_tokens is None or overlap is None:
        resolved_settings = settings or get_settings()
    resolved_max_tokens = resolved_settings.chunk_max_tokens if max_tokens is None else max_tokens
    resolved_overlap = resolved_settings.chunk_overlap if overlap is None else overlap
    if resolved_max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")
    if not 0 <= resolved_overlap < 1:
        raise ValueError("overlap must be greater than or equal to zero and less than one.")

    resolved_encoder = encoder or tiktoken.get_encoding("cl100k_base")
    overlap_tokens = min(
        round(resolved_max_tokens * resolved_overlap),
        resolved_max_tokens - 1,
    )
    chunks: list[ChunkRow] = []

    for raw_section in sections:
        section = _validate_section(raw_section)
        section_chunks = _boundary_chunks(
            section["text"],
            max_tokens=resolved_max_tokens,
            overlap_tokens=overlap_tokens,
            encoder=resolved_encoder,
        )
        for chunk_index, (chunk_text_value, token_count) in enumerate(section_chunks):
            chunks.append(
                ChunkRow(
                    ticker=section["ticker"],
                    cik=section["cik"],
                    company_name=section["company_name"],
                    accession_no=section["accession_no"],
                    form_type=section["form_type"],
                    filing_date=section["filing_date"],
                    period_of_report=section["period_of_report"],
                    fiscal_year=section["fiscal_year"],
                    fiscal_period=section["fiscal_period"],
                    section=section["section"],
                    chunk_index=chunk_index,
                    text=chunk_text_value,
                    token_count=token_count,
                    source_url=section["source_url"],
                )
            )
    return chunks


def _boundary_chunks(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    encoder: TokenEncoder,
) -> list[tuple[str, int]]:
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []

    chunks: list[tuple[str, int]] = []
    start_index = 0
    while start_index < len(words):
        hard_end_index = _maximum_end_index(
            text,
            words,
            start_index=start_index,
            max_tokens=max_tokens,
            encoder=encoder,
        )
        end_index = _preferred_end_index(
            text,
            words,
            start_index=start_index,
            hard_end_index=hard_end_index,
            max_tokens=max_tokens,
            encoder=encoder,
        )
        chunk = text[words[start_index].start() : words[end_index].end()].strip()
        token_count = len(encoder.encode(chunk))
        chunks.append((chunk, token_count))
        if end_index == len(words) - 1:
            break
        start_index = _overlap_start_index(
            text,
            words,
            current_start_index=start_index,
            end_index=end_index,
            overlap_tokens=overlap_tokens,
            encoder=encoder,
        )
    return chunks


def _maximum_end_index(
    text: str,
    words: list[re.Match[str]],
    *,
    start_index: int,
    max_tokens: int,
    encoder: TokenEncoder,
) -> int:
    start = words[start_index].start()
    if len(encoder.encode(text[start : words[start_index].end()])) > max_tokens:
        return start_index

    low = start_index
    high = len(words) - 1
    best = start_index
    while low <= high:
        middle = (low + high) // 2
        candidate = text[start : words[middle].end()]
        if len(encoder.encode(candidate)) <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _preferred_end_index(
    text: str,
    words: list[re.Match[str]],
    *,
    start_index: int,
    hard_end_index: int,
    max_tokens: int,
    encoder: TokenEncoder,
) -> int:
    minimum_fill = max(1, round(max_tokens * 0.6))
    start = words[start_index].start()
    for candidate_index in range(hard_end_index, start_index - 1, -1):
        if not _is_natural_end(text, words, candidate_index):
            continue
        candidate = text[start : words[candidate_index].end()]
        if len(encoder.encode(candidate)) >= minimum_fill:
            return candidate_index
        break
    return hard_end_index


def _is_natural_end(
    text: str,
    words: list[re.Match[str]],
    word_index: int,
) -> bool:
    word = words[word_index].group()
    if re.search(r"[.!?][\"'”’)}\]]*$", word):
        return True
    if word_index == len(words) - 1:
        return True
    gap = text[words[word_index].end() : words[word_index + 1].start()]
    return "\n" in gap


def _overlap_start_index(
    text: str,
    words: list[re.Match[str]],
    *,
    current_start_index: int,
    end_index: int,
    overlap_tokens: int,
    encoder: TokenEncoder,
) -> int:
    if overlap_tokens == 0 or end_index == current_start_index:
        return end_index + 1

    low = current_start_index + 1
    high = end_index
    best = end_index + 1
    end = words[end_index].end()
    while low <= high:
        middle = (low + high) // 2
        suffix = text[words[middle].start() : end]
        if len(encoder.encode(suffix)) <= overlap_tokens:
            best = middle
            high = middle - 1
        else:
            low = middle + 1
    return best


def _validate_section(section: Mapping[str, object]) -> ChunkRow:
    required_strings = (
        "ticker",
        "company_name",
        "accession_no",
        "form_type",
        "section",
        "text",
        "source_url",
    )
    values: dict[str, object] = dict(section)
    for field in required_strings:
        value = values.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string.")
        values[field] = value.strip()
    cik = values.get("cik")
    if isinstance(cik, bool) or not isinstance(cik, int) or cik <= 0:
        raise ValueError("cik must be a positive integer.")
    fiscal_year = values.get("fiscal_year")
    if fiscal_year is not None and (
        isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int) or fiscal_year <= 0
    ):
        raise ValueError("fiscal_year must be a positive integer when present.")
    fiscal_period = values.get("fiscal_period")
    if fiscal_period is not None and fiscal_period not in {"FY", "Q1", "Q2", "Q3", "Q4"}:
        raise ValueError("fiscal_period must be FY or Q1 through Q4 when present.")

    return ChunkRow(
        ticker=str(values["ticker"]),
        cik=cik,
        company_name=str(values["company_name"]),
        accession_no=str(values["accession_no"]),
        form_type=str(values["form_type"]),
        filing_date=values.get("filing_date"),
        period_of_report=values.get("period_of_report"),
        fiscal_year=fiscal_year,
        fiscal_period=None if fiscal_period is None else str(fiscal_period),
        section=str(values["section"]),
        chunk_index=0,
        text=str(values["text"]),
        token_count=0,
        source_url=str(values["source_url"]),
    )
