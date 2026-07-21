"""Clean SEC filing HTML and extract narrative sections."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import TypedDict

from bs4 import BeautifulSoup
from bs4.element import Tag

from ffa.ingestion.unstructured.fetch import FilingMetadata

logger = logging.getLogger(__name__)

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
_MIN_SUBSTANTIVE_SECTION_CHARS = 2_000
_REPAIR_MIN_GROWTH_FACTOR = 1.2
_MAX_LOGGED_HEADING_CANDIDATES = 30
_LATE_ITEM_INDEX_MIN_ITEMS = 8
_LATE_ITEM_INDEX_MAX_SPAN = 48
_LATE_ITEM_INDEX_TAIL_FRACTION = 0.75
_ACCOUNTING_NOTE_LOOKAHEAD_LINES = 12
_CANDIDATE_DOMINANCE_FACTOR = 1.2
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

# Version 1 of the deliberately narrow, reviewable alternative-title vocabulary.
# Unknown titles are logged but never promoted to section boundaries automatically.
_ALTERNATIVE_SECTION_TITLES_V1 = {
    "MD&A": (
        "Management's Discussion and Analysis",
        "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    ),
    "Notes": (
        "Notes to Consolidated Financial Statements",
        "Notes to the Consolidated Financial Statements",
    ),
}

# A fallback span is accepted only when one of these explicit, heading-like end
# markers is present. This is intentionally conservative to prevent contamination.
_ALTERNATIVE_SECTION_END_TITLES_V1 = {
    "MD&A": (
        "Management's Report on Internal Control Over Financial Reporting",
        "Consolidated Financial Statements",
        "Financial Statements",
        "Report of Independent Registered Public Accounting Firm",
        "Notes to Consolidated Financial Statements",
        "Notes to the Consolidated Financial Statements",
    ),
    "Notes": (
        "Supplementary Information",
        "Schedule II - Valuation and Qualifying Accounts",
        "Exhibit Index",
        "Item 16. Form 10-K Summary",
        "Signatures",
    ),
}

# Version 1 of the standalone titles used only after a compact Item index near
# the end of a 10-K proves that the historical Part/Item state machine selected
# navigation entries instead of the substantive sections preceding them.
_PRE_ITEM_SECTION_TITLES_V1 = {
    "MD&A": _ALTERNATIVE_SECTION_TITLES_V1["MD&A"],
    "Risk Factors": ("Risk Factors",),
    "Notes": _ALTERNATIVE_SECTION_TITLES_V1["Notes"],
}
_PRE_ITEM_SECTION_END_TITLES_V1 = {
    "MD&A": ("Risk Factors",),
    "Risk Factors": ("Properties",),
    "Notes": (
        "Management’s Assessment of Internal Control Over Financial Reporting",
        "Report of Independent Registered Public Accounting Firm",
        "Report of Independent Registered Public Accounting Firm on Internal Control "
        "Over Financial Reporting",
    ),
}

_ACCOUNTING_NOTES_END_TITLES_V1 = (
    "Report of Independent Registered Public Accounting Firm",
    "Supplementary Information",
    "Schedule II - Valuation and Qualifying Accounts",
    "Exhibit Index",
    "Item 16. Form 10-K Summary",
    "Signatures",
)

_REFERENCE_STUB_PATTERNS = (
    re.compile(r"\bappears?\s+on\s+pages?\b", re.IGNORECASE),
    re.compile(r"\bsee\s+(?:pages?|part|item)\b", re.IGNORECASE),
    re.compile(r"\bincorporat(?:ed|ion)\s+by\s+reference\b", re.IGNORECASE),
    re.compile(
        r"\b(?:set\s+forth|included)\b.{0,180}"
        r"\b(?:annual\s+report|part\s+iv|item\s+15|pages?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
)


@dataclass(frozen=True)
class _DocumentSignals:
    """DOM evidence retained before the filing is flattened into text lines."""

    heading_signatures: frozenset[str]
    anchored_item_occurrences: frozenset[tuple[str, int]]
    heading_titles: tuple[str, ...]


@dataclass(frozen=True)
class _BoundedCandidate:
    """Strongly bounded recovery candidate retained for ambiguity checks."""

    start_index: int
    end_index: int
    text: str


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
    """Convert filing HTML into grounded MD&A, risk, and notes sections.

    The historical largest-candidate behavior remains the default. A section is
    repaired only when strong DOM and text signals prove repeated page headers,
    a short incorporation reference, a late Item index, or an accounting Notes
    body with an unambiguous first-note marker and closing boundary.
    """
    filing, html = _unwrap_document(document)
    soup = _parse_filing_markup(html)
    _remove_non_content(soup)
    _remove_data_tables(soup)
    signals = _collect_document_signals(soup)
    lines = _remove_repeated_boilerplate(_normalized_lines(_block_aware_text(soup)))

    candidates = _extract_historical_candidates(lines, filing["form_type"])
    selected = {
        section_name: max(variants, key=lambda value: (len(value), value))
        for section_name, variants in candidates.items()
    }
    repair_failures: dict[str, str] = {}

    for section_name, primary_text in tuple(selected.items()):
        continuation_result = _extract_repeated_header_continuation(
            lines,
            form_type=filing["form_type"],
            section_name=section_name,
            signals=signals,
        )
        if continuation_result is not None:
            continuation, anchored_start = continuation_result
            anchor_repairs_small_primary = (
                anchored_start and len(primary_text) < _MIN_SUBSTANTIVE_SECTION_CHARS
            )
            proves_fragmentation = len(continuation) >= (
                len(primary_text) * _REPAIR_MIN_GROWTH_FACTOR
            )
            if anchor_repairs_small_primary or proves_fragmentation:
                selected[section_name] = continuation

        current_text = selected[section_name]
        if not _is_reference_stub(current_text):
            continue
        alternative, failure_reason = _extract_alternative_section(
            lines,
            section_name=section_name,
            signals=signals,
        )
        if alternative is not None and len(alternative) > len(current_text):
            selected[section_name] = alternative
        elif failure_reason is not None:
            repair_failures[section_name] = failure_reason

    if filing["form_type"] == "10-K":
        late_item_index = _find_late_item_index(lines)
        if late_item_index is not None:
            for section_name in _TEN_K_SECTIONS.values():
                current_text = selected.get(section_name, "")
                if len(current_text) >= _MIN_SUBSTANTIVE_SECTION_CHARS:
                    continue
                recovered, failure_reason = _extract_pre_item_section(
                    lines,
                    section_name=section_name,
                    late_item_index=late_item_index,
                )
                if recovered is not None and len(recovered) > len(current_text):
                    selected[section_name] = recovered
                    repair_failures.pop(section_name, None)
                elif failure_reason is not None:
                    repair_failures.setdefault(section_name, failure_reason)

        current_notes = selected.get("Notes", "")
        if len(current_notes) < _MIN_SUBSTANTIVE_SECTION_CHARS:
            recovered_notes, failure_reason = _extract_accounting_notes(lines)
            if recovered_notes is not None and len(recovered_notes) > len(current_notes):
                selected["Notes"] = recovered_notes
                repair_failures.pop("Notes", None)
            elif failure_reason is not None:
                repair_failures.setdefault("Notes", failure_reason)

    if not selected:
        fallback_text = "\n".join(lines).strip()
        if not fallback_text:
            raise ValueError("SEC filing HTML contains no usable text.")
        selected["Full Filing"] = fallback_text

    selected = {
        section_name: _strip_unambiguous_page_chrome(section_text)
        for section_name, section_text in selected.items()
    }

    _log_unfetched_companion_document(filing, selected)

    _log_unresolved_sections(
        filing,
        selected,
        signals=signals,
        repair_failures=repair_failures,
    )

    return sorted(
        (
            _section_row(filing, section_name, section_text)
            for section_name, section_text in selected.items()
        ),
        key=lambda section: section["section"],
    )


def _extract_historical_candidates(
    lines: list[str],
    form_type: str,
) -> dict[str, list[str]]:
    """Reproduce the original line-state extraction without repair behavior."""
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
            current_section = _canonical_section(form_type, current_part, item)
            current_lines = []
            title = _normalize_line(item_match.group("title"))
            if current_section is not None and title:
                current_lines.append(title)
            continue

        if current_section is not None:
            current_lines.append(line)
    flush()
    return candidates


def _extract_repeated_header_continuation(
    lines: list[str],
    *,
    form_type: str,
    section_name: str,
    signals: _DocumentSignals,
) -> tuple[str, bool] | None:
    """Join a canonical 10-K section split by repeated PART or ITEM page headers."""
    if form_type != "10-K":
        return None
    target_item = next(
        (item for item, canonical in _TEN_K_SECTIONS.items() if canonical == section_name),
        None,
    )
    if target_item is None:
        return None

    item_occurrences: list[tuple[int, tuple[str, int]]] = []
    item_signature_counts: Counter[str] = Counter()
    part_count = 0
    for index, line in enumerate(lines):
        item_match = _ITEM_PATTERN.match(line)
        if item_match is not None and item_match.group("item").upper() == target_item:
            signature = _heading_signature(line)
            occurrence = item_signature_counts[signature]
            item_signature_counts[signature] += 1
            item_occurrences.append((index, (signature, occurrence)))
        if _PART_PATTERN.match(line) is not None:
            part_count += 1

    if len(item_occurrences) < _REPEATED_BOILERPLATE_MIN_COUNT and (
        part_count < _REPEATED_BOILERPLATE_MIN_COUNT
    ):
        return None

    document_line_counts = Counter(_heading_signature(line) for line in lines)
    variants: list[tuple[bool, str]] = []
    for start_index, occurrence_key in item_occurrences:
        start_match = _ITEM_PATTERN.match(lines[start_index])
        if start_match is None:
            continue
        body: list[str] = []
        title = _normalize_line(start_match.group("title"))
        if title:
            body.append(title)
        has_reliable_end = False

        for line in lines[start_index + 1 :]:
            item_match = _ITEM_PATTERN.match(line)
            if item_match is not None:
                if item_match.group("item").upper() == target_item:
                    continue
                has_reliable_end = True
                break
            if _PART_PATTERN.match(line) is not None:
                continue
            body.append(line)

        if not has_reliable_end:
            continue
        repaired = _clean_repaired_lines(body, document_line_counts=document_line_counts)
        if repaired:
            variants.append(
                (
                    occurrence_key in signals.anchored_item_occurrences,
                    repaired,
                )
            )

    if not variants:
        return None

    # An evocative, unique item anchor is stronger evidence than length or
    # repetition. Links/hrefs are deliberately not included in this signal.
    anchored_variants = [variant for anchored, variant in variants if anchored]
    if anchored_variants:
        return max(anchored_variants, key=lambda value: (len(value), value)), True
    eligible = [variant for _, variant in variants]
    return max(eligible, key=lambda value: (len(value), value)), False


def _extract_alternative_section(
    lines: list[str],
    *,
    section_name: str,
    signals: _DocumentSignals,
) -> tuple[str | None, str | None]:
    """Resolve an incorporation reference through explicit standalone headings."""
    raw_aliases = _ALTERNATIVE_SECTION_TITLES_V1.get(section_name)
    raw_boundaries = _ALTERNATIVE_SECTION_END_TITLES_V1.get(section_name)
    if raw_aliases is None or raw_boundaries is None:
        return None, "no versioned alternative-title mapping exists"

    aliases = {_heading_signature(alias) for alias in raw_aliases}
    start_indexes = [
        index
        for index, line in enumerate(lines)
        if _heading_signature(line) in aliases
        and _heading_signature(line) in signals.heading_signatures
    ]
    if not start_indexes:
        return None, "no allow-listed standalone heading candidate was found"

    boundaries = tuple(_heading_signature(boundary) for boundary in raw_boundaries)
    document_line_counts = Counter(_heading_signature(line) for line in lines)
    variants: list[str] = []
    saw_unbounded_candidate = False
    for start_index in start_indexes:
        body = [lines[start_index]]
        has_reliable_end = False
        for line in lines[start_index + 1 :]:
            signature = _heading_signature(line)
            if signature in aliases:
                continue
            if _is_alternative_boundary(
                line,
                signature=signature,
                boundary_signatures=boundaries,
                heading_signatures=signals.heading_signatures,
            ):
                has_reliable_end = True
                break
            body.append(line)

        if not has_reliable_end:
            saw_unbounded_candidate = True
            continue
        repaired = _clean_repaired_lines(body, document_line_counts=document_line_counts)
        if repaired:
            variants.append(repaired)

    if not variants:
        if saw_unbounded_candidate:
            return None, "an allow-listed heading was found without a reliable closing boundary"
        return None, "allow-listed heading candidates contained no usable bounded text"
    return max(variants, key=lambda value: (len(value), value)), None


def _find_late_item_index(lines: list[str]) -> int | None:
    """Return the start of a compact, tail-positioned 10-K Item navigation index."""
    tail_start = int(len(lines) * _LATE_ITEM_INDEX_TAIL_FRACTION)
    item_occurrences = [
        (index, match.group("item").upper())
        for index, line in enumerate(lines)
        if index >= tail_start and (match := _ITEM_PATTERN.match(line)) is not None
    ]
    for offset, (start_index, _) in enumerate(item_occurrences):
        cluster = [
            (index, item)
            for index, item in item_occurrences[offset:]
            if index - start_index <= _LATE_ITEM_INDEX_MAX_SPAN
        ]
        items = {item for _, item in cluster}
        if len(cluster) < _LATE_ITEM_INDEX_MIN_ITEMS or not {"1A", "7", "8"}.issubset(items):
            continue
        cluster_end = cluster[-1][0]
        part_count = sum(
            _PART_PATTERN.match(line) is not None
            for line in lines[max(tail_start, start_index - 2) : cluster_end + 2]
        )
        if part_count >= 2:
            return start_index
    return None


def _extract_pre_item_section(
    lines: list[str],
    *,
    section_name: str,
    late_item_index: int,
) -> tuple[str | None, str | None]:
    """Recover a substantive section preceding a proven late Item index."""
    raw_aliases = _PRE_ITEM_SECTION_TITLES_V1.get(section_name)
    raw_boundaries = _PRE_ITEM_SECTION_END_TITLES_V1.get(section_name)
    if raw_aliases is None or raw_boundaries is None:
        return None, "no versioned pre-Item title mapping exists"

    aliases = {_heading_signature(alias) for alias in raw_aliases}
    boundaries = {_heading_signature(boundary) for boundary in raw_boundaries}
    start_indexes = [
        index
        for index, line in enumerate(lines[:late_item_index])
        if _heading_signature(line) in aliases
    ]
    candidates: list[_BoundedCandidate] = []
    document_line_counts = Counter(_heading_signature(line) for line in lines)
    for start_index in start_indexes:
        end_index = next(
            (
                index
                for index in range(start_index + 1, late_item_index)
                if _heading_signature(lines[index]) in boundaries
            ),
            None,
        )
        if end_index is None:
            continue
        body = [
            line
            for line in lines[start_index:end_index]
            if (_heading_signature(line) not in aliases or line == lines[start_index])
            and not _is_standalone_annual_report_page_header(line)
        ]
        repaired = _clean_repaired_lines(
            body,
            document_line_counts=document_line_counts,
        )
        if len(repaired) >= _MIN_SUBSTANTIVE_SECTION_CHARS:
            candidates.append(
                _BoundedCandidate(
                    start_index=start_index,
                    end_index=end_index,
                    text=repaired,
                )
            )

    return _select_unambiguous_candidate(
        candidates,
        empty_reason="no substantive section was bounded before the late Item index",
        ambiguous_reason="multiple pre-Item section candidates remained ambiguous",
    )


def _extract_accounting_notes(lines: list[str]) -> tuple[str | None, str | None]:
    """Select the accounting-body occurrence of a duplicated Notes title."""
    aliases = {_heading_signature(alias) for alias in _ALTERNATIVE_SECTION_TITLES_V1["Notes"]}
    start_indexes = [
        index for index, line in enumerate(lines) if _heading_signature(line) in aliases
    ]
    if len(start_indexes) < 2:
        return None, "fewer than two accounting Notes title occurrences were found"
    if not any(
        (match := _ITEM_PATTERN.match(line)) is not None and match.group("item").upper() == "8"
        for line in lines
    ):
        return None, "duplicated Notes titles were found without an Item 8 context"

    boundary_signatures = {
        _heading_signature(boundary) for boundary in _ACCOUNTING_NOTES_END_TITLES_V1
    }
    document_line_counts = Counter(_heading_signature(line) for line in lines)
    candidates: list[_BoundedCandidate] = []
    for start_index in start_indexes:
        if not _has_nearby_first_note(lines, start_index):
            continue
        end_index = _find_accounting_notes_end(
            lines,
            start_index=start_index,
            boundary_signatures=boundary_signatures,
        )
        if end_index is None:
            continue
        body = [
            line
            for line in lines[start_index:end_index]
            if _heading_signature(line) not in aliases or line == lines[start_index]
            if not _is_accounting_notes_page_chrome(line)
        ]
        repaired = _clean_repaired_lines(
            body,
            document_line_counts=document_line_counts,
        )
        if len(repaired) >= _MIN_SUBSTANTIVE_SECTION_CHARS:
            candidates.append(
                _BoundedCandidate(
                    start_index=start_index,
                    end_index=end_index,
                    text=repaired,
                )
            )

    return _select_unambiguous_candidate(
        candidates,
        empty_reason=(
            "duplicated Notes titles had no substantive candidate with a nearby first note "
            "and reliable end"
        ),
        ambiguous_reason="multiple accounting Notes bodies remained ambiguous",
    )


def _has_nearby_first_note(lines: list[str], start_index: int) -> bool:
    aliases = {_heading_signature(alias) for alias in _ALTERNATIVE_SECTION_TITLES_V1["Notes"]}
    lookahead = lines[start_index + 1 : start_index + 1 + _ACCOUNTING_NOTE_LOOKAHEAD_LINES]
    for line in lookahead:
        if _heading_signature(line) in aliases:
            return False
        if re.match(r"^note\s+1(?:$|[\s.\-–—:])", line, re.IGNORECASE):
            return True
        if re.match(r"^1(?:[.\-–—:]|\s+)\s*\S{3}", line):
            return True
    return False


def _is_standalone_annual_report_page_header(line: str) -> bool:
    return bool(
        re.fullmatch(
            r".{1,100}\s+20\d{2}\s+annual report\s+\d{1,3}",
            _normalize_line(line),
            re.IGNORECASE,
        )
    )


def _is_accounting_notes_page_chrome(line: str) -> bool:
    signature = _heading_signature(line)
    return signature == "financial table of contents" or bool(
        re.fullmatch(
            r"(?:millions|thousands) of dollars(?:, except .{1,80})?",
            _normalize_line(line),
            re.IGNORECASE,
        )
    )


def _find_accounting_notes_end(
    lines: list[str],
    *,
    start_index: int,
    boundary_signatures: set[str],
) -> int | None:
    for index in range(start_index + 1, len(lines)):
        line = lines[index]
        signature = _heading_signature(line)
        item_match = _ITEM_PATTERN.match(line)
        if signature in boundary_signatures:
            return index
        if _PART_PATTERN.match(line) is not None and signature == "part iv":
            return index
        if item_match is not None and item_match.group("item").upper() in {"15", "16"}:
            return index

    last_item_15 = max(
        (
            index
            for index, line in enumerate(lines)
            if (match := _ITEM_PATTERN.match(line)) is not None
            and match.group("item").upper() == "15"
        ),
        default=-1,
    )
    return len(lines) if start_index > last_item_15 else None


def _select_unambiguous_candidate(
    candidates: list[_BoundedCandidate],
    *,
    empty_reason: str,
    ambiguous_reason: str,
) -> tuple[str | None, str | None]:
    if not candidates:
        return None, empty_reason

    # Repeated page titles inside one accounting body create nested candidates
    # with the same closing boundary. Keep the earliest occurrence so the
    # continuations do not masquerade as independent, ambiguous sections.
    by_end: dict[int, _BoundedCandidate] = {}
    for candidate in sorted(candidates, key=lambda value: value.start_index):
        by_end.setdefault(candidate.end_index, candidate)
    deduplicated = list(by_end.values())

    ranked = sorted(deduplicated, key=lambda value: (len(value.text), value.text), reverse=True)
    if len(ranked) > 1 and len(ranked[0].text) < (
        len(ranked[1].text) * _CANDIDATE_DOMINANCE_FACTOR
    ):
        return None, ambiguous_reason
    return ranked[0].text, None


def _is_alternative_boundary(
    line: str,
    *,
    signature: str,
    boundary_signatures: tuple[str, ...],
    heading_signatures: frozenset[str],
) -> bool:
    item_match = _ITEM_PATTERN.match(line)
    if item_match is not None and item_match.group("item").upper() == "16":
        return True
    if signature not in heading_signatures:
        return False
    return any(
        signature == boundary
        or signature.startswith(f"{boundary}:")
        or signature.startswith(f"{boundary} -")
        for boundary in boundary_signatures
    )


def _is_reference_stub(text: str) -> bool:
    return len(text) < _MIN_SUBSTANTIVE_SECTION_CHARS and any(
        pattern.search(text) for pattern in _REFERENCE_STUB_PATTERNS
    )


def _clean_repaired_lines(
    lines: list[str],
    *,
    document_line_counts: Counter[str],
) -> str:
    kept: list[str] = []
    for index, line in enumerate(lines):
        signature = _heading_signature(line)
        if _is_page_number(line) or _is_continuation_label(line) or _is_definite_page_chrome(line):
            continue
        if (
            index > 0
            and document_line_counts[signature] >= _REPEATED_BOILERPLATE_MIN_COUNT
            and _is_repeated_page_chrome(line, signature=signature)
        ):
            continue
        if not kept or kept[-1] != line:
            kept.append(line)
    return "\n".join(kept).strip()


def _is_page_number(line: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", line.strip()))


def _is_continuation_label(line: str) -> bool:
    return bool(re.fullmatch(r"\(?\s*continued\s*\)?[.:]?", line.strip(), re.IGNORECASE))


def _is_definite_page_chrome(line: str) -> bool:
    normalized = _normalize_line(line)
    return bool(
        re.fullmatch(
            r"(?:[^|/]{1,100}[|/]\s*)?(?:20\d{2}\s+)?form\s+10\s*[-–—]\s*[kq]",
            normalized,
            re.IGNORECASE,
        )
    )


def _strip_unambiguous_page_chrome(text: str) -> str:
    lines = text.splitlines()
    kept = [
        line
        for line in lines
        if _heading_signature(line) != "table of contents"
        and not _is_continuation_label(line)
        and not _is_definite_page_chrome(line)
    ]
    return "\n".join(kept).strip()


def _is_repeated_page_chrome(line: str, *, signature: str) -> bool:
    lowered = signature.casefold()
    return bool(
        _PART_PATTERN.match(line)
        or _ITEM_PATTERN.match(line)
        or lowered == "table of contents"
        or lowered.endswith(" and subsidiaries")
        or re.search(r"\bform\s+10\s*-\s*[kq]\b", lowered)
        or "annual report" in lowered
        or signature
        in {
            _heading_signature(title)
            for titles in _ALTERNATIVE_SECTION_TITLES_V1.values()
            for title in titles
        }
    )


def _collect_document_signals(soup: BeautifulSoup) -> _DocumentSignals:
    """Collect heading and unique evocative-anchor evidence from the DOM."""
    id_counts = Counter(
        anchor for tag in soup.find_all(True) if (anchor := _tag_anchor(tag)) is not None
    )
    heading_signatures: set[str] = set()
    anchored_item_occurrences: set[tuple[str, int]] = set()
    heading_titles: dict[str, str] = {}

    for raw_tag in soup.find_all(True):
        if not isinstance(raw_tag, Tag):
            continue
        text = _normalize_line(raw_tag.get_text(" ", strip=True))
        if not text or len(text) > 240:
            continue
        signature = _heading_signature(text)
        if _is_heading_like_tag(raw_tag):
            heading_signatures.add(signature)
            heading_titles.setdefault(signature, text)

    item_signature_counts: Counter[str] = Counter()
    for raw_tag in soup.find_all(_BLOCK_TAGS):
        if not isinstance(raw_tag, Tag) or not _is_leaf_item_block(raw_tag):
            continue
        text = _normalize_line(raw_tag.get_text(" ", strip=True))
        item_match = _ITEM_PATTERN.match(text)
        if item_match is None:
            continue
        signature = _heading_signature(text)
        occurrence = item_signature_counts[signature]
        item_signature_counts[signature] += 1
        anchor = _tag_anchor(raw_tag)
        if (
            anchor is not None
            and id_counts[anchor] == 1
            and _is_evocative_item_anchor(anchor, item_match.group("item"))
        ):
            anchored_item_occurrences.add((signature, occurrence))

    return _DocumentSignals(
        heading_signatures=frozenset(heading_signatures),
        anchored_item_occurrences=frozenset(anchored_item_occurrences),
        heading_titles=tuple(heading_titles[signature] for signature in sorted(heading_titles)),
    )


def _is_leaf_item_block(tag: Tag) -> bool:
    text = _normalize_line(tag.get_text(" ", strip=True))
    if not text or len(text) > 240 or _ITEM_PATTERN.match(text) is None:
        return False
    return not any(
        isinstance(descendant, Tag)
        and (descendant.name or "").lower() in _BLOCK_TAGS
        and _normalize_line(descendant.get_text(" ", strip=True)) == text
        for descendant in tag.find_all(_BLOCK_TAGS)
    )


def _is_heading_like_tag(tag: Tag) -> bool:
    name = (tag.name or "").lower()
    if name in {"h1", "h2", "h3", "h4", "h5", "h6", "strong", "b"}:
        return True
    if str(tag.get("role", "")).casefold() == "heading":
        return True
    own_style = str(tag.get("style", ""))
    if _has_heading_style(own_style):
        return True
    classes = " ".join(str(value) for value in tag.get("class", []))
    if re.search(r"(?:^|[-_\s])(?:heading|title)(?:$|[-_\s])", classes, re.IGNORECASE):
        return True

    # SEC generators commonly put the visible title in one bold span inside a
    # neutral div or paragraph. Only accept a short container whose descendant
    # carries explicit heading style.
    if name in {"div", "p", "section", "td"}:
        return any(
            isinstance(child, Tag) and _has_heading_style(str(child.get("style", "")))
            for child in tag.find_all(True)
        )
    return False


def _has_heading_style(style: str) -> bool:
    normalized = style.replace(" ", "").casefold()
    if "font-weight:bold" in normalized or "text-align:center" in normalized:
        return True
    weight_match = re.search(r"font-weight:(\d{3})", normalized)
    return weight_match is not None and int(weight_match.group(1)) >= 600


def _tag_anchor(tag: Tag) -> str | None:
    for attribute in ("id", "name"):
        value = tag.get(attribute)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_evocative_item_anchor(anchor: str, item: str) -> bool:
    compact_anchor = re.sub(r"[^a-z0-9]", "", anchor.casefold())
    compact_item = re.sub(r"[^a-z0-9]", "", item.casefold())
    return "item" in compact_anchor and f"item{compact_item}" in compact_anchor


def _heading_signature(value: str) -> str:
    normalized = _normalize_line(value).casefold()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = re.sub(r"[‐‑‒–—]", "-", normalized)
    normalized = re.sub(r"\s*-\s*", " - ", normalized)
    normalized = re.sub(r"\s*\(continued\)\s*$", "", normalized)
    return normalized.rstrip(" .:;-–—")


def _log_unresolved_sections(
    filing: FilingMetadata,
    selected: Mapping[str, str],
    *,
    signals: _DocumentSignals,
    repair_failures: Mapping[str, str],
) -> None:
    expected_sections = (
        tuple(_TEN_K_SECTIONS.values())
        if filing["form_type"] == "10-K"
        else tuple(dict.fromkeys(_TEN_Q_SECTIONS.values()))
    )
    candidate_titles = signals.heading_titles[:_MAX_LOGGED_HEADING_CANDIDATES]
    for section_name in expected_sections:
        candidate_size = len(selected.get(section_name, ""))
        if candidate_size >= _MIN_SUBSTANTIVE_SECTION_CHARS:
            continue
        reason = repair_failures.get(
            section_name,
            "canonical section remained below the substantive-text threshold "
            "without a validated repair signal",
        )
        logger.warning(
            "SEC narrative section unresolved: ticker=%s cik=%s accession=%s "
            "section=%s candidate_size=%s heading_candidates=%s reason=%s",
            filing["ticker"],
            filing["cik"],
            filing["accession_no"],
            section_name,
            candidate_size,
            candidate_titles,
            reason,
        )


def _log_unfetched_companion_document(
    filing: FilingMetadata,
    selected: Mapping[str, str],
) -> None:
    combined_text = "\n".join(selected.values())
    if not (
        re.search(r"\bannual report to shareholders\b", combined_text, re.IGNORECASE)
        and re.search(
            r"\bincorporat(?:ed|ion)\s+into\s+this\s+item\s+by\s+reference\b",
            combined_text,
            re.IGNORECASE,
        )
    ):
        return
    logger.warning(
        "SEC narrative content unavailable: ticker=%s cik=%s accession=%s "
        "reason=content incorporated in companion document not fetched",
        filing["ticker"],
        filing["cik"],
        filing["accession_no"],
    )


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
