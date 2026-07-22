"""Grounded answer generation and deterministic evidence validation."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

from openai import OpenAI

from ffa.agent.schemas import Answer, Citation, NumberFact
from ffa.config import Settings, get_settings
from ffa.monitoring.tracing import record_openai_response
from ffa.retrieval.base import Chunk

if TYPE_CHECKING:
    from ffa.agent.router import AgentContext

logger = logging.getLogger(__name__)

_GENERATION_INSTRUCTIONS = """You write grounded financial answers from supplied evidence.

Rules:
- Numeric claims may use only the supplied facts and must copy their metric, fiscal period, value,
  and unit exactly into Answer.numbers.
- Never calculate, estimate, transform, round, compare, or derive a number. A requested ratio,
  growth rate, or delta is usable only when it already exists as a supplied fact.
- Narrative claims may use only the supplied filing chunks. Add a Citation using the exact
  source_url for every narrative claim presented as sourced.
- Never invent a source URL, accession number, section, fact, or number.
- If the evidence is insufficient, state that limitation instead of filling the gap.
- Return only the structured Answer required by the response schema.
"""
_UNSUPPORTED_CLAIMS_TEXT = (
    "Unsupported claims were removed. Only the structured facts and citations below passed "
    "grounding validation."
)
_NO_GROUNDED_ANSWER_TEXT = "The generated response could not be grounded in the retrieved evidence."
_PROMPT_DISCLOSURE_BLOCKED_TEXT = "The response was withheld by the safety policy."
_PROMPT_NGRAM_SIZE = 7
_PROMPT_DISCLOSURE_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(?:system|developer|hidden|internal)\s+(?:prompt|message|instructions?|rules?)\b",
        r"\b(?:my|the)\s+(?:system|developer)\s+(?:prompt|message|instructions?)\b",
        r"(?:^|\n)\s*(?:system|developer)\s*:",
        r"\b(?:here (?:is|are)|the following (?:is|are))\b.{0,60}"
        r"\b(?:prompt|instructions?|rules?)\b",
        r"\b(?:prompt|message|instructions?|r[eè]gles?)\s+"
        r"(?:syst[eè]me|d[eé]veloppeur|internes?|cach[eé]es?)\b",
        r"\bvoici\b.{0,60}\b(?:prompt|instructions?|r[eè]gles?)\b",
    )
)

type NumberKey = tuple[str, int, str, float, str]
type GenerationCallable = Callable[[str, AgentContext], Answer]


class GenerationProvider(Protocol):
    """Backend contract for structured grounded-answer generation."""

    def generate_answer(
        self,
        question: str,
        context: AgentContext,
        *,
        model: str,
    ) -> Answer:
        """Return one schema-validated answer draft."""
        ...


class OpenAIGenerationProvider:
    """OpenAI Responses API adapter for structured Answer generation."""

    def __init__(self, client: OpenAI) -> None:
        """Initialize the provider with an OpenAI SDK client."""
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> OpenAIGenerationProvider:
        """Build the provider from central application settings."""
        resolved_settings = settings or get_settings()
        if resolved_settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY must be configured before generating answers.")
        api_key = resolved_settings.openai_api_key.get_secret_value().strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be configured before generating answers.")
        return cls(OpenAI(api_key=api_key))

    def generate_answer(
        self,
        question: str,
        context: AgentContext,
        *,
        model: str,
    ) -> Answer:
        """Generate one Answer through native structured output."""
        response = self._client.responses.parse(
            model=model,
            instructions=_GENERATION_INSTRUCTIONS,
            input=_generation_input(question, context),
            text_format=Answer,
            metadata={"trace_id": context.trace_id},
        )
        record_openai_response(response, model=model)
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("The model did not return a structured answer.")
        return parsed


class AnswerGenerator:
    """Generate an answer draft and enforce local grounding invariants."""

    def __init__(self, *, provider: GenerationProvider, model: str) -> None:
        """Initialize the generator with explicit model configuration."""
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("OPENAI_MODEL must be configured before generating answers.")
        self._provider = provider
        self._model = normalized_model

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> AnswerGenerator:
        """Build the generator from central application settings."""
        resolved_settings = settings or get_settings()
        return cls(
            provider=OpenAIGenerationProvider.from_settings(resolved_settings),
            model=resolved_settings.openai_model,
        )

    def generate(self, question: str, context: AgentContext) -> Answer:
        """Return a structured answer after deterministic evidence checks."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty.")
        draft = self._provider.generate_answer(
            normalized_question,
            context,
            model=self._model,
        )
        if _contains_prompt_disclosure(draft.text):
            logger.warning("Generated answer withheld by safety policy.")
            return Answer(text=_PROMPT_DISCLOSURE_BLOCKED_TEXT, grounded=False)
        return _validate_grounding(draft, context)


def generate(question: str, context: AgentContext) -> Answer:
    """Generate and locally validate a grounded answer."""
    return _get_default_generator().generate(question, context)


@lru_cache(maxsize=1)
def _get_default_generator() -> AnswerGenerator:
    """Reuse the OpenAI client and configured model across requests."""
    return AnswerGenerator.from_settings(get_settings())


def _generation_input(question: str, context: AgentContext) -> str:
    """Serialize only typed evidence and routing metadata for generation."""
    payload = {
        "question": question,
        "trace_id": context.trace_id,
        "route": context.route,
        "facts": [fact.model_dump(mode="json") for fact in context.facts],
        "chunks": [_chunk_evidence(chunk) for chunk in context.chunks],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _chunk_evidence(chunk: Chunk) -> dict[str, object]:
    """Select chunk fields that may ground narrative claims and citations."""
    return {
        "accession_no": chunk["accession_no"],
        "ticker": chunk["ticker"],
        "fiscal_year": chunk["fiscal_year"],
        "fiscal_period": chunk["fiscal_period"],
        "section": chunk["section"],
        "text": chunk["text"],
        "source_url": chunk["source_url"],
    }


def _validate_grounding(draft: Answer, context: AgentContext) -> Answer:
    """Remove unsupported facts and citations, and retract unsafe prose."""
    allowed_numbers = {_number_key(fact) for fact in context.facts}
    valid_numbers: list[NumberFact] = []
    seen_numbers: set[NumberKey] = set()
    invalid_number = False
    for number in draft.numbers:
        key = _number_key(number)
        if key not in allowed_numbers:
            invalid_number = True
            continue
        if key not in seen_numbers:
            valid_numbers.append(number)
            seen_numbers.add(key)

    valid_citations, invalid_citation = _validated_citations(draft.citations, context.chunks)
    missing_narrative_citation = bool(context.chunks) and not valid_citations
    has_grounding_issue = invalid_number or invalid_citation or missing_narrative_citation

    if has_grounding_issue:
        text = (
            _UNSUPPORTED_CLAIMS_TEXT
            if valid_numbers or valid_citations
            else _NO_GROUNDED_ANSWER_TEXT
        )
    else:
        text = draft.text
    return Answer(
        text=text,
        numbers=valid_numbers,
        citations=valid_citations,
        grounded=not has_grounding_issue,
    )


def _number_key(fact: NumberFact) -> NumberKey:
    """Return the exact identity required for numeric grounding."""
    return (
        fact.metric,
        fact.fiscal_year,
        fact.fiscal_period,
        fact.value,
        fact.unit,
    )


def _validated_citations(
    citations: list[Citation],
    chunks: list[Chunk],
) -> tuple[list[Citation], bool]:
    """Keep citations that resolve exactly to retrieved chunk metadata."""
    chunks_by_url: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_url.setdefault(chunk["source_url"], []).append(chunk)

    valid: list[Citation] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    invalid = False
    for citation in citations:
        candidates = chunks_by_url.get(citation.source_url, [])
        matching = [
            chunk
            for chunk in candidates
            if (citation.section is None or citation.section == chunk["section"])
            and (citation.accession_no is None or citation.accession_no == chunk["accession_no"])
        ]
        if not matching:
            invalid = True
            continue
        chunk = matching[0]
        canonical = Citation(
            source_url=chunk["source_url"],
            section=chunk["section"],
            accession_no=chunk["accession_no"],
        )
        key = (canonical.source_url, canonical.section, canonical.accession_no)
        if key not in seen:
            valid.append(canonical)
            seen.add(key)
    return valid, invalid


def _contains_prompt_disclosure(text: str) -> bool:
    """Detect direct, labeled, or verbatim disclosure of protected instructions."""
    normalized_text = unicodedata.normalize("NFKC", text).casefold()
    if any(pattern.search(normalized_text) is not None for pattern in _PROMPT_DISCLOSURE_PATTERNS):
        return True
    normalized_words = _normalized_instruction_words(normalized_text)
    return any(ngram in normalized_words for ngram in _protected_prompt_ngrams())


@lru_cache(maxsize=1)
def _protected_prompt_ngrams() -> frozenset[str]:
    """Return exact word windows that identify verbatim generation instructions."""
    normalized_prompt = _normalized_instruction_words(_GENERATION_INSTRUCTIONS)
    words = normalized_prompt.split()
    return frozenset(
        " ".join(words[index : index + _PROMPT_NGRAM_SIZE])
        for index in range(len(words) - _PROMPT_NGRAM_SIZE + 1)
    )


def _normalized_instruction_words(text: str) -> str:
    """Normalize instruction text for punctuation-independent comparison."""
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
