"""Structured intent classification, entity extraction, and query rewriting."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Protocol

from openai import OpenAI

from ffa.agent.schemas import Entities, Intent, Understanding
from ffa.common.entities import (
    Company,
    EntityResolutionError,
    normalize_ticker,
    resolve_universe,
)
from ffa.config import Settings, get_settings

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTIONS = """You classify questions for a financial fundamentals agent.
Return only the structured Understanding object required by the response schema.

Intent definitions:
- numeric: requests reported numeric financial facts or comparisons that must use SQL.
- narrative: requests explanations, disclosures, risks, or management commentary from filings.
- hybrid: genuinely requires both reported numbers and narrative filing evidence.
- out_of_scope: unrelated to public-company fundamentals or SEC filings.

Extract canonical ticker symbols when a company is named, but do not invent a ticker.
Extract fiscal years and fiscal periods exactly as requested; valid periods are FY and Q1-Q4.
Extract a filing section only when the question clearly targets it.
Use only these exact corpus names:
- MD&A for an explicit management discussion and analysis request.
- Risk Factors for an explicit question about disclosed risks.
- Notes for an explicit question about financial statement notes.
Otherwise return an empty sections list. Never force a section for a purely numeric question.
Use snake_case metric names when a financial metric is explicit. Rewrite the question into a
self-contained retrieval/query formulation. Never calculate, estimate, or supply a numeric answer.
For out-of-scope questions, return empty entity lists and preserve the request as rewritten_query.
"""

type EntityResolver = Callable[[Sequence[str]], list[Company]]


class UnderstandingError(RuntimeError):
    """Raised when a question cannot be converted to a safe typed understanding."""


class UnderstandingProvider(Protocol):
    """Backend contract for one structured understanding request."""

    def create_understanding(self, question: str, *, model: str) -> Understanding:
        """Return a schema-validated understanding for one question."""
        ...


class OpenAIUnderstandingProvider:
    """OpenAI Responses API adapter using native structured output parsing."""

    def __init__(self, client: OpenAI) -> None:
        """Initialize the adapter with an OpenAI SDK client."""
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> OpenAIUnderstandingProvider:
        """Build the adapter from central application settings."""
        resolved_settings = settings or get_settings()
        if resolved_settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY must be configured before understanding questions.")
        api_key = resolved_settings.openai_api_key.get_secret_value().strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be configured before understanding questions.")
        return cls(OpenAI(api_key=api_key))

    def create_understanding(self, question: str, *, model: str) -> Understanding:
        """Make one structured-output Responses API call."""
        response = self._client.responses.parse(
            model=model,
            instructions=_SYSTEM_INSTRUCTIONS,
            input=question,
            text_format=Understanding,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise UnderstandingError("The question could not be classified.")
        return parsed


def understand(
    question: str,
    *,
    provider: UnderstandingProvider | None = None,
    entity_resolver: EntityResolver | None = None,
    settings: Settings | None = None,
) -> Understanding:
    """Classify and resolve a question through one structured LLM call.

    Unknown or malformed ticker candidates are converted to ``out_of_scope`` with
    no CIKs, preventing downstream tools from receiving an unresolved identifier.
    """
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Question must not be empty.")

    resolved_settings = settings or get_settings()
    model = _select_classifier_model(resolved_settings)
    resolved_provider = provider or OpenAIUnderstandingProvider.from_settings(resolved_settings)
    draft = resolved_provider.create_understanding(normalized_question, model=model)

    if draft.intent is Intent.OUT_OF_SCOPE:
        return draft.model_copy(update={"entities": Entities()})
    if not draft.entities.tickers:
        return draft

    try:
        tickers = _normalize_tickers(draft.entities.tickers)
        resolver = entity_resolver or _settings_resolver(resolved_settings)
        companies = resolver(tickers)
        if len(companies) != len(tickers):
            raise EntityResolutionError("Entity resolver returned an incomplete result.")
    except EntityResolutionError:
        logger.warning("Question contains an unresolved company identifier.")
        return draft.model_copy(
            update={
                "intent": Intent.OUT_OF_SCOPE,
                "entities": draft.entities.model_copy(update={"ciks": []}),
            }
        )

    resolved_entities = draft.entities.model_copy(
        update={
            "tickers": [company["ticker"] for company in companies],
            "ciks": [company["cik"] for company in companies],
        }
    )
    return draft.model_copy(update={"entities": resolved_entities})


def _select_classifier_model(settings: Settings) -> str:
    """Select the classifier model, falling back to the primary chat model."""
    classifier_model = settings.openai_classifier_model.strip()
    primary_model = settings.openai_model.strip()
    selected_model = classifier_model or primary_model
    if not selected_model:
        raise ValueError("OPENAI_CLASSIFIER_MODEL or OPENAI_MODEL must be configured.")
    return selected_model


def _normalize_tickers(tickers: Sequence[str]) -> list[str]:
    """Normalize and deduplicate LLM ticker candidates."""
    return list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers))


def _settings_resolver(settings: Settings) -> EntityResolver:
    """Bind the existing SEC entity resolver to the active settings."""

    def resolve(tickers: Sequence[str]) -> list[Company]:
        return resolve_universe(tickers, settings=settings)

    return resolve
