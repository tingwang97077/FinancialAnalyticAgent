"""Token, cost, latency, and query-log metrics for agent requests."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

from ffa.config import Settings

_ONE_MILLION = Decimal("1000000")

type CallKind = Literal["model", "embedding"]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Normalized token counters from one OpenAI response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


@dataclass(slots=True)
class QueryMetrics:
    """Aggregated telemetry persisted for one API request."""

    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: Decimal = Decimal("0")

    def add(self, usage: TokenUsage, cost_usd: Decimal) -> None:
        """Accumulate one model or embedding call."""
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cached_tokens += usage.cached_tokens
        self.cost_usd += cost_usd


@dataclass(frozen=True, slots=True)
class ModelPrices:
    """Input, cached-input, and output prices in USD per million tokens."""

    input: Decimal | None
    cached_input: Decimal | None
    output: Decimal | None


@dataclass(frozen=True, slots=True)
class PriceBook:
    """Environment-backed prices and the configured model identities."""

    mini_model: str
    nano_model: str
    embedding_model: str
    mini: ModelPrices
    nano: ModelPrices
    embedding: Decimal | None

    @classmethod
    def from_settings(cls, settings: Settings) -> PriceBook:
        """Build a price book without supplying application-side price defaults."""
        return cls(
            mini_model=settings.openai_model.strip(),
            nano_model=settings.openai_classifier_model.strip(),
            embedding_model=settings.openai_embedding_model.strip(),
            mini=ModelPrices(
                input=settings.price_mini_input,
                cached_input=settings.price_mini_cached_input,
                output=settings.price_mini_output,
            ),
            nano=ModelPrices(
                input=settings.price_nano_input,
                cached_input=settings.price_nano_cached_input,
                output=settings.price_nano_output,
            ),
            embedding=settings.price_embedding,
        )

    def cost(self, *, model: str, usage: TokenUsage, kind: CallKind = "model") -> Decimal:
        """Calculate the cost of one call using the price set for its actual model."""
        normalized_model = model.strip()
        if kind == "embedding":
            if normalized_model != self.embedding_model or self.embedding is None:
                return Decimal("0")
            return Decimal(usage.input_tokens) * self.embedding / _ONE_MILLION

        prices = self._model_prices(normalized_model)
        if prices is None:
            return Decimal("0")
        uncached_tokens = max(usage.input_tokens - usage.cached_tokens, 0)
        return (
            Decimal(uncached_tokens) * (prices.input or Decimal("0"))
            + Decimal(usage.cached_tokens) * (prices.cached_input or Decimal("0"))
            + Decimal(usage.output_tokens) * (prices.output or Decimal("0"))
        ) / _ONE_MILLION

    def _model_prices(self, model: str) -> ModelPrices | None:
        """Select prices by configured model identity, never by a hardcoded model name."""
        if self.nano_model and model == self.nano_model:
            return self.nano
        if self.mini_model and model == self.mini_model:
            return self.mini
        return None


def extract_token_usage(response: object) -> TokenUsage:
    """Normalize Responses, Chat Completions, and Embeddings usage shapes."""
    usage = _member(response, "usage")
    if usage is None:
        return TokenUsage()
    input_tokens = _nonnegative_int(
        _member(usage, "input_tokens", _member(usage, "prompt_tokens", 0))
    )
    output_tokens = _nonnegative_int(
        _member(usage, "output_tokens", _member(usage, "completion_tokens", 0))
    )
    details = _member(
        usage,
        "input_tokens_details",
        _member(usage, "prompt_tokens_details"),
    )
    cached_tokens = min(
        _nonnegative_int(_member(details, "cached_tokens", 0)),
        input_tokens,
    )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )


def insert_query_log(
    db: Session,
    *,
    trace_id: str,
    session_id: str | None,
    question: str,
    intent: str,
    route: str,
    grounded: bool,
    metrics: QueryMetrics,
) -> None:
    """Persist the complete monitoring row for one successful question."""
    db.execute(
        text(
            """
            INSERT INTO query_logs (
                trace_id,
                session_id,
                question,
                intent,
                route,
                latency_ms,
                input_tokens,
                output_tokens,
                cached_tokens,
                cost_usd,
                grounded
            )
            VALUES (
                :trace_id,
                :session_id,
                :question,
                :intent,
                :route,
                :latency_ms,
                :input_tokens,
                :output_tokens,
                :cached_tokens,
                CAST(:cost_usd AS NUMERIC),
                :grounded
            )
            """
        ),
        {
            "trace_id": trace_id,
            "session_id": session_id,
            "question": question,
            "intent": intent,
            "route": route,
            "latency_ms": metrics.latency_ms,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "cached_tokens": metrics.cached_tokens,
            "cost_usd": str(metrics.cost_usd),
            "grounded": grounded,
        },
    )


def _member(value: object, name: str, default: Any = None) -> Any:
    """Read one member from either an SDK object or a mapping."""
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _nonnegative_int(value: object) -> int:
    """Coerce SDK counters while rejecting booleans and invalid values."""
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(parsed, 0)
