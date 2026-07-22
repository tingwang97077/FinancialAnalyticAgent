"""Optional Langfuse tracing and request-local OpenAI usage collection."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal
from time import perf_counter_ns
from typing import Any

from langfuse import Langfuse

from ffa.config import Settings
from ffa.monitoring.metrics import (
    CallKind,
    PriceBook,
    QueryMetrics,
    TokenUsage,
    extract_token_usage,
)

logger = logging.getLogger(__name__)

_TRACE_ID_PATTERN = re.compile(r"[0-9a-f]{32}", flags=re.IGNORECASE)
_LANGFUSE_STAGE_TYPES = {
    "guardrails.check_input": "guardrail",
    "understand": "generation",
    "router": "generation",
    "sql_tool": "generation",
    "retrieval_tool": "embedding",
    "generation": "generation",
}


@dataclass(slots=True)
class StageMetrics:
    """Latency and model usage accumulated by one pipeline span."""

    name: str
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: Decimal = Decimal("0")
    models: set[str] = field(default_factory=set)

    def add(self, *, model: str, usage: TokenUsage, cost_usd: Decimal) -> None:
        """Attach one model call to this stage."""
        self.models.add(model)
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cached_tokens += usage.cached_tokens
        self.cost_usd += cost_usd


@dataclass(slots=True)
class RequestTrace:
    """Mutable state for one globally correlated API request."""

    trace_id: str
    price_book: PriceBook
    metrics: QueryMetrics = field(default_factory=QueryMetrics)
    stages: list[StageMetrics] = field(default_factory=list)
    route: str | None = None
    grounded: bool | None = None
    _client: Any | None = None

    def set_result(self, *, route: str, grounded: bool) -> None:
        """Record the safe output metadata attached to the root trace."""
        self.route = route
        self.grounded = grounded

    def add_usage(self, *, model: str, usage: TokenUsage, kind: CallKind) -> None:
        """Aggregate one OpenAI call at request and active-stage levels."""
        cost_usd = self.price_book.cost(model=model, usage=usage, kind=kind)
        self.metrics.add(usage, cost_usd)
        stage = _CURRENT_STAGE.get()
        if stage is not None:
            stage.add(model=model, usage=usage, cost_usd=cost_usd)


class RequestTracer:
    """Create request traces while degrading to local metrics when disabled."""

    def __init__(self, *, settings: Settings, langfuse_client: Any | None = None) -> None:
        """Initialize pricing and an optional prebuilt Langfuse client."""
        self._price_book = PriceBook.from_settings(settings)
        self._client = langfuse_client

    @classmethod
    def from_settings(cls, settings: Settings) -> RequestTracer:
        """Build a tracer only when both Langfuse keys are non-empty."""
        public_key = _secret_value(settings.langfuse_public_key)
        secret_key = _secret_value(settings.langfuse_secret_key)
        if not public_key or not secret_key:
            return cls(settings=settings)
        arguments: dict[str, object] = {
            "public_key": public_key,
            "secret_key": secret_key,
        }
        if endpoint := settings.langfuse_endpoint:
            arguments["host"] = endpoint
        try:
            client = Langfuse(**arguments)
        except Exception:
            logger.exception("Langfuse initialization failed; tracing is disabled.")
            client = None
        return cls(settings=settings, langfuse_client=client)

    @property
    def enabled(self) -> bool:
        """Return whether remote Langfuse emission is active."""
        return self._client is not None

    @contextmanager
    def trace(
        self,
        *,
        trace_id: str,
        question: str,
        session_id: str | None,
    ) -> Iterator[RequestTrace]:
        """Collect one request and correlate it with the public API trace ID."""
        state = RequestTrace(
            trace_id=trace_id,
            price_book=self._price_book,
            _client=self._client,
        )
        token = _CURRENT_TRACE.set(state)
        started_at = perf_counter_ns()
        span_context = self._root_span(state, question=question)
        try:
            with span_context as root_span:
                self._update_current_trace(
                    question=question,
                    session_id=session_id,
                    trace_id=state.trace_id,
                )
                try:
                    yield state
                finally:
                    state.metrics.latency_ms = _elapsed_ms(started_at)
                    _update_span(root_span, state.metrics, output=_trace_output(state))
        finally:
            _CURRENT_TRACE.reset(token)

    def flush(self) -> None:
        """Flush queued telemetry without making application shutdown fail."""
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception:
            logger.exception("Langfuse flush failed during shutdown.")

    def _update_current_trace(
        self,
        *,
        question: str,
        session_id: str | None,
        trace_id: str,
    ) -> None:
        """Attach public correlation metadata after the root span is current."""
        if self._client is None:
            return
        trace_metadata: dict[str, object] = {
            "name": "ffa.ask",
            "input": {"question": question},
            "metadata": {"trace_id": trace_id},
        }
        if session_id:
            trace_metadata["session_id"] = session_id
        try:
            self._client.update_current_trace(**trace_metadata)
        except Exception:
            logger.exception("Langfuse trace update failed; request continues.")

    def _root_span(
        self,
        state: RequestTrace,
        *,
        question: str,
    ) -> Any:
        """Return the global Langfuse trace span or a no-op context."""
        if self._client is None:
            return nullcontext(None)
        langfuse_trace_id = (
            state.trace_id
            if _TRACE_ID_PATTERN.fullmatch(state.trace_id)
            else Langfuse.create_trace_id(seed=state.trace_id)
        )
        try:
            context = self._client.start_as_current_span(
                trace_context={"trace_id": langfuse_trace_id},
                name="ffa.ask",
                input={"question": question},
                metadata={"trace_id": state.trace_id},
            )
            return _safe_span_context(context)
        except Exception:
            logger.exception("Langfuse trace creation failed; request continues without export.")
            state._client = None
            return nullcontext(None)


_CURRENT_TRACE: ContextVar[RequestTrace | None] = ContextVar(
    "ffa_monitoring_trace",
    default=None,
)
_CURRENT_STAGE: ContextVar[StageMetrics | None] = ContextVar(
    "ffa_monitoring_stage",
    default=None,
)


@contextmanager
def observe_step(name: str) -> Iterator[StageMetrics | None]:
    """Create a timed pipeline span when an API trace is active."""
    state = _CURRENT_TRACE.get()
    if state is None:
        yield None
        return

    stage = StageMetrics(name=name)
    state.stages.append(stage)
    stage_token = _CURRENT_STAGE.set(stage)
    started_at = perf_counter_ns()
    span_context: Any = nullcontext(None)
    if state._client is not None:
        try:
            span_context = _safe_span_context(
                state._client.start_as_current_observation(
                    name=name,
                    as_type=_LANGFUSE_STAGE_TYPES.get(name, "span"),
                )
            )
        except Exception:
            logger.exception("Langfuse span creation failed; local metrics continue.")
    try:
        with span_context as remote_span:
            try:
                yield stage
            finally:
                stage.latency_ms = _elapsed_ms(started_at)
                _update_span(remote_span, stage)
    finally:
        _CURRENT_STAGE.reset(stage_token)


def record_openai_response(
    response: object,
    *,
    model: str,
    kind: CallKind = "model",
) -> None:
    """Attach SDK usage to the current request and innermost stage."""
    state = _CURRENT_TRACE.get()
    if state is None:
        return
    state.add_usage(
        model=model,
        usage=extract_token_usage(response),
        kind=kind,
    )


def _update_span(
    span: Any | None,
    metrics: QueryMetrics | StageMetrics,
    *,
    output: dict[str, object] | None = None,
) -> None:
    """Attach normalized telemetry without allowing exporter errors to escape."""
    if span is None:
        return
    usage_details = {
        "input": metrics.input_tokens,
        "output": metrics.output_tokens,
        "cached_input": metrics.cached_tokens,
        "total": metrics.input_tokens + metrics.output_tokens,
    }
    metadata: dict[str, object] = {
        "latency_ms": metrics.latency_ms,
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "cached_tokens": metrics.cached_tokens,
        "cost_usd": str(metrics.cost_usd),
    }
    model: str | None = None
    if isinstance(metrics, StageMetrics) and metrics.models:
        metadata["models"] = sorted(metrics.models)
        if len(metrics.models) == 1:
            model = next(iter(metrics.models))
    try:
        span.update(
            output=output,
            metadata=metadata,
            model=model,
            usage_details=usage_details,
            cost_details={"total": float(metrics.cost_usd)},
        )
    except Exception:
        logger.exception("Langfuse span update failed; request continues.")


def _trace_output(state: RequestTrace) -> dict[str, object]:
    """Return non-sensitive root output metadata."""
    return {"route": state.route, "grounded": state.grounded}


def _elapsed_ms(started_at: int) -> int:
    """Return an integer duration while preserving sub-millisecond work as 1 ms."""
    elapsed_ns = max(perf_counter_ns() - started_at, 0)
    return max(1, (elapsed_ns + 999_999) // 1_000_000)


def _secret_value(value: object) -> str:
    """Read an optional SecretStr without logging it."""
    if value is None:
        return ""
    getter = getattr(value, "get_secret_value", None)
    if getter is None:
        return ""
    return str(getter()).strip()


@contextmanager
def _safe_span_context(context: Any) -> Iterator[Any | None]:
    """Prevent exporter context failures from affecting application behavior."""
    try:
        span = context.__enter__()
    except Exception:
        logger.exception("Langfuse span start failed; request continues.")
        yield None
        return

    error: tuple[object | None, object | None, object | None] = (None, None, None)
    try:
        yield span
    except BaseException:
        error = sys.exc_info()
        raise
    finally:
        try:
            context.__exit__(*error)
        except Exception:
            logger.exception("Langfuse span end failed; request continues.")
