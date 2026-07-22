"""Tests for request tracing, token extraction, and environment-priced costs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from ffa.agent.guardrails import GuardResult
from ffa.agent.router import PlannedToolCall, run_agent
from ffa.agent.schemas import Answer, Entities, Intent, NumberFact, Understanding
from ffa.config import Settings
from ffa.monitoring.metrics import PriceBook, TokenUsage, extract_token_usage
from ffa.monitoring.tracing import RequestTracer, observe_step, record_openai_response


class FakeSpan:
    """Capture Langfuse span updates without network access."""

    def __init__(self, name: str, observation_type: str = "span") -> None:
        self.name = name
        self.observation_type = observation_type
        self.updates: list[dict[str, object]] = []

    def update(self, **values: object) -> None:
        self.updates.append(values)


class FakeSpanContext(AbstractContextManager[FakeSpan]):
    """Minimal context manager returned by the fake Langfuse client."""

    def __init__(self, span: FakeSpan) -> None:
        self.span = span

    def __enter__(self) -> FakeSpan:
        return self.span

    def __exit__(self, *args: object) -> None:
        del args


class FakeLangfuse:
    """Record emitted root and child spans."""

    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []
        self.trace_updates: list[dict[str, object]] = []
        self.flushed = False

    def start_as_current_span(self, *, name: str, **_: object) -> FakeSpanContext:
        span = FakeSpan(name)
        self.spans.append(span)
        return FakeSpanContext(span)

    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: str,
        **_: object,
    ) -> FakeSpanContext:
        span = FakeSpan(name, as_type)
        self.spans.append(span)
        return FakeSpanContext(span)

    def update_current_trace(self, **values: object) -> None:
        self.trace_updates.append(values)

    def flush(self) -> None:
        self.flushed = True


class UsagePlanner:
    """Emit planner usage and return the required hybrid calls."""

    def plan(self, intent: Intent, *, trace_id: str) -> list[PlannedToolCall]:
        del intent, trace_id
        record_openai_response(_response(80, 20, cached=10), model="configured-mini")
        return [
            PlannedToolCall(name="sql_tool", call_id="sql"),
            PlannedToolCall(name="retrieval_tool", call_id="retrieval"),
        ]


def test_cached_tokens_use_cached_rate_and_models_select_distinct_prices() -> None:
    prices = PriceBook.from_settings(_priced_settings())
    usage = TokenUsage(input_tokens=1_000, output_tokens=200, cached_tokens=400)

    mini_cost = prices.cost(model="configured-mini", usage=usage)
    nano_cost = prices.cost(model="configured-nano", usage=usage)

    assert mini_cost == Decimal("0.003")
    assert nano_cost == Decimal("0.0003")
    assert mini_cost != nano_cost


def test_usage_extraction_supports_cached_responses_and_embedding_shapes() -> None:
    responses_usage = extract_token_usage(_response(900, 75, cached=300))
    embedding_usage = extract_token_usage(
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=240, total_tokens=240))
    )

    assert responses_usage == TokenUsage(900, 75, 300)
    assert embedding_usage == TokenUsage(240, 0, 0)


def test_pipeline_emits_all_stage_spans_and_aggregates_usage() -> None:
    fake_langfuse = FakeLangfuse()
    tracer = RequestTracer(
        settings=_priced_settings(),
        langfuse_client=fake_langfuse,
    )
    understanding = Understanding(
        intent=Intent.HYBRID,
        entities=Entities(tickers=["AAPL"], metrics=["net_income"]),
        rewritten_query="Apple net income and management commentary",
    )
    fact = NumberFact(
        metric="net_income",
        fiscal_year=2024,
        fiscal_period="FY",
        value=1,
        unit="USD",
    )

    def classify(_: str) -> Understanding:
        record_openai_response(_response(100, 30, cached=20), model="configured-nano")
        return understanding

    def run_sql(_: Understanding) -> list[NumberFact]:
        record_openai_response(_response(200, 50, cached=40), model="configured-mini")
        return [fact]

    def retrieve(_: Understanding) -> list[Any]:
        record_openai_response(
            _response(60, 0),
            model="configured-embedding",
            kind="embedding",
        )
        return []

    def generate(_: str, __: object) -> Answer:
        record_openai_response(_response(120, 40, cached=30), model="configured-mini")
        return Answer(text="Grounded.", numbers=[fact], grounded=True)

    with tracer.trace(
        trace_id="0123456789abcdef0123456789abcdef",
        question="Hybrid question",
        session_id="session-monitoring",
    ) as trace:
        result = run_agent(
            "Hybrid question",
            trace_id=trace.trace_id,
            guardrail_checker=lambda _: GuardResult(allowed=True, reason="Allowed."),
            understanding_fn=classify,
            planner=UsagePlanner(),
            sql_runner=run_sql,
            retrieval_runner=retrieve,
            answer_generator=generate,
        )
        trace.set_result(route=result.context.route, grounded=result.answer.grounded)

    assert [stage.name for stage in trace.stages] == [
        "guardrails.check_input",
        "understand",
        "router",
        "sql_tool",
        "retrieval_tool",
        "generation",
    ]
    assert [span.name for span in fake_langfuse.spans] == [
        "ffa.ask",
        "guardrails.check_input",
        "understand",
        "router",
        "sql_tool",
        "retrieval_tool",
        "generation",
    ]
    assert [span.observation_type for span in fake_langfuse.spans] == [
        "span",
        "guardrail",
        "generation",
        "generation",
        "generation",
        "embedding",
        "generation",
    ]
    assert trace.metrics.input_tokens == 560
    assert trace.metrics.output_tokens == 140
    assert trace.metrics.cached_tokens == 100
    assert trace.metrics.cost_usd > 0
    assert all(span.updates for span in fake_langfuse.spans)
    assert fake_langfuse.trace_updates[0]["session_id"] == "session-monitoring"


def test_missing_langfuse_keys_disable_export_without_disabling_metrics() -> None:
    settings = _priced_settings(
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )
    tracer = RequestTracer.from_settings(settings)

    with (
        tracer.trace(trace_id="trace-no-langfuse", question="Question", session_id=None) as trace,
        observe_step("understand"),
    ):
        record_openai_response(_response(100, 20, cached=25), model="configured-nano")

    assert tracer.enabled is False
    assert trace.metrics.input_tokens == 100
    assert trace.metrics.output_tokens == 20
    assert trace.metrics.cached_tokens == 25
    assert trace.metrics.cost_usd > 0
    tracer.flush()


def test_langfuse_base_url_is_used_when_host_is_empty() -> None:
    settings = Settings(
        _env_file=None,
        langfuse_host="",
        langfuse_base_url="https://langfuse.example.test",
        price_mini_input="",
    )

    assert settings.langfuse_endpoint == "https://langfuse.example.test"
    assert settings.price_mini_input is None


def _response(input_tokens: int, output_tokens: int, *, cached: int = 0) -> object:
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached),
        )
    )


def _priced_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "openai_model": "configured-mini",
        "openai_classifier_model": "configured-nano",
        "openai_embedding_model": "configured-embedding",
        "price_mini_input": "2",
        "price_mini_cached_input": "0.5",
        "price_mini_output": "8",
        "price_nano_input": "0.2",
        "price_nano_cached_input": "0.05",
        "price_nano_output": "0.8",
        "price_embedding": "0.02",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)
