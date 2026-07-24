from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event
from time import perf_counter

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from openai import OpenAIError
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import OperationalError

import ffa.api.main as api_main_module
from ffa.agent.errors import (
    GeneratedSQLExecutionError,
    GeneratedSQLRejectedError,
    IncompleteComparisonError,
)
from ffa.agent.router import AgentContext, AgentRun
from ffa.agent.schemas import Answer, Citation, NumberFact
from ffa.api.deps import AgentRunner
from ffa.api.main import create_app
from ffa.config import Settings


class FakeAgentRunner:
    def __init__(self, answer: Answer, *, route: str = "sql_tool") -> None:
        self.answer = answer
        self.route = route
        self.calls: list[tuple[str, str]] = []

    def __call__(self, question: str, *, trace_id: str | None = None) -> AgentRun:
        assert trace_id is not None
        self.calls.append((question, trace_id))
        return AgentRun(
            answer=self.answer,
            context=AgentContext(route=self.route, trace_id=trace_id),
        )


class UnavailableAgentRunner:
    def __call__(self, question: str, *, trace_id: str | None = None) -> AgentRun:
        del question, trace_id
        raise OpenAIError("upstream unavailable")


class FailingAgentRunner:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def __call__(self, question: str, *, trace_id: str | None = None) -> AgentRun:
        del question, trace_id
        raise self._error


@pytest.mark.asyncio
async def test_ask_returns_raw_answer_trace_and_persists_minimal_query_log(
    tmp_path: Path,
) -> None:
    fact = NumberFact(
        metric="net_income",
        fiscal_year=2024,
        fiscal_period="FY",
        value=93_736_000_000,
        unit="USD",
    )
    runner = FakeAgentRunner(
        Answer(
            text="Apple reported FY2024 net income of USD 93,736,000,000.",
            numbers=[fact],
            grounded=True,
        )
    )
    app, engine = _test_app(tmp_path, runner)

    async with _client(app) as client:
        response = await client.post(
            "/ask",
            json={
                "question": "What was Apple's net income in FY2024?",
                "session_id": "session-123",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["trace_id"] == runner.calls[0][1]
    assert payload["numbers"] == [fact.model_dump()]
    assert payload["citations"] == []
    assert payload["grounded"] is True

    with engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                SELECT trace_id, session_id, question, intent, route, grounded,
                       latency_ms, input_tokens, output_tokens, cached_tokens, cost_usd
                FROM query_logs
                """
                )
            )
            .mappings()
            .one()
        )
    assert row["latency_ms"] >= 1
    assert dict(row) == {
        "trace_id": payload["trace_id"],
        "session_id": "session-123",
        "question": "What was Apple's net income in FY2024?",
        "intent": "numeric",
        "route": "sql_tool",
        "grounded": 1,
        "latency_ms": row["latency_ms"],
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "cost_usd": 0,
    }


@pytest.mark.asyncio
async def test_ask_preserves_citations_without_api_side_rewriting(tmp_path: Path) -> None:
    citation = Citation(
        source_url="https://www.sec.gov/Archives/example.htm",
        section="Risk Factors",
        accession_no="0000000000-26-000001",
    )
    runner = FakeAgentRunner(
        Answer(text="A cited risk.", citations=[citation], grounded=True),
        route="retrieval_tool",
    )
    app, _ = _test_app(tmp_path, runner)

    async with _client(app) as client:
        response = await client.post("/ask", json={"question": "What risks are cited?"})

    assert response.status_code == 200
    assert response.json()["citations"] == [citation.model_dump()]


@pytest.mark.asyncio
async def test_feedback_returns_204_and_persists_the_rating(tmp_path: Path) -> None:
    app, engine = _test_app(tmp_path, FakeAgentRunner(Answer(text="Unused.")))

    async with _client(app) as client:
        response = await client.post(
            "/feedback",
            json={"trace_id": "trace-feedback", "rating": -1, "comment": "Needs detail."},
        )

    assert response.status_code == 204
    assert response.content == b""
    with engine.connect() as connection:
        row = (
            connection.execute(text("SELECT trace_id, rating, comment FROM feedback"))
            .mappings()
            .one()
        )
    assert dict(row) == {
        "trace_id": "trace-feedback",
        "rating": -1,
        "comment": "Needs detail.",
    }


@pytest.mark.asyncio
async def test_healthz_checks_database_connectivity(tmp_path: Path) -> None:
    app, _ = _test_app(tmp_path, FakeAgentRunner(Answer(text="Unused.")))

    async with _client(app) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


@pytest.mark.asyncio
async def test_empty_question_and_invalid_rating_are_rejected(tmp_path: Path) -> None:
    runner = FakeAgentRunner(Answer(text="Must not run."))
    app, _ = _test_app(tmp_path, runner)

    async with _client(app) as client:
        empty_question = await client.post("/ask", json={"question": "   "})
        invalid_rating = await client.post(
            "/feedback",
            json={"trace_id": "trace", "rating": 0},
        )

    assert empty_question.status_code == 422
    assert invalid_rating.status_code == 422
    assert runner.calls == []


@pytest.mark.asyncio
async def test_openai_failure_returns_sanitized_503(tmp_path: Path) -> None:
    app, _ = _test_app(tmp_path, UnavailableAgentRunner())

    async with _client(app) as client:
        response = await client.post("/ask", json={"question": "A valid financial question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "The answer service is temporarily unavailable."}
    assert "upstream unavailable" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        GeneratedSQLRejectedError("unsafe generated SQL detail"),
        GeneratedSQLExecutionError("invalid generated SQL detail"),
        IncompleteComparisonError("missing comparison detail"),
    ],
)
async def test_generated_query_failures_return_honest_sanitized_502(
    tmp_path: Path,
    error: Exception,
) -> None:
    app, _ = _test_app(tmp_path, FailingAgentRunner(error))

    async with _client(app) as client:
        response = await client.post("/ask", json={"question": "A numeric question"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The financial query could not be completed safely."}
    assert "configured" not in response.text
    assert "database" not in response.text
    assert str(error) not in response.text


@pytest.mark.asyncio
async def test_database_failure_returns_sanitized_503(tmp_path: Path) -> None:
    app, engine = _test_app(tmp_path, FakeAgentRunner(Answer(text="Unused.")))

    @event.listens_for(engine, "before_cursor_execute")
    def fail_database(*_: object) -> None:
        raise OperationalError("SELECT 1", {}, RuntimeError("database secret detail"))

    async with _client(app) as client:
        response = await client.get("/healthz")

    assert response.status_code == 503
    assert response.json() == {"detail": "The database is unavailable."}
    assert "database secret detail" not in response.text


@pytest.mark.asyncio
async def test_lifespan_preloads_retrieval_once_per_application_worker(tmp_path: Path) -> None:
    preload_calls: list[str] = []
    app, _ = _test_app(
        tmp_path,
        FakeAgentRunner(Answer(text="Unused.")),
        retrieval_preloader=lambda: preload_calls.append("loaded"),
    )

    async with app.router.lifespan_context(app):
        assert preload_calls == ["loaded"]


@pytest.mark.asyncio
async def test_lifespan_continues_when_retrieval_preload_fails(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def unavailable_model() -> None:
        raise RuntimeError("model unavailable")

    app, _ = _test_app(
        tmp_path,
        FakeAgentRunner(Answer(text="Unused.")),
        retrieval_preloader=unavailable_model,
    )

    async with app.router.lifespan_context(app):
        assert "startup continues with lazy fallback" in caplog.text


@pytest.mark.asyncio
async def test_lifespan_bounds_a_stalled_retrieval_preload(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = Event()
    monkeypatch.setattr(api_main_module, "_RERANKER_PRELOAD_TIMEOUT_SECONDS", 0.01)
    app, _ = _test_app(
        tmp_path,
        FakeAgentRunner(Answer(text="Unused.")),
        retrieval_preloader=release.wait,
    )

    started_at = perf_counter()
    async with app.router.lifespan_context(app):
        elapsed = perf_counter() - started_at
    release.set()

    assert elapsed < 0.2
    assert "preload timed out" in caplog.text


def _test_app(
    tmp_path: Path,
    runner: AgentRunner,
    *,
    retrieval_preloader: Callable[[], None] | None = None,
) -> tuple[FastAPI, Engine]:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api.sqlite3'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE query_logs (
                    trace_id TEXT NOT NULL,
                    session_id TEXT,
                    question TEXT NOT NULL,
                    intent TEXT,
                    route TEXT,
                    latency_ms INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cached_tokens INTEGER,
                    cost_usd NUMERIC,
                    grounded BOOLEAN,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE feedback (
                    trace_id TEXT NOT NULL,
                    rating SMALLINT NOT NULL,
                    comment TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
    settings = Settings(_env_file=None, database_url="sqlite+pysqlite://")
    return (
        create_app(
            settings=settings,
            engine=engine,
            agent_runner=runner,
            retrieval_preloader=retrieval_preloader or (lambda: None),
        ),
        engine,
    )


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")
