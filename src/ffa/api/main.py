"""FastAPI application factory for the Financial Fundamentals Agent."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import Engine

from ffa.agent.router import run_agent
from ffa.agent.tools.retrieval_tool import preload_retrieval
from ffa.api.deps import AgentRunner
from ffa.api.routes import router
from ffa.common.db import create_rw_engine, create_session_factory
from ffa.config import Settings, get_settings
from ffa.monitoring.tracing import RequestTracer

logger = logging.getLogger(__name__)

_RERANKER_PRELOAD_TIMEOUT_SECONDS = 30.0


def create_app(
    *,
    settings: Settings | None = None,
    engine: Engine | None = None,
    agent_runner: AgentRunner | None = None,
    request_tracer: RequestTracer | None = None,
    retrieval_preloader: Callable[[], None] | None = None,
) -> FastAPI:
    """Create the API with injectable settings, database, and agent dependencies."""
    resolved_settings = settings or get_settings()
    owns_engine = engine is None
    resolved_engine = engine or create_rw_engine(resolved_settings.database_url)
    resolved_tracer = request_tracer or RequestTracer.from_settings(resolved_settings)
    resolved_preloader = retrieval_preloader or preload_retrieval

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            await _run_preloader(resolved_preloader)
            logger.info("Retrieval reranker preloaded for this API worker.")
        except TimeoutError:
            logger.warning(
                "Retrieval reranker preload timed out; startup continues with lazy fallback."
            )
        except Exception:
            logger.exception(
                "Retrieval reranker preload failed; startup continues with lazy fallback."
            )
        yield
        resolved_tracer.flush()
        if owns_engine:
            resolved_engine.dispose()

    application = FastAPI(
        title="Financial Fundamentals Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.engine = resolved_engine
    application.state.session_factory = create_session_factory(resolved_engine)
    application.state.agent_runner = agent_runner or run_agent
    application.state.request_tracer = resolved_tracer
    application.include_router(router)
    return application


async def _run_preloader(preloader: Callable[[], None]) -> None:
    """Run blocking model loading with a bounded, shutdown-safe wait."""
    loop = asyncio.get_running_loop()
    completion: asyncio.Future[None] = loop.create_future()

    def finish(error: Exception | None) -> None:
        if completion.done():
            return
        if error is None:
            completion.set_result(None)
        else:
            completion.set_exception(error)

    def load() -> None:
        try:
            preloader()
        except Exception as exc:
            error: Exception | None = exc
        else:
            error = None
        try:
            loop.call_soon_threadsafe(finish, error)
        except RuntimeError:
            return

    threading.Thread(
        target=load,
        name="ffa-reranker-preload",
        daemon=True,
    ).start()
    await asyncio.wait_for(
        completion,
        timeout=_RERANKER_PRELOAD_TIMEOUT_SECONDS,
    )


app = create_app()
