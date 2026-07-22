"""FastAPI dependencies for application settings, database sessions, and the agent."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from fastapi import Request
from sqlalchemy.orm import Session

from ffa.agent.router import AgentRun
from ffa.config import Settings
from ffa.monitoring.tracing import RequestTracer


class AgentRunner(Protocol):
    """Callable contract implemented by the end-to-end agent entry point."""

    def __call__(
        self,
        question: str,
        *,
        trace_id: str | None = None,
    ) -> AgentRun:
        """Run the grounded agent pipeline for one question."""
        ...


async def get_settings(request: Request) -> Settings:
    """Return the settings instance owned by the FastAPI application."""
    return request.app.state.settings


async def get_db_session(request: Request) -> AsyncIterator[Session]:
    """Yield a request-scoped read-write database session."""
    session: Session = request.app.state.session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def get_agent_runner(request: Request) -> AgentRunner:
    """Return the configured end-to-end agent entry point."""
    return request.app.state.agent_runner


async def get_request_tracer(request: Request) -> RequestTracer:
    """Return the request tracer owned by the FastAPI application."""
    return request.app.state.request_tracer
