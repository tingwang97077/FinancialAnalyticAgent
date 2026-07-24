"""HTTP routes for grounded questions, user feedback, and health checks."""

from __future__ import annotations

import logging
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response, status
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ffa.agent.errors import (
    GeneratedSQLExecutionError,
    GeneratedSQLRejectedError,
    GeneratedSQLResultError,
    IncompleteComparisonError,
)
from ffa.agent.schemas import Answer
from ffa.api.deps import (
    AgentRunner,
    get_agent_runner,
    get_db_session,
    get_request_tracer,
)
from ffa.monitoring.metrics import insert_query_log
from ffa.monitoring.tracing import RequestTracer

logger = logging.getLogger(__name__)

router = APIRouter()

DbSession = Annotated[Session, Depends(get_db_session)]
InjectedAgentRunner = Annotated[AgentRunner, Depends(get_agent_runner)]
InjectedRequestTracer = Annotated[RequestTracer, Depends(get_request_tracer)]

_INTENT_BY_ROUTE = {
    "sql_tool": "numeric",
    "retrieval_tool": "narrative",
    "hybrid": "hybrid",
    "out_of_scope": "out_of_scope",
    "blocked": "blocked",
}


class _ApiModel(BaseModel):
    """Strict base model for public API payloads."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AskRequest(_ApiModel):
    """Question submitted to the grounded financial agent."""

    question: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = Field(default=None, max_length=255)


class AskResponse(Answer):
    """Grounded answer with the end-to-end trace identifier."""

    trace_id: str = Field(min_length=1)


class FeedbackRequest(_ApiModel):
    """User rating associated with a prior answer trace."""

    trace_id: str = Field(min_length=1, max_length=255)
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=4_000)


class HealthResponse(_ApiModel):
    """Liveness and database connectivity status."""

    status: Literal["ok"] = "ok"
    database: Literal["ok"] = "ok"


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    db: DbSession,
    agent_runner: InjectedAgentRunner,
    request_tracer: InjectedRequestTracer,
) -> AskResponse:
    """Run the existing agent pipeline and persist its enriched query trace."""
    trace_id = uuid4().hex
    try:
        with request_tracer.trace(
            trace_id=trace_id,
            question=payload.question,
            session_id=payload.session_id,
        ) as request_trace:
            run = agent_runner(payload.question, trace_id=trace_id)
            request_trace.set_result(
                route=run.context.route,
                grounded=run.answer.grounded,
            )
        if run.context.trace_id != trace_id:
            raise RuntimeError("The agent returned an inconsistent trace identifier.")
        insert_query_log(
            db,
            trace_id=run.context.trace_id,
            session_id=payload.session_id,
            question=payload.question,
            intent=_INTENT_BY_ROUTE.get(run.context.route, "unknown"),
            route=run.context.route,
            grounded=run.answer.grounded,
            metrics=request_trace.metrics,
        )
        db.commit()
    except OpenAIError as exc:
        db.rollback()
        logger.warning("Agent upstream unavailable", extra={"trace_id": trace_id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The answer service is temporarily unavailable.",
        ) from exc
    except (
        GeneratedSQLRejectedError,
        GeneratedSQLExecutionError,
        GeneratedSQLResultError,
        IncompleteComparisonError,
    ) as exc:
        db.rollback()
        logger.warning("Generated financial query failed", extra={"trace_id": trace_id})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The financial query could not be completed safely.",
        ) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("Agent database unavailable", extra={"trace_id": trace_id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The database is temporarily unavailable.",
        ) from exc
    except ValueError as exc:
        db.rollback()
        logger.error("Agent configuration invalid", extra={"trace_id": trace_id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The answer service is not configured correctly.",
        ) from exc
    except Exception as exc:
        db.rollback()
        logger.error("Agent request failed", extra={"trace_id": trace_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed.",
        ) from exc

    return AskResponse(trace_id=trace_id, **run.answer.model_dump())


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def submit_feedback(payload: FeedbackRequest, db: DbSession) -> Response:
    """Persist an explicit positive or negative rating for a prior trace."""
    try:
        db.execute(
            text(
                """
                INSERT INTO feedback (trace_id, rating, comment)
                VALUES (:trace_id, :rating, :comment)
                """
            ),
            {
                "trace_id": payload.trace_id,
                "rating": payload.rating,
                "comment": payload.comment,
            },
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("Feedback database unavailable", extra={"trace_id": payload.trace_id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The database is temporarily unavailable.",
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/healthz", response_model=HealthResponse)
async def healthz(db: DbSession) -> HealthResponse:
    """Report liveness only when the application database is reachable."""
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("Health check database unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The database is unavailable.",
        ) from exc
    return HealthResponse()
