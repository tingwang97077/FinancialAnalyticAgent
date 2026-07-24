"""Typed failures raised by agent tools and orchestration."""

from __future__ import annotations


class AgentToolError(RuntimeError):
    """Base class for expected tool failures that are not configuration errors."""


class GeneratedSQLRejectedError(AgentToolError):
    """Raised when generated SQL does not pass the read-only validator."""


class GeneratedSQLExecutionError(AgentToolError):
    """Raised when PostgreSQL rejects the syntax of a generated query."""


class GeneratedSQLResultError(AgentToolError):
    """Raised when generated SQL returns rows outside the NumberFact contract."""


class IncompleteComparisonError(AgentToolError):
    """Raised when SQL omits required delta or percentage facts."""


class FinancialDataUnavailableError(AgentToolError):
    """Raised when no canonical fact exists for a supported numeric request."""
