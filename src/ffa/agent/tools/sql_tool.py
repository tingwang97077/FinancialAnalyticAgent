"""Deterministic numeric tool backed exclusively by read-only PostgreSQL."""

from __future__ import annotations

import json
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, text

from ffa.agent.guardrails import ValidatedSQL, validate_sql
from ffa.agent.schemas import Intent, NumberFact, Understanding
from ffa.common.db import create_readonly_engine
from ffa.config import Settings, get_settings
from ffa.monitoring.tracing import record_openai_response

_STATEMENT_TIMEOUT_MS = 5_000
_SET_READ_ONLY = text("SET TRANSACTION READ ONLY")
_SET_STATEMENT_TIMEOUT = text("SELECT set_config('statement_timeout', :statement_timeout, true)")
_FIXED_SQL_SCHEMA = """Allowed PostgreSQL schema (no other tables or columns exist):

companies(
  cik BIGINT PRIMARY KEY,
  ticker TEXT,
  name TEXT,
  sic TEXT,
  updated_at TIMESTAMPTZ
)

filings(
  accession_no TEXT PRIMARY KEY,
  cik BIGINT REFERENCES companies(cik),
  form_type TEXT,
  filing_date DATE,
  period_of_report DATE,
  primary_doc_url TEXT
)

financial_facts(
  id BIGINT PRIMARY KEY,
  cik BIGINT REFERENCES companies(cik),
  ticker TEXT,
  metric TEXT,
  taxonomy_tag TEXT,
  unit TEXT,
  fiscal_year INT,
  fiscal_period TEXT,
  period_start DATE,
  period_end DATE,
  value NUMERIC,
  form_type TEXT,
  filing_date DATE,
  accession_no TEXT REFERENCES filings(accession_no),
  source_url TEXT
)
"""
_SQL_SYSTEM_INSTRUCTIONS = f"""You generate one PostgreSQL SELECT for a financial facts tool.
{_FIXED_SQL_SCHEMA}

Rules:
- Query only the documented schema and always qualify columns in joins.
- Return exactly five columns named metric, fiscal_year, fiscal_period, value, and unit.
- Return numeric facts only; never produce explanatory text.
- Perform every calculation in SQL. Growth, ratios, deltas, and comparisons must be SQL arithmetic,
  aggregates, window expressions, or self-joins over financial_facts.
- The request JSON includes required_comparisons. When it is non-empty, the result MUST contain
  both derived rows named by delta_metric and percent_change_metric for every requirement.
- Compute each delta as the SQL expression to_value - from_value. Compute each percentage change
  as 100 * (to_value - from_value) / NULLIF(from_value, 0), entirely inside PostgreSQL.
- Copy the source unit to the delta row, use the literal unit 'percent' for the percentage row,
  and assign both derived rows to the comparison's to fiscal year and period.
- Use the exact derived metric names supplied in required_comparisons. Returning only the two
  underlying period values does not satisfy a comparison request.
- The parsed root statement must be a SELECT. When multiple fact rows require UNION ALL, put every
  UNION branch inside a derived table and use one outer SELECT for the five required columns.
- Never calculate a final value yourself and never place a computed answer as a numeric literal.
- Use fiscal_year and fiscal_period as declared; do not infer calendar periods.
- Prefer the most recent filing_date when duplicate reported facts could exist.
- Do not use SELECT *, write operations, system catalogs, comments, or multiple statements.
- Use an explicit LIMIT no greater than 100; the application also enforces this limit.

Return only the structured SQL payload required by the response schema.
"""


class GeneratedSQL(BaseModel):
    """Structured SQL payload returned by the model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sql: str = Field(min_length=1)


class SQLGenerationProvider:
    """OpenAI structured-output adapter for SQL generation."""

    def __init__(self, client: OpenAI) -> None:
        """Initialize the provider with an OpenAI SDK client."""
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> SQLGenerationProvider:
        """Build the provider from central application settings."""
        resolved_settings = settings or get_settings()
        if resolved_settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY must be configured before generating SQL.")
        api_key = resolved_settings.openai_api_key.get_secret_value().strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be configured before generating SQL.")
        return cls(OpenAI(api_key=api_key))

    def generate_sql(self, understanding: Understanding, *, model: str) -> GeneratedSQL:
        """Generate one SQL payload through native structured output."""
        response = self._client.responses.parse(
            model=model,
            instructions=_SQL_SYSTEM_INSTRUCTIONS,
            input=_sql_generation_input(understanding),
            text_format=GeneratedSQL,
        )
        record_openai_response(response, model=model)
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("The model did not return a SQL query.")
        return parsed


class SqlTool:
    """Generate, validate, execute, and type numeric SQL results."""

    def __init__(
        self,
        *,
        provider: SQLGenerationProvider,
        readonly_engine: Engine,
        model: str,
        statement_timeout_ms: int = _STATEMENT_TIMEOUT_MS,
    ) -> None:
        """Initialize the tool with explicit read-only dependencies."""
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("OPENAI_MODEL must be configured before generating SQL.")
        if (
            isinstance(statement_timeout_ms, bool)
            or not isinstance(statement_timeout_ms, int)
            or statement_timeout_ms <= 0
        ):
            raise ValueError("Statement timeout must be a positive integer.")
        self._provider = provider
        self._readonly_engine = readonly_engine
        self._model = normalized_model
        self._statement_timeout_ms = statement_timeout_ms

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> SqlTool:
        """Build the tool exclusively from ``DATABASE_URL_READONLY``."""
        resolved_settings = settings or get_settings()
        readonly_url = _validated_readonly_url(resolved_settings)
        return cls(
            provider=SQLGenerationProvider.from_settings(resolved_settings),
            readonly_engine=create_readonly_engine(readonly_url),
            model=resolved_settings.openai_model,
        )

    def run(self, understanding: Understanding) -> list[NumberFact]:
        """Return typed facts after guarded SQL execution."""
        if understanding.intent not in {Intent.NUMERIC, Intent.HYBRID}:
            raise ValueError("The SQL tool only accepts numeric or hybrid understanding.")
        generated = self._provider.generate_sql(understanding, model=self._model)
        validated = validate_sql(generated.sql)
        facts = self._execute(validated)
        _validate_required_comparison_facts(understanding, facts)
        return facts

    def _execute(self, query: ValidatedSQL) -> list[NumberFact]:
        """Execute a validated query inside a bounded read-only transaction."""
        with self._readonly_engine.connect() as connection, connection.begin():
            connection.execute(_SET_READ_ONLY)
            connection.execute(
                _SET_STATEMENT_TIMEOUT,
                {"statement_timeout": f"{self._statement_timeout_ms}ms"},
            )
            rows = connection.execute(text(query.sql)).mappings().all()
        return [NumberFact.model_validate(dict(row)) for row in rows]


def sql_tool(understanding: Understanding) -> list[NumberFact]:
    """Answer a numeric question with validated SQL and typed database facts."""
    return _get_default_sql_tool().run(understanding)


@lru_cache(maxsize=1)
def _get_default_sql_tool() -> SqlTool:
    """Reuse the OpenAI client and pooled read-only engine across requests."""
    return SqlTool.from_settings(get_settings())


def _validated_readonly_url(settings: Settings) -> str:
    """Reject missing or accidentally shared read-write database configuration."""
    readonly_url = settings.database_url_readonly.strip()
    if not readonly_url:
        raise ValueError("DATABASE_URL_READONLY must be configured for the SQL tool.")
    if readonly_url == settings.database_url.strip():
        raise ValueError("DATABASE_URL_READONLY must differ from DATABASE_URL.")
    return readonly_url


def _sql_generation_input(understanding: Understanding) -> str:
    """Serialize the understanding with deterministic comparison requirements."""
    payload = {
        "understanding": understanding.model_dump(mode="json"),
        "required_comparisons": _comparison_requirements(understanding),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _comparison_requirements(understanding: Understanding) -> list[dict[str, object]]:
    """Describe derived SQL facts required by multi-period questions."""
    entities = understanding.entities
    requirements: list[dict[str, object]] = []
    if len(entities.fiscal_years) > 1:
        requirements.extend(
            _dimension_requirements(
                entities.metrics,
                dimension="fiscal_year",
                values=entities.fiscal_years,
                metric_suffix="yoy",
            )
        )
    if len(entities.fiscal_periods) > 1:
        requirements.extend(
            _dimension_requirements(
                entities.metrics,
                dimension="fiscal_period",
                values=entities.fiscal_periods,
                metric_suffix="period",
            )
        )
    return requirements


def _dimension_requirements(
    metrics: list[str],
    *,
    dimension: str,
    values: list[int] | list[str],
    metric_suffix: str,
) -> list[dict[str, object]]:
    """Build exact metric names and direction for one comparison dimension."""
    return [
        {
            "source_metric": metric,
            "dimension": dimension,
            "from_value": values[0],
            "to_value": values[-1],
            "delta_metric": f"{metric}_{metric_suffix}_delta",
            "percent_change_metric": f"{metric}_{metric_suffix}_percent_change",
        }
        for metric in metrics
    ]


def _validate_required_comparison_facts(
    understanding: Understanding,
    facts: list[NumberFact],
) -> None:
    """Reject comparison results that omit SQL-derived delta or percentage facts."""
    requirements = _comparison_requirements(understanding)
    required_metrics = {
        str(requirement[field])
        for requirement in requirements
        for field in ("delta_metric", "percent_change_metric")
    }
    returned_metrics = {fact.metric for fact in facts}
    if missing_metrics := required_metrics.difference(returned_metrics):
        missing_count = len(missing_metrics)
        raise RuntimeError(f"SQL result is missing {missing_count} required comparison fact(s).")
