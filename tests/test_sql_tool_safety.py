"""Safety and deterministic-execution tests for the numeric SQL tool."""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

import ffa.agent.tools.sql_tool as sql_tool_module
from ffa.agent.guardrails import SQLValidationError, validate_sql
from ffa.agent.schemas import Entities, Intent, NumberFact, Understanding
from ffa.agent.tools.sql_tool import GeneratedSQL, SQLGenerationProvider, SqlTool
from ffa.config import Settings

_VALID_SELECT = """
SELECT
    ff.metric,
    ff.fiscal_year,
    ff.fiscal_period,
    ff.value,
    ff.unit
FROM financial_facts AS ff
WHERE ff.ticker = 'AAPL' AND ff.metric = 'net_income'
"""

_YOY_SQL = """
SELECT
    comparison.metric,
    comparison.fiscal_year,
    comparison.fiscal_period,
    comparison.value,
    comparison.unit
FROM (
    SELECT
        ff.metric,
        ff.fiscal_year,
        ff.fiscal_period,
        ff.value,
        ff.unit
    FROM financial_facts AS ff
    WHERE ff.ticker = 'AAPL'
      AND ff.metric = 'net_income'
      AND ff.fiscal_year IN (2023, 2024)
      AND ff.fiscal_period = 'FY'
    UNION ALL
    SELECT
        ff.metric || '_yoy_delta' AS metric,
        2024 AS fiscal_year,
        'FY' AS fiscal_period,
        MAX(CASE WHEN ff.fiscal_year = 2024 THEN ff.value END)
          - MAX(CASE WHEN ff.fiscal_year = 2023 THEN ff.value END) AS value,
        MAX(ff.unit) AS unit
    FROM financial_facts AS ff
    WHERE ff.ticker = 'AAPL'
      AND ff.metric = 'net_income'
      AND ff.fiscal_year IN (2023, 2024)
      AND ff.fiscal_period = 'FY'
    GROUP BY ff.metric
    UNION ALL
    SELECT
        ff.metric || '_yoy_percent_change' AS metric,
        2024 AS fiscal_year,
        'FY' AS fiscal_period,
        100 * (
            MAX(CASE WHEN ff.fiscal_year = 2024 THEN ff.value END)
            - MAX(CASE WHEN ff.fiscal_year = 2023 THEN ff.value END)
        ) / NULLIF(
            MAX(CASE WHEN ff.fiscal_year = 2023 THEN ff.value END),
            0
        ) AS value,
        'percent' AS unit
    FROM financial_facts AS ff
    WHERE ff.ticker = 'AAPL'
      AND ff.metric = 'net_income'
      AND ff.fiscal_year IN (2023, 2024)
      AND ff.fiscal_period = 'FY'
    GROUP BY ff.metric
) AS comparison
ORDER BY comparison.fiscal_year, comparison.metric
"""


class FakeSQLProvider:
    def __init__(self, sql: str) -> None:
        self.sql = sql
        self.calls: list[tuple[Understanding, str]] = []

    def generate_sql(self, understanding: Understanding, *, model: str) -> GeneratedSQL:
        self.calls.append((understanding, model))
        return GeneratedSQL(sql=self.sql)


class FakeResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> FakeResult:
        return self

    def all(self) -> list[dict[str, object]]:
        return self._rows


class FakeTransaction:
    def __enter__(self) -> FakeTransaction:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeReadOnlyConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def __enter__(self) -> FakeReadOnlyConnection:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def begin(self) -> FakeTransaction:
        return FakeTransaction()

    def execute(
        self,
        statement: object,
        parameters: dict[str, object] | None = None,
    ) -> FakeResult:
        sql = str(statement)
        self.calls.append((sql, parameters))
        normalized = sql.lstrip().upper()
        if normalized.startswith(("INSERT", "UPDATE", "DELETE", "DROP")):
            raise PermissionError("read-only transaction")
        if normalized.startswith("SET TRANSACTION") or "SET_CONFIG" in normalized:
            return FakeResult([])
        return FakeResult(self.rows)


class FakeReadOnlyEngine:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.connection = FakeReadOnlyConnection(rows)

    def connect(self) -> FakeReadOnlyConnection:
        return self.connection


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO financial_facts (metric) VALUES ('x')",
        "UPDATE financial_facts SET value = 0",
        "DELETE FROM financial_facts",
        "DROP TABLE financial_facts",
        "ALTER TABLE financial_facts ADD COLUMN unsafe TEXT",
        "CREATE TABLE unsafe_copy AS SELECT * FROM financial_facts",
        _VALID_SELECT + "; DELETE FROM financial_facts",
        _VALID_SELECT.replace("financial_facts", "doc_chunks"),
        _VALID_SELECT.replace("financial_facts", "query_logs"),
        _VALID_SELECT.replace("financial_facts", "information_schema.tables"),
        _VALID_SELECT.replace("ff.value", "ff.unknown_value"),
    ],
)
def test_validate_sql_rejects_unsafe_statements(sql: str) -> None:
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


def test_validate_sql_accepts_allowlisted_select_and_adds_limit() -> None:
    result = validate_sql(_VALID_SELECT)

    assert result.tables == ("financial_facts",)
    assert result.sql.endswith("LIMIT 100")


def test_validate_sql_rejects_dangerous_function_and_writable_cte() -> None:
    dangerous = _VALID_SELECT.replace("ff.value", "pg_sleep(1) AS value")
    writable_cte = """
    WITH removed AS (DELETE FROM financial_facts RETURNING metric)
    SELECT metric, 2024 AS fiscal_year, 'FY' AS fiscal_period,
           0 AS value, 'USD' AS unit
    FROM removed
    """

    with pytest.raises(SQLValidationError):
        validate_sql(dangerous)
    with pytest.raises(SQLValidationError):
        validate_sql(writable_cte)


def test_validate_sql_rejects_literal_numeric_answer() -> None:
    literal_answer = _VALID_SELECT.replace("ff.value", "42 AS value")

    with pytest.raises(SQLValidationError, match="derived from database columns"):
        validate_sql(literal_answer)


def test_sql_tool_performs_yoy_arithmetic_in_sql_and_returns_typed_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    understanding = _numeric_understanding()
    provider = FakeSQLProvider(_YOY_SQL)
    engine = FakeReadOnlyEngine(
        [
            _fact_row("net_income", 2023, Decimal("96995000000")),
            _fact_row("net_income", 2024, Decimal("93736000000")),
            _fact_row("net_income_yoy_delta", 2024, Decimal("-3259000000")),
            _fact_row(
                "net_income_yoy_percent_change",
                2024,
                Decimal("-3.3600"),
                unit="percent",
            ),
        ]
    )
    tool = SqlTool(
        provider=provider,  # type: ignore[arg-type]
        readonly_engine=engine,  # type: ignore[arg-type]
        model="configured-primary-model",
    )
    monkeypatch.setattr(sql_tool_module, "_get_default_sql_tool", lambda: tool)

    facts = sql_tool_module.sql_tool(understanding)

    executed_sql = engine.connection.calls[-1][0]
    assert "MAX(CASE WHEN ff.fiscal_year = 2024 THEN ff.value END) - MAX(" in executed_sql
    assert facts == [
        NumberFact(
            metric="net_income",
            fiscal_year=2023,
            fiscal_period="FY",
            value=96_995_000_000,
            unit="USD",
        ),
        NumberFact(
            metric="net_income",
            fiscal_year=2024,
            fiscal_period="FY",
            value=93_736_000_000,
            unit="USD",
        ),
        NumberFact(
            metric="net_income_yoy_delta",
            fiscal_year=2024,
            fiscal_period="FY",
            value=-3_259_000_000,
            unit="USD",
        ),
        NumberFact(
            metric="net_income_yoy_percent_change",
            fiscal_year=2024,
            fiscal_period="FY",
            value=-3.36,
            unit="percent",
        ),
    ]
    assert engine.connection.calls[0][0] == "SET TRANSACTION READ ONLY"
    assert engine.connection.calls[1][1] == {"statement_timeout": "5000ms"}
    assert provider.calls == [(understanding, "configured-primary-model")]


def test_sql_tool_rejects_comparison_without_sql_derived_facts() -> None:
    engine = FakeReadOnlyEngine(
        [
            _fact_row("net_income", 2023, Decimal("96995000000")),
            _fact_row("net_income", 2024, Decimal("93736000000")),
        ]
    )
    tool = SqlTool(
        provider=FakeSQLProvider(_VALID_SELECT),  # type: ignore[arg-type]
        readonly_engine=engine,  # type: ignore[arg-type]
        model="configured-primary-model",
    )

    with pytest.raises(RuntimeError, match="required comparison fact"):
        tool.run(_hybrid_understanding())


def test_sql_tool_rejects_write_before_opening_readonly_connection() -> None:
    engine = FakeReadOnlyEngine([])
    tool = SqlTool(
        provider=FakeSQLProvider("UPDATE financial_facts SET value = 0"),  # type: ignore[arg-type]
        readonly_engine=engine,  # type: ignore[arg-type]
        model="configured-primary-model",
    )

    with pytest.raises(SQLValidationError):
        tool.run(_numeric_understanding())

    assert engine.connection.calls == []


def test_sql_tool_factory_uses_only_database_url_readonly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readonly_url = "postgresql+psycopg://ffa_ro:pass@db:5432/ffa"
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        openai_model="configured-primary-model",
        database_url="postgresql+psycopg://ffa_app:pass@db:5432/ffa",
        database_url_readonly=readonly_url,
    )
    engine = FakeReadOnlyEngine([])
    engine_urls: list[str] = []

    def create_engine(url: str) -> FakeReadOnlyEngine:
        engine_urls.append(url)
        return engine

    monkeypatch.setattr(sql_tool_module, "create_readonly_engine", create_engine)
    monkeypatch.setattr(
        sql_tool_module.SQLGenerationProvider,
        "from_settings",
        lambda _: FakeSQLProvider(_VALID_SELECT),
    )

    tool = SqlTool.from_settings(settings)

    assert isinstance(tool, SqlTool)
    assert engine_urls == [readonly_url]


def test_sql_tool_factory_rejects_missing_or_shared_readonly_url() -> None:
    shared_url = "postgresql+psycopg://same@db:5432/ffa"
    shared_settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        openai_model="configured-primary-model",
        database_url=shared_url,
        database_url_readonly=shared_url,
    )
    missing_settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        openai_model="configured-primary-model",
        database_url_readonly="",
    )

    with pytest.raises(ValueError, match="must differ"):
        SqlTool.from_settings(shared_settings)
    with pytest.raises(ValueError, match="must be configured"):
        SqlTool.from_settings(missing_settings)


def test_openai_sql_provider_uses_structured_output_and_fixed_schema() -> None:
    calls: list[dict[str, Any]] = []

    class FakeResponses:
        def parse(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(output_parsed=GeneratedSQL(sql=_VALID_SELECT))

    provider = SQLGenerationProvider(  # type: ignore[arg-type]
        SimpleNamespace(responses=FakeResponses())
    )

    understanding = _hybrid_understanding()
    result = provider.generate_sql(understanding, model="configured-primary-model")

    assert result.sql == _VALID_SELECT.strip()
    assert calls[0]["model"] == "configured-primary-model"
    assert calls[0]["text_format"] is GeneratedSQL
    assert "financial_facts(" in calls[0]["instructions"]
    assert "doc_chunks" not in calls[0]["instructions"]
    assert "Perform every calculation in SQL" in calls[0]["instructions"]
    assert "Returning only the two" in calls[0]["instructions"]
    assert "UNION branch inside a derived table" in calls[0]["instructions"]
    payload = json.loads(calls[0]["input"])
    assert payload["understanding"] == understanding.model_dump(mode="json")
    assert payload["required_comparisons"] == [
        {
            "source_metric": "net_income",
            "dimension": "fiscal_year",
            "from_value": 2023,
            "to_value": 2024,
            "delta_metric": "net_income_yoy_delta",
            "percent_change_metric": "net_income_yoy_percent_change",
        }
    ]


def _numeric_understanding() -> Understanding:
    return Understanding(
        intent=Intent.NUMERIC,
        entities=Entities(
            tickers=["AAPL"],
            ciks=[320193],
            metrics=["net_income"],
            fiscal_years=[2023, 2024],
            fiscal_periods=["FY"],
        ),
        rewritten_query="Compare Apple net income in fiscal 2023 and fiscal 2024",
    )


def _hybrid_understanding() -> Understanding:
    return Understanding(
        intent=Intent.HYBRID,
        entities=Entities(
            tickers=["AAPL"],
            ciks=[320193],
            metrics=["net_income"],
            fiscal_years=[2023, 2024],
            sections=["MD&A"],
        ),
        rewritten_query=(
            "How did Apple's net income change from fiscal 2023 to fiscal 2024, "
            "and what does management say about it?"
        ),
    )


def _fact_row(
    metric: str,
    fiscal_year: int,
    value: Decimal,
    *,
    unit: str = "USD",
) -> dict[str, object]:
    return {
        "metric": metric,
        "fiscal_year": fiscal_year,
        "fiscal_period": "FY",
        "value": value,
        "unit": unit,
    }
