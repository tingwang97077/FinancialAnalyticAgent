"""Input safety checks for moderation and prompt injection."""

from __future__ import annotations

import base64
import binascii
import html
import logging
import re
import unicodedata
from typing import Protocol
from urllib.parse import unquote

import sqlglot
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import Scope, traverse_scope

from ffa.config import Settings, get_settings

logger = logging.getLogger(__name__)

_MODERATION_MODEL = "omni-moderation-latest"
_MAX_QUESTION_CHARACTERS = 10_000
_ACCEPTED_REASON = "Input accepted."
_BLOCKED_REASON = "I cannot process this request."
_MAX_SQL_ROWS = 100
_REQUIRED_RESULT_COLUMNS = frozenset({"metric", "fiscal_year", "fiscal_period", "value", "unit"})
_TABLE_COLUMNS = {
    "financial_facts": frozenset(
        {
            "id",
            "cik",
            "ticker",
            "metric",
            "taxonomy_tag",
            "unit",
            "fiscal_year",
            "fiscal_period",
            "period_start",
            "period_end",
            "value",
            "form_type",
            "filing_date",
            "accession_no",
            "source_url",
        }
    ),
    "companies": frozenset({"cik", "ticker", "name", "sic", "updated_at"}),
    "filings": frozenset(
        {
            "accession_no",
            "cik",
            "form_type",
            "filing_date",
            "period_of_report",
            "primary_doc_url",
        }
    ),
}
_ALLOWED_SQL_FUNCTIONS = frozenset(
    {
        "ABS",
        "AND",
        "AVG",
        "CASE",
        "CAST",
        "CEIL",
        "COALESCE",
        "COUNT",
        "DATE_TRUNC",
        "EXTRACT",
        "FLOOR",
        "GREATEST",
        "IF",
        "LAG",
        "LEAD",
        "LEAST",
        "LOWER",
        "MAX",
        "MIN",
        "NULLIF",
        "NOT",
        "OR",
        "POWER",
        "ROUND",
        "ROW_NUMBER",
        "SUM",
        "UPPER",
    }
)
_FORBIDDEN_SQL_NODES = (
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Create,
    exp.Delete,
    exp.Drop,
    exp.Grant,
    exp.Insert,
    exp.Into,
    exp.Lock,
    exp.Merge,
    exp.Pragma,
    exp.Set,
    exp.Transaction,
    exp.TruncateTable,
    exp.Update,
    exp.Use,
)
_INJECTION_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(?:ignore|disregard|forget|override|bypass|discard)\b.{0,160}"
        r"\b(?:previous|prior|above|earlier|system|developer|safety|hidden|initial|your)\b"
        r".{0,40}\b(?:instructions?|rules?|messages?|prompt)\b",
        r"\b(?:ignore|disregard|forget|override|bypass|discard)\b.{0,160}"
        r"\b(?:instructions?|rules?|messages?|prompt)\b.{0,40}"
        r"\b(?:previous|prior|above|earlier|system|developer|safety|hidden|initial)\b",
        r"\b(?:follow|obey|execute|adopt)\b.{0,80}\b(?:new|these|my)\b.{0,40}"
        r"\b(?:instructions?|rules?|prompt)\b.{0,40}\b(?:instead|now)\b",
        r"\b(?:reveal|show|print|quote|repeat|expose|leak)\b.{0,100}"
        r"\b(?:system|developer|hidden|initial)\b.{0,50}\b(?:prompt|instructions?|message)\b",
        r"\b(?:act as|pretend to be|you are now)\b.{0,100}"
        r"\b(?:system|developer|unrestricted|jailbroken)\b",
        r"(?:^|\n)\s*(?:system|developer)\s*:",
        r"\b(?:reveal|show|print|return|expose|leak)\b.{0,50}"
        r"\b(?:your|hidden|internal|system|developer)\b.{0,40}"
        r"\b(?:api[- ]?keys?|secrets?|credentials?)\b",
        r"\b(?:ignore(?:r|z)?|oublie(?:r|z)?|contourne(?:r|z)?|outrepasse(?:r|z)?)\b"
        r".{0,160}\b(?:toutes?\s+les\s+)?(?:instructions?|r[eè]gles?|directives?)\b"
        r".{0,50}\b(?:pr[eé]c[eé]dentes?|ant[eé]rieures?|ci-dessus|syst[eè]me|d[eé]veloppeur)\b",
        r"\bne\s+(?:tiens|tenez)\s+pas\s+compte\b.{0,120}"
        r"\b(?:instructions?|r[eè]gles?|directives?|prompt)\b",
        r"\b(?:r[eé]v[eè]le(?:r|z)?|affiche(?:r|z)?|divulgue(?:r|z)?|expose(?:r|z)?)\b"
        r".{0,120}\b(?:prompt|instructions?|message)\b.{0,60}"
        r"\b(?:syst[eè]me|d[eé]veloppeur|cach[eé])\b",
    )
)
_BASE64_TOKEN_PATTERN = re.compile(r"(?<![\w+/=-])[A-Za-z0-9+/_-]{16,}={0,2}(?![\w+/=-])")
_LEETSPEAK_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)


class GuardResult(BaseModel):
    """Safe result consumed by downstream request orchestration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    reason: str


class ValidatedSQL(BaseModel):
    """A normalized, bounded query that passed every SQL safety check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sql: str = Field(min_length=1)
    tables: tuple[str, ...]


class SQLValidationError(ValueError):
    """Raised when generated SQL violates the read-only query policy."""


class ModerationProvider(Protocol):
    """Backend contract for content moderation."""

    def is_flagged(self, text: str) -> bool:
        """Return whether the moderation service flags the text."""
        ...


class OpenAIModerationProvider:
    """OpenAI moderation endpoint adapter."""

    def __init__(self, client: OpenAI) -> None:
        """Initialize the adapter with an OpenAI SDK client."""
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> OpenAIModerationProvider:
        """Build the adapter from central application settings."""
        resolved_settings = settings or get_settings()
        if resolved_settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY must be configured before moderating input.")
        api_key = resolved_settings.openai_api_key.get_secret_value().strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be configured before moderating input.")
        return cls(OpenAI(api_key=api_key))

    def is_flagged(self, text: str) -> bool:
        """Submit text to the OpenAI moderation endpoint."""
        response = self._client.moderations.create(model=_MODERATION_MODEL, input=text)
        if not response.results:
            raise RuntimeError("OpenAI moderation returned no result.")
        return bool(response.results[0].flagged)


def check_input(
    question: str,
    *,
    moderation_provider: ModerationProvider | None = None,
    settings: Settings | None = None,
) -> GuardResult:
    """Allow safe questions and return a generic refusal for blocked input.

    Heuristic prompt-injection checks run before the remote moderation request so
    obviously adversarial input cannot trigger any further external processing.
    Moderation failures fail closed and are logged without exposing implementation
    details to the caller.
    """
    if not isinstance(question, str):
        return _blocked_result()
    normalized_question = unicodedata.normalize("NFKC", question).strip()
    if not normalized_question or len(normalized_question) > _MAX_QUESTION_CHARACTERS:
        return _blocked_result()
    if _looks_like_prompt_injection(normalized_question):
        logger.warning("Input blocked by safety policy.")
        return _blocked_result()

    try:
        provider = moderation_provider or OpenAIModerationProvider.from_settings(settings)
        if provider.is_flagged(normalized_question):
            logger.warning("Input blocked by safety policy.")
            return _blocked_result()
    except Exception:
        logger.exception("Input moderation failed; request blocked.")
        return _blocked_result()

    return GuardResult(allowed=True, reason=_ACCEPTED_REASON)


def validate_sql(sql: str) -> ValidatedSQL:
    """Validate and bound one read-only PostgreSQL SELECT with ``sqlglot``.

    The validator uses an allow-list for statements, physical tables, columns, and
    SQL functions. It never includes the rejected SQL text in public exceptions.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise SQLValidationError("SQL must be a non-empty SELECT statement.")
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except (ParseError, ValueError) as exc:
        raise SQLValidationError("SQL could not be parsed as PostgreSQL.") from exc
    if len(statements) != 1 or statements[0] is None:
        raise SQLValidationError("Exactly one SQL statement is required.")

    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise SQLValidationError("Only a SELECT query is permitted.")
    if any(isinstance(node, _FORBIDDEN_SQL_NODES) for node in statement.walk()):
        raise SQLValidationError("The SELECT contains a prohibited SQL operation.")
    if any(True for _ in statement.find_all(exp.Star)):
        raise SQLValidationError("Wildcard column selection is not permitted.")

    _validate_sql_functions(statement)
    tables = _validate_sql_sources_and_columns(statement)
    _validate_result_contract(statement)
    _validate_value_derivation(statement)
    bounded_statement = _apply_row_limit(statement)
    return ValidatedSQL(
        sql=bounded_statement.sql(dialect="postgres"),
        tables=tuple(sorted(tables)),
    )


def _looks_like_prompt_injection(question: str) -> bool:
    """Detect high-confidence instruction-hierarchy manipulation attempts."""
    return any(
        pattern.search(candidate) is not None
        for candidate in _text_variants(question)
        for pattern in _INJECTION_PATTERNS
    )


def _text_variants(text: str) -> tuple[str, ...]:
    """Expose plain, simply encoded, and obfuscated text to injection checks."""
    normalized = _strip_format_characters(unicodedata.normalize("NFKC", text))
    variants = {normalized, unquote(normalized), html.unescape(normalized)}
    for candidate in tuple(variants):
        variants.add(candidate.translate(_LEETSPEAK_TRANSLATION))
        variants.update(_decoded_base64_fragments(candidate))
    for candidate in tuple(variants):
        variants.add(candidate.translate(_LEETSPEAK_TRANSLATION))
    return tuple(variants)


def _decoded_base64_fragments(text: str) -> set[str]:
    """Decode printable UTF-8 Base64 fragments without accepting arbitrary binary data."""
    decoded_fragments: set[str] = set()
    for match in _BASE64_TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        padded = token + "=" * (-len(token) % 4)
        try:
            decoded = base64.b64decode(padded, altchars=b"-_", validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if len(decoded) > 4_000:
            continue
        if decoded and all(character.isprintable() or character.isspace() for character in decoded):
            decoded_fragments.add(_strip_format_characters(decoded))
    return decoded_fragments


def _strip_format_characters(text: str) -> str:
    """Remove invisible Unicode format characters used to split detection terms."""
    return "".join(character for character in text if unicodedata.category(character) != "Cf")


def _blocked_result() -> GuardResult:
    """Return the same non-diagnostic refusal for every blocking condition."""
    return GuardResult(allowed=False, reason=_BLOCKED_REASON)


def _validate_sql_functions(statement: exp.Select) -> None:
    """Allow only deterministic functions needed for financial calculations."""
    for function in statement.find_all(exp.Func):
        name = function.name if isinstance(function, exp.Anonymous) else function.sql_name()
        if name.upper() not in _ALLOWED_SQL_FUNCTIONS:
            raise SQLValidationError("The SELECT uses a function outside the allow-list.")


def _validate_sql_sources_and_columns(statement: exp.Select) -> set[str]:
    """Validate physical sources and resolve columns inside every query scope."""
    referenced_tables: set[str] = set()
    for scope in traverse_scope(statement):
        source_columns: dict[str, frozenset[str]] = {}
        for alias, source in scope.sources.items():
            normalized_alias = alias.lower()
            if isinstance(source, exp.Table):
                table_name = source.name.lower()
                schema_name = source.db.lower()
                if source.catalog or schema_name not in {"", "public"}:
                    raise SQLValidationError("Only allow-listed public tables may be queried.")
                if table_name not in _TABLE_COLUMNS:
                    raise SQLValidationError(
                        "The SELECT references a table outside the allow-list."
                    )
                referenced_tables.add(table_name)
                source_columns[normalized_alias] = _TABLE_COLUMNS[table_name]
            elif isinstance(source, Scope):
                output_columns = frozenset(
                    name.lower() for name in source.expression.named_selects if name and name != "*"
                )
                source_columns[normalized_alias] = output_columns
            else:
                raise SQLValidationError("The SELECT contains an unsupported data source.")

        available_columns = frozenset().union(*source_columns.values())
        projection_aliases = frozenset(
            name.lower() for name in scope.expression.named_selects if name
        )
        for column in scope.columns:
            column_name = column.name.lower()
            qualifier = column.table.lower()
            if qualifier:
                allowed_columns = source_columns.get(qualifier)
                if allowed_columns is None or column_name not in allowed_columns:
                    raise SQLValidationError(
                        "The SELECT references a column outside the allow-list."
                    )
            elif column_name not in available_columns | projection_aliases:
                raise SQLValidationError("The SELECT references a column outside the allow-list.")

        for join in scope.expression.find_all(exp.Join):
            if join.args.get("using"):
                raise SQLValidationError("JOIN columns must be explicitly qualified with ON.")

    if not referenced_tables:
        raise SQLValidationError("The SELECT must read from an allow-listed table.")
    return referenced_tables


def _validate_result_contract(statement: exp.Select) -> None:
    """Require exactly the columns needed to build ``NumberFact`` objects."""
    selected_columns = [name.lower() for name in statement.named_selects]
    if (
        len(selected_columns) != len(_REQUIRED_RESULT_COLUMNS)
        or set(selected_columns) != _REQUIRED_RESULT_COLUMNS
    ):
        raise SQLValidationError(
            "The SELECT must return exactly metric, fiscal_year, fiscal_period, value, and unit."
        )


def _validate_value_derivation(statement: exp.Select) -> None:
    """Reject numeric answers supplied as literals instead of database expressions."""
    for scope in traverse_scope(statement):
        if not isinstance(scope.expression, exp.Select):
            continue
        for projection in scope.expression.selects:
            if projection.alias_or_name.lower() != "value":
                continue
            if not any(True for _ in projection.find_all(exp.Column)):
                raise SQLValidationError(
                    "The value result must be derived from database columns in SQL."
                )


def _apply_row_limit(statement: exp.Select) -> exp.Select:
    """Add or clamp a literal top-level row limit."""
    limit = statement.args.get("limit")
    if limit is None:
        return statement.limit(_MAX_SQL_ROWS, copy=True)
    limit_expression = limit.expression
    if not isinstance(limit_expression, exp.Literal) or not limit_expression.is_int:
        raise SQLValidationError("LIMIT must be a positive integer literal.")
    row_count = int(limit_expression.this)
    if row_count <= 0:
        raise SQLValidationError("LIMIT must be a positive integer literal.")
    if row_count > _MAX_SQL_ROWS:
        return statement.limit(_MAX_SQL_ROWS, copy=True)
    return statement
