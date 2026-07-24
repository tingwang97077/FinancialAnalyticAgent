"""Evaluate numeric correctness and narrative generation on frozen gold sets."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import random
import statistics
import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from langchain_core.outputs import Generation, LLMResult
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, text

from ffa.agent.generation import AnswerGenerator
from ffa.agent.router import AgentContext, AgentRun, run_agent
from ffa.agent.schemas import Answer, NumberFact
from ffa.common.db import create_rw_engine
from ffa.config import Settings, get_settings
from ffa.evaluation.build_groundtruth import (
    RetrievalGroundTruth,
    SourceChunk,
    load_groundtruth,
    load_source_chunks,
)
from ffa.monitoring.metrics import PriceBook, QueryMetrics, extract_token_usage
from ffa.monitoring.tracing import RequestTracer, record_openai_response
from ffa.retrieval.base import Chunk

logger = logging.getLogger(__name__)

DEFAULT_NUMERIC_GOLD_PATH = Path("evaluation/generation_numeric_gold.json")
DEFAULT_NARRATIVE_SET_PATH = Path("evaluation/generation_narrative_set.json")
DEFAULT_RETRIEVAL_GROUNDTRUTH_PATH = Path("evaluation/retrieval_groundtruth.json")
DEFAULT_SEED = 42
DIRECT_CASE_COUNT = 34
COMPARISON_CASE_COUNT = 8
NARRATIVE_CASE_COUNT = 30
PROMPT_ORDER = ("production", "citation_first")
_NARRATIVE_SECTIONS = ("MD&A", "Risk Factors", "Notes")
_METRIC_LABELS = {
    "revenue": "revenue",
    "net_income": "net income",
    "total_assets": "total assets",
    "total_liabilities": "total liabilities",
    "cash_and_equivalents": "cash and cash equivalents",
}
_CITATION_FIRST_INSTRUCTIONS = """You write concise, citation-first financial answers.

Rules:
- Use only the supplied typed facts and filing chunks.
- State only claims directly supported by the supplied evidence.
- Keep the response focused on the question and omit background that is not necessary.
- Every narrative claim must have a Citation copied exactly from its supporting chunk.
- Numeric claims must copy supplied NumberFact fields exactly. Never calculate or transform
  a number.
- If evidence is missing or conflicting, state the limitation directly.
- Never invent a number, source URL, accession number, section, or fact.
- Return only the structured Answer required by the response schema.
"""

type FactKey = tuple[str, int, str, Decimal, str]


class _EvalModel(BaseModel):
    """Strict immutable base model for persisted evaluation data."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class GoldFact(_EvalModel):
    """Exact expected or allowed NumberFact represented without JSON float loss."""

    metric: str = Field(min_length=1)
    fiscal_year: int = Field(gt=0)
    fiscal_period: str = Field(min_length=1)
    value: str = Field(min_length=1)
    unit: str = Field(min_length=1)

    @classmethod
    def from_values(
        cls,
        *,
        metric: str,
        fiscal_year: int,
        fiscal_period: str,
        value: Decimal | float | int | str,
        unit: str,
    ) -> GoldFact:
        """Create a stable fact from database or derived numeric values."""
        return cls(
            metric=metric,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            value=str(value),
            unit=unit,
        )


class NumericGoldCase(_EvalModel):
    """One answerable numeric question or deliberate no-answer trap."""

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    required_facts: list[GoldFact] = Field(default_factory=list)
    allowed_facts: list[GoldFact] = Field(default_factory=list)
    expect_refusal: bool = False
    case_type: str = Field(min_length=1)


class NumericGoldSet(_EvalModel):
    """Frozen numeric benchmark derived from canonical financial facts."""

    version: int = 1
    seed: int
    generated_at: datetime
    source: str = "financial_facts"
    cases: list[NumericGoldCase]


class NarrativeCase(_EvalModel):
    """Stable narrative question linked to one known source chunk."""

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_accession_no: str = Field(min_length=1)
    expected_section: str = Field(min_length=1)
    expected_chunk_index: int = Field(ge=0)
    ticker: str = Field(min_length=1)

    @property
    def identity(self) -> tuple[str, str, int]:
        """Return the chunk identity stable across database reloads."""
        return (
            self.expected_accession_no,
            self.expected_section,
            self.expected_chunk_index,
        )


class NarrativeSet(_EvalModel):
    """Frozen narrative benchmark sampled from retrieval ground truth."""

    version: int = 1
    seed: int
    generated_at: datetime
    retrieval_corpus_fingerprint: str
    cases: list[NarrativeCase]


class NumericCaseOutcome(_EvalModel):
    """Observed numeric result retained for deterministic scoring."""

    case_id: str
    answer: Answer | None = None
    error: str | None = None


class NumericMetrics(_EvalModel):
    """Aggregate exactness, refusal, grounding, and false-number metrics."""

    question_count: int
    answerable_count: int
    trap_count: int
    exact_match_rate: float
    grounded_rate: float
    empty_or_refusal_rate: float
    false_number_rate: float
    correct_refusal_rate: float
    execution_error_rate: float
    false_number_case_count: int
    false_number_count: int
    failed_case_ids: list[str]
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: Decimal = Decimal("0")


class NarrativeCandidate(_EvalModel):
    """One answer and its shared retrieval evidence for RAGAS."""

    case_id: str
    question: str
    reference: str
    contexts: list[str]
    answer: Answer


class NarrativePromptMetrics(_EvalModel):
    """RAGAS and deterministic grounding metrics for one prompt."""

    prompt_name: str
    question_count: int
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    grounded_rate: float
    citation_rate: float
    execution_error_count: int
    execution_error_rate: float
    composite_score: float
    ragas_nan_count: int
    duration_seconds: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: Decimal
    eval_run_id: int | None = None


class GenerationEvaluationSummary(_EvalModel):
    """Complete numeric and narrative generation evaluation report."""

    seed: int
    numeric_gold_path: str
    narrative_set_path: str
    retrieval_strategy: str
    generation_model: str
    judge_model: str
    numeric_metrics: NumericMetrics
    prompt_results: list[NarrativePromptMetrics]
    best_prompt: str
    production_prompt: str = "production"
    production_confirmed: bool
    numeric_eval_run_id: int | None = None
    total_duration_seconds: float
    total_cost_usd: Decimal


@dataclass(frozen=True, slots=True)
class DatabaseFact:
    """Canonical database fact used while constructing numeric cases."""

    ticker: str
    metric: str
    fiscal_year: int
    fiscal_period: str
    value: Decimal
    unit: str

    def gold(self, *, metric: str | None = None) -> GoldFact:
        """Return the persisted representation of this fact."""
        return GoldFact.from_values(
            metric=metric or self.metric,
            fiscal_year=self.fiscal_year,
            fiscal_period=self.fiscal_period,
            value=self.value,
            unit=self.unit,
        )


class AgentRunner(Protocol):
    """Callable real or fake agent contract used by numeric evaluation."""

    def __call__(self, question: str, *, trace_id: str | None = None) -> AgentRun:
        """Return an answer and the exact evidence context."""
        ...


class UsageAccumulator:
    """Thread-safe usage and environment-priced cost accumulator."""

    def __init__(self, settings: Settings) -> None:
        """Initialize counters with the configured model price book."""
        self.metrics = QueryMetrics()
        self._price_book = PriceBook.from_settings(settings)
        self._lock = Lock()

    def add_response(
        self,
        response: object,
        *,
        model: str,
        kind: str = "model",
    ) -> None:
        """Add one OpenAI response usage without exposing request content."""
        usage = extract_token_usage(response)
        cost = self._price_book.cost(
            model=model,
            usage=usage,
            kind="embedding" if kind == "embedding" else "model",
        )
        with self._lock:
            self.metrics.add(usage, cost)


class EvaluationGenerationProvider:
    """Evaluation-only structured generation provider with an alternate prompt."""

    def __init__(self, *, client: OpenAI, instructions: str) -> None:
        """Initialize one prompt candidate without changing production code."""
        self._client = client
        self._instructions = instructions

    def generate_answer(
        self,
        question: str,
        context: AgentContext,
        *,
        model: str,
    ) -> Answer:
        """Generate a structured candidate from the shared agent context."""
        response = self._client.responses.parse(
            model=model,
            instructions=self._instructions,
            input=_generation_input(question, context),
            text_format=Answer,
            metadata={"trace_id": context.trace_id, "evaluation_prompt": "citation_first"},
        )
        record_openai_response(response, model=model)
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("The evaluation prompt did not return a structured answer.")
        return parsed


def build_generation_sets(
    engine: Engine,
    *,
    numeric_path: Path = DEFAULT_NUMERIC_GOLD_PATH,
    narrative_path: Path = DEFAULT_NARRATIVE_SET_PATH,
    retrieval_groundtruth_path: Path = DEFAULT_RETRIEVAL_GROUNDTRUTH_PATH,
    seed: int = DEFAULT_SEED,
) -> tuple[NumericGoldSet, NarrativeSet]:
    """Build and persist both frozen generation evaluation sets."""
    facts = load_database_facts(engine)
    numeric_set = build_numeric_gold_set(facts, seed=seed)
    retrieval_set = load_groundtruth(retrieval_groundtruth_path)
    narrative_set = build_narrative_set(retrieval_set, seed=seed)
    _write_model(numeric_path, numeric_set)
    _write_model(narrative_path, narrative_set)
    return numeric_set, narrative_set


def load_database_facts(engine: Engine) -> list[DatabaseFact]:
    """Load one latest canonical fact per ticker, metric, period, year, and unit."""
    statement = text(
        """
        SELECT DISTINCT ON (
            ticker, metric, fiscal_year, fiscal_period, unit
        )
            ticker,
            metric,
            fiscal_year,
            fiscal_period,
            value,
            unit
        FROM financial_facts
        WHERE fiscal_period IN ('FY', 'Q1', 'Q2', 'Q3', 'Q4')
          AND metric IN (
              'revenue',
              'net_income',
              'total_assets',
              'total_liabilities',
              'cash_and_equivalents'
          )
        ORDER BY
            ticker,
            metric,
            fiscal_year,
            fiscal_period,
            unit,
            filing_date DESC NULLS LAST,
            id DESC
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()
    return [
        DatabaseFact(
            ticker=str(row["ticker"]),
            metric=str(row["metric"]),
            fiscal_year=int(row["fiscal_year"]),
            fiscal_period=str(row["fiscal_period"]),
            value=Decimal(str(row["value"])),
            unit=str(row["unit"]),
        )
        for row in rows
    ]


def build_numeric_gold_set(
    facts: Sequence[DatabaseFact],
    *,
    seed: int = DEFAULT_SEED,
    direct_count: int = DIRECT_CASE_COUNT,
    comparison_count: int = COMPARISON_CASE_COUNT,
) -> NumericGoldSet:
    """Select balanced direct and comparison cases plus fixed no-answer traps."""
    if not facts:
        raise ValueError("Cannot build a numeric gold set without facts.")
    direct_facts = select_direct_facts(facts, count=direct_count, seed=seed)
    comparison_pairs = select_comparison_pairs(
        facts,
        count=comparison_count,
        seed=seed,
    )
    cases: list[NumericGoldCase] = []
    for index, fact in enumerate(direct_facts, start=1):
        label = _METRIC_LABELS[fact.metric]
        expected = fact.gold()
        cases.append(
            NumericGoldCase(
                case_id=f"direct-{index:03d}",
                question=_direct_question(fact, label=label),
                required_facts=[expected],
                allowed_facts=[expected],
                case_type="direct",
            )
        )
    for index, (from_fact, to_fact) in enumerate(comparison_pairs, start=1):
        cases.append(_comparison_case(index, from_fact, to_fact))
    cases.extend(_trap_cases())
    return NumericGoldSet(
        seed=seed,
        generated_at=datetime.now(UTC),
        cases=cases,
    )


def select_direct_facts(
    facts: Sequence[DatabaseFact],
    *,
    count: int,
    seed: int,
) -> list[DatabaseFact]:
    """Select latest facts in metric-and-period-balanced seeded order."""
    latest: dict[tuple[str, str, str], DatabaseFact] = {}
    for fact in facts:
        key = (fact.ticker, fact.metric, fact.fiscal_period)
        current = latest.get(key)
        if current is None or fact.fiscal_year > current.fiscal_year:
            latest[key] = fact
    periods = ("FY", "Q1", "Q2", "Q3", "Q4")
    grouped: dict[tuple[str, str], list[DatabaseFact]] = {
        (metric, period): [] for metric in _METRIC_LABELS for period in periods
    }
    for fact in latest.values():
        grouped[(fact.metric, fact.fiscal_period)].append(fact)
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)

    selected: list[DatabaseFact] = []
    strata = list(grouped)
    while len(selected) < count:
        made_progress = False
        for stratum in strata:
            values = grouped[stratum]
            if values and len(selected) < count:
                selected.append(values.pop())
                made_progress = True
        if not made_progress:
            break
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} direct numeric cases are available.")
    return selected


def select_comparison_pairs(
    facts: Sequence[DatabaseFact],
    *,
    count: int,
    seed: int,
) -> list[tuple[DatabaseFact, DatabaseFact]]:
    """Select seeded adjacent-year comparisons across distinct ticker/metric pairs."""
    by_series: dict[tuple[str, str, str], list[DatabaseFact]] = {}
    for fact in facts:
        if fact.fiscal_period != "FY":
            continue
        by_series.setdefault((fact.ticker, fact.metric, fact.unit), []).append(fact)
    candidates: list[tuple[DatabaseFact, DatabaseFact]] = []
    for series in by_series.values():
        ordered = sorted(series, key=lambda fact: fact.fiscal_year)
        by_year = {fact.fiscal_year: fact for fact in ordered}
        for from_year, from_fact in by_year.items():
            to_fact = by_year.get(from_year + 1)
            if to_fact is not None and from_year >= 2021 and from_fact.value != 0:
                candidates.append((from_fact, to_fact))
    rng = random.Random(seed + 1)
    rng.shuffle(candidates)

    selected: list[tuple[DatabaseFact, DatabaseFact]] = []
    seen_tickers: set[str] = set()
    for pair in candidates:
        if pair[0].ticker in seen_tickers:
            continue
        selected.append(pair)
        seen_tickers.add(pair[0].ticker)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} numeric comparison cases are available.")
    return selected


def build_narrative_set(
    retrieval_set: RetrievalGroundTruth,
    *,
    seed: int = DEFAULT_SEED,
    count: int = NARRATIVE_CASE_COUNT,
) -> NarrativeSet:
    """Select an equal number of stable questions from each canonical section."""
    if count % len(_NARRATIVE_SECTIONS) != 0:
        raise ValueError("Narrative case count must be divisible by the section count.")
    per_section = count // len(_NARRATIVE_SECTIONS)
    rng = random.Random(seed)
    selected = []
    for section in _NARRATIVE_SECTIONS:
        candidates = [pair for pair in retrieval_set.pairs if pair.expected_section == section]
        rng.shuffle(candidates)
        if len(candidates) < per_section:
            raise ValueError(f"Not enough narrative cases for section {section}.")
        selected.extend(candidates[:per_section])
    rng.shuffle(selected)
    return NarrativeSet(
        seed=seed,
        generated_at=datetime.now(UTC),
        retrieval_corpus_fingerprint=retrieval_set.corpus_fingerprint,
        cases=[
            NarrativeCase(
                case_id=f"narrative-{index:03d}",
                question=pair.question,
                expected_accession_no=pair.expected_accession_no,
                expected_section=pair.expected_section,
                expected_chunk_index=pair.expected_chunk_index,
                ticker=pair.ticker,
            )
            for index, pair in enumerate(selected, start=1)
        ],
    )


def evaluate_generation(
    *,
    numeric_path: Path = DEFAULT_NUMERIC_GOLD_PATH,
    narrative_path: Path = DEFAULT_NARRATIVE_SET_PATH,
    engine: Engine | None = None,
    settings: Settings | None = None,
    agent_runner: AgentRunner = run_agent,
    persist: bool = True,
) -> GenerationEvaluationSummary:
    """Run numeric exactness and two-prompt narrative evaluation."""
    started_at = perf_counter()
    resolved_settings = settings or get_settings()
    owns_engine = engine is None
    resolved_engine = engine or create_rw_engine(resolved_settings.database_url)
    try:
        numeric_set = _load_model(numeric_path, NumericGoldSet)
        narrative_set = _load_model(narrative_path, NarrativeSet)

        numeric_metrics = run_numeric_evaluation(
            numeric_set,
            settings=resolved_settings,
            agent_runner=agent_runner,
        )
        candidates, narrative_generation_metrics = generate_narrative_candidates(
            narrative_set,
            resolved_engine,
            settings=resolved_settings,
            agent_runner=agent_runner,
        )

        prompt_results: list[NarrativePromptMetrics] = []
        for prompt_name in PROMPT_ORDER:
            metrics = run_ragas_evaluation(
                candidates[prompt_name],
                prompt_name=prompt_name,
                settings=resolved_settings,
            )
            generation_metrics = narrative_generation_metrics[prompt_name]
            prompt_results.append(
                metrics.model_copy(
                    update={
                        "duration_seconds": round(
                            metrics.duration_seconds
                            + float(generation_metrics["duration_seconds"]),
                            3,
                        ),
                        "input_tokens": (
                            metrics.input_tokens + int(generation_metrics["input_tokens"])
                        ),
                        "output_tokens": (
                            metrics.output_tokens + int(generation_metrics["output_tokens"])
                        ),
                        "cached_tokens": (
                            metrics.cached_tokens + int(generation_metrics["cached_tokens"])
                        ),
                        "cost_usd": (
                            metrics.cost_usd + Decimal(str(generation_metrics["cost_usd"]))
                        ),
                        "execution_error_count": int(generation_metrics["execution_error_count"]),
                        "execution_error_rate": _rate(
                            int(generation_metrics["execution_error_count"]),
                            metrics.question_count,
                        ),
                    }
                )
            )

        best_prompt = select_best_prompt(prompt_results)
        numeric_run_id: int | None = None
        if persist:
            run_ids = persist_generation_runs(
                resolved_engine,
                numeric_path=numeric_path,
                narrative_path=narrative_path,
                settings=resolved_settings,
                numeric_metrics=numeric_metrics,
                prompt_results=prompt_results,
            )
            numeric_run_id = run_ids["numeric_gold"]
            prompt_results = [
                result.model_copy(update={"eval_run_id": run_ids[result.prompt_name]})
                for result in prompt_results
            ]
        total_cost = numeric_metrics.cost_usd + sum(
            (result.cost_usd for result in prompt_results),
            start=Decimal("0"),
        )
        return GenerationEvaluationSummary(
            seed=numeric_set.seed,
            numeric_gold_path=str(numeric_path),
            narrative_set_path=str(narrative_path),
            retrieval_strategy=resolved_settings.retrieval_strategy,
            generation_model=resolved_settings.openai_model,
            judge_model=_judge_model(resolved_settings),
            numeric_metrics=numeric_metrics,
            prompt_results=prompt_results,
            best_prompt=best_prompt,
            production_confirmed=best_prompt == "production",
            numeric_eval_run_id=numeric_run_id,
            total_duration_seconds=round(perf_counter() - started_at, 3),
            total_cost_usd=total_cost,
        )
    finally:
        if owns_engine:
            resolved_engine.dispose()


def run_numeric_evaluation(
    gold_set: NumericGoldSet,
    *,
    settings: Settings,
    agent_runner: AgentRunner = run_agent,
) -> NumericMetrics:
    """Execute the real agent for every numeric case and score exact facts."""
    started_at = perf_counter()
    outcomes: list[NumericCaseOutcome] = []
    tracer = RequestTracer(settings=settings)
    with tracer.trace(
        trace_id=uuid4().hex,
        question="evaluate numeric generation gold set",
        session_id=None,
    ) as evaluation_trace:
        for case in gold_set.cases:
            try:
                run = agent_runner(
                    case.question,
                    trace_id=f"eval-generation-{case.case_id}",
                )
            except Exception as exc:
                outcomes.append(
                    NumericCaseOutcome(
                        case_id=case.case_id,
                        error=type(exc).__name__,
                    )
                )
            else:
                outcomes.append(
                    NumericCaseOutcome(
                        case_id=case.case_id,
                        answer=run.answer,
                    )
                )
    metrics = score_numeric_outcomes(gold_set.cases, outcomes)
    return metrics.model_copy(
        update={
            "duration_seconds": round(perf_counter() - started_at, 3),
            "input_tokens": evaluation_trace.metrics.input_tokens,
            "output_tokens": evaluation_trace.metrics.output_tokens,
            "cached_tokens": evaluation_trace.metrics.cached_tokens,
            "cost_usd": evaluation_trace.metrics.cost_usd,
        }
    )


def score_numeric_outcomes(
    cases: Sequence[NumericGoldCase],
    outcomes: Sequence[NumericCaseOutcome],
) -> NumericMetrics:
    """Compute strict numeric correctness with false-number errors separated."""
    if not cases:
        raise ValueError("At least one numeric gold case is required.")
    outcome_by_id = {outcome.case_id: outcome for outcome in outcomes}
    exact_matches = 0
    grounded = 0
    empty_or_refusal = 0
    correct_refusals = 0
    execution_errors = 0
    false_number_cases = 0
    false_number_count = 0
    responses_with_numbers = 0
    failed_case_ids: list[str] = []

    for case in cases:
        outcome = outcome_by_id.get(case.case_id)
        if outcome is None or outcome.answer is None:
            execution_errors += 1
            empty_or_refusal += 1
            failed_case_ids.append(case.case_id)
            continue
        answer = outcome.answer
        grounded += int(answer.grounded)
        if not answer.numbers:
            empty_or_refusal += 1
        else:
            responses_with_numbers += 1
        returned = {_number_fact_key(fact) for fact in answer.numbers}
        required = {_gold_fact_key(fact) for fact in case.required_facts}
        allowed = {_gold_fact_key(fact) for fact in case.allowed_facts}
        false_facts = returned.difference(allowed)
        if false_facts:
            false_number_cases += 1
            false_number_count += len(false_facts)

        if case.expect_refusal:
            if not returned:
                correct_refusals += 1
            else:
                failed_case_ids.append(case.case_id)
            continue
        if required.issubset(returned) and not false_facts:
            exact_matches += 1
        else:
            failed_case_ids.append(case.case_id)

    answerable_count = sum(not case.expect_refusal for case in cases)
    trap_count = len(cases) - answerable_count
    return NumericMetrics(
        question_count=len(cases),
        answerable_count=answerable_count,
        trap_count=trap_count,
        exact_match_rate=_rate(exact_matches, answerable_count),
        grounded_rate=_rate(grounded, len(cases)),
        empty_or_refusal_rate=_rate(empty_or_refusal, len(cases)),
        false_number_rate=_rate(false_number_cases, responses_with_numbers),
        correct_refusal_rate=_rate(correct_refusals, trap_count),
        execution_error_rate=_rate(execution_errors, len(cases)),
        false_number_case_count=false_number_cases,
        false_number_count=false_number_count,
        failed_case_ids=failed_case_ids,
    )


def generate_narrative_candidates(
    narrative_set: NarrativeSet,
    engine: Engine,
    *,
    settings: Settings,
    agent_runner: AgentRunner = run_agent,
) -> tuple[
    dict[str, list[NarrativeCandidate]],
    dict[str, dict[str, int | float | Decimal]],
]:
    """Generate both prompt answers over identical real-agent retrieval contexts."""
    corpus_by_identity = {chunk.identity: chunk for chunk in load_source_chunks(engine)}
    strict_generator = AnswerGenerator(
        provider=EvaluationGenerationProvider(
            client=_openai_client(settings),
            instructions=_CITATION_FIRST_INSTRUCTIONS,
        ),
        model=settings.openai_model,
    )
    candidates = {prompt: [] for prompt in PROMPT_ORDER}
    production_tracer = RequestTracer(settings=settings)
    strict_tracer = RequestTracer(settings=settings)

    production_started = perf_counter()
    production_errors = 0
    with production_tracer.trace(
        trace_id=uuid4().hex,
        question="evaluate production narrative prompt",
        session_id=None,
    ) as production_trace:
        runs: list[tuple[NarrativeCase, SourceChunk, AgentRun]] = []
        for case in narrative_set.cases:
            reference = corpus_by_identity.get(case.identity)
            if reference is None:
                raise RuntimeError(f"Narrative reference chunk is missing: {case.identity}.")
            run, failed = _run_narrative_agent(
                case,
                agent_runner=agent_runner,
            )
            production_errors += int(failed)
            runs.append((case, reference, run))
            candidates["production"].append(
                _narrative_candidate(case, reference, run.context, run.answer)
            )
    production_seconds = perf_counter() - production_started

    strict_started = perf_counter()
    strict_errors = 0
    with strict_tracer.trace(
        trace_id=uuid4().hex,
        question="evaluate citation-first narrative prompt",
        session_id=None,
    ) as strict_trace:
        for case, reference, run in runs:
            if run.context.route == "evaluation_error":
                answer = run.answer
                strict_errors += 1
            else:
                try:
                    answer = strict_generator.generate(case.question, run.context)
                except Exception as exc:
                    strict_errors += 1
                    logger.warning(
                        "Citation-first generation failed during evaluation",
                        extra={
                            "case_id": case.case_id,
                            "error_type": type(exc).__name__,
                        },
                    )
                    answer = _failed_evaluation_answer()
            candidates["citation_first"].append(
                _narrative_candidate(case, reference, run.context, answer)
            )
    strict_seconds = perf_counter() - strict_started
    return candidates, {
        "production": {
            **_trace_metrics(production_trace.metrics, production_seconds),
            "execution_error_count": production_errors,
        },
        "citation_first": {
            **_trace_metrics(strict_trace.metrics, strict_seconds),
            "execution_error_count": strict_errors,
        },
    }


def run_ragas_evaluation(
    candidates: Sequence[NarrativeCandidate],
    *,
    prompt_name: str,
    settings: Settings,
) -> NarrativePromptMetrics:
    """Evaluate one prompt with four RAGAS metrics and configured judge models."""
    if not candidates:
        raise ValueError("At least one narrative candidate is required.")
    started_at = perf_counter()
    ragas = _load_ragas()
    usage = UsageAccumulator(settings)
    judge_model = _judge_model(settings)
    ragas_llm = _build_ragas_llm(
        ragas["BaseRagasLLM"],
        settings=settings,
        model=judge_model,
        usage=usage,
    )
    ragas_embeddings = _build_ragas_embeddings(
        ragas["BaseRagasEmbeddings"],
        settings=settings,
        usage=usage,
    )
    dataset = ragas["EvaluationDataset"].from_list(
        [
            {
                "user_input": candidate.question,
                "response": candidate.answer.text,
                "retrieved_contexts": candidate.contexts,
                "reference": candidate.reference,
            }
            for candidate in candidates
        ]
    )
    metrics = [
        ragas["Faithfulness"](),
        ragas["AnswerRelevancy"](),
        ragas["LLMContextPrecisionWithReference"](name="context_precision"),
        ragas["LLMContextRecall"](),
    ]
    result = ragas["evaluate"](
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=ragas["RunConfig"](
            timeout=180,
            max_retries=3,
            max_wait=30,
            max_workers=3,
            seed=DEFAULT_SEED,
        ),
        raise_exceptions=False,
        show_progress=True,
        batch_size=5,
    )
    scores = {
        name: _penalized_mean(result[name], expected_count=len(candidates))
        for name in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        )
    }
    nan_count = sum(not math.isfinite(float(value)) for name in scores for value in result[name])
    grounded_rate = _rate(
        sum(candidate.answer.grounded for candidate in candidates),
        len(candidates),
    )
    citation_rate = _rate(
        sum(bool(candidate.answer.citations) for candidate in candidates),
        len(candidates),
    )
    execution_error_count = sum(
        candidate.answer == _failed_evaluation_answer() for candidate in candidates
    )
    return NarrativePromptMetrics(
        prompt_name=prompt_name,
        question_count=len(candidates),
        faithfulness=scores["faithfulness"],
        answer_relevancy=scores["answer_relevancy"],
        context_precision=scores["context_precision"],
        context_recall=scores["context_recall"],
        grounded_rate=grounded_rate,
        citation_rate=citation_rate,
        execution_error_count=execution_error_count,
        execution_error_rate=_rate(execution_error_count, len(candidates)),
        composite_score=round(statistics.fmean(scores.values()), 6),
        ragas_nan_count=nan_count,
        duration_seconds=round(perf_counter() - started_at, 3),
        input_tokens=usage.metrics.input_tokens,
        output_tokens=usage.metrics.output_tokens,
        cached_tokens=usage.metrics.cached_tokens,
        cost_usd=usage.metrics.cost_usd,
    )


def select_best_prompt(results: Sequence[NarrativePromptMetrics]) -> str:
    """Select by composite RAGAS score, then faithfulness and citation rate."""
    if not results:
        raise ValueError("At least one prompt result is required.")
    return max(
        results,
        key=lambda result: (
            result.composite_score,
            result.faithfulness,
            result.citation_rate,
            -PROMPT_ORDER.index(result.prompt_name),
        ),
    ).prompt_name


def persist_generation_runs(
    engine: Engine,
    *,
    numeric_path: Path,
    narrative_path: Path,
    settings: Settings,
    numeric_metrics: NumericMetrics,
    prompt_results: Sequence[NarrativePromptMetrics],
) -> dict[str, int]:
    """Persist numeric and narrative generation evaluations atomically."""
    statement = text(
        """
        INSERT INTO eval_runs (run_type, config, metrics)
        VALUES ('generation', CAST(:config AS JSONB), CAST(:metrics AS JSONB))
        RETURNING id
        """
    )
    entries: list[tuple[str, dict[str, object], dict[str, object]]] = [
        (
            "numeric_gold",
            {
                "evaluation": "numeric_gold",
                "dataset_path": str(numeric_path),
                "seed": DEFAULT_SEED,
                "question_count": numeric_metrics.question_count,
                "generation_model": settings.openai_model,
                "retrieval_strategy": settings.retrieval_strategy,
            },
            numeric_metrics.model_dump(mode="json"),
        )
    ]
    for result in prompt_results:
        entries.append(
            (
                result.prompt_name,
                {
                    "evaluation": "narrative_ragas",
                    "prompt": result.prompt_name,
                    "prompt_hash": _prompt_hash(result.prompt_name),
                    "dataset_path": str(narrative_path),
                    "seed": DEFAULT_SEED,
                    "question_count": result.question_count,
                    "generation_model": settings.openai_model,
                    "judge_model": _judge_model(settings),
                    "embedding_model": settings.openai_embedding_model,
                    "retrieval_strategy": settings.retrieval_strategy,
                },
                result.model_dump(mode="json", exclude={"eval_run_id"}),
            )
        )

    run_ids: dict[str, int] = {}
    with engine.begin() as connection:
        for name, config, metrics in entries:
            row = connection.execute(
                statement,
                {
                    "config": json.dumps(config, sort_keys=True),
                    "metrics": json.dumps(metrics, sort_keys=True),
                },
            ).one()
            run_ids[name] = int(row[0])
    return run_ids


def _comparison_case(
    index: int,
    from_fact: DatabaseFact,
    to_fact: DatabaseFact,
) -> NumericGoldCase:
    label = _METRIC_LABELS[from_fact.metric]
    delta = to_fact.value - from_fact.value
    percent = Decimal("100") * delta / from_fact.value
    delta_fact = GoldFact.from_values(
        metric=f"{from_fact.metric}_yoy_delta",
        fiscal_year=to_fact.fiscal_year,
        fiscal_period=to_fact.fiscal_period,
        value=float(delta),
        unit=from_fact.unit,
    )
    percent_fact = GoldFact.from_values(
        metric=f"{from_fact.metric}_yoy_percent_change",
        fiscal_year=to_fact.fiscal_year,
        fiscal_period=to_fact.fiscal_period,
        value=float(percent),
        unit="percent",
    )
    return NumericGoldCase(
        case_id=f"comparison-{index:03d}",
        question=(
            f"How did {from_fact.ticker}'s {label} change from "
            f"FY{from_fact.fiscal_year} to FY{to_fact.fiscal_year}? "
            "Return both the delta and percentage change."
        ),
        required_facts=[delta_fact, percent_fact],
        allowed_facts=[
            from_fact.gold(),
            to_fact.gold(),
            delta_fact,
            percent_fact,
        ],
        case_type="comparison",
    )


def _direct_question(fact: DatabaseFact, *, label: str) -> str:
    period = (
        f"FY{fact.fiscal_year}"
        if fact.fiscal_period == "FY"
        else f"{fact.fiscal_period} FY{fact.fiscal_year}"
    )
    return f"What was {fact.ticker}'s {label} in {period}?"


def _trap_cases() -> list[NumericGoldCase]:
    questions = (
        ("trap-excluded-tsla", "What was TSLA's net income in FY2024?"),
        ("trap-excluded-xom", "What was XOM's revenue in FY2024?"),
        ("trap-excluded-wfc", "What was WFC's total assets in FY2024?"),
        ("trap-year-aapl", "What was AAPL's revenue in FY1850?"),
        ("trap-year-msft", "What was MSFT's net income in FY2099?"),
        ("trap-metric-aapl", "What was AAPL's gross margin in FY2024?"),
    )
    return [
        NumericGoldCase(
            case_id=case_id,
            question=question,
            expect_refusal=True,
            case_type="no_answer",
        )
        for case_id, question in questions
    ]


def _narrative_candidate(
    case: NarrativeCase,
    reference: SourceChunk,
    context: AgentContext,
    answer: Answer,
) -> NarrativeCandidate:
    return NarrativeCandidate(
        case_id=case.case_id,
        question=case.question,
        reference=reference.text,
        contexts=[chunk["text"] for chunk in context.chunks],
        answer=answer,
    )


def _run_narrative_agent(
    case: NarrativeCase,
    *,
    agent_runner: AgentRunner,
) -> tuple[AgentRun, bool]:
    trace_id = f"eval-generation-{case.case_id}"
    try:
        return agent_runner(case.question, trace_id=trace_id), False
    except Exception as exc:
        logger.warning(
            "Agent execution failed during narrative evaluation",
            extra={
                "case_id": case.case_id,
                "error_type": type(exc).__name__,
            },
        )
        return (
            AgentRun(
                answer=_failed_evaluation_answer(),
                context=AgentContext(route="evaluation_error", trace_id=trace_id),
            ),
            True,
        )


def _failed_evaluation_answer() -> Answer:
    return Answer(text="No answer was generated.", grounded=False)


def _generation_input(question: str, context: AgentContext) -> str:
    payload = {
        "question": question,
        "trace_id": context.trace_id,
        "route": context.route,
        "data_unavailable": context.data_unavailable,
        "facts": [fact.model_dump(mode="json") for fact in context.facts],
        "chunks": [_chunk_evidence(chunk) for chunk in context.chunks],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _chunk_evidence(chunk: Chunk) -> dict[str, object]:
    return {
        "accession_no": chunk["accession_no"],
        "ticker": chunk["ticker"],
        "fiscal_year": chunk["fiscal_year"],
        "fiscal_period": chunk["fiscal_period"],
        "section": chunk["section"],
        "text": chunk["text"],
        "source_url": chunk["source_url"],
    }


def _load_ragas() -> dict[str, Any]:
    """Load RAGAS 0.3 with compatibility shims for locked transitive packages."""
    import pyarrow

    if not hasattr(pyarrow, "PyExtensionType"):
        pyarrow.PyExtensionType = pyarrow.ExtensionType
    module_name = "langchain_community.chat_models.vertexai"
    if module_name not in sys.modules:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            shim = types.ModuleType(module_name)
            shim.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules[module_name] = shim

    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings.base import BaseRagasEmbeddings
    from ragas.llms.base import BaseRagasLLM
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.run_config import RunConfig

    return {
        "EvaluationDataset": EvaluationDataset,
        "evaluate": evaluate,
        "BaseRagasEmbeddings": BaseRagasEmbeddings,
        "BaseRagasLLM": BaseRagasLLM,
        "AnswerRelevancy": AnswerRelevancy,
        "Faithfulness": Faithfulness,
        "LLMContextPrecisionWithReference": LLMContextPrecisionWithReference,
        "LLMContextRecall": LLMContextRecall,
        "RunConfig": RunConfig,
    }


def _build_ragas_llm(
    base_class: type[Any],
    *,
    settings: Settings,
    model: str,
    usage: UsageAccumulator,
) -> Any:
    api_key = _api_key(settings)

    class ResponsesRagasLLM(base_class):
        def __init__(self) -> None:
            super().__init__()
            self._client = OpenAI(api_key=api_key)
            self._async_client = AsyncOpenAI(api_key=api_key)

        def is_finished(self, response: LLMResult) -> bool:
            del response
            return True

        def generate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = None,
            stop: list[str] | None = None,
            callbacks: object = None,
        ) -> LLMResult:
            del temperature, stop, callbacks
            responses = [
                self._client.responses.create(
                    model=model,
                    input=prompt.to_string(),
                )
                for _ in range(n)
            ]
            return _ragas_llm_result(responses, model=model, usage=usage)

        async def agenerate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = None,
            stop: list[str] | None = None,
            callbacks: object = None,
        ) -> LLMResult:
            del temperature, stop, callbacks
            responses = await asyncio.gather(
                *[
                    self._async_client.responses.create(
                        model=model,
                        input=prompt.to_string(),
                    )
                    for _ in range(n)
                ]
            )
            return _ragas_llm_result(responses, model=model, usage=usage)

    return ResponsesRagasLLM()


def _build_ragas_embeddings(
    base_class: type[Any],
    *,
    settings: Settings,
    usage: UsageAccumulator,
) -> Any:
    api_key = _api_key(settings)
    model = settings.openai_embedding_model
    dimensions = settings.embedding_dim

    class ResponsesRagasEmbeddings(base_class):
        def __init__(self) -> None:
            super().__init__()
            self._client = OpenAI(api_key=api_key)
            self._async_client = AsyncOpenAI(api_key=api_key)

        def embed_query(self, value: str) -> list[float]:
            return self.embed_documents([value])[0]

        def embed_documents(self, values: list[str]) -> list[list[float]]:
            response = self._client.embeddings.create(
                model=model,
                input=values,
                dimensions=dimensions,
            )
            usage.add_response(response, model=model, kind="embedding")
            return [
                list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)
            ]

        async def aembed_query(self, value: str) -> list[float]:
            return (await self.aembed_documents([value]))[0]

        async def aembed_documents(self, values: list[str]) -> list[list[float]]:
            response = await self._async_client.embeddings.create(
                model=model,
                input=values,
                dimensions=dimensions,
            )
            usage.add_response(response, model=model, kind="embedding")
            return [
                list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)
            ]

    return ResponsesRagasEmbeddings()


def _ragas_llm_result(
    responses: Sequence[Any],
    *,
    model: str,
    usage: UsageAccumulator,
) -> LLMResult:
    input_tokens = 0
    output_tokens = 0
    generations: list[Generation] = []
    for response in responses:
        usage.add_response(response, model=model)
        token_usage = extract_token_usage(response)
        input_tokens += token_usage.input_tokens
        output_tokens += token_usage.output_tokens
        generations.append(Generation(text=response.output_text))
    return LLMResult(
        generations=[generations],
        llm_output={
            "token_usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            }
        },
    )


def _number_fact_key(fact: NumberFact) -> FactKey:
    return (
        fact.metric,
        fact.fiscal_year,
        fact.fiscal_period,
        Decimal(str(fact.value)),
        fact.unit,
    )


def _gold_fact_key(fact: GoldFact) -> FactKey:
    return (
        fact.metric,
        fact.fiscal_year,
        fact.fiscal_period,
        Decimal(fact.value),
        fact.unit,
    )


def _penalized_mean(values: Sequence[float], *, expected_count: int) -> float:
    if expected_count <= 0:
        raise ValueError("expected_count must be positive.")
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return round(sum(finite) / expected_count, 6)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _judge_model(settings: Settings) -> str:
    return settings.openai_classifier_model.strip() or settings.openai_model.strip()


def _api_key(settings: Settings) -> str:
    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY must be configured for generation evaluation.")
    value = settings.openai_api_key.get_secret_value().strip()
    if not value:
        raise ValueError("OPENAI_API_KEY must be configured for generation evaluation.")
    return value


def _openai_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=_api_key(settings))


def _prompt_hash(prompt_name: str) -> str:
    prompt = (
        _CITATION_FIRST_INSTRUCTIONS
        if prompt_name == "citation_first"
        else "production:ffa.agent.generation"
    )
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _trace_metrics(
    metrics: QueryMetrics,
    duration_seconds: float,
) -> dict[str, int | float | Decimal]:
    return {
        "duration_seconds": round(duration_seconds, 3),
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "cached_tokens": metrics.cached_tokens,
        "cost_usd": metrics.cost_usd,
    }


def _write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _load_model[T: BaseModel](path: Path, model_type: type[T]) -> T:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation set is missing: {path}. Run with --build.")
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-gold", type=Path, default=DEFAULT_NUMERIC_GOLD_PATH)
    parser.add_argument("--narrative-set", type=Path, default=DEFAULT_NARRATIVE_SET_PATH)
    parser.add_argument(
        "--retrieval-groundtruth",
        type=Path,
        default=DEFAULT_RETRIEVAL_GROUNDTRUTH_PATH,
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--no-persist", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build optional frozen sets, execute evaluation, and print JSON."""
    arguments = _parse_args()
    settings = get_settings()
    engine = create_rw_engine(settings.database_url)
    try:
        if arguments.build or arguments.build_only:
            numeric_set, narrative_set = build_generation_sets(
                engine,
                numeric_path=arguments.numeric_gold,
                narrative_path=arguments.narrative_set,
                retrieval_groundtruth_path=arguments.retrieval_groundtruth,
            )
            print(
                json.dumps(
                    {
                        "numeric_cases": len(numeric_set.cases),
                        "narrative_cases": len(narrative_set.cases),
                    },
                    indent=2,
                )
            )
        if arguments.build_only:
            return
        summary = evaluate_generation(
            numeric_path=arguments.numeric_gold,
            narrative_path=arguments.narrative_set,
            engine=engine,
            settings=settings,
            persist=not arguments.no_persist,
        )
        print(summary.model_dump_json(indent=2))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
