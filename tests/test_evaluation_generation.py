"""Tests for numeric gold scoring and narrative prompt evaluation helpers."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from ffa.agent.schemas import Answer, NumberFact
from ffa.config import Settings
from ffa.evaluation.build_groundtruth import (
    GroundTruthBuildStats,
    GroundTruthPair,
    RetrievalGroundTruth,
)
from ffa.evaluation.eval_generation import (
    DatabaseFact,
    GoldFact,
    NarrativePromptMetrics,
    NumericCaseOutcome,
    NumericGoldCase,
    _penalized_mean,
    _run_narrative_agent,
    build_narrative_set,
    build_numeric_gold_set,
    persist_generation_runs,
    score_numeric_outcomes,
    select_best_prompt,
)


class FakePersistResult:
    """Minimal SQLAlchemy result returning one generated ID."""

    def __init__(self, row_id: int) -> None:
        self._row_id = row_id

    def one(self) -> tuple[int]:
        return (self._row_id,)


class FakePersistConnection:
    """Capture generation eval JSON payloads."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def execute(self, _: object, parameters: dict[str, str]) -> FakePersistResult:
        self.calls.append(parameters)
        return FakePersistResult(len(self.calls))


class FakeBeginContext(AbstractContextManager[FakePersistConnection]):
    """Expose one fake transaction."""

    def __init__(self, connection: FakePersistConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakePersistConnection:
        return self._connection

    def __exit__(self, *args: object) -> None:
        del args


class FakePersistEngine:
    """Provide one captured transaction for eval_runs writes."""

    def __init__(self) -> None:
        self.connection = FakePersistConnection()

    def begin(self) -> FakeBeginContext:
        return FakeBeginContext(self.connection)


def test_numeric_gold_build_is_seeded_balanced_and_includes_traps() -> None:
    facts = [
        _database_fact(ticker, metric, year, value)
        for ticker, year, value in (
            ("AAPL", 2023, 100),
            ("AAPL", 2024, 110),
            ("MSFT", 2023, 200),
            ("MSFT", 2024, 180),
            ("JPM", 2023, 300),
            ("JPM", 2024, 330),
        )
        for metric in ("revenue", "net_income")
    ]

    first = build_numeric_gold_set(
        facts,
        seed=42,
        direct_count=4,
        comparison_count=2,
    )
    second = build_numeric_gold_set(
        facts,
        seed=42,
        direct_count=4,
        comparison_count=2,
    )

    assert first.cases == second.cases
    assert len(first.cases) == 12
    assert sum(case.case_type == "direct" for case in first.cases) == 4
    assert sum(case.case_type == "comparison" for case in first.cases) == 2
    assert sum(case.expect_refusal for case in first.cases) == 6
    comparison = next(case for case in first.cases if case.case_type == "comparison")
    assert {fact.metric for fact in comparison.required_facts} == {
        f"{comparison.allowed_facts[0].metric}_yoy_delta",
        f"{comparison.allowed_facts[0].metric}_yoy_percent_change",
    }


def test_numeric_scoring_separates_missing_answers_from_false_numbers() -> None:
    expected = GoldFact(
        metric="net_income",
        fiscal_year=2024,
        fiscal_period="FY",
        value="100",
        unit="USD",
    )
    cases = [
        NumericGoldCase(
            case_id="correct",
            question="Correct?",
            required_facts=[expected],
            allowed_facts=[expected],
            case_type="direct",
        ),
        NumericGoldCase(
            case_id="wrong",
            question="Wrong?",
            required_facts=[expected],
            allowed_facts=[expected],
            case_type="direct",
        ),
        NumericGoldCase(
            case_id="trap",
            question="Unavailable?",
            expect_refusal=True,
            case_type="no_answer",
        ),
    ]
    outcomes = [
        NumericCaseOutcome(
            case_id="correct",
            answer=Answer(
                text="Correct.",
                numbers=[_number_fact(100)],
                grounded=True,
            ),
        ),
        NumericCaseOutcome(
            case_id="wrong",
            answer=Answer(
                text="Wrong.",
                numbers=[_number_fact(101)],
                grounded=True,
            ),
        ),
        NumericCaseOutcome(
            case_id="trap",
            answer=Answer(text="Data unavailable.", grounded=True),
        ),
    ]

    metrics = score_numeric_outcomes(cases, outcomes)

    assert metrics.exact_match_rate == 0.5
    assert metrics.grounded_rate == 1.0
    assert metrics.empty_or_refusal_rate == 0.333333
    assert metrics.false_number_rate == 0.5
    assert metrics.correct_refusal_rate == 1.0
    assert metrics.execution_error_rate == 0.0
    assert metrics.false_number_case_count == 1
    assert metrics.false_number_count == 1
    assert metrics.failed_case_ids == ["wrong"]


def test_narrative_set_is_seeded_and_balanced_by_section() -> None:
    retrieval_set = _retrieval_groundtruth()

    first = build_narrative_set(retrieval_set, seed=42, count=6)
    second = build_narrative_set(retrieval_set, seed=42, count=6)

    assert first.cases == second.cases
    assert {
        section: sum(case.expected_section == section for case in first.cases)
        for section in (
            "MD&A",
            "Risk Factors",
            "Notes",
        )
    } == {
        "MD&A": 2,
        "Risk Factors": 2,
        "Notes": 2,
    }


def test_best_prompt_uses_composite_then_faithfulness() -> None:
    production = _prompt_metrics("production", composite=0.8, faithfulness=0.9)
    citation_first = _prompt_metrics(
        "citation_first",
        composite=0.81,
        faithfulness=0.85,
    )

    assert select_best_prompt([production, citation_first]) == "citation_first"


def test_narrative_agent_failure_is_counted_without_aborting() -> None:
    case = build_narrative_set(_retrieval_groundtruth(), seed=42, count=3).cases[0]

    def failing_runner(question: str, *, trace_id: str | None = None) -> Any:
        del question, trace_id
        raise RuntimeError("invalid generated SQL")

    run, failed = _run_narrative_agent(case, agent_runner=failing_runner)

    assert failed is True
    assert run.context.route == "evaluation_error"
    assert run.answer.grounded is False
    assert run.answer.numbers == []
    assert _penalized_mean([1.0, float("nan")], expected_count=2) == 0.5


def test_generation_runs_persist_numeric_and_both_prompt_configs() -> None:
    engine = FakePersistEngine()
    numeric_metrics = score_numeric_outcomes(
        [
            NumericGoldCase(
                case_id="trap",
                question="Unavailable?",
                expect_refusal=True,
                case_type="no_answer",
            )
        ],
        [
            NumericCaseOutcome(
                case_id="trap",
                answer=Answer(text="Unavailable.", grounded=True),
            )
        ],
    )
    prompt_results = [
        _prompt_metrics("production", composite=0.8, faithfulness=0.9),
        _prompt_metrics("citation_first", composite=0.81, faithfulness=0.85),
    ]

    run_ids = persist_generation_runs(
        engine,  # type: ignore[arg-type]
        numeric_path=Path("evaluation/generation_numeric_gold.json"),
        narrative_path=Path("evaluation/generation_narrative_set.json"),
        settings=Settings(
            _env_file=None,
            openai_model="configured-primary",
            openai_classifier_model="configured-judge",
            retrieval_strategy="vector_rerank",
        ),
        numeric_metrics=numeric_metrics,
        prompt_results=prompt_results,
    )

    assert run_ids == {
        "numeric_gold": 1,
        "production": 2,
        "citation_first": 3,
    }
    configs = [json.loads(call["config"]) for call in engine.connection.calls]
    assert [config["evaluation"] for config in configs] == [
        "numeric_gold",
        "narrative_ragas",
        "narrative_ragas",
    ]
    assert configs[1]["prompt"] == "production"
    assert configs[2]["prompt"] == "citation_first"
    assert all(config["seed"] == 42 for config in configs)


def _database_fact(
    ticker: str,
    metric: str,
    fiscal_year: int,
    value: int,
) -> DatabaseFact:
    return DatabaseFact(
        ticker=ticker,
        metric=metric,
        fiscal_year=fiscal_year,
        fiscal_period="FY",
        value=Decimal(value),
        unit="USD",
    )


def _number_fact(value: float) -> NumberFact:
    return NumberFact(
        metric="net_income",
        fiscal_year=2024,
        fiscal_period="FY",
        value=value,
        unit="USD",
    )


def _prompt_metrics(
    prompt_name: str,
    *,
    composite: float,
    faithfulness: float,
) -> NarrativePromptMetrics:
    return NarrativePromptMetrics(
        prompt_name=prompt_name,
        question_count=30,
        faithfulness=faithfulness,
        answer_relevancy=0.8,
        context_precision=0.8,
        context_recall=0.8,
        grounded_rate=1.0,
        citation_rate=1.0,
        execution_error_count=0,
        execution_error_rate=0.0,
        composite_score=composite,
        ragas_nan_count=0,
        duration_seconds=1.0,
        input_tokens=10,
        output_tokens=5,
        cached_tokens=0,
        cost_usd=Decimal("0.01"),
    )


def _retrieval_groundtruth() -> RetrievalGroundTruth:
    pairs = []
    chunk_id = 1
    for section in ("MD&A", "Risk Factors", "Notes"):
        for index in range(3):
            pairs.append(
                GroundTruthPair(
                    pair_id=f"{section}-{index}",
                    question=f"What does AAPL disclose for {section} case {index}?",
                    expected_chunk_id=chunk_id,
                    expected_accession_no=f"accession-{chunk_id}",
                    expected_section=section,
                    expected_chunk_index=index,
                    ticker="AAPL",
                    source="synthetic",
                    lexical_overlap=0.2,
                )
            )
            chunk_id += 1
    return RetrievalGroundTruth(
        version=1,
        seed=42,
        generated_at=datetime.now(UTC),
        generation_model="configured-classifier",
        corpus_fingerprint="fingerprint",
        eligible_sections=("MD&A", "Risk Factors", "Notes"),
        filter_policy={},
        stats=GroundTruthBuildStats(
            candidates_sampled=9,
            generated_questions=9,
            quality_passed=9,
            retained_synthetic=9,
            rejected_synthetic=0,
            manual_questions=0,
            rejection_reasons={},
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            cost_usd=Decimal("0"),
            elapsed_seconds=0.1,
        ),
        pairs=pairs,
    )


def json_load(value: str) -> dict[str, Any]:
    return json.loads(value)
