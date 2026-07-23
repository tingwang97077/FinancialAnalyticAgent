"""Build a reproducible retrieval ground truth from real filing chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol
from uuid import uuid4

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, text

from ffa.common.db import create_rw_engine
from ffa.config import Settings, get_settings
from ffa.monitoring.tracing import RequestTracer, record_openai_response

logger = logging.getLogger(__name__)

GROUNDTRUTH_VERSION = 1
DEFAULT_SEED = 42
DEFAULT_SYNTHETIC_TARGET = 90
DEFAULT_CANDIDATE_COUNT = 260
DEFAULT_BATCH_SIZE = 20
DEFAULT_OUTPUT_PATH = Path("evaluation/retrieval_groundtruth.json")
ELIGIBLE_SECTIONS = ("MD&A", "Risk Factors", "Notes")

_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", flags=re.IGNORECASE)
_META_PHRASES = (
    "according to the passage",
    "according to the chunk",
    "in the provided text",
    "in this document",
    "in the excerpt",
    "the passage",
    "the chunk",
)
_CORPORATE_SUFFIXES = frozenset(
    {
        "and",
        "company",
        "co",
        "corp",
        "corporation",
        "group",
        "holdings",
        "inc",
        "incorporated",
        "limited",
        "llc",
        "plc",
        "the",
    }
)
_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "according",
        "after",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "before",
        "between",
        "by",
        "can",
        "company",
        "could",
        "did",
        "do",
        "does",
        "during",
        "explain",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "management",
        "might",
        "of",
        "on",
        "or",
        "reported",
        "say",
        "says",
        "the",
        "their",
        "they",
        "to",
        "under",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "why",
        "will",
        "with",
        "would",
    }
)
_GENERIC_CONTENT_WORDS = frozenset(
    {
        "business",
        "disclose",
        "disclosed",
        "disclosure",
        "discuss",
        "filing",
        "important",
        "investor",
        "investors",
        "mention",
        "risk",
        "risks",
    }
)

_QUESTION_INSTRUCTIONS = """You create a retrieval benchmark for SEC filing analysis.
For every supplied source chunk, write exactly one natural English question whose answer is
contained in that chunk.

Requirements:
- Write the question as a financial analyst would ask it.
- Keep it concise, ideally 15 to 30 words and never more than 45 words.
- Mention the issuer by its common name or ticker so the question is specific without
  metadata filters.
- Target one concrete disclosure, accounting treatment, business driver, or risk unique enough that
  an unrelated filing chunk could not answer it.
- Paraphrase aggressively with synonyms and a natural formulation.
- Do not copy a phrase of five or more consecutive words from the source.
- Before returning, compare the draft with the source and rewrite every sequence of five shared
  consecutive words. Prefer different verbs, nouns, and sentence structure over source terminology.
- Do not quote the source and do not mention a passage, chunk, excerpt, supplied text, or benchmark.
- Do not include the answer in the question.
- Mark is_specific=false if a focused, independently answerable question cannot be formed.

Return one structured item for every input chunk and preserve each chunk_id exactly.
"""


class SourceChunk(BaseModel):
    """Real database chunk available for synthetic question generation."""

    model_config = ConfigDict(frozen=True)

    id: int
    accession_no: str
    ticker: str
    company_name: str
    section: str
    chunk_index: int
    token_count: int
    text: str
    source_url: str

    @property
    def identity(self) -> tuple[str, str, int]:
        """Return the stable filing identity used across re-ingestion IDs."""
        return (self.accession_no, self.section, self.chunk_index)

    @property
    def size_bucket(self) -> str:
        """Classify source length for stratified sampling."""
        if self.token_count < 400:
            return "short"
        if self.token_count < 680:
            return "medium"
        return "long"


class GeneratedQuestion(BaseModel):
    """One typed LLM output associated with an input chunk."""

    chunk_id: int
    question: str
    is_specific: bool


class GeneratedQuestionBatch(BaseModel):
    """Structured output envelope for a batch generation request."""

    questions: list[GeneratedQuestion]


@dataclass(frozen=True, slots=True)
class QuestionGenerationResult:
    """Validated provider output before deterministic quality filtering."""

    questions: tuple[GeneratedQuestion, ...]


class QuestionGenerator(Protocol):
    """Pluggable structured question-generation backend."""

    def generate(
        self,
        chunks: Sequence[SourceChunk],
        *,
        model: str,
    ) -> QuestionGenerationResult:
        """Generate exactly one typed question per source chunk."""
        ...


class OpenAIQuestionGenerator:
    """Generate benchmark questions with OpenAI native structured output."""

    def __init__(self, client: OpenAI) -> None:
        """Initialize the provider with an explicit SDK client."""
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAIQuestionGenerator:
        """Create the provider from the configured application credentials."""
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY must be configured to build ground truth.")
        api_key = settings.openai_api_key.get_secret_value().strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be configured to build ground truth.")
        return cls(OpenAI(api_key=api_key))

    def generate(
        self,
        chunks: Sequence[SourceChunk],
        *,
        model: str,
    ) -> QuestionGenerationResult:
        """Generate one structured question for every supplied source."""
        payload = [
            {
                "chunk_id": chunk.id,
                "ticker": chunk.ticker,
                "company_name": chunk.company_name,
                "section": chunk.section,
                "source_text": chunk.text,
            }
            for chunk in chunks
        ]
        response = self._client.responses.parse(
            model=model,
            instructions=_QUESTION_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=GeneratedQuestionBatch,
        )
        record_openai_response(response, model=model)
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI did not return structured benchmark questions.")
        return QuestionGenerationResult(questions=tuple(parsed.questions))


class GroundTruthPair(BaseModel):
    """One reproducible question and its single expected source chunk."""

    model_config = ConfigDict(frozen=True)

    pair_id: str
    question: str
    expected_chunk_id: int
    expected_accession_no: str
    expected_section: str
    expected_chunk_index: int
    ticker: str
    source: Literal["synthetic", "manual"]
    lexical_overlap: float = Field(ge=0, le=1)


class GroundTruthBuildStats(BaseModel):
    """Quality, usage, and duration counters for one dataset build."""

    candidates_sampled: int
    generated_questions: int
    quality_passed: int
    retained_synthetic: int
    rejected_synthetic: int
    manual_questions: int
    rejection_reasons: dict[str, int]
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: Decimal
    elapsed_seconds: float


class RetrievalGroundTruth(BaseModel):
    """Versioned retrieval benchmark persisted independently of eval runs."""

    version: int
    seed: int
    generated_at: datetime
    generation_model: str
    corpus_fingerprint: str
    eligible_sections: tuple[str, ...]
    filter_policy: dict[str, object]
    stats: GroundTruthBuildStats
    pairs: list[GroundTruthPair]


@dataclass(frozen=True, slots=True)
class ManualQuestion:
    """Hand-written control question resolved by stable chunk identity."""

    question: str
    ticker: str
    accession_no: str
    section: str
    chunk_index: int

    @property
    def identity(self) -> tuple[str, str, int]:
        """Return the referenced stable filing identity."""
        return (self.accession_no, self.section, self.chunk_index)


MANUAL_QUESTIONS: tuple[ManualQuestion, ...] = (
    ManualQuestion(
        question=(
            "How could trade restrictions disrupt Apple's access to components and its global "
            "supply network?"
        ),
        ticker="AAPL",
        accession_no="0000320193-25-000079",
        section="Risk Factors",
        chunk_index=1,
    ),
    ManualQuestion(
        question=(
            "What does JPMorgan disclose about delinquency and defaults among recently modified "
            "wholesale loans?"
        ),
        ticker="JPM",
        accession_no="0001628280-26-008131",
        section="Notes",
        chunk_index=70,
    ),
    ManualQuestion(
        question=(
            "Which businesses feed Microsoft's commercial cloud indicators, and what are those "
            "indicators intended to show investors?"
        ),
        ticker="MSFT",
        accession_no="0000950170-25-100235",
        section="MD&A",
        chunk_index=3,
    ),
    ManualQuestion(
        question=(
            "How might tighter semiconductor export rules constrain NVIDIA's overseas demand and "
            "operations?"
        ),
        ticker="NVDA",
        accession_no="0001045810-26-000021",
        section="Risk Factors",
        chunk_index=22,
    ),
    ManualQuestion(
        question=(
            "What offset the benefit of higher AWS sales when Amazon explained the change in that "
            "segment's operating profit?"
        ),
        ticker="AMZN",
        accession_no="0001018724-26-000004",
        section="MD&A",
        chunk_index=11,
    ),
    ManualQuestion(
        question=(
            "Why does Alphabet expect compliance burdens from digital-platform and AI regulation "
            "to keep increasing?"
        ),
        ticker="GOOGL",
        accession_no="0001652044-26-000018",
        section="Risk Factors",
        chunk_index=15,
    ),
    ManualQuestion(
        question=(
            "What reserve does Johnson & Johnson report for resolving claims alleging cancer from "
            "body-powder products?"
        ),
        ticker="JNJ",
        accession_no="0000200406-26-000016",
        section="Notes",
        chunk_index=59,
    ),
    ManualQuestion(
        question=(
            "How do Walmart's inventory costing methods differ across its U.S., international, and "
            "Sam's Club businesses?"
        ),
        ticker="WMT",
        accession_no="0000104169-26-000055",
        section="Notes",
        chunk_index=5,
    ),
    ManualQuestion(
        question=(
            "Why can Berkshire Hathaway's insurance underwriting result swing sharply between "
            "reporting periods?"
        ),
        ticker="BRK-B",
        accession_no="0001193125-26-083899",
        section="MD&A",
        chunk_index=2,
    ),
    ManualQuestion(
        question=(
            "Which project sanctions and license renewals contributed to Chevron's recent reserve "
            "additions in the Americas and Nigeria?"
        ),
        ticker="CVX",
        accession_no="0000093410-26-000078",
        section="Notes",
        chunk_index=43,
    ),
)


def build_retrieval_groundtruth(
    *,
    engine: Engine | None = None,
    settings: Settings | None = None,
    generator: QuestionGenerator | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    seed: int = DEFAULT_SEED,
    synthetic_target: int = DEFAULT_SYNTHETIC_TARGET,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    manual_questions: Sequence[ManualQuestion] = MANUAL_QUESTIONS,
) -> RetrievalGroundTruth:
    """Generate, quality-filter, validate, and persist retrieval ground truth."""
    _validate_build_limits(
        synthetic_target=synthetic_target,
        candidate_count=candidate_count,
        batch_size=batch_size,
    )
    started_at = perf_counter()
    resolved_settings = settings or get_settings()
    model = _select_generation_model(resolved_settings)
    owns_engine = engine is None
    resolved_engine = engine or create_rw_engine(resolved_settings.database_url)
    try:
        corpus = load_source_chunks(resolved_engine)
        manual_pairs = resolve_manual_questions(corpus, manual_questions)
        manual_identities = {question.identity for question in manual_questions}
        eligible_candidates = [chunk for chunk in corpus if chunk.identity not in manual_identities]
        candidates = balanced_sample(
            eligible_candidates,
            count=candidate_count,
            seed=seed,
        )
        provider = generator or OpenAIQuestionGenerator.from_settings(resolved_settings)
        tracer = RequestTracer(settings=resolved_settings)
        generated: list[tuple[SourceChunk, GeneratedQuestion]] = []
        rejection_reasons: Counter[str] = Counter()

        with tracer.trace(
            trace_id=uuid4().hex,
            question="build retrieval ground truth",
            session_id=None,
        ) as generation_trace:
            for offset in range(0, len(candidates), batch_size):
                batch = candidates[offset : offset + batch_size]
                result = provider.generate(batch, model=model)
                mapped, batch_rejections = _match_generated_questions(batch, result.questions)
                generated.extend(mapped)
                rejection_reasons.update(batch_rejections)

        accepted: list[tuple[SourceChunk, GeneratedQuestion, float]] = []
        seen_questions: set[str] = set()
        for chunk, draft in generated:
            accepted_question, reason, overlap = assess_question_quality(chunk, draft)
            normalized_question = " ".join(_words(draft.question))
            if accepted_question and normalized_question in seen_questions:
                accepted_question = False
                reason = "duplicate_question"
            if not accepted_question:
                rejection_reasons[reason] += 1
                continue
            seen_questions.add(normalized_question)
            accepted.append((chunk, draft, overlap))

        retained = balanced_sample_questions(
            accepted,
            count=min(synthetic_target, len(accepted)),
            seed=seed,
        )
        if len(retained) < synthetic_target:
            raise RuntimeError(
                f"Only {len(retained)} synthetic questions passed quality filtering; "
                f"{synthetic_target} are required. Rejections: "
                f"{dict(sorted(rejection_reasons.items()))}."
            )
        synthetic_pairs = [
            _synthetic_pair(index, chunk, draft, overlap)
            for index, (chunk, draft, overlap) in enumerate(retained, start=1)
        ]
        elapsed_seconds = perf_counter() - started_at
        metrics = generation_trace.metrics
        dataset = RetrievalGroundTruth(
            version=GROUNDTRUTH_VERSION,
            seed=seed,
            generated_at=datetime.now(UTC),
            generation_model=model,
            corpus_fingerprint=corpus_fingerprint(corpus),
            eligible_sections=ELIGIBLE_SECTIONS,
            filter_policy={
                "question_words": {"min": 8, "max": 50},
                "issuer_required": True,
                "minimum_specific_content_words": 4,
                "maximum_lexical_overlap": 0.75,
                "maximum_copied_ngram": 4,
                "meta_references_rejected": list(_META_PHRASES),
            },
            stats=GroundTruthBuildStats(
                candidates_sampled=len(candidates),
                generated_questions=len(generated),
                quality_passed=len(accepted),
                retained_synthetic=len(synthetic_pairs),
                rejected_synthetic=len(candidates) - len(accepted),
                manual_questions=len(manual_pairs),
                rejection_reasons=dict(sorted(rejection_reasons.items())),
                input_tokens=metrics.input_tokens,
                output_tokens=metrics.output_tokens,
                cached_tokens=metrics.cached_tokens,
                cost_usd=metrics.cost_usd,
                elapsed_seconds=round(elapsed_seconds, 3),
            ),
            pairs=[*synthetic_pairs, *manual_pairs],
        )
        persist_groundtruth(dataset, output_path)
        return dataset
    finally:
        if owns_engine:
            resolved_engine.dispose()


def load_source_chunks(engine: Engine) -> list[SourceChunk]:
    """Load the complete eligible narrative corpus in stable order."""
    with engine.connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                SELECT
                    dc.id,
                    dc.accession_no,
                    dc.ticker,
                    c.name AS company_name,
                    dc.section,
                    dc.chunk_index,
                    dc.token_count,
                    dc.text,
                    dc.source_url
                FROM doc_chunks AS dc
                JOIN companies AS c ON c.cik = dc.cik
                WHERE dc.section IN ('MD&A', 'Risk Factors', 'Notes')
                ORDER BY
                    dc.ticker,
                    dc.section,
                    dc.accession_no,
                    dc.chunk_index,
                    dc.id
                """
                )
            )
            .mappings()
            .all()
        )
    if not rows:
        raise RuntimeError("No eligible narrative chunks are available for evaluation.")
    return [SourceChunk.model_validate(dict(row)) for row in rows]


def balanced_sample(
    chunks: Sequence[SourceChunk],
    *,
    count: int,
    seed: int,
) -> list[SourceChunk]:
    """Sample round-robin across ticker, section, and source-size strata."""
    if count <= 0:
        raise ValueError("Sample count must be positive.")
    if count > len(chunks):
        raise ValueError("Sample count exceeds the available chunk population.")
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[SourceChunk]] = defaultdict(list)
    for chunk in sorted(chunks, key=_source_sort_key):
        groups[(chunk.ticker, chunk.section, chunk.size_bucket)].append(chunk)
    queues: dict[tuple[str, str, str], deque[SourceChunk]] = {}
    for key, values in groups.items():
        rng.shuffle(values)
        queues[key] = deque(values)
    keys = list(queues)
    rng.shuffle(keys)

    selected: list[SourceChunk] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            queue = queues[key]
            if queue and len(selected) < count:
                selected.append(queue.popleft())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError("Stratified sampling exhausted the corpus unexpectedly.")
    return selected


def balanced_sample_questions(
    questions: Sequence[tuple[SourceChunk, GeneratedQuestion, float]],
    *,
    count: int,
    seed: int,
) -> list[tuple[SourceChunk, GeneratedQuestion, float]]:
    """Retain accepted questions without losing source-stratum diversity."""
    if count > len(questions):
        raise ValueError("Question sample count exceeds accepted questions.")
    rng = random.Random(seed + 1)
    groups: dict[
        tuple[str, str, str],
        list[tuple[SourceChunk, GeneratedQuestion, float]],
    ] = defaultdict(list)
    for item in questions:
        chunk = item[0]
        groups[(chunk.ticker, chunk.section, chunk.size_bucket)].append(item)
    queues: dict[
        tuple[str, str, str],
        deque[tuple[SourceChunk, GeneratedQuestion, float]],
    ] = {}
    for key, values in groups.items():
        rng.shuffle(values)
        queues[key] = deque(values)
    keys = list(queues)
    rng.shuffle(keys)

    selected: list[tuple[SourceChunk, GeneratedQuestion, float]] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            queue = queues[key]
            if queue and len(selected) < count:
                selected.append(queue.popleft())
                progressed = True
        if not progressed:
            break
    return selected


def assess_question_quality(
    chunk: SourceChunk,
    generated: GeneratedQuestion,
) -> tuple[bool, str, float]:
    """Apply deterministic specificity and anti-copying checks."""
    question = generated.question.strip()
    question_words = _words(question)
    if not generated.is_specific:
        return False, "model_marked_generic", 0.0
    if not question.endswith("?"):
        return False, "missing_question_mark", 0.0
    if not 8 <= len(question_words) <= 50:
        return False, "invalid_length", 0.0
    lowered = question.casefold()
    if any(phrase in lowered for phrase in _META_PHRASES):
        return False, "meta_reference", 0.0

    issuer_terms = _issuer_terms(chunk)
    if not issuer_terms.intersection(question_words):
        return False, "issuer_missing", 0.0
    content_words = {
        word
        for word in question_words
        if word not in _STOP_WORDS
        and word not in _GENERIC_CONTENT_WORDS
        and word not in issuer_terms
        and len(word) > 2
    }
    if len(content_words) < 4:
        return False, "too_generic", 0.0

    source_words = _words(chunk.text)
    source_set = set(source_words)
    overlap = len(content_words.intersection(source_set)) / len(content_words)
    if _contains_copied_ngram(question_words, source_words, n=5):
        return False, "copied_phrase", overlap
    if overlap > 0.75:
        return False, "excessive_lexical_overlap", overlap
    return True, "accepted", overlap


def resolve_manual_questions(
    corpus: Sequence[SourceChunk],
    manual_questions: Sequence[ManualQuestion],
) -> list[GroundTruthPair]:
    """Resolve hand-written questions against stable real chunk identities."""
    by_identity = {chunk.identity: chunk for chunk in corpus}
    pairs: list[GroundTruthPair] = []
    for index, manual in enumerate(manual_questions, start=1):
        chunk = by_identity.get(manual.identity)
        if chunk is None:
            raise RuntimeError(
                "Manual ground-truth chunk is missing: "
                f"{manual.accession_no}/{manual.section}/{manual.chunk_index}."
            )
        if chunk.ticker != manual.ticker:
            raise RuntimeError("Manual ground-truth ticker does not match the resolved chunk.")
        content_words = {
            word for word in _words(manual.question) if word not in _STOP_WORDS and len(word) > 2
        }
        source_words = set(_words(chunk.text))
        overlap = (
            len(content_words.intersection(source_words)) / len(content_words)
            if content_words
            else 0.0
        )
        pairs.append(
            GroundTruthPair(
                pair_id=f"manual-{index:03d}",
                question=manual.question,
                expected_chunk_id=chunk.id,
                expected_accession_no=chunk.accession_no,
                expected_section=chunk.section,
                expected_chunk_index=chunk.chunk_index,
                ticker=chunk.ticker,
                source="manual",
                lexical_overlap=round(overlap, 6),
            )
        )
    return pairs


def corpus_fingerprint(corpus: Sequence[SourceChunk]) -> str:
    """Hash stable identities and texts to detect corpus drift."""
    digest = hashlib.sha256()
    for chunk in sorted(corpus, key=_source_sort_key):
        text_digest = hashlib.sha256(chunk.text.encode()).hexdigest()
        digest.update(
            (
                f"{chunk.accession_no}|{chunk.section}|{chunk.chunk_index}|"
                f"{chunk.ticker}|{text_digest}\n"
            ).encode()
        )
    return digest.hexdigest()


def persist_groundtruth(dataset: RetrievalGroundTruth, output_path: Path) -> None:
    """Persist the complete benchmark as stable, human-reviewable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        dataset.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def load_groundtruth(path: Path = DEFAULT_OUTPUT_PATH) -> RetrievalGroundTruth:
    """Load and validate a previously persisted retrieval benchmark."""
    return RetrievalGroundTruth.model_validate_json(path.read_text(encoding="utf-8"))


def _match_generated_questions(
    batch: Sequence[SourceChunk],
    generated: Sequence[GeneratedQuestion],
) -> tuple[list[tuple[SourceChunk, GeneratedQuestion]], Counter[str]]:
    expected = {chunk.id: chunk for chunk in batch}
    matched: list[tuple[SourceChunk, GeneratedQuestion]] = []
    rejections: Counter[str] = Counter()
    seen_ids: set[int] = set()
    for question in generated:
        if question.chunk_id not in expected:
            rejections["unexpected_chunk_id"] += 1
            continue
        if question.chunk_id in seen_ids:
            rejections["duplicate_chunk_id"] += 1
            continue
        seen_ids.add(question.chunk_id)
        matched.append((expected[question.chunk_id], question))
    missing_count = len(expected) - len(seen_ids)
    if missing_count:
        rejections["missing_generation"] += missing_count
    return matched, rejections


def _synthetic_pair(
    index: int,
    chunk: SourceChunk,
    generated: GeneratedQuestion,
    overlap: float,
) -> GroundTruthPair:
    return GroundTruthPair(
        pair_id=f"synthetic-{index:03d}",
        question=generated.question.strip(),
        expected_chunk_id=chunk.id,
        expected_accession_no=chunk.accession_no,
        expected_section=chunk.section,
        expected_chunk_index=chunk.chunk_index,
        ticker=chunk.ticker,
        source="synthetic",
        lexical_overlap=round(overlap, 6),
    )


def _contains_copied_ngram(
    question_words: Sequence[str],
    source_words: Sequence[str],
    *,
    n: int,
) -> bool:
    if len(question_words) < n or len(source_words) < n:
        return False
    source_ngrams = {
        tuple(source_words[offset : offset + n]) for offset in range(len(source_words) - n + 1)
    }
    return any(
        tuple(question_words[offset : offset + n]) in source_ngrams
        for offset in range(len(question_words) - n + 1)
    )


def _issuer_terms(chunk: SourceChunk) -> set[str]:
    terms = set(_words(chunk.ticker))
    terms.update(
        word
        for word in _words(chunk.company_name)
        if word not in _CORPORATE_SUFFIXES and len(word) > 2
    )
    return terms


def _words(value: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_PATTERN.finditer(value)]


def _source_sort_key(chunk: SourceChunk) -> tuple[str, str, str, int, int]:
    return (
        chunk.ticker,
        chunk.section,
        chunk.accession_no,
        chunk.chunk_index,
        chunk.id,
    )


def _select_generation_model(settings: Settings) -> str:
    model = settings.openai_classifier_model.strip() or settings.openai_model.strip()
    if not model:
        raise ValueError("OPENAI_CLASSIFIER_MODEL or OPENAI_MODEL must be configured.")
    return model


def _validate_build_limits(
    *,
    synthetic_target: int,
    candidate_count: int,
    batch_size: int,
) -> None:
    if synthetic_target <= 0:
        raise ValueError("Synthetic target must be positive.")
    if candidate_count < synthetic_target:
        raise ValueError("Candidate count must be at least the synthetic target.")
    if batch_size <= 0:
        raise ValueError("Generation batch size must be positive.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--synthetic-target",
        type=int,
        default=DEFAULT_SYNTHETIC_TARGET,
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=DEFAULT_CANDIDATE_COUNT,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    """Build the configured retrieval benchmark and print a compact report."""
    arguments = _parse_args()
    dataset = build_retrieval_groundtruth(
        output_path=arguments.output,
        seed=arguments.seed,
        synthetic_target=arguments.synthetic_target,
        candidate_count=arguments.candidate_count,
        batch_size=arguments.batch_size,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "pairs": len(dataset.pairs),
                "model": dataset.generation_model,
                "stats": dataset.stats.model_dump(mode="json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
