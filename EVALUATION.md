# Evaluation

Financial Fundamentals Agent evaluates retrieval and answer generation separately so that search quality,
numeric correctness, narrative grounding, and operational failures remain distinguishable. All datasets
use seed `42`, and every run is persisted in PostgreSQL `eval_runs` with its configuration and metrics as
JSONB.

## Methodology

| Evaluation | Dataset | Size | Ground truth |
|---|---|---:|---|
| Retrieval | `evaluation/retrieval_groundtruth.json` | 100 questions | One expected `doc_chunks.id` per question |
| Numeric generation | `evaluation/generation_numeric_gold.json` | 48 questions | Exact typed facts derived from `financial_facts` |
| Narrative generation | `evaluation/generation_narrative_set.json` | 30 questions | Expected SEC filing chunks and sections |

The retrieval set contains 90 LLM-generated analyst-style paraphrases and 10 hand-written controls.
It spans MD&A (35 questions), Notes (27), and Risk Factors (38). Synthetic questions were generated
without copying the source passage: copied phrases, excessive lexical overlap, generic questions, and
missing generations were rejected. From 260 sampled candidates, 257 questions were generated, 163
passed the quality filter, and 90 were retained alongside the 10 manual controls.

The retrieval comparison uses the same 100 questions, corpus, OpenAI
`text-embedding-3-small` query embeddings, no metadata filters, and cutoffs `k=5` and `k=10` for every
configuration. Reranked configurations use `cross-encoder/ms-marco-MiniLM-L6-v2`. The metrics are hit
rate, mean reciprocal rank (MRR), recall, and normalized discounted cumulative gain (NDCG).

The numeric set contains 34 direct fact questions, eight year-over-year comparisons, and six
deliberately unanswerable traps. The narrative set is balanced across MD&A, Notes, and Risk Factors
(10 questions each). Its two prompt variants use the same `vector_rerank` retrieval path and evidence;
RAGAS judging uses the configured `gpt-5.4-nano`, while answer generation uses the configured
`gpt-5.4-mini`.

## Retrieval benchmark

| Configuration | Hit@5 | MRR@5 | Recall@5 | NDCG@5 | Hit@10 | MRR@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `text_search` | 0.380 | 0.242 | 0.380 | 0.277 | 0.440 | 0.250 | 0.440 | 0.296 |
| `vector_search` | 0.750 | 0.546 | 0.750 | 0.597 | 0.870 | 0.563 | 0.870 | 0.637 |
| `hybrid_rrf` | 0.640 | 0.453 | 0.640 | 0.500 | 0.730 | 0.466 | 0.730 | 0.529 |
| `hybrid_rrf_rerank` | 0.680 | 0.535 | 0.680 | 0.572 | 0.730 | 0.542 | 0.730 | 0.589 |
| **`vector_rerank`** | **0.840** | **0.652** | **0.840** | **0.699** | **0.870** | **0.656** | **0.870** | **0.709** |

`vector_rerank` is the clear winner and is the production default. Against the previous production
configuration, `hybrid_rrf_rerank`, it improves Hit@5 by `0.160`, Recall@10 by `0.140`, and NDCG@10 by
`0.120`. On this corpus and paraphrased benchmark, the lexical OR query returns a very broad candidate
set and RRF displaces relevant vector candidates. Additional persisted ablations bounded the lexical
leg before fusion, but none justified retaining hybrid retrieval in production. The lexical and hybrid
implementations remain available for future experiments with a more selective lexical query or weighted
fusion.

## Numeric generation benchmark

| Metric | Persisted result |
|---|---:|
| Answerable questions | 42 |
| No-answer traps | 6 |
| Exact match rate on answerable questions | 0.738 |
| Grounded rate | 0.875 |
| Empty or refusal rate | 0.354 |
| Correct refusal rate on traps | 1.000 |
| Execution error rate | 0.125 (6/48) |
| **False-number rate** | **0.000 (0/48)** |

The critical safety result is a zero false-number rate: the agent did not return an incorrect number as
grounded. All six unanswerable traps were refused correctly. The persisted run did record six execution
errors, concentrated in year-over-year comparisons and cash-equivalent questions, which reduced exact
match rather than producing fabricated values. The subsequent runtime hardening normalized canonical
metric synonyms and explicit fiscal periods and introduced typed, honest error mapping; this table
remains the immutable persisted benchmark snapshot rather than claiming an unmeasured post-fix score.

## Narrative generation benchmark

| Prompt | Faithfulness | Answer relevancy | Context precision | Context recall | Composite | Citation rate | Grounded rate | Execution errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Production** | 0.852 | **0.676** | **0.743** | **0.732** | **0.751** | 0.933 | 0.900 | 0.067 (2/30) |
| Citation-first | **0.859** | 0.639 | 0.733 | 0.716 | 0.737 | 0.933 | 0.900 | 0.067 (2/30) |

The citation-first prompt slightly improves faithfulness (`+0.006`) but loses answer relevancy, context
precision, context recall, and overall composite score. The production prompt therefore remains selected:
its composite score is `0.751` versus `0.737`, with the same citation and grounded rates.

## Reproducibility and run records

- Seed: `42` for all three datasets.
- Retrieval ground-truth generation: 175,348 input tokens, 167,680 cached input tokens, 11,803 output
  tokens, approximately USD `0.01964095`, and 96.635 seconds.
- Persisted numeric run: 89,951 input tokens, 20,996 output tokens, approximately USD `0.13542710`, and
  438.504 seconds.
- Persisted production-prompt narrative run: approximately USD `0.39501711` and 768.146 seconds.
- Persisted citation-first narrative run: approximately USD `0.29344257` and 579.013 seconds.
- Retrieval configurations, lexical-leg ablations, numeric metrics, narrative metrics, model names,
  corpus fingerprints, durations, token usage, and costs remain queryable in `eval_runs`.

The JSON datasets are versioned so the same questions can be rerun, while `eval_runs` preserves the
database-side history of configurations and results.
