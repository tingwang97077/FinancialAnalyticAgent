# Financial Fundamentals Agent

## Architecture

### Design principles

Financial Fundamentals Agent (`ffa`) separates numeric evidence from narrative evidence:

- **Numeric questions** are answered from canonical SEC XBRL facts through generated, validated,
  read-only PostgreSQL queries. PostgreSQL performs every calculation; the LLM never calculates or
  estimates a financial number.
- **Narrative questions** are answered from retrieved excerpts of SEC filings. Every sourced
  narrative claim must resolve to a retrieved chunk and its SEC URL.
- **Hybrid questions** combine typed SQL facts with cited filing excerpts.
- **Typed boundaries** use Pydantic models and typed dictionaries between understanding, tools,
  generation, API, and evaluation layers.
- **Fail-closed validation** blocks unsafe input, unauthorized SQL, unsupported numbers, invalid
  citations, and prompt disclosure.
- **Traceability** propagates one `trace_id` through the API, agent, Langfuse spans, `query_logs`,
  and user feedback.

The application targets Python 3.12 and uses `uv.lock` as the reproducible dependency source. Apache
Airflow has a separate dependency environment and image.

### System overview

#### System architecture

```mermaid
flowchart LR
    User["Financial analyst"] --> UI["Streamlit UI"]
    UI -->|"POST /ask and /feedback"| API["FastAPI API"]
    API --> Agent["Grounded agent"]
    Agent --> SQLTool["Validated SQL tool"]
    Agent --> Retrieval["Vector retrieval and reranker"]
    Agent --> OpenAI["OpenAI APIs"]
    SQLTool --> PG[("PostgreSQL 17 and pgvector")]
    Retrieval --> PG
    API --> PG
    API -. "optional traces" .-> Langfuse["Langfuse Cloud"]

    Airflow["Airflow 3 components"] --> DAGs["Thin ingestion DAGs"]
    DAGs --> Ingestion["ffa ingestion package"]
    Ingestion --> SEC["SEC EDGAR"]
    Ingestion --> OpenAI
    Ingestion --> PG

    Grafana["Grafana"] --> PG
```

The UI is an HTTP client of the API and contains no agent logic. FastAPI owns the request lifecycle,
agent invocation, query logging, feedback persistence, health checks, tracing, and process-local
reranker warm-up.

The agent uses OpenAI structured outputs for question understanding, SQL generation, tool planning,
and final answer generation. Local application code retains control of routing constraints, SQL
authorization, database execution, numeric grounding, citation validation, and error mapping.

Airflow orchestrates the ingestion functions implemented under `src/ffa/ingestion/`; business logic
does not live in the DAG files. PostgreSQL is the shared store for source data, retrieval documents,
operational telemetry, feedback, and evaluation runs.

### Data ingestion

#### Data ingestion flow

```mermaid
flowchart TD
    Airflow["Weekly Airflow Task SDK DAGs"] --> Structured["Structured ingestion"]
    Airflow --> Unstructured["Unstructured ingestion"]

    SEC["SEC EDGAR"] --> SecClient["SecEdgarClient"]
    SecClient --> CompanyFacts["Companyfacts JSON"]
    CompanyFacts --> Normalize["Explicit XBRL normalization"]
    Normalize --> Restatements["Fiscal alignment and latest restatement selection"]
    Restatements --> FactLoad["Transactional fact UPSERT"]
    FactLoad --> Facts[("financial_facts")]
    FactLoad --> State[("ingestion_state")]

    SecClient --> Submissions["Submissions and filing documents"]
    Submissions --> Discovery["Incremental filing discovery"]
    Discovery --> Clean["XHTML parsing and narrative cleaning"]
    Clean --> Sections["MD&A, Risk Factors, and Notes"]
    Sections --> Chunk["Boundary-aware chunking with overlap"]
    Chunk --> Embed["OpenAI embeddings"]
    Embed --> ChunkLoad["Atomic delete then insert by accession"]
    ChunkLoad --> Chunks[("doc_chunks")]
    ChunkLoad --> State
```

`SecEdgarClient` applies an identifying user agent, an SEC-compatible request-rate limit, retries,
allowed-host validation, and a persistent disk cache. Scheduled DAG fetches bypass selected cached
metadata so new filings remain discoverable.

The structured branch maps reviewed XBRL tags to five canonical metrics: `revenue`, `net_income`,
`total_assets`, `total_liabilities`, and `cash_and_equivalents`. Unknown tags are logged and ignored
rather than inferred. The normalizer preserves SEC-declared `fy`, `fp`, `start`, and `end`, stores raw
values and units, distinguishes instant from duration facts, and selects the most recently filed
restatement. Loading is incremental and transactional, using `ON CONFLICT DO UPDATE` while advancing
`SEC_EDGAR_STRUCTURED` state.

The unstructured branch discovers supported 10-K, 10-K/A, 10-Q, and 10-Q/A filings, including SEC
submission history. Missing primary documents and issuers without an exploitable 10-K are reported as
skips. Filing fiscal metadata is joined from structured facts by accession rather than inferred from
calendar dates.

The cleaner selects XML parsing for XHTML/inline-XBRL, removes hidden content, repeated pagination
boilerplate, and large numeric tables, then extracts canonical narrative sections with bounded,
fail-closed repairs for repeated headings and incorporation references. Chunking targets
`CHUNK_MAX_TOKENS`, prefers sentence or line boundaries, never splits words, and preserves configurable
overlap. Embedded chunks are replaced per accession in one transaction, preventing stale chunks from
earlier segmentations. The DAGs publish the `facts_ready` and `chunks_ready` Airflow Assets.

### Storage

| Component | Purpose | Key properties |
|---|---|---|
| `companies` | SEC issuer identity | Primary key `cik`; unique normalized ticker |
| `filings` | Filing provenance | One row per accession with form, dates, and primary SEC URL |
| `financial_facts` | Canonical numeric evidence | Raw value and unit, taxonomy tag, fiscal context, filing provenance, indexed financial lookup |
| `doc_chunks` | Narrative retrieval corpus | Section metadata, source URL, generated English `tsvector`, and `vector(1536)` embedding |
| `ingestion_state` | Incremental ingestion cursors | Separate state per source and CIK |
| `query_logs` | Operational request telemetry | Trace, session, question, route, intent, grounding, latency, tokens, and cost |
| `feedback` | Explicit user ratings | Positive or negative rating and optional comment associated with a `trace_id` |
| `eval_runs` | Reproducible evaluation history | Evaluation type plus JSONB configuration and metrics |

`doc_chunks` has an HNSW index using `vector_cosine_ops`, queried with pgvector's cosine-distance
operator `<=>`. Its generated `text_tsv` column retains a GIN index for lexical experiments. Chunk
identity is unique across `(accession_no, section, chunk_index)`.

The configured corpus deliberately excludes XOM, WFC, and TSLA because they do not currently provide
both usable structured facts and safely segmented narrative sections. CVX retains structured facts,
Notes, and Risk Factors, but its MD&A is unavailable in the current corpus.

### Retrieval

The default online path is:

`rewritten query → OpenAI embedding → VectorSearchIndex → top-k chunks → local CrossEncoderReranker → top-n chunks`

`VectorSearchIndex` performs cosine search through pgvector and sets `hnsw.ef_search` for each search
connection. The API preloads `cross-encoder/ms-marco-MiniLM-L6-v2` once per worker during startup, with
a bounded timeout and lazy-loading fallback.

`RETRIEVAL_STRATEGY` supports four runtime configurations:

- `vector_rerank` — vector search followed by cross-encoder reranking; production default.
- `hybrid_rerank` — text/vector Reciprocal Rank Fusion followed by reranking.
- `vector` — vector search without cross-encoder reranking.
- `hybrid` — text/vector RRF without reranking.

`TextSearchIndex` uses the GIN-backed `text_tsv` column and `ts_rank_cd`. `HybridSearchIndex` fuses
ranks rather than incomparable lexical and cosine scores. Both remain available for controlled
evaluation.

Ticker, fiscal year, fiscal period, and section filters are translated into parameterized SQL
predicates. Lists become `IN (...)` clauses; filtering is never deferred to Python. A hybrid request
that finds no narrative evidence with numeric period filters may retry without only those period
filters while retaining company and section constraints. Query embeddings are prepared once and reused
across this fallback.

The frozen 100-question benchmark selected `vector_rerank`: it recorded Hit@5 `0.840`, Recall@10
`0.870`, and NDCG@10 `0.709`, compared with `0.680`, `0.730`, and `0.589` for hybrid RRF plus
reranking.

> Repository consistency note: runtime configuration in `Settings` and `RetrievalPipeline` defaults to
> `vector_rerank`. `eval_retrieval.py` still contains the legacy label
> `PRODUCTION_APPROACH = "hybrid_rrf_rerank"`, and the router's tool description retains legacy hybrid
> wording. Neither label changes the runtime retrieval implementation.

### Agent and question routing

#### Online question flow

```mermaid
flowchart TD
    User["User question"] --> UI["Streamlit chat"]
    UI --> Ask["FastAPI POST /ask"]
    Ask --> Guard["Input guardrails"]
    Guard -->|"blocked"| Blocked["Generic safe refusal"]
    Guard -->|"allowed"| Understand["Structured Understanding"]
    Understand --> Router["Intent-bound function-call planner"]

    Router -->|"numeric"| SQL["SQL tool"]
    Router -->|"narrative"| Retrieve["Retrieval tool"]
    Router -->|"hybrid"| SQL
    Router -->|"hybrid"| Retrieve
    Router -->|"out_of_scope"| Scope["Scope refusal"]

    SQL --> Context["AgentContext facts and chunks"]
    Retrieve --> Context
    Context --> Generate["Structured Answer generation"]
    Generate --> Validate["Numeric grounding and citation validation"]

    Blocked --> Log["Persist query log and trace"]
    Scope --> Log
    Validate --> Log
    Log --> Response["AskResponse with trace_id"]
    Response --> Display["Streamlit answer"]
    Display -. "optional rating" .-> Feedback["FastAPI POST /feedback"]
    Feedback --> FeedbackTable[("feedback")]
```

`Understanding` contains an `Intent`, normalized `Entities`, and a self-contained `rewritten_query`.
`Entities` carries tickers, resolved SEC CIKs, canonical metrics, fiscal years, fiscal periods, and
optional canonical sections. Unknown ticker resolution becomes `out_of_scope` instead of propagating
an unsafe identifier.

`AgentContext` carries typed `NumberFact` rows, retrieved `Chunk` objects, the selected route, `trace_id`,
and a data-unavailable flag. `AgentRun` returns both this evidence context and the public `Answer`. An
`Answer` contains text, exact `NumberFact` objects, validated `Citation` objects, and a `grounded` flag.

| Intent | Tool path | Expected evidence |
|---|---|---|
| `numeric` | `sql_tool` | Typed facts computed or selected by PostgreSQL |
| `narrative` | `retrieval_tool` | Reranked SEC filing chunks with source URLs |
| `hybrid` | Both tools | SQL facts plus cited narrative chunks |
| `out_of_scope` | No tool | Scope refusal |
| Blocked input | No understanding or tool call | Generic safety refusal |

### Grounding and security

Input protection combines OpenAI moderation with local prompt-injection detection. The heuristics
inspect English and French instruction-hierarchy attacks, Unicode normalization, URL/HTML decoding,
simple Base64 fragments, leetspeak, and invisible formatting characters. Clear attacks are blocked
before downstream processing; moderation failure also fails closed. Generation adds a separate
disclosure barrier that detects protected-instruction patterns and verbatim prompt fragments.

For numeric requests, an LLM proposes one structured SQL payload against a fixed schema containing only
`companies`, `filings`, and `financial_facts`. `validate_sql()` uses `sqlglot` to enforce:

- one PostgreSQL root `SELECT`;
- strict table, column, and function allow-lists;
- no wildcard, DDL, DML, commands, multiple statements, system catalogs, or unauthorized sources;
- exactly `metric`, `fiscal_year`, `fiscal_period`, `value`, and `unit` in the result;
- a database-derived `value`;
- a maximum top-level limit of 100 rows.

Execution exclusively uses `DATABASE_URL_READONLY`, whose URL must differ from `DATABASE_URL`.
PostgreSQL additionally receives `SET TRANSACTION READ ONLY` and a five-second statement timeout.
Comparisons require SQL-derived delta and percentage `NumberFact` rows. Missing required comparison rows
are rejected.

Generation may only copy supplied facts into `Answer.numbers`. Local validation removes any number that
does not exactly match metric, fiscal year, fiscal period, value, and unit from the SQL result. Citations
must resolve to retrieved chunk URL, section, and accession metadata. Missing or invalid evidence makes
the answer ungrounded and retracts unsupported prose. Missing canonical data returns an honest
data-unavailable response rather than an infrastructure error.

### API and user interface

FastAPI exposes:

| Endpoint | Responsibility |
|---|---|
| `POST /ask` | Run the existing end-to-end agent, persist telemetry, and return `Answer` plus `trace_id` |
| `POST /feedback` | Store a `+1` or `-1` rating and optional comment for a trace |
| `GET /healthz` | Verify API liveness and PostgreSQL connectivity |

The API maps upstream model failures, generated-SQL failures, database failures, invalid configuration,
and unexpected errors to bounded HTTP responses without exposing stack traces or secrets.

Streamlit calls these endpoints with `httpx`. It keeps conversation and feedback state in
`st.session_state`, formats raw API numbers only for display, presents citations as SEC links, and shows
an explicit warning for `grounded=false`. Its Dashboard tab links to the provisioned Grafana dashboard.

### Observability

`RequestTracer` creates one `ffa.ask` trace and timed spans for `guardrails.check_input`, `understand`,
`router`, `sql_tool`, `retrieval_tool`, and `generation` when applicable. The public `trace_id` is
propagated into OpenAI metadata and Langfuse correlation metadata.

OpenAI usage is normalized across response and embedding APIs. Input, output, and cached-input tokens
are aggregated, and costs are calculated from environment-provided prices for the actual configured
model. `query_logs` stores total latency, tokens, cached tokens, cost, route, intent, and grounding.
Missing or invalid Langfuse credentials disable only remote export; local metrics and request processing
continue normally.

Grafana provisioning is entirely code-driven. Its PostgreSQL datasource uses `postgres:5432` and the
`ffa_ro` role. The `FFA Overview` dashboard contains request volume, average and p95 latency, cumulative
cost, intent distribution, grounded-response rate, and positive-feedback rate panels.

### Evaluation

Retrieval evaluation uses `evaluation/retrieval_groundtruth.json`, a seed-42 set of 100 questions: 90
quality-filtered LLM paraphrases and 10 hand-written controls. Quality filtering rejects generic
questions, copied phrases, excessive lexical overlap, and missing generation. The same unfiltered
question set and cutoffs `k=5` and `k=10` compare text search, vector search, hybrid RRF, hybrid RRF plus
reranking, and vector search plus reranking using hit rate, MRR, recall, and NDCG.

Generation evaluation has two complementary datasets:

- 48 numeric cases: 34 direct facts, eight YoY comparisons, and six no-answer traps derived from
  `financial_facts`.
- 30 section-balanced narrative cases evaluated with RAGAS faithfulness, answer relevancy, context
  precision, and context recall.

The narrative benchmark compares the production prompt with a citation-first alternative under the
same retrieved context. The recorded benchmark retained the production prompt. Numeric scoring
separately tracks exact fact matches, grounding, empty/refused answers, correct trap refusals, execution
errors, and the critical false-number rate. Each configuration is persisted as an `eval_runs` row with
its reproducibility metadata and metrics.

### Container topology

| Service | Internal dependency | Published port | Responsibility |
|---|---|---:|---|
| `postgres` | None | 5432 | PostgreSQL 17, pgvector, schema initialization, application and read-only roles |
| `api` | Healthy `postgres` | 8000 | FastAPI, agent execution, request telemetry, reranker preload |
| `ui` | Healthy `api` | 8501 | Streamlit HTTP client |
| `grafana` | Healthy `postgres` | 3000 | Provisioned operational dashboard |
| `airflow-init` | Healthy `postgres` | — | Metadata migration and administrator creation |
| `airflow-apiserver` | `postgres`, completed `airflow-init` | 8080 | Airflow 3 API and execution endpoint |
| `airflow-scheduler` | `postgres`, completed `airflow-init` | — | DAG scheduling with `LocalExecutor` |
| `airflow-dag-processor` | `postgres`, completed `airflow-init` | — | Independent DAG parsing |
| `airflow-triggerer` | `postgres`, completed `airflow-init` | — | Deferred-task trigger processing |

API and UI share `ffa-app:local`, built from `python:3.12-slim` by `docker/app.Dockerfile` with
`uv sync --frozen --no-dev`. Airflow uses its dedicated `apache/airflow:3.3.0-python3.12` image and
installs `ffa` separately; Airflow is not part of the application environment.

Compose overrides host-local URLs with `postgres:5432`, `http://api:8000`, and `/data/sec_cache`. It
mounts `./dags` read-only into every Airflow component. Named volumes preserve PostgreSQL data, Grafana
state, Airflow logs, and the shared SEC cache. Health checks and conditional `depends_on` relationships
enforce startup ordering.

### Key configuration

| Variable | Used by | Purpose |
|---|---|---|
| `DATABASE_URL` | API, ingestion, retrieval, evaluation | Read-write PostgreSQL connection |
| `DATABASE_URL_READONLY` | SQL tool | Dedicated read-only numeric-query connection |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | PostgreSQL and Compose | Database bootstrap identity |
| `GRAFANA_DATABASE_PASSWORD` | `ffa_ro`, Grafana, SQL tool URL | Password for the read-only PostgreSQL role |
| `OPENAI_API_KEY` | Agent, embeddings, evaluation | OpenAI authentication |
| `OPENAI_MODEL` | SQL planning, routing, generation | Primary configured model |
| `OPENAI_CLASSIFIER_MODEL` | Understanding and evaluation | Lower-cost classifier, with fallback to `OPENAI_MODEL` |
| `OPENAI_EMBEDDING_MODEL` | Ingestion and retrieval | Shared document/query embedding model |
| `RETRIEVAL_STRATEGY` | Retrieval pipeline | Select `vector_rerank`, `hybrid_rerank`, `vector`, or `hybrid` |
| `RETRIEVAL_TOP_K`, `RERANK_TOP_N` | Retrieval pipeline | Candidate and final result counts |
| `EMBEDDING_DIM` | Ingestion and retrieval | Expected embedding width |
| `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP` | Chunker | Chunk size and overlap policy |
| `SEC_USER_AGENT`, `SEC_MAX_RPS`, `SEC_CACHE_DIR` | SEC client | Identification, throttling, and persistent cache |
| `TICKER_UNIVERSE` | Entity resolution and DAGs | Ordered ingestion universe |
| `LANGFUSE_HOST`, `LANGFUSE_BASE_URL` | Tracing | Langfuse endpoint; host takes precedence |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | Tracing | Optional Langfuse credentials |
| `PRICE_MINI_INPUT`, `PRICE_MINI_CACHED_INPUT`, `PRICE_MINI_OUTPUT` | Metrics | Primary-model prices per million tokens |
| `PRICE_NANO_INPUT`, `PRICE_NANO_CACHED_INPUT`, `PRICE_NANO_OUTPUT` | Metrics | Classifier-model prices per million tokens |
| `PRICE_EMBEDDING` | Metrics | Embedding price per million input tokens |
| `API_BASE_URL`, `GRAFANA_BASE_URL` | Streamlit | Backend and dashboard locations |
| `GF_SECURITY_ADMIN_USER`, `GF_SECURITY_ADMIN_PASSWORD` | Grafana | Dashboard administrator credentials |
| `AIRFLOW__CORE__EXECUTOR`, `AIRFLOW__CORE__LOAD_EXAMPLES` | Airflow | Executor and example-DAG configuration |
| `AIRFLOW_ADMIN_USERNAME`, `AIRFLOW_ADMIN_PASSWORD` | Airflow init | Initial administrator |
| `AIRFLOW_FERNET_KEY`, `AIRFLOW_API_JWT_SECRET`, `AIRFLOW_UID` | Airflow | Encryption, API signing, and container ownership |

### Local and Docker execution

The existing [Run](#run) section documents both supported modes. Local development runs PostgreSQL
through Compose while API and UI run through `uv`; all service URLs use `localhost`. Full-stack mode
uses `docker compose up --build`, internal service names, health checks, and named persistent volumes.

## Run

### Full stack with Docker Compose

Prerequisites:

- Docker Compose v2 with at least 4 GB of memory available to Airflow.
- Copy `.env.example` to `.env` and configure the required secrets:
  `OPENAI_API_KEY`, `SEC_USER_AGENT`, `POSTGRES_PASSWORD`,
  `GRAFANA_DATABASE_PASSWORD`, `GF_SECURITY_ADMIN_PASSWORD`,
  `AIRFLOW_ADMIN_PASSWORD`, `AIRFLOW_FERNET_KEY`, and
  `AIRFLOW_API_JWT_SECRET`.
- Keep password values URL-safe because they are interpolated into PostgreSQL
  connection URLs by Docker Compose.
- Generate an Airflow Fernet key with:

  ```bash
  uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  ```

- Generate the Airflow API JWT secret with:

  ```bash
  openssl rand -hex 32
  ```

Build and start PostgreSQL, the API, UI, Airflow 3, and Grafana with one command:

```bash
docker compose up --build
```

The one-shot `airflow-init` service migrates the Airflow metadata tables and creates
the configured administrator before the long-running Airflow components start.
Docker Compose overrides the local connection settings with container-network
addresses:

- PostgreSQL: `postgres:5432`
- UI to API: `http://api:8000`
- SEC cache: `/data/sec_cache`

The complete stack is available at:

| Component | URL |
|---|---|
| FastAPI | <http://localhost:8000> |
| FastAPI documentation | <http://localhost:8000/docs> |
| Streamlit | <http://localhost:8501> |
| Grafana | <http://localhost:3000> |
| Airflow 3 | <http://localhost:8080> |

Use `GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD` for Grafana and
`AIRFLOW_ADMIN_USERNAME` / `AIRFLOW_ADMIN_PASSWORD` for Airflow. Stop the stack
without deleting its named volumes with:

```bash
docker compose down
```

### Local development

Prerequisites:

- Create `.env` from `.env.example` and configure `OPENAI_API_KEY`.
- Configure `GF_SECURITY_ADMIN_PASSWORD` and set `GRAFANA_DATABASE_PASSWORD` to the
  password of the read-only `ffa_ro` database role.
- When running the API and UI directly on the host, set the database hosts in `.env` to
  `localhost`. Keep the URL passwords aligned with `POSTGRES_PASSWORD` and
  `GRAFANA_DATABASE_PASSWORD`, for example:

  ```dotenv
  DATABASE_URL=postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}
  DATABASE_URL_READONLY=postgresql+psycopg://ffa_ro:${GRAFANA_DATABASE_PASSWORD}@localhost:5432/${POSTGRES_DB}
  ```

- Start PostgreSQL before starting the API or UI:

  ```bash
  docker compose up -d postgres
  ```

Start the FastAPI backend with automatic reload:

```bash
make api
```

The API listens on `http://localhost:8000`. Its interactive OpenAPI documentation is available at
<http://localhost:8000/docs>.

Once the Streamlit application is available, start it with:

```bash
make ui
```

Start the provisioned Grafana service with:

```bash
docker compose up -d grafana
```

Grafana is available at <http://localhost:3000>. Sign in with `GF_SECURITY_ADMIN_USER`
and `GF_SECURITY_ADMIN_PASSWORD` from `.env`; the **FFA Overview** dashboard and its
PostgreSQL datasource are provisioned automatically.

## Manual UI corner-case check

This visual check complements the automated API and UI tests. Start PostgreSQL, the API, and the UI,
then open <http://localhost:8501> and verify:

1. Before the first answer, no feedback form or empty feedback request is present.
2. Stop the API, submit a question, and confirm that the chat displays an API-unreachable message
   without losing the earlier conversation or crashing the Streamlit process.
3. Display numeric answers and confirm that negative values, zero, percentages, and share counts are
   formatted for display while the API response remains unchanged.
4. Display a `grounded=false` response and confirm that a red **Not grounded** badge and warning appear
   before the answer text and facts.
5. Submit feedback containing accents, quotes, angle brackets, an ampersand, a newline, and an emoji;
   confirm that success is shown once and the feedback buttons are no longer offered.
6. Confirm that citations open their SEC `source_url` and display their section and accession number.
7. Confirm that the Dashboard tab links to the provisioned Grafana instance.
