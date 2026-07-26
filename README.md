# Financial Fundamentals Analytic Agent

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![OPENAI](https://img.shields.io/badge/Provider-OpenAI-yellow)
![OPENAI](https://img.shields.io/badge/LLM-GPT-orange)
![AGenticRAG](https://img.shields.io/badge/AgenticRAG-Production--Ready-purple)
![Monitoring](https://img.shields.io/badge/Monitoring-Grafana-success)
![Observability](https://img.shields.io/badge/Observability-Langfuse--Cloud-red)
![UI](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit">)
![IaC](https://img.shields.io/badge/IaC-Terraform-%23844FBA?logo=terraform&logoColor=white)
![Orchestration](https://img.shields.io/badge/Orchestration-Airflow-%23017CEE?logo=apacheairflow&logoColor=white)
![Runtime](https://img.shields.io/badge/Runtime-Docker%20Compose-%232496ED?logo=docker&logoColor=white)
![Data Ingestion](https://img.shields.io/badge/Data%20Ingestion-Python-%233776AB?logo=python&logoColor=white)

</p>

## Project overview

Financial Fundamentals Agent is an AI-powered research assistant for asking natural-language questions
about the financial fundamentals and SEC disclosures of a curated universe of public companies. Numeric
answers come from validated, read-only SQL over normalized XBRL facts, narrative answers come from cited
SEC filing excerpts selected by vector retrieval and reranking, and hybrid answers combine both paths.
The LLM understands, routes, and explains evidence but never calculates a financial number; unsupported
claims are retracted or marked unavailable. The repository covers ingestion, retrieval, agent
orchestration, FastAPI, Streamlit, monitoring, evaluation, and Airflow scheduling. It is neither a
general-purpose chatbot nor an investment-advice engine: its governing principle is **evidence before
fluency**.

## Demo

### Streamlit UI
![alt text](images/streamlit_ui.png)

### Monitoring Grafana
![alt text](images/grafana.png)

### Observability Langfuse
![alt text](images/langfuse.png)


## Data universe

The corpus covers 27 curated US public companies across technology, banking, payments, healthcare,
consumer goods, retail, media, industrials, energy, and a diversified holding company. SEC EDGAR is the
only source: the pipeline ingests `10-K`, `10-K/A`, `10-Q`, and `10-Q/A` filings as normalized XBRL facts
and narrative MD&A, Risk Factors, and Notes sections. XOM, WFC, and TSLA are excluded because their
current filings cannot provide both usable structured facts and safely segmented narrative evidence.
SEC data is not distributed with the repository; follow the ingestion step in
[Getting started](#getting-started) to build the corpus.

## Architecture at a glance

- Numeric questions use canonical XBRL facts and strictly validated read-only PostgreSQL.
- Narrative questions use SEC excerpts selected by pgvector search and a local cross-encoder reranker.
- Hybrid questions combine typed SQL facts with cited filing evidence.
- Pydantic boundaries and fail-closed validation protect routing, SQL, numbers, and citations.
- One `trace_id` connects API responses, Langfuse spans, query logs, and feedback.
- OpenAI is the supported LLM provider; PostgreSQL 17, pgvector, Airflow 3, FastAPI, Streamlit, and
  Grafana form the surrounding platform.

### System architecture

```mermaid
flowchart LR
    User["User"] --> UI["Streamlit UI"]
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

### Data ingestion flow

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

### Online question flow

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

→ Full architecture details: see [ARCHITECTURE.md](ARCHITECTURE.md).

→ Evaluation results: see [EVALUATION.md](EVALUATION.md).

## Getting started

SEC data is not bundled with the repository. A successful installation therefore includes both database
initialization and ingestion before the agent is started.

### Prerequisites

Both installation paths require Docker Engine or Docker Desktop with Docker Compose v2. Allocate at
least 4 GB of memory if Airflow will run. Local development additionally requires
[uv](https://docs.astral.sh/uv/) and Python 3.12.

Choose one path:

- [Option A — Full Docker stack](#full-docker-stack): one containerized stack, recommended for
  evaluation.
- [Option B — Local development](#local-development): PostgreSQL and Grafana in Docker, with
  the API and UI running through `uv`.

<a id="full-docker-stack"></a>

### Option A — Full Docker stack

#### Step A1 — Clone the repository

```bash
git clone https://github.com/tingwang97077/FinancialAnalyticAgent.git
cd FinancialAnalyticAgent
cp .env.example .env
```

#### Step A2 — Configure `.env`

Replace every required placeholder. At minimum, configure `OPENAI_API_KEY`, `OPENAI_MODEL`,
`OPENAI_CLASSIFIER_MODEL`, `OPENAI_EMBEDDING_MODEL`, `POSTGRES_USER`, `POSTGRES_PASSWORD`,
`POSTGRES_DB`, `GRAFANA_DATABASE_PASSWORD`, `DATABASE_URL`, `DATABASE_URL_READONLY`,
`SEC_USER_AGENT`, `GF_SECURITY_ADMIN_PASSWORD`, `AIRFLOW_ADMIN_USERNAME`, and
`AIRFLOW_ADMIN_PASSWORD`. Keep passwords URL-safe because Compose interpolates them into PostgreSQL
connection URLs.

`SEC_USER_AGENT` must identify the application and a monitored contact email:

```dotenv
SEC_USER_AGENT=FinancialFundamentalsAgent you@example.com
```

`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` are optional; clear their placeholders
when Langfuse is not used. See [Key configuration](ARCHITECTURE.md#key-configuration) for all variables.
Only OpenAI models and APIs are supported.

#### Step A3 — Generate Airflow secrets

Generate a Fernet key with the published Airflow image:

```bash
docker run --rm --entrypoint python louvuol/ffa-airflow:3.3.0 \
  -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Generate the API JWT secret:

```bash
openssl rand -hex 32
```

Copy the outputs into `AIRFLOW_FERNET_KEY` and `AIRFLOW_API_JWT_SECRET` in `.env`.

#### Step A4 — Initialize PostgreSQL

```bash
docker compose up -d postgres
docker compose exec postgres sh -lc 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

On the first start, PostgreSQL executes `sql/*.sql` to install pgvector, create the schema, and configure
the application and read-only roles.

#### Step A5 — Ingest SEC data

This step is mandatory. Without it, `financial_facts` and `doc_chunks` are empty and the agent cannot
answer grounded questions.

```bash
docker compose run --rm --build api python -m ffa.ingestion.run
```

The incremental runner downloads SEC companyfacts, normalizes and loads XBRL facts, discovers filings,
cleans and chunks narrative sections, creates OpenAI embeddings, and loads the configured
`TICKER_UNIVERSE`. Its JSON summary reports facts, filings, chunks, skipped issuers, and failures.

A measured latest-annual-filing build of the approximately 30-company development corpus processed about
1.63 million embedding tokens in roughly two minutes and cost approximately USD 0.033 at the configured
`text-embedding-3-small` price. A fresh complete filing history can take tens of minutes to several hours
and costs more; actual duration and cost depend on filing volume, network latency, cache state, and the
configured embedding price.

#### Step A6 — Start the complete stack

The project images are also published on Docker Hub:

| Image | Purpose |
|---|---|
| [louvuol/**ffa-app**](https://hub.docker.com/repository/docker/louvuol/ffa-app) | Shared FastAPI and Streamlit application image |
| [louvuol/**ffa-airflow**](https://hub.docker.com/repository/docker/louvuol/ffa-airflow) | Airflow 3 image containing the `ffa` package used by the ingestion DAGs |

```bash
docker pull louvuol/ffa-app:1.0
docker pull louvuol/ffa-airflow:3.3.0
docker compose up --build
```

The published images target `linux/amd64` only. Apple Silicon systems (M1–M4) and some Windows ARM
machines run them through emulation, which is slower, particularly for PyTorch. On ARM64,
`docker compose up --build` is preferred because it builds native images for the host.

The one-shot `airflow-init` service migrates Airflow metadata and creates the configured administrator.
Compose replaces host-local settings with internal addresses:

- PostgreSQL: `postgres:5432`
- UI to API: `http://api:8000`
- SEC cache: `/data/sec_cache`

#### Step A7 — Verify and use the stack

```bash
curl --fail http://localhost:8000/healthz
```

| Component | URL |
|---|---|
| FastAPI | <http://localhost:8000> |
| FastAPI documentation | <http://localhost:8000/docs> |
| Streamlit | <http://localhost:8501> |
| Grafana | <http://localhost:3000> |
| Airflow 3 | <http://localhost:8080> |

Use `GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD` for Grafana and
`AIRFLOW_ADMIN_USERNAME` / `AIRFLOW_ADMIN_PASSWORD` for Airflow. Open Streamlit and ask:

> What was Apple's net income in FY2024?

The response should contain a SQL-backed `NumberFact` and display a green **Grounded** status. If it does
not, rerun Step A5 and inspect the ingestion JSON summary.

#### Step A8 — Refresh data or stop the stack

For later refreshes, use the Airflow UI or trigger the structured DAG first, wait for success, and then
trigger the unstructured DAG:

```bash
docker compose exec airflow-apiserver airflow dags trigger structured_ingestion_dag
docker compose exec airflow-apiserver airflow dags trigger unstructured_ingestion_dag
```

Stop the stack without deleting named volumes:

```bash
docker compose down
```

<a id="local-development"></a>

### Option B — Local development

#### Step B1 — Clone and install

```bash
git clone https://github.com/tingwang97077/FinancialAnalyticAgent.git
cd FinancialAnalyticAgent
cp .env.example .env
uv sync --frozen
```

#### Step B2 — Configure `.env` for localhost

Replace every required placeholder. Configuring only `OPENAI_API_KEY` is insufficient. Configure
`OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_CLASSIFIER_MODEL`, `OPENAI_EMBEDDING_MODEL`,
`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `GRAFANA_DATABASE_PASSWORD`, `SEC_USER_AGENT`,
`GF_SECURITY_ADMIN_PASSWORD`, and the Airflow credentials used by Compose.

Both database URLs must use `localhost` when the ingestion runner, API, and UI execute on the host. Keep
their passwords aligned with `POSTGRES_PASSWORD` and `GRAFANA_DATABASE_PASSWORD`:

```dotenv
DATABASE_URL=postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}
DATABASE_URL_READONLY=postgresql+psycopg://ffa_ro:${GRAFANA_DATABASE_PASSWORD}@localhost:5432/${POSTGRES_DB}
SEC_USER_AGENT=FinancialFundamentalsAgent you@example.com
```

Keep passwords URL-safe. Clear the optional Langfuse placeholders when Langfuse is not used, and consult
[Key configuration](ARCHITECTURE.md#key-configuration) for the complete list. Only OpenAI models and
APIs are supported.

#### Step B3 — Generate Airflow secrets when Airflow is used

```bash
uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
openssl rand -hex 32
```

Store the outputs in `AIRFLOW_FERNET_KEY` and `AIRFLOW_API_JWT_SECRET`. The Compose file validates its
required substitutions, so keep these values non-empty before using Compose services.

#### Step B4 — Initialize PostgreSQL

```bash
docker compose up -d postgres
docker compose exec postgres sh -lc 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

The first start installs pgvector, creates the schema, and configures the application and read-only
roles from `sql/*.sql`.

#### Step B5 — Ingest SEC data

Ingestion is mandatory before the agent can answer questions:

```bash
make ingest
```

This executes the same incremental structured and unstructured pipelines described in Option A. It calls
SEC EDGAR and the OpenAI embeddings API and emits a JSON summary. Runtime and cost depend on filing
volume; the measured latest-annual-filing reference processed 1.63 million tokens in roughly two minutes
for approximately USD 0.033, while a fresh complete history can take substantially longer.

#### Step B6 — Start the API, UI, and Grafana

Keep the API and UI commands running in separate terminals:

```bash
make api
```

```bash
make ui
```

Start the provisioned Grafana service:

```bash
docker compose up -d grafana
```

| Component | URL |
|---|---|
| FastAPI | <http://localhost:8000> |
| FastAPI documentation | <http://localhost:8000/docs> |
| Streamlit | <http://localhost:8501> |
| Grafana | <http://localhost:3000> |

Sign in to Grafana with `GF_SECURITY_ADMIN_USER` and
`GF_SECURITY_ADMIN_PASSWORD`; the **FFA Overview** dashboard and PostgreSQL datasource are provisioned
automatically.

#### Step B7 — Verify the local installation

```bash
curl --fail http://localhost:8000/healthz
```

Open Streamlit and ask:

> What was Apple's net income in FY2024?

The result should contain a SQL-backed `NumberFact` with a green **Grounded** status. If it does not,
rerun `make ingest` and inspect its JSON summary.

Stop `make api` and `make ui` with `Ctrl+C`. Stop PostgreSQL and Grafana without deleting their named
volumes with:

```bash
docker compose down
```

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
