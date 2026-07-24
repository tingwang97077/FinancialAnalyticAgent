# Financial Fundamentals Agent

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
