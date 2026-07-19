CREATE EXTENSION IF NOT EXISTS vector;

-- Reference: company identity
CREATE TABLE companies (
    cik          BIGINT PRIMARY KEY,
    ticker       TEXT UNIQUE NOT NULL,
    name         TEXT NOT NULL,
    sic          TEXT,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Reference: one row per filing document
CREATE TABLE filings (
    accession_no     TEXT PRIMARY KEY,
    cik              BIGINT NOT NULL REFERENCES companies(cik),
    form_type        TEXT NOT NULL,
    filing_date      DATE NOT NULL,
    period_of_report DATE,
    primary_doc_url  TEXT NOT NULL
);

-- Structured facts (SEC_EDGAR_STRUCTURED): one row per metric, period, and filing
CREATE TABLE financial_facts (
    id             BIGSERIAL PRIMARY KEY,
    cik            BIGINT NOT NULL REFERENCES companies(cik),
    ticker         TEXT NOT NULL,
    metric         TEXT NOT NULL,
    taxonomy_tag   TEXT NOT NULL,
    unit           TEXT NOT NULL,
    fiscal_year    INT NOT NULL,
    fiscal_period  TEXT NOT NULL,
    period_start   DATE,
    period_end     DATE,
    value          NUMERIC NOT NULL,
    form_type      TEXT NOT NULL,
    filing_date    DATE NOT NULL,
    accession_no   TEXT REFERENCES filings(accession_no),
    source_url     TEXT NOT NULL,
    UNIQUE (cik, metric, fiscal_year, fiscal_period, taxonomy_tag, filing_date)
);
CREATE INDEX idx_facts_lookup
    ON financial_facts (ticker, metric, fiscal_year, fiscal_period);

-- Unstructured chunks (SEC_EDGAR_UNSTRUCTURED)
CREATE TABLE doc_chunks (
    id             BIGSERIAL PRIMARY KEY,
    accession_no   TEXT NOT NULL REFERENCES filings(accession_no),
    cik            BIGINT NOT NULL REFERENCES companies(cik),
    ticker         TEXT NOT NULL,
    fiscal_year    INT,
    fiscal_period  TEXT,
    section        TEXT NOT NULL,
    chunk_index    INT NOT NULL,
    text           TEXT NOT NULL,
    token_count    INT NOT NULL,
    embedding      vector(1536) NOT NULL,
    text_tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    source_url     TEXT NOT NULL
);
CREATE INDEX idx_chunks_hnsw
    ON doc_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_tsv
    ON doc_chunks USING gin (text_tsv);
CREATE INDEX idx_chunks_meta
    ON doc_chunks (ticker, fiscal_year, fiscal_period);

-- Incremental ingestion bookkeeping
CREATE TABLE ingestion_state (
    source           TEXT NOT NULL,
    cik              BIGINT NOT NULL,
    last_accession   TEXT,
    last_filing_date DATE,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, cik)
);

-- Observability
CREATE TABLE query_logs (
    id            BIGSERIAL PRIMARY KEY,
    trace_id      TEXT NOT NULL,
    session_id    TEXT,
    question      TEXT NOT NULL,
    intent        TEXT,
    route         TEXT,
    latency_ms    INT,
    input_tokens  INT,
    output_tokens INT,
    cost_usd      NUMERIC,
    grounded      BOOLEAN,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE feedback (
    id          BIGSERIAL PRIMARY KEY,
    trace_id    TEXT NOT NULL,
    rating      SMALLINT NOT NULL,
    comment     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE eval_runs (
    id          BIGSERIAL PRIMARY KEY,
    run_type    TEXT NOT NULL,
    config      JSONB NOT NULL,
    metrics     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
