\getenv ffa_readonly_password FFA_READONLY_PASSWORD

\if :{?ffa_readonly_password}
\else
    \echo 'FFA_READONLY_PASSWORD must be configured.'
    \quit
\endif

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ffa_ro') THEN
        CREATE ROLE ffa_ro WITH
            LOGIN
            NOSUPERUSER
            NOCREATEDB
            NOCREATEROLE
            NOREPLICATION
            NOBYPASSRLS;
    END IF;
END
$$;

SELECT format('ALTER ROLE ffa_ro PASSWORD %L', :'ffa_readonly_password') \gexec

ALTER ROLE ffa_ro WITH
    LOGIN
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOREPLICATION
    NOBYPASSRLS;
ALTER ROLE ffa_ro SET default_transaction_read_only TO on;

REVOKE ALL PRIVILEGES ON DATABASE ffa FROM ffa_ro;
GRANT CONNECT ON DATABASE ffa TO ffa_ro;

REVOKE ALL PRIVILEGES ON SCHEMA public FROM ffa_ro;
GRANT USAGE ON SCHEMA public TO ffa_ro;

REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM ffa_ro;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM ffa_ro;

GRANT SELECT ON TABLE companies, filings, financial_facts, query_logs, feedback TO ffa_ro;
