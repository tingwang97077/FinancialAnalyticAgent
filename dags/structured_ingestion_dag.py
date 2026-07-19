"""Airflow DAG for weekly structured SEC EDGAR ingestion."""

import pendulum
from airflow.sdk import Asset, dag, task

FACTS_READY = Asset("facts_ready")


@dag(
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ingestion", "structured"],
)
def structured_ingestion_dag() -> None:
    """Orchestrate companyfacts fetch, normalization, and loading."""

    @task
    def resolve_universe() -> list[dict[str, object]]:
        from ffa.common.entities import resolve_universe as resolve

        return list(resolve())

    @task
    def fetch_companyfacts(company: dict[str, object]) -> dict[str, object]:
        from ffa.ingestion.structured.fetch import fetch_companyfacts as fetch

        # Scheduled runs bypass persistent HTTP cache so new SEC filings are visible.
        return dict(fetch(company, use_cache=False))

    @task
    def normalize_facts(raw: dict[str, object]) -> list[dict[str, object]]:
        from ffa.ingestion.structured.normalize import normalize_facts as normalize

        return [dict(row) for row in normalize(raw)]

    @task(outlets=[FACTS_READY])
    def load_facts(rows: list[dict[str, object]]) -> int:
        from ffa.ingestion.structured.load import load_facts as load

        return load(rows)

    companies = resolve_universe()
    raw_companyfacts = fetch_companyfacts.expand(company=companies)
    normalized_rows = normalize_facts.expand(raw=raw_companyfacts)
    load_facts.expand(rows=normalized_rows)


structured_ingestion_dag()
