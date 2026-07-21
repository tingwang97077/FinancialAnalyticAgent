"""Airflow DAG for weekly unstructured SEC EDGAR ingestion."""

import pendulum
from airflow.sdk import Asset, dag, task

CHUNKS_READY = Asset("chunks_ready")


@dag(
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ingestion", "unstructured"],
)
def unstructured_ingestion_dag() -> None:
    """Orchestrate filing discovery, cleaning, chunking, embedding, and loading."""

    @task
    def resolve_universe() -> list[dict[str, object]]:
        from ffa.common.entities import resolve_universe as resolve

        return list(resolve())

    @task
    def discover_filings(company: dict[str, object]) -> dict[str, object]:
        from ffa.ingestion.unstructured.fetch import discover_filings as discover

        # Scheduled runs bypass the current submissions cache so new filings are visible.
        return dict(discover(company, use_cache=False))

    @task
    def report_discoveries(
        discoveries: list[dict[str, object]],
    ) -> dict[str, list[dict[str, object]]]:
        from ffa.ingestion.unstructured.fetch import summarize_filing_discoveries

        report = summarize_filing_discoveries(discoveries)  # type: ignore[arg-type]
        return {
            "filings": [dict(filing) for filing in report["filings"]],
            "ok": [dict(summary) for summary in report["ok"]],
            "skipped": [dict(summary) for summary in report["skipped"]],
        }

    @task
    def flatten_filings(
        report: dict[str, list[dict[str, object]]],
    ) -> list[dict[str, object]]:
        return report["filings"]

    @task
    def fetch_documents(filing: dict[str, object]) -> dict[str, object]:
        from ffa.ingestion.unstructured.fetch import fetch_documents as fetch

        return dict(fetch(filing))

    @task
    def clean_text(document: dict[str, object]) -> list[dict[str, object]]:
        from ffa.ingestion.unstructured.clean import clean_text as clean

        return [dict(section) for section in clean(document)]

    @task
    def chunk_text(sections: list[dict[str, object]]) -> list[dict[str, object]]:
        from ffa.ingestion.unstructured.chunk import chunk_text as chunk

        return [dict(row) for row in chunk(sections)]

    @task
    def embed_chunks(chunks: list[dict[str, object]]) -> list[dict[str, object]]:
        from ffa.ingestion.unstructured.embed import embed_chunks as embed

        return [dict(row) for row in embed(chunks)]

    @task(outlets=[CHUNKS_READY])
    def load_chunks(chunks: list[dict[str, object]]) -> int:
        from ffa.ingestion.unstructured.load import load_chunks as load

        return load(chunks)

    companies = resolve_universe()
    discoveries = discover_filings.expand(company=companies)
    discovery_report = report_discoveries(discoveries)
    filings = flatten_filings(discovery_report)
    documents = fetch_documents.expand(filing=filings)
    sections = clean_text.expand(document=documents)
    chunks = chunk_text.expand(sections=sections)
    embedded_chunks = embed_chunks.expand(chunks=chunks)
    load_chunks.expand(chunks=embedded_chunks)


unstructured_ingestion_dag()
