"""Static contract tests for Grafana dashboards-as-code provisioning."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
GRAFANA_ROOT = ROOT / "monitoring" / "grafana"


def test_postgres_datasource_uses_compose_host_and_read_only_role() -> None:
    datasource = (GRAFANA_ROOT / "datasources" / "postgres.yml").read_text()

    assert "url: postgres:5432" in datasource
    assert "user: ffa_ro" in datasource
    assert "password: $GRAFANA_DATABASE_PASSWORD" in datasource
    assert "localhost" not in datasource


def test_dashboard_provisions_six_expected_panels() -> None:
    dashboard = json.loads((GRAFANA_ROOT / "dashboards" / "ffa_overview.json").read_text())
    panels = dashboard["panels"]
    titles = {panel["title"] for panel in panels}

    assert dashboard["uid"] == "ffa-overview"
    assert len(panels) == 6
    assert titles == {
        "Request volume",
        "Latency: average and p95",
        "Cumulative cost (USD)",
        "Intent distribution",
        "Grounded response rate",
        "Positive feedback rate",
    }
    assert all(panel["datasource"]["uid"] == "ffa-postgres" for panel in panels)

    queries = " ".join(target["rawSql"] for panel in panels for target in panel["targets"])
    assert "query_logs" in queries
    assert "feedback" in queries
    assert "percentile_cont(0.95)" in queries
    assert "COALESCE" in queries


def test_dashboard_provider_loads_the_versioned_json_directory() -> None:
    provider = (GRAFANA_ROOT / "dashboard-provider.yml").read_text()

    assert "path: /var/lib/grafana/dashboards" in provider
    assert "allowUiUpdates: false" in provider
