from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from ffa.ui.app import _ask_api, _feedback_api, _format_number, _grafana_base_url


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (93_736_000_000, "USD", "93,74 Md$"),
        (-3.3599670086086912, "percent", "-3,36 %"),
        (-2_500_000_000, "USD", "-2,50 Md$"),
        (0, "USD", "0 $"),
        (1_520_000_000, "shares", "1,52 Md actions"),
        (925, "USD", "925 $"),
    ],
)
def test_format_number_is_readable_without_changing_input(
    value: float,
    unit: str,
    expected: str,
) -> None:
    original = value

    rendered = _format_number(value, unit)

    assert rendered == expected
    assert value == original


def test_ask_api_posts_to_backend_and_validates_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        calls.append({"url": url, **kwargs})
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "text": "Grounded answer.",
                "numbers": [],
                "citations": [],
                "grounded": True,
                "trace_id": "trace-ui",
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    answer, error = _ask_api(
        "http://localhost:8000",
        question="What was net income?",
        session_id="session-ui",
    )

    assert error is None
    assert answer is not None
    assert answer.trace_id == "trace-ui"
    assert calls[0]["url"] == "http://localhost:8000/ask"
    assert calls[0]["json"] == {
        "question": "What was net income?",
        "session_id": "session-ui",
    }


def test_feedback_api_posts_rating_comment_and_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    comment = 'Très utile <>& "quoted" 😊\nSecond line'

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        calls.append({"url": url, **kwargs})
        return httpx.Response(204, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)

    error = _feedback_api(
        "http://localhost:8000",
        trace_id="trace-ui",
        rating=-1,
        comment=comment,
    )

    assert error is None
    assert calls[0]["url"] == "http://localhost:8000/feedback"
    assert calls[0]["json"] == {
        "trace_id": "trace-ui",
        "rating": -1,
        "comment": comment,
    }


def test_ask_api_reports_connection_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_post(*_: object, **__: object) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fail_post)

    answer, error = _ask_api(
        "http://localhost:8000",
        question="A question",
        session_id="session-ui",
    )

    assert answer is None
    assert error is not None
    assert "make api" in error


def test_initial_ui_has_no_feedback_action_or_empty_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def unexpected_post(*args: object, **kwargs: object) -> httpx.Response:
        calls.append((args, kwargs))
        raise AssertionError("The initial UI must not call the API.")

    monkeypatch.setattr(httpx, "post", unexpected_post)
    app_path = Path(__file__).parents[1] / "src" / "ffa" / "ui" / "app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=20)

    assert not app.exception
    assert len(app.button) == 0
    assert calls == []


def test_grafana_url_uses_environment_with_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRAFANA_BASE_URL", raising=False)
    assert _grafana_base_url() == "http://localhost:3000"

    monkeypatch.setenv("GRAFANA_BASE_URL", "https://grafana.example.test/root/")
    assert _grafana_base_url() == "https://grafana.example.test/root"
