"""Tests for the SEC EDGAR HTTP client."""

from pathlib import Path

import httpx
import pytest

from ffa.common.sec_client import SecEdgarClient


class FakeTime:
    """Deterministic clock and sleep implementation."""

    def __init__(self) -> None:
        self.now = 0.0
        self.delays: list[float] = []

    def monotonic(self) -> float:
        """Return the current fake monotonic time."""
        return self.now

    def sleep(self, delay: float) -> None:
        """Advance fake time by a requested delay."""
        self.delays.append(delay)
        self.now += delay


def test_requires_identifying_user_agent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contact email"):
        SecEdgarClient(user_agent="anonymous", cache_dir=tmp_path)


def test_companyfacts_uses_padded_cik_headers_and_cache(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"cik": 320193})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = SecEdgarClient(
            user_agent="FFA tests@example.com",
            cache_dir=tmp_path,
            http_client=http_client,
        )
        first = client.fetch_companyfacts(320193)
        second = client.fetch_companyfacts(320193)

    assert first == second == {"cik": 320193}
    assert len(requests) == 1
    assert requests[0].url.path.endswith("/CIK0000320193.json")
    assert requests[0].headers["User-Agent"] == "FFA tests@example.com"
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_retries_transient_responses_without_real_sleep(tmp_path: Path) -> None:
    fake_time = FakeTime()
    response_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal response_count
        response_count += 1
        if response_count == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = SecEdgarClient(
            user_agent="FFA tests@example.com",
            max_rps=10,
            cache_dir=tmp_path,
            http_client=http_client,
            clock=fake_time.monotonic,
            sleep=fake_time.sleep,
        )
        assert client.fetch_submissions(1, use_cache=False) == {"ok": True}

    assert response_count == 2
    assert 0.25 in fake_time.delays


def test_rejects_non_sec_urls(tmp_path: Path) -> None:
    client = SecEdgarClient(user_agent="FFA tests@example.com", cache_dir=tmp_path)
    try:
        with pytest.raises(ValueError, match="allowed sec.gov host"):
            client.get_bytes("https://example.com/filing.html")
    finally:
        client.close()
