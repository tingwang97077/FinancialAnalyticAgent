"""HTTP client for SEC EDGAR resources."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from ffa.config import Settings, get_settings

SEC_DATA_BASE_URL = "https://data.sec.gov"
SEC_WWW_BASE_URL = "https://www.sec.gov"
COMPANY_TICKERS_URL = f"{SEC_WWW_BASE_URL}/files/company_tickers.json"

_ALLOWED_SEC_HOSTS = frozenset({"sec.gov", "www.sec.gov", "data.sec.gov"})
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_USER_AGENT_PATTERN = re.compile(r"^\S.*\s+[^\s@]+@[^\s@]+\.[^\s@]+$")

type JsonObject = dict[str, Any]


class SecClientError(RuntimeError):
    """Base exception for SEC client failures."""


class SecPayloadError(SecClientError):
    """Raised when an SEC JSON response is not an object."""


class SecEdgarClient:
    """Fetch SEC EDGAR resources with identification, throttling, retries, and caching."""

    def __init__(
        self,
        *,
        user_agent: str,
        max_rps: float = 8,
        cache_dir: str | Path = "/data/sec_cache",
        timeout_seconds: float = 30,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
        http_client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize the SEC client.

        Args:
            user_agent: Identifying value in ``AppName contact@email.com`` format.
            max_rps: Maximum request rate, capped at the SEC limit of 10 requests/second.
            cache_dir: Directory used for persistent response caching.
            timeout_seconds: Per-request HTTP timeout.
            max_retries: Number of retries after a transient failure.
            retry_backoff_seconds: Initial exponential retry delay.
            http_client: Optional injected synchronous HTTP client.
            clock: Monotonic clock, injectable for deterministic tests.
            sleep: Sleep function, injectable for deterministic tests.

        Raises:
            ValueError: If identification, rate, timeout, or retry settings are invalid.
        """
        normalized_user_agent = user_agent.strip()
        if not _USER_AGENT_PATTERN.fullmatch(normalized_user_agent):
            raise ValueError(
                "SEC_USER_AGENT must identify the application and a contact email, "
                'for example "FinancialFundamentalsAgent contact@example.com".'
            )
        if not 0 < max_rps <= 10:
            raise ValueError(
                "SEC request rate must be greater than 0 and at most 10 requests/second."
            )
        if timeout_seconds <= 0:
            raise ValueError("SEC request timeout must be greater than 0 seconds.")
        if max_retries < 0:
            raise ValueError("SEC maximum retries must not be negative.")
        if retry_backoff_seconds < 0:
            raise ValueError("SEC retry backoff must not be negative.")

        self._headers = {
            "User-Agent": normalized_user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
        }
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._sleep = sleep
        self._rate_limiter = _RateLimiter(max_rps=max_rps, clock=clock, sleep=sleep)
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> Self:
        """Create a client from central application settings."""
        resolved_settings = settings or get_settings()
        return cls(
            user_agent=resolved_settings.sec_user_agent,
            max_rps=resolved_settings.sec_max_rps,
            cache_dir=resolved_settings.sec_cache_dir,
            http_client=http_client,
        )

    def __enter__(self) -> Self:
        """Return this client as a context-managed resource."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close resources owned by this client."""
        self.close()

    def close(self) -> None:
        """Close the internally created HTTP client, if any."""
        if self._owns_http_client:
            self._http_client.close()

    def fetch_company_tickers(self, *, use_cache: bool = True) -> JsonObject:
        """Return the SEC ticker-to-CIK dataset."""
        return self.get_json(COMPANY_TICKERS_URL, use_cache=use_cache)

    def fetch_companyfacts(self, cik: int, *, use_cache: bool = True) -> JsonObject:
        """Return XBRL company facts for a CIK."""
        url = f"{SEC_DATA_BASE_URL}/api/xbrl/companyfacts/CIK{_format_cik(cik)}.json"
        return self.get_json(url, use_cache=use_cache)

    def fetch_submissions(self, cik: int, *, use_cache: bool = True) -> JsonObject:
        """Return filing submission metadata for a CIK."""
        url = f"{SEC_DATA_BASE_URL}/submissions/CIK{_format_cik(cik)}.json"
        return self.get_json(url, use_cache=use_cache)

    def fetch_filing_document(self, url: str, *, use_cache: bool = True) -> str:
        """Return a filing document as decoded text."""
        return self.get_text(url, use_cache=use_cache)

    def get_json(self, url: str, *, use_cache: bool = True) -> JsonObject:
        """Fetch and decode a JSON object from an allowed SEC URL."""
        payload = self.get_bytes(url, use_cache=use_cache)
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SecPayloadError(f"SEC response from {url} is not valid JSON.") from exc
        if not isinstance(decoded, dict):
            raise SecPayloadError(f"SEC response from {url} must be a JSON object.")
        return decoded

    def get_text(self, url: str, *, use_cache: bool = True) -> str:
        """Fetch and decode text from an allowed SEC URL."""
        return self.get_bytes(url, use_cache=use_cache).decode("utf-8", errors="replace")

    def get_bytes(self, url: str, *, use_cache: bool = True) -> bytes:
        """Fetch raw bytes from an allowed SEC URL, optionally using the disk cache."""
        _validate_sec_url(url)
        cache_path = self._cache_path(url)
        if use_cache and cache_path.is_file():
            return cache_path.read_bytes()

        payload = self._request(url)
        if use_cache:
            self._write_cache(cache_path, payload)
        return payload

    def _request(self, url: str) -> bytes:
        for attempt in range(self._max_retries + 1):
            try:
                self._rate_limiter.wait()
                response = self._http_client.get(url, headers=self._headers)
            except httpx.TransportError:
                if attempt >= self._max_retries:
                    raise
                self._sleep(self._backoff_delay(attempt))
                continue

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                self._sleep(self._retry_delay(attempt, response))
                continue

            response.raise_for_status()
            return response.content

        raise AssertionError("SEC retry loop ended unexpectedly.")

    def _retry_delay(self, attempt: int, response: httpx.Response) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return min(max(float(retry_after), 0), 60)
            except ValueError:
                pass
        return self._backoff_delay(attempt)

    def _backoff_delay(self, attempt: int) -> float:
        return min(self._retry_backoff_seconds * (2**attempt), 60)

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        suffix = Path(urlsplit(url).path).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
            suffix = ".cache"
        return self._cache_dir / f"{digest}{suffix}"

    @staticmethod
    def _write_cache(cache_path: Path, payload: bytes) -> None:
        temporary_path = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
        try:
            temporary_path.write_bytes(payload)
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)


class _RateLimiter:
    def __init__(
        self,
        *,
        max_rps: float,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        self._interval_seconds = 1 / max_rps
        self._clock = clock
        self._sleep = sleep
        self._next_request_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            delay = self._next_request_at - now
            if delay > 0:
                self._sleep(delay)
                now = self._clock()
            self._next_request_at = max(now, self._next_request_at) + self._interval_seconds


def _format_cik(cik: int) -> str:
    if isinstance(cik, bool) or not isinstance(cik, int) or not 0 < cik <= 9_999_999_999:
        raise ValueError("CIK must be a positive integer with at most 10 digits.")
    return f"{cik:010d}"


def _validate_sec_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_SEC_HOSTS:
        raise ValueError("SEC client URLs must use HTTPS and an allowed sec.gov host.")
