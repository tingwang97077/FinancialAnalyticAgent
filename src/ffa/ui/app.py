"""Streamlit client for the Financial Fundamentals Agent API."""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any
from uuid import uuid4

import httpx
import streamlit as st
from pydantic import ValidationError

from ffa.api.routes import AskResponse

_DEFAULT_API_BASE_URL = "http://localhost:8000"
_DEFAULT_GRAFANA_BASE_URL = "http://localhost:3000"
_ASK_TIMEOUT_SECONDS = 300.0
_FEEDBACK_TIMEOUT_SECONDS = 15.0
_PERCENT_UNITS = frozenset({"%", "percent", "percentage"})
_SHARE_UNITS = frozenset({"share", "shares"})
_SCALES = (
    (Decimal("1000000000000"), "T"),
    (Decimal("1000000000"), "Md"),
    (Decimal("1000000"), "M"),
    (Decimal("1000"), "k"),
)


def main() -> None:
    """Render the chat client and the monitoring dashboard link."""
    st.set_page_config(
        page_title="Financial Fundamentals Analytic Agent",
        page_icon="📊",
        layout="wide",
    )
    _initialize_session_state()
    api_base_url = _api_base_url()

    st.title("Financial Fundamentals Analytic Agent")
    st.caption(f"API: {api_base_url}")
    chat_tab, dashboard_tab = st.tabs(["Chat", "Dashboard"])

    with chat_tab:
        _render_chat(api_base_url)

    with dashboard_tab:
        st.header("Monitoring dashboard")
        st.markdown(
            "Open the provisioned [FFA Grafana dashboard]"
            f"({_grafana_base_url()}) to inspect requests, latency, costs, intents, "
            "grounding, and feedback."
        )
        st.caption("Grafana must be running locally with `docker compose up -d grafana`.")


def _render_chat(api_base_url: str) -> None:
    """Render persisted messages and handle a new chat submission."""
    for message in st.session_state.messages:
        _render_message(message, api_base_url)

    question = st.chat_input("Ask about financial facts or SEC filings")
    if not question:
        return

    user_message = {"role": "user", "text": question}
    st.session_state.messages.append(user_message)
    _render_message(user_message, api_base_url)

    with st.spinner("Grounding the answer in SEC data..."):
        answer, error = _ask_api(
            api_base_url,
            question=question,
            session_id=st.session_state.session_id,
        )

    if error is not None:
        assistant_message = {"role": "assistant", "error": error}
    else:
        assert answer is not None
        assistant_message = {"role": "assistant", "answer": answer.model_dump(mode="json")}
    st.session_state.messages.append(assistant_message)
    _render_message(assistant_message, api_base_url)


def _render_message(message: dict[str, Any], api_base_url: str) -> None:
    """Render one user or assistant message from session history."""
    role = str(message["role"])
    with st.chat_message(role):
        if role == "user":
            st.markdown(str(message["text"]))
            return
        if "error" in message:
            st.error(str(message["error"]))
            return

        try:
            answer = AskResponse.model_validate(message["answer"])
        except ValidationError:
            st.error("A stored API response is no longer valid. Please submit the question again.")
            return
        _render_answer(answer, api_base_url)


def _render_answer(answer: AskResponse, api_base_url: str) -> None:
    """Render grounded status, text, raw facts as formatted rows, and citations."""
    if answer.grounded:
        st.badge("Grounded", icon=":material/verified:", color="green")
    else:
        st.badge("Not grounded", icon=":material/warning:", color="red")
        st.error(
            "This answer could not be fully grounded in the retrieved evidence. "
            "Do not treat unsupported claims or numbers as reliable."
        )

    st.markdown(answer.text)

    if answer.numbers:
        st.subheader("Financial facts")
        rows = [
            {
                "Metric": fact.metric.replace("_", " ").title(),
                "Period": f"{fact.fiscal_year} {fact.fiscal_period}",
                "Value": _format_number(fact.value, fact.unit),
                "Unit": fact.unit,
            }
            for fact in answer.numbers
        ]
        st.dataframe(rows, hide_index=True, width="stretch")

    if answer.citations:
        st.subheader("Sources")
        for index, citation in enumerate(answer.citations, start=1):
            section = citation.section or "SEC filing"
            accession = f" — {citation.accession_no}" if citation.accession_no else ""
            st.markdown(f"{index}. [{section}{accession}]({citation.source_url})")

    _render_feedback(answer.trace_id, api_base_url)


def _render_feedback(trace_id: str, api_base_url: str) -> None:
    """Render a per-answer feedback form and persist its status in the session."""
    submitted = st.session_state.feedback.get(trace_id)
    if submitted is not None:
        st.success(f"Feedback submitted ({submitted}).")
        return

    with st.form(key=f"feedback-{trace_id}"):
        comment = st.text_input("Optional feedback comment")
        positive_column, negative_column = st.columns(2)
        positive = positive_column.form_submit_button("👍 Helpful", use_container_width=True)
        negative = negative_column.form_submit_button("👎 Not helpful", use_container_width=True)

    if not positive and not negative:
        return
    rating = 1 if positive else -1
    error = _feedback_api(
        api_base_url,
        trace_id=trace_id,
        rating=rating,
        comment=comment or None,
    )
    if error is not None:
        st.error(error)
        return
    label = "helpful" if rating == 1 else "not helpful"
    st.session_state.feedback[trace_id] = label
    st.success(f"Feedback submitted ({label}).")


def _ask_api(
    api_base_url: str,
    *,
    question: str,
    session_id: str,
) -> tuple[AskResponse | None, str | None]:
    """Submit a question to the API and validate its response contract."""
    try:
        response = httpx.post(
            f"{api_base_url}/ask",
            json={"question": question, "session_id": session_id},
            timeout=_ASK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return AskResponse.model_validate(response.json()), None
    except httpx.ConnectError:
        return None, "The API is unreachable. Start it with `make api` and try again."
    except httpx.TimeoutException:
        return None, "The API did not respond before the request timed out."
    except httpx.HTTPStatusError as exc:
        return None, _http_error_message(exc.response)
    except (ValueError, ValidationError):
        return None, "The API returned an invalid response."
    except httpx.HTTPError:
        return None, "The API request failed. Please try again."


def _feedback_api(
    api_base_url: str,
    *,
    trace_id: str,
    rating: int,
    comment: str | None,
) -> str | None:
    """Submit feedback through the API and return a user-safe error when needed."""
    try:
        response = httpx.post(
            f"{api_base_url}/feedback",
            json={"trace_id": trace_id, "rating": rating, "comment": comment},
            timeout=_FEEDBACK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return None
    except httpx.ConnectError:
        return "Feedback could not be submitted because the API is unreachable."
    except httpx.TimeoutException:
        return "Feedback submission timed out. Please try again."
    except httpx.HTTPStatusError as exc:
        return _http_error_message(exc.response)
    except httpx.HTTPError:
        return "Feedback could not be submitted. Please try again."


def _format_number(value: float, unit: str) -> str:
    """Format a numeric fact for display without changing its stored raw value."""
    decimal_value = Decimal(str(value))
    normalized_unit = unit.strip().casefold()
    if normalized_unit in _PERCENT_UNITS:
        return f"{_localized_decimal(decimal_value, decimals=2)} %"

    scale, prefix = _display_scale(decimal_value)
    scaled_value = decimal_value / scale
    formatted_value = _localized_decimal(scaled_value, decimals=2 if scale != 1 else 0)
    if normalized_unit == "usd":
        suffix = f"{prefix}$" if prefix else "$"
    elif normalized_unit in _SHARE_UNITS:
        suffix = f"{prefix} actions" if prefix else "actions"
    else:
        suffix = f"{prefix} {unit}".strip()
    return f"{formatted_value} {suffix}".strip()


def _display_scale(value: Decimal) -> tuple[Decimal, str]:
    """Select the largest readable magnitude for an absolute value."""
    absolute_value = abs(value)
    for scale, prefix in _SCALES:
        if absolute_value >= scale:
            return scale, prefix
    return Decimal(1), ""


def _localized_decimal(value: Decimal, *, decimals: int) -> str:
    """Render a decimal with grouped thousands and a comma decimal separator."""
    rendered = f"{value:,.{decimals}f}"
    return rendered.replace(",", "\N{NO-BREAK SPACE}").replace(".", ",")


def _http_error_message(response: httpx.Response) -> str:
    """Return an API error without exposing internal exception details."""
    detail: object = None
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail")
    except ValueError:
        pass
    if isinstance(detail, str) and detail.strip():
        return f"The API returned HTTP {response.status_code}: {detail}"
    return f"The API returned HTTP {response.status_code}."


def _api_base_url() -> str:
    """Read the API base URL from the environment with a local default."""
    configured = os.getenv("API_BASE_URL", _DEFAULT_API_BASE_URL).strip()
    return (configured or _DEFAULT_API_BASE_URL).rstrip("/")


def _grafana_base_url() -> str:
    """Read the Grafana URL from the environment with a local default."""
    configured = os.getenv("GRAFANA_BASE_URL", _DEFAULT_GRAFANA_BASE_URL).strip()
    return (configured or _DEFAULT_GRAFANA_BASE_URL).rstrip("/")


def _initialize_session_state() -> None:
    """Initialize conversation and feedback state for the browser session."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}


if __name__ == "__main__":
    main()
