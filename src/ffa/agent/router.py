"""Intent-bound function calling and end-to-end agent orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Protocol
from uuid import uuid4

from openai import OpenAI

from ffa.agent.generation import GenerationCallable, generate
from ffa.agent.guardrails import GuardResult, check_input
from ffa.agent.schemas import Answer, Intent, NumberFact, Understanding
from ffa.agent.tools.retrieval_tool import retrieval_tool
from ffa.agent.tools.sql_tool import sql_tool
from ffa.agent.understanding import understand
from ffa.config import Settings, get_settings
from ffa.retrieval.base import Chunk

_OUT_OF_SCOPE_REFUSAL = (
    "I can only answer questions about public-company financial fundamentals and SEC filings."
)
_ROUTE_BY_INTENT = MappingProxyType(
    {
        Intent.NUMERIC: "sql_tool",
        Intent.NARRATIVE: "retrieval_tool",
        Intent.HYBRID: "hybrid",
        Intent.OUT_OF_SCOPE: "out_of_scope",
    }
)
_TOOLS_BY_INTENT = MappingProxyType(
    {
        Intent.NUMERIC: ("sql_tool",),
        Intent.NARRATIVE: ("retrieval_tool",),
        Intent.HYBRID: ("sql_tool", "retrieval_tool"),
        Intent.OUT_OF_SCOPE: (),
    }
)
_TOOL_DEFINITIONS = MappingProxyType(
    {
        "sql_tool": {
            "type": "function",
            "name": "sql_tool",
            "description": (
                "Return typed numeric facts by generating and executing validated read-only SQL."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "retrieval_tool": {
            "type": "function",
            "name": "retrieval_tool",
            "description": ("Return reranked filing chunks from hybrid text and vector retrieval."),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
)
_PLANNER_INSTRUCTIONS = """The intent is already classified and the available tools are fixed.
Call every provided tool exactly once and call no other tool.
Use an empty JSON object for arguments.
Do not answer the question, calculate a number, rewrite tool arguments, or emit explanatory prose.
"""

type GuardrailCallable = Callable[[str], GuardResult]
type UnderstandingCallable = Callable[[str], Understanding]
type SqlToolCallable = Callable[[Understanding], list[NumberFact]]
type RetrievalToolCallable = Callable[[Understanding], list[Chunk]]


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Evidence and trace metadata passed to grounded generation."""

    facts: list[NumberFact] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    route: str = "out_of_scope"
    trace_id: str = ""


@dataclass(frozen=True, slots=True)
class AgentRun:
    """End-to-end result retaining both the answer and its traceable context."""

    answer: Answer
    context: AgentContext


@dataclass(frozen=True, slots=True)
class PlannedToolCall:
    """Validated function call emitted by the OpenAI planner."""

    name: str
    call_id: str


class RoutingError(RuntimeError):
    """Raised when function calling deviates from the intent-bound route."""


class FunctionCallPlanner(Protocol):
    """Backend contract for intent-bound OpenAI function calling."""

    def plan(self, intent: Intent, *, trace_id: str) -> list[PlannedToolCall]:
        """Return the exact tool calls required by an intent."""
        ...


class OpenAIFunctionCallPlanner:
    """Expose local tools through strict OpenAI function definitions."""

    def __init__(self, *, client: OpenAI, model: str) -> None:
        """Initialize the planner with explicit OpenAI configuration."""
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("OPENAI_MODEL must be configured before routing tools.")
        self._client = client
        self._model = normalized_model

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> OpenAIFunctionCallPlanner:
        """Build the planner from central application settings."""
        resolved_settings = settings or get_settings()
        if resolved_settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY must be configured before routing tools.")
        api_key = resolved_settings.openai_api_key.get_secret_value().strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be configured before routing tools.")
        return cls(
            client=OpenAI(api_key=api_key),
            model=resolved_settings.openai_model,
        )

    def plan(self, intent: Intent, *, trace_id: str) -> list[PlannedToolCall]:
        """Request strict function calls constrained to the classified intent."""
        expected_tools = _TOOLS_BY_INTENT[intent]
        if not expected_tools:
            return []
        tools = [_TOOL_DEFINITIONS[name] for name in expected_tools]
        tool_choice: object = (
            {"type": "function", "name": expected_tools[0]}
            if len(expected_tools) == 1
            else "required"
        )
        response = self._client.responses.create(
            model=self._model,
            instructions=_PLANNER_INSTRUCTIONS,
            input=json.dumps({"intent": intent.value}),
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=len(expected_tools) > 1,
            metadata={"trace_id": trace_id},
        )
        calls: list[PlannedToolCall] = []
        for item in response.output:
            if item.type != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise RoutingError("The tool planner returned invalid arguments.") from exc
            if arguments != {}:
                raise RoutingError("The tool planner returned unexpected arguments.")
            calls.append(PlannedToolCall(name=item.name, call_id=item.call_id))
        return calls


def route_understanding(
    understanding: Understanding,
    *,
    trace_id: str,
    planner: FunctionCallPlanner | None = None,
    sql_runner: SqlToolCallable = sql_tool,
    retrieval_runner: RetrievalToolCallable = retrieval_tool,
) -> AgentContext:
    """Execute exactly the tools prescribed by the classified intent."""
    route = _ROUTE_BY_INTENT[understanding.intent]
    expected_tools = _TOOLS_BY_INTENT[understanding.intent]
    if not expected_tools:
        return AgentContext(route=route, trace_id=trace_id)

    resolved_planner = planner or _get_default_planner()
    calls = resolved_planner.plan(understanding.intent, trace_id=trace_id)
    call_names = [call.name for call in calls]
    if len(call_names) != len(expected_tools) or set(call_names) != set(expected_tools):
        raise RoutingError("The tool planner did not follow the classified intent.")

    facts = sql_runner(understanding) if "sql_tool" in expected_tools else []
    chunks = retrieval_runner(understanding) if "retrieval_tool" in expected_tools else []
    return AgentContext(
        facts=facts,
        chunks=chunks,
        route=route,
        trace_id=trace_id,
    )


def run_agent(
    question: str,
    *,
    trace_id: str | None = None,
    guardrail_checker: GuardrailCallable = check_input,
    understanding_fn: UnderstandingCallable = understand,
    planner: FunctionCallPlanner | None = None,
    sql_runner: SqlToolCallable = sql_tool,
    retrieval_runner: RetrievalToolCallable = retrieval_tool,
    answer_generator: GenerationCallable = generate,
) -> AgentRun:
    """Run guardrails, understanding, intent-bound tools, and grounded generation."""
    resolved_trace_id = _resolve_trace_id(trace_id)
    guard_result = guardrail_checker(question)
    if not guard_result.allowed:
        context = AgentContext(route="blocked", trace_id=resolved_trace_id)
        return AgentRun(
            answer=Answer(text=guard_result.reason, grounded=True),
            context=context,
        )

    understanding = understanding_fn(question)
    context = route_understanding(
        understanding,
        trace_id=resolved_trace_id,
        planner=planner,
        sql_runner=sql_runner,
        retrieval_runner=retrieval_runner,
    )
    if understanding.intent is Intent.OUT_OF_SCOPE:
        return AgentRun(
            answer=Answer(text=_OUT_OF_SCOPE_REFUSAL, grounded=True),
            context=context,
        )
    return AgentRun(
        answer=answer_generator(question, context),
        context=context,
    )


@lru_cache(maxsize=1)
def _get_default_planner() -> OpenAIFunctionCallPlanner:
    """Reuse the OpenAI function-calling client across requests."""
    return OpenAIFunctionCallPlanner.from_settings(get_settings())


def _resolve_trace_id(trace_id: str | None) -> str:
    """Use a caller trace ID or create a process-independent identifier."""
    if trace_id is None:
        return uuid4().hex
    normalized = trace_id.strip()
    if not normalized:
        raise ValueError("trace_id must not be empty.")
    return normalized
