"""The only module that talks to the Claude API."""

from __future__ import annotations

from typing import TypeVar

import anthropic
from pydantic import BaseModel

from app import config

T = TypeVar("T", bound=BaseModel)
_client: anthropic.Anthropic | None = None


class LLMError(Exception):
    pass


def _get() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(timeout=90.0, max_retries=2)
    return _client


def structured(system: str, user: str, schema: type[T], effort: str = "medium") -> T:
    """One call that must return an instance of `schema` (goal reader, fixer)."""
    try:
        resp = _get().messages.parse(
            model=config.MODEL,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": effort},
        )
    except anthropic.APIStatusError as e:
        raise LLMError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not reach the Claude API") from e
    if resp.stop_reason == "refusal":
        raise LLMError("The model declined this request")
    if resp.parsed_output is None:
        raise LLMError(f"No structured output (stop_reason={resp.stop_reason})")
    return resp.parsed_output


def tool_turn(system: str, messages: list, tools: list, effort: str = "medium"):
    """One planner step. The response may contain tool_use blocks."""
    try:
        resp = _get().messages.create(
            model=config.MODEL,
            max_tokens=16000,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"},
            output_config={"effort": effort},
        )
    except anthropic.APIStatusError as e:
        raise LLMError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not reach the Claude API") from e
    if resp.stop_reason == "refusal":
        raise LLMError("The model declined this request")
    return resp
