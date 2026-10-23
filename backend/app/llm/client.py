"""The only module that talks to the LLM provider (Groq, OpenAI-compatible chat completions)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TypeVar

import groq
from pydantic import BaseModel, ValidationError

from app import config

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger("uvicorn.error")
_client: groq.Groq | None = None


class LLMError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class TurnResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # The assistant message to append to the conversation before sending tool results.
    assistant_message: dict = field(default_factory=dict)


def _get() -> groq.Groq:
    global _client
    if _client is None:
        # Retries are handled in _create so a rate limit never blocks for long.
        _client = groq.Groq(api_key=config.GROQ_API_KEY, timeout=45.0, max_retries=0)
    return _client


def _retry_after(e: groq.RateLimitError) -> float | None:
    try:
        return float(e.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def _create(role: str, model: str, effort: str, **kwargs):
    extra = {"reasoning_effort": effort} if model.startswith("openai/gpt-oss") else {}
    for attempt in range(2):
        started = time.monotonic()
        try:
            resp = _get().chat.completions.create(model=model, **extra, **kwargs)
        except groq.RateLimitError as e:
            wait = _retry_after(e)
            log.warning("[llm] %s %s rate limited (retry-after %s s)", role, model, wait)
            if attempt == 0 and wait is not None and wait <= config.MAX_RATE_LIMIT_WAIT_SECONDS:
                time.sleep(wait)
                continue
            raise LLMError("Groq rate limit reached; using the offline fallback for this step") from e
        except groq.APIStatusError as e:
            log.warning("[llm] %s %s error %s: %s", role, model, e.status_code, e.message)
            raise LLMError(f"Groq API error {e.status_code}: {e.message}") from e
        except groq.APIConnectionError as e:
            log.warning("[llm] %s %s connection error", role, model)
            raise LLMError("Could not reach the Groq API") from e
        usage = getattr(resp, "usage", None)
        log.info("[llm] %s %s %.1fs tokens in=%s out=%s", role, model, time.monotonic() - started,
                 getattr(usage, "prompt_tokens", "?"), getattr(usage, "completion_tokens", "?"))
        return resp
    raise LLMError("Groq rate limit reached")


def structured(system: str, user: str, schema: type[T], effort: str = "medium", role: str = "structured",
               model: str | None = None) -> T:
    """One call that must return an instance of `schema` (goal reader, fixer).

    Uses JSON object mode with the schema in the prompt, validates with Pydantic, and retries once with the
    validation error if the first answer does not match.
    """
    model = model or config.MODEL
    schema_text = json.dumps(schema.model_json_schema())
    messages = [
        {"role": "system", "content": f"{system}\n\nRespond with a single JSON object that matches this JSON schema, "
                                      f"and nothing else:\n{schema_text}"},
        {"role": "user", "content": user},
    ]
    last_error = ""
    for _ in range(2):
        resp = _create(role, model, effort, messages=messages, response_format={"type": "json_object"},
                       max_completion_tokens=4096)
        content = resp.choices[0].message.content or ""
        try:
            return schema.model_validate_json(content)
        except ValidationError as e:
            last_error = str(e)[:800]
            messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"That JSON does not match the schema: {last_error}\nReturn the corrected JSON object only."},
            ]
    raise LLMError(f"Model output did not match {schema.__name__}: {last_error[:200]}")


def _function_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


def tool_turn(system: str, messages: list, tools: list, effort: str = "medium", model: str | None = None) -> TurnResult:
    """One planner step. `messages` holds user/assistant/tool messages; the system prompt is added here."""
    resp = _create(
        "planner", model or config.FAST_MODEL, effort,
        messages=[{"role": "system", "content": system}, *messages],
        tools=_function_tools(tools),
        tool_choice="auto",
        max_completion_tokens=2048,
    )
    msg = resp.choices[0].message
    calls = []
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args if isinstance(args, dict) else {}))
    assistant = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        assistant["tool_calls"] = [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                                   for tc in msg.tool_calls]
    return TurnResult(text=msg.content or "", tool_calls=calls, assistant_message=assistant)
