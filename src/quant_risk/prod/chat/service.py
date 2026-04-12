"""ChatService: orchestrates OpenAI calls, tool execution, and SSE streaming."""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

import duckdb
from openai import OpenAI
from sqlalchemy.orm import Session

from quant_risk.prod.auth.models import User
from quant_risk.prod.chat.prompts import SYSTEM_PROMPT
from quant_risk.prod.chat.tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from quant_risk.prod.serving.duckdb import ServingDB

logger = logging.getLogger(__name__)

MAX_TOOL_CYCLES = 10


def chat_stream(
    messages: list[dict[str, str]],
    user: User,
    db: Session,
    serving_db: ServingDB,
    research_db: duckdb.DuckDBPyConnection,
    api_key: str,
    model: str,
) -> Generator[str, None, None]:
    """Run the chat completion loop and yield SSE-formatted lines.

    Yields
    ------
    str
        Lines in SSE format: ``data: {"type": "...", ...}\\n\\n``
    """
    client = OpenAI(api_key=api_key)

    openai_messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for msg in messages:
        openai_messages.append({"role": msg["role"], "content": msg["content"]})

    for cycle in range(MAX_TOOL_CYCLES):
        response = client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=TOOL_DEFINITIONS,
            stream=True,
        )

        tool_calls_accum: dict[int, dict] = {}
        content_started = False

        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Accumulate tool calls across chunks
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_accum:
                        tool_calls_accum[idx] = {
                            "id": tc.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc.id:
                        tool_calls_accum[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls_accum[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls_accum[idx]["arguments"] += tc.function.arguments

            # Stream text content
            if delta.content:
                content_started = True
                yield _sse({"type": "token", "content": delta.content})

            # Check for finish
            finish = chunk.choices[0].finish_reason if chunk.choices else None
            if finish == "stop":
                yield _sse({"type": "done"})
                return
            if finish == "tool_calls":
                break  # Process tool calls below

        if not tool_calls_accum:
            # No tool calls and no stop — shouldn't happen, but handle gracefully
            yield _sse({"type": "done"})
            return

        # Build the assistant message with tool calls for the conversation
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls_accum.values()
            ],
        }
        openai_messages.append(assistant_msg)

        # Execute each tool call
        for tc in tool_calls_accum.values():
            name = tc["name"]
            yield _sse({"type": "tool", "name": name, "status": "start"})

            try:
                params = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                params = {}

            executor = TOOL_REGISTRY.get(name)
            if executor is None:
                result = {"error": f"Unknown tool '{name}'."}
            else:
                try:
                    result = executor(params, user, db, serving_db, research_db)
                except Exception as exc:
                    logger.exception("Tool '%s' failed", name)
                    result = {"error": f"Tool execution failed: {exc}"}

            yield _sse({"type": "tool", "name": name, "status": "end"})

            openai_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, default=str),
            })

    # Exhausted max cycles
    yield _sse({"type": "token", "content": "I've reached the maximum number of tool calls for this request. Please try a simpler question."})
    yield _sse({"type": "done"})


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"
