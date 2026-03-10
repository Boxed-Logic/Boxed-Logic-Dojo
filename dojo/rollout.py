from __future__ import annotations
import json
import re
from concurrent.futures import ThreadPoolExecutor, wait
from uuid import uuid4

from .episode import Episode
from .tools import ToolRegistry


def process_tool_calls(
    active: list[Episode],
    decoded_texts: list[str],
    token_id_lists: list[list[int]],
    registry: ToolRegistry,
    executor: ThreadPoolExecutor,
) -> None:
    """Shared tool-call processing for rollout engines.

    For each active episode, detects tool calls in decoded text, appends
    messages, executes tools via the thread pool, and appends results.
    """
    tool_futures = []

    for ep, decoded, token_ids in zip(active, decoded_texts, token_id_lists):
        ep.completion_token_ids.append(token_ids)
        ep.turn += 1

        tool_call = _extract_tool_call(decoded)

        if tool_call is not None:
            ep.messages.append({
                "role": "assistant",
                "content": decoded,
                "tool_calls": [tool_call],
            })
            future = executor.submit(registry.execute, ep.id, tool_call)
            tool_futures.append((ep, tool_call, future))
        else:
            ep.messages.append({"role": "assistant", "content": decoded})
            ep.done = True

    if tool_futures:
        wait([f for _, _, f in tool_futures])
        for ep, tool_call, future in tool_futures:
            try:
                result = future.result()
            except Exception as e:
                result = f"Error: {e}"
            ep.messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            })


# ──────────────────────────────────────────────────────────────────────────────
# Tool call parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

_XML_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _extract_tool_call(text: str) -> dict | None:
    """Try to parse a tool call from model output.

    Attempts:
    1. <tool_call>...</tool_call> XML tags → JSON payload
    2. Bare JSON object with 'name' and 'arguments' keys

    Returns an OpenAI-format tool call dict, or None.
    """
    # Strategy 1: XML tags
    match = _XML_PATTERN.search(text)
    if match:
        payload = _try_parse_json(match.group(1).strip())
        if payload and "name" in payload:
            return _build_tool_call_dict(payload)

    # Strategy 2: bare JSON (find first '{')
    brace_idx = text.find("{")
    if brace_idx != -1:
        payload = _try_parse_json(text[brace_idx:])
        if payload and "name" in payload and "arguments" in payload:
            return _build_tool_call_dict(payload)

    return None


def _try_parse_json(s: str) -> dict | None:
    try:
        result = json.loads(s)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    return None


def _build_tool_call_dict(payload: dict) -> dict:
    args = payload.get("arguments", {})
    return {
        "id": f"call_{uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": payload["name"],
            "arguments": json.dumps(args) if not isinstance(args, str) else args,
        },
    }
