from __future__ import annotations
import json
from contextvars import ContextVar
from typing import Callable
from .episode import EpisodeStore

_episode_id: ContextVar[str] = ContextVar("episode_id")


def get_episode_id() -> str:
    """Return the episode ID for the currently executing tool call."""
    return _episode_id.get()


def tool(name: str, description: str, parameters: dict):
    """Decorator that attaches OpenAI-format tool schema to the function."""
    def decorator(fn: Callable) -> Callable:
        fn._tool_name = name
        fn._tool_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        return fn
    return decorator


class ToolRegistry:
    """Registry of @tool-decorated callables with per-episode context injection."""

    def __init__(self, tools: list[Callable], store: EpisodeStore):
        """Register tools and bind them to the episode store.

        Args:
            tools: List of functions decorated with @tool.
            store: Shared EpisodeStore for per-episode state access.

        Raises:
            ValueError: If any function in ``tools`` is not decorated with @tool.
        """
        self._store = store
        self._tools: dict[str, Callable] = {}
        for fn in tools:
            if not hasattr(fn, "_tool_name"):
                raise ValueError(f"Function {fn.__name__!r} is not decorated with @tool")
            self._tools[fn._tool_name] = fn

    @property
    def schemas(self) -> list[dict]:
        """Return OpenAI-format tool schemas for all registered tools."""
        return [fn._tool_schema for fn in self._tools.values()]

    def execute(self, episode_id: str, tool_call_dict: dict) -> str:
        """Execute a tool call and return the result as a string.

        Sets the episode_id context variable so tools can call get_episode_id().

        Args:
            episode_id: ID of the episode making the tool call.
            tool_call_dict: OpenAI-format tool call dict with a "function" key.

        Returns:
            String result of the tool, or an error message string on failure.
        """
        fn_info = tool_call_dict.get("function", {})
        name = fn_info.get("name", "")
        raw_args = fn_info.get("arguments", "{}")

        if name not in self._tools:
            return f"Error: unknown tool '{name}'"

        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as e:
            return f"Error: invalid tool arguments JSON: {e}"

        token = _episode_id.set(episode_id)
        try:
            result = self._tools[name](**args)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
        finally:
            _episode_id.reset(token)
