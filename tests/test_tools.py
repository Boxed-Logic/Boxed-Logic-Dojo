"""Unit tests for dojo/tools.py — no GPU required."""
import json
import threading
import time
import pytest

from dojo.episode import EpisodeStore
from dojo.tools import tool, ToolRegistry, get_episode_id, _episode_id


@tool(
    name="echo",
    description="Echo the input back",
    parameters={
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    },
)
def echo_tool(message: str) -> str:
    return f"echo:{message}"


@tool(
    name="id_checker",
    description="Returns the current episode_id from ContextVar",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
def id_checker_tool() -> str:
    return get_episode_id()


class TestToolDecorator:
    def test_schema_attached(self):
        assert hasattr(echo_tool, "_tool_schema")
        assert echo_tool._tool_schema["function"]["name"] == "echo"

    def test_tool_name_attached(self):
        assert echo_tool._tool_name == "echo"

    def test_fn_still_callable(self):
        assert echo_tool(message="hello") == "echo:hello"


class TestToolRegistry:
    def _make_registry(self, tools=None):
        store = EpisodeStore()
        return ToolRegistry(tools or [echo_tool], store), store

    def test_schemas_property(self):
        registry, _ = self._make_registry()
        schemas = registry.schemas
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "echo"

    def test_execute_known_tool(self):
        registry, _ = self._make_registry()
        tool_call = {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "echo", "arguments": json.dumps({"message": "world"})},
        }
        result = registry.execute("ep-1", tool_call)
        assert result == "echo:world"

    def test_execute_unknown_tool(self):
        registry, _ = self._make_registry()
        tool_call = {
            "id": "call_xyz",
            "type": "function",
            "function": {"name": "nonexistent", "arguments": "{}"},
        }
        result = registry.execute("ep-1", tool_call)
        assert "unknown tool" in result.lower() or "error" in result.lower()

    def test_execute_malformed_json_args(self):
        registry, _ = self._make_registry()
        tool_call = {
            "id": "call_bad",
            "type": "function",
            "function": {"name": "echo", "arguments": "{not valid json"},
        }
        result = registry.execute("ep-1", tool_call)
        assert "error" in result.lower()

    def test_undecorated_function_raises(self):
        def raw_fn():
            pass
        store = EpisodeStore()
        with pytest.raises(ValueError, match="not decorated"):
            ToolRegistry([raw_fn], store)

    def test_contextvar_set_during_execution(self):
        store = EpisodeStore()
        registry = ToolRegistry([id_checker_tool], store)
        tool_call = {
            "id": "call_id",
            "type": "function",
            "function": {"name": "id_checker", "arguments": "{}"},
        }
        result = registry.execute("episode-42", tool_call)
        assert result == "episode-42"

    def test_contextvar_reset_after_execution(self):
        """ContextVar must be reset after tool execution (thread-safety)."""
        store = EpisodeStore()
        registry = ToolRegistry([id_checker_tool], store)
        tool_call = {
            "id": "call_id",
            "type": "function",
            "function": {"name": "id_checker", "arguments": "{}"},
        }
        # Set a different value in the current context before the call
        token = _episode_id.set("outer-context")
        try:
            result = registry.execute("episode-99", tool_call)
            assert result == "episode-99"
            # After execute, the ContextVar should be reset to "outer-context"
            assert get_episode_id() == "outer-context"
        finally:
            _episode_id.reset(token)

    def test_contextvar_isolation_across_threads(self):
        """Each thread gets its own ContextVar value."""
        store = EpisodeStore()
        registry = ToolRegistry([id_checker_tool], store)
        tool_call_template = {
            "id": "call_t",
            "type": "function",
            "function": {"name": "id_checker", "arguments": "{}"},
        }
        results = {}
        errors = []

        def worker(ep_id):
            try:
                tc = {**tool_call_template}
                result = registry.execute(ep_id, tc)
                results[ep_id] = result
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"ep-{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for i in range(10):
            assert results[f"ep-{i}"] == f"ep-{i}", f"Thread isolation failed for ep-{i}"
