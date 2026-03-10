"""Unit tests for dojo/rollout.py — no GPU required."""
import json
import pytest
from concurrent.futures import ThreadPoolExecutor

from dojo.episode import Episode, EpisodeStore
from dojo.rollout import _extract_tool_call, _try_parse_json, _build_tool_call_dict, process_tool_calls
from dojo.tools import ToolRegistry, tool


@tool(
    name="add",
    description="Return a + b",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    },
)
def _add_tool(a, b):
    return a + b


class TestExtractToolCall:
    def test_xml_tag_format(self):
        text = '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
        result = _extract_tool_call(text)
        assert result is not None
        assert result["function"]["name"] == "calculator"
        args = json.loads(result["function"]["arguments"])
        assert args["expression"] == "2+2"

    def test_xml_tag_with_surrounding_text(self):
        text = "Let me calculate that.\n<tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"3*4\"}}</tool_call>\nDone."
        result = _extract_tool_call(text)
        assert result is not None
        assert result["function"]["name"] == "calculator"

    def test_bare_json_format(self):
        text = 'Here is my call: {"name": "echo", "arguments": {"message": "hello"}}'
        result = _extract_tool_call(text)
        assert result is not None
        assert result["function"]["name"] == "echo"

    def test_no_tool_call(self):
        text = "The answer is 42. I don't need any tools."
        result = _extract_tool_call(text)
        assert result is None

    def test_malformed_json_in_xml_tags(self):
        text = "<tool_call>{not valid json}</tool_call>"
        result = _extract_tool_call(text)
        assert result is None

    def test_bare_json_missing_name(self):
        text = '{"arguments": {"x": 1}}'
        result = _extract_tool_call(text)
        assert result is None

    def test_bare_json_missing_arguments_still_needs_name(self):
        # Bare JSON without 'arguments' key should not match
        text = '{"name": "calculator"}'
        result = _extract_tool_call(text)
        assert result is None

    def test_xml_tag_missing_name(self):
        text = '<tool_call>{"arguments": {"expression": "1+1"}}</tool_call>'
        result = _extract_tool_call(text)
        assert result is None

    def test_tool_call_dict_structure(self):
        text = '<tool_call>{"name": "my_tool", "arguments": {"a": 1}}</tool_call>'
        result = _extract_tool_call(text)
        assert result is not None
        assert "id" in result
        assert result["id"].startswith("call_")
        assert result["type"] == "function"
        assert "function" in result
        assert "name" in result["function"]
        assert "arguments" in result["function"]

    def test_arguments_serialized_to_string(self):
        text = '<tool_call>{"name": "calc", "arguments": {"x": 5}}</tool_call>'
        result = _extract_tool_call(text)
        assert isinstance(result["function"]["arguments"], str)
        args = json.loads(result["function"]["arguments"])
        assert args["x"] == 5

    def test_multiline_xml_tag(self):
        text = """<tool_call>
{
  "name": "calculator",
  "arguments": {
    "expression": "100 / 4"
  }
}
</tool_call>"""
        result = _extract_tool_call(text)
        assert result is not None
        assert result["function"]["name"] == "calculator"


class TestTryParseJson:
    def test_valid_dict(self):
        result = _try_parse_json('{"a": 1}')
        assert result == {"a": 1}

    def test_invalid_json(self):
        result = _try_parse_json("{not json}")
        assert result is None

    def test_non_dict_json(self):
        result = _try_parse_json("[1, 2, 3]")
        assert result is None

    def test_empty_dict(self):
        result = _try_parse_json("{}")
        assert result == {}


class TestBuildToolCallDict:
    def test_basic_structure(self):
        payload = {"name": "my_fn", "arguments": {"x": 1}}
        result = _build_tool_call_dict(payload)
        assert result["type"] == "function"
        assert result["function"]["name"] == "my_fn"
        assert result["id"].startswith("call_")

    def test_unique_ids(self):
        payload = {"name": "fn", "arguments": {}}
        ids = {_build_tool_call_dict(payload)["id"] for _ in range(20)}
        assert len(ids) == 20, "Tool call IDs should be unique"

    def test_string_args_passthrough(self):
        payload = {"name": "fn", "arguments": '{"x": 1}'}
        result = _build_tool_call_dict(payload)
        assert result["function"]["arguments"] == '{"x": 1}'

    def test_dict_args_serialized(self):
        payload = {"name": "fn", "arguments": {"a": "b"}}
        result = _build_tool_call_dict(payload)
        assert json.loads(result["function"]["arguments"]) == {"a": "b"}


# ── process_tool_calls ────────────────────────────────────────────────────────

class TestProcessToolCalls:
    def _registry(self):
        return ToolRegistry([_add_tool], EpisodeStore())

    def _run(self, episodes, texts, token_ids, registry=None):
        if registry is None:
            registry = self._registry()
        with ThreadPoolExecutor(max_workers=4) as executor:
            process_tool_calls(episodes, texts, token_ids, registry, executor)

    # ── done / not-done ───────────────────────────────────────────────────────

    def test_no_tool_call_marks_done(self):
        ep = Episode()
        self._run([ep], ["Plain answer."], [[1, 2, 3]])
        assert ep.done is True

    def test_tool_call_does_not_mark_done(self):
        ep = Episode()
        text = '<tool_call>{"name": "add", "arguments": {"a": 3, "b": 4}}</tool_call>'
        self._run([ep], [text], [[1]])
        assert ep.done is False

    # ── message structure ─────────────────────────────────────────────────────

    def test_no_tool_call_appends_assistant_message(self):
        ep = Episode()
        self._run([ep], ["Hello!"], [[1]])
        assert ep.messages[-1] == {"role": "assistant", "content": "Hello!"}

    def test_tool_call_appends_assistant_with_tool_calls_field(self):
        ep = Episode()
        text = '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}}</tool_call>'
        self._run([ep], [text], [[1]])
        asst = next(m for m in ep.messages if m["role"] == "assistant")
        assert "tool_calls" in asst
        assert asst["tool_calls"][0]["function"]["name"] == "add"

    def test_tool_result_appended_after_execution(self):
        ep = Episode()
        text = '<tool_call>{"name": "add", "arguments": {"a": 5, "b": 6}}</tool_call>'
        self._run([ep], [text], [[1]])
        tool_msg = next((m for m in ep.messages if m["role"] == "tool"), None)
        assert tool_msg is not None
        assert tool_msg["content"] == "11"

    def test_tool_result_tool_call_id_matches_assistant(self):
        ep = Episode()
        text = '<tool_call>{"name": "add", "arguments": {"a": 0, "b": 0}}</tool_call>'
        self._run([ep], [text], [[1]])
        asst = next(m for m in ep.messages if m["role"] == "assistant")
        tool_msg = next(m for m in ep.messages if m["role"] == "tool")
        assert tool_msg["tool_call_id"] == asst["tool_calls"][0]["id"]

    # ── turn counter / token storage ──────────────────────────────────────────

    def test_turn_incremented(self):
        ep = Episode()
        assert ep.turn == 0
        self._run([ep], ["answer"], [[1]])
        assert ep.turn == 1

    def test_turn_incremented_for_tool_call(self):
        ep = Episode()
        text = '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 1}}</tool_call>'
        self._run([ep], [text], [[1]])
        assert ep.turn == 1

    def test_completion_token_ids_stored(self):
        ep = Episode()
        self._run([ep], ["answer"], [[10, 20, 30]])
        assert ep.completion_token_ids == [[10, 20, 30]]

    # ── tool error handling ───────────────────────────────────────────────────

    def test_tool_error_still_appends_result_message(self):
        """Tool errors are caught by ToolRegistry and returned as error strings."""
        ep = Episode()
        # Pass a string for 'a' so a+b raises TypeError inside the tool
        text = '<tool_call>{"name": "add", "arguments": {"a": "oops", "b": 1}}</tool_call>'
        self._run([ep], [text], [[1]])
        tool_msg = next((m for m in ep.messages if m["role"] == "tool"), None)
        assert tool_msg is not None
        assert isinstance(tool_msg["content"], str)
        assert "error" in tool_msg["content"].lower()

    # ── multiple episodes ─────────────────────────────────────────────────────

    def test_multiple_episodes_all_processed(self):
        eps = [Episode(), Episode()]
        self._run(eps, ["answer one", "answer two"], [[1], [2]])
        assert all(ep.done for ep in eps)
        assert eps[0].messages[-1]["content"] == "answer one"
        assert eps[1].messages[-1]["content"] == "answer two"

    def test_mixed_tool_and_plain_episodes(self):
        ep_plain = Episode()
        ep_tool = Episode()
        tool_text = '<tool_call>{"name": "add", "arguments": {"a": 2, "b": 3}}</tool_call>'
        self._run([ep_plain, ep_tool], ["no tool here", tool_text], [[1], [2]])
        assert ep_plain.done is True
        assert ep_tool.done is False
        assert any(m["role"] == "tool" for m in ep_tool.messages)
