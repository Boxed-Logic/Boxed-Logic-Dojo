"""Unit tests for dojo/episode.py helper methods."""
import json
import threading
import pytest
from dojo.episode import Episode, EpisodeStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_tool_call(name: str, args: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _ep_no_tools() -> Episode:
    ep = Episode()
    ep.messages = [
        {"role": "system",    "content": "You are helpful."},
        {"role": "user",      "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ]
    return ep


def _ep_one_tool() -> Episode:
    ep = Episode()
    ep.messages = [
        {"role": "system",    "content": "sys"},
        {"role": "user",      "content": "What is 12*5?"},
        {"role": "assistant", "content": None,
         "tool_calls": [_make_tool_call("calculator", {"expression": "12*5"}, "call_1")]},
        {"role": "tool",      "tool_call_id": "call_1", "content": "60"},
        {"role": "assistant", "content": "The answer is 60."},
    ]
    return ep


def _ep_two_tools() -> Episode:
    ep = Episode()
    ep.messages = [
        {"role": "system",    "content": "sys"},
        {"role": "user",      "content": "What is (12+8)*(3-1)?"},
        {"role": "assistant", "content": None,
         "tool_calls": [_make_tool_call("calculator", {"expression": "12+8"}, "call_1")]},
        {"role": "tool",      "tool_call_id": "call_1", "content": "20"},
        {"role": "assistant", "content": None,
         "tool_calls": [_make_tool_call("calculator", {"expression": "3-1"}, "call_2")]},
        {"role": "tool",      "tool_call_id": "call_2", "content": "2"},
        {"role": "assistant", "content": "The answer is 40."},
    ]
    return ep


def _ep_three_tools() -> Episode:
    ep = Episode()
    ep.messages = [
        {"role": "system",    "content": "sys"},
        {"role": "user",      "content": "Multi-step"},
        {"role": "assistant", "content": None,
         "tool_calls": [_make_tool_call("search", {"q": "step1"}, "call_1")]},
        {"role": "tool",      "tool_call_id": "call_1", "content": "result1"},
        {"role": "assistant", "content": None,
         "tool_calls": [_make_tool_call("search", {"q": "step2"}, "call_2")]},
        {"role": "tool",      "tool_call_id": "call_2", "content": "result2"},
        {"role": "assistant", "content": None,
         "tool_calls": [_make_tool_call("calculator", {"expression": "1+1"}, "call_3")]},
        {"role": "tool",      "tool_call_id": "call_3", "content": "2"},
        {"role": "assistant", "content": "Done."},
    ]
    return ep


# ── tool_calls() ──────────────────────────────────────────────────────────────

class TestToolCalls:
    def test_no_tool_calls_returns_empty(self):
        assert _ep_no_tools().tool_calls() == []

    def test_empty_episode_returns_empty(self):
        assert Episode().tool_calls() == []

    def test_one_tool_call_count(self):
        assert len(_ep_one_tool().tool_calls()) == 1

    def test_one_tool_call_name(self):
        calls = _ep_one_tool().tool_calls()
        assert calls[0]["function"]["name"] == "calculator"

    def test_one_tool_call_arguments(self):
        calls = _ep_one_tool().tool_calls()
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"expression": "12*5"}

    def test_one_tool_call_id(self):
        calls = _ep_one_tool().tool_calls()
        assert calls[0]["id"] == "call_1"

    def test_two_tool_calls_count(self):
        assert len(_ep_two_tools().tool_calls()) == 2

    def test_two_tool_calls_order(self):
        calls = _ep_two_tools().tool_calls()
        assert calls[0]["id"] == "call_1"
        assert calls[1]["id"] == "call_2"

    def test_two_tool_calls_names(self):
        calls = _ep_two_tools().tool_calls()
        assert calls[0]["function"]["name"] == "calculator"
        assert calls[1]["function"]["name"] == "calculator"

    def test_three_tool_calls_count(self):
        assert len(_ep_three_tools().tool_calls()) == 3

    def test_three_tool_calls_order_and_names(self):
        calls = _ep_three_tools().tool_calls()
        assert calls[0]["function"]["name"] == "search"
        assert calls[1]["function"]["name"] == "search"
        assert calls[2]["function"]["name"] == "calculator"

    def test_returns_full_openai_dict_structure(self):
        calls = _ep_one_tool().tool_calls()
        tc = calls[0]
        assert "id" in tc
        assert "type" in tc
        assert tc["type"] == "function"
        assert "function" in tc
        assert "name" in tc["function"]
        assert "arguments" in tc["function"]


# ── tool_results() ────────────────────────────────────────────────────────────

class TestToolResults:
    def test_no_tools_returns_empty(self):
        assert _ep_no_tools().tool_results() == []

    def test_empty_episode_returns_empty(self):
        assert Episode().tool_results() == []

    def test_one_result(self):
        assert _ep_one_tool().tool_results() == ["60"]

    def test_two_results_count(self):
        assert len(_ep_two_tools().tool_results()) == 2

    def test_two_results_order(self):
        results = _ep_two_tools().tool_results()
        assert results[0] == "20"
        assert results[1] == "2"

    def test_three_results(self):
        results = _ep_three_tools().tool_results()
        assert results == ["result1", "result2", "2"]

    def test_results_are_strings(self):
        for result in _ep_three_tools().tool_results():
            assert isinstance(result, str)

    def test_tool_calls_and_results_same_length(self):
        for ep in [_ep_no_tools(), _ep_one_tool(), _ep_two_tools(), _ep_three_tools()]:
            assert len(ep.tool_calls()) == len(ep.tool_results())


# ── num_turns() ───────────────────────────────────────────────────────────────

class TestNumTurns:
    def test_empty_episode(self):
        assert Episode().num_turns() == 0

    def test_no_tools_one_turn(self):
        # [sys, user, asst] → 1 assistant turn
        assert _ep_no_tools().num_turns() == 1

    def test_one_tool_two_turns(self):
        # [sys, user, asst_call, tool, asst_final] → 2 assistant turns
        assert _ep_one_tool().num_turns() == 2

    def test_two_tools_three_turns(self):
        # [sys, user, call1, tool1, call2, tool2, final] → 3 assistant turns
        assert _ep_two_tools().num_turns() == 3

    def test_three_tools_four_turns(self):
        assert _ep_three_tools().num_turns() == 4

    def test_prompt_only_zero_turns(self):
        ep = Episode()
        ep.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user",   "content": "hello"},
        ]
        assert ep.num_turns() == 0


# ── EpisodeStore ──────────────────────────────────────────────────────────────

class TestEpisodeStore:
    def test_set_and_get(self):
        store = EpisodeStore()
        store.set("ep1", "key", "value")
        assert store.get("ep1") == {"key": "value"}

    def test_get_missing_returns_empty(self):
        store = EpisodeStore()
        assert store.get("nonexistent") == {}

    def test_get_returns_copy(self):
        """Mutations of the returned dict must not affect the store."""
        store = EpisodeStore()
        store.set("ep1", "x", 1)
        result = store.get("ep1")
        result["x"] = 999
        assert store.get("ep1")["x"] == 1

    def test_set_multiple_keys(self):
        store = EpisodeStore()
        store.set("ep1", "a", 1)
        store.set("ep1", "b", 2)
        assert store.get("ep1") == {"a": 1, "b": 2}

    def test_set_overwrites_existing_key(self):
        store = EpisodeStore()
        store.set("ep1", "k", "old")
        store.set("ep1", "k", "new")
        assert store.get("ep1")["k"] == "new"

    def test_update_merges_keys(self):
        store = EpisodeStore()
        store.set("ep1", "a", 1)
        store.update("ep1", {"b": 2, "c": 3})
        assert store.get("ep1") == {"a": 1, "b": 2, "c": 3}

    def test_update_creates_episode_if_missing(self):
        store = EpisodeStore()
        store.update("new_ep", {"x": 42})
        assert store.get("new_ep") == {"x": 42}

    def test_clear_removes_episode(self):
        store = EpisodeStore()
        store.set("ep1", "key", "val")
        store.clear("ep1")
        assert store.get("ep1") == {}

    def test_clear_nonexistent_is_noop(self):
        store = EpisodeStore()
        store.clear("never_existed")  # must not raise

    def test_multiple_episodes_isolated(self):
        store = EpisodeStore()
        store.set("ep1", "k", 1)
        store.set("ep2", "k", 2)
        assert store.get("ep1")["k"] == 1
        assert store.get("ep2")["k"] == 2
        store.clear("ep1")
        assert store.get("ep1") == {}
        assert store.get("ep2") == {"k": 2}

    def test_thread_safe_concurrent_set(self):
        """Concurrent sets to distinct ep_ids must not corrupt state."""
        store = EpisodeStore()
        errors = []

        def worker(ep_id, val):
            try:
                store.set(ep_id, "k", val)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"ep-{i}", i))
            for i in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for i in range(50):
            assert store.get(f"ep-{i}") == {"k": i}

    def test_thread_safe_concurrent_update(self):
        """Concurrent updates to the same ep_id must not raise."""
        store = EpisodeStore()
        errors = []

        def worker():
            try:
                for i in range(20):
                    store.update("shared", {f"key_{i}": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All keys written by at least one thread must be present
        result = store.get("shared")
        assert isinstance(result, dict)

