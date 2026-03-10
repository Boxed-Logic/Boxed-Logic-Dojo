from __future__ import annotations
import threading
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class Episode:
    id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[dict] = field(default_factory=list)
    completion_token_ids: list[list[int]] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    turn: int = 0

    def tool_calls(self) -> list[dict]:
        """All tool call dicts made across the episode, in order.

        Each entry is an OpenAI-format tool call dict:
        {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
        """
        calls = []
        for msg in self.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                calls.extend(msg["tool_calls"])
        return calls

    def tool_results(self) -> list[str]:
        """All tool result strings across the episode, in order."""
        return [
            msg["content"]
            for msg in self.messages
            if msg.get("role") == "tool"
        ]

    def num_turns(self) -> int:
        """Number of assistant turns taken (including tool-call turns)."""
        return sum(1 for msg in self.messages if msg.get("role") == "assistant")


class EpisodeStore:
    """Thread-safe key-value store keyed by episode_id."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}

    def get(self, ep_id: str) -> dict:
        with self._lock:
            return dict(self._data.get(ep_id, {}))

    def set(self, ep_id: str, key: str, val):
        with self._lock:
            if ep_id not in self._data:
                self._data[ep_id] = {}
            self._data[ep_id][key] = val

    def update(self, ep_id: str, updates: dict):
        with self._lock:
            if ep_id not in self._data:
                self._data[ep_id] = {}
            self._data[ep_id].update(updates)

    def clear(self, ep_id: str):
        with self._lock:
            self._data.pop(ep_id, None)
