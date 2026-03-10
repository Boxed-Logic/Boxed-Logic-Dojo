"""Reward functions for GSM8K math reasoning with tool use.

Three reward components encourage distinct behaviours:

  think_reward   — fractional compliance: fraction of assistant turns that
                   contain a <think>...</think> block.  Fractional rather
                   than positional so the signal degrades gracefully when
                   the model is still learning the format.

  tool_use_reward — binary: 1.0 if the model made at least one tool call
                   during the episode, 0.0 otherwise.

  answer_reward  — binary: 1.0 if the extracted final answer matches the
                   gold value within a 0.01 tolerance.

make_reward_fn() combines all three into a single callable accepted by
GRPOTrainer.  Total reward range is [0.0, 1.5]; relative scale only —
GRPOTrainer normalises rewards within each group via z-scoring.
"""
from __future__ import annotations
import re
from typing import Callable

from dojo.episode import Episode

_THINK_RE  = re.compile(r'<think>.*?</think>', re.DOTALL)
_ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
_NUMBER_RE = re.compile(r'-?[\d,]+\.?\d*')


def think_reward(episode: Episode) -> float:
    """Return the fraction of assistant turns that contain a <think> block.

    Tool-call assistant messages typically have content=None; these are
    guarded with ``or ""`` and count against compliance if they lack a
    think block, which is correct — the model should reason before calling
    a tool as well as before giving a final answer.

    Returns 0.0 if there are no assistant messages.
    """
    assistant_msgs = [m for m in episode.messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        return 0.0
    hits = sum(
        1 for msg in assistant_msgs
        if _THINK_RE.search(msg.get("content") or "")
    )
    return hits / len(assistant_msgs)


def tool_use_reward(episode: Episode) -> float:
    """Return 1.0 if the model made at least one tool call, else 0.0."""
    return 1.0 if episode.tool_calls() else 0.0


def extract_final_answer(episode: Episode) -> float | None:
    """Extract the model's final numeric answer from the last assistant message.

    Two strategies are tried in order:
      1. Look for an explicit <answer>...</answer> tag and parse its content.
      2. Fall back to the last number appearing anywhere in the message text.

    The fallback handles models that haven't yet learned the tagging format
    but still write the correct number at the end of their response.

    Returns None if there are no assistant messages, if the last assistant
    message has no text content (e.g. a tool-call-only turn), or if no
    number can be parsed.
    """
    # Find the last assistant message.  content=None means no assistant messages
    # exist at all; content="" means the last assistant turn had no text (e.g. a
    # tool-call-only turn) — both cases correctly return None below.
    content = None
    for msg in reversed(episode.messages):
        if msg.get("role") == "assistant":
            content = msg.get("content") or ""
            break
    if content is None:
        return None

    # Strategy 1: explicit <answer> tag.
    m = _ANSWER_RE.search(content)
    if m:
        try:
            return float(m.group(1).strip().replace(",", ""))
        except ValueError:
            pass

    # Strategy 2: last number in the response text.
    numbers = _NUMBER_RE.findall(content)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


def answer_reward(episode: Episode, gold: float) -> float:
    """Return 1.0 if the extracted final answer matches gold within 0.01."""
    pred = extract_final_answer(episode)
    return 1.0 if pred is not None and abs(pred - gold) < 0.01 else 0.0


def make_reward_fn(
    w_correct: float = 1.0,
    w_think: float = 0.3,
    w_tool: float = 0.2,
) -> Callable:
    """Return a reward function compatible with GRPOTrainer.

    The returned function has the signature:
        reward_fn(episodes: list[Episode], row: dict) -> list[float]

    Weights:
        w_correct  weight for answer_reward   (default 1.0)
        w_think    weight for think_reward     (default 0.3)
        w_tool     weight for tool_use_reward  (default 0.2)

    Total reward range: [0.0, 1.5].
    """
    def reward_fn(episodes: list[Episode], row: dict) -> list[float]:
        gold = float(row["answer"])
        return [
            w_correct * answer_reward(ep, gold)
            + w_think  * think_reward(ep)
            + w_tool   * tool_use_reward(ep)
            for ep in episodes
        ]
    return reward_fn
