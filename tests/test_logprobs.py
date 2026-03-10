"""Unit tests for dojo/logprobs.py — no GPU required.

MockTokenizer design
--------------------
Every message produces exactly 3 tokens:  [ROLE_MARKER, content_tok, ROLE_END]
add_generation_prompt appends:            [ASST_HEADER]   (1 token)

Token IDs:
  10=SYS_MARKER  11=SYS_END
  20=USR_MARKER  21=USR_END
  30=ASST_HEADER 31=ASST_END   ← 30 is emitted both by template AND add_generation_prompt
  40=TOOL_MARKER 41=TOOL_END
  content tokens: 100+

Because ASST_HEADER(30) is the last token emitted by add_generation_prompt,
the assistant *content* span begins at position len(without_asst_with_gen_prompt)
and ends at len(with_asst).  That span = [content_tok, ASST_END] = 2 tokens.
ASST_HEADER itself is always masked 0.

Position arithmetic for any episode
------------------------------------
Each non-assistant, non-tool message block = 3 tokens
Each assistant block = 3 tokens, mask=1 on last 2 (content + ASST_END)
Each tool block      = 3 tokens, mask=0 on all 3
add_generation_prompt = +1 token (ASST_HEADER)

So for a sequence [sys, user, asst1, tool1, asst2, tool2, asst3]:
  Block offsets:  0   3     6      9     12     15     18
  Lengths:        3   3     3      3     3      3      3   = 21 total

  asst1 span (mask=1): [7, 8]     (positions within full sequence)
  asst2 span (mask=1): [13, 14]
  asst3 span (mask=1): [19, 20]

  tool spans (mask=0): [9-11], [15-17]
"""
import pytest
import torch

from dojo.episode import Episode
from dojo.logprobs import _build_episode_sequence, build_token_sequences

# ── Token ID constants ────────────────────────────────────────────────────────
SYS_MARKER  = 10;  SYS_END   = 11
USR_MARKER  = 20;  USR_END   = 21
ASST_HEADER = 30;  ASST_END  = 31
TOOL_MARKER = 40;  TOOL_END  = 41
PAD         = 0


class MockTokenizer:
    """Fixed 3-token-per-message tokenizer for deterministic position tests.

    Every message → [ROLE_MARKER, content_tok, ROLE_END]
    add_generation_prompt → appends [ASST_HEADER]

    Content is encoded as a single token: 100 + stable index derived from string.
    The exact content token value is irrelevant to the mask; what matters is that
    every message block is exactly 3 tokens so positions are fully predictable.
    """

    def __init__(self):
        self.pad_token_id = PAD
        self.eos_token_id = 1
        self._registry: dict[str, int] = {}
        self._next = 100

    def _content_tok(self, s: str) -> int:
        if s not in self._registry:
            self._registry[s] = self._next
            self._next += 1
        return self._registry[s]

    def _msg_content_key(self, msg: dict) -> str:
        role = msg.get("role", "")
        content = msg.get("content")
        if content is not None:
            return f"{role}:{str(content)}"
        # tool_calls assistant: encode by tool name
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            name = tool_calls[0].get("function", {}).get("name", "unknown")
            return f"{role}:call:{name}"
        return f"{role}:empty"

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        tools=None,
    ) -> list[int]:
        tokens = []
        for msg in messages:
            role = msg.get("role", "")
            ctok = self._content_tok(self._msg_content_key(msg))
            if role == "system":
                tokens += [SYS_MARKER, ctok, SYS_END]
            elif role == "user":
                tokens += [USR_MARKER, ctok, USR_END]
            elif role == "assistant":
                tokens += [ASST_HEADER, ctok, ASST_END]
            elif role == "tool":
                tokens += [TOOL_MARKER, ctok, TOOL_END]
        if add_generation_prompt:
            tokens += [ASST_HEADER]
        return tokens


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ep(messages: list[dict]) -> Episode:
    ep = Episode()
    ep.messages = messages
    return ep


def _sys():
    return {"role": "system", "content": "You are helpful."}


def _usr(text="What is 2+2?"):
    return {"role": "user", "content": text}


def _asst(text):
    return {"role": "assistant", "content": text}


def _asst_call(tool_name, args):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_{tool_name}",
            "type": "function",
            "function": {"name": tool_name, "arguments": str(args)},
        }],
    }


def _tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": str(content)}


# ── Single-turn (no tool calls) ───────────────────────────────────────────────

class TestSingleTurn:
    """[sys, user, asst]
    Full sequence (9 tokens):
      pos: 0=SYS_M  1=sys_c  2=SYS_E
           3=USR_M  4=usr_c  5=USR_E
           6=ASST_H 7=asst_c 8=ASST_E
    Expected mask: [0,0,0,0,0,0,0,1,1]
    """

    def setup_method(self):
        self.tok = MockTokenizer()
        self.ep = _ep([_sys(), _usr(), _asst("4")])
        self.ids, self.mask = _build_episode_sequence(self.ep, self.tok, 512)

    def test_sequence_length(self):
        assert len(self.ids) == 9

    def test_exact_mask(self):
        expected = torch.tensor([0,0,0,0,0,0,0,1,1], dtype=torch.float)
        assert torch.equal(self.mask, expected), f"mask={self.mask.tolist()}"

    def test_prompt_tokens_are_zero(self):
        assert self.mask[:6].sum() == 0   # system + user blocks

    def test_asst_header_is_zero(self):
        assert self.mask[6].item() == 0   # ASST_HEADER(30) not masked

    def test_asst_content_and_end_are_one(self):
        assert self.mask[7].item() == 1   # content token
        assert self.mask[8].item() == 1   # ASST_END

    def test_token_ids_correct(self):
        ids = self.ids.tolist()
        assert ids[0] == SYS_MARKER
        assert ids[2] == SYS_END
        assert ids[3] == USR_MARKER
        assert ids[5] == USR_END
        assert ids[6] == ASST_HEADER
        assert ids[8] == ASST_END


# ── Single tool call (one tool round-trip) ────────────────────────────────────

class TestOneToolCall:
    """[sys, user, asst_call, tool_result, asst_final]
    Full sequence (15 tokens):
      pos: 0-2   sys block
           3-5   user block
           6-8   asst_call block    → mask[7:9]=1
           9-11  tool_result block  → mask=0
           12-14 asst_final block   → mask[13:15]=1
    Expected mask: [0,0,0,0,0,0,0,1,1,0,0,0,0,1,1]
    """

    def setup_method(self):
        self.tok = MockTokenizer()
        messages = [
            _sys(),
            _usr(),
            _asst_call("calculator", {"expression": "2+2"}),
            _tool_result("call_calculator", "4"),
            _asst("The answer is 4"),
        ]
        self.ep = _ep(messages)
        self.ids, self.mask = _build_episode_sequence(self.ep, self.tok, 512)

    def test_sequence_length(self):
        assert len(self.ids) == 15

    def test_exact_mask(self):
        expected = torch.tensor([0,0,0,0,0,0,0,1,1,0,0,0,0,1,1], dtype=torch.float)
        assert torch.equal(self.mask, expected), f"mask={self.mask.tolist()}"

    def test_tool_call_assistant_is_masked(self):
        """The model wrote the tool call — it must receive gradient."""
        assert self.mask[7].item() == 1   # tool call content token
        assert self.mask[8].item() == 1   # ASST_END after tool call

    def test_tool_call_header_not_masked(self):
        assert self.mask[6].item() == 0   # ASST_HEADER before tool call

    def test_tool_result_tokens_are_zero(self):
        """Tool result came from the environment — must NOT get gradient."""
        assert self.mask[9:12].sum() == 0

    def test_final_answer_is_masked(self):
        assert self.mask[13].item() == 1
        assert self.mask[14].item() == 1

    def test_final_answer_header_not_masked(self):
        assert self.mask[12].item() == 0

    def test_masked_token_count(self):
        # 2 assistant turns × 2 masked tokens each = 4
        assert self.mask.sum().item() == 4


# ── Two tool calls ────────────────────────────────────────────────────────────

class TestTwoToolCalls:
    """[sys, user, asst_call1, tool1, asst_call2, tool2, asst_final]
    Full sequence (21 tokens):
      pos: 0-2   sys block
           3-5   user block
           6-8   asst_call1 block   → mask[7:9]=1
           9-11  tool1 block        → mask=0
           12-14 asst_call2 block   → mask[13:15]=1
           15-17 tool2 block        → mask=0
           18-20 asst_final block   → mask[19:21]=1
    Expected mask: [0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1]
    """

    def setup_method(self):
        self.tok = MockTokenizer()
        messages = [
            _sys(),
            _usr("What is (12+8) * (3-1)?"),
            _asst_call("calculator", {"expression": "12+8"}),
            _tool_result("call_calculator", "20"),
            _asst_call("calculator", {"expression": "3-1"}),
            _tool_result("call_calculator", "2"),
            _asst("The answer is 40"),
        ]
        self.ep = _ep(messages)
        self.ids, self.mask = _build_episode_sequence(self.ep, self.tok, 512)

    def test_sequence_length(self):
        assert len(self.ids) == 21

    def test_exact_mask(self):
        expected = torch.tensor(
            [0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1], dtype=torch.float
        )
        assert torch.equal(self.mask, expected), f"mask={self.mask.tolist()}"

    def test_first_tool_call_masked(self):
        assert self.mask[6].item() == 0   # header
        assert self.mask[7].item() == 1   # content
        assert self.mask[8].item() == 1   # end

    def test_first_tool_result_not_masked(self):
        assert self.mask[9:12].sum() == 0

    def test_second_tool_call_masked(self):
        assert self.mask[12].item() == 0  # header
        assert self.mask[13].item() == 1  # content
        assert self.mask[14].item() == 1  # end

    def test_second_tool_result_not_masked(self):
        assert self.mask[15:18].sum() == 0

    def test_final_answer_masked(self):
        assert self.mask[18].item() == 0  # header
        assert self.mask[19].item() == 1  # content
        assert self.mask[20].item() == 1  # end

    def test_masked_token_count(self):
        # 3 assistant turns × 2 masked tokens each = 6
        assert self.mask.sum().item() == 6

    def test_no_tool_headers_masked(self):
        """ASST_HEADER tokens (pos 6, 12, 18) must all be 0."""
        for pos in [6, 12, 18]:
            assert self.mask[pos].item() == 0, f"ASST_HEADER at pos {pos} should be 0"


# ── Three tool calls ──────────────────────────────────────────────────────────

class TestThreeToolCalls:
    """[sys, user, call1, tool1, call2, tool2, call3, tool3, final]
    Full sequence (27 tokens), 4 assistant turns.
    Expected masked positions: [7,8], [13,14], [19,20], [25,26]
    """

    def setup_method(self):
        self.tok = MockTokenizer()
        messages = [
            _sys(),
            _usr("Multi-step problem"),
            _asst_call("search", {"q": "step1"}),
            _tool_result("call_search", "result1"),
            _asst_call("search", {"q": "step2"}),
            _tool_result("call_search", "result2"),
            _asst_call("search", {"q": "step3"}),
            _tool_result("call_search", "result3"),
            _asst("Final answer here"),
        ]
        self.ep = _ep(messages)
        self.ids, self.mask = _build_episode_sequence(self.ep, self.tok, 512)

    def test_sequence_length(self):
        assert len(self.ids) == 27

    def test_exact_mask(self):
        expected = torch.zeros(27)
        for span_start in [7, 13, 19, 25]:
            expected[span_start:span_start + 2] = 1.0
        assert torch.equal(self.mask, expected), f"mask={self.mask.tolist()}"

    def test_masked_token_count(self):
        # 4 assistant turns × 2 = 8
        assert self.mask.sum().item() == 8

    def test_all_tool_result_blocks_zero(self):
        for block_start in [9, 15, 21]:
            assert self.mask[block_start:block_start + 3].sum() == 0, \
                f"tool block at {block_start} has non-zero mask"

    def test_all_asst_headers_zero(self):
        for pos in [6, 12, 18, 24]:
            assert self.mask[pos].item() == 0, f"ASST_HEADER at {pos} should be masked 0"


# ── Truncation ────────────────────────────────────────────────────────────────

class TestTruncation:
    """When the sequence exceeds max_total_tokens, it should be truncated from
    the LEFT (keeping the most recent / completion tokens)."""

    def test_truncated_length_respected(self):
        tok = MockTokenizer()
        ep = _ep([_sys(), _usr(), _asst("ans")])
        # Full length = 9; truncate to 5 → keeps last 5 tokens
        ids, mask = _build_episode_sequence(ep, tok, max_total_tokens=5)
        assert len(ids) == 5
        assert len(mask) == 5

    def test_truncation_keeps_completion_tokens(self):
        tok = MockTokenizer()
        ep = _ep([_sys(), _usr(), _asst("ans")])
        # Full: 9 tokens, mask=[0,0,0,0,0,0,0,1,1]
        # Truncated to 5: keeps last 5 tokens = positions [4..8]
        # Original mask[4:9] = [0,0,0,1,1]
        ids, mask = _build_episode_sequence(ep, tok, max_total_tokens=5)
        expected = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float)
        assert torch.equal(mask, expected), f"mask={mask.tolist()}"

    def test_no_truncation_when_within_limit(self):
        tok = MockTokenizer()
        ep = _ep([_sys(), _usr(), _asst("ans")])
        ids, mask = _build_episode_sequence(ep, tok, max_total_tokens=512)
        assert len(ids) == 9


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_episode(self):
        tok = MockTokenizer()
        ep = Episode()
        ids, mask = _build_episode_sequence(ep, tok, 512)
        assert ids.shape == mask.shape

    def test_prompt_only_no_assistant(self):
        """Episode with only system+user (no assistant turn yet) → mask all zeros."""
        tok = MockTokenizer()
        ep = _ep([_sys(), _usr()])
        ids, mask = _build_episode_sequence(ep, tok, 512)
        assert mask.sum().item() == 0

    def test_multiple_user_turns(self):
        """Interleaved user/assistant without tools."""
        tok = MockTokenizer()
        ep = _ep([
            _sys(),
            _usr("Q1"), _asst("A1"),
            _usr("Q2"), _asst("A2"),
        ])
        ids, mask = _build_episode_sequence(ep, tok, 512)
        # seq: [sys][usr][asst1][usr][asst2] = 5*3=15 tokens
        assert len(ids) == 15
        expected = torch.tensor([0,0,0,0,0,0,0,1,1,0,0,0,0,1,1], dtype=torch.float)
        assert torch.equal(mask, expected), f"mask={mask.tolist()}"


# ── Batching (build_token_sequences) ─────────────────────────────────────────

class TestBatchTokenSequences:
    """Test that build_token_sequences correctly right-pads and stacks episodes."""

    def test_output_shapes(self):
        tok = MockTokenizer()
        episodes = [
            _ep([_sys(), _usr(), _asst("A")]),
            _ep([_sys(), _usr(), _asst_call("calc", {}), _tool_result("x", "4"), _asst("B")]),
        ]
        input_ids, comp_mask, attn_mask = build_token_sequences(episodes, tok, 512)
        assert input_ids.shape[0] == 2
        assert comp_mask.shape[0] == 2
        assert attn_mask.shape[0] == 2
        assert input_ids.shape[1] == comp_mask.shape[1] == attn_mask.shape[1]

    def test_shorter_episode_is_right_padded(self):
        """Shorter episode should have PAD tokens at the end, not the start."""
        tok = MockTokenizer()
        short_ep = _ep([_sys(), _usr(), _asst("A")])        # 9 tokens
        long_ep  = _ep([_sys(), _usr(),
                         _asst_call("c", {}),
                         _tool_result("x", "4"),
                         _asst("B")])                        # 15 tokens

        input_ids, comp_mask, attn_mask = build_token_sequences([short_ep, long_ep], tok, 512)

        # short_ep (row 0) padded to length 15: last 6 positions should be PAD
        assert (input_ids[0, 9:] == PAD).all(), \
            f"expected PAD after position 9, got {input_ids[0, 9:].tolist()}"

    def test_padding_has_zero_attention(self):
        tok = MockTokenizer()
        short_ep = _ep([_sys(), _usr(), _asst("A")])
        long_ep  = _ep([_sys(), _usr(), _asst_call("c", {}), _tool_result("x", "4"), _asst("B")])
        _, _, attn_mask = build_token_sequences([short_ep, long_ep], tok, 512)

        # Short episode: first 9 positions attend, last 6 are 0
        assert attn_mask[0, :9].sum() == 9
        assert attn_mask[0, 9:].sum() == 0

    def test_padding_has_zero_completion_mask(self):
        tok = MockTokenizer()
        short_ep = _ep([_sys(), _usr(), _asst("A")])
        long_ep  = _ep([_sys(), _usr(), _asst_call("c", {}), _tool_result("x", "4"), _asst("B")])
        _, comp_mask, _ = build_token_sequences([short_ep, long_ep], tok, 512)
        assert comp_mask[0, 9:].sum() == 0

    def test_mask_values_preserved_after_batching(self):
        """Mask values for each episode should match what _build_episode_sequence returns."""
        tok = MockTokenizer()
        ep1 = _ep([_sys(), _usr(), _asst("A")])
        ep2 = _ep([_sys(), _usr(), _asst_call("c", {}), _tool_result("x", "4"), _asst("B")])

        _, comp_mask, _ = build_token_sequences([ep1, ep2], tok, 512)

        # ep1 mask (padded to 15): [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0]
        ep1_expected = torch.tensor([0,0,0,0,0,0,0,1,1,0,0,0,0,0,0], dtype=torch.float)
        assert torch.equal(comp_mask[0], ep1_expected), f"ep1 mask={comp_mask[0].tolist()}"

        # ep2 mask (15 tokens, no padding): [0,0,0,0,0,0,0,1,1,0,0,0,0,1,1]
        ep2_expected = torch.tensor([0,0,0,0,0,0,0,1,1,0,0,0,0,1,1], dtype=torch.float)
        assert torch.equal(comp_mask[1], ep2_expected), f"ep2 mask={comp_mask[1].tolist()}"

    def test_dtypes(self):
        tok = MockTokenizer()
        eps = [_ep([_sys(), _usr(), _asst("A")])]
        input_ids, comp_mask, attn_mask = build_token_sequences(eps, tok, 512)
        assert input_ids.dtype == torch.long
        assert comp_mask.dtype == torch.float
        assert attn_mask.dtype == torch.long
