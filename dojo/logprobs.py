from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .episode import Episode


def build_token_sequences(
    episodes: list["Episode"],
    tokenizer,
    max_total_tokens: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build right-padded token sequences for the forward pass.

    For each episode, produces:
    - full token sequence (all messages)
    - completion mask: 1 only on assistant-turn tokens

    The mask is built incrementally per-message so that tool result tokens
    between assistant turns are correctly masked to 0.

    Returns:
        input_ids: (N, L) long
        completion_mask: (N, L) float — 1 on assistant tokens only
        attention_mask: (N, L) long — 1 on real tokens, 0 on padding
    """
    all_input_ids = []
    all_masks = []

    for ep in episodes:
        ids, mask = _build_episode_sequence(ep, tokenizer, max_total_tokens)
        all_input_ids.append(ids)
        all_masks.append(mask)

    max_len = max(ids.shape[0] for ids in all_input_ids)

    padded_input_ids = []
    padded_masks = []
    padded_attention = []
    pad_id = tokenizer.pad_token_id

    for ids, mask in zip(all_input_ids, all_masks):
        pad_len = max_len - ids.shape[0]
        real_len = ids.shape[0]

        padded_ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
        padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.float)])
        attn = torch.cat([torch.ones(real_len, dtype=torch.long),
                          torch.zeros(pad_len, dtype=torch.long)])

        padded_input_ids.append(padded_ids)
        padded_masks.append(padded_mask)
        padded_attention.append(attn)

    return (
        torch.stack(padded_input_ids),
        torch.stack(padded_masks),
        torch.stack(padded_attention),
    )


def _build_episode_sequence(
    ep: "Episode",
    tokenizer,
    max_total_tokens: int,
) -> tuple[Tensor, Tensor]:
    """Build ids and completion mask for a single episode.

    Strategy: tokenize prefix up to each message boundary to find exact
    assistant-turn token spans, then set mask=1 for those spans only.
    """
    messages = ep.messages
    if not messages:
        return torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.float)

    # Tokenize full sequence
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )
    if not isinstance(full_ids, list):
        full_ids = full_ids["input_ids"]
    full_ids = torch.tensor(full_ids, dtype=torch.long)

    mask = torch.zeros(len(full_ids), dtype=torch.float)

    # Find assistant turn spans by binary-tokenizing prefixes
    prefix_messages = []

    for i, msg in enumerate(messages):
        prefix_messages.append(msg)

        if msg["role"] == "assistant":
            # Tokenize everything up to and including this assistant message
            with_asst = tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors=None,
            )
            if not isinstance(with_asst, list):
                with_asst = with_asst["input_ids"]
            cur_len = len(with_asst)

            # Without this assistant message — use add_generation_prompt=True
            # to get the prompt prefix that precedes the assistant tokens
            without_asst = tokenizer.apply_chat_template(
                prefix_messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
            if not isinstance(without_asst, list):
                without_asst = without_asst["input_ids"]
            asst_start = len(without_asst)
            asst_end = cur_len

            if asst_start < asst_end and asst_end <= len(full_ids):
                mask[asst_start:asst_end] = 1.0

    # Truncate from left if over max_total_tokens (keep completions)
    if len(full_ids) > max_total_tokens:
        excess = len(full_ids) - max_total_tokens
        full_ids = full_ids[excess:]
        mask = mask[excess:]

    return full_ids, mask


def compute_logprobs(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
    completion_mask: Tensor,
) -> Tensor:
    """Compute per-token log-probs under the current model policy.

    Passes use_cache=False to avoid hybrid-cache issues with Mamba models and
    to ensure a clean forward pass with no residual recurrent state.

    Returns:
        (N, L) — log-probs on completion tokens, 0 elsewhere.
        Padded back to original length so positions align with input_ids.
    """
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = output.logits  # (N, L, V)

    shift_logits = logits[:, :-1, :]  # (N, L-1, V)
    shift_labels = input_ids[:, 1:]  # (N, L-1)
    shift_mask = completion_mask[:, 1:]  # (N, L-1)

    lp = F.log_softmax(shift_logits, dim=-1)
    lp = lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (N, L-1)
    lp = lp * shift_mask

    # Pad back to (N, L) so token positions align with input_ids
    lp = F.pad(lp, (1, 0), value=0.0)  # (N, L)
    return lp


def compute_ref_logprobs(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
    completion_mask: Tensor,
) -> Tensor:
    """Compute per-token log-probs under the reference (base) model.

    Uses PEFT's disable_adapter() context manager so no second model is needed.
    Must be called with base LoRA model (not merged).

    Note on hybrid Mamba models: we pass use_cache=False inside compute_logprobs,
    which means every call starts from fresh recurrent state. This avoids any
    SSM state contamination between policy and reference forward passes.
    """
    with torch.no_grad(), model.disable_adapter():
        return compute_logprobs(model, input_ids, attention_mask, completion_mask)
