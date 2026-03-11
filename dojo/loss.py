from __future__ import annotations
import torch
from torch import Tensor


def normalize_per_group(rewards: Tensor, group_size: int) -> Tensor:
    """Normalize rewards within each group of size `group_size`.

    Args:
        rewards: shape (N,) where N = num_prompts * group_size
        group_size: number of generations per prompt (G)

    Returns:
        advantages: shape (N,) — zero-mean, unit-variance within each group.
                    All-same-reward groups return 0 advantages.
    """
    n = rewards.shape[0]
    g = group_size
    assert n % g == 0, f"rewards length {n} not divisible by group_size {g}"
    grouped = rewards.view(-1, g)  # (num_prompts, G)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    advantages = (grouped - mean) / (std + 1e-8)
    return advantages.view(-1)


def grpo_loss(
    log_probs: Tensor,
    ref_log_probs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    epsilon: float = 0.2,
    beta: float = 0.0,
) -> Tensor:
    """Compute GRPO clipped policy gradient loss.

    Args:
        log_probs: (N, L) — per-token log-probs under current policy, 0 on non-completion tokens
        ref_log_probs: (N, L) — per-token log-probs under reference (base) policy
        advantages: (N,) — normalized group advantages
        mask: (N, L) — 1 on completion tokens, 0 elsewhere
        epsilon: clip ratio (default 0.2)
        beta: KL penalty coefficient (0 = disabled)

    Returns:
        scalar loss
    """
    ratio = torch.exp(log_probs - ref_log_probs)  # (N, L)
    ratio = torch.clamp(ratio, 1e-4, 10.0)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    adv = advantages.unsqueeze(1)  # (N, 1) — broadcasts to (N, L)
    pg = -torch.min(ratio * adv, clipped * adv)  # (N, L)

    if beta > 0:
        kl = log_probs - ref_log_probs
        per_token_loss = (pg + beta * kl) * mask
    else:
        per_token_loss = pg * mask

    return per_token_loss.sum() / (mask.sum() + 1e-8)
