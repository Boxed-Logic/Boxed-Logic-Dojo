"""Boxed-Logic Dojo: a minimal GRPO training library with tool-use support."""
from .config import GRPOConfig
from .episode import Episode, EpisodeStore
from .tools import ToolRegistry, tool, get_episode_id
from .logprobs import build_token_sequences, compute_logprobs, compute_ref_logprobs
from .loss import grpo_loss, normalize_per_group
from .trainer import GRPOTrainer
from .vllm_rollout import VLLMRolloutEngine
from .dataset import load_dataset

__all__ = [
    "GRPOConfig",
    "Episode",
    "EpisodeStore",
    "ToolRegistry",
    "tool",
    "get_episode_id",
    "VLLMRolloutEngine",
    "build_token_sequences",
    "compute_logprobs",
    "compute_ref_logprobs",
    "grpo_loss",
    "normalize_per_group",
    "GRPOTrainer",
    "load_dataset",
]
