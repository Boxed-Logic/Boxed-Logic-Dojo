from __future__ import annotations
from dataclasses import dataclass, field, fields, MISSING


@dataclass
class GRPOConfig:
    """Configuration for a GRPO training run.

    Required fields have no defaults; optional fields include vLLM settings,
    HF Hub upload, memory efficiency knobs, and W&B logging.
    """
    # Model
    model_name: str
    lora_rank: int
    lora_alpha: int
    target_modules: list

    # Rollout
    num_generations: int
    max_turns: int
    max_completion_length: int
    max_total_tokens: int

    # Training
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_epochs: int

    # GRPO
    epsilon: float
    beta: float
    temperature: float
    top_p: float

    # Misc
    seed: int
    device: str
    torch_dtype: str  # "bfloat16", "float16", "float32", or "auto"
    output_dir: str
    log_every: int

    # vLLM
    vllm_gpu_memory_utilization: float = 0.4  # leave room for training
    vllm_sync_every: int = 1  # sync weights every N gradient steps
    vllm_enable_sleep_mode: bool = True  # offload vLLM weights during train phases
    vllm_enable_lora: bool = True  # enable native LoRA serving; False for eval-only
    vllm_max_lora_rank: int | None = None  # defaults to lora_rank if None

    # Hugging Face Hub (optional — reads HF_TOKEN from env)
    hf_repo_id: str | None = None  # e.g. "myorg/my-model"; None = disabled
    hf_push_every: int = 0  # push every N gradient steps; 0 = disabled
    hf_push_final: bool = True  # push adapter weights at end of training

    # Memory efficiency: split logprob forward passes into microbatches.
    # With Mamba hybrid models the SSM chunk-scan creates a 6-D intermediate
    # tensor (b, chunks, chunk_size, state, heads, head_dim) that grows
    # linearly with batch size and can OOM even on large GPUs.
    # Set to None to process all episodes at once (old behavior).
    logprob_micro_batch_size: int | None = 4

    # Enable gradient checkpointing on the base model.
    # Recomputes activations during backward instead of storing them,
    # drastically reducing activation memory at the cost of ~30% extra compute.
    # Requires use_cache=False in compute_logprobs (already the case).
    use_gradient_checkpointing: bool = True

    # Weights & Biases (optional — reads WANDB_API_KEY from env)
    wandb_project: str | None = None  # e.g. "grpo-math"; None = disabled
    wandb_run_name: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "GRPOConfig":
        valid = {f.name for f in fields(cls)}
        provided = {k: v for k, v in d.items() if k in valid}
        required = {
            f.name for f in fields(cls)
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing = required - provided.keys()
        if missing:
            raise ValueError(
                f"GRPOConfig: missing required fields from config: {sorted(missing)}"
            )
        return cls(**provided)
