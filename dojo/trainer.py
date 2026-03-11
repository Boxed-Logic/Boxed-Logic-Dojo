from __future__ import annotations
import dataclasses
import logging
import os
import shutil
from pathlib import Path
from typing import Callable

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GRPOConfig
from .episode import Episode, EpisodeStore
from .logprobs import build_token_sequences, compute_logprobs, compute_ref_logprobs
from .loss import grpo_loss, normalize_per_group
from .tools import ToolRegistry
from .vllm_rollout import VLLMRolloutEngine

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """End-to-end GRPO trainer: rollout → reward → loss → optimizer step."""

    def __init__(
        self,
        config: GRPOConfig,
        tools: list[Callable],
        reward_fn: Callable[[list[Episode], dict], list[float]],
        episode_init_fn: Callable | None = None,
    ):
        """Initialize model, tokenizer, optimizer, vLLM engine, and logging.

        Args:
            config: Full GRPO training configuration.
            tools: List of @tool-decorated callables available to the model.
            reward_fn: Maps (episodes, dataset_row) → per-episode reward floats.
            episode_init_fn: Optional callback invoked with (episode_id, row) at
                the start of each episode to populate the episode store.
        """
        self.config = config
        self.reward_fn = reward_fn
        self.episode_init_fn = episode_init_fn

        torch.manual_seed(config.seed)

        # ── Tokenizer ────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ── Model (LoRA) ──────────────────────────────────────────────────────
        _DTYPE_MAP = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        if config.torch_dtype not in _DTYPE_MAP:
            raise ValueError(
                f"torch_dtype must be one of {list(_DTYPE_MAP)}, got {config.torch_dtype!r}"
            )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=_DTYPE_MAP[config.torch_dtype],
            device_map=config.device,
            trust_remote_code=True,
        )

        if config.use_gradient_checkpointing:
            # Must be enabled before PEFT wrapping.
            # use_reentrant=False avoids issues with ContextVar / custom autograd.
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        lora_cfg = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_cfg)
        if config.use_gradient_checkpointing:
            # Required for PEFT + gradient checkpointing: ensures input
            # embeddings have requires_grad=True so the graph is connected.
            self.model.enable_input_require_grads()
        self.model.print_trainable_parameters()

        # ── Optimizer (LoRA params only) ──────────────────────────────────────
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.learning_rate)

        # ── Support infrastructure ────────────────────────────────────────────
        self.store = EpisodeStore()
        self.registry = ToolRegistry(tools, self.store)

        # ── Weights & Biases ──────────────────────────────────────────────────
        self._wandb = None
        if config.wandb_project:
            import wandb
            wandb.login(key=os.environ["WANDB_API_KEY"])
            self._wandb = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=dataclasses.asdict(config),
            )

        # ── Hugging Face Hub ──────────────────────────────────────────────────
        self._hf_api = None
        if config.hf_repo_id:
            from huggingface_hub import HfApi
            self._hf_api = HfApi(token=os.environ["HF_TOKEN"])

        self.rollout_engine = VLLMRolloutEngine(
            tokenizer=self.tokenizer,
            config=config,
            registry=self.registry,
            store=self.store,
            episode_init_fn=episode_init_fn,
        )

    def train(self, dataset):
        """Run the full GRPO training loop over ``dataset``.

        Iterates for ``config.num_epochs`` epochs, processing the dataset in
        batches. Each batch runs rollout, reward scoring, advantage normalization,
        microbatched log-prob computation, and an optimizer step.

        Args:
            dataset: HuggingFace Dataset (or list of dicts) with at least
                ``system_prompt`` and ``prompt`` fields per row.
        """
        cfg = self.config
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        global_step = 0
        self.optimizer.zero_grad()

        for epoch in range(cfg.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{cfg.num_epochs}")

            # Iterate dataset in batches of batch_size rows
            for batch_start in range(0, len(dataset), cfg.batch_size):
                batch_rows = dataset[batch_start : batch_start + cfg.batch_size]
                batch_rows = _rows_to_list(batch_rows)

                # ── Phase 1: Rollout ──────────────────────────────────────────
                self.model.eval()

                if cfg.vllm_enable_sleep_mode:
                    self.rollout_engine.wake_up()

                all_episode_groups = self.rollout_engine.rollout_batch(batch_rows)
                for episodes, row in zip(all_episode_groups, batch_rows):
                    rewards = self.reward_fn(episodes, row)
                    for ep, r in zip(episodes, rewards):
                        ep.reward = r

                if cfg.vllm_enable_sleep_mode:
                    self.rollout_engine.sleep()

                flat_episodes = [ep for group in all_episode_groups for ep in group]

                if not flat_episodes:
                    continue

                # ── Phase 2: Token sequences ──────────────────────────────────
                input_ids, comp_mask, attn_mask = build_token_sequences(
                    flat_episodes, self.tokenizer, cfg.max_total_tokens
                )
                device = next(self.model.parameters()).device
                input_ids = input_ids.to(device)
                comp_mask = comp_mask.to(device)
                attn_mask = attn_mask.to(device)

                # ── Phase 3: Advantages (computed before model forward) ───────
                rewards_t = torch.tensor(
                    [ep.reward for ep in flat_episodes],
                    dtype=torch.float32,
                    device=device,
                )
                advantages = normalize_per_group(rewards_t, cfg.num_generations)

                # ── Phase 4: Microbatched log-probs + loss + backward ─────────
                # Splits the (N, L) batch into smaller chunks so the Mamba SSM
                # chunk-scan intermediate tensor (b, c, l, s, h, n) doesn't OOM.
                self.model.train()

                n_episodes = input_ids.shape[0]
                mb_size = cfg.logprob_micro_batch_size or n_episodes
                total_mask_tokens = comp_mask.sum().item()

                loss = torch.tensor(0.0, device=device)  # accumulated for logging

                for mb_start in range(0, n_episodes, mb_size):
                    mb_end = min(mb_start + mb_size, n_episodes)
                    mb_ids = input_ids[mb_start:mb_end]
                    mb_attn = attn_mask[mb_start:mb_end]
                    mb_comp = comp_mask[mb_start:mb_end]
                    mb_adv = advantages[mb_start:mb_end]

                    mb_log_probs = compute_logprobs(
                        self.model, mb_ids, mb_attn, mb_comp
                    )

                    mb_ref_log_probs = compute_ref_logprobs(
                        self.model, mb_ids, mb_attn, mb_comp
                    )

                    mb_loss = grpo_loss(
                        mb_log_probs,
                        mb_ref_log_probs,
                        mb_adv,
                        mb_comp,
                        epsilon=cfg.epsilon,
                        beta=cfg.beta,
                    )

                    # Weight by this microbatch's share of completion tokens so
                    # the accumulated gradient equals that of the full-batch loss.
                    mb_tokens = mb_comp.sum().item()
                    weight = mb_tokens / (total_mask_tokens + 1e-8)

                    (mb_loss * weight / cfg.gradient_accumulation_steps).backward()

                    loss = loss + mb_loss.detach() * weight
                    torch.cuda.empty_cache()

                global_step += 1

                if global_step % cfg.gradient_accumulation_steps == 0:
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Sync LoRA adapter to vLLM periodically
                    if global_step % cfg.vllm_sync_every == 0:
                        self._sync_weights_to_vllm()

                    # Push PEFT adapter weights to HF Hub periodically
                    if (
                        cfg.hf_repo_id
                        and cfg.hf_push_every > 0
                        and global_step % cfg.hf_push_every == 0
                    ):
                        self._push_to_hub(tag=f"step-{global_step}")

                # ── Logging ───────────────────────────────────────────────────
                if global_step % cfg.log_every == 0:
                    mean_reward = rewards_t.mean().item()
                    reward_std = rewards_t.std().item()
                    logger.info(
                        f"step={global_step} loss={loss.item():.4f} "
                        f"mean_reward={mean_reward:.3f} reward_std={reward_std:.3f}"
                    )
                    # Diagnostic: log sample generation when all rewards are 0
                    if mean_reward == 0.0 and flat_episodes:
                        sample = flat_episodes[0]
                        last_asst = None
                        for msg in reversed(sample.messages):
                            if msg.get("role") == "assistant":
                                last_asst = (msg.get("content") or "")[:300]
                                break
                        logger.warning(
                            "REWARD COLLAPSE at step=%d — sample generation: %r",
                            global_step, last_asst,
                        )
                    if self._wandb is not None:
                        self._wandb.log(
                            {
                                "loss": loss.item(),
                                "mean_reward": mean_reward,
                                "reward_std": reward_std,
                                "epoch": epoch,
                            },
                            step=global_step,
                        )

                # ── Cleanup episode store ─────────────────────────────────────
                for ep in flat_episodes:
                    self.store.clear(ep.id)

        # Final save
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

        if cfg.hf_repo_id and cfg.hf_push_final:
            self._push_to_hub(tag="final")

        if self._wandb is not None:
            self._wandb.finish()

        self.rollout_engine.cleanup()

    def _sync_weights_to_vllm(self) -> None:
        """Save LoRA adapter to disk and load into vLLM for next rollout."""
        adapter_dir = Path(self.config.output_dir) / ".lora_adapter"
        self.model.save_pretrained(adapter_dir)
        self.rollout_engine.load_lora_adapter(str(adapter_dir))

    def _push_to_hub(self, tag: str) -> None:
        """Push PEFT adapter weights (not base weights) to the HF Hub."""
        cfg = self.config
        tmp_path = Path(cfg.output_dir) / f".hf_push_{tag}"
        self.model.save_pretrained(tmp_path)
        self._hf_api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=cfg.hf_repo_id,
            repo_type="model",
            commit_message=f"grpo-dojo: {tag}",
        )
        logger.info(f"Pushed adapter weights to {cfg.hf_repo_id} ({tag})")
        if self._wandb is not None:
            self._wandb.log({"hf_push": tag})
        shutil.rmtree(tmp_path, ignore_errors=True)


def _rows_to_list(batch) -> list[dict]:
    """Convert HuggingFace Dataset batch (dict of lists) to list of dicts."""
    if isinstance(batch, list):
        return batch
    # HF Dataset slice returns dict of lists
    if isinstance(batch, dict):
        keys = list(batch.keys())
        n = len(batch[keys[0]])
        return [{k: batch[k][i] for k in keys} for i in range(n)]
    return list(batch)
