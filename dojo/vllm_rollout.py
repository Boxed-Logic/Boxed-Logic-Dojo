"""vLLM-based rollout engine — mirrors the RolloutEngine interface."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import GRPOConfig
from .episode import Episode, EpisodeStore
from .rollout import process_tool_calls
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


class VLLMRolloutEngine:
    """Rollout engine backed by vLLM for high-throughput inference.

    Mirrors the interface of RolloutEngine (rollout_group, cleanup) so
    GRPOTrainer needs minimal changes. vLLM handles variable-length prompts
    natively — no left-padding required.
    """

    def __init__(
        self,
        tokenizer,
        config: GRPOConfig,
        registry: ToolRegistry,
        store: EpisodeStore,
        episode_init_fn=None,
    ):
        from vllm import LLM, SamplingParams

        self.tokenizer = tokenizer
        self.config = config
        self.registry = registry
        self.store = store

        self._sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_completion_length,
        )

        # Initialize vLLM from the sync directory so reload_weights works
        # without update_config (which can fail on models with non-init
        # dataclass fields like attention_chunk_size on Granite).
        model_path = config.model_name
        if config.vllm_enable_sleep_mode:
            model_path = self._prepare_sync_dir(config)

        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            enforce_eager=True,  # required for in-place weight updates
            enable_sleep_mode=config.vllm_enable_sleep_mode,
            trust_remote_code=True,  # for Granite hybrid and other custom models
        )

        self._episode_init_fn = episode_init_fn

        # Size pool for rollout_batch
        self._executor = ThreadPoolExecutor(
            max_workers=config.batch_size * config.num_generations * 2
        )

    @staticmethod
    def _prepare_sync_dir(config: GRPOConfig) -> str:
        """Pre-populate the sync directory with the original model snapshot.

        Returns the sync_dir path for vLLM to load from.  By initializing
        vLLM from this directory, reload_weights() can re-read updated
        weights without needing collective_rpc("update_config").
        """
        import shutil
        from huggingface_hub import snapshot_download

        sync_dir = Path(config.output_dir).resolve() / ".vllm_sync"

        # Always start fresh so vLLM loads base model weights matching
        # the fresh LoRA (stale merged weights from a prior run would
        # cause a mismatch).
        if sync_dir.exists():
            shutil.rmtree(sync_dir)
        sync_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = Path(snapshot_download(config.model_name))
        for src_file in snapshot_path.iterdir():
            (sync_dir / src_file.name).symlink_to(src_file)
        logger.info("Prepared vLLM sync dir from %s", config.model_name)

        return str(sync_dir)

    def rollout_group(
        self, system_prompt: str, user_prompt: str, row: dict | None = None
    ) -> list[Episode]:
        """Run num_generations episodes for a single (system, user) pair."""
        synthetic_row = dict(row) if row else {}
        synthetic_row.setdefault("system_prompt", system_prompt)
        synthetic_row.setdefault("prompt", user_prompt)
        return self.rollout_batch([synthetic_row])[0]

    def rollout_batch(self, rows: list[dict]) -> list[list[Episode]]:
        """Run rollouts for an entire batch of rows in one interleaved loop.

        All num_generations episodes across all rows are batched into a single
        llm.generate() call each turn, maximising GPU utilisation.

        Returns a list of episode groups (one group per row), in the same order
        as ``rows``.
        """
        cfg = self.config

        # Build one group of episodes per row
        groups: list[list[Episode]] = []
        for row in rows:
            system_prompt = row.get("system_prompt", "")
            user_prompt = row.get("prompt", "")
            eps = [Episode() for _ in range(cfg.num_generations)]
            for ep in eps:
                ep.messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                if self._episode_init_fn is not None:
                    self._episode_init_fn(ep.id, row)
            groups.append(eps)

        all_episodes = [ep for group in groups for ep in group]

        for _turn in range(cfg.max_turns):
            active = [ep for ep in all_episodes if not ep.done]
            if not active:
                break

            prompts = [
                self.tokenizer.apply_chat_template(
                    ep.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=self.registry.schemas if self.registry.schemas else None,
                )
                for ep in active
            ]

            outputs = self.llm.generate(prompts, self._sampling_params)

            decoded_texts = [out.outputs[0].text for out in outputs]
            token_id_lists = [list(out.outputs[0].token_ids) for out in outputs]

            process_tool_calls(
                active, decoded_texts, token_id_lists,
                self.registry, self._executor,
            )

        for ep in all_episodes:
            ep.done = True

        return groups

    def reinitialize(self) -> None:
        """Destroy and recreate the vLLM engine from the sync directory.

        This is used instead of reload_weights() because vLLM's in-place
        weight reload corrupts hybrid Mamba models (producing gibberish).
        A fresh LLM() uses the same initial-load code path that works.
        """
        import gc
        import torch
        from vllm import LLM

        cfg = self.config
        sync_dir = str(Path(cfg.output_dir).resolve() / ".vllm_sync")

        del self.llm
        gc.collect()
        torch.cuda.empty_cache()

        self.llm = LLM(
            model=sync_dir,
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            enforce_eager=True,
            enable_sleep_mode=cfg.vllm_enable_sleep_mode,
            trust_remote_code=True,
        )
        logger.info("vLLM engine reinitialized from %s", sync_dir)

    def sleep(self) -> None:
        """Offload vLLM weights to CPU to free GPU memory for training."""
        self.llm.sleep(level=1)

    def wake_up(self) -> None:
        """Restore vLLM weights to GPU after training step."""
        self.llm.wake_up()

    def cleanup(self):
        """Shut down the thread pool executor."""
        self._executor.shutdown(wait=True)
