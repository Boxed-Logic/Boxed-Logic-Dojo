"""vLLM-based rollout engine — mirrors the RolloutEngine interface."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from .config import GRPOConfig
from .episode import Episode, EpisodeStore
from .rollout import process_tool_calls
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


class VLLMRolloutEngine:
    """Rollout engine backed by vLLM for high-throughput inference.

    Uses vLLM's native LoRA serving to swap adapter weights near-instantly
    instead of reinitializing the engine.  The base model stays loaded;
    only the small LoRA adapter is updated between training steps.
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

        llm_kwargs = dict(
            model=config.model_name,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            enforce_eager=True,
            enable_sleep_mode=config.vllm_enable_sleep_mode,
            trust_remote_code=True,
        )
        if config.vllm_enable_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = config.vllm_max_lora_rank or config.lora_rank

        self.llm = LLM(**llm_kwargs)

        self._episode_init_fn = episode_init_fn
        self._lora_id = 0
        self._lora_request = None

        # Size pool for rollout_batch
        self._executor = ThreadPoolExecutor(
            max_workers=config.batch_size * config.num_generations * 2
        )

    def load_lora_adapter(self, lora_path: str) -> None:
        """Register a new LoRA adapter for subsequent generate() calls."""
        from vllm.lora.request import LoRARequest

        self._lora_id += 1
        self._lora_request = LoRARequest(
            lora_name=f"adapter-{self._lora_id}",
            lora_int_id=self._lora_id,
            lora_path=lora_path,
        )
        logger.info("Loaded LoRA adapter %d from %s", self._lora_id, lora_path)

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

            outputs = self.llm.generate(
                prompts, self._sampling_params, lora_request=self._lora_request
            )

            decoded_texts = [out.outputs[0].text for out in outputs]
            token_id_lists = [list(out.outputs[0].token_ids) for out in outputs]

            process_tool_calls(
                active, decoded_texts, token_id_lists,
                self.registry, self._executor,
            )

        for ep in all_episodes:
            ep.done = True

        return groups

    def sleep(self) -> None:
        """Offload vLLM weights to CPU to free GPU memory for training."""
        self.llm.sleep(level=1)

    def wake_up(self) -> None:
        """Restore vLLM weights to GPU after training step."""
        self.llm.wake_up()

    def cleanup(self):
        """Shut down the thread pool executor."""
        self._executor.shutdown(wait=True)
