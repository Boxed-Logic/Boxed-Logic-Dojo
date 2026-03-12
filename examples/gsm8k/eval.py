"""vLLM-based pass-rate evaluation for GSM8K.

Runs greedy rollouts (temperature=0) over the test set using
VLLMRolloutEngine directly — no training machinery is involved.
Prints a running pass rate after each batch and a final summary.

Usage:
    python eval.py                             # use config.json defaults
    python eval.py --config config.json
    python eval.py --model ./gsm8k_output      # auto-detects PEFT adapter
    python eval.py --adapter ./gsm8k_output    # explicit adapter path
"""
from __future__ import annotations
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer
from dojo import GRPOConfig, VLLMRolloutEngine, ToolRegistry, EpisodeStore, load_dataset
from tools import calculate
from rewards import extract_final_answer

HERE = Path(__file__).parent


def _make_eval_config(model: str, cfg: dict) -> GRPOConfig:
    """Build a GRPOConfig suitable for eval-only use with VLLMRolloutEngine.

    GRPOConfig requires all fields to be supplied even though many are only
    used by the training loop.  Training-irrelevant fields receive minimal
    sentinel values; comments mark which fields are genuinely used by the
    rollout engine.
    """
    return GRPOConfig(
        model_name=model,
        # --- Rollout (used by VLLMRolloutEngine) ---
        num_generations=1,                              # single sample per problem
        max_turns=cfg["max_turns"],
        max_completion_length=cfg["max_completion_length"],
        max_total_tokens=cfg["max_total_tokens"],
        temperature=0.0,                                # greedy decoding
        top_p=1.0,                                      # required by vLLM when temperature=0
        vllm_gpu_memory_utilization=0.9,                # full GPU available — no training
        vllm_enable_sleep_mode=False,                   # no training loop to call wake_up()
        vllm_enable_lora=False,                          # eval-only; no adapter swapping
        vllm_sync_every=1,                              # unused; required field
        # --- LoRA (structurally required; never used by vLLM) ---
        lora_rank=1,
        lora_alpha=1,
        target_modules=["q_proj"],
        # --- Training (never used) ---
        batch_size=cfg["batch_size"],
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        num_epochs=1,
        # --- GRPO (never used) ---
        epsilon=0.2,
        beta=0.0,
        # --- Misc ---
        seed=42,
        device="cuda",
        torch_dtype="bfloat16",
        output_dir="/tmp/eval_unused",
        log_every=1,
    )


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Evaluate a model on the GSM8K test set using vLLM rollouts."
    )
    parser.add_argument(
        "--config",
        default=str(HERE / "config.json"),
        help="Path to config JSON file (default: config.json next to this script).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Base model name or path. Overrides config.json eval.model.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to a PEFT/LoRA adapter directory to evaluate (loaded on top of the base model).",
    )
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())["eval"]

    # Resolve adapter and base model.  If --adapter is given, the base model
    # is --model (or config default).  For backward compat, if --model points
    # to a PEFT adapter dir (contains adapter_config.json) and no --adapter
    # flag, auto-detect and split.
    adapter_path = args.adapter
    base_model = args.model or cfg["model"]

    if adapter_path is None and Path(base_model).is_dir():
        ac = Path(base_model) / "adapter_config.json"
        if ac.exists():
            adapter_path = base_model
            base_model = json.loads(ac.read_text()).get(
                "base_model_name_or_path", cfg["model"]
            )

    rows = load_dataset(cfg["test_data"])
    # load_dataset returns a list[dict] for local JSONL files.  If an HF
    # Dataset object is returned instead, convert it to a list of dicts.
    if not isinstance(rows, list):
        keys = list(rows.features.keys())
        rows = [{k: rows[k][i] for k in keys} for i in range(len(rows))]

    if cfg.get("limit") is not None:
        rows = rows[: cfg["limit"]]

    config = _make_eval_config(base_model, cfg)
    if adapter_path:
        config.vllm_enable_lora = True
        # Read adapter rank so vLLM sets max_lora_rank correctly
        ac_file = Path(adapter_path) / "adapter_config.json"
        if ac_file.exists():
            config.lora_rank = json.loads(ac_file.read_text()).get("r", config.lora_rank)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    store = EpisodeStore()
    registry = ToolRegistry([calculate], store)
    engine = VLLMRolloutEngine(
        tokenizer=tokenizer,
        config=config,
        registry=registry,
        store=store,
    )

    if adapter_path:
        engine.load_lora_adapter(str(Path(adapter_path).resolve()))

    correct = total = 0
    for i in range(0, len(rows), cfg["batch_size"]):
        batch = rows[i : i + cfg["batch_size"]]
        groups = engine.rollout_batch(batch)
        for episodes, row in zip(groups, batch):
            ep = episodes[0]
            pred = extract_final_answer(ep)
            gold = float(row["answer"])
            if pred is not None and abs(pred - gold) < 0.01:
                correct += 1
            total += 1
        print(f"  {total}/{len(rows)} | pass rate: {correct/total:.1%}")

    engine.cleanup()
    if total == 0:
        print("\nNo rows evaluated.")
    else:
        print(f"\nPass rate: {correct}/{total} = {correct/total:.2%}")


if __name__ == "__main__":
    main()
