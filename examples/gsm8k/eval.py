"""vLLM-based pass-rate evaluation for GSM8K.

Runs greedy rollouts (temperature=0) over the test set using
VLLMRolloutEngine directly — no training machinery is involved.
Prints a running pass rate after each batch and a final summary.

Usage:
    python eval.py                             # use config.json defaults
    python eval.py --config config.json
    python eval.py --model ./gsm8k_output      # override model (post-training eval)
"""
from __future__ import annotations
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
        help="Model name or path. Overrides config.json eval.model (useful for post-training eval).",
    )
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())["eval"]
    model = args.model or cfg["model"]

    rows = load_dataset(cfg["test_data"])
    # load_dataset returns a list[dict] for local JSONL files.  If an HF
    # Dataset object is returned instead, convert it to a list of dicts.
    if not isinstance(rows, list):
        keys = list(rows.features.keys())
        rows = [{k: rows[k][i] for k in keys} for i in range(len(rows))]

    if cfg.get("limit") is not None:
        rows = rows[: cfg["limit"]]

    config = _make_eval_config(model, cfg)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
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
