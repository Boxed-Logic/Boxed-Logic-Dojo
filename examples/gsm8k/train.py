"""GRPO training script for GSM8K math reasoning.

Reads all hyperparameters from config.json and runs GRPOTrainer with the
calculate() tool and the composite reward function from rewards.py.

Usage:
    python train.py
    python train.py --config config.json
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

from dojo import GRPOConfig, GRPOTrainer, load_dataset
from tools import calculate
from rewards import make_reward_fn

HERE = Path(__file__).parent


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Train a model on GSM8K with GRPO. All settings are read from config.json."
    )
    parser.add_argument(
        "--config",
        default=str(HERE / "config.json"),
        help="Path to config JSON file (default: config.json next to this script).",
    )
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())["train"]

    dataset = load_dataset(cfg["data"])

    lora_rank = cfg["lora_rank"]
    config = GRPOConfig(
        model_name=cfg["model"],
        lora_rank=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=cfg["target_modules"],
        num_generations=cfg["num_generations"],
        max_turns=cfg["max_turns"],
        max_completion_length=cfg["max_completion_length"],
        max_total_tokens=cfg["max_total_tokens"],
        batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_epochs=cfg["num_epochs"],
        epsilon=cfg["epsilon"],
        beta=cfg["beta"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        seed=cfg["seed"],
        device=cfg["device"],
        torch_dtype=cfg["torch_dtype"],
        output_dir=cfg["output_dir"],
        log_every=cfg["log_every"],
        vllm_gpu_memory_utilization=cfg["vllm_gpu_memory_utilization"],
        vllm_sync_every=cfg["vllm_sync_every"],
        vllm_enable_sleep_mode=cfg["vllm_enable_sleep_mode"],
        wandb_project=cfg.get("wandb_project"),
    )

    trainer = GRPOTrainer(
        config=config,
        tools=[calculate],
        reward_fn=make_reward_fn(),
    )
    trainer.train(dataset)


if __name__ == "__main__":
    main()
