<div align="center">

[![Tests](https://github.com/Boxed-Logic/Boxed-Logic-Dojo/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Boxed-Logic/Boxed-Logic-Dojo/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

# Boxed-Logic Dojo

**Minimal GRPO training library with multi-turn tool-use support for LLMs.**

</div>

---

## What is Dojo?

Dojo implements [Group Relative Policy Optimization (GRPO)](https://arxiv.org/abs/2402.03300) for fine-tuning language models with LoRA. It natively supports multi-turn rollouts where the model can call tools between turns, scored by user-defined reward functions. Inference is powered by vLLM with sleep/wake GPU sharing so training and rollouts run on a single GPU.

## Features

- **GRPO with clipped surrogate + KL penalty** — policy gradient loss with configurable epsilon clipping and beta-weighted KL divergence
- **Multi-turn tool use** — models call tools mid-conversation for up to `max_turns` rounds per episode
- **`@tool` decorator** — attach OpenAI-format JSON schemas to plain Python functions
- **Per-episode state** — thread-safe `EpisodeStore` lets tools read/write state scoped to the current episode via `get_episode_id()`
- **LoRA fine-tuning via PEFT** — reference log-probs computed by disabling the adapter (no second model copy needed)
- **vLLM sleep/wake GPU sharing** — vLLM offloads weights to CPU during training, wakes up for rollouts; merged LoRA weights synced to disk for in-place reload
- **Microbatched log-prob computation** — configurable `logprob_micro_batch_size` to avoid OOM on hybrid architectures
- **Gradient checkpointing** — recompute activations during backward to reduce memory at ~30% extra compute
- **W&B and HF Hub integration** — optional Weights & Biases logging and periodic adapter pushes to Hugging Face Hub
- **Flexible data loading** — `load_dataset()` accepts local JSONL files or HuggingFace dataset names

## Quickstart

```bash
pip install -e .
pip install vllm  # required for rollout engine
```

```python
from dojo import GRPOConfig, GRPOTrainer, load_dataset
from dojo.tools import tool

# 1. Define a tool
@tool(
    name="calculate",
    description="Evaluate an arithmetic expression.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g. '(12 + 8) * 3'"}
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))

# 2. Define a reward function
def reward_fn(episodes, row):
    gold = float(row["answer"])
    return [
        1.0 if abs(float(ep.messages[-1].get("content", "0")) - gold) < 0.01 else 0.0
        for ep in episodes
    ]

# 3. Configure and train
config = GRPOConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    lora_rank=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    num_generations=8, max_turns=5,
    max_completion_length=512, max_total_tokens=2048,
    batch_size=4, gradient_accumulation_steps=4,
    learning_rate=5e-5, num_epochs=2,
    epsilon=0.2, beta=0.01,
    temperature=0.9, top_p=0.95,
    seed=42, device="auto", torch_dtype="bfloat16",
    output_dir="./output", log_every=1,
)

dataset = load_dataset("train.jsonl")
trainer = GRPOTrainer(config=config, tools=[calculate], reward_fn=reward_fn)
trainer.train(dataset)
```

## Example: GSM8K Math Reasoning

The `examples/gsm8k/` directory contains a complete example that trains a model on GSM8K math word problems using a `calculate` tool and a composite reward (answer correctness + think-block compliance + tool use).

```bash
cd examples/gsm8k
python train.py                          # train with config.json defaults
python eval.py --model ./gsm8k_output    # evaluate the trained model
```

All hyperparameters are read from `config.json`.

## Architecture

| Module | Purpose |
|--------|---------|
| `dojo/config.py` | `GRPOConfig` dataclass — all hyperparameters for training, rollout, vLLM, W&B, HF Hub |
| `dojo/trainer.py` | `GRPOTrainer` — end-to-end training loop: rollout, reward, loss, optimizer step |
| `dojo/tools.py` | `@tool` decorator, `ToolRegistry`, `get_episode_id()` — tool schema and execution with per-episode context |
| `dojo/episode.py` | `Episode` dataclass (messages, rewards, token IDs) and thread-safe `EpisodeStore` |
| `dojo/vllm_rollout.py` | `VLLMRolloutEngine` — batched multi-turn inference with vLLM, sleep/wake GPU sharing |
| `dojo/rollout.py` | Shared tool-call parsing and execution logic |
| `dojo/logprobs.py` | `build_token_sequences`, `compute_logprobs`, `compute_ref_logprobs` — tokenization and forward passes |
| `dojo/loss.py` | `grpo_loss` (clipped surrogate + KL), `normalize_per_group` (per-group z-score advantages) |
| `dojo/weight_sync.py` | `sync_lora_weights_to_disk` — merge LoRA weights and save safetensors for vLLM reload |
| `dojo/dataset.py` | `load_dataset` — loads local JSONL files or HuggingFace datasets |

## API Reference

### `GRPOConfig`

```python
GRPOConfig(model_name, lora_rank, lora_alpha, target_modules, ...)
GRPOConfig.from_dict(d)  # build from a dict, ignoring unknown keys
```

Dataclass holding all hyperparameters. See `dojo/config.py` for the full list of fields and defaults.

### `GRPOTrainer`

```python
trainer = GRPOTrainer(config, tools, reward_fn, episode_init_fn=None)
trainer.train(dataset)
```

- `tools` — list of `@tool`-decorated functions
- `reward_fn(episodes: list[Episode], row: dict) -> list[float]`
- `episode_init_fn(episode_id: str, row: dict) -> None` — optional callback to populate the episode store at the start of each episode

### `@tool`

```python
@tool(name="my_tool", description="...", parameters={...})
def my_tool(arg1: str) -> str: ...
```

Attaches an OpenAI-format tool schema. The decorated function is called with JSON-parsed arguments during rollout.

### `ToolRegistry`

```python
registry = ToolRegistry(tools, store)
registry.schemas      # list of OpenAI-format tool schema dicts
registry.execute(episode_id, tool_call_dict)  # run a tool call
```

### `Episode` / `EpisodeStore`

```python
episode.messages      # list of chat messages
episode.reward        # float reward value
episode.tool_calls()  # list of tool call dicts
episode.tool_results()  # list of tool result strings
```

`EpisodeStore` is a thread-safe key-value store. Tools access the current episode via `get_episode_id()`.

### `VLLMRolloutEngine`

```python
engine = VLLMRolloutEngine(tokenizer, config, registry, store)
engine.rollout_batch(rows)   # batched multi-turn rollout
engine.sleep()               # offload vLLM weights to CPU
engine.wake_up()             # reload weights for inference
```

### `load_dataset`

```python
load_dataset("train.jsonl")              # local JSONL file
load_dataset("gsm8k", split="train")     # HuggingFace dataset
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style guidelines, and the pull request process.

## License

MIT — see [LICENSE](LICENSE) for details.
