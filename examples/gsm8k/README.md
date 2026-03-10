# GSM8K Math Reasoning

Trains a model to solve grade-school math word problems using a calculator tool, with GRPO reinforcement learning.

## Training

```bash
python train.py
python train.py --config path/to/config.json
```

All hyperparameters are read from `config.json` (the single source of truth). Defaults:
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Output:** `./gsm8k_output`

## Evaluation

```bash
python eval.py
python eval.py --model ./gsm8k_output
```

Runs greedy rollouts (temperature=0) over the test set and prints a running pass rate. Use `--model` to point at a fine-tuned checkpoint for post-training evaluation.

## Reward Design

The composite reward in `rewards.py` has three components:

| Component | Weight | Type | Description |
|---|---|---|---|
| `answer_reward` | 1.0 | binary | 1.0 if extracted answer matches gold within 0.01 |
| `think_reward` | 0.3 | fractional | Fraction of assistant turns containing a `<think>` block |
| `tool_use_reward` | 0.2 | binary | 1.0 if the model made at least one tool call |

**Total range:** [0.0, 1.5]

GRPO normalizes rewards within each group via z-scoring, so only relative scale between components matters — not absolute magnitude.

**Customizing weights:** Edit the `w_correct`, `w_think`, and `w_tool` defaults in `make_reward_fn()` in `rewards.py`.
