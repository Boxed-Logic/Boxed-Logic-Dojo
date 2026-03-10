"""Download GSM8K and write train.jsonl + test.jsonl.

Each output row contains three fields:
  system_prompt  — shared instruction injected at the start of every episode
  prompt         — the math question
  answer         — the numeric answer as a float

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --output-dir /path/to/dir
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from datasets import load_dataset as hf_load

SYSTEM_PROMPT = (
    "You are a helpful assistant that uses tools to solve math problems step by step.\n"
    "Before every tool call and every final answer, write your reasoning between <think> and </think>.\n"
    "Always end your final response with your numeric answer between <answer> and </answer>."
)


def extract_answer(raw: str) -> float | None:
    """Parse the numeric answer from a GSM8K answer string.

    GSM8K answers end with '#### <number>', e.g. '#### 42' or '#### 1,234'.
    Returns the number as a float, or None if the format is not found or
    the text after '####' cannot be converted to a float.
    """
    parts = raw.split("#### ")
    if len(parts) < 2:
        return None
    try:
        return float(parts[-1].strip().replace(",", ""))
    except ValueError:
        return None


def process_split(hf_dataset, out_path: Path) -> tuple[int, int]:
    """Write one JSONL file for a dataset split.

    Iterates over rows, extracts the numeric answer, and writes qualifying
    rows to ``out_path``. Rows whose answer cannot be parsed are skipped.

    Returns:
        (written, skipped) counts.
    """
    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in hf_dataset:
            answer = extract_answer(row["answer"])
            if answer is None:
                skipped += 1
                continue
            record = {
                "system_prompt": SYSTEM_PROMPT,
                "prompt": row["question"],
                "answer": answer,
            }
            f.write(json.dumps(record) + "\n")
            written += 1
    return written, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Download GSM8K from HuggingFace and write train.jsonl + test.jsonl."
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write JSONL files (default: current directory).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, filename in [("train", "train.jsonl"), ("test", "test.jsonl")]:
        # dojo.load_dataset does not accept the name= parameter required by
        # GSM8K, so we call the HuggingFace API directly here.
        ds = hf_load("openai/gsm8k", name="main", split=split)
        out_path = out_dir / filename
        written, skipped = process_split(ds, out_path)
        print(f"{split}: wrote {written} rows, skipped {skipped} rows → {out_path}")


if __name__ == "__main__":
    main()
