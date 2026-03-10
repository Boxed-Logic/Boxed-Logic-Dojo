from __future__ import annotations
import json
from pathlib import Path


def load_dataset(source: str, split: str = "train"):
    """Load a dataset from a local JSONL file or a HuggingFace Hub identifier.

    Args:
        source: Either a path ending in ``.jsonl`` / ``.jsonl.gz``, or a
                HuggingFace dataset name (e.g. ``"openai/gsm8k"``).
        split:  HuggingFace split to load (ignored for local files).

    Returns:
        A list of dicts (JSONL) or a HuggingFace Dataset object — both are
        accepted by ``GRPOTrainer.train()``.
    """
    p = Path(source)
    if source.endswith(".jsonl") or source.endswith(".jsonl.gz"):
        return _load_jsonl(p)

    # Fall back to HuggingFace
    from datasets import load_dataset as hf_load
    return hf_load(source, split=split)


def _load_jsonl(path: Path) -> list[dict]:
    opener = _get_opener(path)
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_opener(path: Path):
    if path.name.endswith(".gz"):
        import gzip
        return gzip.open
    return open
