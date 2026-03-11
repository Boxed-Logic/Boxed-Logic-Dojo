"""Utilities for syncing LoRA-merged weights to disk for vLLM reload."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import safetensors.torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Cached on first successful call so we only hit HF once per process.
_original_weight_map: dict[str, str] | None = None


def _get_original_weight_map(model_name: str) -> dict[str, str] | None:
    """Return {weight_name: shard_filename} from the original HF model.

    Returns None if the map cannot be fetched (callers should fall back to
    single-file save).
    """
    global _original_weight_map
    if _original_weight_map is not None:
        return _original_weight_map

    try:
        idx_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(idx_path) as f:
            result = json.load(f)["weight_map"]
    except Exception:
        # Single-file model
        try:
            sf_path = hf_hub_download(model_name, "model.safetensors")
            from safetensors import safe_open
            with safe_open(sf_path, framework="pt") as f:
                result = {k: "model.safetensors" for k in f.keys()}
        except Exception:
            logger.warning("Cannot fetch original weight map for %s", model_name)
            return None

    _original_weight_map = result
    return _original_weight_map


def sync_lora_weights_to_disk(model, model_name: str, output_dir: str) -> Path:
    """Merge LoRA into base weights and save to disk for vLLM to reload.

    Saves weights using the SAME shard layout as the original HF model so
    that vLLM's reload_weights processes them identically to the initial load.

    Returns the directory containing the saved weights.
    """
    sync_dir = Path(output_dir).resolve() / ".vllm_sync"
    sync_dir.mkdir(parents=True, exist_ok=True)

    # 1. Merge LoRA into base weights (in-place, reversible)
    model.merge_adapter()

    # 2. Build state dict with clean names matching the original HF model.
    #    Skip LoRA-specific params — vLLM only needs merged base weights.
    state_dict = {}
    for name, tensor in model.state_dict().items():
        if "lora_" in name or "modules_to_save" in name:
            continue
        clean = name
        if clean.startswith("base_model.model."):
            clean = clean[len("base_model.model."):]
        clean = clean.replace(".base_layer", "")
        state_dict[clean] = tensor.cpu()

    # 3. Unmerge so LoRA training continues (preserves optimizer state)
    model.unmerge_adapter()

    # 4. Copy config.json from original model (vLLM needs it to validate)
    config_dst = sync_dir / "config.json"
    if not config_dst.exists() or config_dst.is_symlink():
        if config_dst.is_symlink():
            config_dst.unlink()
        try:
            src = hf_hub_download(model_name, "config.json")
            shutil.copyfile(src, config_dst)
        except Exception:
            model.get_base_model().config.to_json_file(str(config_dst))

    # 5. Remove old weight files (symlinks from init OR real files from prior sync)
    for p in list(sync_dir.glob("model*.safetensors*")):
        p.unlink()

    # 6. Try to match the original model's shard layout so vLLM's
    #    reload_weights processes weights identically to the initial load.
    original_map = _get_original_weight_map(model_name)

    if original_map is not None:
        # Filter out extra weights not in the original (e.g. lm_head.weight
        # from tied embeddings) — extras can confuse vLLM's reload pipeline.
        original_names = set(original_map.keys())

        extra = set(state_dict.keys()) - original_names
        if extra:
            logger.info("WEIGHT SYNC: dropping %d extra weights not in original model: %s",
                         len(extra), sorted(extra))
            for k in extra:
                del state_dict[k]

        missing = original_names - set(state_dict.keys())
        if missing:
            logger.error("WEIGHT SYNC: %d weights MISSING vs original model: %s",
                          len(missing), sorted(missing)[:20])

        # Save into the same shard files the original model uses.
        shard_to_keys: dict[str, list[str]] = {}
        for wname, shard_file in original_map.items():
            if wname in state_dict:
                shard_to_keys.setdefault(shard_file, []).append(wname)

        for shard_file, keys in shard_to_keys.items():
            shard_dict = {k: state_dict[k] for k in keys}
            safetensors.torch.save_file(shard_dict, str(sync_dir / shard_file))

        # Write index matching original layout
        weight_map = {k: v for k, v in original_map.items() if k in state_dict}
        n_shards = len(shard_to_keys)
    else:
        # Fallback: save all weights in a single file
        safetensors.torch.save_file(state_dict, str(sync_dir / "model.safetensors"))
        weight_map = {k: "model.safetensors" for k in state_dict}
        n_shards = 1

    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (sync_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    logger.info("Merged weights saved to %s for vLLM reload (%d shards, %d weights)",
                sync_dir, n_shards, len(state_dict))
    return sync_dir
