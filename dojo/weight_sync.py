"""Utilities for syncing LoRA-merged weights to disk for vLLM reload."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import safetensors.torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def sync_lora_weights_to_disk(model, model_name: str, output_dir: str) -> Path:
    """Merge LoRA into base weights and save to disk for vLLM to reload.

    Returns the directory containing the saved weights.
    """
    sync_dir = Path(output_dir).resolve() / ".vllm_sync"
    sync_dir.mkdir(parents=True, exist_ok=True)

    # 1. Merge LoRA into base weights (in-place, reversible)
    model.merge_adapter()

    # 2. Build state dict with clean names vLLM expects.
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
        # Unlink symlink first to avoid writing through to the HF cache
        if config_dst.is_symlink():
            config_dst.unlink()
        try:
            src = hf_hub_download(model_name, "config.json")
            shutil.copyfile(src, config_dst)
        except Exception:
            model.get_base_model().config.to_json_file(str(config_dst))

    # 5. Remove symlinked weight files so we don't write through to HF cache,
    #    and clean up stale shards that won't match our single-file layout.
    for p in list(sync_dir.glob("model*.safetensors*")):
        if p.is_symlink():
            p.unlink()

    # 6. Verify saved weight names match the original model's names.
    #    Name mismatches cause vLLM reload_weights to silently skip params.
    _verify_weight_names(state_dict, model_name, sync_dir)

    # 7. Save weights as a single safetensors file
    safetensors.torch.save_file(state_dict, str(sync_dir / "model.safetensors"))

    # 8. Write the index file so vLLM finds the weights
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {k: "model.safetensors" for k in state_dict},
    }
    (sync_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    logger.info("Merged weights saved to %s for vLLM reload", sync_dir)
    return sync_dir


def _verify_weight_names(
    state_dict: dict[str, "torch.Tensor"],
    model_name: str,
    sync_dir: Path,
) -> None:
    """Compare saved weight names against the original model's safetensors names.

    Logs warnings for any mismatches, which would cause vLLM reload_weights
    to silently skip parameters.
    """
    import torch  # noqa: F811

    # Collect original weight names from the HF safetensors index
    original_names: set[str] = set()
    try:
        idx_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(idx_path) as f:
            original_index = json.load(f)
        original_names = set(original_index.get("weight_map", {}).keys())
    except Exception:
        # Single-file model — read keys directly from the safetensors file
        try:
            sf_path = hf_hub_download(model_name, "model.safetensors")
            from safetensors import safe_open
            with safe_open(sf_path, framework="pt") as f:
                original_names = set(f.keys())
        except Exception as e:
            logger.warning("Cannot verify weight names against original model: %s", e)
            return

    saved_names = set(state_dict.keys())

    missing_from_save = original_names - saved_names
    extra_in_save = saved_names - original_names

    if missing_from_save:
        logger.error(
            "WEIGHT SYNC: %d weights in original model MISSING from saved file: %s",
            len(missing_from_save),
            sorted(missing_from_save)[:20],
        )
    if extra_in_save:
        logger.warning(
            "WEIGHT SYNC: %d extra weights in saved file not in original model: %s",
            len(extra_in_save),
            sorted(extra_in_save)[:20],
        )
    if not missing_from_save and not extra_in_save:
        logger.info("WEIGHT SYNC: all %d weight names match original model", len(saved_names))
    else:
        logger.info(
            "WEIGHT SYNC: saved=%d, original=%d, missing=%d, extra=%d",
            len(saved_names), len(original_names),
            len(missing_from_save), len(extra_in_save),
        )
