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
    if not config_dst.exists():
        try:
            src = hf_hub_download(model_name, "config.json")
            shutil.copyfile(src, config_dst)
        except Exception:
            model.get_base_model().config.to_json_file(str(config_dst))

    # 5. Save weights as a single safetensors file
    safetensors.torch.save_file(state_dict, str(sync_dir / "model.safetensors"))

    # 6. Write the index file so vLLM finds the weights
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {k: "model.safetensors" for k in state_dict},
    }
    (sync_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    logger.info("Merged weights saved to %s for vLLM reload", sync_dir)
    return sync_dir
