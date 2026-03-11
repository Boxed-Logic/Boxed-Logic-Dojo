"""Unit tests for dojo/weight_sync.py — no GPU required."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import dojo.weight_sync as _ws
from dojo.weight_sync import sync_lora_weights_to_disk


@pytest.fixture(autouse=True)
def _reset_weight_map_cache():
    """Clear the cached original weight map between tests."""
    _ws._original_weight_map = None
    yield
    _ws._original_weight_map = None


def _make_model(state_dict=None):
    model = MagicMock()
    if state_dict is None:
        state_dict = {"layers.0.weight": torch.zeros(2, 2)}
    model.state_dict.return_value = state_dict
    return model


def _run(model, tmp_path, hub_side_effect=Exception("no hub")):
    saved = {}

    def capture(sd, path):
        saved.update(sd)

    with patch("dojo.weight_sync.safetensors.torch.save_file", side_effect=capture), \
         patch("dojo.weight_sync.hf_hub_download", side_effect=hub_side_effect):
        result = sync_lora_weights_to_disk(model, "test/model", str(tmp_path))

    return result, saved


# ── merge / unmerge ───────────────────────────────────────────────────────────

class TestMergeUnmerge:
    def test_merge_adapter_called(self, tmp_path):
        model = _make_model()
        _run(model, tmp_path)
        model.merge_adapter.assert_called_once()

    def test_unmerge_adapter_called(self, tmp_path):
        model = _make_model()
        _run(model, tmp_path)
        model.unmerge_adapter.assert_called_once()

    def test_unmerge_called_after_merge(self, tmp_path):
        model = _make_model()
        order = []
        model.merge_adapter.side_effect = lambda: order.append("merge")
        model.unmerge_adapter.side_effect = lambda: order.append("unmerge")
        _run(model, tmp_path)
        assert order == ["merge", "unmerge"]


# ── weight name cleaning ──────────────────────────────────────────────────────

class TestWeightNameCleaning:
    def test_lora_params_excluded(self, tmp_path):
        sd = {
            "layers.0.weight": torch.zeros(2, 2),
            "layers.0.lora_A.weight": torch.zeros(2, 2),
            "lora_B.weight": torch.zeros(2, 2),
            "modules_to_save.head.weight": torch.zeros(2, 2),
        }
        _, saved = _run(_make_model(sd), tmp_path)
        assert "layers.0.weight" in saved
        assert not any("lora_" in k for k in saved)
        assert not any("modules_to_save" in k for k in saved)

    def test_base_model_prefix_stripped(self, tmp_path):
        sd = {"base_model.model.layers.0.weight": torch.zeros(2, 2)}
        _, saved = _run(_make_model(sd), tmp_path)
        assert "layers.0.weight" in saved
        assert "base_model.model.layers.0.weight" not in saved

    def test_base_layer_suffix_stripped(self, tmp_path):
        sd = {"layers.0.weight.base_layer": torch.zeros(2, 2)}
        _, saved = _run(_make_model(sd), tmp_path)
        assert "layers.0.weight" in saved
        assert "layers.0.weight.base_layer" not in saved

    def test_both_prefix_and_suffix_stripped(self, tmp_path):
        sd = {"base_model.model.layers.0.weight.base_layer": torch.zeros(2, 2)}
        _, saved = _run(_make_model(sd), tmp_path)
        assert "layers.0.weight" in saved

    def test_weights_moved_to_cpu(self, tmp_path):
        tensor = MagicMock()
        tensor.cpu.return_value = torch.zeros(2, 2)
        sd = {"layers.0.weight": tensor}
        _run(_make_model(sd), tmp_path)
        tensor.cpu.assert_called_once()


# ── index file ────────────────────────────────────────────────────────────────

class TestIndexFile:
    def test_index_file_exists(self, tmp_path):
        _run(_make_model(), tmp_path)
        index_path = tmp_path / ".vllm_sync" / "model.safetensors.index.json"
        assert index_path.exists()

    def test_index_has_metadata_and_total_size(self, tmp_path):
        _run(_make_model(), tmp_path)
        index = json.loads(
            (tmp_path / ".vllm_sync" / "model.safetensors.index.json").read_text()
        )
        assert "metadata" in index
        assert "total_size" in index["metadata"]
        assert isinstance(index["metadata"]["total_size"], int)

    def test_index_weight_map_keys_match_saved(self, tmp_path):
        sd = {
            "layers.0.weight": torch.zeros(2, 2),
            "layers.1.weight": torch.zeros(4, 4),
        }
        _run(_make_model(sd), tmp_path)
        index = json.loads(
            (tmp_path / ".vllm_sync" / "model.safetensors.index.json").read_text()
        )
        weight_map = index["weight_map"]
        assert set(weight_map.keys()) == {"layers.0.weight", "layers.1.weight"}
        assert all(v == "model.safetensors" for v in weight_map.values())

    def test_index_total_size_is_non_negative(self, tmp_path):
        _run(_make_model(), tmp_path)
        index = json.loads(
            (tmp_path / ".vllm_sync" / "model.safetensors.index.json").read_text()
        )
        assert index["metadata"]["total_size"] >= 0


# ── config.json handling ──────────────────────────────────────────────────────

class TestConfigJson:
    def test_config_copied_from_hf_hub(self, tmp_path):
        fake_config = tmp_path / "fake_config.json"
        fake_config.write_text('{"model_type": "test"}')
        model = _make_model()
        with patch("dojo.weight_sync.safetensors.torch.save_file"), \
             patch("dojo.weight_sync.hf_hub_download", return_value=str(fake_config)):
            sync_lora_weights_to_disk(model, "test/model", str(tmp_path))
        config_dst = tmp_path / ".vllm_sync" / "config.json"
        assert config_dst.exists()

    def test_config_fallback_to_model_config_on_hub_error(self, tmp_path):
        model = _make_model()
        with patch("dojo.weight_sync.safetensors.torch.save_file"), \
             patch("dojo.weight_sync.hf_hub_download", side_effect=Exception("hub down")):
            sync_lora_weights_to_disk(model, "test/model", str(tmp_path))
        model.get_base_model.return_value.config.to_json_file.assert_called_once()

    def test_config_not_overwritten_if_already_exists(self, tmp_path):
        """Second call should not re-download config.json."""
        sync_dir = tmp_path / ".vllm_sync"
        sync_dir.mkdir(parents=True, exist_ok=True)
        existing = sync_dir / "config.json"
        existing.write_text('{"existing": true}')
        model = _make_model()
        with patch("dojo.weight_sync.safetensors.torch.save_file"), \
             patch("dojo.weight_sync.hf_hub_download", side_effect=Exception("no hub")):
            sync_lora_weights_to_disk(model, "test/model", str(tmp_path))
        # config.json should be preserved (not re-downloaded or overwritten)
        assert json.loads(existing.read_text()) == {"existing": True}


# ── return value ──────────────────────────────────────────────────────────────

class TestReturnValue:
    def test_returns_sync_dir_path(self, tmp_path):
        result, _ = _run(_make_model(), tmp_path)
        assert result == (tmp_path / ".vllm_sync").resolve()

    def test_sync_dir_created(self, tmp_path):
        _run(_make_model(), tmp_path)
        assert (tmp_path / ".vllm_sync").is_dir()
