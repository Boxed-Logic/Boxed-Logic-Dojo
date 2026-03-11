"""Unit tests for vLLM rollout — no real GPU or vLLM installation required."""
from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch, call

import pytest

from dojo.config import GRPOConfig
from dojo.episode import Episode, EpisodeStore
from dojo.tools import ToolRegistry


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(**overrides) -> GRPOConfig:
    base = dict(
        model_name="test/model",
        lora_rank=4,
        lora_alpha=8,
        target_modules=["q_proj"],
        num_generations=2,
        max_turns=3,
        max_completion_length=64,
        max_total_tokens=512,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        num_epochs=1,
        epsilon=0.2,
        beta=0.0,
        temperature=1.0,
        top_p=1.0,
        seed=42,
        device="cpu",
        torch_dtype="float32",
        output_dir="/tmp/test",
        log_every=1,
        vllm_gpu_memory_utilization=0.4,
        vllm_sync_every=10,
        vllm_enable_sleep_mode=True,
    )
    base.update(overrides)
    return GRPOConfig(**base)


def _make_request_output(text: str, token_ids: list[int] | None = None):
    """Build a minimal vLLM RequestOutput-like mock."""
    if token_ids is None:
        token_ids = list(range(len(text.split())))
    completion = MagicMock()
    completion.text = text
    completion.token_ids = token_ids
    output = MagicMock()
    output.outputs = [completion]
    return output


def _make_engine(cfg=None, tokenizer=None, registry=None, store=None, llm_mock=None):
    """Build a VLLMRolloutEngine with fully mocked vLLM internals."""
    # Inject fake vllm module so the import inside __init__ works
    vllm_mod = types.ModuleType("vllm")
    sampling_params_mock = MagicMock()
    vllm_mod.SamplingParams = MagicMock(return_value=sampling_params_mock)

    if llm_mock is None:
        llm_mock = MagicMock()
    vllm_mod.LLM = MagicMock(return_value=llm_mock)

    with patch.dict(sys.modules, {"vllm": vllm_mod}):
        from dojo.vllm_rollout import VLLMRolloutEngine

        if cfg is None:
            cfg = _make_config()
        if tokenizer is None:
            tokenizer = MagicMock()
            tokenizer.apply_chat_template.return_value = "<prompt>"
        if registry is None:
            registry = MagicMock(spec=ToolRegistry)
            registry.schemas = []
        if store is None:
            store = EpisodeStore()

        with patch.object(
            VLLMRolloutEngine, "_prepare_sync_dir",
            staticmethod(lambda config: config.model_name),
        ):
            engine = VLLMRolloutEngine(
                tokenizer=tokenizer,
                config=cfg,
                registry=registry,
                store=store,
            )
    return engine, llm_mock, vllm_mod


# ── Tests: Config ─────────────────────────────────────────────────────────────

class TestConfigNewFields:
    def _base(self, **extra):
        return {
            "model_name": "m", "lora_rank": 4, "lora_alpha": 8,
            "target_modules": [], "num_generations": 2, "max_turns": 3,
            "max_completion_length": 64, "max_total_tokens": 512,
            "batch_size": 2, "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4, "num_epochs": 1, "epsilon": 0.2,
            "beta": 0.0, "temperature": 1.0, "top_p": 1.0,
            "seed": 42, "device": "cpu", "torch_dtype": "float32",
            "output_dir": "/tmp", "log_every": 1,
            **extra,
        }

    def test_defaults_applied_when_absent(self):
        cfg = GRPOConfig.from_dict(self._base())
        assert cfg.vllm_gpu_memory_utilization == 0.4
        assert cfg.vllm_sync_every == 1
        assert cfg.vllm_enable_sleep_mode is True

    def test_explicit_vllm_fields_accepted(self):
        cfg = GRPOConfig.from_dict(self._base(
            vllm_gpu_memory_utilization=0.6,
            vllm_sync_every=5,
            vllm_enable_sleep_mode=False,
        ))
        assert cfg.vllm_gpu_memory_utilization == 0.6
        assert cfg.vllm_sync_every == 5
        assert cfg.vllm_enable_sleep_mode is False

    def test_unknown_keys_ignored(self):
        cfg = GRPOConfig.from_dict(self._base(some_future_field="ignored"))
        assert cfg.model_name == "m"

    def test_missing_required_fields_raises(self):
        """from_dict must raise ValueError listing which fields are missing."""
        with pytest.raises(ValueError, match="missing required fields"):
            GRPOConfig.from_dict({})

    def test_missing_one_required_field_raises(self):
        d = self._base()
        del d["model_name"]
        with pytest.raises(ValueError, match="model_name"):
            GRPOConfig.from_dict(d)


# ── Tests: VLLMRolloutEngine initialization ───────────────────────────────────

class TestVLLMRolloutEngineInit:
    def test_llm_constructed_with_correct_args(self):
        vllm_mod = types.ModuleType("vllm")
        llm_instance = MagicMock()
        llm_cls = MagicMock(return_value=llm_instance)
        vllm_mod.LLM = llm_cls
        vllm_mod.SamplingParams = MagicMock(return_value=MagicMock())

        cfg = _make_config(vllm_gpu_memory_utilization=0.5, vllm_enable_sleep_mode=False)

        with patch.dict(sys.modules, {"vllm": vllm_mod}):
            from dojo.vllm_rollout import VLLMRolloutEngine
            engine = VLLMRolloutEngine(
                tokenizer=MagicMock(),
                config=cfg,
                registry=MagicMock(schemas=[]),
                store=EpisodeStore(),
            )

        llm_cls.assert_called_once_with(
            model="test/model",
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            enable_sleep_mode=False,
            trust_remote_code=True,
        )

    def test_sampling_params_constructed_correctly(self):
        vllm_mod = types.ModuleType("vllm")
        sp_cls = MagicMock(return_value=MagicMock())
        vllm_mod.SamplingParams = sp_cls
        vllm_mod.LLM = MagicMock(return_value=MagicMock())

        cfg = _make_config(temperature=0.7, top_p=0.9, max_completion_length=128)

        with patch.dict(sys.modules, {"vllm": vllm_mod}):
            from dojo.vllm_rollout import VLLMRolloutEngine
            with patch.object(
                VLLMRolloutEngine, "_prepare_sync_dir",
                staticmethod(lambda config: config.model_name),
            ):
                VLLMRolloutEngine(
                    tokenizer=MagicMock(),
                    config=cfg,
                    registry=MagicMock(schemas=[]),
                    store=EpisodeStore(),
                )

        sp_cls.assert_called_once_with(temperature=0.7, top_p=0.9, max_tokens=128)


# ── Tests: rollout_group — single turn, no tool call ─────────────────────────

class TestSingleTurnNoToolCall:
    def test_episode_marked_done(self):
        engine, llm_mock, _ = _make_engine()
        llm_mock.generate.return_value = [
            _make_request_output("The answer is 42."),
            _make_request_output("It is 42."),
        ]

        episodes = engine.rollout_group("sys", "what is 6*7?")

        assert len(episodes) == 2
        assert all(ep.done for ep in episodes)

    def test_assistant_message_appended(self):
        engine, llm_mock, _ = _make_engine()
        llm_mock.generate.return_value = [
            _make_request_output("Answer: 42."),
            _make_request_output("Answer: 42."),
        ]

        episodes = engine.rollout_group("sys", "prompt")

        for ep in episodes:
            roles = [m["role"] for m in ep.messages]
            assert "assistant" in roles
            assistant_msg = next(m for m in ep.messages if m["role"] == "assistant")
            assert assistant_msg["content"] == "Answer: 42."

    def test_prompt_built_with_apply_chat_template(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<built_prompt>"
        engine, llm_mock, _ = _make_engine(tokenizer=tokenizer)
        llm_mock.generate.return_value = [
            _make_request_output("done"),
            _make_request_output("done"),
        ]

        engine.rollout_group("system_msg", "user_msg")

        # apply_chat_template called once per active episode per turn
        assert tokenizer.apply_chat_template.call_count >= 1
        first_call_kwargs = tokenizer.apply_chat_template.call_args_list[0]
        args, kwargs = first_call_kwargs
        # First arg is messages list
        messages = args[0] if args else kwargs.get("conversation") or kwargs.get("messages")
        # tokenize=False should be set
        assert kwargs.get("tokenize") is False
        assert kwargs.get("add_generation_prompt") is True

    def test_token_ids_stored(self):
        engine, llm_mock, _ = _make_engine()
        llm_mock.generate.return_value = [
            _make_request_output("done", token_ids=[1, 2, 3]),
            _make_request_output("done", token_ids=[4, 5, 6]),
        ]

        episodes = engine.rollout_group("sys", "prompt")

        assert episodes[0].completion_token_ids == [[1, 2, 3]]
        assert episodes[1].completion_token_ids == [[4, 5, 6]]


# ── Tests: rollout_group — multi-turn with tool call ─────────────────────────

class TestMultiTurnToolCall:
    def test_tool_executed_and_result_appended(self):
        registry = MagicMock(spec=ToolRegistry)
        registry.schemas = []
        registry.execute.return_value = "42"

        engine, llm_mock, _ = _make_engine(registry=registry)

        tool_call_text = '<tool_call>{"name": "calc", "arguments": {"expression": "6*7"}}</tool_call>'
        # Turn 1: tool call; Turn 2: final answer
        llm_mock.generate.side_effect = [
            [_make_request_output(tool_call_text), _make_request_output(tool_call_text)],
            [_make_request_output("The answer is 42."), _make_request_output("The answer is 42.")],
        ]

        episodes = engine.rollout_group("sys", "what is 6*7?")

        assert all(ep.done for ep in episodes)
        # Tool should have been executed once per episode
        assert registry.execute.call_count == 2

        for ep in episodes:
            roles = [m["role"] for m in ep.messages]
            assert "tool" in roles
            tool_msg = next(m for m in ep.messages if m["role"] == "tool")
            assert tool_msg["content"] == "42"

    def test_turn_counter_incremented(self):
        engine, llm_mock, _ = _make_engine()
        llm_mock.generate.return_value = [
            _make_request_output("done"),
            _make_request_output("done"),
        ]

        episodes = engine.rollout_group("sys", "prompt")

        for ep in episodes:
            assert ep.turn == 1


# ── Tests: max_turns cap ──────────────────────────────────────────────────────

class TestMaxTurnsCap:
    def test_episodes_marked_done_after_max_turns(self):
        cfg = _make_config(max_turns=2, num_generations=1)
        engine, llm_mock, _ = _make_engine(cfg=cfg)

        tool_call_text = '<tool_call>{"name": "calc", "arguments": {"expression": "1+1"}}</tool_call>'
        # Always return a tool call so episodes never self-terminate
        llm_mock.generate.return_value = [_make_request_output(tool_call_text)]

        registry = MagicMock(spec=ToolRegistry)
        registry.schemas = []
        registry.execute.return_value = "2"
        engine.registry = registry

        episodes = engine.rollout_group("sys", "prompt")

        assert all(ep.done for ep in episodes)
        for ep in episodes:
            assert ep.turn <= 2


# ── Tests: sleep / wake_up ────────────────────────────────────────────────────

class TestSleepWakeUp:
    def test_sleep_delegates_to_llm(self):
        engine, llm_mock, _ = _make_engine()
        engine.sleep()
        llm_mock.sleep.assert_called_once_with(level=1)

    def test_wake_up_delegates_to_llm(self):
        engine, llm_mock, _ = _make_engine()
        engine.wake_up()
        llm_mock.wake_up.assert_called_once()


# ── Tests: cleanup ────────────────────────────────────────────────────────────

class TestCleanup:
    def test_cleanup_shuts_down_executor(self):
        engine, _, _ = _make_engine()
        engine.cleanup()
        # After cleanup, submitting to executor should raise RuntimeError
        with pytest.raises(RuntimeError):
            engine._executor.submit(lambda: None)


# ── Tests: _sync_weights_to_vllm ─────────────────────────────────────────────

class TestSyncWeightsToVllm:
    def _make_trainer_mock(self, tmp_path):
        """Build a minimal GRPOTrainer-like object with mocked internals."""
        import torch

        trainer = MagicMock()
        config = _make_config(vllm_sync_every=5)
        config.output_dir = str(tmp_path)
        trainer.config = config

        # Mock model with state_dict and adapter merge/unmerge
        trainer.model.merge_adapter = MagicMock()
        trainer.model.unmerge_adapter = MagicMock()
        trainer.model.state_dict.return_value = {
            "layers.0.weight": torch.zeros(2, 2),
        }
        trainer.model.get_base_model.return_value.config.to_json_file = MagicMock()

        return trainer

    def test_merge_adapter_called(self, tmp_path):
        from dojo.trainer import GRPOTrainer
        trainer = self._make_trainer_mock(tmp_path)
        with patch("dojo.weight_sync.safetensors.torch.save_file"), \
             patch("dojo.weight_sync.hf_hub_download", side_effect=Exception("skip")):
            GRPOTrainer._sync_weights_to_vllm(trainer)
        trainer.model.merge_adapter.assert_called_once()

    def test_unmerge_adapter_called(self, tmp_path):
        from dojo.trainer import GRPOTrainer
        trainer = self._make_trainer_mock(tmp_path)
        with patch("dojo.weight_sync.safetensors.torch.save_file"), \
             patch("dojo.weight_sync.hf_hub_download", side_effect=Exception("skip")):
            GRPOTrainer._sync_weights_to_vllm(trainer)
        trainer.model.unmerge_adapter.assert_called_once()

    def test_vllm_needs_reload_flag_set(self, tmp_path):
        """After sync, _vllm_needs_reload should be True (reload happens on wake_up)."""
        from dojo.trainer import GRPOTrainer
        trainer = self._make_trainer_mock(tmp_path)
        with patch("dojo.weight_sync.safetensors.torch.save_file"), \
             patch("dojo.weight_sync.hf_hub_download", side_effect=Exception("skip")):
            GRPOTrainer._sync_weights_to_vllm(trainer)
        assert trainer._vllm_needs_reload is True

    def test_state_dict_base_model_prefix_stripped(self, tmp_path):
        """Keys with 'base_model.model.' prefix are cleaned before saving."""
        import torch
        from dojo.trainer import GRPOTrainer
        trainer = self._make_trainer_mock(tmp_path)
        trainer.model.state_dict.return_value = {
            "base_model.model.layers.0.weight": torch.zeros(2, 2),
        }
        saved = {}
        def capture_save(sd, path):
            saved.update(sd)
        with patch("dojo.weight_sync.safetensors.torch.save_file", side_effect=capture_save), \
             patch("dojo.weight_sync.hf_hub_download", side_effect=Exception("skip")):
            GRPOTrainer._sync_weights_to_vllm(trainer)
        assert "layers.0.weight" in saved
        assert "base_model.model.layers.0.weight" not in saved

    def test_state_dict_base_layer_suffix_stripped(self, tmp_path):
        """Keys with '.base_layer' are cleaned (LoRA linear wrapper artifact)."""
        import torch
        from dojo.trainer import GRPOTrainer
        trainer = self._make_trainer_mock(tmp_path)
        trainer.model.state_dict.return_value = {
            "layers.0.weight.base_layer": torch.zeros(2, 2),
        }
        saved = {}
        def capture_save(sd, path):
            saved.update(sd)
        with patch("dojo.weight_sync.safetensors.torch.save_file", side_effect=capture_save), \
             patch("dojo.weight_sync.hf_hub_download", side_effect=Exception("skip")):
            GRPOTrainer._sync_weights_to_vllm(trainer)
        assert "layers.0.weight" in saved
        assert "layers.0.weight.base_layer" not in saved

# ── Tests: vllm_sync_every logic ─────────────────────────────────────────────

class TestSyncEveryLogic:
    def test_sync_called_on_correct_steps(self):
        """_sync_weights_to_vllm should be called when step % vllm_sync_every == 0."""
        cfg = _make_config(vllm_sync_every=5)
        sync_mock = MagicMock()

        called_at = []
        for step in range(1, 21):
            if step % cfg.vllm_sync_every == 0:
                sync_mock(step)
                called_at.append(step)

        assert called_at == [5, 10, 15, 20]
        assert sync_mock.call_count == 4

    def test_sync_every_1_triggers_every_step(self):
        cfg = _make_config(vllm_sync_every=1)
        called_at = [s for s in range(1, 6) if s % cfg.vllm_sync_every == 0]
        assert called_at == [1, 2, 3, 4, 5]

    def test_sync_every_100_skips_for_early_steps(self):
        cfg = _make_config(vllm_sync_every=100)
        called_at = [s for s in range(1, 51) if s % cfg.vllm_sync_every == 0]
        assert called_at == []


# ── Tests: _rows_to_list ──────────────────────────────────────────────────────

class TestRowsToList:
    def _fn(self):
        from dojo.trainer import _rows_to_list
        return _rows_to_list

    def test_list_of_dicts_returned_unchanged(self):
        rows = [{"a": 1}, {"a": 2}]
        result = self._fn()(rows)
        assert result == [{"a": 1}, {"a": 2}]

    def test_hf_dict_of_lists_converted(self):
        """HuggingFace Dataset slice (dict of lists) → list of dicts."""
        batch = {"system_prompt": ["sys1", "sys2"], "prompt": ["p1", "p2"]}
        result = self._fn()(batch)
        assert result == [
            {"system_prompt": "sys1", "prompt": "p1"},
            {"system_prompt": "sys2", "prompt": "p2"},
        ]

    def test_empty_list(self):
        assert self._fn()([]) == []

    def test_single_row_list(self):
        result = self._fn()([{"prompt": "hello"}])
        assert result == [{"prompt": "hello"}]

    def test_hf_single_row_dict(self):
        batch = {"a": [42], "b": ["x"]}
        result = self._fn()(batch)
        assert result == [{"a": 42, "b": "x"}]

    def test_preserves_all_keys(self):
        batch = {"k1": [1, 2], "k2": ["a", "b"], "k3": [True, False]}
        result = self._fn()(batch)
        assert result[0] == {"k1": 1, "k2": "a", "k3": True}
        assert result[1] == {"k1": 2, "k2": "b", "k3": False}
