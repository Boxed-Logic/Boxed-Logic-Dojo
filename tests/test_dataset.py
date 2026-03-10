"""Unit tests for dojo/dataset.py — no network or GPU required."""
import gzip
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dojo.dataset import load_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _write_jsonl_gz(path: Path, rows: list[dict]) -> Path:
    content = "\n".join(json.dumps(r) for r in rows) + "\n"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(content)
    return path


# ── JSONL loading ─────────────────────────────────────────────────────────────

class TestLoadJsonl:
    def test_basic_rows(self, tmp_path):
        rows = [{"prompt": "Q1", "answer": "A1"}, {"prompt": "Q2", "answer": "A2"}]
        p = _write_jsonl(tmp_path / "data.jsonl", rows)
        result = load_dataset(str(p))
        assert result == rows

    def test_returns_list_of_dicts(self, tmp_path):
        rows = [{"x": 1}]
        p = _write_jsonl(tmp_path / "data.jsonl", rows)
        result = load_dataset(str(p))
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = load_dataset(str(p))
        assert result == []

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"a": 1}\n\n{"a": 2}\n\n')
        result = load_dataset(str(p))
        assert len(result) == 2

    def test_single_row(self, tmp_path):
        p = _write_jsonl(tmp_path / "single.jsonl", [{"prompt": "hi"}])
        result = load_dataset(str(p))
        assert result == [{"prompt": "hi"}]

    def test_nested_values_preserved(self, tmp_path):
        rows = [{"meta": {"split": "train"}, "values": [1, 2, 3]}]
        p = _write_jsonl(tmp_path / "data.jsonl", rows)
        result = load_dataset(str(p))
        assert result[0]["meta"] == {"split": "train"}
        assert result[0]["values"] == [1, 2, 3]

    def test_unicode_preserved(self, tmp_path):
        rows = [{"text": "こんにちは"}, {"text": "café"}]
        p = _write_jsonl(tmp_path / "data.jsonl", rows)
        result = load_dataset(str(p))
        assert result[0]["text"] == "こんにちは"
        assert result[1]["text"] == "café"


# ── gzipped JSONL loading ─────────────────────────────────────────────────────

class TestLoadJsonlGz:
    def test_basic_rows(self, tmp_path):
        rows = [{"prompt": "Q1"}, {"prompt": "Q2"}]
        p = _write_jsonl_gz(tmp_path / "data.jsonl.gz", rows)
        result = load_dataset(str(p))
        assert result == rows

    def test_returns_list_of_dicts(self, tmp_path):
        p = _write_jsonl_gz(tmp_path / "data.jsonl.gz", [{"x": 1}])
        result = load_dataset(str(p))
        assert isinstance(result, list)

    def test_empty_gz(self, tmp_path):
        p = _write_jsonl_gz(tmp_path / "empty.jsonl.gz", [])
        result = load_dataset(str(p))
        assert result == []


# ── HuggingFace routing ───────────────────────────────────────────────────────

class TestHuggingFaceRouting:
    def _hf_mock(self):
        """Return a mock hf_load and the datasets module stub to inject."""
        import sys
        import types
        fake_ds = MagicMock()
        hf_load_mock = MagicMock(return_value=fake_ds)
        datasets_mod = types.ModuleType("datasets")
        datasets_mod.load_dataset = hf_load_mock
        return hf_load_mock, datasets_mod, fake_ds

    def test_jsonl_source_does_not_call_hf(self, tmp_path):
        rows = [{"prompt": "hi"}]
        p = _write_jsonl(tmp_path / "data.jsonl", rows)
        hf_mock, datasets_mod, _ = self._hf_mock()
        with patch.dict("sys.modules", {"datasets": datasets_mod}):
            load_dataset(str(p))
        hf_mock.assert_not_called()

    def test_jsonl_gz_source_does_not_call_hf(self, tmp_path):
        p = _write_jsonl_gz(tmp_path / "data.jsonl.gz", [{"x": 1}])
        hf_mock, datasets_mod, _ = self._hf_mock()
        with patch.dict("sys.modules", {"datasets": datasets_mod}):
            load_dataset(str(p))
        hf_mock.assert_not_called()

    def test_hf_name_calls_hf_load(self):
        hf_mock, datasets_mod, fake_ds = self._hf_mock()
        with patch.dict("sys.modules", {"datasets": datasets_mod}):
            result = load_dataset("some/hf-dataset")
        hf_mock.assert_called_once()
        assert result is fake_ds

    def test_hf_split_forwarded(self):
        hf_mock, datasets_mod, _ = self._hf_mock()
        with patch.dict("sys.modules", {"datasets": datasets_mod}):
            load_dataset("some/dataset", split="validation")
        _, kwargs = hf_mock.call_args
        assert kwargs.get("split") == "validation"
