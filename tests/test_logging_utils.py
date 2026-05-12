"""Tests for ``utils/logging.py`` and ``utils/seed.py``."""
from __future__ import annotations

import json
import random as py_random
from pathlib import Path

import numpy as np
import pytest
import torch

from cellfluxv2.utils.logging import append_jsonl, format_metrics, read_jsonl
from cellfluxv2.utils.seed import seed_everything


# ============================================================================
# append_jsonl / read_jsonl
# ============================================================================

def test_jsonl_roundtrip_single_record(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    append_jsonl(path, {"step": 0, "loss": 1.5})
    records = read_jsonl(path)
    assert records == [{"step": 0, "loss": 1.5}]


def test_jsonl_multiple_records(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    for i in range(3):
        append_jsonl(path, {"step": i, "loss": float(i) * 0.5})
    records = read_jsonl(path)
    assert len(records) == 3
    assert records[0]["step"] == 0
    assert records[2]["loss"] == 1.0


def test_jsonl_keys_sorted_in_file(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    append_jsonl(path, {"z": 1, "a": 2, "m": 3})
    text = path.read_text().strip()
    # JSON serialized with sort_keys=True puts keys alphabetically.
    assert text == '{"a": 2, "m": 3, "z": 1}'


def test_jsonl_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "nested" / "logs" / "train.jsonl"
    assert not path.parent.exists()
    append_jsonl(path, {"hello": "world"})
    assert path.exists()


def test_jsonl_non_dict_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="dict"):
        append_jsonl(tmp_path / "x.jsonl", [1, 2, 3])  # type: ignore[arg-type]


def test_jsonl_read_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "missing.jsonl")


def test_jsonl_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n')
    records = read_jsonl(path)
    assert records == [{"a": 1}, {"b": 2}]


# ============================================================================
# format_metrics
# ============================================================================

def test_format_metrics_sorted_keys():
    out = format_metrics({"loss": 1.2345, "acc": 0.9})
    assert out == "acc=0.9000 loss=1.2345"


def test_format_metrics_integer_value():
    out = format_metrics({"step": 100})
    assert out == "step=100"


def test_format_metrics_scientific_for_small_values():
    out = format_metrics({"grad": 1e-7})
    assert "grad=1.000e-07" in out


def test_format_metrics_scientific_for_huge_values():
    out = format_metrics({"big": 1.5e8})
    assert "big=1.500e+08" in out


def test_format_metrics_zero_uses_decimal():
    out = format_metrics({"loss": 0.0})
    assert out == "loss=0.0000"


def test_format_metrics_with_prefix():
    out = format_metrics({"loss": 1.0}, prefix="[train]")
    assert out == "[train] loss=1.0000"


def test_format_metrics_non_dict_raises():
    with pytest.raises(ValueError, match="dict"):
        format_metrics([1, 2, 3])  # type: ignore[arg-type]


# ============================================================================
# seed_everything
# ============================================================================

def test_seed_everything_reproduces_torch():
    seed_everything(42)
    a = torch.randn(5)
    seed_everything(42)
    b = torch.randn(5)
    torch.testing.assert_close(a, b)


def test_seed_everything_reproduces_numpy():
    seed_everything(42)
    a = np.random.rand(5)
    seed_everything(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_seed_everything_reproduces_python_random():
    seed_everything(42)
    a = [py_random.random() for _ in range(5)]
    seed_everything(42)
    b = [py_random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_different_seed_diverges():
    seed_everything(0)
    a = torch.randn(5)
    seed_everything(1)
    b = torch.randn(5)
    assert not torch.allclose(a, b)


def test_seed_everything_negative_raises():
    with pytest.raises(ValueError, match=">= 0"):
        seed_everything(-1)


def test_seed_everything_bool_raises():
    with pytest.raises(ValueError, match="int"):
        seed_everything(True)  # type: ignore[arg-type]


def test_seed_everything_deterministic_sets_cudnn_flags():
    seed_everything(0, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
