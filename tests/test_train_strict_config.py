"""Trainer-level strict-config tests.

Two strictness contracts the trainers must honour:

1. ``stage1.build_model`` / ``stage2.build_model`` pass raw config values
   into ``DiTVelocity`` so the model's strict validation runs on the YAML
   the user wrote. Coercing here (``bool(...)``, ``float(...)``) would
   silently turn typos like ``"false"`` into ``True`` or ``True`` into
   ``1.0``.

2. ``stage1.build_split`` / ``stage2.build_split`` raise
   ``FileNotFoundError`` when ``missing_addresses_path`` is set but the
   file does not exist. The previous behaviour was to silently skip the
   filter, which would hide config typos and run on unfiltered data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cellfluxv2.data.metadata import REQUIRED_COLUMNS
from cellfluxv2.train.stage1 import (
    Stage1Config,
    build_model as build_model_stage1,
    build_split as build_split_stage1,
)
from cellfluxv2.train.stage2 import (
    Stage2Config,
    build_model as build_model_stage2,
    build_split as build_split_stage2,
)


def _stage1_cfg(model_overrides: dict) -> Stage1Config:
    model = {
        "hidden_dim": 32,
        "depth": 1,
        "num_heads": 2,
        "dropout": 0.0,
        "balance_conditioning": True,
        "time_scale": 1.0,
        "condition_scale": 1.0,
    }
    model.update(model_overrides)
    return Stage1Config(
        seed=0,
        stage=1,
        model=model,
        training={"max_steps": 1, "batch_size": 2, "num_workers": 0},
        data={},
        wandb={},
    )


def _stage2_cfg(model_overrides: dict) -> Stage2Config:
    model = {
        "hidden_dim": 32,
        "depth": 1,
        "num_heads": 2,
        "dropout": 0.0,
        "balance_conditioning": True,
        "time_scale": 1.0,
        "condition_scale": 1.0,
    }
    model.update(model_overrides)
    return Stage2Config(
        seed=0,
        stage=2,
        model=model,
        training={"max_steps": 1, "batch_size": 2, "num_workers": 0},
        data={},
        wandb={},
    )


# ---------- baseline: clean configs build successfully ----------------------

def test_build_model_stage1_baseline_succeeds():
    build_model_stage1(_stage1_cfg({}))


def test_build_model_stage2_baseline_succeeds():
    build_model_stage2(_stage2_cfg({}))


# ---------- balance_conditioning strictness ---------------------------------

@pytest.mark.parametrize("builder, cfg_factory", [
    (build_model_stage1, _stage1_cfg),
    (build_model_stage2, _stage2_cfg),
])
def test_build_model_balance_conditioning_string_raises(builder, cfg_factory):
    with pytest.raises(ValueError, match="balance_conditioning"):
        builder(cfg_factory({"balance_conditioning": "false"}))


@pytest.mark.parametrize("builder, cfg_factory", [
    (build_model_stage1, _stage1_cfg),
    (build_model_stage2, _stage2_cfg),
])
def test_build_model_balance_conditioning_int_raises(builder, cfg_factory):
    with pytest.raises(ValueError, match="balance_conditioning"):
        builder(cfg_factory({"balance_conditioning": 1}))


# ---------- time_scale strictness ------------------------------------------

@pytest.mark.parametrize("builder, cfg_factory", [
    (build_model_stage1, _stage1_cfg),
    (build_model_stage2, _stage2_cfg),
])
def test_build_model_time_scale_bool_raises(builder, cfg_factory):
    with pytest.raises(ValueError, match="time_scale"):
        builder(cfg_factory({"time_scale": True}))


# ---------- condition_scale strictness -------------------------------------

@pytest.mark.parametrize("builder, cfg_factory", [
    (build_model_stage1, _stage1_cfg),
    (build_model_stage2, _stage2_cfg),
])
def test_build_model_condition_scale_string_raises(builder, cfg_factory):
    with pytest.raises(ValueError, match="condition_scale"):
        builder(cfg_factory({"condition_scale": "1.0"}))


# ---------- missing_addresses_path strictness -------------------------------

def _write_metadata_csv(tmp_path: Path) -> Path:
    """Tiny valid metadata CSV with one treated and one control row."""
    rows = [
        {
            "experiment_name": "expA",
            "plate": 1,
            "address": "A01",
            "treatment": "aspirin",
            "SMILES": "CC(=O)O",
            "perturbation_type": "COMPOUND",
        },
        {
            "experiment_name": "expA",
            "plate": 1,
            "address": "C01",
            "treatment": "EMPTY_control",
            "SMILES": "",
            "perturbation_type": "EMPTY_control",
        },
    ]
    df = pd.DataFrame(rows)
    for c in REQUIRED_COLUMNS:
        assert c in df.columns
    path = tmp_path / "metadata.csv"
    df.to_csv(path, index=False)
    return path


def _stage1_cfg_for_split(
    metadata_path: Path, missing_addresses_path: Path | str | None
) -> Stage1Config:
    return Stage1Config(
        seed=0,
        stage=1,
        model={},
        training={},
        data={
            "metadata_path": str(metadata_path),
            "missing_addresses_path": (
                str(missing_addresses_path) if missing_addresses_path else None
            ),
        },
        wandb={},
    )


def _stage2_cfg_for_split(
    metadata_path: Path, missing_addresses_path: Path | str | None
) -> Stage2Config:
    return Stage2Config(
        seed=0,
        stage=2,
        model={},
        training={},
        data={
            "metadata_path": str(metadata_path),
            "missing_addresses_path": (
                str(missing_addresses_path) if missing_addresses_path else None
            ),
        },
        wandb={},
    )


@pytest.mark.parametrize("builder, cfg_factory", [
    (build_split_stage1, _stage1_cfg_for_split),
    (build_split_stage2, _stage2_cfg_for_split),
])
def test_build_split_missing_addresses_absent_does_not_raise(
    tmp_path: Path, builder, cfg_factory
):
    """No ``missing_addresses_path`` key (or None) means no filter, no raise."""
    metadata_csv = _write_metadata_csv(tmp_path)
    cfg = cfg_factory(metadata_csv, None)
    split, report = builder(cfg)
    assert report is None
    assert len(split.treated) == 1
    assert len(split.control) == 1


@pytest.mark.parametrize("builder, cfg_factory", [
    (build_split_stage1, _stage1_cfg_for_split),
    (build_split_stage2, _stage2_cfg_for_split),
])
def test_build_split_missing_addresses_set_but_missing_file_raises(
    tmp_path: Path, builder, cfg_factory
):
    """Strict: a config that names a file requires the file to exist."""
    metadata_csv = _write_metadata_csv(tmp_path)
    bogus = tmp_path / "does_not_exist.csv"
    cfg = cfg_factory(metadata_csv, bogus)
    with pytest.raises(FileNotFoundError, match="missing_addresses_path"):
        builder(cfg)
