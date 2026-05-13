"""Unit-level tests for ``cellfluxv2.train.stage2``.

These tests deliberately do not hit real rxrx3 data or GCS. They cover:

* the YAML at ``configs/stage2.yaml`` parses into ``Stage2Config``;
* CLI overrides (including ``--init-ckpt``) round-trip through
  ``apply_overrides``;
* ``build_pair_index_from_split`` builds a real, strict same-(experiment,
  plate) ``PairIndex`` — there is no empty-PairIndex shortcut anywhere
  in the Stage 2 trainer (Stage 1's escape hatch is intentionally
  absent);
* ``build_pair_index_from_split`` raises when a treated group has lost
  all its controls (e.g. after missing-address filtering).

End-to-end real-data behaviour lives in ``scripts/smoke_stage2_tiny.py``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cellfluxv2.data.metadata import REQUIRED_COLUMNS, MetadataSplit
from cellfluxv2.data.pair_index import PairIndex
from cellfluxv2.train.stage2 import (
    Stage2Config,
    apply_overrides,
    build_pair_index_from_split,
    load_config,
    parse_args,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2_YAML = REPO_ROOT / "configs" / "stage2.yaml"


# ---------- synthetic split helpers ----------------------------------------

def _row(
    experiment: str,
    plate: int,
    address: str,
    treatment: str | None,
    smiles: str | None,
    metadata_idx: int,
) -> dict:
    return {
        "experiment_name": experiment,
        "plate": plate,
        "address": address,
        "treatment": treatment,
        "SMILES": smiles,
        "perturbation_type": "COMPOUND",
        "metadata_idx": metadata_idx,
    }


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    cols = list(REQUIRED_COLUMNS) + ["metadata_idx"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def _synthetic_split() -> MetadataSplit:
    """Two plates, each with 2 treated + 2 controls. Valid for Stage 2."""
    treated = _make_df(
        [
            _row("expA", 1, "A01", "aspirin", "CC(=O)O", 10),
            _row("expA", 1, "A02", "tylenol", "C", 11),
            _row("expA", 2, "A03", "aspirin", "CC(=O)O", 12),
            _row("expA", 2, "A04", "tylenol", "C", 13),
        ]
    )
    control = _make_df(
        [
            _row("expA", 1, "C01", "DMSO", None, 20),
            _row("expA", 1, "C02", "DMSO", None, 21),
            _row("expA", 2, "C03", "DMSO", None, 22),
            _row("expA", 2, "C04", "DMSO", None, 23),
        ]
    )
    vocab = set(treated["SMILES"].astype(str).unique())
    return MetadataSplit(treated=treated, control=control, smiles_vocab=vocab)


# ============================================================================
# A. Stage 2 config + overrides
# ============================================================================

def test_stage2_yaml_loads_and_is_stage_2():
    cfg = load_config(STAGE2_YAML)
    assert isinstance(cfg, Stage2Config)
    assert cfg.stage == 2
    # Spec defaults that the trainer relies on:
    assert cfg.model["balance_conditioning"] is True
    assert float(cfg.model["time_scale"]) == 1.0
    assert float(cfg.model["condition_scale"]) == 1.0
    assert int(cfg.training["batch_size"]) == 128
    assert float(cfg.training["lr"]) == pytest.approx(1e-4)
    assert float(cfg.training["source_noise_p"]) == pytest.approx(0.1)
    assert float(cfg.training["source_noise_sigma"]) == pytest.approx(1.0)
    # init_ckpt key is present but null by default.
    assert "init_ckpt" in cfg.training
    assert cfg.training["init_ckpt"] is None


def test_stage2_yaml_rejects_wrong_stage(tmp_path):
    """Stage 2 loader must reject a stage=1 YAML."""
    bad = tmp_path / "wrong.yaml"
    bad.write_text(
        "seed: 0\n"
        "stage: 1\n"
        "model: {}\n"
        "training: {}\n"
        "data: {}\n"
    )
    with pytest.raises(ValueError, match="stage=2"):
        load_config(bad)


def test_apply_overrides_carries_init_ckpt_and_friends():
    cfg = load_config(STAGE2_YAML)
    cfg = apply_overrides(
        cfg,
        {
            "max_steps": 7,
            "batch_size": 32,
            "num_workers": 2,
            "device": "cpu",
            "output_dir": "runs/foo",
            "init_ckpt": "runs/stage1_cond_balance_1k/final.pt",
            "wandb_disabled": True,
        },
    )
    assert cfg.training["max_steps"] == 7
    assert cfg.training["batch_size"] == 32
    assert cfg.training["num_workers"] == 2
    assert cfg.training["device"] == "cpu"
    assert cfg.training["output_dir"] == "runs/foo"
    assert cfg.training["init_ckpt"] == "runs/stage1_cond_balance_1k/final.pt"
    assert cfg.wandb["enabled"] is False


def test_apply_overrides_no_init_ckpt_leaves_yaml_default():
    cfg = load_config(STAGE2_YAML)
    cfg = apply_overrides(cfg, {})
    assert cfg.training["init_ckpt"] is None


def test_parse_args_supports_init_ckpt_flag():
    ns = parse_args(
        [
            "--config", "configs/stage2.yaml",
            "--init-ckpt", "runs/stage1_cond_balance_1k/final.pt",
            "--max-steps", "10",
            "--batch-size", "16",
            "--num-workers", "0",
            "--device", "cpu",
            "--output-dir", "runs/x",
        ]
    )
    assert ns.init_ckpt == "runs/stage1_cond_balance_1k/final.pt"
    assert ns.max_steps == 10
    assert ns.batch_size == 16
    assert ns.num_workers == 0
    assert ns.device == "cpu"
    assert ns.output_dir == "runs/x"


# ============================================================================
# B. PairIndex on the filtered split — no empty shortcut
# ============================================================================

def test_build_pair_index_from_split_returns_real_pairindex():
    split = _synthetic_split()
    pi = build_pair_index_from_split(split)
    assert isinstance(pi, PairIndex)
    # Two plates → two groups.
    assert len(pi.groups) == 2
    assert len(pi.treated_to_group) == 4
    # Every treated metadata_idx must map to a non-empty control pool.
    for ti, key in pi.treated_to_group.items():
        controls = pi.groups[key]["control"]
        assert len(controls) > 0, f"treated {ti} → group {key} has no controls"
        # Strict same-plate invariant.
        for ci in controls:
            row = split.control.loc[split.control["metadata_idx"] == ci].iloc[0]
            assert (str(row["experiment_name"]), int(row["plate"])) == key


def test_build_pair_index_raises_when_plate_lost_all_controls():
    """If a treated plate has no controls, build_pair_index must raise.

    This is the silent-skip path we explicitly do NOT take in Stage 2.
    Compare to Stage 1, which uses an empty-PairIndex shortcut because
    Stage 1 never reads controls.
    """
    split = _synthetic_split()
    # Drop every control on plate 2, leaving its treated rows orphaned.
    surviving = split.control[split.control["plate"] != 2].copy()
    broken = MetadataSplit(
        treated=split.treated, control=surviving, smiles_vocab=split.smiles_vocab
    )
    with pytest.raises(ValueError, match=r"no controls"):
        build_pair_index_from_split(broken)


def test_stage2_module_does_not_construct_empty_pair_index():
    """The empty-PairIndex sentinel is Stage 1's escape hatch only; the
    Stage 2 module must never construct one (sanity grep)."""
    src = (REPO_ROOT / "src" / "cellfluxv2" / "train" / "stage2.py").read_text()
    assert "PairIndex(groups={}" not in src, (
        "stage2.py is constructing an empty PairIndex — that is the Stage 1 "
        "shortcut and is forbidden for Stage 2"
    )
    # Positive check: stage2 must call the real builder somewhere.
    assert "build_pair_index" in src, (
        "stage2.py does not call build_pair_index — Stage 2 needs a real pair index"
    )
