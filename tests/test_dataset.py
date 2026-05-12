from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from cellfluxv2.data.dataset import CellFluxDataset
from cellfluxv2.data.fingerprints import FingerprintCache
from cellfluxv2.data.metadata import EMPTY_CONTROL, REQUIRED_COLUMNS, split_metadata
from cellfluxv2.data.pair_index import build_pair_index
from cellfluxv2.data.plate_cache import PlateCache


# ---------- fixtures --------------------------------------------------------

def _row(exp, plate, address, treatment, smiles, ptype="COMPOUND"):
    return {
        "experiment_name": exp,
        "plate": plate,
        "address": address,
        "treatment": treatment,
        "SMILES": smiles,
        "perturbation_type": ptype,
    }


def _make_metadata_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[list(REQUIRED_COLUMNS)]


def _make_fp_cache(treatment_names: list[str], n_bits: int = 1024) -> FingerprintCache:
    fps = np.zeros((len(treatment_names), n_bits), dtype=np.uint8)
    for i in range(len(treatment_names)):
        # Distinguishable: each treatment gets its own pair of set bits.
        fps[i, i % n_bits] = 1
        fps[i, (i + 7) % n_bits] = 1
    return FingerprintCache(
        fps=fps,
        smiles=np.array([f"SMI_{t}" for t in treatment_names], dtype=str),
        treatments=np.array(treatment_names, dtype=str),
        radius=2,
        n_bits=n_bits,
        use_chirality=False,
    )


def _write_plate_npz(
    root: Path,
    experiment: str,
    plate: int,
    wells: list[str],
    n_per_well: int = 3,
    seed: int = 0,
) -> Path:
    exp_dir = root / experiment
    exp_dir.mkdir(exist_ok=True)
    well_arr = np.array(sum([[w] * n_per_well for w in wells], []), dtype="<U4")
    n = len(well_arr)
    rng = np.random.default_rng(seed)
    arrays = {
        "latent": rng.standard_normal((n, 169, 8)).astype(np.float16),
        "well": well_arr,
        "fov": np.arange(n, dtype=np.int16),
        "y": np.arange(n, dtype=np.int16),
        "x": np.arange(n, dtype=np.int16),
        "cell": np.arange(n, dtype=np.int16),
    }
    p = exp_dir / f"Plate{plate}.npz"
    np.savez(p, **arrays)
    return p


def _build_dataset(
    tmp_path: Path,
    stage: int,
    seed: int = 0,
    *,
    drop_treatment_fp: str | None = None,
    drop_npz_address: str | None = None,
) -> CellFluxDataset:
    """Build a full dataset over a synthetic single-plate experiment.

    By default the metadata has:
      - 4 treated wells (drugA at A01 and A02, drugB at A03, drugC at A04)
      - 2 control wells (C01, C02)
    All on (expA, plate 1).
    """
    rows = [
        _row("expA", 1, "A01", "drugA", "CCO"),     # idx 0 -> treated
        _row("expA", 1, "A02", "drugA", "CCO"),     # idx 1 -> treated
        _row("expA", 1, "A03", "drugB", "CCC"),     # idx 2 -> treated
        _row("expA", 1, "A04", "drugC", "CCCC"),    # idx 3 -> treated
        _row("expA", 1, "C01", EMPTY_CONTROL, None),  # idx 4 -> control
        _row("expA", 1, "C02", EMPTY_CONTROL, None),  # idx 5 -> control
    ]
    df = _make_metadata_df(rows)
    split = split_metadata(df)
    pair_idx = build_pair_index(split.treated, split.control)

    npz_wells = ["A01", "A02", "A03", "A04", "C01", "C02"]
    if drop_npz_address is not None:
        npz_wells = [w for w in npz_wells if w != drop_npz_address]
    _write_plate_npz(tmp_path, "expA", 1, npz_wells)

    cache = PlateCache(tmp_path)
    treatments = ["drugA", "drugB", "drugC"]
    if drop_treatment_fp is not None:
        treatments = [t for t in treatments if t != drop_treatment_fp]
    fp_cache = _make_fp_cache(treatments)
    mean = torch.zeros(8)
    std = torch.ones(8)
    return CellFluxDataset(split, pair_idx, cache, fp_cache, mean, std, stage, seed)


# ---------- constructor validation ------------------------------------------

def test_dataset_rejects_bad_stage(tmp_path: Path):
    with pytest.raises(ValueError, match="stage must be 1 or 2"):
        _build_dataset(tmp_path, stage=3)


def test_dataset_rejects_bad_mean_shape(tmp_path: Path):
    rows = [
        _row("expA", 1, "A01", "drugA", "CCO"),
        _row("expA", 1, "C01", EMPTY_CONTROL, None),
    ]
    split = split_metadata(_make_metadata_df(rows))
    pair_idx = build_pair_index(split.treated, split.control)
    _write_plate_npz(tmp_path, "expA", 1, ["A01", "C01"])
    cache = PlateCache(tmp_path)
    fp_cache = _make_fp_cache(["drugA"])
    with pytest.raises(ValueError, match="mean shape"):
        CellFluxDataset(split, pair_idx, cache, fp_cache, torch.zeros(4), torch.ones(8), stage=1, rng_seed=0)


def test_dataset_rejects_wrong_fp_n_bits(tmp_path: Path):
    rows = [
        _row("expA", 1, "A01", "drugA", "CCO"),
        _row("expA", 1, "C01", EMPTY_CONTROL, None),
    ]
    split = split_metadata(_make_metadata_df(rows))
    pair_idx = build_pair_index(split.treated, split.control)
    _write_plate_npz(tmp_path, "expA", 1, ["A01", "C01"])
    cache = PlateCache(tmp_path)
    fp_cache = _make_fp_cache(["drugA"], n_bits=2048)
    with pytest.raises(ValueError, match="n_bits"):
        CellFluxDataset(split, pair_idx, cache, fp_cache, torch.zeros(8), torch.ones(8), stage=1, rng_seed=0)


# ---------- __len__ ---------------------------------------------------------

def test_dataset_len_equals_treated_count(tmp_path: Path):
    ds = _build_dataset(tmp_path, stage=1)
    assert len(ds) == 4


# ---------- shapes / dtypes / meta ------------------------------------------

def test_dataset_item_shapes_and_dtypes(tmp_path: Path):
    ds = _build_dataset(tmp_path, stage=1)
    item = ds[0]
    assert item["x0"].shape == (169, 8)
    assert item["x1"].shape == (169, 8)
    assert item["condition"].shape == (1024,)
    assert item["x0"].dtype == torch.float32
    assert item["x1"].dtype == torch.float32
    assert item["condition"].dtype == torch.float32


def test_dataset_item_meta_stage1(tmp_path: Path):
    ds = _build_dataset(tmp_path, stage=1)
    m = ds[0]["meta"]
    assert m["stage"] == 1
    assert m["control_metadata_idx"] is None
    assert m["control_address"] is None
    assert m["experiment_name"] == "expA"
    assert m["plate"] == 1
    assert m["treated_address"] in ("A01", "A02", "A03", "A04")
    assert m["treatment"] in ("drugA", "drugB", "drugC")


def test_dataset_item_meta_stage2(tmp_path: Path):
    ds = _build_dataset(tmp_path, stage=2, seed=0)
    m = ds[0]["meta"]
    assert m["stage"] == 2
    assert m["control_metadata_idx"] in (4, 5)
    assert m["control_address"] in ("C01", "C02")


# ---------- stage semantics -------------------------------------------------

def test_dataset_stage1_x0_is_gaussian_not_plate_latent(tmp_path: Path):
    """In stage 1, x0 is fresh noise — extremely unlikely to equal x1 from the plate."""
    ds = _build_dataset(tmp_path, stage=1)
    item = ds[0]
    assert not torch.allclose(item["x0"], item["x1"])
    # And not the plate's latent for the same well either.
    plate_obj = ds.plate_cache.get("expA", 1)
    addr = item["meta"]["treated_address"]
    rows = plate_obj.rows_for(addr)
    for r in rows:
        cell_latent = torch.from_numpy(plate_obj.latent[r])
        assert not torch.allclose(item["x0"], cell_latent)


def test_dataset_stage1_x0_no_double_normalization(tmp_path: Path):
    """With unit mean/std, stage-1 x0 should be near-standard-normal (mean ~0, std ~1)."""
    ds = _build_dataset(tmp_path, stage=1, seed=123)
    # Aggregate over a few items for stability.
    xs = torch.stack([ds[i]["x0"] for i in range(4)])  # (4, 169, 8)
    # Roughly unit-variance noise — much wider tolerance than asymptotic.
    assert abs(xs.mean().item()) < 0.1
    assert 0.7 < xs.std().item() < 1.3


def test_dataset_stage2_x0_comes_from_control_well(tmp_path: Path):
    """Stage-2 x0 must be one of the control well's latent rows."""
    ds = _build_dataset(tmp_path, stage=2, seed=0)
    item = ds[0]
    plate_obj = ds.plate_cache.get(item["meta"]["experiment_name"], item["meta"]["plate"])
    rows = plate_obj.rows_for(item["meta"]["control_address"])
    # Reverse the normalization on item["x0"] and compare against the raw latents.
    x0_unnorm = item["x0"] * ds.std + ds.mean
    matched = any(
        torch.allclose(x0_unnorm, torch.from_numpy(plate_obj.latent[r]), atol=1e-3)
        for r in rows
    )
    assert matched


def test_dataset_stage2_pairing_is_same_plate(tmp_path: Path):
    """meta.experiment_name and meta.plate must match between treated and control."""
    ds = _build_dataset(tmp_path, stage=2)
    for idx in range(len(ds)):
        m = ds[idx]["meta"]
        # Look up the control row in the underlying df.
        ctrl_row = ds.split.control.loc[m["control_metadata_idx"]]
        assert str(ctrl_row["experiment_name"]) == m["experiment_name"]
        assert int(ctrl_row["plate"]) == m["plate"]


# ---------- error paths -----------------------------------------------------

def test_dataset_missing_treatment_fp_raises(tmp_path: Path):
    """Loading a row whose treatment is absent from the FP cache raises KeyError."""
    ds = _build_dataset(tmp_path, stage=1, drop_treatment_fp="drugB")
    # idx=2 references drugB
    with pytest.raises(KeyError, match="drugB"):
        _ = ds[2]


def test_dataset_missing_address_raises(tmp_path: Path):
    """Loading a row whose address is missing from the NPZ raises KeyError. No fallback."""
    ds = _build_dataset(tmp_path, stage=1, drop_npz_address="A03")
    with pytest.raises(KeyError, match="A03"):
        _ = ds[2]


def test_dataset_no_nan_in_outputs(tmp_path: Path):
    """Sanity: synthetic data has no NaNs, and the dataset's finiteness assertion holds."""
    ds = _build_dataset(tmp_path, stage=2)
    for i in range(len(ds)):
        item = ds[i]
        assert torch.isfinite(item["x0"]).all()
        assert torch.isfinite(item["x1"]).all()
        assert torch.isfinite(item["condition"]).all()


# ---------- determinism -----------------------------------------------------

def test_dataset_deterministic_under_fixed_seed_stage1(tmp_path: Path):
    ds_a = _build_dataset(tmp_path, stage=1, seed=42)
    ds_b = _build_dataset(tmp_path, stage=1, seed=42)
    for i in range(len(ds_a)):
        a, b = ds_a[i], ds_b[i]
        torch.testing.assert_close(a["x0"], b["x0"])
        torch.testing.assert_close(a["x1"], b["x1"])
        assert a["meta"]["treated_address"] == b["meta"]["treated_address"]


def test_dataset_deterministic_under_fixed_seed_stage2(tmp_path: Path):
    ds_a = _build_dataset(tmp_path, stage=2, seed=42)
    ds_b = _build_dataset(tmp_path, stage=2, seed=42)
    for i in range(len(ds_a)):
        a, b = ds_a[i], ds_b[i]
        torch.testing.assert_close(a["x0"], b["x0"])
        torch.testing.assert_close(a["x1"], b["x1"])
        assert a["meta"]["control_metadata_idx"] == b["meta"]["control_metadata_idx"]
        assert a["meta"]["control_address"] == b["meta"]["control_address"]


def test_dataset_different_seed_varies_sampling(tmp_path: Path):
    """Across two different seeds, at least one of the 4 items differs in its sampled control."""
    ds_a = _build_dataset(tmp_path, stage=2, seed=0)
    ds_b = _build_dataset(tmp_path, stage=2, seed=999)
    any_diff = any(
        ds_a[i]["meta"]["control_metadata_idx"]
        != ds_b[i]["meta"]["control_metadata_idx"]
        for i in range(len(ds_a))
    )
    assert any_diff


def test_dataset_does_not_use_global_rng(tmp_path: Path):
    """Mutating numpy / torch global RNGs between samples must not affect output."""
    ds_a = _build_dataset(tmp_path, stage=1, seed=7)
    a0 = ds_a[0]
    np.random.seed(12345)
    torch.manual_seed(67890)
    a0_again = ds_a[0]
    torch.testing.assert_close(a0["x0"], a0_again["x0"])
    torch.testing.assert_close(a0["x1"], a0_again["x1"])


# ---------- batch collation -------------------------------------------------

def test_dataset_collate_static_method_shapes(tmp_path: Path):
    ds = _build_dataset(tmp_path, stage=2)
    batch = CellFluxDataset.collate([ds[i] for i in range(4)])
    assert batch["x0"].shape == (4, 169, 8)
    assert batch["x1"].shape == (4, 169, 8)
    assert batch["condition"].shape == (4, 1024)
    assert isinstance(batch["meta"], dict)
    assert len(batch["meta"]["treated_address"]) == 4
    assert batch["meta"]["stage"] == [2, 2, 2, 2]


def test_dataset_dataloader_batch_shapes(tmp_path: Path):
    ds = _build_dataset(tmp_path, stage=1)
    loader = DataLoader(ds, batch_size=4, collate_fn=CellFluxDataset.collate)
    batch = next(iter(loader))
    assert batch["x0"].shape == (4, 169, 8)
    assert batch["x1"].shape == (4, 169, 8)
    assert batch["condition"].shape == (4, 1024)
