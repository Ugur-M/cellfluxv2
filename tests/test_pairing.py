"""Tests for metadata.py and pair_index.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellfluxv2.data.metadata import (
    EMPTY_CONTROL,
    REQUIRED_COLUMNS,
    MetadataSplit,
    load_metadata,
    split_metadata,
)
from cellfluxv2.data.pair_index import PairIndex, build_pair_index


# ---------- builders --------------------------------------------------------

def _row(
    experiment: str,
    plate: int,
    address: str,
    treatment: str | None,
    smiles: str | None,
    ptype: str = "COMPOUND",
) -> dict:
    return {
        "experiment_name": experiment,
        "plate": plate,
        "address": address,
        "treatment": treatment,
        "SMILES": smiles,
        "perturbation_type": ptype,
    }


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with all required columns; missing fields → NaN."""
    df = pd.DataFrame(rows)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[list(REQUIRED_COLUMNS)]


# ---------- metadata.py: filtering ------------------------------------------

def test_split_metadata_treated_and_control_definitions():
    df = _make_df([
        _row("compound-001", 1, "A01", "aspirin", "CC(=O)O"),
        _row("compound-001", 1, "A02", EMPTY_CONTROL, None),
        _row("compound-001", 1, "A03", "CRISPR_control", None),
        _row("compound-001", 1, "A04", "gene", None, ptype="CRISPR"),
        _row("compound-001", 1, "A05", "no_smiles", None),
        _row("compound-001", 1, "A06", "empty_smiles", ""),
    ])
    split = split_metadata(df)
    assert list(split.treated["treatment"]) == ["aspirin"]
    assert list(split.control["treatment"]) == [EMPTY_CONTROL]
    assert split.smiles_vocab == {"CC(=O)O"}


def test_split_metadata_smiles_vocab_unique():
    df = _make_df([
        _row("e1", 1, "A01", "drug_a", "CCO"),
        _row("e1", 1, "A02", "drug_a", "CCO"),  # duplicate SMILES
        _row("e1", 1, "A03", "drug_b", "CCC"),
    ])
    split = split_metadata(df)
    assert split.smiles_vocab == {"CCO", "CCC"}


def test_split_metadata_raises_on_missing_columns():
    df = pd.DataFrame({"treatment": ["x"], "SMILES": ["CCO"]})
    with pytest.raises(ValueError, match="missing required columns"):
        split_metadata(df)


# ---------- metadata.py: metadata_idx preservation --------------------------

def test_split_metadata_preserves_metadata_idx():
    df = _make_df([
        _row("e1", 1, "A01", "drug_a", "CCO"),             # idx=0 -> treated
        _row("e1", 1, "A02", EMPTY_CONTROL, None),          # idx=1 -> control
        _row("e1", 1, "A03", "drug_b", "CCC"),             # idx=2 -> treated
        _row("e2", 1, "A01", EMPTY_CONTROL, None),          # idx=3 -> control
    ])
    split = split_metadata(df)
    assert "metadata_idx" in split.treated.columns
    assert "metadata_idx" in split.control.columns
    assert sorted(split.treated["metadata_idx"].tolist()) == [0, 2]
    assert sorted(split.control["metadata_idx"].tolist()) == [1, 3]


def test_split_metadata_does_not_mutate_input():
    df = _make_df([_row("e1", 1, "A01", "drug_a", "CCO")])
    cols_before = list(df.columns)
    _ = split_metadata(df)
    assert list(df.columns) == cols_before  # `metadata_idx` not added to caller's df


def test_load_metadata_round_trip(tmp_path: Path):
    df = _make_df([
        _row("e1", 1, "A01", "drug_a", "CCO"),
        _row("e1", 1, "A02", EMPTY_CONTROL, None),
    ])
    p = tmp_path / "meta.csv"
    df.to_csv(p, index=False)
    split = load_metadata(p)
    assert len(split.treated) == 1
    assert len(split.control) == 1


# ---------- pair_index.py: strict (experiment, plate) pairing ----------------

def test_pair_index_pairs_within_same_experiment_and_plate():
    """Treated A/p1 samples only from controls at A/p1; treated B/p1 only from B/p1."""
    df = _make_df([
        _row("A", 1, "W01", "drug", "CCO"),
        _row("A", 1, "W02", EMPTY_CONTROL, None),
        _row("B", 1, "W01", "drug", "CCC"),
        _row("B", 1, "W02", EMPTY_CONTROL, None),
    ])
    split = split_metadata(df)
    idx = build_pair_index(split.treated, split.control)
    rng = np.random.default_rng(0)

    t_a = int(split.treated[split.treated["experiment_name"] == "A"].iloc[0]["metadata_idx"])
    t_b = int(split.treated[split.treated["experiment_name"] == "B"].iloc[0]["metadata_idx"])
    c_a = int(split.control[split.control["experiment_name"] == "A"].iloc[0]["metadata_idx"])
    c_b = int(split.control[split.control["experiment_name"] == "B"].iloc[0]["metadata_idx"])

    assert idx.sample_control(t_a, rng) == c_a
    assert idx.sample_control(t_b, rng) == c_b


def test_pair_index_never_falls_back_across_experiments():
    """Same plate integer, different experiment names → no fallback, must raise."""
    df = _make_df([
        _row("A", 1, "W01", "drug", "CCO"),                # treated A/1, no A/1 control
        _row("B", 1, "W02", EMPTY_CONTROL, None),          # control B/1 only
    ])
    split = split_metadata(df)
    with pytest.raises(ValueError, match="no controls"):
        build_pair_index(split.treated, split.control)


def test_pair_index_raises_when_treated_group_has_no_controls():
    df = _make_df([
        _row("A", 1, "W01", "drug", "CCO"),                # treated A/1
        _row("A", 2, "W02", EMPTY_CONTROL, None),          # control A/2 (different plate)
    ])
    split = split_metadata(df)
    with pytest.raises(ValueError, match="no controls"):
        build_pair_index(split.treated, split.control)


def test_pair_index_lists_failing_groups_in_error():
    df = _make_df([
        _row("A", 1, "W01", "drug", "CCO"),
        _row("B", 2, "W01", "drug", "CCO"),
        _row("C", 3, "W01", EMPTY_CONTROL, None),  # the only control, mismatched
    ])
    split = split_metadata(df)
    with pytest.raises(ValueError) as exc:
        build_pair_index(split.treated, split.control)
    msg = str(exc.value)
    assert "A" in msg and "B" in msg  # both failing keys listed


# ---------- pair_index.py: sample within group only --------------------------

def test_pair_index_every_sample_matches_group_key():
    rows: list[dict] = []
    for exp in ("A", "B"):
        for plate in (1, 2):
            for w in range(3):
                rows.append(_row(exp, plate, f"T{w:02d}", "drug", "CCO"))
            for w in range(4):
                rows.append(_row(exp, plate, f"C{w:02d}", EMPTY_CONTROL, None))
    split = split_metadata(_make_df(rows))
    idx = build_pair_index(split.treated, split.control)

    treated_lookup = split.treated.set_index("metadata_idx")
    control_lookup = split.control.set_index("metadata_idx")
    rng = np.random.default_rng(1234)

    for t_idx in split.treated["metadata_idx"]:
        c_idx = idx.sample_control(int(t_idx), rng)
        t_row = treated_lookup.loc[t_idx]
        c_row = control_lookup.loc[c_idx]
        assert t_row["experiment_name"] == c_row["experiment_name"]
        assert int(t_row["plate"]) == int(c_row["plate"])


# ---------- pair_index.py: determinism --------------------------------------

def test_pair_index_deterministic_under_seed():
    """Same Generator seed → identical draw sequence."""
    df = _make_df([
        _row("A", 1, "W01", "drug", "CCO"),
        _row("A", 1, "W02", EMPTY_CONTROL, None),
        _row("A", 1, "W03", EMPTY_CONTROL, None),
        _row("A", 1, "W04", EMPTY_CONTROL, None),
        _row("A", 1, "W05", EMPTY_CONTROL, None),
    ])
    split = split_metadata(df)
    idx = build_pair_index(split.treated, split.control)
    t = int(split.treated.iloc[0]["metadata_idx"])

    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    seq_a = [idx.sample_control(t, rng_a) for _ in range(20)]
    seq_b = [idx.sample_control(t, rng_b) for _ in range(20)]
    assert seq_a == seq_b
    # And the draws actually vary across the pool (otherwise determinism is trivial)
    assert len(set(seq_a)) > 1


# ---------- pair_index.py: integrity ----------------------------------------

def test_pair_index_preserves_metadata_idx():
    df = _make_df([
        _row("A", 1, "W01", "drug", "CCO"),
        _row("A", 1, "W02", EMPTY_CONTROL, None),
        _row("A", 1, "W03", "drug", "CCC"),
    ])
    split = split_metadata(df)
    idx = build_pair_index(split.treated, split.control)
    all_idxs: set[int] = set()
    for group in idx.groups.values():
        all_idxs.update(group["treated"])
        all_idxs.update(group["control"])
    assert all_idxs == {0, 1, 2}  # the original df.index values


def test_pair_index_unknown_treated_idx_raises():
    df = _make_df([
        _row("A", 1, "W01", "drug", "CCO"),
        _row("A", 1, "W02", EMPTY_CONTROL, None),
    ])
    split = split_metadata(df)
    idx = build_pair_index(split.treated, split.control)
    with pytest.raises(KeyError):
        idx.sample_control(9999, np.random.default_rng(0))


def test_pair_index_len_equals_treated_count():
    df = _make_df([
        _row("A", 1, "W01", "drug", "CCO"),
        _row("A", 1, "W02", "drug", "CCC"),
        _row("A", 1, "W03", EMPTY_CONTROL, None),
    ])
    split = split_metadata(df)
    idx = build_pair_index(split.treated, split.control)
    assert len(idx) == len(split.treated)


def test_pair_index_raises_on_missing_key_columns():
    treated = pd.DataFrame({"metadata_idx": [0], "plate": [1]})  # missing experiment_name
    control = pd.DataFrame({"metadata_idx": [1], "experiment_name": ["A"], "plate": [1]})
    with pytest.raises(ValueError, match="experiment_name"):
        build_pair_index(treated, control)
