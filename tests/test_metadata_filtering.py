"""Tests for ``filter_split_by_missing_addresses``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cellfluxv2.data.metadata import (
    EMPTY_CONTROL,
    REQUIRED_COLUMNS,
    filter_split_by_missing_addresses,
    split_metadata,
)


# ---------- builders --------------------------------------------------------

def _row(exp, plate, address, treatment, smiles, ptype="COMPOUND"):
    return {
        "experiment_name": exp,
        "plate": plate,
        "address": address,
        "treatment": treatment,
        "SMILES": smiles,
        "perturbation_type": ptype,
    }


def _make_df(rows):
    df = pd.DataFrame(rows)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[list(REQUIRED_COLUMNS)]


def _make_split():
    """5 treated, 3 controls; metadata_idx aligns with row order in `_make_df`."""
    rows = [
        _row("A", 1, "W01", "drugA", "CCO"),            # idx 0 treated
        _row("A", 1, "W02", "drugB", "CCC"),            # idx 1 treated
        _row("A", 1, "W03", "drugC", "CCCC"),           # idx 2 treated
        _row("A", 1, "W04", "drugA", "CCO"),            # idx 3 treated
        _row("A", 1, "W05", "drugD", "CCCCC"),          # idx 4 treated
        _row("A", 1, "C01", EMPTY_CONTROL, None),       # idx 5 control
        _row("A", 1, "C02", EMPTY_CONTROL, None),       # idx 6 control
        _row("A", 1, "C03", EMPTY_CONTROL, None),       # idx 7 control
    ]
    return split_metadata(_make_df(rows))


def _write_missing_csv(path: Path, entries: list[dict]) -> Path:
    cols = ["role", "experiment", "plate", "address", "treatment", "metadata_idx"]
    if not entries:
        pd.DataFrame(columns=cols).to_csv(path, index=False)
    else:
        df = pd.DataFrame(entries)[cols]
        df.to_csv(path, index=False)
    return path


# ---------- tests -----------------------------------------------------------

def test_filter_drops_listed_treated_rows(tmp_path):
    split = _make_split()
    csv_path = _write_missing_csv(
        tmp_path / "missing.csv",
        [
            {"role": "treated", "experiment": "A", "plate": 1, "address": "W02",
             "treatment": "drugB", "metadata_idx": 1},
            {"role": "treated", "experiment": "A", "plate": 1, "address": "W04",
             "treatment": "drugA", "metadata_idx": 3},
        ],
    )
    filtered, report = filter_split_by_missing_addresses(split, csv_path)
    # idx 1 and 3 should be dropped; 0, 2, 4 kept.
    kept = sorted(filtered.treated["metadata_idx"].astype(int).tolist())
    assert kept == [0, 2, 4]
    assert report["dropped_treated_rows"] == 2
    assert report["filtered_treated_rows"] == 3
    assert report["raw_treated_rows"] == 5


def test_filter_drops_listed_control_rows(tmp_path):
    split = _make_split()
    csv_path = _write_missing_csv(
        tmp_path / "missing.csv",
        [
            {"role": "control", "experiment": "A", "plate": 1, "address": "C02",
             "treatment": EMPTY_CONTROL, "metadata_idx": 6},
        ],
    )
    filtered, report = filter_split_by_missing_addresses(split, csv_path)
    kept = sorted(filtered.control["metadata_idx"].astype(int).tolist())
    assert kept == [5, 7]
    assert report["dropped_control_rows"] == 1
    assert report["raw_control_rows"] == 3
    # Treated untouched.
    assert report["dropped_treated_rows"] == 0


def test_filter_only_drops_by_role(tmp_path):
    """A 'treated'-roled missing entry must not affect the control DataFrame even
    if its metadata_idx happens to match a control row's index."""
    split = _make_split()
    csv_path = _write_missing_csv(
        tmp_path / "missing.csv",
        [
            # Treated-role row referring to metadata_idx=5 (which is a control row).
            {"role": "treated", "experiment": "A", "plate": 1, "address": "X",
             "treatment": "x", "metadata_idx": 5},
        ],
    )
    filtered, report = filter_split_by_missing_addresses(split, csv_path)
    # No control row should be dropped.
    assert report["dropped_control_rows"] == 0
    assert report["dropped_treated_rows"] == 0  # no treated row has metadata_idx=5


def test_filter_does_not_mutate_input_split(tmp_path):
    split = _make_split()
    treated_before = split.treated.copy()
    control_before = split.control.copy()
    vocab_before = set(split.smiles_vocab)
    csv_path = _write_missing_csv(
        tmp_path / "missing.csv",
        [
            {"role": "treated", "experiment": "A", "plate": 1, "address": "W02",
             "treatment": "drugB", "metadata_idx": 1},
        ],
    )
    _, _ = filter_split_by_missing_addresses(split, csv_path)
    pd.testing.assert_frame_equal(split.treated, treated_before)
    pd.testing.assert_frame_equal(split.control, control_before)
    assert split.smiles_vocab == vocab_before


def test_filter_preserves_metadata_idx(tmp_path):
    split = _make_split()
    csv_path = _write_missing_csv(
        tmp_path / "missing.csv",
        [
            {"role": "treated", "experiment": "A", "plate": 1, "address": "W02",
             "treatment": "drugB", "metadata_idx": 1},
        ],
    )
    filtered, _ = filter_split_by_missing_addresses(split, csv_path)
    # metadata_idx column survives + DataFrame index preserved (no reset).
    assert "metadata_idx" in filtered.treated.columns
    assert list(filtered.treated.index) == [0, 2, 3, 4]
    assert list(filtered.treated["metadata_idx"].astype(int)) == [0, 2, 3, 4]


def test_filter_split_uses_metadata_idx_as_explicit_index(tmp_path):
    """Regression guard: the filtered split must have ``metadata_idx`` as
    its explicit DataFrame index (not via the RangeIndex-equals-metadata_idx
    coincidence). Covers four invariants downstream code relies on:

    1. ``metadata_idx`` is still present as a column.
    2. ``split.control.loc[m_idx]["metadata_idx"] == m_idx``.
    3. ``split.treated.loc[m_idx]["metadata_idx"] == m_idx`` for a
       surviving treated row.
    4. ``.iloc`` positional access still works — row order is unchanged
       by the index promotion.
    """
    split = _make_split()
    csv_path = _write_missing_csv(
        tmp_path / "missing.csv",
        [
            {"role": "treated", "experiment": "A", "plate": 1, "address": "W02",
             "treatment": "drugB", "metadata_idx": 1},
        ],
    )
    filtered, _ = filter_split_by_missing_addresses(split, csv_path)

    # (1) metadata_idx column survives the index promotion.
    assert "metadata_idx" in filtered.treated.columns
    assert "metadata_idx" in filtered.control.columns

    # (2) .loc on a control metadata_idx returns the row whose
    #     metadata_idx column equals the looked-up label.
    for ctrl_m_idx in [5, 6, 7]:  # all 3 control rows survive
        row = filtered.control.loc[ctrl_m_idx]
        assert int(row["metadata_idx"]) == ctrl_m_idx

    # (3) Same invariant for the surviving treated rows (idx 1 was
    #     dropped, so 0/2/3/4 remain).
    for t_m_idx in [0, 2, 3, 4]:
        row = filtered.treated.loc[t_m_idx]
        assert int(row["metadata_idx"]) == t_m_idx

    # (4) Positional .iloc access matches the surviving row order — row 0
    #     is still the first surviving treated row, etc.
    surviving_order = [0, 2, 3, 4]
    for pos, expected_m_idx in enumerate(surviving_order):
        row = filtered.treated.iloc[pos]
        assert int(row["metadata_idx"]) == expected_m_idx

    # Bonus invariant: the index name is cleared so it does not collide
    # with the same-named column, which would break a stray reset_index.
    assert filtered.treated.index.name is None
    assert filtered.control.index.name is None


def test_filter_recomputes_smiles_vocab(tmp_path):
    """Dropping drugB rows must remove 'CCC' from the vocab."""
    split = _make_split()
    assert "CCC" in split.smiles_vocab
    csv_path = _write_missing_csv(
        tmp_path / "missing.csv",
        [
            {"role": "treated", "experiment": "A", "plate": 1, "address": "W02",
             "treatment": "drugB", "metadata_idx": 1},
        ],
    )
    filtered, _ = filter_split_by_missing_addresses(split, csv_path)
    # Only drugB used "CCC"; vocab should lose it.
    assert "CCC" not in filtered.smiles_vocab
    # Other SMILES still present.
    assert "CCO" in filtered.smiles_vocab


def test_filter_empty_csv_returns_identical_counts(tmp_path):
    split = _make_split()
    csv_path = _write_missing_csv(tmp_path / "empty.csv", [])
    filtered, report = filter_split_by_missing_addresses(split, csv_path)
    assert report["dropped_treated_rows"] == 0
    assert report["dropped_control_rows"] == 0
    assert report["filtered_treated_rows"] == report["raw_treated_rows"] == 5
    assert report["filtered_control_rows"] == report["raw_control_rows"] == 3
    assert filtered.smiles_vocab == split.smiles_vocab


def test_filter_missing_required_columns_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"role": ["treated"], "metadata_idx": [0]}).to_csv(csv_path, index=False)
    split = _make_split()
    with pytest.raises(ValueError, match="missing required columns"):
        filter_split_by_missing_addresses(split, csv_path)


def test_filter_missing_csv_file_raises(tmp_path):
    split = _make_split()
    with pytest.raises(FileNotFoundError):
        filter_split_by_missing_addresses(split, tmp_path / "nope.csv")


def test_filter_report_includes_csv_path(tmp_path):
    split = _make_split()
    csv_path = _write_missing_csv(tmp_path / "missing.csv", [])
    _, report = filter_split_by_missing_addresses(split, csv_path)
    assert report["missing_csv_path"] == str(csv_path)
