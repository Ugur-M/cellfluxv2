"""Tests for scripts/precompute_fingerprints.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# `scripts/` is on sys.path via conftest.py
from precompute_fingerprints import (  # noqa: E402
    select_treated,
    validate_and_dedupe_treatments,
)

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
L_ALA = "N[C@@H](C)C(=O)O"
D_ALA = "N[C@H](C)C(=O)O"

SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "precompute_fingerprints.py"
)


def _write_metadata(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run_script(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


# -- select_treated -----------------------------------------------------------

def test_select_treated_excludes_controls_crispr_and_blank():
    df = pd.DataFrame(
        {
            "treatment": [
                "aspirin",
                "EMPTY_control",
                "CRISPR_control",
                None,
                "no_smiles",
            ],
            "SMILES": [ASPIRIN, "", "", "", None],
            "perturbation_type": [
                "COMPOUND",
                "COMPOUND",
                "COMPOUND",
                "COMPOUND",
                "COMPOUND",
            ],
        }
    )
    out = select_treated(df)
    assert list(out["treatment"]) == ["aspirin"]


# -- validate_and_dedupe_treatments ------------------------------------------

def test_validate_dedupe_passes_through_single_smiles():
    df = pd.DataFrame(
        {
            "treatment": ["aspirin", "caffeine"],
            "SMILES": [ASPIRIN, CAFFEINE],
        }
    )
    out_df, info = validate_and_dedupe_treatments(
        df, radius=2, n_bits=1024, use_chirality=False
    )
    assert info == {}
    assert len(out_df) == 2


def test_validate_dedupe_allows_fp_equivalent_stereoisomers():
    """L/D-alanine collapse to the same FP under chirality=False; dedupe must keep one."""
    df = pd.DataFrame(
        {
            "treatment": ["alanine", "alanine"],
            "SMILES": [L_ALA, D_ALA],
        }
    )
    out_df, info = validate_and_dedupe_treatments(
        df, radius=2, n_bits=1024, use_chirality=False
    )
    assert set(info) == {"alanine"}
    assert sorted(info["alanine"]) == sorted([L_ALA, D_ALA])
    kept = out_df[out_df["treatment"] == "alanine"]
    assert len(kept) == 1
    assert kept["SMILES"].iloc[0] == sorted([L_ALA, D_ALA])[0]  # lex-smallest canonical


def test_validate_dedupe_raises_on_distinct_fps():
    """Two genuinely different compounds under the same treatment must raise."""
    df = pd.DataFrame(
        {
            "treatment": ["fake", "fake"],
            "SMILES": [ASPIRIN, CAFFEINE],
        }
    )
    with pytest.raises(ValueError, match="non-equivalent Morgan FPs"):
        validate_and_dedupe_treatments(
            df, radius=2, n_bits=1024, use_chirality=False
        )


def test_validate_dedupe_chirality_on_makes_enantiomers_distinct():
    """Under chirality=True, L/D-alanine become FP-distinct; validator must raise."""
    df = pd.DataFrame(
        {
            "treatment": ["alanine", "alanine"],
            "SMILES": [L_ALA, D_ALA],
        }
    )
    with pytest.raises(ValueError, match="non-equivalent Morgan FPs"):
        validate_and_dedupe_treatments(
            df, radius=2, n_bits=1024, use_chirality=True
        )


# -- script: hard-fail on parse failures -------------------------------------

def test_script_hard_fails_on_parse_failure(tmp_path: Path):
    """ANY SMILES parse failure → non-zero exit and no cache file written."""
    csv_path = _write_metadata(
        tmp_path / "meta.csv",
        [
            {"treatment": "good", "SMILES": ASPIRIN, "perturbation_type": "COMPOUND"},
            {"treatment": "bad", "SMILES": "totally-not-smiles-{}", "perturbation_type": "COMPOUND"},
        ],
    )
    out_path = tmp_path / "fp.npz"
    res = _run_script(["--metadata", str(csv_path), "--output", str(out_path)])
    assert res.returncode != 0, f"expected failure, got stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "failed to parse" in res.stderr.lower()
    assert not out_path.exists(), "partial cache must not be written on parse failure"


def test_script_succeeds_on_clean_input(tmp_path: Path):
    """Smoke: clean metadata produces a valid cache with use_chirality=False."""
    csv_path = _write_metadata(
        tmp_path / "meta.csv",
        [
            {"treatment": "aspirin", "SMILES": ASPIRIN, "perturbation_type": "COMPOUND"},
            {"treatment": "caffeine", "SMILES": CAFFEINE, "perturbation_type": "COMPOUND"},
        ],
    )
    out_path = tmp_path / "fp.npz"
    res = _run_script(["--metadata", str(csv_path), "--output", str(out_path)])
    assert res.returncode == 0, f"unexpected failure: stderr={res.stderr!r}"
    assert out_path.exists()
    with np.load(out_path, allow_pickle=False) as data:
        assert data["fps"].shape == (2, 1024)
        assert bool(data["use_chirality"]) is False


# -- script: --dedupe-report --------------------------------------------------

def test_script_writes_dedupe_report(tmp_path: Path):
    """--dedupe-report writes one row per multi-SMILES treatment."""
    csv_path = _write_metadata(
        tmp_path / "meta.csv",
        [
            {"treatment": "alanine", "SMILES": L_ALA, "perturbation_type": "COMPOUND"},
            {"treatment": "alanine", "SMILES": D_ALA, "perturbation_type": "COMPOUND"},
            {"treatment": "aspirin", "SMILES": ASPIRIN, "perturbation_type": "COMPOUND"},
        ],
    )
    out_path = tmp_path / "fp.npz"
    report_path = tmp_path / "dedupe.csv"
    res = _run_script(
        [
            "--metadata", str(csv_path),
            "--output", str(out_path),
            "--dedupe-report", str(report_path),
        ]
    )
    assert res.returncode == 0, f"unexpected failure: stderr={res.stderr!r}"
    assert report_path.exists()
    report = pd.read_csv(report_path)
    assert len(report) == 1
    row = report.iloc[0]
    assert row["treatment"] == "alanine"
    assert int(row["n_variants"]) == 2
    assert str(row["use_chirality"]).lower() == "false"
    dropped = json.loads(row["variants_dropped"])
    assert len(dropped) == 1
    # Canonical is lex-smallest; dropped is the other.
    assert row["canonical_smiles"] == sorted([L_ALA, D_ALA])[0]
    assert dropped[0] == sorted([L_ALA, D_ALA])[1]


def test_script_dedupe_report_empty_when_no_duplicates(tmp_path: Path):
    """Report is written (header-only) even when no treatments need dedup."""
    csv_path = _write_metadata(
        tmp_path / "meta.csv",
        [
            {"treatment": "aspirin", "SMILES": ASPIRIN, "perturbation_type": "COMPOUND"},
        ],
    )
    report_path = tmp_path / "dedupe.csv"
    res = _run_script(
        [
            "--metadata", str(csv_path),
            "--output", str(tmp_path / "fp.npz"),
            "--dedupe-report", str(report_path),
        ]
    )
    assert res.returncode == 0
    assert report_path.exists()
    report = pd.read_csv(report_path)
    assert len(report) == 0
    assert list(report.columns) == [
        "treatment",
        "n_variants",
        "canonical_smiles",
        "variants_dropped",
        "radius",
        "n_bits",
        "use_chirality",
    ]


# -- script: --use-chirality --------------------------------------------------

def test_script_use_chirality_flag_propagates_to_cache(tmp_path: Path):
    csv_path = _write_metadata(
        tmp_path / "meta.csv",
        [
            {"treatment": "aspirin", "SMILES": ASPIRIN, "perturbation_type": "COMPOUND"},
        ],
    )
    out_path = tmp_path / "fp.npz"
    res = _run_script(
        [
            "--metadata", str(csv_path),
            "--output", str(out_path),
            "--use-chirality",
        ]
    )
    assert res.returncode == 0, f"unexpected failure: stderr={res.stderr!r}"
    with np.load(out_path, allow_pickle=False) as data:
        assert bool(data["use_chirality"]) is True
