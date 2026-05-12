"""Load and split the rxrx3 metadata CSV.

Splits rows into:
- treated:  ``perturbation_type == "COMPOUND"`` AND
            ``treatment != "EMPTY_control"`` AND non-empty SMILES.
- control:  ``treatment == "EMPTY_control"``.

The original CSV row index is preserved as a ``metadata_idx`` column on
both DataFrames before any filtering, so downstream code can map back
to the source CSV row without depending on positional indexing.

``filter_split_by_missing_addresses`` consumes a CSV of rows to drop
(produced upstream by an address-validation tool) and produces a
filtered split + a report. The dataset itself stays strict — it never
silently skips missing addresses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "experiment_name",
    "plate",
    "address",  # per-plate well coordinate, e.g. "AD37"; NPZ files store the same string under key "well"
    "treatment",
    "SMILES",
    "perturbation_type",
)
EMPTY_CONTROL = "EMPTY_control"


@dataclass
class MetadataSplit:
    """Treated and control row splits from one rxrx3 metadata CSV."""

    treated: pd.DataFrame
    control: pd.DataFrame
    smiles_vocab: set[str]

    def __post_init__(self) -> None:
        for df, name in ((self.treated, "treated"), (self.control, "control")):
            if "metadata_idx" not in df.columns:
                raise ValueError(f"{name} DataFrame is missing `metadata_idx`")
            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    raise ValueError(f"{name} DataFrame is missing column {col!r}")


def load_metadata(csv_path: str | Path) -> MetadataSplit:
    """Read the rxrx3 metadata CSV and split into treated / control."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    return split_metadata(df)


def split_metadata(df: pd.DataFrame) -> MetadataSplit:
    """Pure split function over an already-loaded metadata DataFrame.

    The input ``df.index`` is captured as ``metadata_idx`` before any
    filtering. The treated / control DataFrames returned are filtered
    views (copies); neither resets nor drops ``metadata_idx``.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"metadata is missing required columns: {missing}; "
            f"present columns: {list(df.columns)}"
        )

    df = df.copy()
    df["metadata_idx"] = df.index

    is_control = df["treatment"] == EMPTY_CONTROL
    is_treated = (
        (df["perturbation_type"] == "COMPOUND")
        & (df["treatment"] != EMPTY_CONTROL)
        & df["SMILES"].notna()
        & (df["SMILES"].astype(str).str.len() > 0)
    )

    treated = df[is_treated].copy()
    control = df[is_control].copy()
    vocab = set(treated["SMILES"].astype(str).unique())

    return MetadataSplit(treated=treated, control=control, smiles_vocab=vocab)


MISSING_CSV_REQUIRED_COLUMNS: tuple[str, ...] = (
    "role",
    "experiment",
    "plate",
    "address",
    "treatment",
    "metadata_idx",
)


def filter_split_by_missing_addresses(
    split: MetadataSplit, missing_csv_path: str | Path
) -> tuple[MetadataSplit, dict]:
    """Drop rows whose ``metadata_idx`` is listed in a missing-addresses CSV.

    The CSV is expected to have one row per metadata row that failed
    upstream address validation, with columns
    ``(role, experiment, plate, address, treatment, metadata_idx)`` —
    ``role`` is ``"treated"`` or ``"control"``. Rows with
    ``role == "treated"`` filter the treated DataFrame; rows with
    ``role == "control"`` filter the control DataFrame.

    The original split is **not** mutated. ``metadata_idx`` is preserved
    on the survivors. ``smiles_vocab`` is recomputed from the filtered
    treated rows. Returns ``(filtered_split, report)``; the report carries
    the raw/filtered/dropped row counts and the source CSV path.
    """
    missing_csv_path = Path(missing_csv_path)
    if not missing_csv_path.exists():
        raise FileNotFoundError(f"missing-addresses CSV not found: {missing_csv_path}")
    missing_df = pd.read_csv(missing_csv_path)

    missing_cols = [c for c in MISSING_CSV_REQUIRED_COLUMNS if c not in missing_df.columns]
    if missing_cols:
        raise ValueError(
            f"missing-addresses CSV at {missing_csv_path} is missing required "
            f"columns: {missing_cols}; present columns: {list(missing_df.columns)}"
        )

    treated_drop_idx = set(
        missing_df.loc[missing_df["role"] == "treated", "metadata_idx"].astype(int).tolist()
    )
    control_drop_idx = set(
        missing_df.loc[missing_df["role"] == "control", "metadata_idx"].astype(int).tolist()
    )

    raw_treated = len(split.treated)
    raw_control = len(split.control)

    treated_keep = ~split.treated["metadata_idx"].astype(int).isin(treated_drop_idx)
    control_keep = ~split.control["metadata_idx"].astype(int).isin(control_drop_idx)
    filtered_treated = split.treated[treated_keep].copy()
    filtered_control = split.control[control_keep].copy()

    vocab = (
        set(filtered_treated["SMILES"].astype(str).unique())
        if len(filtered_treated) > 0
        else set()
    )

    filtered_split = MetadataSplit(
        treated=filtered_treated,
        control=filtered_control,
        smiles_vocab=vocab,
    )
    report = {
        "raw_treated_rows": raw_treated,
        "raw_control_rows": raw_control,
        "filtered_treated_rows": len(filtered_treated),
        "filtered_control_rows": len(filtered_control),
        "dropped_treated_rows": raw_treated - len(filtered_treated),
        "dropped_control_rows": raw_control - len(filtered_control),
        "missing_csv_path": str(missing_csv_path),
    }
    return filtered_split, report
