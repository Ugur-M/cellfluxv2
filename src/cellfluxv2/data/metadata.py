"""Load and split the rxrx3 metadata CSV.

Splits rows into:
- treated:  ``perturbation_type == "COMPOUND"`` AND
            ``treatment != "EMPTY_control"`` AND non-empty SMILES.
- control:  ``treatment == "EMPTY_control"``.

The original CSV row index is preserved as a ``metadata_idx`` column on
both DataFrames before any filtering, so downstream code can map back
to the source CSV row without depending on positional indexing.
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
