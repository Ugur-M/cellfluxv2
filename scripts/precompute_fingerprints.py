"""Precompute Morgan (ECFP) fingerprints for the rxrx3 compound vocabulary.

Reads the rxrx3 metadata CSV, filters to treated compound rows, validates
that every treatment maps to a single SMILES, computes Morgan fingerprints
for each unique ``(treatment, SMILES)`` pair, and writes the table to an
``.npz`` the dataset loader can mmap.

Usage:
    python scripts/precompute_fingerprints.py \\
        --metadata /teamspace/studios/this_studio/2DGen/data/rxrx3_core/metadata_rxrx3_core_normalized.csv \\
        --output data/fingerprints_morgan_r2_b1024.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from cellfluxv2.data.fingerprints import compute_morgan  # noqa: E402

CONTROL_TREATMENTS = {"EMPTY_control", "CRISPR_control"}


def select_treated(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that should appear in the compound fingerprint table."""
    return df[
        (df["perturbation_type"] == "COMPOUND")
        & df["treatment"].notna()
        & (~df["treatment"].isin(CONTROL_TREATMENTS))
        & df["SMILES"].notna()
        & (df["SMILES"].astype(str).str.len() > 0)
    ]


def validate_and_dedupe_treatments(
    treated: pd.DataFrame, radius: int, n_bits: int
) -> pd.DataFrame:
    """Reconcile treatments that map to multiple SMILES.

    Some rxrx3 treatments carry several SMILES strings that differ only in
    stereochemistry annotations (`[C@H]` vs `[C@@H]`, `|r|` markers, etc.).
    Default Morgan fingerprints are stereo-blind, so these variants
    produce identical bit vectors and are safe to collapse.

    For each treatment with multiple SMILES:
      - If all variants produce **identical** Morgan FPs at the requested
        ``(radius, n_bits)``, keep the lexicographically smallest SMILES
        as canonical and drop the rest (logged).
      - If any variants produce **different** FPs, raise — that is the
        genuinely unexpected case the user asked us to surface.
    """
    counts = treated.groupby("treatment")["SMILES"].nunique()
    multi = counts[counts > 1]
    if len(multi) == 0:
        return treated

    canonical: dict[str, str] = {}
    truly_unexpected: list[tuple[str, list[str], str]] = []

    for treatment in multi.index:
        variants = sorted(
            treated.loc[treated["treatment"] == treatment, "SMILES"].unique().tolist()
        )
        try:
            fps = [compute_morgan(s, radius=radius, n_bits=n_bits) for s in variants]
        except ValueError as e:
            truly_unexpected.append((treatment, variants, f"RDKit parse failure: {e}"))
            continue
        base = fps[0]
        if not all(np.array_equal(base, fp) for fp in fps[1:]):
            truly_unexpected.append(
                (treatment, variants, f"Morgan FPs differ across variants (r={radius}, b={n_bits})")
            )
            continue
        canonical[treatment] = variants[0]  # lexicographically smallest

    if truly_unexpected:
        details = "\n".join(
            f"  - {t}: {reason}\n      variants: {v}"
            for t, v, reason in truly_unexpected
        )
        raise ValueError(
            f"{len(truly_unexpected)} treatment(s) map to multiple SMILES with "
            f"non-equivalent Morgan FPs; this is genuinely unexpected and "
            f"requires metadata resolution:\n{details}"
        )

    print(
        f"[fp] note: {len(canonical)} treatment(s) had multiple SMILES that "
        f"produce identical Morgan FPs (stereo annotations); keeping "
        f"lexicographically smallest variant:",
        file=sys.stderr,
    )
    for t, c in list(canonical.items())[:10]:
        print(f"    - {t}: kept {c!r}", file=sys.stderr)
    if len(canonical) > 10:
        print(f"    ... and {len(canonical) - 10} more", file=sys.stderr)

    keep_mask = pd.Series(True, index=treated.index)
    for treatment, canonical_smiles in canonical.items():
        is_treatment = treated["treatment"] == treatment
        is_noncanonical = is_treatment & (treated["SMILES"] != canonical_smiles)
        keep_mask &= ~is_noncanonical
    return treated[keep_mask]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=1024)
    args = p.parse_args()

    df = pd.read_csv(args.metadata)
    print(f"[fp] metadata: {len(df)} rows from {args.metadata}", file=sys.stderr)

    treated = select_treated(df)
    print(f"[fp] treated compound rows: {len(treated)}", file=sys.stderr)

    treated = validate_and_dedupe_treatments(treated, args.radius, args.n_bits)

    pairs = (
        treated[["treatment", "SMILES"]]
        .drop_duplicates()
        .sort_values(["treatment", "SMILES"])
        .reset_index(drop=True)
    )
    print(
        f"[fp] unique (treatment, SMILES) pairs: {len(pairs)} "
        f"(unique treatments={treated['treatment'].nunique()}, "
        f"unique SMILES={treated['SMILES'].nunique()})",
        file=sys.stderr,
    )

    fps_list: list[np.ndarray] = []
    smiles_list: list[str] = []
    treatments_list: list[str] = []
    failures: list[tuple[str, str, str]] = []

    for _, row in pairs.iterrows():
        try:
            fp = compute_morgan(row["SMILES"], radius=args.radius, n_bits=args.n_bits)
        except ValueError as e:
            failures.append((row["treatment"], row["SMILES"], str(e)))
            continue
        fps_list.append(fp)
        smiles_list.append(row["SMILES"])
        treatments_list.append(row["treatment"])

    if failures:
        print(
            f"[fp] WARNING: {len(failures)} SMILES failed to parse (first 5):",
            file=sys.stderr,
        )
        for t, s, e in failures[:5]:
            print(f"    - {t}: {s!r} -> {e}", file=sys.stderr)

    if not fps_list:
        raise RuntimeError("No fingerprints produced; nothing to write.")

    fps = np.stack(fps_list, axis=0).astype(np.uint8)
    smiles = np.array(smiles_list, dtype=str)
    treatments = np.array(treatments_list, dtype=str)

    assert fps.shape == (len(smiles), args.n_bits)
    assert len(treatments) == len(smiles)
    if not ((fps == 0) | (fps == 1)).all():
        raise RuntimeError("Computed fingerprints contain non-binary values.")

    empties = np.where(fps.sum(axis=1) == 0)[0]
    if len(empties) > 0:
        print(
            f"[fp] WARNING: {len(empties)} all-zero fingerprint(s) "
            "(possible RDKit edge case on very small molecules):",
            file=sys.stderr,
        )
        for i in empties[:5]:
            print(f"    - {treatments[i]}: {smiles[i]!r}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fps=fps,
        smiles=smiles,
        treatments=treatments,
        radius=np.int32(args.radius),
        n_bits=np.int32(args.n_bits),
    )
    size_mb = args.output.stat().st_size / 1e6
    print(
        f"[fp] wrote {args.output} ({size_mb:.2f} MB): "
        f"{len(fps)} fingerprints x {args.n_bits} bits, radius={args.radius}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
