"""Precompute Morgan (ECFP) fingerprints for the rxrx3 compound vocabulary.

Reads the rxrx3 metadata CSV, filters to treated compound rows, computes
Morgan fingerprints for each unique ``(treatment, SMILES)`` pair, and
writes the table to an ``.npz`` the dataset loader can mmap.

Multi-SMILES treatments are allowed only when every variant produces an
identical Morgan fingerprint under the configured ``(radius, n_bits,
use_chirality)`` settings — this is the expected stereo-annotation case
(``[C@H]`` vs ``[C@@H]`` under the default stereo-blind Morgan). If any
variant pair produces *different* fingerprints, the script raises so the
metadata can be reconciled.

Any SMILES that fails to parse raises and no cache is written, so we
never produce a partial table.

Usage:
    python scripts/precompute_fingerprints.py \\
        --metadata /teamspace/studios/this_studio/2DGen/data/rxrx3_core/metadata_rxrx3_core_normalized.csv \\
        --output data/fingerprints_morgan_r2_b1024.npz \\
        [--use-chirality] \\
        [--dedupe-report data/dedupe_report.csv]
"""
from __future__ import annotations

import argparse
import json
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
    treated: pd.DataFrame,
    radius: int,
    n_bits: int,
    use_chirality: bool,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Reconcile treatments that map to multiple SMILES.

    For each treatment with multiple SMILES:
      - If every variant produces **identical** Morgan FPs at the requested
        ``(radius, n_bits, use_chirality)``, keep the lexicographically
        smallest SMILES as canonical and drop the rest.
      - If any variants produce **different** FPs, raise — the genuinely
        unexpected case.

    Returns
    -------
    (filtered_df, dedupe_info)
        ``filtered_df`` is ``treated`` with non-canonical SMILES removed.
        ``dedupe_info`` maps each multi-SMILES treatment to the full sorted
        list of its SMILES variants (the first entry is the canonical one
        kept; the rest were dropped).
    """
    counts = treated.groupby("treatment")["SMILES"].nunique()
    multi = counts[counts > 1]
    if len(multi) == 0:
        return treated, {}

    dedupe_info: dict[str, list[str]] = {}
    truly_unexpected: list[tuple[str, list[str], str]] = []

    for treatment in multi.index:
        variants = sorted(
            treated.loc[treated["treatment"] == treatment, "SMILES"].unique().tolist()
        )
        try:
            fps = [
                compute_morgan(s, radius=radius, n_bits=n_bits, use_chirality=use_chirality)
                for s in variants
            ]
        except ValueError as e:
            truly_unexpected.append((treatment, variants, f"RDKit parse failure: {e}"))
            continue
        base = fps[0]
        if not all(np.array_equal(base, fp) for fp in fps[1:]):
            truly_unexpected.append(
                (
                    treatment,
                    variants,
                    f"Morgan FPs differ across variants "
                    f"(r={radius}, b={n_bits}, chirality={use_chirality})",
                )
            )
            continue
        dedupe_info[treatment] = variants  # variants[0] is canonical

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

    if dedupe_info:
        print(
            f"[fp] note: {len(dedupe_info)} treatment(s) had multiple SMILES that "
            f"produce identical Morgan FPs under the configured settings; "
            f"keeping lexicographically smallest variant:",
            file=sys.stderr,
        )
        for t, v in list(dedupe_info.items())[:10]:
            print(f"    - {t}: kept {v[0]!r}", file=sys.stderr)
        if len(dedupe_info) > 10:
            print(f"    ... and {len(dedupe_info) - 10} more", file=sys.stderr)

    keep_mask = pd.Series(True, index=treated.index)
    for treatment, variants in dedupe_info.items():
        canonical = variants[0]
        is_treatment = treated["treatment"] == treatment
        is_noncanonical = is_treatment & (treated["SMILES"] != canonical)
        keep_mask &= ~is_noncanonical
    return treated[keep_mask], dedupe_info


def write_dedupe_report(
    dedupe_info: dict[str, list[str]],
    out_path: Path,
    radius: int,
    n_bits: int,
    use_chirality: bool,
) -> None:
    """Write one row per treatment with multiple SMILES variants."""
    rows = []
    for treatment, variants in sorted(dedupe_info.items()):
        canonical = variants[0]
        dropped = variants[1:]
        rows.append(
            {
                "treatment": treatment,
                "n_variants": len(variants),
                "canonical_smiles": canonical,
                "variants_dropped": json.dumps(dropped),
                "radius": radius,
                "n_bits": n_bits,
                "use_chirality": use_chirality,
            }
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "treatment",
            "n_variants",
            "canonical_smiles",
            "variants_dropped",
            "radius",
            "n_bits",
            "use_chirality",
        ],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=1024)
    p.add_argument(
        "--use-chirality",
        action="store_true",
        help="Encode stereo information in Morgan FPs (default off).",
    )
    p.add_argument(
        "--dedupe-report",
        type=Path,
        default=None,
        help="Optional CSV: one row per treatment with multiple SMILES variants.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.metadata)
    print(f"[fp] metadata: {len(df)} rows from {args.metadata}", file=sys.stderr)

    treated = select_treated(df)
    print(f"[fp] treated compound rows: {len(treated)}", file=sys.stderr)

    treated, dedupe_info = validate_and_dedupe_treatments(
        treated, args.radius, args.n_bits, args.use_chirality
    )

    if args.dedupe_report is not None:
        write_dedupe_report(
            dedupe_info,
            args.dedupe_report,
            args.radius,
            args.n_bits,
            args.use_chirality,
        )
        print(
            f"[fp] wrote dedupe report ({len(dedupe_info)} rows) to "
            f"{args.dedupe_report}",
            file=sys.stderr,
        )

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
            fp = compute_morgan(
                row["SMILES"],
                radius=args.radius,
                n_bits=args.n_bits,
                use_chirality=args.use_chirality,
            )
        except ValueError as e:
            failures.append((row["treatment"], row["SMILES"], str(e)))
            continue
        fps_list.append(fp)
        smiles_list.append(row["SMILES"])
        treatments_list.append(row["treatment"])

    if failures:
        head = "\n".join(f"  - {t}: {s!r} -> {e}" for t, s, e in failures[:20])
        tail = f"\n  ... and {len(failures) - 20} more" if len(failures) > 20 else ""
        raise ValueError(
            f"{len(failures)} SMILES failed to parse; refusing to write a "
            f"partial fingerprint cache. Resolve in metadata before re-running.\n"
            f"{head}{tail}"
        )

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
        use_chirality=np.bool_(args.use_chirality),
    )
    size_mb = args.output.stat().st_size / 1e6
    print(
        f"[fp] wrote {args.output} ({size_mb:.2f} MB): "
        f"{len(fps)} fingerprints x {args.n_bits} bits, "
        f"radius={args.radius}, use_chirality={args.use_chirality}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
