"""Smoke: how does metadata map onto the per-plate NPZ rows?

Two questions to answer end-to-end on real data:

1. For a sampled treated/control metadata row, does its ``address``
   appear in the matching NPZ's ``well`` array?  (sanity check on the
   experiment/plate -> NPZ path resolution and on the address column.)

2. Does ``metadata_idx`` equal the NPZ row index?  We test the
   hypothesis directly: for sampled rows where ``metadata_idx <
   len(npz["well"])``, compare ``metadata.address`` to
   ``npz["well"][metadata_idx]``.  If the alignment rate is **not** 100%,
   the dataset must use the per-plate ``address -> row_indices`` map
   that ``Plate`` already builds at load time.

Usage:
    python scripts/smoke_latent_mapping.py \\
        --metadata /teamspace/studios/this_studio/2DGen/data/rxrx3_core/metadata_rxrx3_core_normalized.csv \\
        --latent-root /teamspace/gcs_connections/sentinal4d/imaging/2d/single_cell/rxrx/rxrx3-core/features/rxrx3_core_mae_latents \\
        --n-samples 200
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from cellfluxv2.data.metadata import load_metadata  # noqa: E402
from cellfluxv2.data.pair_index import build_pair_index  # noqa: E402
from cellfluxv2.data.plate_cache import PlateCache  # noqa: E402


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) == 0:
        return df
    return df.sample(n=min(n, len(df)), random_state=seed)


def _fmt_stats(label: str, counts: list[int]) -> str:
    if not counts:
        return f"  {label}: (none)"
    return (
        f"  {label}: min={min(counts)} "
        f"med={int(statistics.median(counts))} "
        f"max={max(counts)} "
        f"(n={len(counts)})"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--latent-root", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    split = load_metadata(args.metadata)
    idx = build_pair_index(split.treated, split.control)

    # Restrict controls to (experiment, plate) groups that exist in the pair
    # index — i.e., controls on compound plates, not on gene/CRISPR plates.
    valid_keys = set(idx.groups.keys())
    control_keys = pd.MultiIndex.from_arrays(
        [
            split.control["experiment_name"].astype(str),
            split.control["plate"].astype(int),
        ]
    )
    control_mask = control_keys.isin(valid_keys)
    treated_sample = _sample(split.treated, args.n_samples, args.seed)
    control_sample = _sample(split.control[control_mask], args.n_samples, args.seed)
    print(
        f"[setup] treated_pool={len(split.treated)} "
        f"compound_control_pool={int(control_mask.sum())} "
        f"sampled treated={len(treated_sample)} control={len(control_sample)}",
        flush=True,
    )
    cache = PlateCache(args.latent_root, max_plates=16)

    print(f"sampled rows: treated={len(treated_sample)} control={len(control_sample)}")
    print(f"(seed={args.seed}, n-samples={args.n_samples})")
    print()

    records: list[dict] = []
    for label, sample in (("treated", treated_sample), ("control", control_sample)):
        for _, row in sample.iterrows():
            exp = str(row["experiment_name"])
            plate = int(row["plate"])
            addr = str(row["address"])
            m_idx = int(row["metadata_idx"])
            rec: dict = {
                "label": label,
                "metadata_idx": m_idx,
                "experiment": exp,
                "plate": plate,
                "address": addr,
                "npz_status": None,
                "addr_in_well": None,
                "n_cells_for_well": None,
                "idx_align_testable": False,
                "idx_align_match": None,
            }
            try:
                plate_obj = cache.get(exp, plate)
                rec["npz_status"] = "OK"
            except (FileNotFoundError, ValueError) as e:
                rec["npz_status"] = f"ERROR: {type(e).__name__}: {e}"
                records.append(rec)
                continue

            rec["addr_in_well"] = bool(addr in plate_obj.address_to_rows)
            rec["n_cells_for_well"] = (
                int(len(plate_obj.rows_for(addr))) if rec["addr_in_well"] else 0
            )
            if m_idx < plate_obj.n_cells():
                rec["idx_align_testable"] = True
                rec["idx_align_match"] = bool(
                    str(plate_obj.well[m_idx]) == addr
                )
            records.append(rec)

    df = pd.DataFrame(records)
    ok = df[df["npz_status"] == "OK"]

    # -- Section A: NPZ access ------------------------------------------------
    print("== NPZ access ==")
    print(f"NPZ open errors: {(df['npz_status'] != 'OK').sum()} / {len(df)}")
    if (df["npz_status"] != "OK").any():
        for s in df.loc[df["npz_status"] != "OK", "npz_status"].head(5):
            print(f"  {s}")
    print()

    # -- Section B: address in well column ------------------------------------
    print("== address in NPZ['well'] ==")
    print(f"rows with address present: {ok['addr_in_well'].sum()} / {len(ok)}")
    missing = ok[~ok["addr_in_well"].fillna(False)]
    if len(missing) > 0:
        print(f"  first 5 missing:")
        for _, r in missing.head(5).iterrows():
            print(f"    - {r['experiment']}/{r['plate']}/{r['address']}")
    cells_per_well = ok.loc[ok["addr_in_well"], "n_cells_for_well"].tolist()
    print(_fmt_stats("cells per matched well", cells_per_well))
    print()

    # -- Section C: metadata_idx == NPZ row index hypothesis ------------------
    print("== metadata_idx == NPZ row index hypothesis ==")
    testable = ok[ok["idx_align_testable"]]
    if len(testable) == 0:
        print("no rows testable (all metadata_idx >= per-plate cell count)")
        print("verdict: cannot confirm alignment; default to address-based lookup.")
    else:
        n_match = int(testable["idx_align_match"].sum())
        n_test = len(testable)
        rate = n_match / n_test
        print(f"tested: {n_test} (sampled rows where metadata_idx < n_cells of their plate)")
        print(f"matched: {n_match} ({rate:.2%})")
        if rate == 1.0:
            print("verdict: direct metadata_idx -> NPZ row indexing IS safe.")
        else:
            print(
                "verdict: metadata_idx != NPZ row. Must use the per-plate "
                "`address -> row_indices` map built by Plate (load_plate)."
            )
            print(
                "         Future dataset.py will sample a row uniformly from "
                "plate.rows_for(address)."
            )


if __name__ == "__main__":
    main()
