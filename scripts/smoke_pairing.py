"""Real-data smoke summary for metadata + pair index.

Loads the rxrx3 metadata CSV, splits into treated/control, and prints:
  - treated / control row counts
  - unique (experiment_name, plate) group counts (treated-only, control-only, both)
  - failing groups (treated keys with no controls — would break pair_index)
  - min / median / max treated and control counts per group

It then attempts to build the pair index and reports success or the
exact ValueError so the smoke output explains why a real run would fail.

Usage:
    python scripts/smoke_pairing.py \\
        --metadata /teamspace/studios/this_studio/2DGen/data/rxrx3_core/metadata_rxrx3_core_normalized.csv
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from cellfluxv2.data.metadata import load_metadata  # noqa: E402
from cellfluxv2.data.pair_index import build_pair_index  # noqa: E402


def _fmt_stats(label: str, counts: list[int]) -> str:
    if not counts:
        return f"  {label}: (no groups)"
    return (
        f"  {label}: min={min(counts)} "
        f"med={int(statistics.median(counts))} "
        f"max={max(counts)} (over {len(counts)} groups)"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, required=True)
    args = p.parse_args()

    split = load_metadata(args.metadata)

    print(f"treated rows: {len(split.treated)}")
    print(f"control rows: {len(split.control)}")
    print(f"unique SMILES (vocab): {len(split.smiles_vocab)}")
    print()

    treated_keys = set(zip(split.treated["experiment_name"], split.treated["plate"]))
    control_keys = set(zip(split.control["experiment_name"], split.control["plate"]))
    both = treated_keys & control_keys
    treated_only = treated_keys - control_keys
    control_only = control_keys - treated_keys

    print("(experiment_name, plate) groups:")
    print(f"  any side: {len(treated_keys | control_keys)}")
    print(f"  with both treated and controls: {len(both)}")
    print(f"  treated-only (failing — no controls): {len(treated_only)}")
    print(f"  control-only (unused — no treated): {len(control_only)}")
    print()

    treated_counts = (
        split.treated.groupby(["experiment_name", "plate"]).size().tolist()
    )
    control_counts = (
        split.control.groupby(["experiment_name", "plate"]).size().tolist()
    )
    print("per-group counts:")
    print(_fmt_stats("treated", treated_counts))
    print(_fmt_stats("control", control_counts))
    print()

    if treated_only:
        print(f"first 10 failing groups (treated with no controls):")
        for key in list(sorted(treated_only))[:10]:
            print(f"  - {key}")
        print()

    try:
        idx = build_pair_index(split.treated, split.control)
        print(
            f"pair index built OK: {len(idx)} treated rows over "
            f"{len(idx.groups)} (experiment, plate) groups."
        )
    except ValueError as e:
        print("pair index FAILED to build:")
        print(str(e))


if __name__ == "__main__":
    main()
