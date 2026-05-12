"""Real-data smoke: build CellFluxDataset for stage 1 and stage 2, pull 8 items, build a batch.

Verifies end-to-end on the rxrx3 mount:
  - dataset constructs from real metadata + pair_index + plate_cache + fp_cache + norm stats
  - per-item shapes / dtypes / finite / range
  - stage-2 (experiment, plate) consistency between treated and control
  - DataLoader batches via CellFluxDataset.collate to the expected shapes

Usage:
    python scripts/smoke_data.py \\
        --metadata /teamspace/studios/this_studio/2DGen/data/rxrx3_core/metadata_rxrx3_core_normalized.csv \\
        --latent-root /teamspace/gcs_connections/sentinal4d/imaging/2d/single_cell/rxrx/rxrx3-core/features/rxrx3_core_mae_latents \\
        --norm-stats /teamspace/studios/this_studio/2DGen/data/rxrx3_core/norm_stats_controls.pt \\
        --fp-cache data/fingerprints_morgan_r2_b1024.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from cellfluxv2.data.dataset import CellFluxDataset  # noqa: E402
from cellfluxv2.data.fingerprints import load_fp_cache  # noqa: E402
from cellfluxv2.data.latent_norm import load_norm_stats  # noqa: E402
from cellfluxv2.data.metadata import load_metadata  # noqa: E402
from cellfluxv2.data.pair_index import build_pair_index  # noqa: E402
from cellfluxv2.data.plate_cache import PlateCache  # noqa: E402


def _tensor_summary(name: str, t: torch.Tensor) -> str:
    return (
        f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={t.min().item():.3f} max={t.max().item():.3f} "
        f"mean={t.mean().item():.3f} std={t.std().item():.3f} "
        f"finite={bool(torch.isfinite(t).all())}"
    )


def _condition_summary(c: torch.Tensor) -> str:
    return (
        f"  condition: shape={tuple(c.shape)} dtype={c.dtype} "
        f"set_bits={int(c.sum().item())} density={c.mean().item():.4f} "
        f"finite={bool(torch.isfinite(c).all())}"
    )


def run_stage(
    stage: int,
    split,
    pair_idx,
    plate_cache,
    fp_cache,
    mean,
    std,
    n_items: int = 8,
    batch_size: int = 8,
    seed: int = 0,
) -> None:
    ds = CellFluxDataset(
        metadata_split=split,
        pair_index=pair_idx,
        plate_cache=plate_cache,
        fp_cache=fp_cache,
        mean=mean,
        std=std,
        stage=stage,
        rng_seed=seed,
    )
    print(f"\n=== Stage {stage} ===", flush=True)
    print(f"len(dataset) = {len(ds)}", flush=True)

    items = [ds[i] for i in range(n_items)]
    print(f"\nFirst item ({n_items} pulled total):", flush=True)
    print(_tensor_summary("x0", items[0]["x0"]), flush=True)
    print(_tensor_summary("x1", items[0]["x1"]), flush=True)
    print(_condition_summary(items[0]["condition"]), flush=True)
    print(f"  meta.treatment = {items[0]['meta']['treatment']!r}", flush=True)
    print(
        f"  meta.experiment_name/plate/treated_address = "
        f"{items[0]['meta']['experiment_name']!r}/{items[0]['meta']['plate']}/"
        f"{items[0]['meta']['treated_address']!r}",
        flush=True,
    )

    # Aggregate stats over the n_items.
    all_x0 = torch.stack([it["x0"] for it in items])
    all_x1 = torch.stack([it["x1"] for it in items])
    print(f"\nAggregate over {n_items} items:", flush=True)
    print(
        f"  x0: range [{all_x0.min().item():.3f}, {all_x0.max().item():.3f}] "
        f"mean={all_x0.mean().item():.3f} std={all_x0.std().item():.3f}",
        flush=True,
    )
    print(
        f"  x1: range [{all_x1.min().item():.3f}, {all_x1.max().item():.3f}] "
        f"mean={all_x1.mean().item():.3f} std={all_x1.std().item():.3f}",
        flush=True,
    )

    treated_addrs = {it["meta"]["treated_address"] for it in items}
    print(f"\nUnique treated addresses sampled (of {n_items}): {len(treated_addrs)}", flush=True)

    if stage == 2:
        control_addrs = {it["meta"]["control_address"] for it in items}
        print(
            f"Unique control addresses sampled (of {n_items}): {len(control_addrs)}",
            flush=True,
        )
        # (experiment, plate) consistency check
        mismatches = 0
        for it in items:
            m = it["meta"]
            ctrl_row = ds.split.control.loc[m["control_metadata_idx"]]
            if (
                str(ctrl_row["experiment_name"]) != m["experiment_name"]
                or int(ctrl_row["plate"]) != m["plate"]
            ):
                mismatches += 1
        print(
            f"Stage 2 pairing check: {n_items - mismatches} / {n_items} items "
            f"have matching (experiment, plate)",
            flush=True,
        )
        assert mismatches == 0
    else:
        # Stage-1 sanity: x0 should be close to N(0, I).
        print(
            f"Stage 1 noise check: x0 mean={all_x0.mean().item():.3f} (target ~0), "
            f"std={all_x0.std().item():.3f} (target ~1)",
            flush=True,
        )

    print(f"\nBuilding DataLoader batch of {batch_size}...", flush=True)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=CellFluxDataset.collate)
    batch = next(iter(loader))
    assert tuple(batch["x0"].shape) == (batch_size, 169, 8), batch["x0"].shape
    assert tuple(batch["x1"].shape) == (batch_size, 169, 8), batch["x1"].shape
    assert tuple(batch["condition"].shape) == (batch_size, 1024), batch["condition"].shape
    print(
        f"Batch shapes OK: x0={tuple(batch['x0'].shape)} "
        f"x1={tuple(batch['x1'].shape)} condition={tuple(batch['condition'].shape)}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--latent-root", type=Path, required=True)
    p.add_argument("--norm-stats", type=Path, required=True)
    p.add_argument("--fp-cache", type=Path, required=True)
    p.add_argument("--n-items", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"[load] metadata: {args.metadata}", flush=True)
    split = load_metadata(args.metadata)
    print(
        f"       treated={len(split.treated)} control={len(split.control)} "
        f"smiles_vocab={len(split.smiles_vocab)}",
        flush=True,
    )

    print(f"[build] pair_index ...", flush=True)
    pair_idx = build_pair_index(split.treated, split.control)
    print(f"        {len(pair_idx.groups)} (experiment, plate) groups", flush=True)

    print(f"[load] fingerprint cache: {args.fp_cache}", flush=True)
    fp_cache = load_fp_cache(args.fp_cache)
    print(
        f"       {len(fp_cache)} compounds, n_bits={fp_cache.n_bits}, "
        f"use_chirality={fp_cache.use_chirality}",
        flush=True,
    )

    print(f"[load] norm stats: {args.norm_stats}", flush=True)
    mean, std = load_norm_stats(args.norm_stats)
    print(
        f"       mean range [{mean.min().item():.3f}, {mean.max().item():.3f}] "
        f"std range [{std.min().item():.3f}, {std.max().item():.3f}]",
        flush=True,
    )

    plate_cache = PlateCache(args.latent_root, max_plates=16)

    run_stage(
        1, split, pair_idx, plate_cache, fp_cache, mean, std,
        n_items=args.n_items, batch_size=args.batch_size, seed=args.seed,
    )
    run_stage(
        2, split, pair_idx, plate_cache, fp_cache, mean, std,
        n_items=args.n_items, batch_size=args.batch_size, seed=args.seed,
    )

    print("\nALL SMOKE CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
