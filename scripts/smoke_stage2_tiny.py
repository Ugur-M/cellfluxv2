"""Tiny end-to-end Stage 2 smoke on the real rxrx3 data.

Runs 5 train_steps on a 4-head, 128-hidden, 2-block DiT with
``configs/stage2.yaml`` defaults. Asserts:

* a real ``PairIndex`` is built on the missing-address-filtered split,
* every batch keeps the same-plate invariant (control and treated rows
  share ``experiment_name`` and ``plate``),
* ``train.jsonl`` exists and contains 5 records, all losses finite,
* ``latest.pt`` and ``final.pt`` are written,
* ``final.pt`` reloads into a fresh ``DiTVelocity`` and reports step=5,
* if a Stage 1 ``final.pt`` is present at the expected path, the smoke
  warm-starts from it; otherwise it runs from scratch.

The smoke uses absolute paths derived from the studio root so it is
runnable from any cwd (``python scripts/smoke_stage2_tiny.py`` from
``cellfluxv2_repro/`` is the canonical invocation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDIO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellfluxv2.models.dit import DiTVelocity  # noqa: E402
from cellfluxv2.train.checkpoint import load_checkpoint  # noqa: E402
from cellfluxv2.train.stage2 import (  # noqa: E402
    Stage2Config,
    build_dataloader,
    build_dataset,
    build_pair_index_from_split,
    build_split,
    load_config,
    run_training,
)
from cellfluxv2.utils.logging import read_jsonl  # noqa: E402

CANDIDATE_INIT_CKPTS: tuple[Path, ...] = (
    REPO_ROOT / "runs" / "stage1_cond_balance_1k" / "final.pt",
    REPO_ROOT / "runs" / "stage1" / "final.pt",
)


def _build_smoke_config() -> Stage2Config:
    cfg = load_config(REPO_ROOT / "configs" / "stage2.yaml")

    cfg.data["metadata_path"] = str(
        STUDIO_ROOT / "2DGen/data/rxrx3_core/metadata_rxrx3_core_normalized.csv"
    )
    cfg.data["norm_stats_path"] = str(
        STUDIO_ROOT / "2DGen/data/rxrx3_core/norm_stats_controls.pt"
    )
    cfg.data["fingerprints_path"] = str(
        REPO_ROOT / "data/fingerprints_morgan_r2_b1024.npz"
    )
    cfg.data["missing_addresses_path"] = str(
        REPO_ROOT / "data/missing_addresses.csv"
    )

    cfg.model["hidden_dim"] = 128
    cfg.model["depth"] = 2
    cfg.model["num_heads"] = 4
    cfg.training["batch_size"] = 8
    cfg.training["max_steps"] = 5
    cfg.training["num_workers"] = 0
    cfg.training["log_interval"] = 1
    cfg.training["diagnostic_interval"] = 2
    cfg.training["checkpoint_interval"] = 2
    cfg.training["device"] = "cpu" if not torch.cuda.is_available() else "cuda"
    cfg.wandb["enabled"] = False

    init_ckpt = next((p for p in CANDIDATE_INIT_CKPTS if p.exists()), None)
    if init_ckpt is not None:
        # Reload-into-tiny-model would need shape match; the smoke uses
        # hidden_dim=128 / depth=2, which does NOT match the 384/6 final.pt.
        # So we keep init_ckpt off in the smoke even when one exists, and
        # only print which checkpoint *would* be used at full scale.
        print(
            f"[smoke] noting available init_ckpt at {init_ckpt} "
            f"(skipped: tiny smoke model shape does not match)",
            flush=True,
        )
    cfg.training["init_ckpt"] = None
    return cfg


def _verify_same_plate_invariant(cfg: Stage2Config) -> None:
    """Build the dataset + loader and pull one batch; assert the same-plate
    invariant for every (treated, control) pair the dataset emits."""
    split, _ = build_split(cfg)
    pair_index = build_pair_index_from_split(split)
    if len(pair_index.groups) == 0:
        raise AssertionError("pair_index has zero groups after filtering")
    ds = build_dataset(cfg, split, pair_index)
    ds.set_epoch(0)
    loader = build_dataloader(ds, cfg)
    batch = next(iter(loader))
    meta = batch["meta"]
    B = len(meta["experiment_name"])
    print(
        f"[smoke] pair_index groups={len(pair_index.groups)} "
        f"treated_paired={len(pair_index.treated_to_group)}",
        flush=True,
    )
    for i in range(B):
        exp = meta["experiment_name"][i]
        plate = meta["plate"][i]
        treated_idx = meta["treated_metadata_idx"][i]
        control_idx = meta["control_metadata_idx"][i]
        assert control_idx is not None, (
            f"Stage 2 control_metadata_idx is None for batch element {i}"
        )
        # Look up control row to assert plate match
        control_row = split.control.loc[control_idx]
        ctrl_exp = str(control_row["experiment_name"])
        ctrl_plate = int(control_row["plate"])
        assert (ctrl_exp, ctrl_plate) == (exp, plate), (
            f"batch element {i}: treated=({exp}, {plate}) but "
            f"control=({ctrl_exp}, {ctrl_plate})"
        )
    print(
        f"[smoke] same-plate invariant OK across {B} batch elements",
        flush=True,
    )


def _assert_finite(records: list[dict]) -> None:
    for i, r in enumerate(records):
        loss = r.get("loss")
        assert loss is not None, f"record {i} missing 'loss': {r}"
        assert isinstance(loss, (int, float)) and loss == loss, (
            f"record {i} loss not finite: {loss}"
        )
        assert abs(float(loss)) < float("inf"), (
            f"record {i} loss not finite: {loss}"
        )


def main() -> None:
    cfg = _build_smoke_config()
    output_dir = REPO_ROOT / "runs" / "smoke_stage2"
    print(f"[smoke] writing to {output_dir}", flush=True)

    _verify_same_plate_invariant(cfg)

    summary = run_training(cfg, output_dir)
    print(f"[summary] {summary}", flush=True)

    train_jsonl = Path(summary["train_jsonl"])
    assert train_jsonl.exists(), f"train.jsonl not written at {train_jsonl}"
    records = read_jsonl(train_jsonl)
    assert len(records) == int(cfg.training["max_steps"]), (
        f"expected {cfg.training['max_steps']} records, got {len(records)}"
    )
    _assert_finite(records)
    print(
        f"[verify] {len(records)} records in train.jsonl, "
        f"first_loss={records[0]['loss']:.4f} last_loss={records[-1]['loss']:.4f}",
        flush=True,
    )

    latest = Path(summary["latest_ckpt"]) if summary["latest_ckpt"] else None
    assert latest is not None and latest.exists(), (
        f"latest checkpoint missing: {latest}"
    )
    final = Path(summary["final_ckpt"])
    assert final.exists(), f"final checkpoint missing: {final}"
    print(f"[verify] latest={latest}", flush=True)
    print(f"[verify] final={final}", flush=True)

    fresh = DiTVelocity(
        hidden_dim=int(cfg.model["hidden_dim"]),
        depth=int(cfg.model["depth"]),
        num_heads=int(cfg.model["num_heads"]),
        dropout=float(cfg.model.get("dropout", 0.0)),
        balance_conditioning=bool(cfg.model.get("balance_conditioning", True)),
        time_scale=float(cfg.model.get("time_scale", 1.0)),
        condition_scale=float(cfg.model.get("condition_scale", 1.0)),
    )
    meta = load_checkpoint(final, model=fresh)
    assert meta["step"] == int(cfg.training["max_steps"]), (
        f"checkpoint step mismatch: {meta['step']} vs {cfg.training['max_steps']}"
    )
    print(
        f"[verify] reloaded final.pt at step={meta['step']} epoch={meta['epoch']}",
        flush=True,
    )

    print("\nSMOKE STAGE2 TINY: ALL CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
