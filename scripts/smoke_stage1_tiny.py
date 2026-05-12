"""Tiny end-to-end Stage 1 smoke on the real rxrx3 data.

Runs 5 train_steps with a 4-head, 128-hidden, 2-block DiT against
``configs/stage1.yaml``. Asserts:

* ``train.jsonl`` exists and contains 5 records.
* All recorded losses are finite.
* A rolling ``latest.pt`` checkpoint was written.
* A ``final.pt`` checkpoint was written and reloads into a fresh model.

The smoke uses absolute paths derived from the studio root so it is
runnable from any cwd (``python scripts/smoke_stage1_tiny.py`` from
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
from cellfluxv2.train.stage1 import (  # noqa: E402
    Stage1Config,
    load_config,
    run_training,
)
from cellfluxv2.utils.logging import read_jsonl  # noqa: E402


def _build_smoke_config() -> Stage1Config:
    """Load the canonical Stage 1 config and shrink it for a 5-step smoke."""
    cfg = load_config(REPO_ROOT / "configs" / "stage1.yaml")

    # Resolve data paths to absolutes so the smoke is cwd-independent.
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
    # latent_root stays absolute as authored.

    # Tiny model + tiny step count.
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
    return cfg


def _assert_finite(records: list[dict]) -> None:
    for i, r in enumerate(records):
        loss = r.get("loss")
        assert loss is not None, f"record {i} missing 'loss': {r}"
        assert isinstance(loss, (int, float)) and loss == loss, (  # NaN check
            f"record {i} loss not finite: {loss}"
        )
        assert abs(float(loss)) < float("inf"), (
            f"record {i} loss not finite: {loss}"
        )


def main() -> None:
    cfg = _build_smoke_config()
    output_dir = REPO_ROOT / "runs" / "smoke_stage1"
    print(f"[smoke] writing to {output_dir}", flush=True)

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
    )
    meta = load_checkpoint(final, model=fresh)
    assert meta["step"] == int(cfg.training["max_steps"]), (
        f"checkpoint step mismatch: {meta['step']} vs {cfg.training['max_steps']}"
    )
    print(f"[verify] reloaded final.pt at step={meta['step']} epoch={meta['epoch']}", flush=True)

    print("\nSMOKE STAGE1 TINY: ALL CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
