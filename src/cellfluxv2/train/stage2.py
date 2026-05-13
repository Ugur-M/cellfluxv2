"""Stage-2 training entry-point: same-plate control latent -> normalized treated latent.

Run:

    python -m cellfluxv2.train.stage2 --config configs/stage2.yaml

CLI overrides:

    --max-steps INT
    --batch-size INT
    --num-workers INT
    --device STR        (e.g. "cuda", "cpu")
    --output-dir PATH
    --init-ckpt PATH    (model-only init, optimizer NOT loaded)

Behaviour:

* Loads the YAML config; CLI flags override the corresponding fields.
* Loads metadata; applies :func:`filter_split_by_missing_addresses` when
  the CSV exists (the dataset itself stays strict).
* Builds a real :func:`build_pair_index` on the **filtered** split — no
  empty-PairIndex shortcut. If any treated ``(experiment_name, plate)``
  group lost all its controls, ``build_pair_index`` raises loudly; that
  is intentional. There is no silent skip.
* Builds ``CellFluxDataset(stage=2)`` so ``__getitem__`` samples the
  source latent from the same-plate control pool via the pair index.
* Optionally warm-starts model weights from ``init_ckpt``. Optimizer
  state is **not** loaded; AdamW always starts fresh for the Stage 2
  objective.
* Same source-noise injection (``source_noise_p / source_noise_sigma``),
  same noisy interpolant, same training loop and logging shape as
  Stage 1, plus diagnostics on a fixed held-out batch every
  ``diagnostic_interval`` steps.

This module is import-safe — :func:`main` is only invoked from
``__main__``. The smoke script imports the smaller helpers directly.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from torch.utils.data import DataLoader

from ..data.dataset import CellFluxDataset
from ..data.fingerprints import load_fp_cache
from ..data.latent_norm import load_norm_stats
from ..data.metadata import (
    MetadataSplit,
    filter_split_by_missing_addresses,
    load_metadata,
)
from ..data.pair_index import PairIndex, build_pair_index
from ..data.plate_cache import PlateCache
from ..data.plate_sampler import PlateGroupedSampler
from ..models.dit import DiTVelocity
from ..train.checkpoint import load_checkpoint, save_checkpoint
from ..train.diagnostics import diagnostic_suite
from ..train.loop import train_step
from ..utils.logging import append_jsonl, format_metrics
from ..utils.seed import seed_everything
from ..utils.wandb_run import WandbRun, flatten_for_wandb


# ---------- config dataclass ------------------------------------------------


@dataclass
class Stage2Config:
    """Parsed + normalized Stage-2 config."""

    seed: int
    stage: int
    model: dict[str, Any]
    training: dict[str, Any]
    data: dict[str, Any]
    wandb: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Stage2Config":
        for k in ("seed", "stage", "model", "training", "data"):
            if k not in raw:
                raise ValueError(f"config missing key {k!r}; have {sorted(raw.keys())}")
        if int(raw["stage"]) != 2:
            raise ValueError(
                f"stage2.py only supports stage=2; got stage={raw['stage']}"
            )
        return cls(
            seed=int(raw["seed"]),
            stage=int(raw["stage"]),
            model=dict(raw["model"]),
            training=dict(raw["training"]),
            data=dict(raw["data"]),
            wandb=dict(raw.get("wandb") or {}),
        )

    def to_serializable(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "stage": self.stage,
            "model": dict(self.model),
            "training": dict(self.training),
            "data": dict(self.data),
            "wandb": dict(self.wandb),
        }


def load_config(path: str | Path) -> Stage2Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"config at {path} must be a YAML mapping; got {type(raw).__name__}"
        )
    return Stage2Config.from_dict(raw)


def apply_overrides(cfg: Stage2Config, overrides: dict[str, Any]) -> Stage2Config:
    """Merge a small set of CLI overrides into the parsed config."""
    if overrides.get("max_steps") is not None:
        cfg.training["max_steps"] = int(overrides["max_steps"])
    if overrides.get("batch_size") is not None:
        cfg.training["batch_size"] = int(overrides["batch_size"])
    if overrides.get("num_workers") is not None:
        cfg.training["num_workers"] = int(overrides["num_workers"])
    if overrides.get("device") is not None:
        cfg.training["device"] = str(overrides["device"])
    if overrides.get("output_dir") is not None:
        cfg.training["output_dir"] = str(overrides["output_dir"])
    if overrides.get("init_ckpt") is not None:
        cfg.training["init_ckpt"] = str(overrides["init_ckpt"])
    if overrides.get("wandb_disabled") is True:
        cfg.wandb["enabled"] = False
    if overrides.get("wandb_project") is not None:
        cfg.wandb["project"] = str(overrides["wandb_project"])
    if overrides.get("wandb_run_name") is not None:
        cfg.wandb["run_name"] = str(overrides["wandb_run_name"])
    if overrides.get("wandb_mode") is not None:
        cfg.wandb["mode"] = str(overrides["wandb_mode"])
    return cfg


# ---------- pipeline builders ----------------------------------------------


def build_split(cfg: Stage2Config) -> tuple[MetadataSplit, Optional[dict[str, Any]]]:
    """Load metadata; optionally drop rows via the missing-addresses CSV.

    Returns ``(split, filter_report or None)``.
    """
    metadata_path = Path(cfg.data["metadata_path"])
    split = load_metadata(metadata_path)
    missing_path = cfg.data.get("missing_addresses_path")
    if not missing_path:
        return split, None
    missing_path = Path(missing_path)
    if not missing_path.exists():
        return split, None
    filtered_split, report = filter_split_by_missing_addresses(split, missing_path)
    return filtered_split, report


def build_pair_index_from_split(split: MetadataSplit) -> PairIndex:
    """Build a strict same-(experiment_name, plate) pair index on the split.

    No empty-PairIndex shortcut. If any treated group has lost all its
    controls after missing-address filtering, ``build_pair_index`` raises
    — Stage 2 cannot run on a group with no control source.
    """
    return build_pair_index(split.treated, split.control)


def build_dataset(
    cfg: Stage2Config, split: MetadataSplit, pair_index: PairIndex
) -> CellFluxDataset:
    """Build the CellFluxDataset(stage=2) on the filtered split + real pair index."""
    latent_root = Path(cfg.data["latent_root"])
    plate_cache = PlateCache(
        latent_root, max_plates=int(cfg.data.get("plate_cache_max_plates", 8))
    )
    fp_cache = load_fp_cache(cfg.data["fingerprints_path"])
    mean, std = load_norm_stats(cfg.data["norm_stats_path"])
    return CellFluxDataset(
        metadata_split=split,
        pair_index=pair_index,
        plate_cache=plate_cache,
        fp_cache=fp_cache,
        mean=mean,
        std=std,
        stage=2,
        rng_seed=int(cfg.data.get("rng_seed", 1337)),
    )


def build_model(cfg: Stage2Config) -> DiTVelocity:
    m = cfg.model
    return DiTVelocity(
        hidden_dim=int(m["hidden_dim"]),
        depth=int(m["depth"]),
        num_heads=int(m["num_heads"]),
        dropout=float(m.get("dropout", 0.0)),
        balance_conditioning=bool(m.get("balance_conditioning", True)),
        time_scale=float(m.get("time_scale", 1.0)),
        condition_scale=float(m.get("condition_scale", 1.0)),
    )


def build_optimizer(model: DiTVelocity, cfg: Stage2Config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training["lr"]),
        weight_decay=float(cfg.training["weight_decay"]),
    )


def build_dataloader(
    dataset: CellFluxDataset, cfg: Stage2Config
) -> DataLoader:
    num_workers = int(cfg.training["num_workers"])
    device = str(cfg.training.get("device", "cpu"))
    treated = dataset.split.treated
    plate_keys = list(
        zip(treated["experiment_name"].tolist(), treated["plate"].tolist())
    )
    plate_to_positions: dict = {}
    for pos, key in enumerate(plate_keys):
        plate_to_positions.setdefault(key, []).append(pos)
    sampler = PlateGroupedSampler(
        plate_to_positions, seed=int(cfg.data.get("rng_seed", 1337))
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.training["batch_size"]),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=CellFluxDataset.collate,
        drop_last=True,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


# ---------- training loop --------------------------------------------------


def _device_str(cfg: Stage2Config) -> str:
    requested = str(cfg.training.get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"config requested device={requested!r} but torch.cuda.is_available() is False"
        )
    return requested


def _snapshot_diagnostic_batch(loader: DataLoader, device: str) -> dict[str, Any]:
    """Pull one batch and clone tensors to CPU so subsequent loader epochs don't reuse them."""
    batch = next(iter(loader))
    snap = {
        "x0": batch["x0"].detach().clone(),
        "x1": batch["x1"].detach().clone(),
        "condition": batch["condition"].detach().clone(),
    }
    if "meta" in batch:
        snap["meta"] = batch["meta"]
    return snap


def _maybe_load_init_ckpt(
    model: DiTVelocity, init_ckpt: Optional[str], device: str
) -> Optional[dict[str, Any]]:
    """Load model-only weights from ``init_ckpt`` if provided.

    Returns the loaded metadata dict (``{step, epoch, config, extra}``)
    when a checkpoint is loaded, or ``None`` when no init was requested.

    Optimizer state is intentionally NOT loaded — the Stage 2 optimizer
    must start fresh because the objective has changed.
    """
    if init_ckpt is None or str(init_ckpt).lower() in ("", "none", "null"):
        warnings.warn(
            "Stage 2 was launched without --init-ckpt / training.init_ckpt. "
            "Training from scratch is supported but usually you want to warm-start "
            "from a balanced Stage 1 final.pt.",
            stacklevel=2,
        )
        return None
    ckpt_path = Path(init_ckpt)
    meta = load_checkpoint(ckpt_path, model=model, map_location=device)
    print(
        f"[init] loaded model weights from {ckpt_path} "
        f"step={meta.get('step')} epoch={meta.get('epoch')}",
        flush=True,
    )
    return meta


def run_training(
    cfg: Stage2Config, output_dir: Path
) -> dict[str, Any]:
    """Run the Stage 2 training loop and return a summary dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(cfg.seed))

    split, filter_report = build_split(cfg)
    if filter_report is not None:
        print(f"[data] filtered split: {filter_report}", flush=True)

    pair_index = build_pair_index_from_split(split)
    print(
        f"[data] pair_index groups={len(pair_index.groups)} "
        f"treated_paired={len(pair_index.treated_to_group)}",
        flush=True,
    )
    if len(pair_index.groups) == 0:
        raise RuntimeError(
            "build_pair_index_from_split returned an empty PairIndex — "
            "Stage 2 cannot run without same-plate control pools"
        )

    dataset = build_dataset(cfg, split, pair_index)
    print(
        f"[data] dataset stage={dataset.stage} len(treated)={len(dataset)} "
        f"smiles_vocab={len(dataset.split.smiles_vocab)}",
        flush=True,
    )

    loader = build_dataloader(dataset, cfg)
    model = build_model(cfg)
    device = _device_str(cfg)
    model = model.to(device)
    init_meta = _maybe_load_init_ckpt(
        model, cfg.training.get("init_ckpt"), device
    )
    optimizer = build_optimizer(model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n_params:,} device={device}", flush=True)

    # Fixed held-out diagnostic batch — pulled once before any epoch
    # boundary, so it is not the same tensors any worker hands back.
    dataset.set_epoch(0)
    diag_batch = _snapshot_diagnostic_batch(loader, device)

    train_jsonl = output_dir / "train.jsonl"
    if train_jsonl.exists():
        train_jsonl.unlink()
    config_dump = output_dir / "config.json"
    config_dump.write_text(json.dumps(cfg.to_serializable(), indent=2, sort_keys=True))

    wandb_run = WandbRun(
        cfg.wandb, config=cfg.to_serializable(), output_dir=output_dir
    )
    if wandb_run.active:
        wandb_run.save(config_dump)

    max_steps = int(cfg.training["max_steps"])
    log_interval = int(cfg.training["log_interval"])
    diag_interval = int(cfg.training["diagnostic_interval"])
    ckpt_interval = int(cfg.training["checkpoint_interval"])
    path_type = str(cfg.training["path_type"])
    path_sigma = float(cfg.training["path_sigma"])
    source_noise_p = float(cfg.training["source_noise_p"])
    source_noise_sigma = float(cfg.training["source_noise_sigma"])
    cond_dropout_p = float(cfg.training["cond_dropout_p"])
    grad_clip_norm = float(cfg.training["grad_clip_norm"])
    lr = float(cfg.training["lr"])
    batch_size = int(cfg.training["batch_size"])

    losses: list[float] = []
    step = 0
    epoch = 0
    samples_seen = 0
    started = time.time()
    last_ckpt: Optional[Path] = None

    while step < max_steps:
        dataset.set_epoch(epoch)
        for batch in loader:
            if step >= max_steps:
                break
            metrics = train_step(
                model,
                batch,
                optimizer,
                device=device,
                path_type=path_type,
                path_sigma=path_sigma,
                source_noise_p=source_noise_p,
                source_noise_sigma=source_noise_sigma,
                cond_dropout_p=cond_dropout_p,
                grad_clip_norm=grad_clip_norm,
            )
            samples_seen += int(batch["x0"].shape[0])
            losses.append(metrics["loss"])

            record: dict[str, Any] = {
                "step": step,
                "epoch": epoch,
                "samples_seen": samples_seen,
                "lr": lr,
                **metrics,
            }

            if diag_interval > 0 and (step == 0 or (step + 1) % diag_interval == 0):
                diag = diagnostic_suite(
                    model, diag_batch, device=device,
                    path_type=path_type, path_sigma=path_sigma,
                )
                record["diagnostics"] = diag

            append_jsonl(train_jsonl, record)
            wandb_run.log(flatten_for_wandb(record), step=step)

            if log_interval > 0 and (step % log_interval == 0 or step == max_steps - 1):
                msg = format_metrics(
                    {
                        "loss": metrics["loss"],
                        "grad_norm": metrics["grad_norm"],
                        "v_pred_rms": metrics["v_pred_rms"],
                        "v_target_rms": metrics["v_target_rms"],
                        "t_mean": metrics["t_mean"],
                        "cond_drop_frac": metrics["cond_drop_frac"],
                        "source_noise_frac": metrics["source_noise_frac"],
                    },
                    prefix=f"[step {step}/{max_steps} epoch {epoch}]",
                )
                print(msg, flush=True)

            if ckpt_interval > 0 and (step + 1) % ckpt_interval == 0:
                latest = output_dir / "latest.pt"
                save_checkpoint(
                    latest,
                    model=model,
                    optimizer=optimizer,
                    step=step + 1,
                    epoch=epoch,
                    config=cfg.to_serializable(),
                    extra={
                        "samples_seen": samples_seen,
                        "init_ckpt": cfg.training.get("init_ckpt"),
                    },
                )
                last_ckpt = latest

            step += 1
        epoch += 1

    # Final checkpoint
    final = output_dir / "final.pt"
    save_checkpoint(
        final,
        model=model,
        optimizer=optimizer,
        step=step,
        epoch=epoch,
        config=cfg.to_serializable(),
        extra={
            "samples_seen": samples_seen,
            "init_ckpt": cfg.training.get("init_ckpt"),
        },
    )
    elapsed = time.time() - started
    print(
        f"[done] step={step} epoch={epoch} elapsed={elapsed:.1f}s "
        f"first_loss={losses[0]:.4f} last_loss={losses[-1]:.4f}",
        flush=True,
    )
    if wandb_run.active:
        wandb_run.log(
            {
                "summary/first_loss": float(losses[0]) if losses else 0.0,
                "summary/last_loss": float(losses[-1]) if losses else 0.0,
                "summary/elapsed_sec": float(elapsed),
                "summary/samples_seen": int(samples_seen),
            },
            step=step,
        )
        wandb_run.save(final)
    wandb_run.finish()

    return {
        "step": step,
        "epoch": epoch,
        "samples_seen": samples_seen,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "elapsed_sec": elapsed,
        "train_jsonl": str(train_jsonl),
        "latest_ckpt": str(last_ckpt) if last_ckpt is not None else None,
        "final_ckpt": str(final),
        "batch_size": batch_size,
        "output_dir": str(output_dir),
        "init_ckpt": cfg.training.get("init_ckpt"),
        "init_ckpt_meta": init_meta,
        "pair_index_groups": len(pair_index.groups),
    }


# ---------- CLI ------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CellFluxV2 Stage 2 trainer (same-plate control -> treated)."
    )
    p.add_argument("--config", required=True, help="Path to a Stage 2 YAML config.")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where train.jsonl + checkpoints go (default: runs/stage2).",
    )
    p.add_argument(
        "--init-ckpt",
        type=str,
        default=None,
        help=(
            "Path to a model-only init checkpoint (typically a balanced "
            "Stage 1 final.pt). Optimizer state is NOT loaded."
        ),
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging even if enabled in the YAML.",
    )
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    cfg = load_config(args.config)
    cfg = apply_overrides(
        cfg,
        {
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "output_dir": args.output_dir,
            "init_ckpt": args.init_ckpt,
            "wandb_disabled": args.no_wandb,
            "wandb_project": args.wandb_project,
            "wandb_run_name": args.wandb_run_name,
            "wandb_mode": args.wandb_mode,
        },
    )
    output_dir = Path(cfg.training.get("output_dir") or "runs/stage2")
    return run_training(cfg, output_dir)


if __name__ == "__main__":
    main()
