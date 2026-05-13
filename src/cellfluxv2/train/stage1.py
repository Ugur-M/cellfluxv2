"""Stage-1 training entry-point: Gaussian noise -> normalized treated latent.

Run:

    python -m cellfluxv2.train.stage1 --config configs/stage1.yaml

CLI overrides:

    --max-steps INT
    --batch-size INT
    --num-workers INT
    --device STR        (e.g. "cuda", "cpu")
    --output-dir PATH

Behaviour:

* Loads the YAML config; CLI flags override the corresponding fields.
* Loads metadata; if ``data.missing_addresses_path`` is set, drops rows
  via :func:`filter_split_by_missing_addresses` before building the
  dataset. The path must exist when set; missing-but-set raises
  ``FileNotFoundError``. The dataset itself stays strict.
* Stage 1 doesn't read control rows, so we don't run
  :func:`build_pair_index` here (it would fail loudly on any plate that
  lost all its controls after filtering, even though Stage 1 wouldn't
  use those controls). An empty ``PairIndex`` is passed to the dataset
  constructor — it'd only be touched on a stage==2 ``__getitem__``.
* Trains the DiT velocity model for ``training.max_steps`` train_steps,
  calling ``dataset.set_epoch(epoch)`` at every fresh DataLoader pass
  (this is what makes stage-1 Gaussian noise resample across epochs).
* Writes per-step metrics to ``<output_dir>/train.jsonl``.
* Saves a rolling ``latest.pt`` every ``checkpoint_interval`` steps,
  and a ``final.pt`` at the end.
* Runs the full latent-diagnostic suite on a fixed held-out batch every
  ``diagnostic_interval`` steps; the diagnostic values are folded into
  the matching train.jsonl record.

This module is import-safe — :func:`main` is only invoked from
``__main__``. The smoke script imports the smaller helpers directly.
"""
from __future__ import annotations

import argparse
import json
import time
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
from ..data.pair_index import PairIndex
from ..data.plate_cache import PlateCache
from ..data.plate_sampler import PlateGroupedSampler
from ..models.dit import DiTVelocity
from ..train.checkpoint import save_checkpoint
from ..train.diagnostics import diagnostic_suite
from ..train.loop import train_step
from ..utils.logging import append_jsonl, format_metrics
from ..utils.seed import seed_everything
from ..utils.wandb_run import WandbRun, flatten_for_wandb


# ---------- config dataclass ------------------------------------------------


@dataclass
class Stage1Config:
    """Parsed + normalized Stage-1 config."""

    seed: int
    stage: int
    model: dict[str, Any]
    training: dict[str, Any]
    data: dict[str, Any]
    wandb: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Stage1Config":
        for k in ("seed", "stage", "model", "training", "data"):
            if k not in raw:
                raise ValueError(f"config missing key {k!r}; have {sorted(raw.keys())}")
        if int(raw["stage"]) != 1:
            raise ValueError(
                f"stage1.py only supports stage=1; got stage={raw['stage']}"
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


def load_config(path: str | Path) -> Stage1Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config at {path} must be a YAML mapping; got {type(raw).__name__}")
    return Stage1Config.from_dict(raw)


def apply_overrides(cfg: Stage1Config, overrides: dict[str, Any]) -> Stage1Config:
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


def build_split(cfg: Stage1Config) -> tuple[MetadataSplit, Optional[dict[str, Any]]]:
    """Load metadata; optionally drop rows via the missing-addresses CSV.

    Returns ``(split, filter_report or None)``. ``filter_report`` is only
    non-None when the CSV is actually applied.
    """
    metadata_path = Path(cfg.data["metadata_path"])
    split = load_metadata(metadata_path)
    missing_path = cfg.data.get("missing_addresses_path")
    if not missing_path:
        return split, None
    # Strict: if the user names a missing-addresses CSV, the file must
    # exist. Silently skipping would hide a config typo and run the
    # trainer on unfiltered data.
    missing_path = Path(missing_path)
    if not missing_path.exists():
        raise FileNotFoundError(
            f"missing_addresses_path is set but file not found: {missing_path}"
        )
    filtered_split, report = filter_split_by_missing_addresses(split, missing_path)
    return filtered_split, report


def build_dataset(
    cfg: Stage1Config, split: MetadataSplit
) -> CellFluxDataset:
    """Build the CellFluxDataset(stage=1) plus its sibling caches.

    Stage 1 never reads controls, so we hand the dataset an empty
    ``PairIndex`` rather than running ``build_pair_index``; that
    function fails loudly on any (experiment, plate) that lost all its
    controls after filtering, which would be a false alarm for Stage 1.
    """
    latent_root = Path(cfg.data["latent_root"])
    plate_cache = PlateCache(
        latent_root, max_plates=int(cfg.data.get("plate_cache_max_plates", 8))
    )
    fp_cache = load_fp_cache(cfg.data["fingerprints_path"])
    mean, std = load_norm_stats(cfg.data["norm_stats_path"])
    empty_pair_index = PairIndex(groups={}, treated_to_group={})
    return CellFluxDataset(
        metadata_split=split,
        pair_index=empty_pair_index,
        plate_cache=plate_cache,
        fp_cache=fp_cache,
        mean=mean,
        std=std,
        stage=1,
        rng_seed=int(cfg.data.get("rng_seed", 0)),
    )


def build_model(cfg: Stage1Config) -> DiTVelocity:
    m = cfg.model
    # `balance_conditioning`, `time_scale`, `condition_scale` are passed
    # through raw so DiTVelocity's strict validation runs on the actual
    # YAML values. Coercing here (bool(...), float(...)) would silently
    # turn typos like "false" into True or True into 1.0.
    return DiTVelocity(
        hidden_dim=int(m["hidden_dim"]),
        depth=int(m["depth"]),
        num_heads=int(m["num_heads"]),
        dropout=float(m.get("dropout", 0.0)),
        balance_conditioning=m.get("balance_conditioning", True),
        time_scale=m.get("time_scale", 1.0),
        condition_scale=m.get("condition_scale", 1.0),
    )


def build_optimizer(model: DiTVelocity, cfg: Stage1Config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training["lr"]),
        weight_decay=float(cfg.training["weight_decay"]),
    )


# Diagnostic-vs-training seed split. The diagnostic snapshot uses
# ``rng_seed + DIAG_SEED_OFFSET`` so its plate/position draw is
# reproducible but does not share state with the training sampler.
DIAG_SEED_OFFSET = 1_000_001


def _plate_to_positions(treated) -> dict:
    """Build ``{(experiment_name, plate): [positions]}`` from a treated split.

    ``positions`` are dataset indices into ``dataset.split.treated`` (i.e.
    the ``.iloc`` positions, not ``metadata_idx``).
    """
    plate_keys = list(
        zip(treated["experiment_name"].tolist(), treated["plate"].tolist())
    )
    out: dict = {}
    for pos, key in enumerate(plate_keys):
        out.setdefault(key, []).append(pos)
    return out


def build_dataloader(
    dataset: CellFluxDataset, cfg: Stage1Config
) -> tuple[DataLoader, PlateGroupedSampler]:
    num_workers = int(cfg.training["num_workers"])
    device = str(cfg.training.get("device", "cpu"))
    plate_to_positions = _plate_to_positions(dataset.split.treated)
    sampler = PlateGroupedSampler(
        plate_to_positions, seed=int(cfg.data.get("rng_seed", 0))
    )
    loader = DataLoader(
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
    return loader, sampler


# ---------- training loop --------------------------------------------------


def _device_str(cfg: Stage1Config) -> str:
    requested = str(cfg.training.get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"config requested device={requested!r} but torch.cuda.is_available() is False"
        )
    return requested


def _snapshot_diagnostic_batch(
    dataset: CellFluxDataset, cfg: Stage1Config, batch_size: int
) -> dict[str, Any]:
    """Build a deterministic diagnostic batch with a separate sampler seed.

    Uses a fresh ``PlateGroupedSampler`` seeded at
    ``rng_seed + DIAG_SEED_OFFSET`` and reads ``dataset[i]`` directly
    (no DataLoader, no workers). The diagnostic batch is therefore a
    deterministic function of ``(rng_seed, dataset state)`` and does
    not share its draw with the training loader's sampler.
    """
    plate_to_positions = _plate_to_positions(dataset.split.treated)
    diag_sampler = PlateGroupedSampler(
        plate_to_positions,
        seed=int(cfg.data.get("rng_seed", 0)) + DIAG_SEED_OFFSET,
    )
    diag_sampler.set_epoch(0)
    positions: list[int] = []
    for p in diag_sampler:
        positions.append(p)
        if len(positions) >= batch_size:
            break
    if len(positions) < batch_size:
        raise ValueError(
            f"diagnostic batch needs {batch_size} positions but dataset only "
            f"yielded {len(positions)}"
        )
    items = [dataset[i] for i in positions]
    batch = CellFluxDataset.collate(items)
    snap = {
        "x0": batch["x0"].detach().clone(),
        "x1": batch["x1"].detach().clone(),
        "condition": batch["condition"].detach().clone(),
    }
    if "meta" in batch:
        snap["meta"] = batch["meta"]
    return snap


def run_training(
    cfg: Stage1Config, output_dir: Path
) -> dict[str, Any]:
    """Run the Stage 1 training loop and return a summary dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(cfg.seed))

    split, filter_report = build_split(cfg)
    if filter_report is not None:
        print(f"[data] filtered split: {filter_report}", flush=True)
    dataset = build_dataset(cfg, split)
    print(
        f"[data] dataset len(treated)={len(dataset)} "
        f"smiles_vocab={len(dataset.split.smiles_vocab)}",
        flush=True,
    )

    loader, sampler = build_dataloader(dataset, cfg)
    model = build_model(cfg)
    device = _device_str(cfg)
    model = model.to(device)
    optimizer = build_optimizer(model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n_params:,} device={device}", flush=True)

    # Diagnostic batch uses a separate sampler seed (rng_seed +
    # DIAG_SEED_OFFSET) so it does not share its draw with the training
    # loader. Determinism is from the seed + dataset state only.
    dataset.set_epoch(0)
    diag_batch = _snapshot_diagnostic_batch(
        dataset, cfg, batch_size=int(cfg.training["batch_size"])
    )

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
        sampler.set_epoch(epoch)
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

            # Diagnostics on a fixed batch every diag_interval steps.
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
                    extra={"samples_seen": samples_seen},
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
        extra={"samples_seen": samples_seen},
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
    }


# ---------- CLI ------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CellFluxV2 Stage 1 trainer (noise -> treated).")
    p.add_argument("--config", required=True, help="Path to a Stage 1 YAML config.")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where train.jsonl + checkpoints go (default: runs/stage1).",
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
            "wandb_disabled": args.no_wandb,
            "wandb_project": args.wandb_project,
            "wandb_run_name": args.wandb_run_name,
            "wandb_mode": args.wandb_mode,
        },
    )
    output_dir = Path(cfg.training.get("output_dir") or "runs/stage1")
    return run_training(cfg, output_dir)


if __name__ == "__main__":
    main()
