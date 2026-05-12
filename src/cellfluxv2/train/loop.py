"""Reusable training-step logic for CellFluxV2-style flow matching.

Stage-agnostic: this module assumes the dataset already returns
``{"x0", "x1", "condition", ...}`` per batch, and so the same
``train_step`` works for Stage 1 (where ``x0`` is Gaussian noise) and
Stage 2 (where ``x0`` is a same-plate control latent).

Three public helpers:

- ``move_batch_to_device`` — validates required keys + tensor shapes
  + finiteness, then moves ``x0`` / ``x1`` / ``condition`` to the
  requested device (and leaves ``meta`` alone).
- ``compute_flow_batch`` — samples ``t``, optionally applies source
  noise augmentation, and computes the interpolant ``x_t`` / target
  ``v_target`` via either the rectified or noisy path.
- ``train_step`` — one optimizer step: ``move_batch_to_device``,
  ``compute_flow_batch``, condition dropout, forward, MSE loss,
  backward, optional grad clip, optimizer step. Returns a dict of
  Python-float metrics.

No checkpointing, EMA, schedulers, or W&B integration here. No
``nan_to_num``. No silent repair.
"""
from __future__ import annotations

from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..flow.cfg import condition_dropout
from ..flow.path import noisy_path, rectified_path, source_noise_augmentation

LATENT_SHAPE: tuple[int, int] = (169, 8)
CONDITION_DIM: int = 1024
REQUIRED_BATCH_KEYS: tuple[str, ...] = ("x0", "x1", "condition")


# ----------------------------- move_batch_to_device -------------------------

def move_batch_to_device(
    batch: dict[str, Any], device: Union[torch.device, str]
) -> dict[str, Any]:
    """Validate the batch tensors and move ``x0`` / ``x1`` / ``condition``.

    ``meta`` (and any other non-required key) is passed through unchanged.
    Returns a new dict; the input is not mutated.
    """
    if not isinstance(batch, dict):
        raise ValueError(f"batch must be a dict; got {type(batch).__name__}")
    for key in REQUIRED_BATCH_KEYS:
        if key not in batch:
            raise ValueError(f"batch is missing required key {key!r}")

    x0 = batch["x0"]
    x1 = batch["x1"]
    condition = batch["condition"]

    for name, t in (("x0", x0), ("x1", x1), ("condition", condition)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"batch[{name!r}] must be a Tensor; got {type(t).__name__}")
        if not t.is_floating_point():
            raise ValueError(f"batch[{name!r}] must be floating; got dtype {t.dtype}")
        if not torch.isfinite(t).all():
            raise ValueError(f"batch[{name!r}] contains non-finite values")

    if x0.ndim != 3 or tuple(x0.shape[-2:]) != LATENT_SHAPE:
        raise ValueError(
            f"batch['x0'] shape {tuple(x0.shape)} must be (B, 169, 8)"
        )
    if x1.shape != x0.shape:
        raise ValueError(
            f"batch['x1'] shape {tuple(x1.shape)} must match x0 shape "
            f"{tuple(x0.shape)}"
        )
    B = x0.shape[0]
    if condition.shape != (B, CONDITION_DIM):
        raise ValueError(
            f"batch['condition'] shape {tuple(condition.shape)} must be "
            f"(B={B}, {CONDITION_DIM})"
        )

    out = dict(batch)
    out["x0"] = x0.to(device)
    out["x1"] = x1.to(device)
    out["condition"] = condition.to(device)
    return out


# ----------------------------- compute_flow_batch ---------------------------

def compute_flow_batch(
    batch: dict[str, Any],
    *,
    path_type: str = "noisy",
    path_sigma: float = 1.0,
    source_noise_p: float = 0.0,
    source_noise_sigma: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> dict[str, Any]:
    """Sample ``t``, optionally augment ``x0``, compute path tensors.

    Returns
    -------
    dict with keys ``{x_t, v_target, t, x0_aug, source_noise_mask, path_eps}``.
    ``path_eps`` is ``None`` for the rectified path.

    The source augmentation, when active, is applied to ``x0`` *before*
    the path computation, so ``v_target = (x1 - x0_aug) + …`` uses the
    augmented source.
    """
    x0 = batch["x0"]
    x1 = batch["x1"]
    B = x0.shape[0]
    device = x0.device
    dtype = x0.dtype

    if path_type not in ("rectified", "noisy"):
        raise ValueError(
            f"path_type must be 'rectified' or 'noisy'; got {path_type!r}"
        )

    # Sample t ~ U(0, 1) on x0's device + dtype.
    if generator is not None:
        t = torch.rand(B, device=device, dtype=dtype, generator=generator)
    else:
        t = torch.rand(B, device=device, dtype=dtype)

    # Source-noise augmentation (separate from the path noise).
    if source_noise_p > 0.0 or source_noise_sigma > 0.0:
        x0_aug, mask, _ = source_noise_augmentation(
            x0,
            p=float(source_noise_p),
            sigma=float(source_noise_sigma),
            generator=generator,
        )
    else:
        x0_aug = x0
        mask = torch.zeros(B, dtype=torch.bool, device=device)

    if path_type == "rectified":
        x_t, v_target = rectified_path(x0_aug, x1, t)
        path_eps: Optional[torch.Tensor] = None
    else:  # "noisy"
        x_t, v_target, path_eps = noisy_path(
            x0_aug, x1, t, sigma=float(path_sigma)
        )

    if not torch.isfinite(x_t).all():
        raise ValueError("x_t contains non-finite values")
    if not torch.isfinite(v_target).all():
        raise ValueError("v_target contains non-finite values")

    return {
        "x_t": x_t,
        "v_target": v_target,
        "t": t,
        "x0_aug": x0_aug,
        "source_noise_mask": mask,
        "path_eps": path_eps,
    }


# ----------------------------- train_step -----------------------------------

def _grad_norm_unclipped(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm().item()) ** 2
    return total ** 0.5


def train_step(
    model: nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: Union[torch.device, str],
    path_type: str = "noisy",
    path_sigma: float = 1.0,
    source_noise_p: float = 0.0,
    source_noise_sigma: float = 0.0,
    cond_dropout_p: float = 0.1,
    grad_clip_norm: Optional[float] = 1.0,
) -> dict[str, float]:
    """One MSE flow-matching training step. Returns Python-float metrics."""
    if not isinstance(model, nn.Module):
        raise ValueError(
            f"model must be an nn.Module; got {type(model).__name__}"
        )
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError(
            f"optimizer must be a torch.optim.Optimizer; got {type(optimizer).__name__}"
        )

    model.train()

    batch = move_batch_to_device(batch, device)
    flow = compute_flow_batch(
        batch,
        path_type=path_type,
        path_sigma=path_sigma,
        source_noise_p=source_noise_p,
        source_noise_sigma=source_noise_sigma,
    )
    x_t = flow["x_t"]
    v_target = flow["v_target"]
    t = flow["t"]

    # Condition dropout (per-sample replace with zero vector).
    condition_dropped, drop_mask = condition_dropout(
        batch["condition"], p=float(cond_dropout_p)
    )

    # Forward + loss.
    v_pred = model(x_t, t, condition_dropped)
    loss = F.mse_loss(v_pred, v_target)
    if not torch.isfinite(loss):
        raise ValueError(f"non-finite loss: {loss.item()}")

    # Backward, optional clip, step.
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if grad_clip_norm is not None:
        if grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be > 0 or None; got {grad_clip_norm}"
            )
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(grad_clip_norm)
            ).item()
        )
    else:
        grad_norm = _grad_norm_unclipped(model)

    optimizer.step()

    # Per-sample L2 norm of v_target, then mean (gives "average velocity magnitude").
    v_target_per_sample_norm = v_target.detach().flatten(1).norm(dim=-1)
    metrics: dict[str, float] = {
        "loss": float(loss.detach().item()),
        "grad_norm": float(grad_norm),
        "v_pred_rms": float(v_pred.detach().pow(2).mean().sqrt().item()),
        "v_target_rms": float(v_target.detach().pow(2).mean().sqrt().item()),
        "v_target_mean_norm": float(v_target_per_sample_norm.mean().item()),
        "t_mean": float(t.detach().mean().item()),
        "cond_drop_frac": float(drop_mask.detach().float().mean().item()),
        "source_noise_frac": float(
            flow["source_noise_mask"].detach().float().mean().item()
        ),
    }
    return metrics
