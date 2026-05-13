"""Latent-space diagnostics for the flow-matching velocity model.

All diagnostics:
  - Run under ``torch.no_grad()`` and call ``model.eval()`` so dropout
    / etc. don't perturb measurements.
  - Do **not** call ``backward`` or ``optimizer.step``.
  - Do **not** mutate model parameters or accumulate gradients.
  - Build their own internal ``(x_t, t, v_target)`` via
    ``compute_flow_batch``; the same triple is reused across the
    "real vs shuffled condition" probes so the comparison isn't
    polluted by a different noise sample.

Grad-norm diagnostics are intentionally deferred to Step 9.
"""
from __future__ import annotations

from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loop import compute_flow_batch, move_batch_to_device


def rms(x: torch.Tensor) -> torch.Tensor:
    """Root-mean-square of all elements of ``x``."""
    return x.pow(2).mean().sqrt()


def _prepare_for_diagnostics(
    model: nn.Module, batch: dict[str, Any], device: Union[torch.device, str]
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Common setup: move batch, eval model, build shared flow batch."""
    if not isinstance(model, nn.Module):
        raise ValueError(
            f"model must be an nn.Module; got {type(model).__name__}"
        )
    batch_dev = move_batch_to_device(batch, device)
    B = batch_dev["x0"].shape[0]
    model.eval()
    return batch_dev, B, model  # second return unused but keeps order obvious


# ---- drug_swap_v_cos -------------------------------------------------------

@torch.no_grad()
def drug_swap_v_cos(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: Union[torch.device, str],
    path_type: str = "noisy",
    path_sigma: float = 1.0,
) -> float:
    """Mean per-sample cosine similarity of ``v(x_t, t, c)`` vs
    ``v(x_t, t, c_shuffled)``.

    A model that uses its conditioning will produce different velocities
    for different conditions, dropping the cosine below 1. A model that
    ignores conditioning produces identical velocities — cosine = 1.

    Raises ``ValueError`` if batch size < 2 (nothing to shuffle).
    """
    if not isinstance(model, nn.Module):
        raise ValueError(f"model must be an nn.Module; got {type(model).__name__}")
    batch_dev = move_batch_to_device(batch, device)
    B = int(batch_dev["x0"].shape[0])
    if B < 2:
        raise ValueError(f"batch size must be >= 2 to shuffle; got {B}")

    model.eval()
    flow = compute_flow_batch(batch_dev, path_type=path_type, path_sigma=path_sigma)
    x_t = flow["x_t"]
    t = flow["t"]
    cond = batch_dev["condition"]

    perm = torch.randperm(B, device=cond.device)
    cond_shuf = cond[perm]

    v_real = model(x_t, t, cond)
    v_shuf = model(x_t, t, cond_shuf)

    cos = F.cosine_similarity(
        v_real.flatten(1), v_shuf.flatten(1), dim=-1
    )
    return float(cos.mean().item())


# ---- mse_real_vs_shuffled --------------------------------------------------

@torch.no_grad()
def mse_real_vs_shuffled(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: Union[torch.device, str],
    path_type: str = "noisy",
    path_sigma: float = 1.0,
) -> dict[str, float]:
    """MSE of ``v_pred`` vs ``v_target`` under the real and shuffled
    conditions, reusing the same ``(x_t, t, v_target)`` for both.

    A positive ``mse_gap_shuffled_minus_real`` means the real condition
    fits the velocity target better than a shuffled one (the model is
    using its condition). A gap near zero means the model is ignoring
    conditioning.
    """
    if not isinstance(model, nn.Module):
        raise ValueError(f"model must be an nn.Module; got {type(model).__name__}")
    batch_dev = move_batch_to_device(batch, device)
    B = int(batch_dev["x0"].shape[0])
    if B < 2:
        raise ValueError(f"batch size must be >= 2 to shuffle; got {B}")

    model.eval()
    flow = compute_flow_batch(batch_dev, path_type=path_type, path_sigma=path_sigma)
    x_t = flow["x_t"]
    t = flow["t"]
    v_target = flow["v_target"]
    cond = batch_dev["condition"]

    perm = torch.randperm(B, device=cond.device)
    cond_shuf = cond[perm]

    v_real = model(x_t, t, cond)
    v_shuf = model(x_t, t, cond_shuf)

    mse_real = float(F.mse_loss(v_real, v_target).item())
    mse_shuf = float(F.mse_loss(v_shuf, v_target).item())
    return {
        "mse_real_condition": mse_real,
        "mse_shuffled_condition": mse_shuf,
        "mse_gap_shuffled_minus_real": mse_shuf - mse_real,
    }


# ---- embedding_rms ---------------------------------------------------------

@torch.no_grad()
def embedding_rms(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: Union[torch.device, str],
) -> dict[str, float]:
    """RMS of the model's internal time and condition embeddings.

    If the model exposes ``get_conditioning_embeddings``, also report the
    post-balancing ("used") and combined RMS so we can tell whether the
    conditioning channel is starved before or after RMSNorm. The legacy
    keys ``condition_embedding_rms`` / ``time_embedding_rms`` remain and
    mirror the *used* values when the balanced path is available, falling
    back to the raw values for models that do not expose the method.
    """
    if not isinstance(model, nn.Module):
        raise ValueError(f"model must be an nn.Module; got {type(model).__name__}")
    if not hasattr(model, "cond_embed"):
        raise ValueError("model is missing the `cond_embed` attribute")
    if not hasattr(model, "time_embed"):
        raise ValueError("model is missing the `time_embed` attribute")

    batch_dev = move_batch_to_device(batch, device)
    cond = batch_dev["condition"]
    B = int(cond.shape[0])

    model.eval()
    t = torch.rand(B, device=cond.device, dtype=cond.dtype)

    if hasattr(model, "get_conditioning_embeddings"):
        bundle = model.get_conditioning_embeddings(t, cond)
        time_raw_rms = float(rms(bundle["time_raw"]).item())
        cond_raw_rms = float(rms(bundle["condition_raw"]).item())
        time_used_rms = float(rms(bundle["time_used"]).item())
        cond_used_rms = float(rms(bundle["condition_used"]).item())
        combined_rms = float(rms(bundle["combined"]).item())
        return {
            "time_embedding_raw_rms": time_raw_rms,
            "condition_embedding_raw_rms": cond_raw_rms,
            "time_embedding_used_rms": time_used_rms,
            "condition_embedding_used_rms": cond_used_rms,
            "combined_conditioning_rms": combined_rms,
            # Backwards-compatible legacy keys mirror the post-balance RMS.
            "time_embedding_rms": time_used_rms,
            "condition_embedding_rms": cond_used_rms,
        }

    # Fallback for models that do not expose get_conditioning_embeddings.
    cond_emb = model.cond_embed(cond)
    time_emb = model.time_embed(t)
    raw_t_rms = float(rms(time_emb).item())
    raw_c_rms = float(rms(cond_emb).item())
    return {
        "time_embedding_raw_rms": raw_t_rms,
        "condition_embedding_raw_rms": raw_c_rms,
        "time_embedding_used_rms": raw_t_rms,
        "condition_embedding_used_rms": raw_c_rms,
        "combined_conditioning_rms": float(rms(time_emb + cond_emb).item()),
        "time_embedding_rms": raw_t_rms,
        "condition_embedding_rms": raw_c_rms,
    }


# ---- diagnostic_suite ------------------------------------------------------

def diagnostic_suite(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: Union[torch.device, str],
    path_type: str = "noisy",
    path_sigma: float = 1.0,
) -> dict[str, float]:
    """Run drug-swap cosine, real-vs-shuffled MSE, and embedding RMS."""
    out: dict[str, float] = {}
    out["drug_swap_v_cos"] = drug_swap_v_cos(
        model, batch, device=device, path_type=path_type, path_sigma=path_sigma
    )
    out.update(
        mse_real_vs_shuffled(
            model,
            batch,
            device=device,
            path_type=path_type,
            path_sigma=path_sigma,
        )
    )
    out.update(embedding_rms(model, batch, device=device))
    return out
