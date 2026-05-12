"""Classifier-free guidance utilities.

Two pieces:

- ``condition_dropout`` — training-time helper that, per sample with
  probability ``p``, replaces the condition row with all zeros (the
  "null" condition the model can learn an unconditional density for).
  Returns the dropped condition and the boolean mask.

- ``apply_cfg`` — inference-time blend
  ``v_cfg = alpha * v_cond + (1 - alpha) * v_uncond``.
  ``alpha = 1`` is no guidance, ``alpha = 0`` is unconditional only,
  ``alpha > 1`` extrapolates further toward the conditional velocity.
"""
from __future__ import annotations

from typing import Optional

import torch


def _validate_condition(c: torch.Tensor) -> None:
    if not isinstance(c, torch.Tensor):
        raise ValueError(f"condition must be a Tensor; got {type(c).__name__}")
    if not c.is_floating_point():
        raise ValueError(f"condition must be floating; got dtype {c.dtype}")
    if c.ndim != 2:
        raise ValueError(
            f"condition must be 2-d (B, D); got shape {tuple(c.shape)}"
        )
    if not torch.isfinite(c).all():
        raise ValueError("condition contains non-finite values")


def condition_dropout(
    condition: torch.Tensor,
    p: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample drop a condition row to all-zeros with probability ``p``.

    Returns ``(dropped_condition, mask)`` with shapes ``(B, D)`` and
    ``(B,)``. The input tensor is **not** mutated; the dropped result
    is a fresh tensor.
    """
    _validate_condition(condition)
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise ValueError(f"p must be a number; got {type(p).__name__}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1]; got {p}")

    B = condition.shape[0]
    if p == 0.0:
        mask = torch.zeros(B, dtype=torch.bool, device=condition.device)
        return condition.clone(), mask

    if generator is not None:
        u = torch.rand(B, device=condition.device, generator=generator)
    else:
        u = torch.rand(B, device=condition.device)
    mask = u < p

    out = condition.clone()
    out[mask] = 0
    return out, mask


def apply_cfg(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Classifier-free guidance blend.

    ``v_cfg = alpha * v_cond + (1 - alpha) * v_uncond``.

    Conventions:
      - ``alpha = 1`` → ``v_cfg == v_cond`` (no guidance).
      - ``alpha = 0`` → ``v_cfg == v_uncond`` (unconditional only).
      - ``alpha > 1`` → extrapolation toward the conditional velocity.

    Raises if shapes mismatch, dtypes are non-floating, any value is
    non-finite, or ``alpha < 0``.
    """
    if not isinstance(v_cond, torch.Tensor):
        raise ValueError(f"v_cond must be a Tensor; got {type(v_cond).__name__}")
    if not isinstance(v_uncond, torch.Tensor):
        raise ValueError(
            f"v_uncond must be a Tensor; got {type(v_uncond).__name__}"
        )
    if not v_cond.is_floating_point():
        raise ValueError(f"v_cond must be floating; got dtype {v_cond.dtype}")
    if not v_uncond.is_floating_point():
        raise ValueError(f"v_uncond must be floating; got dtype {v_uncond.dtype}")
    if v_cond.shape != v_uncond.shape:
        raise ValueError(
            f"v_cond shape {tuple(v_cond.shape)} != "
            f"v_uncond shape {tuple(v_uncond.shape)}"
        )
    if not torch.isfinite(v_cond).all():
        raise ValueError("v_cond contains non-finite values")
    if not torch.isfinite(v_uncond).all():
        raise ValueError("v_uncond contains non-finite values")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be a number; got {type(alpha).__name__}")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0; got {alpha}")

    return alpha * v_cond + (1.0 - alpha) * v_uncond
