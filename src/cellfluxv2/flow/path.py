"""Pure-torch flow-matching path utilities.

Implements the rectified and noisy interpolant paths used by the
CellFluxV2-style reproduction:

  Rectified:    x_t = (1 - t) x0 + t x1
                v_target = x1 - x0
  Noisy:        x_t = (1 - t) x0 + t x1 + sin²(π t) · σ · ε
                v_target = (x1 - x0) + π sin(2π t) · σ · ε

``source_noise_augmentation`` is a *separate* utility that adds optional
Gaussian perturbations to ``x0`` *before* the interpolation path is
computed. It is intentionally not folded into ``noisy_path`` so that
the path noise and the source augmentation can be reasoned about and
configured independently.

All functions are pure torch: no numpy, no pandas, no data loading, no
model code, no global mutable state, no ``nan_to_num``, no silent
repair. Every shape or finiteness violation raises ``ValueError``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

LATENT_SHAPE: tuple[int, int] = (169, 8)


# ----------------------------- validation -----------------------------------

def _validate_latent(name: str, x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor; got {type(x).__name__}")
    if not x.is_floating_point():
        raise ValueError(
            f"{name} must be a floating-point tensor; got dtype {x.dtype}"
        )
    if x.ndim != 3:
        raise ValueError(
            f"{name} must be 3-d (B, 169, 8); got ndim={x.ndim}, "
            f"shape={tuple(x.shape)}"
        )
    if tuple(x.shape[-2:]) != LATENT_SHAPE:
        raise ValueError(
            f"{name} shape {tuple(x.shape)} must end in {LATENT_SHAPE}"
        )
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values")


def _validate_pair(x0: torch.Tensor, x1: torch.Tensor) -> None:
    _validate_latent("x0", x0)
    _validate_latent("x1", x1)
    if x0.shape != x1.shape:
        raise ValueError(
            f"x0 shape {tuple(x0.shape)} != x1 shape {tuple(x1.shape)}"
        )


def _validate_eps(eps: torch.Tensor, x0: torch.Tensor) -> None:
    if not isinstance(eps, torch.Tensor):
        raise ValueError(f"eps must be a torch.Tensor; got {type(eps).__name__}")
    if eps.shape != x0.shape:
        raise ValueError(
            f"eps shape {tuple(eps.shape)} != x0 shape {tuple(x0.shape)}"
        )
    if not torch.isfinite(eps).all():
        raise ValueError("eps contains non-finite values")


# ----------------------------- broadcast_t ----------------------------------

def broadcast_t(
    t: float | int | torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Broadcast a time value to ``(B, 1, 1)`` matching ``x``'s batch.

    Accepts:
      - Python scalar ``float`` / ``int``.
      - 0-d ``Tensor``.
      - 1-d ``Tensor`` of shape ``(B,)``.

    Moves to ``x.device`` and casts to ``x.dtype``. Raises if ``t`` is
    outside ``[0, 1]`` or contains non-finite values.
    """
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"x must be a torch.Tensor; got {type(x).__name__}")
    if x.ndim < 1:
        raise ValueError("x must have at least one (batch) dimension")
    B = x.shape[0]

    if isinstance(t, bool):
        raise ValueError("t must be a number, not bool")
    if isinstance(t, (int, float)):
        t_tensor = torch.tensor(float(t), device=x.device, dtype=x.dtype)
    elif isinstance(t, torch.Tensor):
        t_tensor = t.to(device=x.device, dtype=x.dtype)
    else:
        raise ValueError(
            f"t must be float, int, or Tensor; got {type(t).__name__}"
        )

    if not torch.isfinite(t_tensor).all():
        raise ValueError("t contains non-finite values")
    if (t_tensor < 0).any() or (t_tensor > 1).any():
        mn = float(t_tensor.min().item())
        mx = float(t_tensor.max().item())
        raise ValueError(
            f"t must be in [0, 1]; got min={mn:.4g}, max={mx:.4g}"
        )

    if t_tensor.ndim == 0:
        t_tensor = t_tensor.expand(B)
    elif t_tensor.ndim == 1:
        if t_tensor.shape[0] != B:
            raise ValueError(
                f"t shape {tuple(t_tensor.shape)} does not match batch size {B}"
            )
    else:
        raise ValueError(
            f"t must be scalar or 1-d (B,); got shape {tuple(t_tensor.shape)}"
        )
    return t_tensor.unsqueeze(-1).unsqueeze(-1)


# ----------------------------- rectified_path -------------------------------

def rectified_path(
    x0: torch.Tensor, x1: torch.Tensor, t: float | int | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard rectified-flow interpolant.

    Returns ``(x_t, v_target)`` with::

        x_t      = (1 - t) x0 + t x1
        v_target = x1 - x0
    """
    _validate_pair(x0, x1)
    t_b = broadcast_t(t, x0)
    x_t = (1 - t_b) * x0 + t_b * x1
    v_target = x1 - x0
    return x_t, v_target


# ----------------------------- noisy_path -----------------------------------

def noisy_path(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: float | int | torch.Tensor,
    eps: Optional[torch.Tensor] = None,
    sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Noisy interpolant path with derivative-matched velocity target.

    Returns ``(x_t, v_target, eps)`` with::

        x_t      = (1 - t) x0 + t x1 + sin²(π t) · σ · ε
        v_target = (x1 - x0) + π sin(2π t) · σ · ε

    The noise term vanishes at ``t = 0`` and ``t = 1`` (because
    ``sin(0) = sin(π) = 0``), so the path still passes exactly through
    ``x0`` and ``x1`` at the endpoints. With ``sigma=0`` this reduces
    exactly to ``rectified_path``.

    If ``eps`` is ``None`` it is sampled via ``torch.randn_like(x0)``.
    """
    _validate_pair(x0, x1)
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise ValueError(f"sigma must be a number; got {type(sigma).__name__}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0; got {sigma}")

    if eps is None:
        eps = torch.randn_like(x0)
    else:
        _validate_eps(eps, x0)

    t_b = broadcast_t(t, x0)
    pi = math.pi
    sin_sq = torch.sin(pi * t_b) ** 2
    deriv = pi * torch.sin(2 * pi * t_b)

    x_t = (1 - t_b) * x0 + t_b * x1 + sin_sq * sigma * eps
    v_target = (x1 - x0) + deriv * sigma * eps
    return x_t, v_target, eps


# ----------------------------- source_noise_augmentation --------------------

def source_noise_augmentation(
    x0: torch.Tensor,
    p: float,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optional Gaussian augmentation of ``x0`` *before* the path computation.

    Per-sample with probability ``p``, add ``sigma * ε`` to ``x0``.
    This is *separate* from the path noise inside :func:`noisy_path`;
    if both are desired, compose them (apply this first, then call
    ``noisy_path`` on the augmented ``x0``).

    Special cases (matches the spec; no silent surprises):
      - ``p == 0``: return ``x0`` unchanged, all-False mask, zero noise
        (no RNG draw at all).
      - ``sigma == 0``: return ``x0`` unchanged, *sampled* mask, zero
        noise (the mask draw still consumes RNG state).

    Returns ``(x0_aug, mask, noise)`` with shapes ``(B, 169, 8)``,
    ``(B,) bool``, ``(B, 169, 8)``.
    """
    _validate_latent("x0", x0)
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise ValueError(f"p must be a number; got {type(p).__name__}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1]; got {p}")
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise ValueError(f"sigma must be a number; got {type(sigma).__name__}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0; got {sigma}")

    B = x0.shape[0]

    if p == 0:
        mask = torch.zeros(B, dtype=torch.bool, device=x0.device)
        noise = torch.zeros_like(x0)
        return x0, mask, noise

    if generator is not None:
        u = torch.rand(B, device=x0.device, generator=generator)
    else:
        u = torch.rand(B, device=x0.device)
    mask = u < p

    if sigma == 0:
        noise = torch.zeros_like(x0)
        return x0, mask, noise

    if generator is not None:
        noise = torch.randn(
            x0.shape, dtype=x0.dtype, device=x0.device, generator=generator
        )
    else:
        noise = torch.randn_like(x0)
    x0_aug = x0 + mask.view(B, 1, 1).to(x0.dtype) * sigma * noise
    return x0_aug, mask, noise
