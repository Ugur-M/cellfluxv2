"""Per-channel z-score normalization for ``(169, 8)`` latents.

The stats file is a ``.pt`` dict with keys ``mean`` and ``std``, each a
``(8,)`` float tensor. ``normalize`` and ``denormalize`` accept either a
single latent of shape ``(169, 8)`` or a batch of shape ``(B, 169, 8)``
and broadcast ``mean`` / ``std`` over the spatial dimension.
"""
from __future__ import annotations

from pathlib import Path

import torch

EXPECTED_CHANNELS = 8
EXPECTED_TOKENS = 169


def load_norm_stats(pt_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load per-channel ``(mean, std)`` stats from a ``.pt`` file.

    Returns
    -------
    (mean, std) : tuple of torch.Tensor
        Both shape ``(8,)``, dtype ``float32``, on CPU.

    Raises
    ------
    FileNotFoundError
        If ``pt_path`` does not exist.
    ValueError
        If the file is not a dict, is missing keys, has wrong shape,
        contains non-finite values, or has any non-positive std.
    """
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"norm stats not found: {pt_path}")

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(
            f"expected dict at {pt_path}, got {type(obj).__name__}"
        )
    missing = {"mean", "std"} - set(obj.keys())
    if missing:
        raise ValueError(
            f"norm stats at {pt_path} missing keys {sorted(missing)}; "
            f"found: {list(obj.keys())}"
        )

    mean = obj["mean"].detach().to(torch.float32).cpu()
    std = obj["std"].detach().to(torch.float32).cpu()

    if mean.shape != (EXPECTED_CHANNELS,):
        raise ValueError(
            f"mean shape {tuple(mean.shape)} must be ({EXPECTED_CHANNELS},)"
        )
    if std.shape != (EXPECTED_CHANNELS,):
        raise ValueError(
            f"std shape {tuple(std.shape)} must be ({EXPECTED_CHANNELS},)"
        )
    if not torch.isfinite(mean).all():
        raise ValueError("mean contains non-finite values")
    if not torch.isfinite(std).all():
        raise ValueError("std contains non-finite values")
    if not (std > 0).all():
        raise ValueError(
            f"std must be strictly positive; got {std.tolist()}"
        )
    return mean, std


def _check_latent_shape(z: torch.Tensor) -> None:
    if z.ndim == 2:
        if tuple(z.shape) != (EXPECTED_TOKENS, EXPECTED_CHANNELS):
            raise ValueError(
                f"expected latent shape (169, 8); got {tuple(z.shape)}"
            )
        return
    if z.ndim == 3:
        if tuple(z.shape[-2:]) != (EXPECTED_TOKENS, EXPECTED_CHANNELS):
            raise ValueError(
                f"expected latent shape (B, 169, 8); got {tuple(z.shape)}"
            )
        return
    raise ValueError(
        f"latent must be 2D (169, 8) or 3D (B, 169, 8); got {z.ndim}D"
    )


def normalize(
    z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Apply per-channel z-score: ``(z - mean) / std``.

    Accepts ``z`` of shape ``(169, 8)`` or ``(B, 169, 8)``;
    ``mean`` and ``std`` must be shape ``(8,)``.
    """
    _check_latent_shape(z)
    return (z - mean) / std


def denormalize(
    z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`normalize`: ``z * std + mean``."""
    _check_latent_shape(z)
    return z * std + mean
