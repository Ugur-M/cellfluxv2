"""Time and condition embeddings for the DiT velocity model.

``SinusoidalTimeEmbed`` turns a scalar time ``t ∈ [0, 1]`` into a
``(B, dim)`` vector via classical sinusoidal/Fourier features followed
by a 2-layer SiLU MLP.

``ConditionEmbed`` projects a Morgan fingerprint (or any fixed-width
chemistry vector) to ``(B, out_dim)`` via a 2-layer SiLU MLP. Both
layers use the PyTorch default (Kaiming-uniform) initialization — the
near-zero initial output of the velocity model is delivered by the
zero-init adaLN-Zero gates in ``DiTBlock``, not by shrinking the
condition embedding to a tiny magnitude before it ever enters the
network.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbed(nn.Module):
    """``t -> (B, dim)`` via sinusoidal features + 2-layer SiLU MLP.

    Accepts ``t`` as a Python scalar (``int``/``float``), a 0-d Tensor,
    a 1-d ``(B,)`` Tensor, or a 2-d ``(B, 1)`` Tensor. Validates that
    ``t`` is finite and in ``[0, 1]``. Outputs ``(B, dim)``.

    Args
    ----
    dim : int
        Output feature width. Must be even (we split half cos / half sin).
    hidden_dim : int | None
        Intermediate MLP width. Defaults to ``dim * 4`` (DiT-style) if
        ``None``.
    max_period : float
        Spread of the Fourier basis. ``10000`` gives a wide span of
        frequencies; the MLP downstream picks what's useful.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        max_period: float = 10000.0,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even; got {dim}")
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        self.dim = dim
        self.max_period = float(max_period)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else dim * 4
        self.fc1 = nn.Linear(dim, self.hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(self.hidden_dim, dim)

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t.unsqueeze(-1) * freqs  # (B, half)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t) -> torch.Tensor:
        if isinstance(t, bool):
            raise ValueError("t must be a number, not bool")
        if isinstance(t, (int, float)):
            t = torch.tensor(
                float(t),
                device=self.fc1.weight.device,
                dtype=self.fc1.weight.dtype,
            )
        elif not isinstance(t, torch.Tensor):
            raise ValueError(
                f"t must be float, int, or Tensor; got {type(t).__name__}"
            )
        else:
            t = t.to(device=self.fc1.weight.device, dtype=self.fc1.weight.dtype)

        if t.ndim == 0:
            t = t.unsqueeze(0)
        elif t.ndim == 1:
            pass
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        else:
            raise ValueError(
                f"t must be scalar, (B,), or (B, 1); got shape {tuple(t.shape)}"
            )

        if not torch.isfinite(t).all():
            raise ValueError("t contains non-finite values")
        if (t < 0).any() or (t > 1).any():
            mn = float(t.min().item())
            mx = float(t.max().item())
            raise ValueError(
                f"t must be in [0, 1]; got min={mn:.4g}, max={mx:.4g}"
            )

        emb = self._sinusoidal(t)
        return self.fc2(self.act(self.fc1(emb)))


class ConditionEmbed(nn.Module):
    """``condition (B, in_dim) -> (B, out_dim)`` via 2-layer SiLU MLP.

    Used to lift Morgan fingerprints (``in_dim=1024``) to the model's
    hidden width. The fingerprint is treated as a generic float vector
    — it is **not** binarized, thresholded, or otherwise modified.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 512,
        out_dim: int = 384,
    ):
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError(
                f"dims must be positive; got in={in_dim} hidden={hidden_dim} out={out_dim}"
            )
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if not isinstance(condition, torch.Tensor):
            raise ValueError(
                f"condition must be a Tensor; got {type(condition).__name__}"
            )
        if not condition.is_floating_point():
            raise ValueError(
                f"condition must be a floating tensor; got dtype {condition.dtype}"
            )
        if condition.ndim != 2:
            raise ValueError(
                f"condition must be 2-d (B, {self.in_dim}); "
                f"got shape {tuple(condition.shape)}"
            )
        if condition.shape[1] != self.in_dim:
            raise ValueError(
                f"condition trailing dim {condition.shape[1]} != in_dim {self.in_dim}"
            )
        if not torch.isfinite(condition).all():
            raise ValueError("condition contains non-finite values")
        return self.fc2(self.act(self.fc1(condition)))
