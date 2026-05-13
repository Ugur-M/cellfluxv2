"""Simple DiT velocity model for CellFluxV2-style flow matching.

Predicts ``v_pred : (B, 169, 8)`` from ``x_t : (B, 169, 8)``, a
per-sample (or scalar) ``t``, and a Morgan-fingerprint condition
``(B, 1024)``.

Architecture (small, paper-faithful, no MMDiT / no cross-attention):

  - ``Linear(8 -> hidden)`` patch / token embedding.
  - Learned positional embedding ``(1, 169, hidden)``, small std init.
  - Sinusoidal ``t`` embedding -> MLP -> ``hidden``.
  - Morgan FP ``(B, 1024)`` -> MLP -> ``hidden``.
  - Optional non-learnable RMSNorm on each side, then a unit-RMS
    weighted sum into the adaLN conditioning vector ``c``.
  - ``depth`` DiT blocks: self-attention over 169 tokens + MLP,
    both adaLN-Zero conditioned on ``c``.
  - Final ``LayerNorm`` + adaLN + ``Linear(hidden -> 8)`` projection.

Block-level adaLN modulation is zero-initialized so each block starts
as identity. The final modulation and projection use small-std init so
the model initial output is near zero **and** gradients still flow
through every learnable parameter on the first backward.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .embeddings import ConditionEmbed, SinusoidalTimeEmbed

LATENT_TOKENS = 169
LATENT_DIM = 8
CONDITION_DIM = 1024


def modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """adaLN modulation: ``x * (1 + scale) + shift``.

    ``x`` has shape ``(B, N, D)``; ``shift`` and ``scale`` have shape
    ``(B, D)`` — they're unsqueezed along the token axis for broadcasting.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """One adaLN-Zero conditioned self-attention + MLP block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        # 6 chunks: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        # Zero-init the final adaLN linear so the block starts as identity:
        # gates = 0 → both residuals contribute zero on the first forward.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, hidden_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        # Self-attention path
        n1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(n1, n1, n1, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        # MLP path
        n2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(n2)
        return x


class DiTVelocity(nn.Module):
    """Simple DiT velocity model.

    Forward:
        ``v_pred = model(x, t, condition)``
        with shapes
        ``x: (B, 169, 8)``,
        ``t: scalar | (B,) | (B, 1)``,
        ``condition: (B, 1024)``,
        ``v_pred: (B, 169, 8)``.

    All shape / dtype / finiteness violations raise ``ValueError``.
    """

    def __init__(
        self,
        latent_tokens: int = LATENT_TOKENS,
        latent_dim: int = LATENT_DIM,
        condition_dim: int = CONDITION_DIM,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        balance_conditioning: bool = True,
        time_scale: float = 1.0,
        condition_scale: float = 1.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")
        if latent_tokens < 1 or latent_dim < 1 or condition_dim < 1:
            raise ValueError(
                f"latent_tokens={latent_tokens}, latent_dim={latent_dim}, "
                f"condition_dim={condition_dim} must all be positive"
            )
        if not isinstance(balance_conditioning, bool):
            raise ValueError(
                f"balance_conditioning must be bool; got {type(balance_conditioning).__name__}"
            )
        if not isinstance(time_scale, (int, float)) or isinstance(time_scale, bool):
            raise ValueError(
                f"time_scale must be a number; got {type(time_scale).__name__}"
            )
        if not isinstance(condition_scale, (int, float)) or isinstance(
            condition_scale, bool
        ):
            raise ValueError(
                f"condition_scale must be a number; got {type(condition_scale).__name__}"
            )
        time_scale = float(time_scale)
        condition_scale = float(condition_scale)
        if time_scale < 0:
            raise ValueError(f"time_scale must be >= 0; got {time_scale}")
        if condition_scale < 0:
            raise ValueError(f"condition_scale must be >= 0; got {condition_scale}")
        if time_scale == 0 and condition_scale == 0:
            raise ValueError("time_scale and condition_scale cannot both be zero")

        self.latent_tokens = int(latent_tokens)
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.balance_conditioning = bool(balance_conditioning)
        self.time_scale = time_scale
        self.condition_scale = condition_scale

        # Token + positional embedding
        self.x_embed = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.empty(1, latent_tokens, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Time and condition embeddings
        self.time_embed = SinusoidalTimeEmbed(dim=hidden_dim)
        cond_inner = max(condition_dim // 2, hidden_dim)
        self.cond_embed = ConditionEmbed(
            in_dim=condition_dim,
            hidden_dim=cond_inner,
            out_dim=hidden_dim,
        )

        # Non-learnable RMSNorms used to balance the time and condition
        # signals before they enter adaLN. ``elementwise_affine=False`` is
        # deliberate: a learnable gamma would let the model re-introduce
        # the magnitude imbalance these norms exist to remove.
        self.time_norm = nn.RMSNorm(hidden_dim, elementwise_affine=False)
        self.cond_norm = nn.RMSNorm(hidden_dim, elementwise_affine=False)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        # Final norm + adaLN + projection back to latent_dim
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        self.final_proj = nn.Linear(hidden_dim, latent_dim)
        # Small-std init on the final layers so:
        #   * initial output is near zero (training starts well-behaved), and
        #   * gradients flow through cond_embed / time_embed / final_proj on
        #     the very first backward (block adaLN_modulation is zero-init,
        #     so the block path doesn't carry the conditioning gradient yet).
        nn.init.normal_(self.final_modulation[-1].weight, std=0.02)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.normal_(self.final_proj.weight, std=0.02)
        nn.init.zeros_(self.final_proj.bias)

    # ---- validation helpers ------------------------------------------------

    def _validate_x(self, x: torch.Tensor) -> int:
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"x must be a Tensor; got {type(x).__name__}")
        if not x.is_floating_point():
            raise ValueError(
                f"x must be a floating tensor; got dtype {x.dtype}"
            )
        if x.ndim != 3:
            raise ValueError(
                f"x must be 3-d (B, {self.latent_tokens}, {self.latent_dim}); "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.latent_tokens or x.shape[2] != self.latent_dim:
            raise ValueError(
                f"x shape {tuple(x.shape)} != expected "
                f"(B, {self.latent_tokens}, {self.latent_dim})"
            )
        if not torch.isfinite(x).all():
            raise ValueError("x contains non-finite values")
        return int(x.shape[0])

    def _validate_condition(self, condition: torch.Tensor, B: int) -> None:
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
                f"condition must be 2-d (B, {self.condition_dim}); "
                f"got shape {tuple(condition.shape)}"
            )
        if condition.shape[0] != B:
            raise ValueError(
                f"condition batch size {condition.shape[0]} != x batch size {B}"
            )
        if condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"condition trailing dim {condition.shape[1]} != "
                f"condition_dim {self.condition_dim}"
            )
        if not torch.isfinite(condition).all():
            raise ValueError("condition contains non-finite values")

    def _expand_t(self, t, B: int, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``t`` to a ``(B,)`` tensor on ``x``'s device / dtype."""
        if isinstance(t, bool):
            raise ValueError("t must be a number, not bool")
        if isinstance(t, (int, float)):
            t = torch.tensor(float(t), device=x.device, dtype=x.dtype)
        elif isinstance(t, torch.Tensor):
            t = t.to(device=x.device, dtype=x.dtype)
        else:
            raise ValueError(
                f"t must be float, int, or Tensor; got {type(t).__name__}"
            )

        if t.ndim == 0:
            t = t.expand(B)
        elif t.ndim == 1:
            if t.shape[0] != B:
                raise ValueError(
                    f"t shape {tuple(t.shape)} does not match batch size {B}"
                )
        elif t.ndim == 2 and tuple(t.shape) == (B, 1):
            t = t.squeeze(-1)
        else:
            raise ValueError(
                f"t must be scalar, (B,), or (B, 1); got shape {tuple(t.shape)}"
            )
        return t

    # ---- conditioning ------------------------------------------------------

    def _combine_conditioning(
        self, time_raw: torch.Tensor, cond_raw: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Apply optional RMSNorm balancing + scaled mix to time/cond.

        Single source of truth for the adaLN conditioning vector. The
        ``forward`` and ``get_conditioning_embeddings`` paths both call
        this so they cannot drift.
        """
        if self.balance_conditioning:
            time_used = self.time_norm(time_raw)
            cond_used = self.cond_norm(cond_raw)
            denom = math.sqrt(self.time_scale ** 2 + self.condition_scale ** 2)
            combined = (
                self.time_scale * time_used + self.condition_scale * cond_used
            ) / denom
        else:
            time_used = time_raw
            cond_used = cond_raw
            combined = self.time_scale * time_used + self.condition_scale * cond_used
        return {
            "time_raw": time_raw,
            "condition_raw": cond_raw,
            "time_used": time_used,
            "condition_used": cond_used,
            "combined": combined,
        }

    def get_conditioning_embeddings(
        self, t, condition: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return time/condition embedding tensors used to drive adaLN.

        Keys:
            ``time_raw``        : ``time_embed(t)``                shape ``(B, hidden_dim)``
            ``condition_raw``   : ``cond_embed(condition)``        shape ``(B, hidden_dim)``
            ``time_used``       : after optional RMSNorm            shape ``(B, hidden_dim)``
            ``condition_used``  : after optional RMSNorm            shape ``(B, hidden_dim)``
            ``combined``        : weighted sum fed to adaLN         shape ``(B, hidden_dim)``

        Used by ``cellfluxv2.train.diagnostics.embedding_rms``.
        """
        if not isinstance(condition, torch.Tensor):
            raise ValueError(
                f"condition must be a Tensor; got {type(condition).__name__}"
            )
        if condition.ndim != 2:
            raise ValueError(
                f"condition must be 2-d (B, {self.condition_dim}); "
                f"got shape {tuple(condition.shape)}"
            )
        B = int(condition.shape[0])
        self._validate_condition(condition, B)
        t = self._expand_t(t, B, condition)
        time_raw = self.time_embed(t)
        cond_raw = self.cond_embed(condition)
        return self._combine_conditioning(time_raw, cond_raw)

    # ---- forward -----------------------------------------------------------

    def forward(
        self, x: torch.Tensor, t, condition: torch.Tensor
    ) -> torch.Tensor:
        B = self._validate_x(x)
        self._validate_condition(condition, B)
        t = self._expand_t(t, B, x)  # SinusoidalTimeEmbed validates [0, 1] / finite

        # Token + positional
        h = self.x_embed(x) + self.pos_embed  # (B, 169, hidden_dim)

        # Conditioning vector (balanced if configured).
        time_raw = self.time_embed(t)
        cond_raw = self.cond_embed(condition)
        c = self._combine_conditioning(time_raw, cond_raw)["combined"]

        # Transformer blocks
        for block in self.blocks:
            h = block(h, c)

        # Final norm + adaLN + projection
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        v = self.final_proj(h)
        return v
