"""Euler ODE sampler for the DiT velocity model.

Integrates ``dx/dt = v_theta(x_t, t, c)`` from ``t = 0`` to ``t = 1`` in
``num_steps`` constant-step forward-Euler updates:

    t_i = i / num_steps,                i = 0, 1, ..., num_steps - 1
    x_{i+1} = x_i + (1 / num_steps) * v_theta(x_i, t_i, c)

Optional classifier-free guidance: when ``guidance_alpha != 1.0``, a
second model forward against the **zero** condition gives ``v_uncond``
and ``apply_cfg`` blends:

    v = alpha * v_cond + (1 - alpha) * v_uncond

Sampling is pure inference — runs under ``torch.no_grad()``, never
calls ``backward``, never toggles ``train()``/``eval()``, and never
mutates the input ``x0`` tensor in place.
"""
from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from .cfg import apply_cfg

LATENT_SHAPE: tuple[int, int] = (169, 8)
CONDITION_DIM: int = 1024


class EulerSampler:
    """Constant-step forward-Euler sampler with optional CFG."""

    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 50,
        guidance_alpha: float = 1.0,
    ):
        if not isinstance(model, nn.Module):
            raise ValueError(
                f"model must be a torch.nn.Module; got {type(model).__name__}"
            )
        if isinstance(num_steps, bool) or not isinstance(num_steps, int):
            raise ValueError(
                f"num_steps must be an int; got {type(num_steps).__name__}"
            )
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {num_steps}")
        if isinstance(guidance_alpha, bool) or not isinstance(
            guidance_alpha, (int, float)
        ):
            raise ValueError(
                f"guidance_alpha must be a number; got {type(guidance_alpha).__name__}"
            )
        if guidance_alpha < 0:
            raise ValueError(
                f"guidance_alpha must be >= 0; got {guidance_alpha}"
            )
        self.model = model
        self.num_steps = int(num_steps)
        self.guidance_alpha = float(guidance_alpha)

    # ---- validation -------------------------------------------------------

    def _validate(self, x0: torch.Tensor, condition: torch.Tensor) -> int:
        if not isinstance(x0, torch.Tensor):
            raise ValueError(f"x0 must be a Tensor; got {type(x0).__name__}")
        if not x0.is_floating_point():
            raise ValueError(f"x0 must be floating; got dtype {x0.dtype}")
        if x0.ndim != 3 or tuple(x0.shape[-2:]) != LATENT_SHAPE:
            raise ValueError(
                f"x0 shape {tuple(x0.shape)} must be (B, 169, 8)"
            )
        if not torch.isfinite(x0).all():
            raise ValueError("x0 contains non-finite values")
        B = int(x0.shape[0])
        if not isinstance(condition, torch.Tensor):
            raise ValueError(
                f"condition must be a Tensor; got {type(condition).__name__}"
            )
        if not condition.is_floating_point():
            raise ValueError(
                f"condition must be floating; got dtype {condition.dtype}"
            )
        if condition.shape != (B, CONDITION_DIM):
            raise ValueError(
                f"condition shape {tuple(condition.shape)} must be "
                f"(B={B}, {CONDITION_DIM})"
            )
        if not torch.isfinite(condition).all():
            raise ValueError("condition contains non-finite values")
        return B

    # ---- sample -----------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        condition: torch.Tensor,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Integrate from ``x0`` at ``t=0`` to ``x1`` at ``t=1``.

        Returns the final ``x1_pred : (B, 169, 8)``. If
        ``return_trajectory`` is true, also returns
        ``traj : (num_steps + 1, B, 169, 8)`` where ``traj[0]`` is
        ``x0`` (a copy) and ``traj[-1]`` equals the final output.

        Output is the absolute generated latent, not a delta.
        """
        B = self._validate(x0, condition)
        dt = 1.0 / self.num_steps

        # Don't mutate the input; work on a copy.
        x = x0.detach().clone()

        traj: list[torch.Tensor] | None = None
        if return_trajectory:
            traj = [x.clone()]

        use_cfg = self.guidance_alpha != 1.0
        zero_condition = (
            torch.zeros_like(condition) if use_cfg else None
        )

        for i in range(self.num_steps):
            t_val = i / self.num_steps
            t = torch.full((B,), t_val, device=x.device, dtype=x.dtype)
            v_cond = self.model(x, t, condition)
            if use_cfg:
                v_uncond = self.model(x, t, zero_condition)
                v = apply_cfg(v_cond, v_uncond, self.guidance_alpha)
            else:
                v = v_cond
            x = x + dt * v
            if traj is not None:
                traj.append(x.clone())

        if not torch.isfinite(x).all():
            raise ValueError("sampled output contains non-finite values")

        if traj is not None:
            return x, torch.stack(traj, dim=0)
        return x
