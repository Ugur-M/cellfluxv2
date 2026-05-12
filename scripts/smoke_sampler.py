"""Smoke test for the Euler sampler + DiT velocity model.

Builds a tiny ``DiTVelocity(hidden_dim=128, depth=2, num_heads=4)``,
runs Euler sampling with and without CFG, and pulls a trajectory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellfluxv2.flow.sampler import EulerSampler  # noqa: E402
from cellfluxv2.models.dit import DiTVelocity  # noqa: E402


def _summary(name: str, t: torch.Tensor) -> str:
    return (
        f"  {name}: shape={tuple(t.shape)} mean={t.mean().item():.4f} "
        f"std={t.std().item():.4f} finite={bool(torch.isfinite(t).all())}"
    )


def main() -> None:
    torch.manual_seed(0)
    model = DiTVelocity(hidden_dim=128, depth=2, num_heads=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n_params:,}", flush=True)

    g = torch.Generator()
    g.manual_seed(1)
    x0 = torch.randn(2, 169, 8, generator=g)
    cond = torch.randint(0, 2, (2, 1024), generator=g).float()
    print(f"[in] x0={tuple(x0.shape)} cond={tuple(cond.shape)}", flush=True)

    # --- no guidance (alpha = 1) -------------------------------------------
    sampler_noguide = EulerSampler(model, num_steps=5, guidance_alpha=1.0)
    x1 = sampler_noguide.sample(x0, cond)
    assert x1.shape == (2, 169, 8), f"unexpected shape {tuple(x1.shape)}"
    assert torch.isfinite(x1).all(), "x1 has non-finite values"
    assert x1 is not x0, "output should be a fresh tensor"
    print("[noguide]", flush=True)
    print(_summary("x1", x1), flush=True)

    # --- with guidance (alpha = 2) -----------------------------------------
    sampler_cfg = EulerSampler(model, num_steps=5, guidance_alpha=2.0)
    x1_cfg = sampler_cfg.sample(x0, cond)
    assert x1_cfg.shape == (2, 169, 8)
    assert torch.isfinite(x1_cfg).all()
    print("[cfg alpha=2]", flush=True)
    print(_summary("x1_cfg", x1_cfg), flush=True)

    # --- trajectory --------------------------------------------------------
    sampler_traj = EulerSampler(model, num_steps=5, guidance_alpha=1.0)
    x1_t, traj = sampler_traj.sample(x0, cond, return_trajectory=True)
    assert traj.shape == (6, 2, 169, 8), f"traj shape {tuple(traj.shape)}"
    assert torch.isfinite(traj).all()
    torch.testing.assert_close(traj[0], x0)
    torch.testing.assert_close(traj[-1], x1_t)
    print(f"[traj] shape={tuple(traj.shape)} step-0==x0 OK, step-N==output OK", flush=True)

    # --- gradient + immutability checks -----------------------------------
    x0_with_grad = torch.randn(2, 169, 8, requires_grad=True)
    out = EulerSampler(model, num_steps=3).sample(x0_with_grad, cond)
    assert not out.requires_grad, "sampling should not propagate requires_grad"
    snap = x0_with_grad.detach().clone()
    EulerSampler(model, num_steps=3).sample(x0_with_grad, cond)
    torch.testing.assert_close(x0_with_grad.detach(), snap)
    print("[checks] no requires_grad on output, x0 unchanged after sampling", flush=True)

    print("\nALL SMOKE CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
