"""Smoke test for the DiT velocity model on a tiny config.

Builds a model with ``hidden_dim=128, depth=4, num_heads=4`` and runs a
single forward + MSE backward to confirm:

  - output shape ``(B, 169, 8)`` and finite,
  - loss is a finite scalar,
  - key parameter buckets receive non-zero gradients on the first
    backward (input projection, time embedding, condition embedding,
    and at least one DiT block).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from cellfluxv2.models.dit import DiTVelocity  # noqa: E402


def _grad_status(model: torch.nn.Module, suffix: str) -> str:
    for name, p in model.named_parameters():
        if name.endswith(suffix):
            if p.grad is None:
                return f"{name}: NO GRAD"
            return (
                f"{name}: |grad|.sum={p.grad.abs().sum().item():.6f}"
            )
    return f"<no param ending in {suffix!r}>"


def main() -> None:
    torch.manual_seed(0)
    model = DiTVelocity(hidden_dim=128, depth=4, num_heads=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n_params:,}", flush=True)

    g = torch.Generator()
    g.manual_seed(1)
    x = torch.randn(4, 169, 8, generator=g)
    t = torch.rand(4, generator=g)
    cond = torch.randint(0, 2, (4, 1024), generator=g).float()

    print(
        f"[in] x={tuple(x.shape)} t={tuple(t.shape)} cond={tuple(cond.shape)}",
        flush=True,
    )

    v = model(x, t, cond)
    assert v.shape == (4, 169, 8), f"unexpected v shape {tuple(v.shape)}"
    assert torch.isfinite(v).all(), "v has non-finite values"
    print(
        f"[out] v={tuple(v.shape)} mean={v.mean().item():.4f} "
        f"std={v.std().item():.4f} max_abs={v.abs().max().item():.4f}",
        flush=True,
    )

    target = torch.randn(4, 169, 8, generator=g)
    loss = F.mse_loss(v, target)
    assert loss.dim() == 0, "loss must be a scalar"
    assert torch.isfinite(loss), "loss is not finite"
    print(f"[loss] mse={loss.item():.4f}", flush=True)

    loss.backward()

    # Confirm gradients on the four required buckets + at least one block param.
    print("[grad] required buckets:", flush=True)
    for suffix in (
        "x_embed.weight",
        "time_embed.fc1.weight",
        "time_embed.fc2.weight",
        "cond_embed.fc1.weight",
        "cond_embed.fc2.weight",
        "final_proj.weight",
        "final_modulation.1.weight",
    ):
        print(f"  {_grad_status(model, suffix)}", flush=True)

    has_block_grad = False
    for name, p in model.named_parameters():
        if name.startswith("blocks.0.") and p.grad is not None:
            if p.grad.abs().sum().item() > 0:
                has_block_grad = True
                print(
                    f"[grad] first block param with grad: {name} "
                    f"|grad|.sum={p.grad.abs().sum().item():.6f}",
                    flush=True,
                )
                break
    assert has_block_grad, "no parameter in blocks.0 received a gradient"

    # Final hard asserts so the script exits non-zero on regression.
    for suffix in (
        "x_embed.weight",
        "time_embed.fc1.weight",
        "cond_embed.fc1.weight",
    ):
        for name, p in model.named_parameters():
            if name.endswith(suffix):
                assert p.grad is not None, f"{name} has no grad"
                assert p.grad.abs().sum().item() > 0, f"{name} grad is zero"

    print("\nALL SMOKE CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
