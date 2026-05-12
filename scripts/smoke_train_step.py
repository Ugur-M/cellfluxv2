"""Smoke test for the training step + diagnostics, end to end.

Builds a tiny DiTVelocity (hidden=128, depth=2, heads=4), an AdamW
optimizer, and a small synthetic batch. Runs 5 rectified steps and 5
noisy steps, then one diagnostic_suite call. Asserts that losses are
finite, at least one parameter changes, and diagnostic values are
finite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellfluxv2.models.dit import DiTVelocity  # noqa: E402
from cellfluxv2.train.diagnostics import diagnostic_suite  # noqa: E402
from cellfluxv2.train.loop import train_step  # noqa: E402


def _make_batch(g: torch.Generator, B: int = 8) -> dict:
    return {
        "x0": torch.randn(B, 169, 8, generator=g),
        "x1": torch.randn(B, 169, 8, generator=g),
        "condition": torch.randint(0, 2, (B, 1024), generator=g).float(),
    }


def main() -> None:
    torch.manual_seed(0)
    model = DiTVelocity(hidden_dim=128, depth=2, num_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n_params:,}", flush=True)

    g = torch.Generator()
    g.manual_seed(1)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    # --- 5 rectified steps, no condition dropout ---------------------------
    rect_losses = []
    print("[rectified, cond_dropout_p=0.0]", flush=True)
    for i in range(5):
        m = train_step(
            model,
            _make_batch(g),
            optimizer,
            device="cpu",
            path_type="rectified",
            cond_dropout_p=0.0,
        )
        rect_losses.append(m["loss"])
        print(
            f"  step {i}: loss={m['loss']:.4f}  grad_norm={m['grad_norm']:.4f}  "
            f"v_pred_rms={m['v_pred_rms']:.4f}  v_target_rms={m['v_target_rms']:.4f}  "
            f"t_mean={m['t_mean']:.3f}",
            flush=True,
        )

    # --- 5 noisy steps, with condition dropout -----------------------------
    noisy_losses = []
    print("[noisy, cond_dropout_p=0.1]", flush=True)
    for i in range(5):
        m = train_step(
            model,
            _make_batch(g),
            optimizer,
            device="cpu",
            path_type="noisy",
            path_sigma=1.0,
            cond_dropout_p=0.1,
        )
        noisy_losses.append(m["loss"])
        print(
            f"  step {i}: loss={m['loss']:.4f}  grad_norm={m['grad_norm']:.4f}  "
            f"cond_drop_frac={m['cond_drop_frac']:.3f}  "
            f"v_pred_rms={m['v_pred_rms']:.4f}",
            flush=True,
        )

    # --- diagnostics -------------------------------------------------------
    print("[diagnostic_suite]", flush=True)
    diag = diagnostic_suite(model, _make_batch(g), device="cpu", path_type="noisy")
    width = max(len(k) for k in diag) + 1
    for k, v in diag.items():
        print(f"  {k:<{width}} {v:.6f}", flush=True)

    # --- assertions --------------------------------------------------------
    for label, losses in (("rectified", rect_losses), ("noisy", noisy_losses)):
        for v in losses:
            assert torch.isfinite(torch.tensor(v)), f"{label} loss not finite: {v}"
        print(
            f"[summary {label}] initial={losses[0]:.4f}  final={losses[-1]:.4f}  "
            f"delta={losses[-1] - losses[0]:+.4f}",
            flush=True,
        )

    for k, v in diag.items():
        assert torch.isfinite(torch.tensor(v)), f"diagnostic {k}={v} not finite"

    changed = any(
        not torch.allclose(p.detach(), before[n])
        for n, p in model.named_parameters()
    )
    assert changed, "training did not change any parameter"

    print("\nALL SMOKE CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
