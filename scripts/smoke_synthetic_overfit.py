"""Smoke: prove the training loop can overfit a fixed synthetic batch.

Builds a tiny ``DiTVelocity(hidden=128, depth=2, heads=4)``, fixes a
synthetic batch ``{x0, x1, condition}`` of 8 samples, and runs 100
rectified train_steps with ``cond_dropout_p=0`` and no source noise.

The asserts:
  - ``final_loss < 0.5 * initial_loss`` (training is actually learning),
  - the model can be saved and reloaded with matching forward output.

This is a *synthetic* sanity check — it does not touch the rxrx3
GCS mount and does not constitute real training.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellfluxv2.models.dit import DiTVelocity  # noqa: E402
from cellfluxv2.train.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from cellfluxv2.train.loop import train_step  # noqa: E402


N_STEPS = 100
RATIO_THRESHOLD = 0.5  # final_loss < ratio * initial_loss


def main() -> None:
    torch.manual_seed(0)
    model = DiTVelocity(hidden_dim=128, depth=2, num_heads=4, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n_params:,}", flush=True)

    # Fixed synthetic batch — the same eight samples every step.
    g = torch.Generator()
    g.manual_seed(1)
    batch = {
        "x0": torch.randn(8, 169, 8, generator=g),
        "x1": torch.randn(8, 169, 8, generator=g),
        "condition": torch.randint(0, 2, (8, 1024), generator=g).float(),
    }

    losses: list[float] = []
    print(f"[train] {N_STEPS} steps rectified path, cond_dropout_p=0, source_noise=0", flush=True)
    for i in range(N_STEPS):
        m = train_step(
            model,
            batch,
            optimizer,
            device="cpu",
            path_type="rectified",
            cond_dropout_p=0.0,
            source_noise_p=0.0,
            source_noise_sigma=0.0,
        )
        losses.append(m["loss"])
        if i % 10 == 0 or i == N_STEPS - 1:
            print(
                f"  step {i:3d}: loss={m['loss']:.4f} "
                f"grad_norm={m['grad_norm']:.4f} v_pred_rms={m['v_pred_rms']:.4f}",
                flush=True,
            )

    initial = losses[0]
    final = losses[-1]
    ratio = final / initial if initial > 0 else float("nan")
    print(
        f"\n[overfit] initial={initial:.4f}  final={final:.4f}  "
        f"ratio={ratio:.4f}  threshold={RATIO_THRESHOLD}",
        flush=True,
    )
    assert final < RATIO_THRESHOLD * initial, (
        f"overfit gate failed: final={final:.4f} >= {RATIO_THRESHOLD} * "
        f"initial={initial:.4f}"
    )

    # Save → load → forward equivalence on a fixed test input.
    test_x = torch.randn(2, 169, 8)
    test_t = torch.tensor([0.3, 0.7])
    test_c = torch.randn(2, 1024)
    model.eval()
    with torch.no_grad():
        v_saved = model(test_x, test_t, test_c)

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "overfit.pt"
        save_checkpoint(
            ckpt, model=model, optimizer=optimizer,
            step=N_STEPS, epoch=0,
            config={"hidden_dim": 128, "depth": 2, "num_heads": 4},
            extra={"smoke": "synthetic_overfit", "final_loss": final},
        )
        print(f"[save] checkpoint at {ckpt}", flush=True)

        torch.manual_seed(999)
        fresh = DiTVelocity(hidden_dim=128, depth=2, num_heads=4, dropout=0.0)
        load_checkpoint(ckpt, model=fresh)
        fresh.eval()
        with torch.no_grad():
            v_loaded = fresh(test_x, test_t, test_c)
        torch.testing.assert_close(v_loaded, v_saved)
        print("[verify] reloaded model matches saved model on fixed input", flush=True)

    print("\nALL SMOKE CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
