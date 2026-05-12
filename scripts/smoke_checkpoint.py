"""Smoke: tiny model + one train_step + save → fresh model + load → forward equivalence."""
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


def main() -> None:
    torch.manual_seed(0)
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n_params:,}", flush=True)

    g = torch.Generator()
    g.manual_seed(1)
    batch = {
        "x0": torch.randn(4, 169, 8, generator=g),
        "x1": torch.randn(4, 169, 8, generator=g),
        "condition": torch.randint(0, 2, (4, 1024), generator=g).float(),
    }

    metrics = train_step(model, batch, optimizer, device="cpu")
    print(
        f"[train_step] loss={metrics['loss']:.4f} grad_norm={metrics['grad_norm']:.4f}",
        flush=True,
    )

    # Fixed input for forward equivalence check.
    test_x = torch.randn(2, 169, 8)
    test_t = torch.tensor([0.3, 0.7])
    test_c = torch.randn(2, 1024)
    model.eval()
    with torch.no_grad():
        v_before = model(test_x, test_t, test_c)
    print(
        f"[pre-save] v.mean={v_before.mean().item():.4f} v.std={v_before.std().item():.4f}",
        flush=True,
    )

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ckpt.pt"
        save_checkpoint(
            ckpt,
            model=model,
            optimizer=optimizer,
            step=1,
            epoch=0,
            config={"hidden_dim": 64, "depth": 2, "num_heads": 4},
            extra={"smoke": True},
        )
        print(f"[save] {ckpt} ({ckpt.stat().st_size} bytes)", flush=True)

        torch.manual_seed(999)  # ensure fresh model differs
        fresh = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
        fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)

        with torch.no_grad():
            v_fresh_before_load = fresh(test_x, test_t, test_c)
        assert not torch.allclose(v_fresh_before_load, v_before), (
            "fresh model should differ before load"
        )

        meta = load_checkpoint(ckpt, model=fresh, optimizer=fresh_opt)
        print(
            f"[load] step={meta['step']} epoch={meta['epoch']} "
            f"config={meta['config']} extra={meta['extra']}",
            flush=True,
        )

        fresh.eval()
        with torch.no_grad():
            v_after = fresh(test_x, test_t, test_c)
        torch.testing.assert_close(v_after, v_before)
        print("[verify] reloaded model outputs match saved model on fixed input", flush=True)

    print("\nALL SMOKE CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
