"""Tests for ``save_checkpoint`` / ``load_checkpoint``."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cellfluxv2.train.checkpoint import load_checkpoint, save_checkpoint


class _TinyModel(nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 8):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


# ---------- save ------------------------------------------------------------

def test_save_checkpoint_creates_file(tmp_path: Path):
    model = _TinyModel()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, step=10, epoch=0)
    assert path.exists()


def test_save_checkpoint_creates_parent_dir(tmp_path: Path):
    model = _TinyModel()
    path = tmp_path / "nested" / "deep" / "ckpt.pt"
    assert not path.parent.exists()
    save_checkpoint(path, model=model, step=0, epoch=0)
    assert path.exists()


def test_save_checkpoint_atomic_no_tmp_leftover(tmp_path: Path):
    model = _TinyModel()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, step=1, epoch=0)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_save_checkpoint_negative_step_raises(tmp_path: Path):
    model = _TinyModel()
    with pytest.raises(ValueError, match="step"):
        save_checkpoint(tmp_path / "x.pt", model=model, step=-1, epoch=0)


def test_save_checkpoint_negative_epoch_raises(tmp_path: Path):
    model = _TinyModel()
    with pytest.raises(ValueError, match="epoch"):
        save_checkpoint(tmp_path / "x.pt", model=model, step=0, epoch=-1)


def test_save_checkpoint_non_module_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="nn.Module"):
        save_checkpoint(
            tmp_path / "x.pt", model="not-a-model", step=0, epoch=0  # type: ignore[arg-type]
        )


def test_save_checkpoint_non_optimizer_raises(tmp_path: Path):
    model = _TinyModel()
    with pytest.raises(ValueError, match="Optimizer"):
        save_checkpoint(
            tmp_path / "x.pt",
            model=model,
            optimizer="not-an-opt",  # type: ignore[arg-type]
            step=0,
            epoch=0,
        )


# ---------- load: roundtrip ------------------------------------------------

def test_load_checkpoint_restores_model_weights(tmp_path: Path):
    torch.manual_seed(0)
    model_a = _TinyModel()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model_a, step=42, epoch=3)

    torch.manual_seed(7)  # ensure fresh model has different init
    model_b = _TinyModel()
    # Confirm pre-load they differ.
    assert not torch.allclose(model_a.lin.weight, model_b.lin.weight)

    meta = load_checkpoint(path, model=model_b)
    torch.testing.assert_close(model_a.lin.weight, model_b.lin.weight)
    torch.testing.assert_close(model_a.lin.bias, model_b.lin.bias)
    assert meta["step"] == 42
    assert meta["epoch"] == 3


def test_load_checkpoint_restores_optimizer_state(tmp_path: Path):
    torch.manual_seed(0)
    model_a = _TinyModel()
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    # Take one step so the optimizer has state (Adam moments).
    x = torch.randn(2, 8)
    target = torch.randn(2, 8)
    loss = ((model_a(x) - target) ** 2).mean()
    loss.backward()
    opt_a.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model_a, optimizer=opt_a, step=1, epoch=0)

    torch.manual_seed(7)
    model_b = _TinyModel()
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    assert opt_b.state == {}  # no state yet
    load_checkpoint(path, model=model_b, optimizer=opt_b)
    assert opt_b.state != {}  # state restored
    # Adam moments should match.
    a_state = next(iter(opt_a.state.values()))
    b_state = next(iter(opt_b.state.values()))
    torch.testing.assert_close(a_state["exp_avg"], b_state["exp_avg"])


def test_load_checkpoint_returns_config_and_extra(tmp_path: Path):
    model = _TinyModel()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        step=5,
        epoch=1,
        config={"hidden_dim": 64, "depth": 2},
        extra={"git_sha": "abc123"},
    )
    fresh = _TinyModel()
    meta = load_checkpoint(path, model=fresh)
    assert meta["step"] == 5
    assert meta["epoch"] == 1
    assert meta["config"] == {"hidden_dim": 64, "depth": 2}
    assert meta["extra"] == {"git_sha": "abc123"}


def test_load_checkpoint_default_empty_config_extra(tmp_path: Path):
    model = _TinyModel()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, step=0, epoch=0)
    fresh = _TinyModel()
    meta = load_checkpoint(path, model=fresh)
    assert meta["config"] == {}
    assert meta["extra"] == {}


def test_load_checkpoint_without_optimizer_in_ckpt(tmp_path: Path):
    """Saving without optimizer, then loading with one passed in, should not crash."""
    model = _TinyModel()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, step=0, epoch=0)
    fresh = _TinyModel()
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    # optimizer_state isn't in the file → optimizer left untouched.
    meta = load_checkpoint(path, model=fresh, optimizer=fresh_opt)
    assert meta["step"] == 0


# ---------- load: error paths ----------------------------------------------

def test_load_checkpoint_missing_file_raises(tmp_path: Path):
    fresh = _TinyModel()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model=fresh)


def test_load_checkpoint_missing_required_keys_raises(tmp_path: Path):
    """A torch.save of a dict missing required keys must raise."""
    path = tmp_path / "bad.pt"
    torch.save({"model_state": {}}, path)  # missing step/epoch/config/extra
    fresh = _TinyModel()
    with pytest.raises(ValueError, match="missing required keys"):
        load_checkpoint(path, model=fresh)


def test_load_checkpoint_non_dict_raises(tmp_path: Path):
    path = tmp_path / "bad.pt"
    torch.save([1, 2, 3], path)
    fresh = _TinyModel()
    with pytest.raises(ValueError, match="must be a dict"):
        load_checkpoint(path, model=fresh)


# ---------- forward equivalence after load ---------------------------------

def test_load_checkpoint_forward_equivalence(tmp_path: Path):
    """After load, model_b should produce the same output as model_a on the same input."""
    torch.manual_seed(0)
    model_a = _TinyModel()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model_a, step=1, epoch=0)

    torch.manual_seed(99)
    model_b = _TinyModel()
    load_checkpoint(path, model=model_b)

    x = torch.randn(4, 8)
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        a = model_a(x)
        b = model_b(x)
    torch.testing.assert_close(a, b)
