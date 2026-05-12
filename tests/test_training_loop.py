import pytest
import torch
import torch.nn as nn

from cellfluxv2.models.dit import DiTVelocity
from cellfluxv2.train.loop import (
    compute_flow_batch,
    move_batch_to_device,
    train_step,
)


# ---------- helpers ---------------------------------------------------------

def _make_batch(B: int = 4, seed: int = 0) -> dict:
    g = torch.Generator()
    g.manual_seed(seed)
    return {
        "x0": torch.randn(B, 169, 8, generator=g),
        "x1": torch.randn(B, 169, 8, generator=g),
        "condition": torch.randint(0, 2, (B, 1024), generator=g).float(),
        "meta": {"experiment_name": ["e" for _ in range(B)]},
    }


def _tiny_model() -> DiTVelocity:
    return DiTVelocity(hidden_dim=64, depth=2, num_heads=4)


# ============================================================================
# move_batch_to_device
# ============================================================================

def test_move_batch_to_device_passes_through_meta():
    batch = _make_batch(B=2)
    out = move_batch_to_device(batch, "cpu")
    assert out["meta"] is batch["meta"]  # meta is not touched


def test_move_batch_to_device_moves_required_keys():
    batch = _make_batch(B=2)
    out = move_batch_to_device(batch, "cpu")
    for k in ("x0", "x1", "condition"):
        assert out[k].device == torch.device("cpu")


def test_move_batch_to_device_does_not_mutate_input():
    batch = _make_batch(B=2)
    snap_x0 = batch["x0"].clone()
    _ = move_batch_to_device(batch, "cpu")
    torch.testing.assert_close(batch["x0"], snap_x0)


def test_move_batch_missing_key_raises():
    batch = _make_batch(B=2)
    del batch["condition"]
    with pytest.raises(ValueError, match="missing required key"):
        move_batch_to_device(batch, "cpu")


def test_move_batch_bad_x0_shape_raises():
    batch = _make_batch(B=2)
    batch["x0"] = torch.randn(2, 100, 8)
    with pytest.raises(ValueError, match=r"\(B, 169, 8\)"):
        move_batch_to_device(batch, "cpu")


def test_move_batch_x1_shape_mismatch_raises():
    batch = _make_batch(B=2)
    batch["x1"] = torch.randn(3, 169, 8)
    with pytest.raises(ValueError, match="match x0"):
        move_batch_to_device(batch, "cpu")


def test_move_batch_condition_shape_mismatch_raises():
    batch = _make_batch(B=2)
    batch["condition"] = torch.randn(2, 512)
    with pytest.raises(ValueError, match=r"\(B=2, 1024\)"):
        move_batch_to_device(batch, "cpu")


def test_move_batch_non_finite_x1_raises():
    batch = _make_batch(B=2)
    batch["x1"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        move_batch_to_device(batch, "cpu")


def test_move_batch_integer_x0_raises():
    batch = _make_batch(B=2)
    batch["x0"] = torch.zeros(2, 169, 8, dtype=torch.int64)
    with pytest.raises(ValueError, match="floating"):
        move_batch_to_device(batch, "cpu")


def test_move_batch_non_dict_raises():
    with pytest.raises(ValueError, match="dict"):
        move_batch_to_device(["x0"], "cpu")  # type: ignore[arg-type]


# ============================================================================
# compute_flow_batch
# ============================================================================

def test_compute_flow_batch_rectified_shapes():
    batch = _make_batch(B=4)
    flow = compute_flow_batch(batch, path_type="rectified")
    assert flow["x_t"].shape == (4, 169, 8)
    assert flow["v_target"].shape == (4, 169, 8)
    assert flow["t"].shape == (4,)
    assert flow["x0_aug"].shape == (4, 169, 8)
    assert flow["source_noise_mask"].shape == (4,)
    assert flow["source_noise_mask"].dtype == torch.bool
    assert flow["path_eps"] is None
    assert torch.isfinite(flow["x_t"]).all()
    assert torch.isfinite(flow["v_target"]).all()


def test_compute_flow_batch_noisy_returns_eps():
    batch = _make_batch(B=4)
    flow = compute_flow_batch(batch, path_type="noisy", path_sigma=1.0)
    assert flow["x_t"].shape == (4, 169, 8)
    assert flow["v_target"].shape == (4, 169, 8)
    assert flow["path_eps"] is not None
    assert flow["path_eps"].shape == (4, 169, 8)
    assert torch.isfinite(flow["x_t"]).all()
    assert torch.isfinite(flow["v_target"]).all()
    assert torch.isfinite(flow["path_eps"]).all()


def test_compute_flow_batch_source_noise_p_one_changes_x0():
    batch = _make_batch(B=4)
    x0_before = batch["x0"].clone()
    flow = compute_flow_batch(
        batch, source_noise_p=1.0, source_noise_sigma=1.0
    )
    assert flow["source_noise_mask"].all()
    # All samples augmented → x0_aug should differ from x0 essentially everywhere.
    assert not torch.allclose(flow["x0_aug"], x0_before)


def test_compute_flow_batch_source_noise_zero_p_keeps_x0():
    batch = _make_batch(B=4)
    flow = compute_flow_batch(
        batch, source_noise_p=0.0, source_noise_sigma=1.0
    )
    assert not flow["source_noise_mask"].any()
    torch.testing.assert_close(flow["x0_aug"], batch["x0"])


def test_compute_flow_batch_unsupported_path_type_raises():
    batch = _make_batch(B=2)
    with pytest.raises(ValueError, match="path_type"):
        compute_flow_batch(batch, path_type="cosine")


def test_compute_flow_batch_t_in_range():
    """Sampled t must be in [0, 1) per torch.rand semantics."""
    batch = _make_batch(B=8)
    flow = compute_flow_batch(batch, path_type="rectified")
    t = flow["t"]
    assert (t >= 0).all() and (t < 1).all()


# ============================================================================
# train_step
# ============================================================================

def test_train_step_returns_finite_metrics():
    torch.manual_seed(0)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_step(model, _make_batch(B=4), optimizer, device="cpu")
    for k, v in metrics.items():
        assert isinstance(v, float), f"{k} should be float, got {type(v)}"
        assert torch.isfinite(torch.tensor(v)), f"{k}={v} is not finite"


def test_train_step_returns_all_expected_keys():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_step(model, _make_batch(B=4), optimizer, device="cpu")
    expected = {
        "loss",
        "grad_norm",
        "v_pred_rms",
        "v_target_rms",
        "v_target_mean_norm",
        "t_mean",
        "cond_drop_frac",
        "source_noise_frac",
    }
    assert set(metrics.keys()) == expected


def test_train_step_updates_at_least_one_parameter():
    torch.manual_seed(0)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    train_step(model, _make_batch(B=4), optimizer, device="cpu")
    changed = any(
        not torch.allclose(p.detach(), before[n])
        for n, p in model.named_parameters()
    )
    assert changed


def test_train_step_cond_dropout_zero():
    """cond_dropout_p=0 → no rows dropped, fraction = 0."""
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_step(
        model, _make_batch(B=4), optimizer, device="cpu", cond_dropout_p=0.0
    )
    assert metrics["cond_drop_frac"] == 0.0


def test_train_step_cond_dropout_one():
    """cond_dropout_p=1 → all rows dropped, fraction = 1."""
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_step(
        model, _make_batch(B=4), optimizer, device="cpu", cond_dropout_p=1.0
    )
    assert metrics["cond_drop_frac"] == 1.0


def test_train_step_source_noise_frac_reported():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_step(
        model,
        _make_batch(B=8),
        optimizer,
        device="cpu",
        source_noise_p=1.0,
        source_noise_sigma=0.5,
    )
    assert metrics["source_noise_frac"] == 1.0


def test_train_step_grad_clip_none_still_returns_grad_norm():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_step(
        model,
        _make_batch(B=4),
        optimizer,
        device="cpu",
        grad_clip_norm=None,
    )
    assert metrics["grad_norm"] >= 0.0
    assert torch.isfinite(torch.tensor(metrics["grad_norm"]))


def test_train_step_grad_clip_invalid_raises():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="grad_clip_norm"):
        train_step(
            model,
            _make_batch(B=4),
            optimizer,
            device="cpu",
            grad_clip_norm=0.0,
        )


def test_train_step_bad_batch_raises():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bad = _make_batch(B=2)
    bad["x0"] = torch.zeros(2, 100, 8)
    with pytest.raises(ValueError, match=r"\(B, 169, 8\)"):
        train_step(model, bad, optimizer, device="cpu")


def test_train_step_non_finite_loss_raises():
    """A model that returns NaN should cause train_step to raise."""

    class _NaNModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Need at least one trainable param for the optimizer.
            self.lin = nn.Linear(8, 8)

        def forward(self, x, t, c):
            # Differentiable nan: 0 * lin(x) + nan keeps requires_grad alive
            # so the optimizer.step path is exercised.
            out = self.lin(x)
            return out * 0.0 + float("nan")

    model = _NaNModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="non-finite loss"):
        train_step(model, _make_batch(B=2), optimizer, device="cpu")


def test_train_step_not_a_module_raises():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    with pytest.raises(ValueError, match="nn.Module"):
        train_step(
            "not-a-model",  # type: ignore[arg-type]
            _make_batch(B=2),
            optimizer,
            device="cpu",
        )


def test_train_step_not_an_optimizer_raises():
    model = _tiny_model()
    with pytest.raises(ValueError, match="Optimizer"):
        train_step(
            model,
            _make_batch(B=2),
            "not-an-optimizer",  # type: ignore[arg-type]
            device="cpu",
        )
