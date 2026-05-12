"""Tests for cfg.py and sampler.py."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cellfluxv2.flow.cfg import apply_cfg, condition_dropout
from cellfluxv2.flow.sampler import EulerSampler


# ============================================================================
# A. condition_dropout
# ============================================================================

def test_condition_dropout_p_zero_drops_none():
    c = torch.randn(4, 1024)
    out, mask = condition_dropout(c, p=0.0)
    assert not mask.any()
    assert mask.shape == (4,)
    assert mask.dtype == torch.bool
    torch.testing.assert_close(out, c)


def test_condition_dropout_p_one_drops_all():
    c = torch.randn(4, 1024)
    out, mask = condition_dropout(c, p=1.0)
    assert mask.all()
    torch.testing.assert_close(out, torch.zeros_like(c))


def test_condition_dropout_p_half_returns_bool_mask_shape_B():
    c = torch.randn(8, 1024)
    out, mask = condition_dropout(c, p=0.5)
    assert out.shape == c.shape
    assert mask.shape == (8,)
    assert mask.dtype == torch.bool


def test_condition_dropout_input_not_mutated():
    c = torch.randn(4, 1024)
    snapshot = c.clone()
    _, _ = condition_dropout(c, p=1.0)
    torch.testing.assert_close(c, snapshot)


def test_condition_dropout_invalid_p_raises():
    c = torch.randn(4, 1024)
    with pytest.raises(ValueError, match="p must be in"):
        condition_dropout(c, p=1.5)
    with pytest.raises(ValueError, match="p must be in"):
        condition_dropout(c, p=-0.1)


def test_condition_dropout_bad_shape_raises():
    with pytest.raises(ValueError, match="2-d"):
        condition_dropout(torch.randn(1024), p=0.5)


def test_condition_dropout_nan_raises():
    c = torch.randn(4, 1024)
    c[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        condition_dropout(c, p=0.5)


def test_condition_dropout_integer_input_raises():
    with pytest.raises(ValueError, match="floating"):
        condition_dropout(torch.zeros(4, 1024, dtype=torch.int64), p=0.5)


def test_condition_dropout_deterministic_with_generator():
    c = torch.randn(8, 1024)
    g1 = torch.Generator()
    g1.manual_seed(42)
    out1, mask1 = condition_dropout(c, p=0.5, generator=g1)
    g2 = torch.Generator()
    g2.manual_seed(42)
    out2, mask2 = condition_dropout(c, p=0.5, generator=g2)
    torch.testing.assert_close(out1, out2)
    assert torch.equal(mask1, mask2)


def test_condition_dropout_dropped_rows_are_exactly_zero():
    c = torch.randn(8, 1024)
    g = torch.Generator()
    g.manual_seed(7)
    out, mask = condition_dropout(c, p=0.5, generator=g)
    for i in range(c.shape[0]):
        if mask[i]:
            torch.testing.assert_close(out[i], torch.zeros_like(out[i]))
        else:
            torch.testing.assert_close(out[i], c[i])


# ============================================================================
# B. apply_cfg
# ============================================================================

def test_apply_cfg_alpha_one_returns_v_cond():
    v_cond = torch.randn(2, 169, 8)
    v_uncond = torch.randn(2, 169, 8)
    out = apply_cfg(v_cond, v_uncond, alpha=1.0)
    torch.testing.assert_close(out, v_cond)


def test_apply_cfg_alpha_zero_returns_v_uncond():
    v_cond = torch.randn(2, 169, 8)
    v_uncond = torch.randn(2, 169, 8)
    out = apply_cfg(v_cond, v_uncond, alpha=0.0)
    torch.testing.assert_close(out, v_uncond)


def test_apply_cfg_alpha_two_gives_extrapolation():
    """alpha=2 with v_cond=1, v_uncond=0 → 2*1 + (1-2)*0 = 2."""
    v_cond = torch.ones(2, 169, 8)
    v_uncond = torch.zeros(2, 169, 8)
    out = apply_cfg(v_cond, v_uncond, alpha=2.0)
    torch.testing.assert_close(out, torch.full_like(v_cond, 2.0))


def test_apply_cfg_arbitrary_alpha_formula():
    v_cond = torch.full((2, 169, 8), 3.0)
    v_uncond = torch.full((2, 169, 8), -1.0)
    out = apply_cfg(v_cond, v_uncond, alpha=1.5)
    # 1.5 * 3 + (1 - 1.5) * (-1) = 4.5 + 0.5 = 5.0
    torch.testing.assert_close(out, torch.full_like(v_cond, 5.0))


def test_apply_cfg_shape_mismatch_raises():
    v_cond = torch.randn(2, 169, 8)
    v_uncond = torch.randn(2, 169, 7)
    with pytest.raises(ValueError, match="shape"):
        apply_cfg(v_cond, v_uncond, alpha=1.5)


def test_apply_cfg_nan_raises():
    v_cond = torch.randn(2, 169, 8)
    v_uncond = torch.randn(2, 169, 8)
    v_cond[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        apply_cfg(v_cond, v_uncond, alpha=1.5)


def test_apply_cfg_negative_alpha_raises():
    v_cond = torch.randn(2, 169, 8)
    v_uncond = torch.randn(2, 169, 8)
    with pytest.raises(ValueError, match="alpha"):
        apply_cfg(v_cond, v_uncond, alpha=-0.5)


def test_apply_cfg_integer_tensor_raises():
    v_cond = torch.zeros(2, 169, 8, dtype=torch.int64)
    v_uncond = torch.zeros(2, 169, 8, dtype=torch.int64)
    with pytest.raises(ValueError, match="floating"):
        apply_cfg(v_cond, v_uncond, alpha=1.5)


# ============================================================================
# Dummy models for sampler tests
# ============================================================================

class _ConstVelocityModel(nn.Module):
    """Always returns a constant-valued field shaped like x."""

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.value = float(value)
        # A throwaway param so it's a valid nn.Module.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, t, condition):
        return torch.full_like(x, self.value)


class _ZeroVelocityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, t, condition):
        return torch.zeros_like(x)


class _ConditionAwareModel(nn.Module):
    """Returns ones-like(x) if any condition row is non-zero; else zeros."""

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, t, condition):
        # Per-sample: True when the condition row is all zeros.
        is_zero_row = (condition.abs().sum(dim=-1) == 0)
        is_zero_row = is_zero_row.view(-1, 1, 1).expand_as(x)
        return torch.where(
            is_zero_row, torch.zeros_like(x), torch.ones_like(x)
        )


# ============================================================================
# C. Constant velocity model
# ============================================================================

def test_sampler_constant_velocity():
    """v ≡ 1, x0 = 0, N=10 → x1 ≈ 1 by exact Euler arithmetic."""
    model = _ConstVelocityModel(value=1.0)
    sampler = EulerSampler(model, num_steps=10, guidance_alpha=1.0)
    x0 = torch.zeros(2, 169, 8)
    cond = torch.randn(2, 1024)
    out = sampler.sample(x0, cond)
    torch.testing.assert_close(out, torch.ones(2, 169, 8))


def test_sampler_num_steps_one_constant_velocity():
    """Smallest valid case: one Euler step with dt=1, v=1, x0=0 → x1 = 1."""
    model = _ConstVelocityModel(value=1.0)
    sampler = EulerSampler(model, num_steps=1, guidance_alpha=1.0)
    x0 = torch.zeros(2, 169, 8)
    cond = torch.randn(2, 1024)
    out = sampler.sample(x0, cond)
    torch.testing.assert_close(out, torch.ones(2, 169, 8))


# ============================================================================
# D. Zero velocity model
# ============================================================================

def test_sampler_zero_velocity_returns_x0():
    model = _ZeroVelocityModel()
    sampler = EulerSampler(model, num_steps=10, guidance_alpha=1.0)
    x0 = torch.randn(2, 169, 8)
    cond = torch.randn(2, 1024)
    out = sampler.sample(x0, cond)
    torch.testing.assert_close(out, x0)


# ============================================================================
# E. Guidance behaviour
# ============================================================================

def test_sampler_guidance_alpha_two_extrapolates():
    """v_cond=1 (non-zero cond), v_uncond=0 (zero cond), alpha=2 → v_cfg=2."""
    model = _ConditionAwareModel()
    sampler = EulerSampler(model, num_steps=10, guidance_alpha=2.0)
    x0 = torch.zeros(2, 169, 8)
    cond = torch.ones(2, 1024)
    out = sampler.sample(x0, cond)
    torch.testing.assert_close(out, torch.full_like(out, 2.0))


def test_sampler_guidance_alpha_one_skips_uncond_call():
    """Counts model invocations: alpha=1 → N calls; alpha>1 → 2N calls."""
    model = _ConstVelocityModel(value=1.0)
    call_count = {"n": 0}
    orig = model.forward

    def counting_forward(*a, **kw):
        call_count["n"] += 1
        return orig(*a, **kw)

    model.forward = counting_forward  # type: ignore[assignment]

    x0 = torch.zeros(1, 169, 8)
    cond = torch.randn(1, 1024)
    EulerSampler(model, num_steps=5, guidance_alpha=1.0).sample(x0, cond)
    assert call_count["n"] == 5
    call_count["n"] = 0
    EulerSampler(model, num_steps=5, guidance_alpha=2.0).sample(x0, cond)
    assert call_count["n"] == 10


# ============================================================================
# F. Trajectory
# ============================================================================

def test_sampler_trajectory_shape():
    model = _ZeroVelocityModel()
    sampler = EulerSampler(model, num_steps=5, guidance_alpha=1.0)
    x0 = torch.randn(3, 169, 8)
    cond = torch.randn(3, 1024)
    out, traj = sampler.sample(x0, cond, return_trajectory=True)
    assert traj.shape == (6, 3, 169, 8)


def test_sampler_trajectory_first_entry_is_x0():
    model = _ZeroVelocityModel()
    sampler = EulerSampler(model, num_steps=5, guidance_alpha=1.0)
    x0 = torch.randn(3, 169, 8)
    cond = torch.randn(3, 1024)
    _, traj = sampler.sample(x0, cond, return_trajectory=True)
    torch.testing.assert_close(traj[0], x0)


def test_sampler_trajectory_last_entry_is_output():
    model = _ConstVelocityModel(value=0.5)
    sampler = EulerSampler(model, num_steps=4, guidance_alpha=1.0)
    x0 = torch.zeros(1, 169, 8)
    cond = torch.randn(1, 1024)
    out, traj = sampler.sample(x0, cond, return_trajectory=True)
    torch.testing.assert_close(traj[-1], out)


def test_sampler_trajectory_intermediates_are_correct():
    """traj[i] for const v=1: x_i = i / num_steps."""
    model = _ConstVelocityModel(value=1.0)
    sampler = EulerSampler(model, num_steps=5, guidance_alpha=1.0)
    x0 = torch.zeros(1, 169, 8)
    cond = torch.randn(1, 1024)
    _, traj = sampler.sample(x0, cond, return_trajectory=True)
    for i in range(6):
        torch.testing.assert_close(
            traj[i], torch.full_like(traj[i], i / 5.0)
        )


# ============================================================================
# G. Validation
# ============================================================================

def test_sampler_num_steps_zero_raises():
    with pytest.raises(ValueError, match="num_steps"):
        EulerSampler(_ZeroVelocityModel(), num_steps=0)


def test_sampler_num_steps_negative_raises():
    with pytest.raises(ValueError, match="num_steps"):
        EulerSampler(_ZeroVelocityModel(), num_steps=-3)


def test_sampler_num_steps_not_int_raises():
    with pytest.raises(ValueError, match="num_steps"):
        EulerSampler(_ZeroVelocityModel(), num_steps=5.5)  # type: ignore[arg-type]


def test_sampler_guidance_alpha_negative_raises():
    with pytest.raises(ValueError, match="guidance_alpha"):
        EulerSampler(_ZeroVelocityModel(), num_steps=10, guidance_alpha=-0.1)


def test_sampler_non_module_raises():
    with pytest.raises(ValueError, match="nn.Module"):
        EulerSampler(model=lambda x, t, c: x, num_steps=10)  # type: ignore[arg-type]


def test_sampler_bad_x_shape_raises():
    sampler = EulerSampler(_ZeroVelocityModel(), num_steps=10)
    with pytest.raises(ValueError, match="shape"):
        sampler.sample(torch.randn(2, 100, 8), torch.randn(2, 1024))


def test_sampler_bad_condition_shape_raises():
    sampler = EulerSampler(_ZeroVelocityModel(), num_steps=10)
    with pytest.raises(ValueError, match="shape"):
        sampler.sample(torch.randn(2, 169, 8), torch.randn(3, 1024))


def test_sampler_nan_x_raises():
    sampler = EulerSampler(_ZeroVelocityModel(), num_steps=10)
    x = torch.randn(2, 169, 8)
    x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        sampler.sample(x, torch.randn(2, 1024))


def test_sampler_nan_condition_raises():
    sampler = EulerSampler(_ZeroVelocityModel(), num_steps=10)
    cond = torch.randn(2, 1024)
    cond[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        sampler.sample(torch.randn(2, 169, 8), cond)


def test_sampler_integer_x_raises():
    sampler = EulerSampler(_ZeroVelocityModel(), num_steps=10)
    with pytest.raises(ValueError, match="floating"):
        sampler.sample(
            torch.zeros(2, 169, 8, dtype=torch.int64),
            torch.randn(2, 1024),
        )


# ============================================================================
# H. Input immutability + no-grad
# ============================================================================

def test_sampler_does_not_mutate_x0():
    sampler = EulerSampler(_ConstVelocityModel(value=1.0), num_steps=10)
    x0 = torch.zeros(2, 169, 8)
    snapshot = x0.clone()
    _ = sampler.sample(x0, torch.randn(2, 1024))
    torch.testing.assert_close(x0, snapshot)


def test_sampler_output_is_new_tensor():
    sampler = EulerSampler(_ZeroVelocityModel(), num_steps=10)
    x0 = torch.randn(2, 169, 8)
    out = sampler.sample(x0, torch.randn(2, 1024))
    assert out is not x0


def test_sampler_output_does_not_require_grad():
    """Sampling runs under no_grad: even if x0 requires grad, output does not."""
    sampler = EulerSampler(_ConstVelocityModel(value=1.0), num_steps=5)
    x0 = torch.randn(2, 169, 8, requires_grad=True)
    cond = torch.randn(2, 1024)
    out = sampler.sample(x0, cond)
    assert not out.requires_grad
    assert out.grad_fn is None


def test_sampler_model_params_no_grad_accumulated():
    """No backward is called; param.grad stays None on a fresh model."""
    model = _ConstVelocityModel(value=1.0)
    # Force the dummy param to require grad so it would track if anything did.
    model._dummy.requires_grad = True
    sampler = EulerSampler(model, num_steps=5, guidance_alpha=2.0)
    sampler.sample(torch.zeros(1, 169, 8), torch.ones(1, 1024))
    assert model._dummy.grad is None
