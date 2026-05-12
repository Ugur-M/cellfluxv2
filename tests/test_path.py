import math

import pytest
import torch

from cellfluxv2.flow.path import (
    LATENT_SHAPE,
    broadcast_t,
    noisy_path,
    rectified_path,
    source_noise_augmentation,
)


def _make_batch(B: int = 4, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(B, *LATENT_SHAPE, generator=g, dtype=dtype)


# ============================================================================
# A. broadcast_t
# ============================================================================

def test_broadcast_t_scalar_float():
    x = _make_batch(B=3)
    out = broadcast_t(0.5, x)
    assert out.shape == (3, 1, 1)
    assert out.dtype == x.dtype
    assert torch.all(out == 0.5)


def test_broadcast_t_scalar_int():
    """Python ints are accepted (and cast to x.dtype)."""
    x = _make_batch(B=2)
    out = broadcast_t(1, x)
    assert out.shape == (2, 1, 1)
    assert torch.all(out == 1.0)


def test_broadcast_t_zero_dim_tensor():
    x = _make_batch(B=4)
    out = broadcast_t(torch.tensor(0.25), x)
    assert out.shape == (4, 1, 1)
    assert torch.all(out == 0.25)


def test_broadcast_t_vector():
    x = _make_batch(B=4)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    out = broadcast_t(t, x)
    assert out.shape == (4, 1, 1)
    torch.testing.assert_close(out.squeeze(-1).squeeze(-1), t)


def test_broadcast_t_wrong_vector_shape_raises():
    x = _make_batch(B=4)
    with pytest.raises(ValueError, match="batch size"):
        broadcast_t(torch.tensor([0.5, 0.5]), x)


def test_broadcast_t_higher_rank_raises():
    x = _make_batch(B=4)
    with pytest.raises(ValueError, match="scalar or 1-d"):
        broadcast_t(torch.zeros(4, 1), x)


def test_broadcast_t_out_of_range_raises_high():
    x = _make_batch(B=2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        broadcast_t(1.5, x)


def test_broadcast_t_out_of_range_raises_low():
    x = _make_batch(B=2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        broadcast_t(-0.1, x)


def test_broadcast_t_vector_out_of_range_raises():
    x = _make_batch(B=2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        broadcast_t(torch.tensor([0.5, 1.1]), x)


def test_broadcast_t_non_finite_raises():
    x = _make_batch(B=2)
    with pytest.raises(ValueError, match="non-finite"):
        broadcast_t(torch.tensor([0.5, float("nan")]), x)


def test_broadcast_t_dtype_cast():
    """Output dtype matches x.dtype even if t is a different dtype."""
    x = _make_batch(B=2, dtype=torch.float64)
    t = torch.tensor([0.0, 1.0], dtype=torch.float32)
    out = broadcast_t(t, x)
    assert out.dtype == torch.float64


def test_broadcast_t_rejects_bool():
    """bool is a subclass of int, but we reject it explicitly."""
    x = _make_batch(B=2)
    with pytest.raises(ValueError, match="bool"):
        broadcast_t(True, x)


def test_broadcast_t_rejects_non_tensor_x():
    with pytest.raises(ValueError, match="x must"):
        broadcast_t(0.5, [1.0, 2.0])  # type: ignore[arg-type]


# ============================================================================
# B. rectified_path
# ============================================================================

def test_rectified_t_zero_returns_x0():
    x0 = _make_batch(seed=1)
    x1 = _make_batch(seed=2)
    x_t, v = rectified_path(x0, x1, 0.0)
    torch.testing.assert_close(x_t, x0)
    torch.testing.assert_close(v, x1 - x0)


def test_rectified_t_one_returns_x1():
    x0 = _make_batch(seed=1)
    x1 = _make_batch(seed=2)
    x_t, v = rectified_path(x0, x1, 1.0)
    torch.testing.assert_close(x_t, x1)
    torch.testing.assert_close(v, x1 - x0)


def test_rectified_midpoint():
    x0 = torch.zeros(2, *LATENT_SHAPE)
    x1 = torch.ones(2, *LATENT_SHAPE)
    x_t, v = rectified_path(x0, x1, 0.5)
    torch.testing.assert_close(x_t, torch.full_like(x0, 0.5))
    torch.testing.assert_close(v, torch.ones_like(x0))


def test_rectified_v_target_does_not_depend_on_t():
    x0 = _make_batch(seed=1)
    x1 = _make_batch(seed=2)
    _, v_a = rectified_path(x0, x1, 0.2)
    _, v_b = rectified_path(x0, x1, 0.7)
    _, v_c = rectified_path(x0, x1, torch.tensor([0.1, 0.4, 0.8, 0.95]))
    torch.testing.assert_close(v_a, v_b)
    torch.testing.assert_close(v_a, v_c)
    torch.testing.assert_close(v_a, x1 - x0)


def test_rectified_vector_t():
    """Per-sample t produces per-sample x_t."""
    x0 = torch.zeros(3, *LATENT_SHAPE)
    x1 = torch.ones(3, *LATENT_SHAPE)
    t = torch.tensor([0.0, 0.5, 1.0])
    x_t, _ = rectified_path(x0, x1, t)
    torch.testing.assert_close(x_t[0], torch.zeros_like(x_t[0]))
    torch.testing.assert_close(x_t[1], torch.full_like(x_t[1], 0.5))
    torch.testing.assert_close(x_t[2], torch.ones_like(x_t[2]))


# ============================================================================
# C. noisy_path
# ============================================================================

def test_noisy_t_zero_returns_x0():
    x0 = _make_batch(seed=1)
    x1 = _make_batch(seed=2)
    eps = _make_batch(seed=3)
    x_t, v, eps_out = noisy_path(x0, x1, 0.0, eps=eps, sigma=1.0)
    # sin²(0) = 0  →  x_t = x0
    torch.testing.assert_close(x_t, x0)
    # sin(0) = 0  →  v = x1 - x0
    torch.testing.assert_close(v, x1 - x0)
    # eps is returned as-is
    assert eps_out is eps


def test_noisy_t_one_returns_x1():
    x0 = _make_batch(seed=1)
    x1 = _make_batch(seed=2)
    eps = _make_batch(seed=3)
    x_t, v, _ = noisy_path(x0, x1, 1.0, eps=eps, sigma=1.0)
    # sin²(π) = 0  →  x_t = x1
    torch.testing.assert_close(x_t, x1)
    # sin(2π) = 0  →  v = x1 - x0
    torch.testing.assert_close(v, x1 - x0)


def test_noisy_known_case_midpoint():
    """x0=0, x1=1, eps=1, t=0.5, sigma=1 → x_t=1.5, v_target=1."""
    x0 = torch.zeros(1, *LATENT_SHAPE)
    x1 = torch.ones(1, *LATENT_SHAPE)
    eps = torch.ones(1, *LATENT_SHAPE)
    x_t, v, _ = noisy_path(x0, x1, 0.5, eps=eps, sigma=1.0)
    # x_t = 0.5 + sin²(0.5π) * 1 = 0.5 + 1 = 1.5
    torch.testing.assert_close(x_t, torch.full_like(x0, 1.5))
    # v = (1-0) + π sin(π) * 1 = 1 + 0 = 1
    torch.testing.assert_close(v, torch.ones_like(x0))


def test_noisy_derivative_correction_at_quarter():
    """t=0.25, eps=ones → v == (x1-x0) + π sin(0.5π) eps = (x1-x0) + π eps."""
    x0 = torch.zeros(1, *LATENT_SHAPE)
    x1 = torch.ones(1, *LATENT_SHAPE)
    eps = torch.ones(1, *LATENT_SHAPE)
    _, v, _ = noisy_path(x0, x1, 0.25, eps=eps, sigma=1.0)
    expected = (x1 - x0) + math.pi * math.sin(0.5 * math.pi) * eps
    torch.testing.assert_close(v, expected)


def test_noisy_sigma_zero_matches_rectified():
    """With sigma=0 the noise terms vanish and noisy_path == rectified_path."""
    x0 = _make_batch(seed=1)
    x1 = _make_batch(seed=2)
    eps = _make_batch(seed=3)
    for t_val in (0.0, 0.25, 0.5, 0.75, 1.0):
        x_n, v_n, _ = noisy_path(x0, x1, t_val, eps=eps, sigma=0.0)
        x_r, v_r = rectified_path(x0, x1, t_val)
        torch.testing.assert_close(x_n, x_r)
        torch.testing.assert_close(v_n, v_r)


def test_noisy_eps_none_creates_finite_eps():
    x0 = _make_batch(seed=1)
    x1 = _make_batch(seed=2)
    x_t, v, eps = noisy_path(x0, x1, 0.5, eps=None, sigma=1.0)
    assert eps.shape == x0.shape
    assert eps.dtype == x0.dtype
    assert torch.isfinite(eps).all()
    assert torch.isfinite(x_t).all()
    assert torch.isfinite(v).all()


def test_noisy_endpoints_independent_of_eps():
    """At t∈{0,1} the noise contribution must be (numerically) zero.

    sin(2π) is ~2.5e-7 in float32, so we use unit-scale eps; the residual
    stays within ``assert_close``'s default float32 tolerance.
    """
    x0 = _make_batch(seed=1)
    x1 = _make_batch(seed=2)
    eps = _make_batch(seed=3)
    x_t0, v0, _ = noisy_path(x0, x1, 0.0, eps=eps, sigma=1.0)
    x_t1, v1, _ = noisy_path(x0, x1, 1.0, eps=eps, sigma=1.0)
    torch.testing.assert_close(x_t0, x0)
    torch.testing.assert_close(x_t1, x1)
    torch.testing.assert_close(v0, x1 - x0)
    torch.testing.assert_close(v1, x1 - x0)


def test_noisy_vector_t():
    """Per-sample t with shared eps gives per-sample x_t."""
    x0 = torch.zeros(3, *LATENT_SHAPE)
    x1 = torch.ones(3, *LATENT_SHAPE)
    eps = torch.ones(3, *LATENT_SHAPE)
    t = torch.tensor([0.0, 0.5, 1.0])
    x_t, _, _ = noisy_path(x0, x1, t, eps=eps, sigma=1.0)
    torch.testing.assert_close(x_t[0], x0[0])
    torch.testing.assert_close(x_t[1], torch.full_like(x_t[1], 1.5))
    torch.testing.assert_close(x_t[2], x1[2])


# ============================================================================
# D. source_noise_augmentation
# ============================================================================

def test_source_noise_p_zero_returns_x0_unchanged():
    x0 = _make_batch(B=4, seed=1)
    x_aug, mask, noise = source_noise_augmentation(x0, p=0.0, sigma=1.0)
    torch.testing.assert_close(x_aug, x0)
    assert not mask.any()
    assert mask.shape == (4,)
    assert mask.dtype == torch.bool
    assert noise.shape == x0.shape
    torch.testing.assert_close(noise, torch.zeros_like(x0))


def test_source_noise_p_one_augments_every_sample():
    x0 = _make_batch(B=8, seed=1)
    x_aug, mask, noise = source_noise_augmentation(x0, p=1.0, sigma=0.5)
    assert mask.all()
    # Each row should differ from x0 — randn is ~unit, sigma=0.5 → nontrivial Δ.
    for i in range(x0.shape[0]):
        assert (x_aug[i] - x0[i]).abs().sum().item() > 0


def test_source_noise_sigma_zero_returns_x0_unchanged():
    x0 = _make_batch(B=4, seed=1)
    x_aug, mask, noise = source_noise_augmentation(x0, p=0.5, sigma=0.0)
    torch.testing.assert_close(x_aug, x0)
    assert mask.shape == (4,)
    assert mask.dtype == torch.bool
    torch.testing.assert_close(noise, torch.zeros_like(x0))


def test_source_noise_outputs_shapes_and_finite():
    x0 = _make_batch(B=4, seed=1)
    x_aug, mask, noise = source_noise_augmentation(x0, p=0.5, sigma=1.0)
    assert x_aug.shape == x0.shape
    assert mask.shape == (4,)
    assert mask.dtype == torch.bool
    assert noise.shape == x0.shape
    assert noise.dtype == x0.dtype
    assert torch.isfinite(x_aug).all()
    assert torch.isfinite(noise).all()


def test_source_noise_deterministic_under_generator():
    x0 = _make_batch(B=4, seed=1)
    g_a = torch.Generator()
    g_a.manual_seed(42)
    a_aug, a_mask, a_noise = source_noise_augmentation(
        x0, p=0.5, sigma=1.0, generator=g_a
    )
    g_b = torch.Generator()
    g_b.manual_seed(42)
    b_aug, b_mask, b_noise = source_noise_augmentation(
        x0, p=0.5, sigma=1.0, generator=g_b
    )
    torch.testing.assert_close(a_aug, b_aug)
    torch.testing.assert_close(a_noise, b_noise)
    assert torch.equal(a_mask, b_mask)


def test_source_noise_x0_aug_formula_matches_spec():
    """x0_aug = x0 + mask[:, None, None] * sigma * noise (per the spec)."""
    x0 = _make_batch(B=4, seed=1)
    g = torch.Generator()
    g.manual_seed(7)
    x_aug, mask, noise = source_noise_augmentation(
        x0, p=0.5, sigma=2.0, generator=g
    )
    expected = x0 + mask.view(-1, 1, 1).to(x0.dtype) * 2.0 * noise
    torch.testing.assert_close(x_aug, expected)


# ============================================================================
# E. validation
# ============================================================================

def test_rectified_shape_mismatch_raises():
    x0 = _make_batch(B=2)
    x1 = _make_batch(B=3)
    with pytest.raises(ValueError, match="shape"):
        rectified_path(x0, x1, 0.5)


def test_rectified_bad_latent_inner_shape_raises():
    x0 = torch.randn(2, 100, 8)
    x1 = torch.randn(2, 100, 8)
    with pytest.raises(ValueError, match=r"\(169, 8\)"):
        rectified_path(x0, x1, 0.5)


def test_rectified_wrong_ndim_raises():
    x0 = torch.randn(169, 8)  # missing batch dim
    x1 = torch.randn(169, 8)
    with pytest.raises(ValueError, match="3-d"):
        rectified_path(x0, x1, 0.5)


def test_rectified_x0_nan_raises():
    x0 = _make_batch()
    x1 = _make_batch()
    x0[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="x0 contains non-finite"):
        rectified_path(x0, x1, 0.5)


def test_rectified_x1_inf_raises():
    x0 = _make_batch()
    x1 = _make_batch()
    x1[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="x1 contains non-finite"):
        rectified_path(x0, x1, 0.5)


def test_rectified_integer_tensors_raise():
    x0 = torch.zeros(2, *LATENT_SHAPE, dtype=torch.int64)
    x1 = torch.zeros(2, *LATENT_SHAPE, dtype=torch.int64)
    with pytest.raises(ValueError, match="floating"):
        rectified_path(x0, x1, 0.5)


def test_noisy_eps_shape_mismatch_raises():
    x0 = _make_batch()
    x1 = _make_batch()
    bad_eps = torch.randn(4, 100, 8)
    with pytest.raises(ValueError, match="eps shape"):
        noisy_path(x0, x1, 0.5, eps=bad_eps)


def test_noisy_eps_nan_raises():
    x0 = _make_batch()
    x1 = _make_batch()
    eps = _make_batch()
    eps[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="eps contains non-finite"):
        noisy_path(x0, x1, 0.5, eps=eps)


def test_noisy_negative_sigma_raises():
    x0 = _make_batch()
    x1 = _make_batch()
    with pytest.raises(ValueError, match="sigma"):
        noisy_path(x0, x1, 0.5, sigma=-0.1)


def test_noisy_non_numeric_sigma_raises():
    x0 = _make_batch()
    x1 = _make_batch()
    with pytest.raises(ValueError, match="sigma"):
        noisy_path(x0, x1, 0.5, sigma="big")  # type: ignore[arg-type]


def test_noisy_t_out_of_range_raises():
    x0 = _make_batch()
    x1 = _make_batch()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        noisy_path(x0, x1, 1.2)


def test_source_noise_invalid_p_high_raises():
    x0 = _make_batch()
    with pytest.raises(ValueError, match="p must be in"):
        source_noise_augmentation(x0, p=1.5, sigma=1.0)


def test_source_noise_invalid_p_low_raises():
    x0 = _make_batch()
    with pytest.raises(ValueError, match="p must be in"):
        source_noise_augmentation(x0, p=-0.1, sigma=1.0)


def test_source_noise_invalid_sigma_raises():
    x0 = _make_batch()
    with pytest.raises(ValueError, match="sigma must be >="):
        source_noise_augmentation(x0, p=0.5, sigma=-0.1)


def test_source_noise_bad_latent_shape_raises():
    x0 = torch.randn(2, 100, 8)
    with pytest.raises(ValueError, match=r"\(169, 8\)"):
        source_noise_augmentation(x0, p=0.5, sigma=1.0)


def test_source_noise_nan_raises():
    x0 = _make_batch()
    x0[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="x0 contains non-finite"):
        source_noise_augmentation(x0, p=0.5, sigma=1.0)


def test_source_noise_integer_tensor_raises():
    x0 = torch.zeros(2, *LATENT_SHAPE, dtype=torch.int64)
    with pytest.raises(ValueError, match="floating"):
        source_noise_augmentation(x0, p=0.5, sigma=1.0)
