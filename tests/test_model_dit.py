import pytest
import torch
import torch.nn.functional as F

from cellfluxv2.models.dit import DiTBlock, DiTVelocity, modulate
from cellfluxv2.models.embeddings import ConditionEmbed, SinusoidalTimeEmbed


# ---------- helpers ---------------------------------------------------------

def _make_x(B: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(B, 169, 8, generator=g)


def _make_cond(B: int = 4, seed: int = 1) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    # Morgan FPs are 0/1 in practice; use floats for the model.
    return torch.randint(0, 2, (B, 1024), generator=g).float()


def _grad_nonzero(model: torch.nn.Module, suffix: str) -> bool:
    """Return True if any param whose name ends in ``suffix`` has a non-zero grad."""
    for name, p in model.named_parameters():
        if name.endswith(suffix) and p.grad is not None and p.grad.abs().sum().item() > 0:
            return True
    return False


# ============================================================================
# A. Time embedding
# ============================================================================

def test_time_embed_scalar():
    m = SinusoidalTimeEmbed(dim=64)
    out = m(0.5)
    assert out.shape == (1, 64)
    assert torch.isfinite(out).all()


def test_time_embed_zero_dim_tensor():
    m = SinusoidalTimeEmbed(dim=64)
    out = m(torch.tensor(0.25))
    assert out.shape == (1, 64)


def test_time_embed_vector():
    m = SinusoidalTimeEmbed(dim=64)
    t = torch.linspace(0.0, 1.0, 8)
    out = m(t)
    assert out.shape == (8, 64)
    assert torch.isfinite(out).all()


def test_time_embed_column_shape():
    m = SinusoidalTimeEmbed(dim=64)
    t = torch.tensor([[0.0], [0.5], [1.0]])
    out = m(t)
    assert out.shape == (3, 64)


def test_time_embed_out_of_range_raises():
    m = SinusoidalTimeEmbed(dim=64)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        m(1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        m(-0.1)


def test_time_embed_nan_raises():
    m = SinusoidalTimeEmbed(dim=64)
    with pytest.raises(ValueError, match="non-finite"):
        m(torch.tensor([0.5, float("nan")]))


def test_time_embed_distinguishes_t_values():
    m = SinusoidalTimeEmbed(dim=64)
    a = m(torch.tensor([0.0]))
    b = m(torch.tensor([1.0]))
    assert not torch.allclose(a, b)


def test_time_embed_dim_must_be_even():
    with pytest.raises(ValueError, match="even"):
        SinusoidalTimeEmbed(dim=7)


def test_time_embed_rejects_bool():
    m = SinusoidalTimeEmbed(dim=64)
    with pytest.raises(ValueError, match="bool"):
        m(True)


# ============================================================================
# B. Condition embedding
# ============================================================================

def test_cond_embed_basic_shape():
    m = ConditionEmbed(in_dim=1024, hidden_dim=512, out_dim=384)
    out = m(torch.randn(4, 1024))
    assert out.shape == (4, 384)
    assert torch.isfinite(out).all()


def test_cond_embed_wrong_ndim_raises():
    m = ConditionEmbed(in_dim=1024)
    with pytest.raises(ValueError, match="2-d"):
        m(torch.randn(1024))


def test_cond_embed_wrong_inner_dim_raises():
    m = ConditionEmbed(in_dim=1024)
    with pytest.raises(ValueError, match="in_dim"):
        m(torch.randn(4, 512))


def test_cond_embed_nan_raises():
    m = ConditionEmbed(in_dim=1024)
    c = torch.randn(4, 1024)
    c[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        m(c)


def test_cond_embed_integer_input_raises():
    m = ConditionEmbed(in_dim=1024)
    with pytest.raises(ValueError, match="floating"):
        m(torch.zeros(4, 1024, dtype=torch.int64))


def test_cond_embed_final_layer_small_std():
    """Per spec, fc2 should be initialized with small std (~0.02)."""
    torch.manual_seed(0)
    m = ConditionEmbed(in_dim=64, hidden_dim=32, out_dim=16)
    assert m.fc2.weight.std().item() < 0.1
    assert torch.all(m.fc2.bias == 0)


# ============================================================================
# C. DiT forward
# ============================================================================

def test_dit_tiny_forward_shape():
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    x = _make_x(B=2)
    t = torch.rand(2)
    cond = _make_cond(B=2)
    v = model(x, t, cond)
    assert v.shape == (2, 169, 8)
    assert torch.isfinite(v).all()


def test_dit_small_config_forward():
    model = DiTVelocity(hidden_dim=128, depth=2, num_heads=4)
    x = _make_x(B=4)
    t = torch.rand(4)
    cond = _make_cond(B=4)
    v = model(x, t, cond)
    assert v.shape == (4, 169, 8)
    assert torch.isfinite(v).all()


def test_dit_scalar_t_forward():
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    x = _make_x(B=2)
    cond = _make_cond(B=2)
    v = model(x, 0.3, cond)
    assert v.shape == (2, 169, 8)


def test_dit_column_t_forward():
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    x = _make_x(B=2)
    cond = _make_cond(B=2)
    v = model(x, torch.tensor([[0.3], [0.7]]), cond)
    assert v.shape == (2, 169, 8)


def test_dit_condition_batch_mismatch_raises():
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    x = _make_x(B=2)
    cond = _make_cond(B=3)
    with pytest.raises(ValueError, match="batch"):
        model(x, 0.5, cond)


def test_dit_bad_x_inner_shape_raises():
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    bad_x = torch.randn(2, 100, 8)
    cond = _make_cond(B=2)
    with pytest.raises(ValueError, match="shape"):
        model(bad_x, 0.5, cond)


def test_dit_integer_x_raises():
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    bad_x = torch.zeros(2, 169, 8, dtype=torch.int64)
    cond = _make_cond(B=2)
    with pytest.raises(ValueError, match="floating"):
        model(bad_x, 0.5, cond)


def test_dit_nan_x_raises():
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    x = _make_x(B=2)
    x[0, 0, 0] = float("nan")
    cond = _make_cond(B=2)
    with pytest.raises(ValueError, match="non-finite"):
        model(x, 0.5, cond)


def test_dit_nan_condition_raises():
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    x = _make_x(B=2)
    cond = _make_cond(B=2)
    cond[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        model(x, 0.5, cond)


def test_dit_t_out_of_range_raises():
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    x = _make_x(B=2)
    cond = _make_cond(B=2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model(x, 1.5, cond)


def test_dit_hidden_dim_not_divisible_by_heads_raises():
    with pytest.raises(ValueError, match="divisible"):
        DiTVelocity(hidden_dim=100, depth=2, num_heads=6)


def test_dit_block_hidden_dim_not_divisible_by_heads_raises():
    with pytest.raises(ValueError, match="divisible"):
        DiTBlock(hidden_dim=100, num_heads=6)


# ============================================================================
# D. Gradient flow
# ============================================================================

def test_dit_gradient_flow():
    """MSE loss backward gives non-zero grads to every learnable bucket."""
    torch.manual_seed(42)
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    x = _make_x(B=2, seed=5)
    t = torch.tensor([0.3, 0.7])
    cond = _make_cond(B=2, seed=6)
    target = torch.randn_like(x)
    v = model(x, t, cond)
    loss = F.mse_loss(v, target)
    assert loss.dim() == 0
    loss.backward()

    assert _grad_nonzero(model, "x_embed.weight"), "x_embed has no grad"
    assert _grad_nonzero(model, "time_embed.fc1.weight"), "time_embed.fc1 has no grad"
    assert _grad_nonzero(model, "time_embed.fc2.weight"), "time_embed.fc2 has no grad"
    assert _grad_nonzero(model, "cond_embed.fc1.weight"), "cond_embed.fc1 has no grad"
    assert _grad_nonzero(model, "cond_embed.fc2.weight"), "cond_embed.fc2 has no grad"
    assert _grad_nonzero(model, "final_proj.weight"), "final_proj.weight has no grad"
    assert _grad_nonzero(model, "final_modulation.1.weight"), (
        "final_modulation linear has no grad"
    )

    # At least one block param has gradient.
    has_block_grad = False
    for name, p in model.named_parameters():
        if name.startswith("blocks.0.") and p.grad is not None:
            if p.grad.abs().sum().item() > 0:
                has_block_grad = True
                break
    assert has_block_grad, "no parameter in blocks.0 has gradient"


def test_dit_pos_embed_gets_gradient():
    torch.manual_seed(0)
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    x = _make_x(B=2)
    cond = _make_cond(B=2)
    v = model(x, 0.5, cond)
    loss = v.pow(2).mean()
    loss.backward()
    assert model.pos_embed.grad is not None
    assert model.pos_embed.grad.abs().sum().item() > 0


# ============================================================================
# E. Determinism
# ============================================================================

def test_dit_two_models_same_seed_match():
    torch.manual_seed(0)
    model_a = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    x = _make_x(B=2, seed=5)
    t = torch.tensor([0.3, 0.7])
    cond = _make_cond(B=2, seed=6)
    out_a = model_a(x, t, cond)

    torch.manual_seed(0)
    model_b = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    out_b = model_b(x, t, cond)

    torch.testing.assert_close(out_a, out_b)


def test_dit_same_model_same_input_match():
    """The model is deterministic for a given input when in eval mode (no dropout)."""
    torch.manual_seed(0)
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4).eval()
    x = _make_x(B=2)
    cond = _make_cond(B=2)
    with torch.no_grad():
        a = model(x, 0.5, cond)
        b = model(x, 0.5, cond)
    torch.testing.assert_close(a, b)


# ============================================================================
# F. Parameter sanity
# ============================================================================

def test_dit_parameter_count_positive():
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_dit_no_nan_or_inf_params_at_init():
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"{name} has non-finite params at init"


def test_dit_output_shape_matches_input():
    model = DiTVelocity(hidden_dim=64, depth=2, num_heads=4)
    for B in (1, 3, 7):
        x = _make_x(B=B, seed=B)
        cond = _make_cond(B=B, seed=B + 10)
        v = model(x, 0.5, cond)
        assert v.shape == x.shape


def test_dit_block_zero_init_means_identity_at_init():
    """With zero-init block adaLN, blocks should pass x through unchanged at init."""
    torch.manual_seed(0)
    block = DiTBlock(hidden_dim=64, num_heads=4)
    x = torch.randn(2, 169, 64)
    cond = torch.randn(2, 64)
    out = block(x, cond)
    # gates are zero → residuals are zero → out == x
    torch.testing.assert_close(out, x)


# ============================================================================
# modulate helper
# ============================================================================

def test_modulate_formula():
    x = torch.ones(2, 169, 8)
    shift = torch.full((2, 8), 0.5)
    scale = torch.full((2, 8), 1.0)
    out = modulate(x, shift, scale)
    # x * (1 + scale) + shift = 1 * 2 + 0.5 = 2.5
    torch.testing.assert_close(out, torch.full_like(x, 2.5))


def test_modulate_zero_shift_zero_scale_is_identity():
    x = torch.randn(2, 169, 8)
    shift = torch.zeros(2, 8)
    scale = torch.zeros(2, 8)
    out = modulate(x, shift, scale)
    torch.testing.assert_close(out, x)
