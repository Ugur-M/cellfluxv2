import pytest
import torch
import torch.nn as nn

from cellfluxv2.models.dit import DiTVelocity
from cellfluxv2.models.embeddings import ConditionEmbed, SinusoidalTimeEmbed
from cellfluxv2.train.diagnostics import (
    diagnostic_suite,
    drug_swap_v_cos,
    embedding_rms,
    mse_real_vs_shuffled,
    rms,
)


# ---------- helpers ---------------------------------------------------------

def _make_batch(B: int = 4, seed: int = 0) -> dict:
    g = torch.Generator()
    g.manual_seed(seed)
    return {
        "x0": torch.randn(B, 169, 8, generator=g),
        "x1": torch.randn(B, 169, 8, generator=g),
        "condition": torch.randint(0, 2, (B, 1024), generator=g).float(),
        "meta": {},
    }


def _tiny_model() -> DiTVelocity:
    return DiTVelocity(hidden_dim=64, depth=2, num_heads=4)


class _CondSensitiveDummy(nn.Module):
    """``forward`` depends on the condition — used to verify diagnostics
    actually pick up condition signal."""

    def __init__(self):
        super().__init__()
        # Required by embedding_rms.
        self.cond_embed = ConditionEmbed(in_dim=1024, hidden_dim=64, out_dim=64)
        self.time_embed = SinusoidalTimeEmbed(dim=64)

    def forward(self, x_t, t, condition):
        # v depends on the first 8 condition bits → shuffled condition
        # produces a different v.
        cond_head = condition[:, :8].view(-1, 1, 8)
        return cond_head.expand_as(x_t).contiguous()


class _CondIgnoringDummy(nn.Module):
    """``forward`` ignores the condition — used to verify drug-swap and
    mse-gap go to their no-signal limits."""

    def __init__(self):
        super().__init__()
        self.cond_embed = ConditionEmbed(in_dim=1024, hidden_dim=64, out_dim=64)
        self.time_embed = SinusoidalTimeEmbed(dim=64)

    def forward(self, x_t, t, condition):
        # Output is x_t itself: identical across any condition.
        return x_t.clone()


# ============================================================================
# rms
# ============================================================================

def test_rms_zero_tensor():
    assert float(rms(torch.zeros(10)).item()) == 0.0


def test_rms_unit_tensor():
    x = torch.ones(10)
    assert float(rms(x).item()) == pytest.approx(1.0)


def test_rms_known_values():
    x = torch.tensor([3.0, 4.0])  # rms = sqrt((9 + 16) / 2) = sqrt(12.5)
    assert float(rms(x).item()) == pytest.approx(12.5 ** 0.5)


# ============================================================================
# drug_swap_v_cos
# ============================================================================

def test_drug_swap_v_cos_returns_finite_float():
    model = _tiny_model()
    cos = drug_swap_v_cos(model, _make_batch(B=4), device="cpu")
    assert isinstance(cos, float)
    assert torch.isfinite(torch.tensor(cos))


def test_drug_swap_v_cos_raises_when_batch_size_one():
    model = _tiny_model()
    with pytest.raises(ValueError, match="batch size"):
        drug_swap_v_cos(model, _make_batch(B=1), device="cpu")


def test_drug_swap_v_cos_one_for_ignoring_model():
    """An ignoring model returns identical v for any condition → cos = 1."""
    model = _CondIgnoringDummy()
    cos = drug_swap_v_cos(model, _make_batch(B=8), device="cpu")
    assert cos == pytest.approx(1.0, abs=1e-5)


def test_drug_swap_v_cos_less_than_one_for_sensitive_model():
    """A sensitive model produces different v under shuffled condition."""
    torch.manual_seed(7)
    model = _CondSensitiveDummy()
    cos = drug_swap_v_cos(model, _make_batch(B=8, seed=1), device="cpu")
    assert cos < 0.999


# ============================================================================
# mse_real_vs_shuffled
# ============================================================================

def test_mse_real_vs_shuffled_returns_required_keys():
    model = _tiny_model()
    out = mse_real_vs_shuffled(model, _make_batch(B=4), device="cpu")
    assert set(out.keys()) == {
        "mse_real_condition",
        "mse_shuffled_condition",
        "mse_gap_shuffled_minus_real",
    }
    for k, v in out.items():
        assert isinstance(v, float), f"{k} should be float"
        assert torch.isfinite(torch.tensor(v)), f"{k}={v} not finite"


def test_mse_gap_zero_for_ignoring_model():
    """v_real == v_shuf when the model ignores condition → gap = 0."""
    model = _CondIgnoringDummy()
    out = mse_real_vs_shuffled(model, _make_batch(B=8), device="cpu")
    assert abs(out["mse_gap_shuffled_minus_real"]) < 1e-6
    assert out["mse_real_condition"] == pytest.approx(
        out["mse_shuffled_condition"], abs=1e-6
    )


def test_mse_real_vs_shuffled_raises_when_batch_size_one():
    model = _tiny_model()
    with pytest.raises(ValueError, match="batch size"):
        mse_real_vs_shuffled(model, _make_batch(B=1), device="cpu")


# ============================================================================
# embedding_rms
# ============================================================================

def test_embedding_rms_returns_positive_finite():
    model = _tiny_model()
    out = embedding_rms(model, _make_batch(B=4), device="cpu")
    assert set(out.keys()) == {
        "condition_embedding_rms",
        "time_embedding_rms",
    }
    for k, v in out.items():
        assert isinstance(v, float)
        assert torch.isfinite(torch.tensor(v)), f"{k}={v}"
        assert v >= 0.0


def test_embedding_rms_raises_without_cond_embed():
    class _Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_embed = SinusoidalTimeEmbed(dim=64)

        def forward(self, x, t, c):
            return torch.zeros_like(x)

    with pytest.raises(ValueError, match="cond_embed"):
        embedding_rms(_Bare(), _make_batch(B=2), device="cpu")


def test_embedding_rms_raises_without_time_embed():
    class _Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.cond_embed = ConditionEmbed(in_dim=1024, hidden_dim=64, out_dim=64)

        def forward(self, x, t, c):
            return torch.zeros_like(x)

    with pytest.raises(ValueError, match="time_embed"):
        embedding_rms(_Bare(), _make_batch(B=2), device="cpu")


# ============================================================================
# diagnostic_suite
# ============================================================================

EXPECTED_SUITE_KEYS = {
    "drug_swap_v_cos",
    "mse_real_condition",
    "mse_shuffled_condition",
    "mse_gap_shuffled_minus_real",
    "condition_embedding_rms",
    "time_embedding_rms",
}


def test_diagnostic_suite_returns_all_keys():
    model = _tiny_model()
    out = diagnostic_suite(model, _make_batch(B=4), device="cpu")
    assert set(out.keys()) == EXPECTED_SUITE_KEYS
    for k, v in out.items():
        assert isinstance(v, float)
        assert torch.isfinite(torch.tensor(v))


# ============================================================================
# No-side-effect guarantees
# ============================================================================

def test_diagnostics_do_not_change_parameters():
    torch.manual_seed(0)
    model = _tiny_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    batch = _make_batch(B=4)
    drug_swap_v_cos(model, batch, device="cpu")
    mse_real_vs_shuffled(model, batch, device="cpu")
    embedding_rms(model, batch, device="cpu")
    diagnostic_suite(model, batch, device="cpu")
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.detach(), before[n])


def test_diagnostics_leave_no_param_gradients():
    torch.manual_seed(0)
    model = _tiny_model()
    # All grads are None on a fresh model.
    for p in model.parameters():
        assert p.grad is None
    batch = _make_batch(B=4)
    drug_swap_v_cos(model, batch, device="cpu")
    mse_real_vs_shuffled(model, batch, device="cpu")
    embedding_rms(model, batch, device="cpu")
    diagnostic_suite(model, batch, device="cpu")
    for n, p in model.named_parameters():
        assert p.grad is None, f"{n} has gradient after diagnostics"
