from pathlib import Path

import pytest
import torch

from cellfluxv2.data.latent_norm import (
    denormalize,
    load_norm_stats,
    normalize,
)


def _write_stats(
    path: Path,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    extra: dict | None = None,
) -> Path:
    obj = {
        "mean": mean if mean is not None else torch.tensor([0.1] * 8, dtype=torch.float32),
        "std": std if std is not None else torch.tensor([0.5] * 8, dtype=torch.float32),
    }
    if extra:
        obj.update(extra)
    torch.save(obj, path)
    return path


# ---------- load_norm_stats -------------------------------------------------

def test_load_norm_stats_basic(tmp_path: Path):
    p = _write_stats(tmp_path / "stats.pt")
    mean, std = load_norm_stats(p)
    assert mean.shape == (8,) and std.shape == (8,)
    assert mean.dtype == torch.float32 and std.dtype == torch.float32


def test_load_norm_stats_passes_through_extra_keys(tmp_path: Path):
    """Real stats file has a `_meta` key; loader must ignore it."""
    p = _write_stats(
        tmp_path / "stats.pt",
        extra={"_meta": {"computed_at": "yesterday", "n_cells": 12345}},
    )
    mean, std = load_norm_stats(p)
    assert mean.shape == (8,)


def test_load_norm_stats_rejects_bad_mean_shape(tmp_path: Path):
    p = _write_stats(
        tmp_path / "stats.pt",
        mean=torch.zeros(4),
        std=torch.ones(8),
    )
    with pytest.raises(ValueError, match="mean shape"):
        load_norm_stats(p)


def test_load_norm_stats_rejects_bad_std_shape(tmp_path: Path):
    p = _write_stats(
        tmp_path / "stats.pt",
        mean=torch.zeros(8),
        std=torch.ones(4),
    )
    with pytest.raises(ValueError, match="std shape"):
        load_norm_stats(p)


def test_load_norm_stats_rejects_zero_std(tmp_path: Path):
    p = _write_stats(tmp_path / "stats.pt", std=torch.zeros(8))
    with pytest.raises(ValueError, match="positive"):
        load_norm_stats(p)


def test_load_norm_stats_rejects_negative_std(tmp_path: Path):
    std = torch.ones(8)
    std[3] = -0.1
    p = _write_stats(tmp_path / "stats.pt", std=std)
    with pytest.raises(ValueError, match="positive"):
        load_norm_stats(p)


def test_load_norm_stats_rejects_nan_mean(tmp_path: Path):
    mean = torch.zeros(8)
    mean[0] = float("nan")
    p = _write_stats(tmp_path / "stats.pt", mean=mean)
    with pytest.raises(ValueError, match="mean contains non-finite"):
        load_norm_stats(p)


def test_load_norm_stats_rejects_inf_std(tmp_path: Path):
    std = torch.ones(8)
    std[2] = float("inf")
    p = _write_stats(tmp_path / "stats.pt", std=std)
    with pytest.raises(ValueError, match="std contains non-finite"):
        load_norm_stats(p)


def test_load_norm_stats_rejects_missing_keys(tmp_path: Path):
    p = tmp_path / "stats.pt"
    torch.save({"mean": torch.zeros(8)}, p)
    with pytest.raises(ValueError, match="missing"):
        load_norm_stats(p)


def test_load_norm_stats_rejects_nondict(tmp_path: Path):
    p = tmp_path / "stats.pt"
    torch.save(torch.zeros(8), p)
    with pytest.raises(ValueError, match="dict"):
        load_norm_stats(p)


def test_load_norm_stats_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_norm_stats(tmp_path / "missing.pt")


# ---------- normalize / denormalize shape support ---------------------------

def test_normalize_accepts_2d():
    mean = torch.tensor([0.1] * 8)
    std = torch.tensor([0.5] * 8)
    z = torch.randn(169, 8)
    out = normalize(z, mean, std)
    assert out.shape == (169, 8)


def test_normalize_accepts_3d():
    mean = torch.tensor([0.1] * 8)
    std = torch.tensor([0.5] * 8)
    z = torch.randn(4, 169, 8)
    out = normalize(z, mean, std)
    assert out.shape == (4, 169, 8)


def test_normalize_rejects_bad_shape_2d():
    mean = torch.tensor([0.1] * 8)
    std = torch.tensor([0.5] * 8)
    with pytest.raises(ValueError, match=r"\(169, 8\)"):
        normalize(torch.randn(13, 8), mean, std)


def test_normalize_rejects_bad_shape_3d():
    mean = torch.tensor([0.1] * 8)
    std = torch.tensor([0.5] * 8)
    with pytest.raises(ValueError, match=r"\(B, 169, 8\)"):
        normalize(torch.randn(2, 13, 8), mean, std)


def test_normalize_rejects_wrong_ndim():
    mean = torch.tensor([0.1] * 8)
    std = torch.tensor([0.5] * 8)
    with pytest.raises(ValueError, match="2D|3D"):
        normalize(torch.randn(8), mean, std)


# ---------- roundtrip -------------------------------------------------------

def test_normalize_denormalize_roundtrip_2d():
    mean = torch.tensor([0.5] * 8)
    std = torch.tensor([2.0] * 8)
    z = torch.randn(169, 8)
    z2 = denormalize(normalize(z, mean, std), mean, std)
    torch.testing.assert_close(z2, z)


def test_normalize_denormalize_roundtrip_3d_anisotropic():
    mean = torch.tensor([-0.3, 0.1, 0.5, -0.2, 0.0, 1.0, -1.0, 0.7])
    std = torch.tensor([0.5, 2.0, 1.0, 0.1, 0.3, 1.5, 0.7, 0.9])
    z = torch.randn(3, 169, 8) * 5.0 + 2.0
    z2 = denormalize(normalize(z, mean, std), mean, std)
    torch.testing.assert_close(z2, z)


def test_normalize_zero_mean_unit_std_is_identity():
    mean = torch.zeros(8)
    std = torch.ones(8)
    z = torch.randn(169, 8)
    torch.testing.assert_close(normalize(z, mean, std), z)


def test_normalize_per_channel_scaling():
    """Each of the 8 channels should be normalized by its own (mean_c, std_c)."""
    mean = torch.arange(8, dtype=torch.float32)
    std = torch.arange(1, 9, dtype=torch.float32)
    z = torch.zeros(169, 8)
    out = normalize(z, mean, std)
    # Channel c becomes (-mean[c]) / std[c]; same for every token in that channel.
    expected = (-mean / std).expand(169, 8)
    torch.testing.assert_close(out, expected)
