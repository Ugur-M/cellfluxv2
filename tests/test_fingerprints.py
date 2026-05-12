from pathlib import Path

import numpy as np
import pytest

from cellfluxv2.data.fingerprints import (
    FingerprintCache,
    compute_morgan,
    load_fp_cache,
)

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
PHLORETIN_EXT = "OC1=CC=C(CCC(=O)C2=C(O)C=C(O)C=C2O)C=C1 |c:9,15,19,t:1,3,12|"
PHLORETIN_PLAIN = "OC1=CC=C(CCC(=O)C2=C(O)C=C(O)C=C2O)C=C1"


def test_compute_morgan_shape_and_dtype():
    fp = compute_morgan(ASPIRIN)
    assert fp.shape == (1024,)
    assert fp.dtype == np.uint8


def test_compute_morgan_is_binary_and_nonempty():
    fp = compute_morgan(CAFFEINE)
    assert ((fp == 0) | (fp == 1)).all()
    assert fp.sum() > 0


def test_compute_morgan_deterministic():
    a = compute_morgan(ASPIRIN)
    b = compute_morgan(ASPIRIN)
    np.testing.assert_array_equal(a, b)


def test_compute_morgan_distinguishes_compounds():
    a = compute_morgan(ASPIRIN)
    c = compute_morgan(CAFFEINE)
    assert not np.array_equal(a, c)


def test_compute_morgan_extended_smiles_matches_plain():
    """The extended `|c:|t:|` suffix should not change connectivity-based ECFP."""
    a = compute_morgan(PHLORETIN_EXT)
    b = compute_morgan(PHLORETIN_PLAIN)
    np.testing.assert_array_equal(a, b)


def test_compute_morgan_raises_on_unparseable():
    with pytest.raises(ValueError, match="parse"):
        compute_morgan("not-a-smiles-{}")


def test_compute_morgan_raises_on_empty():
    with pytest.raises(ValueError, match="non-empty"):
        compute_morgan("")


def test_compute_morgan_respects_n_bits():
    for n in (512, 1024, 2048):
        fp = compute_morgan(ASPIRIN, n_bits=n)
        assert fp.shape == (n,)
        assert fp.dtype == np.uint8


def test_compute_morgan_radius_changes_output():
    fp_r1 = compute_morgan(CAFFEINE, radius=1)
    fp_r3 = compute_morgan(CAFFEINE, radius=3)
    assert not np.array_equal(fp_r1, fp_r3)


def _make_cache_npz(tmp_path: Path, n_bits: int = 1024) -> Path:
    fps = np.zeros((3, n_bits), dtype=np.uint8)
    fps[0, 7] = 1
    fps[1, 42] = 1
    fps[2, n_bits - 1] = 1
    smiles = np.array([ASPIRIN, CAFFEINE, PHLORETIN_PLAIN], dtype=str)
    treatments = np.array(["aspirin", "caffeine", "Phloretin"], dtype=str)
    out = tmp_path / "fp.npz"
    np.savez_compressed(
        out,
        fps=fps,
        smiles=smiles,
        treatments=treatments,
        radius=np.int32(2),
        n_bits=np.int32(n_bits),
    )
    return out


def test_fingerprint_cache_roundtrip(tmp_path: Path):
    path = _make_cache_npz(tmp_path)
    cache = load_fp_cache(path)
    assert len(cache) == 3
    assert cache.n_bits == 1024
    assert cache.radius == 2
    # SMILES-keyed lookup
    fp0 = cache.by_smiles(ASPIRIN)
    assert fp0.shape == (1024,) and fp0.dtype == np.uint8 and fp0[7] == 1
    # treatment-keyed lookup pointing at the same row
    fp0_t = cache.by_treatment("aspirin")
    np.testing.assert_array_equal(fp0, fp0_t)


def test_fingerprint_cache_membership(tmp_path: Path):
    cache = load_fp_cache(_make_cache_npz(tmp_path))
    assert cache.has_smiles(PHLORETIN_PLAIN)
    assert not cache.has_smiles("nope")
    assert cache.has_treatment("Phloretin")
    assert not cache.has_treatment("nope")


def test_fingerprint_cache_missing_key_raises(tmp_path: Path):
    cache = load_fp_cache(_make_cache_npz(tmp_path))
    with pytest.raises(KeyError, match="SMILES"):
        cache.by_smiles("missing")
    with pytest.raises(KeyError, match="treatment"):
        cache.by_treatment("missing")


def test_fingerprint_cache_rejects_nonbinary():
    with pytest.raises(ValueError, match="binary"):
        FingerprintCache(
            fps=np.array([[0, 1, 2]], dtype=np.uint8),
            smiles=np.array(["x"], dtype=str),
            treatments=np.array(["X"], dtype=str),
            radius=2,
            n_bits=3,
        )


def test_fingerprint_cache_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="n_bits"):
        FingerprintCache(
            fps=np.array([[0, 1]], dtype=np.uint8),
            smiles=np.array(["x"], dtype=str),
            treatments=np.array(["X"], dtype=str),
            radius=2,
            n_bits=1024,
        )


def test_fingerprint_cache_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        FingerprintCache(
            fps=np.zeros((2, 4), dtype=np.uint8),
            smiles=np.array(["x"], dtype=str),
            treatments=np.array(["x", "y"], dtype=str),
            radius=2,
            n_bits=4,
        )


def test_fingerprint_cache_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="uint8"):
        FingerprintCache(
            fps=np.zeros((1, 4), dtype=np.float32),
            smiles=np.array(["x"], dtype=str),
            treatments=np.array(["X"], dtype=str),
            radius=2,
            n_bits=4,
        )


def test_load_fp_cache_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_fp_cache(tmp_path / "does-not-exist.npz")
