from pathlib import Path

import numpy as np
import pytest

from cellfluxv2.data.plate_cache import (
    Plate,
    PlateCache,
    REQUIRED_ARRAYS,
    load_plate,
)


def _make_npz(
    path: Path,
    n: int = 10,
    wells: list[str] | None = None,
    latent_dtype=np.float16,
    inner_shape: tuple[int, int] = (169, 8),
    with_nan: bool = False,
    drop_keys: tuple[str, ...] = (),
    bad_sibling: str | None = None,
) -> Path:
    rng = np.random.default_rng(0)
    arrays = {
        "latent": rng.standard_normal((n, *inner_shape)).astype(latent_dtype),
        "well": np.array(
            wells if wells is not None else [f"W{i:02d}" for i in range(n)],
            dtype="<U4",
        ),
        "fov": np.arange(n, dtype=np.int16),
        "y": np.arange(n, dtype=np.int16) * 10,
        "x": np.arange(n, dtype=np.int16) * 20,
        "cell": np.arange(n, dtype=np.int16),
    }
    if with_nan:
        arrays["latent"][0, 0, 0] = np.nan
    if bad_sibling is not None:
        arrays[bad_sibling] = np.zeros(n + 3, dtype=np.int16)
    for k in drop_keys:
        del arrays[k]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


# ---------- load_plate: happy path ------------------------------------------

def test_load_plate_basic(tmp_path: Path):
    p = _make_npz(tmp_path / "Plate1.npz")
    plate = load_plate(p)
    assert plate.latent.shape == (10, 169, 8)
    assert plate.latent.dtype == np.float32  # cast from fp16
    assert plate.n_cells() == 10
    assert isinstance(plate.address_to_rows, dict)


def test_load_plate_accepts_float32_latent(tmp_path: Path):
    p = _make_npz(tmp_path / "Plate1.npz", latent_dtype=np.float32)
    plate = load_plate(p)
    assert plate.latent.dtype == np.float32


# ---------- load_plate: validation errors -----------------------------------

@pytest.mark.parametrize("drop", REQUIRED_ARRAYS)
def test_load_plate_rejects_missing_array(tmp_path: Path, drop: str):
    p = _make_npz(tmp_path / "Plate1.npz", drop_keys=(drop,))
    with pytest.raises(ValueError, match="missing required arrays"):
        load_plate(p)


def test_load_plate_rejects_nan_latent(tmp_path: Path):
    p = _make_npz(tmp_path / "Plate1.npz", with_nan=True)
    with pytest.raises(ValueError, match="non-finite"):
        load_plate(p)


def test_load_plate_rejects_inf_latent(tmp_path: Path):
    p = tmp_path / "Plate1.npz"
    n = 4
    latent = np.zeros((n, 169, 8), dtype=np.float16)
    latent[1, 50, 3] = np.float16(np.inf)
    np.savez(
        p,
        latent=latent,
        well=np.array([f"W{i}" for i in range(n)], dtype="<U4"),
        fov=np.arange(n, dtype=np.int16),
        y=np.arange(n, dtype=np.int16),
        x=np.arange(n, dtype=np.int16),
        cell=np.arange(n, dtype=np.int16),
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_plate(p)


@pytest.mark.parametrize("bad", ["well", "fov", "y", "x", "cell"])
def test_load_plate_rejects_sibling_length_mismatch(tmp_path: Path, bad: str):
    p = _make_npz(tmp_path / "Plate1.npz", bad_sibling=bad)
    with pytest.raises(ValueError, match=f"sibling '{bad}'"):
        load_plate(p)


def test_load_plate_rejects_bad_latent_dtype(tmp_path: Path):
    p = _make_npz(tmp_path / "Plate1.npz", latent_dtype=np.float64)
    with pytest.raises(ValueError, match="dtype"):
        load_plate(p)


def test_load_plate_rejects_bad_latent_inner_shape(tmp_path: Path):
    p = _make_npz(tmp_path / "Plate1.npz", inner_shape=(169, 7))
    with pytest.raises(ValueError, match=r"\(N, 169, 8\)"):
        load_plate(p)


def test_load_plate_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_plate(tmp_path / "nope.npz")


# ---------- Plate.address_to_rows mapping -----------------------------------

def test_address_to_rows_single_cell_per_well(tmp_path: Path):
    p = _make_npz(
        tmp_path / "Plate1.npz",
        n=4,
        wells=["AA01", "AA02", "AA03", "AA04"],
    )
    plate = load_plate(p)
    assert set(plate.address_to_rows.keys()) == {"AA01", "AA02", "AA03", "AA04"}
    for i, addr in enumerate(["AA01", "AA02", "AA03", "AA04"]):
        np.testing.assert_array_equal(plate.address_to_rows[addr], [i])


def test_address_to_rows_multiple_cells_per_well(tmp_path: Path):
    p = _make_npz(
        tmp_path / "Plate1.npz",
        n=6,
        wells=["AA01", "AA01", "AA02", "AA02", "AA02", "AB01"],
    )
    plate = load_plate(p)
    np.testing.assert_array_equal(plate.address_to_rows["AA01"], [0, 1])
    np.testing.assert_array_equal(plate.address_to_rows["AA02"], [2, 3, 4])
    np.testing.assert_array_equal(plate.address_to_rows["AB01"], [5])


def test_address_to_rows_dtype_is_int64(tmp_path: Path):
    p = _make_npz(tmp_path / "Plate1.npz", n=3)
    plate = load_plate(p)
    for rows in plate.address_to_rows.values():
        assert rows.dtype == np.int64


def test_plate_rows_for_missing_address_raises(tmp_path: Path):
    p = _make_npz(tmp_path / "Plate1.npz")
    plate = load_plate(p)
    with pytest.raises(KeyError, match="ZZ99"):
        plate.rows_for("ZZ99")


# ---------- PlateCache LRU --------------------------------------------------

def _seed_root(tmp_path: Path, experiments: tuple[str, ...]) -> Path:
    for exp in experiments:
        (tmp_path / exp).mkdir()
        _make_npz(tmp_path / exp / "Plate1.npz")
    return tmp_path


def test_plate_cache_plate_path_format(tmp_path: Path):
    cache = PlateCache(tmp_path)
    assert cache.plate_path("compound-001", 11) == tmp_path / "compound-001" / "Plate11.npz"


def test_plate_cache_reuses_within_window(tmp_path: Path):
    root = _seed_root(tmp_path, ("exp",))
    cache = PlateCache(root)
    a = cache.get("exp", 1)
    b = cache.get("exp", 1)
    assert a is b


def test_plate_cache_lru_eviction(tmp_path: Path):
    root = _seed_root(tmp_path, ("expA", "expB", "expC"))
    cache = PlateCache(root, max_plates=2)
    cache.get("expA", 1)
    cache.get("expB", 1)
    assert len(cache._cache) == 2
    a_first = cache.get("expA", 1)
    cache.get("expC", 1)  # should evict the LRU, which is now expB (expA was just touched)
    assert len(cache._cache) == 2
    # expA should still be cached (was touched after first miss)
    assert cache.get("expA", 1) is a_first


def test_plate_cache_missing_file_raises(tmp_path: Path):
    cache = PlateCache(tmp_path)
    with pytest.raises(FileNotFoundError):
        cache.get("does-not-exist", 1)


def test_plate_cache_rejects_invalid_max_plates(tmp_path: Path):
    with pytest.raises(ValueError, match="max_plates"):
        PlateCache(tmp_path, max_plates=0)
