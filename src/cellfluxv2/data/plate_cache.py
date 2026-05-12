"""Per-plate NPZ latent loader with a small LRU cache.

Each NPZ stores one rxrx3 plate's worth of cell latents and per-cell
siblings (``well``, ``fov``, ``y``, ``x``, ``cell``). Required schema:

    latent: (N, 169, 8) float16 or float32
    well:   (N,) unicode  (the per-plate well address, e.g. "AD37")
    fov:    (N,) int
    y:      (N,) int
    x:      (N,) int
    cell:   (N,) int

Layout on disk:

    <latent_root>/<experiment_name>/Plate<plate>.npz

The loader validates every plate it touches (shapes, dtype, sibling
lengths, finiteness after fp16->fp32 cast) and **never silently repairs
NaNs**. Each loaded plate exposes a precomputed
``address_to_rows: dict[str, np.ndarray]`` that downstream code uses to
look up all latent row indices belonging to a given well address.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_ARRAYS: tuple[str, ...] = ("latent", "well", "fov", "y", "x", "cell")
LATENT_SHAPE_PER_CELL: tuple[int, int] = (169, 8)


@dataclass
class Plate:
    """One loaded plate's arrays plus a built ``address -> row_indices`` map."""

    path: Path
    latent: np.ndarray            # (N, 169, 8) float32
    well: np.ndarray              # (N,) unicode (Python str-compatible)
    fov: np.ndarray
    y: np.ndarray
    x: np.ndarray
    cell: np.ndarray
    address_to_rows: dict[str, np.ndarray]

    def n_cells(self) -> int:
        return int(self.latent.shape[0])

    def rows_for(self, address: str) -> np.ndarray:
        rows = self.address_to_rows.get(address)
        if rows is None:
            raise KeyError(f"address {address!r} not in plate {self.path.name}")
        return rows


def _build_address_to_rows(well: np.ndarray) -> dict[str, np.ndarray]:
    """Return ``{address: row_indices_array}`` over a plate's well column."""
    out: dict[str, np.ndarray] = {}
    # np.unique with inverse is O(N log N) and avoids per-cell dict overhead.
    unique_wells, inverse = np.unique(well, return_inverse=True)
    for i, addr in enumerate(unique_wells):
        out[str(addr)] = np.where(inverse == i)[0].astype(np.int64)
    return out


def load_plate(npz_path: str | Path) -> Plate:
    """Validate and load one NPZ plate file into memory."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"plate NPZ not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        missing = [k for k in REQUIRED_ARRAYS if k not in data.files]
        if missing:
            raise ValueError(
                f"NPZ {npz_path} missing required arrays: {missing}; "
                f"found: {list(data.files)}"
            )
        latent = data["latent"]
        well = data["well"]
        fov = data["fov"]
        y = data["y"]
        x = data["x"]
        cell = data["cell"]

    if latent.ndim != 3 or tuple(latent.shape[1:]) != LATENT_SHAPE_PER_CELL:
        raise ValueError(
            f"NPZ {npz_path} latent shape {latent.shape} must be (N, 169, 8)"
        )
    if latent.dtype not in (np.float16, np.float32):
        raise ValueError(
            f"NPZ {npz_path} latent dtype {latent.dtype} must be float16 or float32"
        )
    N = int(latent.shape[0])
    for name, arr in (("well", well), ("fov", fov), ("y", y), ("x", x), ("cell", cell)):
        if arr.shape != (N,):
            raise ValueError(
                f"NPZ {npz_path} sibling '{name}' shape {arr.shape} must be ({N},)"
            )

    latent_f32 = latent.astype(np.float32, copy=False) if latent.dtype != np.float32 else latent
    if not np.isfinite(latent_f32).all():
        n_bad = int(np.sum(~np.isfinite(latent_f32)))
        raise ValueError(
            f"NPZ {npz_path} latent has {n_bad} non-finite value(s) after "
            f"casting to float32; refusing to silently repair"
        )

    return Plate(
        path=npz_path,
        latent=latent_f32,
        well=well,
        fov=fov,
        y=y,
        x=x,
        cell=cell,
        address_to_rows=_build_address_to_rows(well),
    )


class PlateCache:
    """LRU cache over per-plate NPZ loads, keyed by resolved absolute path."""

    def __init__(self, latent_root: str | Path, max_plates: int = 8):
        self.latent_root = Path(latent_root)
        if max_plates < 1:
            raise ValueError(f"max_plates must be >= 1, got {max_plates}")
        self.max_plates = max_plates
        self._cache: OrderedDict[Path, Plate] = OrderedDict()

    def plate_path(self, experiment_name: str, plate: int) -> Path:
        return self.latent_root / experiment_name / f"Plate{int(plate)}.npz"

    def get(self, experiment_name: str, plate: int) -> Plate:
        path = self.plate_path(experiment_name, plate).resolve()
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        loaded = load_plate(path)
        self._cache[path] = loaded
        while len(self._cache) > self.max_plates:
            self._cache.popitem(last=False)
        return loaded
