"""Morgan (ECFP) fingerprint computation and a precomputed-cache loader.

`compute_morgan` is RDKit-only and used by the offline precompute script.
`FingerprintCache` / `load_fp_cache` are used at training time and do not
require RDKit; they read the `.npz` produced by
`scripts/precompute_fingerprints.py`.

Default `n_bits=1024` matches the reproduction baseline. 2048 is reserved
for an explicit ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=8)
def _morgan_generator(radius: int, n_bits: int, use_chirality: bool):
    from rdkit.Chem import rdFingerprintGenerator
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=use_chirality
    )


def compute_morgan(
    smiles: str,
    radius: int = 2,
    n_bits: int = 1024,
    use_chirality: bool = False,
) -> np.ndarray:
    """Compute a Morgan (ECFP) fingerprint bit vector for a single SMILES.

    Parameters
    ----------
    smiles : str
        Input SMILES; extended-SMILES ``|...|`` annotations are tolerated.
    radius : int
        ECFP radius (ECFP4 = radius 2).
    n_bits : int
        Width of the bit vector.
    use_chirality : bool
        If ``False`` (default), the fingerprint is stereo-blind — L/D
        enantiomers map to identical bit vectors. If ``True``, the RDKit
        ``includeChirality`` flag is passed through so stereo information
        contributes to the bits.

    Returns
    -------
    np.ndarray
        Shape ``(n_bits,)``, dtype ``uint8``, values in ``{0, 1}``.

    Raises
    ------
    ValueError
        If RDKit cannot parse the SMILES (even after stripping the
        extended-SMILES ``|...|`` annotation suffix used by rxrx3).
    """
    from rdkit import Chem
    from rdkit.DataStructs import ConvertToNumpyArray

    if not isinstance(smiles, str) or not smiles:
        raise ValueError(f"SMILES must be a non-empty string, got {smiles!r}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        bar = smiles.find("|")
        if bar != -1:
            mol = Chem.MolFromSmiles(smiles[:bar].strip())
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")

    gen = _morgan_generator(radius, n_bits, use_chirality)
    bv = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    ConvertToNumpyArray(bv, arr)
    return arr


@dataclass(frozen=True)
class FingerprintCache:
    """In-memory view over a precomputed Morgan fingerprint table.

    One row per unique ``(treatment, SMILES)`` pair. Both keys are exposed
    because metadata may reference compounds either by their human-readable
    treatment name or by their canonical SMILES.
    """

    fps: np.ndarray            # (N, n_bits) uint8
    smiles: np.ndarray         # (N,) <U
    treatments: np.ndarray     # (N,) <U
    radius: int
    n_bits: int
    use_chirality: bool

    def __post_init__(self) -> None:
        n = len(self.fps)
        if self.fps.ndim != 2 or self.fps.shape[1] != self.n_bits:
            raise ValueError(
                f"fps shape {self.fps.shape} mismatches n_bits={self.n_bits}"
            )
        if self.fps.dtype != np.uint8:
            raise ValueError(f"fps dtype {self.fps.dtype} must be uint8")
        if len(self.smiles) != n or len(self.treatments) != n:
            raise ValueError(
                f"length mismatch: fps={n} smiles={len(self.smiles)} "
                f"treatments={len(self.treatments)}"
            )
        if not ((self.fps == 0) | (self.fps == 1)).all():
            raise ValueError("fps must be binary {0, 1}")
        if not isinstance(self.use_chirality, bool):
            raise ValueError(
                f"use_chirality must be bool, got {type(self.use_chirality).__name__}"
            )

    @cached_property
    def _smiles_to_idx(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i, s in enumerate(self.smiles):
            out.setdefault(str(s), i)
        return out

    @cached_property
    def _treatment_to_idx(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i, t in enumerate(self.treatments):
            out.setdefault(str(t), i)
        return out

    def by_smiles(self, smiles: str) -> np.ndarray:
        idx = self._smiles_to_idx.get(smiles)
        if idx is None:
            raise KeyError(f"SMILES not in fingerprint cache: {smiles!r}")
        return self.fps[idx]

    def by_treatment(self, treatment: str) -> np.ndarray:
        idx = self._treatment_to_idx.get(treatment)
        if idx is None:
            raise KeyError(f"treatment not in fingerprint cache: {treatment!r}")
        return self.fps[idx]

    def has_smiles(self, smiles: str) -> bool:
        return smiles in self._smiles_to_idx

    def has_treatment(self, treatment: str) -> bool:
        return treatment in self._treatment_to_idx

    def __len__(self) -> int:
        return len(self.fps)


def load_fp_cache(path: str | Path) -> FingerprintCache:
    """Load a precomputed fingerprint table from `.npz`.

    Expected keys in the npz: ``fps`` (uint8, ``(N, n_bits)``), ``smiles``
    (``<U``, ``(N,)``), ``treatments`` (``<U``, ``(N,)``), ``radius`` (int),
    ``n_bits`` (int), ``use_chirality`` (bool). Raises if any key is
    missing so we never silently load a cache with unknown settings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"fingerprint cache not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"fps", "smiles", "treatments", "radius", "n_bits", "use_chirality"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"fingerprint cache at {path} is missing keys {sorted(missing)}; "
                "regenerate via scripts/precompute_fingerprints.py"
            )
        return FingerprintCache(
            fps=np.ascontiguousarray(data["fps"], dtype=np.uint8),
            smiles=np.asarray(data["smiles"]),
            treatments=np.asarray(data["treatments"]),
            radius=int(data["radius"]),
            n_bits=int(data["n_bits"]),
            use_chirality=bool(data["use_chirality"]),
        )
