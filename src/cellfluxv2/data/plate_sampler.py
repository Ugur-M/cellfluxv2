"""Plate-grouped sampler.

`shuffle=True` mixes ~50 plates into every batch under random sampling,
which thrashes the per-worker LRU `PlateCache` and starves the GPU on the
GCS-FUSE mount. This sampler emits dataset positions in plate-grouped
order: positions for plate A in shuffled order, then plate B, and so on.
Plate order is reshuffled every epoch.

A worker called for a plate-coherent batch fetches its NPZ once and serves
all 512 cells from memory. Same trade-off as the 2DGen pipeline: each
gradient step sees cells from a single plate, which is fine here because
controls (when used in Stage 2) are already same-plate by construction.

Reproducibility contract: ``set_epoch(epoch)`` is the only writer for the
sampler's epoch counter. ``__iter__`` is a pure generator that seeds its
RNG from ``self._seed + self._epoch`` and never mutates internal state.
The trainer is responsible for calling ``set_epoch(epoch)`` whenever it
calls ``dataset.set_epoch(epoch)``; the two stay in lockstep.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class PlateGroupedSampler(Sampler[int]):
    def __init__(
        self,
        plate_to_positions: Dict[Hashable, Sequence[int]],
        seed: int = 0,
    ) -> None:
        super().__init__(None)
        self._plate_to_positions: Dict[Hashable, List[int]] = {
            k: list(v) for k, v in plate_to_positions.items()
        }
        self._seed = int(seed)
        self._epoch = 0
        self._length = sum(len(v) for v in self._plate_to_positions.values())

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ValueError(f"epoch must be an int; got {type(epoch).__name__}")
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0; got {epoch}")
        self._epoch = int(epoch)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def seed(self) -> int:
        return self._seed

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self._seed + self._epoch)
        plates = list(self._plate_to_positions.keys())
        rng.shuffle(plates)
        for plate in plates:
            positions = self._plate_to_positions[plate].copy()
            rng.shuffle(positions)
            for p in positions:
                yield int(p)

    def __len__(self) -> int:
        return self._length
