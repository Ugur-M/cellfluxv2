"""``CellFluxDataset`` — produces (x0, x1, condition, meta) per index.

Stage 1: ``x0`` is fresh Gaussian noise; ``x1`` is the (normalized)
treated cell latent.
Stage 2: ``x0`` is a (normalized) **same-plate** control cell latent
sampled via the ``PairIndex``; ``x1`` is the (normalized) treated cell
latent. The noise interpolation lives in the flow-path code, not here.

The dataset composes the modules from earlier steps:
  - ``MetadataSplit``  — which rows are treated vs control, with metadata_idx preserved
  - ``PairIndex``       — strict same-(experiment, plate) sampling, no fallback
  - ``PlateCache``      — validated NPZ loader; latent row lookup via ``plate.rows_for(address)``
  - ``FingerprintCache``— Morgan FPs keyed by ``treatment`` (not raw SMILES)
  - ``mean / std``      — per-channel norm stats from ``latent_norm.load_norm_stats``

Determinism: ``__getitem__(idx)`` uses ``np.random.default_rng((rng_seed, idx))``,
so the same ``(rng_seed, idx)`` always yields the same control row, the
same latent row, and (stage 1) the same noise. No reliance on global
numpy / torch / random state.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .fingerprints import FingerprintCache
from .metadata import MetadataSplit
from .pair_index import PairIndex
from .plate_cache import PlateCache

LATENT_SHAPE = (169, 8)
CONDITION_DIM = 1024


class CellFluxDataset(Dataset):
    """Per-row (x0, x1, condition, meta) producer for flow-matching training."""

    def __init__(
        self,
        metadata_split: MetadataSplit,
        pair_index: PairIndex,
        plate_cache: PlateCache,
        fp_cache: FingerprintCache,
        mean: torch.Tensor,
        std: torch.Tensor,
        stage: int,
        rng_seed: int,
    ):
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        if tuple(mean.shape) != (LATENT_SHAPE[1],):
            raise ValueError(
                f"mean shape {tuple(mean.shape)} must be ({LATENT_SHAPE[1]},)"
            )
        if tuple(std.shape) != (LATENT_SHAPE[1],):
            raise ValueError(
                f"std shape {tuple(std.shape)} must be ({LATENT_SHAPE[1]},)"
            )
        if fp_cache.n_bits != CONDITION_DIM:
            raise ValueError(
                f"FingerprintCache n_bits must be {CONDITION_DIM} for this "
                f"dataset; got {fp_cache.n_bits}"
            )

        self.split = metadata_split
        self.pair_index = pair_index
        self.plate_cache = plate_cache
        self.fp_cache = fp_cache
        self.mean = mean.detach().to(torch.float32)
        self.std = std.detach().to(torch.float32)
        self.stage = int(stage)
        self.rng_seed = int(rng_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch counter used in per-item seeding.

        Per-item RNG seeds are ``(rng_seed, epoch, idx)``, so the same
        seed + same epoch + same idx is reproducible, while a different
        epoch resamples stage-1 noise and (when the pool has > 1 option)
        the stage-2 control row and the per-well latent row.
        """
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise ValueError(f"epoch must be an int; got {type(epoch).__name__}")
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0; got {epoch}")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.split.treated)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = np.random.default_rng((self.rng_seed, self.epoch, int(idx)))

        treated_row = self.split.treated.iloc[idx]
        experiment = str(treated_row["experiment_name"])
        plate_num = int(treated_row["plate"])
        treated_address = str(treated_row["address"])
        treatment = str(treated_row["treatment"])
        treated_m_idx = int(treated_row["metadata_idx"])

        plate_obj = self.plate_cache.get(experiment, plate_num)
        treated_rows = plate_obj.rows_for(treated_address)  # raises KeyError on miss
        treated_row_idx = int(rng.choice(treated_rows))

        x1 = torch.from_numpy(plate_obj.latent[treated_row_idx])
        x1 = (x1 - self.mean) / self.std  # new tensor; cache untouched

        condition = torch.from_numpy(
            self.fp_cache.by_treatment(treatment).astype(np.float32)
        )

        control_m_idx: int | None = None
        control_address: str | None = None
        if self.stage == 1:
            # x0 is N(0, I) already in normalized latent space — do NOT apply
            # (x - mean) / std on top of it.
            x0 = torch.from_numpy(
                rng.standard_normal(LATENT_SHAPE).astype(np.float32)
            )
        else:
            control_m_idx = self.pair_index.sample_control(treated_m_idx, rng)
            control_row = self.split.control.loc[control_m_idx]
            ctrl_exp = str(control_row["experiment_name"])
            ctrl_plate = int(control_row["plate"])
            if ctrl_exp != experiment or ctrl_plate != plate_num:
                raise RuntimeError(
                    f"PairIndex returned cross-plate control: "
                    f"treated=({experiment}, {plate_num}) vs "
                    f"control=({ctrl_exp}, {ctrl_plate}) — pair_index is broken"
                )
            control_address = str(control_row["address"])
            control_rows = plate_obj.rows_for(control_address)
            control_row_idx = int(rng.choice(control_rows))
            x0 = torch.from_numpy(plate_obj.latent[control_row_idx])
            x0 = (x0 - self.mean) / self.std

        assert tuple(x0.shape) == LATENT_SHAPE, f"x0 shape {tuple(x0.shape)}"
        assert tuple(x1.shape) == LATENT_SHAPE, f"x1 shape {tuple(x1.shape)}"
        assert tuple(condition.shape) == (CONDITION_DIM,), (
            f"condition shape {tuple(condition.shape)}"
        )
        assert x0.dtype == torch.float32 and x1.dtype == torch.float32
        assert torch.isfinite(x0).all(), "x0 has non-finite values"
        assert torch.isfinite(x1).all(), "x1 has non-finite values"
        assert torch.isfinite(condition).all(), "condition has non-finite values"

        return {
            "x0": x0,
            "x1": x1,
            "condition": condition,
            "meta": {
                "treated_metadata_idx": treated_m_idx,
                "control_metadata_idx": control_m_idx,
                "experiment_name": experiment,
                "plate": plate_num,
                "treated_address": treated_address,
                "control_address": control_address,
                "treatment": treatment,
                "stage": self.stage,
            },
        }

    @staticmethod
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Custom collate. ``meta`` keys with ``None`` values would otherwise
        break ``torch.utils.data.default_collate``, so meta becomes
        per-key list-of-values instead of being tensorized.
        """
        out = {
            "x0": torch.stack([b["x0"] for b in batch]),
            "x1": torch.stack([b["x1"] for b in batch]),
            "condition": torch.stack([b["condition"] for b in batch]),
        }
        meta_keys = list(batch[0]["meta"].keys())
        out["meta"] = {k: [b["meta"][k] for b in batch] for k in meta_keys}
        return out
