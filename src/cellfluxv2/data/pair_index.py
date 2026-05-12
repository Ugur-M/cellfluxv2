"""Same-(experiment, plate) treated→control pair index.

Pairing is strict: a treated row at ``(experiment_name, plate) = (E, P)``
samples controls only from the exact same key. There is no fallback to
"same experiment, any plate", no fallback to "any experiment, same plate
integer", and no fallback to a global control pool. If any treated group
has zero controls under this key, ``build_pair_index`` raises — it is
not the right call to silently widen the pool.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KEY_COLS: tuple[str, ...] = ("experiment_name", "plate")
GroupKey = tuple[str, int]


@dataclass
class PairIndex:
    """Maps each treated ``metadata_idx`` to its same-key control pool.

    Attributes
    ----------
    groups : dict[(experiment_name, plate), {"treated": list[int], "control": list[int]}]
        Per-group metadata_idx lists.
    treated_to_group : dict[int, (experiment_name, plate)]
        Fast reverse lookup from a treated metadata_idx to its group key.
    """

    groups: dict[GroupKey, dict[str, list[int]]]
    treated_to_group: dict[int, GroupKey]

    def __len__(self) -> int:
        return len(self.treated_to_group)

    def sample_control(self, treated_idx: int, rng: np.random.Generator) -> int:
        """Sample one control metadata_idx from the same group as ``treated_idx``.

        Determinism comes from the caller-provided ``rng`` — same ``Generator``
        state → same draw. Raises ``KeyError`` if the treated index is unknown.
        """
        key = self.treated_to_group.get(treated_idx)
        if key is None:
            raise KeyError(f"treated metadata_idx not in pair index: {treated_idx}")
        controls = self.groups[key]["control"]
        return int(rng.choice(controls))


def _group_by_key(df: pd.DataFrame) -> dict[GroupKey, list[int]]:
    out: dict[GroupKey, list[int]] = {}
    for key, sub in df.groupby(list(KEY_COLS), sort=False):
        out[(str(key[0]), int(key[1]))] = sub["metadata_idx"].tolist()
    return out


def build_pair_index(treated: pd.DataFrame, control: pd.DataFrame) -> PairIndex:
    """Build a strict same-(experiment_name, plate) pair index.

    Raises
    ------
    ValueError
        If any treated ``(experiment_name, plate)`` group has no controls.
    """
    for df, name in ((treated, "treated"), (control, "control")):
        for col in KEY_COLS:
            if col not in df.columns:
                raise ValueError(f"{name} DataFrame is missing key column {col!r}")
        if "metadata_idx" not in df.columns:
            raise ValueError(f"{name} DataFrame is missing `metadata_idx`")

    treated_groups = _group_by_key(treated)
    control_groups = _group_by_key(control)

    missing = [k for k in treated_groups if not control_groups.get(k)]
    if missing:
        head = "\n".join(f"  - {k}" for k in missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise ValueError(
            f"{len(missing)} treated (experiment_name, plate) group(s) have no "
            f"controls; strict same-plate pairing has no fallback so this is "
            f"fatal:\n{head}{more}"
        )

    groups: dict[GroupKey, dict[str, list[int]]] = {}
    treated_to_group: dict[int, GroupKey] = {}
    for key, treated_idxs in treated_groups.items():
        groups[key] = {
            "treated": treated_idxs,
            "control": control_groups[key],
        }
        for ti in treated_idxs:
            treated_to_group[int(ti)] = key

    return PairIndex(groups=groups, treated_to_group=treated_to_group)
