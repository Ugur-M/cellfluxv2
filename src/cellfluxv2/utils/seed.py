"""Single entry-point for seeding Python / numpy / torch RNGs."""
from __future__ import annotations

import random as _py_random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python ``random``, NumPy, and Torch (CPU + all CUDA devices).

    When ``deterministic=True``, also flips cuDNN into deterministic mode
    (``deterministic = True``, ``benchmark = False``). This affects
    convolution kernel selection on GPUs and is a no-op on CPU-only runs.

    Raises ``ValueError`` if ``seed`` is not a non-negative ``int``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int; got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0; got {seed}")

    _py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
